"""Ahead-of-time compilation of torch callbacks for the ``generic_torch`` mode.

``generic_torch`` has the same Python-facing API as the ``generic`` environment
(six callables on the config dict, see GENERIC.md), but instead of calling the
callables from C++ through pybind11 on every expansion, ``create_planner``:

1. exports each callable with :func:`torch.export.export` (dynamic batch dim),
2. compiles it with AOTInductor into a ``.pt2`` package (cached on disk, keyed
   by a hash of the exported graph + captured constants + torch version +
   device), and
3. hands the package paths to the C++ ``Environment`` (generic_torch.h), which
   runs them with ``torch::inductor::AOTIModelPackageLoader``. The search loop
   never takes the GIL or calls Python.

Compiled models (see generic_torch.h):

* ``expand``    dynamics -> final state -> (cost, validity, heuristic), fused
* ``validity``  ``[B, n] -> [B]``
* ``heuristic`` ``[B, n] -> [B]``
* ``goal_test`` ``[n] -> bool``
* ``sample_controls`` ``() -> [K, c]`` -- only if the sampler is random (uses
  ops tagged ``nondeterministic_seeded``, e.g. ``torch.rand``). A deterministic
  sampler is evaluated once here and passed to C++ as a constant tensor.

Randomness: graphs containing random ops are compiled with Inductor's
``fallback_random=True``, so they draw from the global ATen generator: seed
with ``torch.manual_seed`` (CPU) and get the same stream generic mode would.

Constraints on the callables (they must be ``torch.export``-able):

* Pure torch ops on the input tensors. No ``.item()`` / ``bool(tensor)`` /
  ``.tolist()`` / numpy, and no Python control flow on tensor *values*
  (use ``torch.where`` / masking instead). ``goal_test`` must therefore return
  a tensor (e.g. ``torch.norm(s - goal) < r``), not a Python bool. The same
  tensor-returning ``goal_test`` also works in ``generic`` mode.
* Python-side state (globals, closures, counters) is read once at export time
  and frozen into the compiled graph as constants. Changing it later has no
  effect on an existing planner; rebuild the planner instead (a changed
  captured tensor changes the cache key, so it recompiles).
* Shape-dependent logic must not specialize on the batch size ``B``.
* For CUDA (``node_info.device: "cuda"``), callables must be device-agnostic:
  move captured constants with ``.to(states.device)``.

Optional (generic_torch only): ``configs["expand_fn"](states, controls) ->
(next_states, cost, valid, heuristic)`` replaces ``dynamics_fn`` + ``cost_fn``
for successor generation. Use it when the cost needs intermediate rollout
results (e.g. per-step collision counts) that generic-mode callbacks would
pass through Python-side caches; ``validity_fn`` / ``heuristic_fn`` are still
required for the start/goal checks.
"""
from __future__ import annotations

import hashlib
import io
import os
import pathlib
import time
import uuid
from typing import Any, Callable, Dict, Optional

import torch

CALLBACK_KEYS = (
    "sample_controls_fn",
    "dynamics_fn",
    "cost_fn",
    "validity_fn",
    "heuristic_fn",
    "goal_test_fn",
)

# Bump to invalidate every cached package (e.g. if the wrapper modules change).
_CACHE_VERSION = "1"
_MAX_BATCH = 1 << 24

_EXPORT_HINT = (
    "generic_torch callbacks must be torch.export-able: use pure torch ops, no "
    ".item()/bool(tensor)/numpy, no Python control flow on tensor values (use "
    "torch.where), and return tensors (goal_test should return e.g. "
    "`torch.norm(s - goal) < r`, not a Python bool)."
)


def default_cache_dir() -> pathlib.Path:
    env = os.environ.get("IGHASTAR_AOTI_CACHE")
    if env:
        return pathlib.Path(env)
    return pathlib.Path.home() / ".cache" / "ighastar" / "generic_torch"


# ----------------------------------------------------------------------------
# Wrapper modules: these define exactly what the C++ side calls.
# ----------------------------------------------------------------------------
class _Expand(torch.nn.Module):
    """dynamics + cost/validity/heuristic on the final state, in one graph.

    Mirrors generic.h: the dynamics output is cast to float32 before it is fed
    to cost/validity/heuristic, and those see the *final* rollout state.
    """

    def __init__(self, dynamics, cost, validity, heuristic):
        super().__init__()
        self.dynamics = dynamics
        self.cost = cost
        self.validity = validity
        self.heuristic = heuristic

    def forward(self, states, controls):
        nxt = self.dynamics(states, controls).to(torch.float32)
        # Contiguous, like generic.h's final state: `nxt[:, -1]` of a [B, T, n]
        # rollout is a strided view, which e.g. `.view(-1)` in a callback rejects.
        final = nxt[:, -1].contiguous() if nxt.dim() == 3 else nxt
        cost = self.cost(states, controls, final).to(torch.float32).reshape(-1)
        valid = self.validity(final).to(torch.float32).reshape(-1)
        h = self.heuristic(final).to(torch.float32).reshape(-1)
        return nxt, cost, valid, h


class _ExpandUser(torch.nn.Module):
    """User-provided ``expand_fn(states, controls) -> (next, cost, valid, h)``.

    ``next`` is ``[B, n]`` or ``[B, T, n]``; cost / valid / h are ``[B]`` and
    refer to the final state, exactly as in the default dynamics+cost path.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, states, controls):
        nxt, cost, valid, h = self.fn(states, controls)
        return (
            nxt.to(torch.float32),
            cost.to(torch.float32).reshape(-1),
            valid.to(torch.float32).reshape(-1),
            h.to(torch.float32).reshape(-1),
        )


class _Batched(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, states):
        return self.fn(states).to(torch.float32).reshape(-1)


class _GoalTest(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, state):
        out = self.fn(state)
        if not isinstance(out, torch.Tensor):
            raise TypeError(
                "goal_test_fn returned a Python %s; return a tensor instead "
                "(e.g. `torch.norm(s - goal) < r`)." % type(out).__name__
            )
        return out.reshape(-1)[0].to(torch.bool)


class _SampleControls(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self):
        return self.fn().to(torch.float32)


# ----------------------------------------------------------------------------
# Export / compile helpers
# ----------------------------------------------------------------------------
def _export(name: str, module: torch.nn.Module, args, dynamic_shapes):
    try:
        return torch.export.export(
            module, args, dynamic_shapes=dynamic_shapes, strict=False
        )
    except Exception as e:  # noqa: BLE001 - re-raise with context
        hint = _EXPORT_HINT
        if "Constraints violated" in str(e):
            hint += (
                "\nThe callback specializes on the batch size B. Some ops pick "
                "an algorithm by size (e.g. torch.cdist switches to a matmul "
                "path for >25 rows); write it with explicit broadcasting "
                "instead, e.g. (a[:, None] - b[None]).norm(dim=-1)."
            )
        raise RuntimeError(
            f"generic_torch: torch.export failed for '{name}': {e}\n{hint}"
        ) from e


def _has_random_ops(ep: torch.export.ExportedProgram) -> bool:
    tag = torch.Tag.nondeterministic_seeded
    for gm in ep.graph_module.modules():
        if not isinstance(gm, torch.fx.GraphModule):
            continue
        for node in gm.graph.nodes:
            if node.op == "call_function" and tag in getattr(node.target, "tags", ()):
                return True
    return False


def _program_key(ep: torch.export.ExportedProgram, extra: str) -> str:
    """Stable hash of an exported program: graph code + every captured tensor."""
    h = hashlib.sha256()
    h.update(f"v{_CACHE_VERSION}|torch={torch.__version__}|{extra}".encode())
    code = ep.graph_module.print_readable(
        print_output=False, include_stride=True, include_device=True
    )
    # Drop "# File: ...:line in fn, code: ..." provenance comments so the key
    # depends on what is computed, not on where the source lines live.
    code = "\n".join(
        ln for ln in code.splitlines() if not ln.lstrip().startswith("#")
    )
    h.update(code.encode())
    h.update(str(ep.range_constraints).encode())
    tensors = dict(ep.state_dict)
    tensors.update(ep.constants)
    for k in sorted(tensors):
        v = tensors[k]
        h.update(k.encode())
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            h.update(f"{t.dtype}{tuple(t.shape)}".encode())
            buf = io.BytesIO()
            torch.save(t, buf)
            h.update(buf.getvalue())
        else:
            h.update(repr(v).encode())
    return h.hexdigest()[:24]


def _compile(
    name: str,
    ep: torch.export.ExportedProgram,
    device: torch.device,
    cache_dir: pathlib.Path,
    inductor_configs: Optional[Dict[str, Any]],
    verbose: bool,
    stats: Dict[str, Any],
) -> str:
    if _has_random_ops(ep):
        # Inductor's own RNG (philox, seeded inside the AOT package) ignores
        # torch.manual_seed, so runs would not be reproducible. fallback_random
        # keeps random ops as ATen calls on the global generator: seedable with
        # torch.manual_seed and bitwise identical to eager / generic mode.
        inductor_configs = {"fallback_random": True, **(inductor_configs or {})}
    key = _program_key(ep, f"{name}|{device}|{sorted((inductor_configs or {}).items())}")
    path = cache_dir / f"{name}-{key}.pt2"
    if path.exists():
        stats["cached"].append(name)
        return str(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    # Unique per builder (process *and* thread): concurrent builds of the same
    # package each write their own temp file, then atomically replace `path`
    # with identical contents, so no builder overwrites or loses another's.
    tmp = cache_dir / f".{name}-{key}.{os.getpid()}.{uuid.uuid4().hex}.tmp.pt2"
    try:
        torch._inductor.aoti_compile_and_package(
            ep, package_path=str(tmp), inductor_configs=inductor_configs
        )
        os.replace(tmp, path)  # atomic: concurrent builders never see a partial file
    finally:
        tmp.unlink(missing_ok=True)  # Only left behind if the compile failed.
    dt = time.perf_counter() - t0
    stats["compiled"].append(name)
    stats["compile_s"][name] = dt
    if verbose:
        print(f"[generic_torch] compiled '{name}' in {dt:.1f}s -> {path}")
    return str(path)


def _check_controls_shape(controls: torch.Tensor, c: int) -> int:
    """K, after checking `controls` (fixed or sampled) is a `[K, c]` tensor with K >= 1."""
    if controls.dim() != 2 or controls.shape[1] != c or controls.shape[0] < 1:
        raise ValueError(
            f"generic_torch: sample_controls must be [K, {c}] with K >= 1, got "
            f"{tuple(controls.shape)}"
        )
    return int(controls.shape[0])


def compile_generic_torch_callbacks(
    configs: Dict[str, Any],
    *,
    device: Optional[str] = None,
    cache_dir: Optional[os.PathLike] = None,
    inductor_configs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Export + AOT-compile the six generic callbacks found in ``configs``.

    Returns the ``_generic_torch`` dict consumed by generic_torch.h:
    ``{"device": str, "packages": {name: path}, "fixed_controls": Tensor|None,
    "stats": {...}}``.
    """
    info = configs["experiment_info_default"]
    node_info = info.get("node_info", {})
    n = int(info["state_dim"])
    c = int(info["control_dim"])
    dev = torch.device(device or node_info.get("device", "cpu"))
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    cache = pathlib.Path(cache_dir) if cache_dir else default_cache_dir()

    # Optional generic_torch-only `expand_fn` replaces dynamics_fn + cost_fn
    # (and the per-expansion validity/heuristic) with one user function, so
    # intermediate rollout results can feed the cost without Python-side caches.
    expand_fn = configs.get("expand_fn")
    required = [
        k for k in CALLBACK_KEYS
        if not (expand_fn is not None and k in ("dynamics_fn", "cost_fn"))
    ]
    missing = [k for k in required if k not in configs]
    if missing:
        raise KeyError(f"generic_torch: missing callbacks on config: {missing}")
    fns: Dict[str, Callable] = {k: configs[k] for k in required}

    stats: Dict[str, Any] = {"compiled": [], "cached": [], "compile_s": {}}
    t_start = time.perf_counter()
    packages: Dict[str, str] = {}

    # --- sample_controls: constant tensor, deterministic fn, or random fn. ---
    sampler = fns["sample_controls_fn"]
    fixed_controls: Optional[torch.Tensor] = None
    if isinstance(sampler, torch.Tensor):
        fixed_controls = sampler.detach().to("cpu", torch.float32).contiguous()
    else:
        # Exporting and probing the sampler runs it, which would advance the
        # global RNG (on cache hits too) and shift a seeded search's first
        # control samples relative to `generic` mode, whose construction never
        # samples. Fork the generators so construction leaves them untouched.
        rng_devices = [dev.index] if dev.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices):
            with torch.device(dev):
                ep_sc = _export("sample_controls", _SampleControls(sampler), (), None)
            if _has_random_ops(ep_sc):
                packages["sample_controls"] = _compile(
                    "sample_controls", ep_sc, dev, cache, inductor_configs, verbose, stats
                )
                # Probe K from the (eager) exported program; same shape every call.
                # Validated like fixed controls: the C++ side reads K * c floats
                # from every sample, so a wrong shape would read out of bounds.
                K = _check_controls_shape(ep_sc.module()(), c)
            else:
                # Deterministic: evaluate once (eager aten, bitwise identical to
                # what generic mode sees each call) and pass as a constant.
                fixed_controls = ep_sc.module()().detach().to("cpu", torch.float32)
    if fixed_controls is not None:
        fixed_controls = fixed_controls.contiguous()
        K = _check_controls_shape(fixed_controls, c)

    # Example batch: any size > 1 works (B is exported as dynamic). Avoid
    # sizes that coincide with the static dims to keep export from unifying.
    ex_b = max(2, K) * 3 + 1
    while ex_b in (n, c, K):
        ex_b += 1
    B = torch.export.Dim("B", min=1, max=_MAX_BATCH)
    states = torch.zeros(ex_b, n, dtype=torch.float32, device=dev)
    controls = torch.zeros(ex_b, c, dtype=torch.float32, device=dev)
    single = torch.zeros(n, dtype=torch.float32, device=dev)

    if expand_fn is not None:
        expand_module: torch.nn.Module = _ExpandUser(expand_fn)
    else:
        expand_module = _Expand(
            fns["dynamics_fn"], fns["cost_fn"], fns["validity_fn"], fns["heuristic_fn"]
        )
    ep_expand = _export("expand", expand_module, (states, controls), ({0: B}, {0: B}))
    ep_valid = _export("validity", _Batched(fns["validity_fn"]), (states,), ({0: B},))
    ep_heur = _export("heuristic", _Batched(fns["heuristic_fn"]), (states,), ({0: B},))
    ep_goal = _export("goal_test", _GoalTest(fns["goal_test_fn"]), (single,), None)

    for name, ep in (
        ("expand", ep_expand),
        ("validity", ep_valid),
        ("heuristic", ep_heur),
        ("goal_test", ep_goal),
    ):
        packages[name] = _compile(name, ep, dev, cache, inductor_configs, verbose, stats)

    stats["total_s"] = time.perf_counter() - t_start
    if verbose:
        print(
            f"[generic_torch] packages ready in {stats['total_s']:.1f}s "
            f"(compiled: {stats['compiled'] or 'none'}, cached: {stats['cached'] or 'none'})"
        )
    return {
        "device": str(dev),
        "packages": packages,
        "fixed_controls": fixed_controls,
        "num_controls": K,
        "stats": stats,
    }
