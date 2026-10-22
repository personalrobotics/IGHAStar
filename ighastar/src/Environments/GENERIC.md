# Generic environment (Python callbacks)

Use the **generic** environment when you define dynamics, cost, validity, heuristic, and goal test in Python. You do not write a C++ `Environment` class.

The planner builds a Torch extension with your state and control sizes. Your callbacks run on batched `torch.Tensor` values.

For a C++ environment instead, see [README.md](README.md).

## Glossary

| Term | Meaning |
|---|---|
| `generic` | Environment type set by `node_info.node_type: "generic"` |
| callback | Python function the planner calls during search |
| edge | One control held from one state to the next state |
| `g` | Path cost from the start to a node |

## When to use generic

Use generic when:

- You own the dynamics model in Python (or JAX, NumPy, MuJoCo, and similar).
- You want to change cost or validity without a C++ rebuild of those rules.
- Your world model lives in Python closures (not a map tensor).

Use a built-in C++ environment (`kinematic`, `kinodynamic`, `simple`, `holonomic_6D`) when you need the packaged car or map pipelines.

## Required callbacks

Attach these six callables on the config dict. YAML cannot store them. Attach them in Python after you load the YAML.

| Key | Signature | Role |
|---|---|---|
| `sample_controls_fn` | `() -> Tensor[K, n_cont]` | Returns the control set for one expansion |
| `dynamics_fn` | `(states[B, n], controls[B, n_cont]) -> [B, n]` or `[B, T, n]` | Rolls each state with each control |
| `cost_fn` | `(states, controls, next_states) -> [B]` | Edge cost for the **final** state of each rollout |
| `validity_fn` | `(states[B, n]) -> [B]` | Nonzero means valid |
| `heuristic_fn` | `(states[B, n]) -> [B]` | Cost-to-go. Encode goal knowledge here |
| `goal_test_fn` | `(state[n]) -> bool` | True when the state is in the goal region |

### Dynamics shapes

- Return `[B, n]` for a single step.
- Return `[B, T, n]` for a multi-step rollout. The node state is the last step. The planner stores intermediate steps on the path.
- `cost_fn`, `validity_fn`, and `heuristic_fn` always see the **final** state.

### Goal and world tensors

For generic search, `goal_test_fn` and `heuristic_fn` own the goal. The planner does not use the `goal` tensor for that test. The planner does not use the `world` tensor. Capture maps or simulators inside your callbacks.

## Config keys

Set these under `experiment_info_default` (see `examples/standalone/Configs/generic_integrator.yml`):

| Key | Role |
|---|---|
| `state_dim` | State size (`n`). Compiled as `-DN_DIMS` |
| `control_dim` | Control size (`n_cont`). Compiled as `-DN_CONT` |
| `hash_dims` | Leading dims used in the grid hash (default: all state dims) |
| `num_controls` | Expected branching factor `K` |
| `resolution` | Start grid size per dim (length `state_dim`) |
| `tolerance` | Dedup tolerance per dim |
| `bounds_lower` / `bounds_upper` | State bounds (length `state_dim`) |
| `max_level` | Number of resolution levels |
| `division_factor` | Resolution shrink factor per level |
| `max_expansions` | Search budget |
| `hysteresis` | Resolution-switch threshold |
| `node_info.node_type` | Must be `"generic"` |
| `node_info.timesteps` | Hint only. Multi-step `T` comes from `dynamics_fn` output |

`create_planner` rebuilds the extension when `(state_dim, control_dim, hash_dims)` change. Keep those three stable across runs when you can.

## How to plan

1. Load a YAML config with `node_type: "generic"`.
2. Attach the six callbacks on the config dict.
3. Call `create_planner(configs, bidirectional=False)` for IGHA\*, or `bidirectional=True` for BiIGHA\*.
4. Call `planner.search(start, goal, world, max_expansions, hysteresis, ignore_goal_validity)`.
5. If search returns true, read the path with `get_best_path()` or `get_best_path_with_controls()`.

Minimal example:

```python
import torch
import yaml
from ighastar.scripts.common_utils import create_planner

with open("examples/standalone/Configs/generic_integrator.yml") as f:
    configs = yaml.safe_load(f)

configs["sample_controls_fn"] = sample_controls
configs["dynamics_fn"] = dynamics
configs["cost_fn"] = cost
configs["validity_fn"] = validity
configs["heuristic_fn"] = heuristic
configs["goal_test_fn"] = goal_test

planner = create_planner(configs, bidirectional=False)
info = configs["experiment_info_default"]
start = torch.tensor([1.0, 1.0], dtype=torch.float32)
goal = torch.zeros(info["state_dim"], dtype=torch.float32)  # unused by goal_test
world = torch.zeros(1, dtype=torch.float32)                 # unused by generic

ok = planner.search(
    start, goal, world, info["max_expansions"], info["hysteresis"], True
)
if ok:
    path = planner.get_best_path()  # [path_len, state_dim + 1], last column is g
```

Run the full point-mass demo:

```bash
python3 examples/standalone/generic_integrator.py
```

## Path output

- `get_best_path()` returns `[path_len, state_dim + 1]`. The last column is `g`.
- Rows are **goal-first**, start-last. Solution cost is `path[0, -1]`.
- `get_best_path_with_controls()` also returns the control sequence along that path.

## Batching rules

The planner expands several nodes at once. For `M` nodes and `K` controls it builds `B = M * K` rows:

1. Call `sample_controls_fn()` once. Expect shape `[K, n_cont]`.
2. Call `dynamics_fn(states[B, n], controls[B, n_cont])`.
3. Call `cost_fn`, `validity_fn`, and `heuristic_fn` on the final states.

Write every callback to accept a batch on dim 0. Do not assume `B == 1`.

## JAX and other backends

Callbacks must accept and return `torch.Tensor` (CPU float32 is enough). Convert as needed:

1. `tensor.detach().cpu().numpy()`
2. Run your model (JAX, NumPy, MuJoCo, …).
3. `torch.from_numpy(...).float()`

For a multi-step edge with per-step rewards, return `[B, T, n]` from `dynamics_fn`. Put the scalar edge cost in `cost_fn` for the final state. You may store intermediate costs in a closure. See a heavier pattern in an external Sumo Spot runner if you use that stack.

## Bidirectional search

Pass `bidirectional=True` to `create_planner`. BiIGHA\* uses the same six callbacks. Near-meet distance for generic is Euclidean on the full state. Tune `tolerance` and `resolution` if meet detection is too strict or too loose.

## generic_torch: compiled torch callbacks

Requires **torch ≥ 2.6** (`aoti_compile_and_package` + `AOTIModelPackageLoader`).

Set `node_info.node_type: "generic_torch"` to keep the same six callbacks but compile them. `create_planner` then:

1. Exports each callback with `torch.export`, with a dynamic batch dimension.
2. Compiles the exported graphs ahead of time with AOTInductor into `.pt2` packages.
3. Builds the planner with `-DUSE_GENERIC_TORCH_ENV` (`generic_torch.h`), which loads the packages with `torch::inductor::AOTIModelPackageLoader` and runs them from C++.

The search loop never calls Python, and `search()` releases the GIL while it runs. Dynamics, cost, validity and heuristic are fused into one compiled call per batched expansion.

```python
configs["experiment_info_default"]["node_info"]["node_type"] = "generic_torch"
configs["experiment_info_default"]["node_info"]["device"] = "cpu"   # or "cuda"
configs["sample_controls_fn"] = sample_controls   # same six keys as generic
...
planner = create_planner(configs)                 # export + compile (cached) + build
ok = planner.search(start, goal, world, max_expansions, hysteresis, True)
```

### What the callbacks must satisfy

The callbacks must be `torch.export`-able:

- Use pure torch ops on the input tensors. Do not use `.item()`, `bool(tensor)`, `.tolist()` or NumPy.
- Do not branch in Python on tensor values. Use `torch.where` or masks instead.
- `goal_test_fn` must return a tensor, for example `torch.norm(s - goal) < r`. A tensor-returning `goal_test_fn` also works in `generic` mode, so the same functions can serve both modes.
- Do not specialize on the batch size `B`. The batch dimension is exported as dynamic.
- Python state that the callbacks read (globals, closures, counters) is frozen into the graph at export time. If you change it, build a new planner. Changing a captured tensor changes the cache key, so the new planner recompiles.
- For CUDA, make the callbacks device-agnostic by moving captured constants with `.to(states.device)`.

If export fails, `create_planner` raises a `RuntimeError` that names the callback.

Generic-mode callbacks sometimes share results through Python-side caches. For example, `dynamics_fn` stores per-step collision counts for `cost_fn` to read. That pattern cannot be compiled. Instead, pass the optional `expand_fn(states, controls) -> (next_states, cost, valid, heuristic)`, which replaces `dynamics_fn` and `cost_fn` for successor generation. The rollout and the cost then live in one compiled graph. `validity_fn` and `heuristic_fn` are still required for the start and goal checks. `expand_fn` is available only in `generic_torch`.

### Parity with generic

Inductor may fuse or reorder float operations, for example inside a norm reduction. The compiled callbacks can therefore differ from eager torch in the last bit. In the integrator example and the tests, results are bitwise identical: same expansions, path and cost. In a 30-obstacle variant, the compiled run expanded 456 nodes against 460 for `generic`, with the same solution cost. Treat `generic_torch` as equivalent up to float rounding, not bitwise identical.

### Controls and randomness

| `sample_controls_fn` | Behavior |
|---|---|
| A `Tensor[K, n_cont]` | Used as the fixed control set |
| A function with no random ops | Evaluated once at build time and used as a fixed set |
| A function with random ops (`torch.rand`, ...) | Compiled and re-run on every expansion |

Graphs that contain random ops are compiled with Inductor's `fallback_random=True`. Random ops then draw from the global ATen generator, so `torch.manual_seed` controls them and they produce the same numbers as the eager callback in `generic` mode. Random ops inside dynamics or cost follow the same rule.

### Caching and compile time

Packages are cached in `~/.cache/ighastar/generic_torch`. Set `IGHASTAR_AOTI_CACHE` to use another directory. The cache key hashes the exported graph, every captured tensor value, the device and the torch version. Editing comments or moving code does not trigger a recompile. Changing what is computed does.

- First build: about 4-10 s per package on CPU, plus about 20 s for the C++ extension. The extension is cached in `~/.cache/torch_extensions` as `ighastar_generic_torch_{state_dim}_{control_dim}_{hash_dims}`.
- Later runs: about 1 s to re-export and hash the callbacks.

### Device

`node_info.device` defaults to `"cpu"`. With `"cuda"`, the packages run on the GPU, and each call copies its batch to the device and back. That copy and the kernel-launch latency make CUDA slower than CPU for small batches like the integrator. CUDA helps only when individual callbacks are expensive.

### Bidirectional search

`bidirectional=True` works because both search threads call compiled code with no Python. The goal-test caveat still applies. Both directions share `goal_test_fn`, so a backward search that starts inside the goal region terminates at once. For BiIGHA\*, use a goal test that never fires and let near-meet connect the two trees.

## Working examples

| File | What it shows |
|---|---|
| `examples/standalone/generic_integrator.py` | Minimal 2D integrator |
| `examples/standalone/generic_torch_integrator.py` | Same problem in `generic_torch` mode, timed against `generic` |
| `examples/standalone/Configs/generic_integrator.yml` | Generic YAML keys |
| `examples/standalone/generic_ur5e_mjx.py` | Skeleton for a MuJoCo MJX arm |
| `ighastar/src/Environments/include/generic.h` | Callback contract in the C++ header |
| `ighastar/src/Environments/include/generic_torch.h` | Compiled-callback environment |
| `ighastar/scripts/generic_torch.py` | Export, compile and cache logic |

## Related

- C++ `Environment` interface: [README.md](README.md)
- Standalone usage notes: [examples/standalone/README.md](../../../../examples/standalone/README.md)
