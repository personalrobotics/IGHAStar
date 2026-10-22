#!/usr/bin/env python3
"""Planner-level tests for the compiled ``generic_torch`` mode.

* end-to-end search on the generic_integrator problem
* parity with ``generic`` (same torch callbacks called from C++ via pybind11):
  identical expansion counts, paths, controls and cost
* the search never calls Python (counting / raising callbacks, GIL released)
* multi-step rollouts, random control samplers, bidirectional search
* AOTInductor package caching and export error messages
* CUDA (skipped when unavailable)

Run:  python -m pytest tests/unit/planner/test_generic_torch_planner.py -v
"""
import pathlib
import sys
import threading
import time

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # generic_problems
sys.path.insert(0, str(HERE.parent.parent.parent))  # repo root

import generic_problems as gp  # noqa: E402
from ighastar.scripts import generic_torch as gt  # noqa: E402
from ighastar.scripts.common_utils import create_planner  # noqa: E402


def plan(mode, cbs, bidirectional=False, device="cpu", **info):
    configs = gp.make_configs(mode, cbs, device=device, **info)
    planner = create_planner(configs, bidirectional=bidirectional)
    return planner, configs


def solve(mode, cbs, **kw):
    planner, configs = plan(mode, cbs, **kw)
    ok = gp.run_search(planner, configs)
    path, controls = planner.get_best_path_with_controls()
    return {
        "ok": ok,
        "expansions": planner.get_profiler_info()[7],
        "path": path,
        "controls": controls,
        "planner": planner,
        "configs": configs,
    }


def assert_same_result(a, b, exact=True):
    assert a["ok"] and b["ok"]
    assert a["expansions"] == b["expansions"]
    assert a["path"].shape == b["path"].shape
    assert a["controls"].shape == b["controls"].shape
    if exact:
        assert torch.equal(a["path"], b["path"])
        assert torch.equal(a["controls"], b["controls"])
    else:
        torch.testing.assert_close(a["path"], b["path"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(a["controls"], b["controls"], atol=1e-6, rtol=0)


@pytest.fixture(scope="module")
def generic_result():
    return solve("generic", gp.callbacks())


@pytest.fixture(scope="module")
def torch_result():
    return solve("generic_torch", gp.callbacks())


# --------------------------------------------------------------------------
# Basic search + parity
# --------------------------------------------------------------------------
def test_generic_torch_finds_path(torch_result):
    r = torch_result
    assert r["ok"]
    path = r["path"]
    # Goal-first: first row is inside the goal region, last row one step from start.
    assert float(torch.norm(path[0, :2] - gp.GOAL)) < gp.GOAL_RADIUS
    assert float(torch.norm(path[-1, :2] - gp.START)) <= gp.DT + 1e-5
    # Straight-line lower bound is ~9.9 - goal radius; detour around obstacle.
    cost = float(path[0, -1])
    assert 9.0 < cost < 13.0
    # Every waypoint is valid.
    assert bool((gp.validity(path[:, :2]) != 0).all())


def test_parity_with_generic(generic_result, torch_result):
    # Inductor fuses the callbacks, but for this problem the float32 results
    # are bitwise identical, so the searches match exactly.
    assert_same_result(generic_result, torch_result, exact=True)


def test_parity_rollout_dynamics():
    cbs = gp.callbacks(dynamics=gp.dynamics_rollout)
    a = solve("generic", cbs)
    b = solve("generic_torch", cbs)
    assert_same_result(a, b, exact=False)
    # Dense path: ROLLOUT_T sub-steps per edge.
    n_edges = round(float(b["path"][0, -1]) / gp.DT)
    assert b["path"].shape[0] == n_edges * gp.ROLLOUT_T


def test_repeated_search_is_deterministic(torch_result):
    r = torch_result
    ok = gp.run_search(r["planner"], r["configs"])
    assert ok
    assert r["planner"].get_profiler_info()[7] == r["expansions"]
    assert torch.equal(r["planner"].get_best_path(), r["path"])


# --------------------------------------------------------------------------
# The search must not call back into Python
# --------------------------------------------------------------------------
def test_search_does_not_call_python():
    counter = gp.CallCounter(gp.callbacks())
    planner, configs = plan("generic_torch", counter.wrapped)
    # torch.export traced every callback while building the planner.
    assert counter.total() > 0
    counter.reset()
    counter.armed = True  # any Python call now raises (and is counted)
    ok = gp.run_search(planner, configs)
    counter.armed = False
    assert ok
    assert planner.get_profiler_info()[7] > 100
    assert counter.total() == 0, counter.counts


def test_generic_mode_does_call_python():
    """Sanity check of the counter itself: generic mode calls every callback."""
    counter = gp.CallCounter(gp.callbacks())
    planner, configs = plan("generic", counter.wrapped)
    assert gp.run_search(planner, configs)
    for key in ("dynamics_fn", "cost_fn", "validity_fn", "heuristic_fn", "goal_test_fn"):
        assert counter.counts[key] > 0, key


def test_search_releases_gil():
    """A Python thread keeps running while generic_torch searches."""
    ticks = []
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            ticks.append(time.perf_counter())

    # Unreachable goal -> the search runs until the expansion budget is spent.
    planner, configs = plan("generic_torch", gp.callbacks(goal_test=gp.goal_test_never))
    th = threading.Thread(target=spin)
    th.start()
    time.sleep(0.01)
    t0 = time.perf_counter()
    ok = planner.search(gp.START.clone(), torch.zeros(2), torch.zeros(1), 20000, 1000, True)
    t1 = time.perf_counter()
    stop.set()
    th.join()
    assert not ok
    during = [t for t in ticks if t0 < t < t1]
    # With the GIL held for the whole search there would be ~no ticks inside.
    assert len(during) > 100, (t1 - t0, len(during))


# --------------------------------------------------------------------------
# Controls: random sampler
# --------------------------------------------------------------------------
def test_inplace_buffer_sampler_is_rejected():
    """Refilling one tensor in place must not be frozen as fixed controls.

    Without a pre-second-call snapshot, torch.equal(a, b) is vacuously true
    after the second write (same storage). That path must not freeze. In-place
    mutation also cannot be AOT-exported, so build fails with a clear error.
    Returning the same module-level Tensor every call remains allowed.
    """
    buf = torch.empty(8, 2)

    def sample_inplace():
        buf.uniform_(-1.0, 1.0)
        return buf

    with pytest.raises(TypeError, match="in place"):
        gt.compile_generic_torch_callbacks(
            gp.make_configs(
                "generic_torch", gp.callbacks(sample_controls=sample_inplace)
            ),
            verbose=False,
        )


def test_random_sample_controls_search():
    """A random sampler is compiled (not frozen) and re-run every expansion.

    Bitwise RNG parity with generic mode is checked at the environment level
    (test_generic_environment.py::test_random_sampler_parity). Whole searches
    with random controls are not compared: the planner core iterates
    pointer-keyed unordered_sets (inactive_Q_v), so with non-grid-aligned
    successors even two generic-mode runs with the same seed can expand nodes
    in different orders.
    """
    cbs = gp.callbacks(sample_controls=gp.sample_controls_random)
    compiled = gt.compile_generic_torch_callbacks(
        gp.make_configs("generic_torch", cbs), verbose=False
    )
    assert "sample_controls" in compiled["packages"]
    assert compiled["fixed_controls"] is None
    counter = gp.CallCounter(cbs)
    planner, configs = plan("generic_torch", counter.wrapped)
    counter.reset()
    torch.manual_seed(0)
    assert gp.run_search(planner, configs)
    assert counter.total() == 0
    path, controls = planner.get_best_path_with_controls()
    assert 9.0 < float(path[0, -1]) < 13.0
    # Controls along the path are unit vectors, but not from the fixed 8-set.
    torch.testing.assert_close(controls.norm(dim=1), torch.ones(len(controls)))
    dists = torch.cdist(controls, gp.CONTROLS).min(dim=1).values
    assert bool((dists > 1e-4).any())


def test_user_expand_fn_replaces_dynamics_and_cost(generic_result):
    cbs = gp.callbacks()
    del cbs["dynamics_fn"], cbs["cost_fn"]
    cbs["expand_fn"] = gp.expand
    r = solve("generic_torch", cbs)
    assert_same_result(generic_result, r, exact=True)


def test_fixed_controls_tensor_accepted():
    a = solve("generic_torch", gp.callbacks(sample_controls=gp.CONTROLS.clone()))
    b = solve("generic", gp.callbacks())
    assert_same_result(a, b, exact=True)


# --------------------------------------------------------------------------
# Bidirectional
# --------------------------------------------------------------------------
def test_bidirectional_generic_torch():
    cbs = gp.callbacks(goal_test=gp.goal_test_never)
    # A 2D point mass: Euclidean goal sampling, no angular dimension (the SE2 default
    # treats dimension 2 as the heading, which this state doesn't have).
    planner, configs = plan(
        "generic_torch",
        cbs,
        bidirectional=True,
        goal_sampling_config={"space_type": "R_N", "angular_dims": []},
    )
    ok = planner.search(
        gp.START.clone(), gp.GOAL.clone(), torch.zeros(1), 50000, 1000, True
    )
    assert ok
    cost = planner.get_best_cost()
    assert 9.0 < cost < 13.0
    path = planner.get_best_path()
    assert path.shape[0] > 0


def test_bidirectional_rejects_angular_dims_outside_the_state():
    # Dimension 2 doesn't exist in this 2D state: sampling around it would read past the pose.
    cbs = gp.callbacks(goal_test=gp.goal_test_never)
    with pytest.raises(ValueError, match="angular_dims"):
        plan(
            "generic_torch",
            cbs,
            bidirectional=True,
            goal_sampling_config={"space_type": "SE2", "angular_dims": [2]},
        )


# --------------------------------------------------------------------------
# Compilation cache + errors
# --------------------------------------------------------------------------
def test_packages_are_cached(torch_result, monkeypatch):
    configs = gp.make_configs("generic_torch", gp.callbacks())

    def boom(*a, **k):
        raise AssertionError("AOTInductor recompiled a cached package")

    monkeypatch.setattr(torch._inductor, "aoti_compile_and_package", boom)
    out = gt.compile_generic_torch_callbacks(configs, verbose=False)
    assert out["stats"]["compiled"] == []
    assert sorted(out["stats"]["cached"]) == ["expand", "goal_test", "heuristic", "validity"]
    for p in out["packages"].values():
        assert pathlib.Path(p).exists()


def test_cache_key_tracks_captured_constants():
    B = torch.export.Dim("B", min=1, max=1 << 20)
    x = torch.zeros(5, 2)

    def key_for(goal):
        mod = gt._Batched(lambda s: torch.norm(s - goal, dim=1))
        ep = torch.export.export(mod, (x,), dynamic_shapes=({0: B},))
        return gt._program_key(ep, "heuristic|cpu")

    assert key_for(torch.tensor([8.0, 8.0])) == key_for(torch.tensor([8.0, 8.0]))
    assert key_for(torch.tensor([8.0, 8.0])) != key_for(torch.tensor([8.0, 7.0]))


def test_python_bool_goal_test_gives_clear_error():
    def bad_goal_test(state):
        return bool(torch.norm(state - gp.GOAL).item() < gp.GOAL_RADIUS)

    configs = gp.make_configs("generic_torch", gp.callbacks(goal_test=bad_goal_test))
    with pytest.raises(RuntimeError, match="goal_test"):
        gt.compile_generic_torch_callbacks(configs, verbose=False)


@pytest.mark.parametrize("shape", [(8, 1), (8, 3), (16,), (0, 2)])
def test_random_sampler_shape_is_validated(shape):
    # A random sampler is compiled rather than passed as a constant, so its shape is only
    # known from a probe; the C++ side reads K * control_dim floats per sample, so a
    # mismatched shape must be rejected up front.
    def bad_sampler():
        return torch.rand(*shape)

    configs = gp.make_configs("generic_torch", gp.callbacks(sample_controls=bad_sampler))
    with pytest.raises(ValueError, match=r"sample_controls must be \[K, 2\]"):
        gt.compile_generic_torch_callbacks(configs, verbose=False)


def test_missing_callback_raises():
    cbs = gp.callbacks()
    del cbs["cost_fn"]
    configs = gp.make_configs("generic_torch", cbs)
    with pytest.raises(KeyError, match="cost_fn"):
        gt.compile_generic_torch_callbacks(configs, verbose=False)


# --------------------------------------------------------------------------
# CUDA
# --------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu(torch_result):
    r = solve("generic_torch", gp.callbacks(), device="cuda")
    # CUDA kernels may round differently (e.g. sqrt/fma), which can change
    # tie-breaking; require the same solution quality, not identical search.
    assert r["ok"]
    assert abs(float(r["path"][0, -1]) - float(torch_result["path"][0, -1])) < 1e-3
    if r["expansions"] == torch_result["expansions"]:
        torch.testing.assert_close(r["path"], torch_result["path"], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
