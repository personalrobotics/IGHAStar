#!/usr/bin/env python3
"""Environment-level unit tests for the generic environments.

Mirrors test_environment.py / test.cpp (validity, successors, goal reaching,
heuristic) for the two callback-driven environments:

* ``generic``       -- generic.h, Python callbacks via py::function
* ``generic_torch`` -- generic_torch.h, the same torch callbacks exported and
  compiled with AOTInductor, run from C++ without Python

Every check runs against both, and the parity tests compare them on identical
inputs. test_generic.cpp is compiled with ``-DUSE_GENERIC_ENV`` or
``-DUSE_GENERIC_TORCH_ENV``.

Run:  python -m pytest tests/unit/environments/test_generic_environment.py -v
"""
import functools
import pathlib
import sys

import pytest
import torch
from torch.utils.cpp_extension import load

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # generic_problems
sys.path.insert(0, str(HERE.parent.parent.parent))  # repo root

import generic_problems as gp  # noqa: E402
from ighastar.scripts.common_utils import BASE_DIR, CXX_STD_FLAG  # noqa: E402
from ighastar.scripts.generic_torch import compile_generic_torch_callbacks  # noqa: E402

MODES = ["generic", "generic_torch"]
MACROS = {"generic": "-DUSE_GENERIC_ENV", "generic_torch": "-DUSE_GENERIC_TORCH_ENV"}


@functools.lru_cache(maxsize=None)
def _test_kernel(mode):
    include = BASE_DIR / "src" / "Environments" / "include"
    extra_includes = [str(include)]
    if sys.platform == "darwin":
        extra_includes.append("/opt/homebrew/opt/boost/include")
    # Suffix the extension name with the dialect so a stale -std=c++20 build
    # (fails on GCC 9) is not reused after switching to c++2a.
    tag = "cxx2a" if "c++2a" in CXX_STD_FLAG else "cxx20"
    return load(
        name=f"{mode}_env_test_2_2_2_{tag}",
        sources=[str(HERE / "test_generic.cpp")],
        extra_include_paths=extra_includes,
        extra_cflags=[
            CXX_STD_FLAG,
            "-O3",
            "-Wno-attributes",
            MACROS[mode],
            "-DN_DIMS=2",
            "-DN_CONT=2",
            "-DHASH_DIMS=2",
        ],
        verbose=False,
    )


def make_tester(mode, **cb_overrides):
    configs = gp.make_configs(mode, gp.callbacks(**cb_overrides))
    if mode == "generic_torch":
        configs["_generic_torch"] = compile_generic_torch_callbacks(configs, verbose=False)
    return _test_kernel(mode).GenericEnvironmentTester(configs)


@pytest.fixture(scope="module", params=MODES)
def tester(request):
    return make_tester(request.param)


@pytest.fixture(scope="module")
def testers():
    return {m: make_tester(m) for m in MODES}


# --------------------------------------------------------------------------
# Test 1/2: validity
# --------------------------------------------------------------------------
def test_validity_valid_states(tester):
    assert tester.check_validity(gp.START, torch.tensor([8.0, 8.0])) == 2


def test_validity_invalid_states(tester):
    in_obstacle = gp.OBSTACLE.clone()
    out_of_bounds = torch.tensor([-1.0, 5.0])
    assert tester.check_validity(in_obstacle, out_of_bounds) == 0


def test_validity_batched_matches_eager(tester):
    g = torch.Generator().manual_seed(0)
    states = torch.rand(257, 2, generator=g) * 12.0 - 1.0
    got = tester.check_validity_batched(states)
    expected = (gp.validity(states) != 0).tolist()
    assert got == expected
    assert any(got) and not all(got)


# --------------------------------------------------------------------------
# Test 3: successor generation
# --------------------------------------------------------------------------
def test_successors(tester):
    poses, g, f = tester.successors(gp.START)
    assert poses.shape == (8, 2)
    # Every successor moved away from the start by DT with edge cost DT.
    assert torch.all(torch.norm(poses - gp.START, dim=1) > 1e-3)
    torch.testing.assert_close(g, torch.full((8,), gp.DT), atol=1e-6, rtol=0)
    torch.testing.assert_close(f, g + gp.heuristic(poses), atol=1e-5, rtol=0)


def test_successors_drop_invalid_children(tester):
    # Next to the lower-left corner: controls pointing out of bounds are pruned.
    poses, _, _ = tester.successors(torch.tensor([0.1, 0.1]))
    assert 0 < poses.shape[0] < 8
    assert bool((gp.validity(poses) != 0).all())


def test_successors_batched_equals_individual(tester):
    parents = torch.tensor([[1.0, 1.0], [2.0, 6.0], [0.1, 0.1], [7.0, 2.5]])
    bp, bg, bf = tester.successors_batched(parents)
    singles = [tester.successors(p) for p in parents]
    torch.testing.assert_close(bp, torch.cat([s[0] for s in singles]))
    torch.testing.assert_close(bg, torch.cat([s[1] for s in singles]))
    torch.testing.assert_close(bf, torch.cat([s[2] for s in singles]))


# --------------------------------------------------------------------------
# Test 4/5: goal reaching
# --------------------------------------------------------------------------
def test_goal_reaching_close(tester):
    assert tester.reached_goal(torch.tensor([8.1, 7.9]))


def test_goal_reaching_far(tester):
    assert not tester.reached_goal(gp.START)


# --------------------------------------------------------------------------
# Test 6: heuristic
# --------------------------------------------------------------------------
def test_heuristic(tester):
    h = tester.heuristic(gp.START)
    assert h >= 0.0
    assert abs(h - float(gp.heuristic(gp.START[None])[0])) < 1e-5
    assert tester.heuristic(gp.GOAL) < 1e-6


# --------------------------------------------------------------------------
# Parity: generic (Python callbacks) vs generic_torch (compiled)
# --------------------------------------------------------------------------
def test_parity_successors(testers):
    g = torch.Generator().manual_seed(1)
    parents = torch.rand(64, 2, generator=g) * 10.0
    a = testers["generic"].successors_batched(parents)
    b = testers["generic_torch"].successors_batched(parents)
    for x, y in zip(a, b):
        assert x.shape == y.shape
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-6)


def test_parity_heuristic_goal_validity(testers):
    g = torch.Generator().manual_seed(2)
    states = torch.rand(128, 2, generator=g) * 10.0
    tg, tt = testers["generic"], testers["generic_torch"]
    assert tg.check_validity_batched(states) == tt.check_validity_batched(states)
    for s in states[:32]:
        assert abs(tg.heuristic(s) - tt.heuristic(s)) < 1e-6
        assert tg.reached_goal(s) == tt.reached_goal(s)


def test_random_sampler_parity():
    """A random sample_controls is compiled with fallback_random, so it draws
    from the global ATen generator exactly like the eager callback: seeded
    expansions are bitwise identical across modes and reproducible."""
    t = {m: make_tester(m, sample_controls=gp.sample_controls_random) for m in MODES}
    parents = torch.tensor([[1.0, 1.0], [2.0, 6.0], [7.0, 2.5]])
    out = {}
    for m in MODES:
        torch.manual_seed(7)
        first = t[m].successors_batched(parents)
        second = t[m].successors_batched(parents)  # fresh controls
        torch.manual_seed(7)
        again = t[m].successors_batched(parents)
        assert not torch.equal(first[0], second[0])
        for x, y in zip(first, again):
            assert torch.equal(x, y)
        out[m] = first
    for x, y in zip(out["generic"], out["generic_torch"]):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_short_resolution_raises(mode):
    """Config vectors shorter than n_dims must fail at construct, not OOB later."""
    configs = gp.make_configs(mode, gp.callbacks(), resolution=[0.5])
    if mode == "generic_torch":
        configs["_generic_torch"] = compile_generic_torch_callbacks(configs, verbose=False)
    with pytest.raises(RuntimeError, match="resolution"):
        _test_kernel(mode).GenericEnvironmentTester(configs)


@pytest.mark.parametrize("mode", MODES)
def test_bounds_reject_out_of_box_children(mode):
    """bounds_lower/upper drop children even when validity_fn would accept them."""

    def always_valid(states):
        return torch.ones(states.shape[0], dtype=torch.float32, device=states.device)

    # Tight box around the start so a +DT control step exits the box.
    configs = gp.make_configs(
        mode,
        gp.callbacks(validity=always_valid),
        bounds_lower=[0.9, 0.9],
        bounds_upper=[1.1, 1.1],
    )
    if mode == "generic_torch":
        configs["_generic_torch"] = compile_generic_torch_callbacks(configs, verbose=False)
    t = _test_kernel(mode).GenericEnvironmentTester(configs)
    poses, _, _ = t.successors(gp.START)
    assert poses.shape[0] == 0


@pytest.mark.parametrize("mode", MODES)
def test_rollout_dynamics_successors(mode):
    """dynamics returning [B, T, n]: node state is the last sub-step."""
    t = make_tester(mode, dynamics=gp.dynamics_rollout)
    poses, g, _ = t.successors(gp.START)
    expected = gp.dynamics(gp.START.expand(8, 2), gp.CONTROLS)
    torch.testing.assert_close(poses, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(g, torch.full((8,), gp.DT), atol=1e-6, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
