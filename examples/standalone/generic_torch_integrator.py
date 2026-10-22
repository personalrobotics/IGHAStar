#!/usr/bin/env python3
"""The generic_integrator.py problem, run in the compiled ``generic_torch`` mode.

Same 2D point-mass / circular obstacle / goal region as generic_integrator.py,
with the same six callback signatures. The only differences from the plain
``generic`` version are what torch.export needs:

* ``goal_test`` returns a tensor (``norm < r``) instead of a Python bool.
* captured constants are moved with ``.to(states.device)`` so the same
  callbacks also compile for CUDA (``--device cuda``).

``create_planner`` exports the callbacks with torch.export, compiles them with
AOTInductor (cached under ~/.cache/ighastar/generic_torch, so reruns skip the
compile), and the C++ search calls the compiled code directly -- no Python and
no GIL during search.

The script then solves the identical problem in ``generic`` mode (C++ calling
the same Python functions through pybind11) and prints the timing of both.

Run with:  python3 examples/standalone/generic_torch_integrator.py [--device cuda]
"""
import argparse
import copy
import os
import time

import numpy as np
import torch
import yaml

from ighastar.scripts.common_utils import BASE_DIR, create_planner

# ----------------------------- problem definition ---------------------------
DT = 0.5
GOAL = torch.tensor([8.0, 8.0], dtype=torch.float32)
GOAL_RADIUS = 0.75
LOWER = torch.tensor([0.0, 0.0], dtype=torch.float32)
UPPER = torch.tensor([10.0, 10.0], dtype=torch.float32)
OBSTACLE = torch.tensor([4.0, 4.0], dtype=torch.float32)
OBSTACLE_RADIUS = 1.5

_angles = torch.linspace(0.0, 2.0 * np.pi, 9)[:-1]
_CONTROLS = torch.stack([torch.cos(_angles), torch.sin(_angles)], dim=1).float()


def sample_controls() -> torch.Tensor:
    """Return the fixed set of 8 unit-length control directions, shape (8, 2)."""
    # Deterministic -> evaluated once at build time and baked in as a constant.
    # (A sampler using torch.rand & co. is compiled and re-run every expansion.)
    return _CONTROLS


def dynamics(states: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
    """Point-mass dynamics: advance each state by its control times DT."""
    return states + controls * DT


def cost(states, controls, next_states):
    """Edge cost: Euclidean distance travelled between states and next_states."""
    return torch.norm(next_states - states, dim=1)


def validity(states: torch.Tensor) -> torch.Tensor:
    """Return 1.0 for states inside the bounds and outside the obstacle, else 0.0."""
    lo, hi = LOWER.to(states.device), UPPER.to(states.device)
    in_bounds = (states >= lo).all(dim=1) & (states <= hi).all(dim=1)
    clear = torch.norm(states - OBSTACLE.to(states.device), dim=1) > OBSTACLE_RADIUS
    return (in_bounds & clear).float()


def heuristic(states: torch.Tensor) -> torch.Tensor:
    """Straight-line distance from each state to the goal."""
    return torch.norm(states - GOAL.to(states.device), dim=1)


def goal_test(state: torch.Tensor) -> torch.Tensor:
    """Return a boolean tensor: True if the state is within GOAL_RADIUS of the goal."""
    # Tensor, not bool(...).item(): export cannot trace data-dependent Python.
    return torch.norm(state - GOAL.to(state.device)) < GOAL_RADIUS


CALLBACKS = {
    "sample_controls_fn": sample_controls,
    "dynamics_fn": dynamics,
    "cost_fn": cost,
    "validity_fn": validity,
    "heuristic_fn": heuristic,
    "goal_test_fn": goal_test,
}


def load_configs(node_type: str, device: str = "cpu") -> dict:
    """Load the generic_integrator.yml config and attach the callbacks.

    Args:
        node_type: Planner mode to set, e.g. "generic" or "generic_torch".
        device: Torch device the callbacks run on ("cpu" or "cuda").

    Returns:
        The config dict, including the six callback entries from CALLBACKS.
    """
    yaml_path = os.path.join(
        BASE_DIR.parent, "examples", "standalone", "Configs", "generic_integrator.yml"
    )
    with open(yaml_path) as f:
        configs = yaml.safe_load(f)
    configs["experiment_info_default"]["node_info"]["node_type"] = node_type
    configs["experiment_info_default"]["node_info"]["device"] = device
    configs.update(CALLBACKS)
    return configs


def solve(planner, configs, repeats: int):
    """Run the search `repeats` times from a fixed start.

    Returns:
        (success, expansions, path, best_time_s); path is a numpy array of states
        with the cost-so-far in the last column, or None if the search failed.
    """
    info = configs["experiment_info_default"]
    start = torch.tensor([1.0, 1.0], dtype=torch.float32)
    goal = torch.zeros(info["state_dim"], dtype=torch.float32)  # unused
    world = torch.zeros(1, dtype=torch.float32)  # unused
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ok = planner.search(
            start, goal, world, info["max_expansions"], info["hysteresis"], True
        )
        times.append(time.perf_counter() - t0)
    expansions = planner.get_profiler_info()[7]
    path = planner.get_best_path().numpy() if ok else None
    return ok, expansions, path, min(times)


def main() -> None:
    """Solve the problem with generic_torch, then with generic, and compare."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="torch device for the compiled callbacks",
    )
    parser.add_argument(
        "--repeats", type=int, default=5,
        help="number of searches per mode; the best time is reported",
    )
    parser.add_argument(
        "--skip-generic", action="store_true",
        help="skip the generic (Python callback) run and the comparison",
    )
    args = parser.parse_args()

    configs = load_configs("generic_torch", args.device)
    t0 = time.perf_counter()
    planner = create_planner(copy.copy(configs), bidirectional=False)
    print(f"generic_torch planner ready in {time.perf_counter() - t0:.1f}s "
          "(export + AOTInductor compile/cache + extension build/load)")
    ok, exp, path, t_torch = solve(planner, configs, args.repeats)
    print(f"[generic_torch/{args.device}] success={ok} expansions={exp} "
          f"search={t_torch * 1e3:.2f} ms")
    if ok:
        print(f"  path: {path.shape[0]} states, solution cost {path[0, -1]:.3f}")
        for i in range(0, path.shape[0], max(1, path.shape[0] // 10)):
            print(f"  state {i}: {np.round(path[i, :-1], 3)}  g={path[i, -1]:.3f}")

    if args.skip_generic:
        return
    gconfigs = load_configs("generic")
    gplanner = create_planner(gconfigs, bidirectional=False)
    gok, gexp, gpath, t_generic = solve(gplanner, gconfigs, args.repeats)
    print(f"[generic (py callbacks)] success={gok} expansions={gexp} "
          f"search={t_generic * 1e3:.2f} ms")
    same = (
        ok == gok and exp == gexp
        and (path is None or (path.shape == gpath.shape and np.allclose(path, gpath, atol=1e-5)))
    )
    print(f"same result as generic: {same}")
    print(f"speedup generic_torch vs generic: {t_generic / t_torch:.1f}x "
          f"(best of {args.repeats} searches each)")


if __name__ == "__main__":
    main()
