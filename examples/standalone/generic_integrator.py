#!/usr/bin/env python3
"""End-to-end test of the generic (OMPL-style) IGHAStar environment.

A 2D point-mass must travel from a start state to a goal region while avoiding
a circular obstacle, using a user-defined 8-direction control set. All of the
problem definition lives in plain Python callbacks operating on batched torch
tensors:

    sample_controls() -> [K, n_cont]
    dynamics(states[B, n_dims], controls[B, n_cont]) -> next_states[B, n_dims]
    cost(states, controls, next_states) -> [B]
    validity(states[B, n_dims]) -> [B]   (nonzero == valid)
    heuristic(states[B, n_dims]) -> [B]
    goal_test(state[n_dims]) -> bool

Run with:  python3 examples/standalone/generic_integrator.py
"""
import os

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

# Fixed 8-direction unit control set. (The user decides whether to return a
# fixed set or freshly sampled controls each call.)
_angles = torch.linspace(0.0, 2.0 * np.pi, 9)[:-1]
_CONTROLS = torch.stack([torch.cos(_angles), torch.sin(_angles)], dim=1).float()


def sample_controls() -> torch.Tensor:
    return _CONTROLS


def dynamics(states: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
    # Single integrator: move one step of length DT in the control direction.
    return states + controls * DT


def cost(states, controls, next_states):
    return torch.norm(next_states - states, dim=1)


def validity(states: torch.Tensor) -> torch.Tensor:
    in_bounds = (states >= LOWER).all(dim=1) & (states <= UPPER).all(dim=1)
    clear = torch.norm(states - OBSTACLE, dim=1) > OBSTACLE_RADIUS
    return (in_bounds & clear).float()


def heuristic(states: torch.Tensor) -> torch.Tensor:
    return torch.norm(states - GOAL, dim=1)


def goal_test(state: torch.Tensor) -> bool:
    return bool(torch.norm(state - GOAL).item() < GOAL_RADIUS)


def main() -> None:
    yaml_path = os.path.join(
        BASE_DIR.parent, "examples", "standalone", "Configs", "generic_integrator.yml"
    )
    with open(yaml_path) as f:
        configs = yaml.safe_load(f)

    # Attach the user callbacks (cannot be stored in YAML).
    configs["sample_controls_fn"] = sample_controls
    configs["dynamics_fn"] = dynamics
    configs["cost_fn"] = cost
    configs["validity_fn"] = validity
    configs["heuristic_fn"] = heuristic
    configs["goal_test_fn"] = goal_test

    info = configs["experiment_info_default"]
    expansion_limit = info["max_expansions"]
    hysteresis = info["hysteresis"]
    state_dim = info["state_dim"]

    planner = create_planner(configs, bidirectional=False)

    start = torch.tensor([1.0, 1.0], dtype=torch.float32)
    goal = torch.zeros(state_dim, dtype=torch.float32)  # unused by goal_test
    world = torch.zeros(1, dtype=torch.float32)  # unused by generic env

    success = planner.search(start, goal, world, expansion_limit, hysteresis, True)
    print("success:", success)

    (
        avg_successor_time,
        avg_goal_check_time,
        avg_overhead_time,
        avg_g_update_time,
        switches,
        max_level_profile,
        q_v_size,
        expansion_counter,
        expansion_list,
        cost_exp_list,
    ) = planner.get_profiler_info()
    print(f"expansions: {expansion_counter}, max_level: {max_level_profile}")
    print(f"preemptive expansions: {planner.get_preemptive_expansions()}")

    if success:
        path = planner.get_best_path().numpy()
        # Path is returned goal-first -> start-last; the last column is g
        # (cost-from-start), so the solution cost is the goal node's g = path[0].
        print(f"path: {path.shape[0]} states, solution cost {path[0, -1]:.3f}")
        # Print a few waypoints
        for i in range(0, path.shape[0], max(1, path.shape[0] // 10)):
            print(f"  state {i}: {np.round(path[i, :-1], 3)}  g={path[i, -1]:.3f}")
    else:
        print("No path found.")


if __name__ == "__main__":
    main()
