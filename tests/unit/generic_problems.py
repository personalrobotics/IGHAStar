"""Shared problem definitions for the generic / generic_torch tests.

All callbacks are plain torch functions that work in *both* modes:
``generic`` calls them from C++ through pybind11, ``generic_torch`` exports and
AOT-compiles them. The only concessions to torch.export are a tensor-returning
``goal_test`` and device-agnostic constants.
"""
import copy
import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

DT = 0.5
GOAL = torch.tensor([8.0, 8.0])
GOAL_RADIUS = 0.75
LOWER = torch.tensor([0.0, 0.0])
UPPER = torch.tensor([10.0, 10.0])
OBSTACLE = torch.tensor([4.0, 4.0])
OBSTACLE_RADIUS = 1.5
ROLLOUT_T = 4

START = torch.tensor([1.0, 1.0])

_angles = torch.linspace(0.0, 2.0 * math.pi, 9)[:-1]
CONTROLS = torch.stack([torch.cos(_angles), torch.sin(_angles)], dim=1).float()


def sample_controls() -> Tensor:
    return CONTROLS


def sample_controls_random() -> Tensor:
    # Random unit directions, fresh every expansion.
    ang = torch.rand(8) * (2.0 * math.pi)
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)


def dynamics(states: Tensor, controls: Tensor) -> Tensor:
    return states + controls * DT


def dynamics_rollout(states: Tensor, controls: Tensor) -> Tensor:
    # [B, T, n]: T equal sub-steps whose last step equals dynamics().
    steps = torch.arange(1, ROLLOUT_T + 1, dtype=states.dtype, device=states.device)
    frac = (steps / ROLLOUT_T).reshape(1, ROLLOUT_T, 1)
    return states.unsqueeze(1) + controls.unsqueeze(1) * DT * frac


def cost(states: Tensor, controls: Tensor, next_states: Tensor) -> Tensor:
    return torch.norm(next_states - states, dim=1)


def validity(states: Tensor) -> Tensor:
    lo, hi = LOWER.to(states.device), UPPER.to(states.device)
    in_bounds = (states >= lo).all(dim=1) & (states <= hi).all(dim=1)
    clear = torch.norm(states - OBSTACLE.to(states.device), dim=1) > OBSTACLE_RADIUS
    return (in_bounds & clear).float()


def heuristic(states: Tensor) -> Tensor:
    return torch.norm(states - GOAL.to(states.device), dim=1)


def goal_test(state: Tensor) -> Tensor:
    return torch.norm(state - GOAL.to(state.device)) < GOAL_RADIUS


def expand(states: Tensor, controls: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # generic_torch-only fused successor function (replaces dynamics + cost).
    nxt = dynamics(states, controls)
    return nxt, cost(states, controls, nxt), validity(nxt), heuristic(nxt)


def goal_test_never(state: Tensor) -> Tensor:
    # For bidirectional search: let the near-meet logic connect the trees.
    return (state[0] != state[0]) & (state[0] == state[0])


def callbacks(**overrides: Callable) -> Dict[str, Callable]:
    cb = {
        "sample_controls_fn": sample_controls,
        "dynamics_fn": dynamics,
        "cost_fn": cost,
        "validity_fn": validity,
        "heuristic_fn": heuristic,
        "goal_test_fn": goal_test,
    }
    cb.update({f"{k}_fn" if not k.endswith("_fn") else k: v for k, v in overrides.items()})
    return cb


def make_configs(
    node_type: str, cbs: Dict[str, Callable], device: str = "cpu", **info_overrides: Any
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "state_dim": 2,
        "control_dim": 2,
        "hash_dims": 2,
        "num_controls": 8,
        "resolution": [0.5, 0.5],
        "tolerance": [0.1, 0.1],
        "bounds_lower": [0.0, 0.0],
        "bounds_upper": [10.0, 10.0],
        "max_level": 4,
        "division_factor": 2.0,
        "max_expansions": 50000,
        "hysteresis": 1000,
        "preemptive_expansion": {"enabled": False, "min_preemptive": 8, "max_preemptive": 32},
        "node_info": {"node_type": node_type, "timesteps": 1, "device": device},
    }
    info.update(info_overrides)
    configs: Dict[str, Any] = {"experiment_info_default": copy.deepcopy(info)}
    configs.update(cbs)
    return configs


def run_search(
    planner: Any, configs: Dict[str, Any], start: Tensor = START, goal: Optional[Tensor] = None
) -> bool:
    info = configs["experiment_info_default"]
    if goal is None:
        goal = torch.zeros(info["state_dim"])
    ok = planner.search(
        start.clone(), goal.clone(), torch.zeros(1), info["max_expansions"], info["hysteresis"], True
    )
    return ok


class CallCounter:
    """Wrap callbacks so every *Python* invocation is counted, and can be armed
    to raise. torch.export traces the wrapped functions (non-strict export
    really calls them), so calls during planner construction are expected; any
    call during search means the search fell back into Python."""

    def __init__(self, cbs: Dict[str, Callable]) -> None:
        self.counts: Dict[str, int] = {k: 0 for k in cbs}
        self.armed: bool = False
        self.wrapped: Dict[str, Callable] = {k: self._wrap(k, fn) for k, fn in cbs.items()}

    def _wrap(self, key: str, fn: Callable) -> Callable:
        def wrapper(*args: Any) -> Any:
            self.counts[key] += 1
            if self.armed:
                raise AssertionError(f"{key} was called from Python during search")
            return fn(*args)

        wrapper.__name__ = getattr(fn, "__name__", key)
        return wrapper

    def reset(self) -> None:
        self.counts = {k: 0 for k in self.counts}

    def total(self) -> int:
        return sum(self.counts.values())
