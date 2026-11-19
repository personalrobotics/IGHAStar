#!/usr/bin/env python3
"""A vertex popped as IGHA* switches resolution level must not be lost.

The search loop pops the lowest-f vertex, then `shift` may decide to switch resolution level
before expanding it. The vertex was already in `Seen`, so breaking out there used to drop it: it
was never expanded nor queued again, and the search could end with an unexpanded vertex below its
best cost, i.e. without having proved its plan optimal (and sometimes with a worse plan). Small
hysteresis thresholds switch levels often, which made this common.

These tests watch the search from outside, through the `generic` environment's Python callbacks:
the goal test runs on every popped vertex, `dynamics_fn` on every expanded one, and `cost_fn` /
`heuristic_fn` give each vertex's g and h.

Run:  python -m pytest tests/unit/planner/test_resolution_switch.py -v
"""
import pathlib
import sys

import pytest
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # generic_problems
sys.path.insert(0, str(HERE.parent.parent.parent))  # repo root

import generic_problems as gp  # noqa: E402
from ighastar.scripts.common_utils import create_planner  # noqa: E402


class SearchWatcher:
    """The generic callbacks, recording pops, expansions and each vertex's g and h."""

    def __init__(self):
        self.base = gp.callbacks()
        self.callbacks = {
            **self.base,
            "dynamics_fn": self.dynamics,
            "cost_fn": self.cost,
            "heuristic_fn": self.heuristic,
            "goal_test_fn": self.goal_test,
        }
        self.g, self.h = {}, {}
        self.expanded = set()
        self.popped = []  # (state key, reached the goal)

    @staticmethod
    def key(state):
        return state.contiguous().numpy().tobytes()

    def dynamics(self, states, controls):
        for state in states:
            self.expanded.add(self.key(state))
        return self.base["dynamics_fn"](states, controls)

    def cost(self, states, controls, next_states):
        costs = self.base["cost_fn"](states, controls, next_states)
        for state, next_state, c in zip(states, next_states, costs):
            g = self.g.get(self.key(state), 0.0) + float(c)  # The start has g = 0.
            k = self.key(next_state)
            self.g[k] = min(g, self.g.get(k, float("inf")))
        return costs

    def heuristic(self, states):
        h = self.base["heuristic_fn"](states)
        for state, value in zip(states, h):
            self.h[self.key(state)] = float(value)
        return h

    def goal_test(self, state):
        reached = self.base["goal_test_fn"](state)
        self.popped.append((self.key(state), bool(reached)))
        return reached

    def f(self, key):
        return self.g.get(key, 0.0) + self.h.get(key, 0.0)


def watched_search(hysteresis):
    watcher = SearchWatcher()
    configs = gp.make_configs("generic", watcher.callbacks, hysteresis=hysteresis)
    planner = create_planner(configs)
    assert gp.run_search(planner, configs)
    assert planner.get_profiler_info()[7] < configs["experiment_info_default"]["max_expansions"]
    best_cost = float(planner.get_best_path()[0, -1])
    return watcher, best_cost


@pytest.mark.parametrize("hysteresis", [0, 1, 2, 3])
def test_no_popped_vertex_below_the_best_cost_is_left_unexpanded(hysteresis):
    """A finished search has expanded every popped vertex with f < its best cost."""
    watcher, best_cost = watched_search(hysteresis)
    left = [
        key
        for key, reached in watcher.popped
        if not reached
        and key not in watcher.expanded
        and watcher.f(key) < best_cost * (1 - 1e-6)
    ]
    assert not left, (
        f"{len(left)} popped vertices with f < best cost {best_cost} were never expanded "
        f"(f = {sorted(watcher.f(k) for k in left)[:5]})"
    )


@pytest.mark.parametrize("hysteresis", [0, 1, 2, 3])
def test_frequent_resolution_switches_keep_the_best_cost(hysteresis):
    """Switching levels often finds the same plan as the default hysteresis (none lost)."""
    _, reference = watched_search(1000)
    _, best_cost = watched_search(hysteresis)
    torch.testing.assert_close(best_cost, reference, rtol=1e-5, atol=0.0)
