#!/usr/bin/env python3
"""Record slot-insertion solutions, then re-solve the same queries with the library.

Pass 1 searches the easiest saved problems and records each solution.
Pass 2 loads that library and searches the same starts, goals, and walls
with shooting MPC. An exact saved query should solve on the first expansion.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import yaml

from ighastar.scripts.common_utils import create_planner

HERE = os.path.dirname(os.path.abspath(__file__))
CONTENT = os.path.join(HERE, "..", "..", "Content", "standalone")
PROBLEMS = os.path.join(CONTENT, "pusht_multi_slot_1T_rearrange.npz")
CONFIG = os.path.join(HERE, "Configs", "pusht_multi_slot_1T_rearrange.yml")
LIBRARY = os.path.join(CONTENT, "pusht_slot_1T_trajlib.npy")
N = 8
REPLAY_LIMIT = 30


def first_sol(planner) -> int:
    cost = np.asarray(planner.get_profiler_info()[9], dtype=np.float64)
    hit = np.where(cost < 1e4)[0]
    return int(hit[0]) if hit.size else -1


def search(planner, start, goal, world, limit, hyst):
    t0 = time.perf_counter()
    ok = bool(
        planner.search(
            torch.from_numpy(start),
            torch.from_numpy(goal),
            torch.from_numpy(world),
            int(limit),
            int(hyst),
            False,
        )
    )
    path = planner.get_best_path()
    if ok and path.numel() == 0:
        ok = False
    cost = float("nan")
    if ok:
        cost = float(path.detach().cpu().numpy()[0, -1])
    return ok, first_sol(planner), cost, time.perf_counter() - t0


def main() -> None:
    data = np.load(PROBLEMS, allow_pickle=True)
    order = np.argsort(data["first_sol_expansions"])[:N]
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    hyst = int(cfg["experiment_info_default"]["hysteresis"])
    planner = create_planner(cfg, bidirectional=False)

    print("record", flush=True)
    recorded = []
    for idx in order:
        prev = int(data["first_sol_expansions"][idx])
        limit = max(8000, prev * 5)
        ok, first, cost, wall = search(
            planner,
            data["starts"][idx],
            data["goals"][idx],
            data["obstacles"][idx],
            limit,
            hyst,
        )
        if ok:
            planner.record_solution_trajectory()
            recorded.append(int(idx))
        print(
            f"  p{int(idx)} prev={prev} limit={limit} ok={ok} "
            f"first={first} cost={cost:.3f} {wall:.1f}s",
            flush=True,
        )
    if not recorded:
        raise SystemExit("no solutions recorded")
    lib = planner.export_trajectory_library()
    if not planner.trajectory_library_links_ok():
        raise SystemExit("library links broken")
    np.save(LIBRARY, np.array(lib, dtype=object))
    print(f"saved {len(lib)} trajectories to {LIBRARY}", flush=True)

    replay = create_planner(cfg, bidirectional=False)
    loaded = np.load(LIBRARY, allow_pickle=True).tolist()
    replay.import_trajectory_library(loaded)
    print("replay", flush=True)
    for idx in recorded:
        prev = int(data["first_sol_expansions"][idx])
        ok, first, cost, wall = search(
            replay,
            data["starts"][idx],
            data["goals"][idx],
            data["obstacles"][idx],
            REPLAY_LIMIT,
            hyst,
        )
        print(
            f"  p{int(idx)} prev={prev} ok={ok} first={first} "
            f"cost={cost:.3f} {wall:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
