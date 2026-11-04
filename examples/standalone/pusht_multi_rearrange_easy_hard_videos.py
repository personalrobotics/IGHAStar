#!/usr/bin/env python3
"""Rank 2T1O_rearrange by expansions to the first solution, then render 10+10 mp4s.

First-solution index matches pusht_hyperparam_sweep: the first cost_exp_list
entry below 1e4. Search still runs to the yaml expansion cap.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from pusht_multi_generate_2T1O_rearrange import (
    CONTENT,
    load_config,
    render_rearrange_video,
)
from ighastar.scripts.common_utils import create_planner

NPZ = os.path.join(CONTENT, "pusht_multi_2T1O_rearrange.npz")
OUT = os.path.join(CONTENT, "pusht_multi_2T1O_rearrange_videos")


def first_solution_expansions(cost_exp) -> int:
    cost = np.asarray(cost_exp, dtype=np.float64)
    hit = np.where(cost < 1e4)[0]
    return int(hit[0]) if hit.size else -1


def main() -> None:
    z = np.load(NPZ, allow_pickle=True)
    starts, goals, obstacles, paths = z["starts"], z["goals"], z["obstacles"], z["paths"]
    n = len(starts)
    if "first_sol_expansions" in z.files:
        first = np.asarray(z["first_sol_expansions"], dtype=np.int64)
        print(f"using saved first_sol_expansions ({n} problems)")
    else:
        cfg = load_config()
        info = cfg["experiment_info_default"]
        limit = int(info["max_expansions"])
        hyst = int(info["hysteresis"])
        planner = create_planner(cfg, bidirectional=False)
        first = np.empty(n, dtype=np.int64)
        for i in range(n):
            ok = bool(
                planner.search(
                    torch.from_numpy(starts[i]),
                    torch.from_numpy(goals[i]),
                    torch.from_numpy(obstacles[i]),
                    limit,
                    hyst,
                    False,
                )
            )
            prof = planner.get_profiler_info()
            first[i] = first_solution_expansions(prof[9]) if ok else -1
            print(f"{i + 1}/{n} first_sol={first[i]} solved={ok}", flush=True)

    order = np.argsort(first, kind="mergesort")
    easy = order[:10]
    hard = order[-10:][::-1]
    print("easy", list(zip(easy.tolist(), first[easy].tolist())))
    print("hard", list(zip(hard.tolist(), first[hard].tolist())))

    for rank, idx in enumerate(easy, start=1):
        render_rearrange_video(
            paths[idx],
            goals[idx],
            obstacles[idx],
            os.path.join(OUT, "easy", f"e{rank:02d}_p{idx:03d}_exp{first[idx]}.mp4"),
        )
    for rank, idx in enumerate(hard, start=1):
        render_rearrange_video(
            paths[idx],
            goals[idx],
            obstacles[idx],
            os.path.join(OUT, "hard", f"h{rank:02d}_p{idx:03d}_exp{first[idx]}.mp4"),
        )

    np.savez(
        NPZ,
        starts=starts,
        goals=goals,
        obstacles=obstacles,
        seeds=z["seeds"],
        paths=paths,
        problem_class=z["problem_class"],
        tee_id=z["tee_id"],
        tee_rgb=z["tee_rgb"],
        max_expansions=z["max_expansions"],
        first_sol_expansions=first,
    )
    print(f"wrote ranking into {NPZ}")


if __name__ == "__main__":
    main()
