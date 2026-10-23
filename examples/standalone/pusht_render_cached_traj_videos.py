#!/usr/bin/env python3
"""Render mp4s from already-cached traj *.npy (no re-planning).

  /home/stark/ighastar-venv/bin/python examples/standalone/pusht_render_cached_traj_videos.py

Sets:
  hard_easy chipmunk / quasistatic  → Content/standalone/pusht_hard_easy_videos_{dyn}/
  cross QS 10×50                    → Content/standalone/pusht_cross_qs_10x50_videos/
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ighastar.scripts.common_utils import BASE_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pusht_render_problem_videos import render_states  # noqa: E402

CONTENT = Path(BASE_DIR).parent / "Content" / "standalone"
DINO = CONTENT / "pusht_dino_problems.npz"
CROSS_NPZ = CONTENT / "pusht_cross_qs_10x50_problems.npz"


def jobs_hard_easy(dyn: str) -> List[Tuple[Path, np.ndarray, Path]]:
    traj_dir = CONTENT / f"pusht_hard_easy_traj_{dyn}"
    out_dir = CONTENT / f"pusht_hard_easy_videos_{dyn}"
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = np.load(DINO)
    goals = problems["goals"]
    jobs = []
    for p in sorted(traj_dir.glob("p*.npy")):
        idx = int(p.stem[1:])
        out = out_dir / f"{p.stem}.mp4"
        if out.is_file():
            continue
        jobs.append((p, goals[idx][2:5].astype(np.float64), out))
    return jobs


def jobs_cross() -> List[Tuple[Path, np.ndarray, Path]]:
    traj_dir = CONTENT / "pusht_cross_qs_10x50_traj"
    out_dir = CONTENT / "pusht_cross_qs_10x50_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(CROSS_NPZ)
    # Map (si, gi) → goal T from stacked goals (same order as start_idx/goal_idx).
    key_to_goal = {}
    for k in range(len(data["start_idx"])):
        si, gi = int(data["start_idx"][k]), int(data["goal_idx"][k])
        key_to_goal[(si, gi)] = data["goals"][k][2:5].astype(np.float64)
    jobs = []
    for p in sorted(traj_dir.glob("s*_g*.npy")):
        # s08_g20
        parts = p.stem.split("_")
        si, gi = int(parts[0][1:]), int(parts[1][1:])
        out = out_dir / f"{p.stem}.mp4"
        if out.is_file():
            continue
        goal_b = key_to_goal.get((si, gi))
        if goal_b is None:
            # Fallback: last block pose on traj.
            arr = np.load(p)
            goal_b = arr[-1, 2:5].astype(np.float64)
        jobs.append((p, goal_b, out))
    return jobs


def _render_one(job: Tuple[str, list, str, int]) -> Tuple[str, float, int]:
    traj_s, goal_b, out_s, fps = job
    path = np.load(traj_s)
    t0 = time.perf_counter()
    render_states(path, np.asarray(goal_b, dtype=np.float64), out_s, fps)
    return out_s, time.perf_counter() - t0, int(path.shape[0])


def run_jobs(jobs: List[Tuple[Path, np.ndarray, Path]], fps: int, workers: int) -> None:
    if not jobs:
        print("  nothing to render (all mp4 present)")
        return
    print(f"  rendering {len(jobs)} videos  workers={workers}")
    payload = [
        (str(traj), goal_b.tolist(), str(out), fps) for traj, goal_b, out in jobs
    ]
    t0 = time.perf_counter()
    done = 0
    if workers <= 1:
        for job in payload:
            out_s, wall, n = _render_one(job)
            done += 1
            sys.stderr.write(
                f"\r  [{done}/{len(payload)}] {Path(out_s).name}  "
                f"T={n}  {wall:.2f}s   "
            )
            sys.stderr.flush()
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_render_one, j) for j in payload]
            for fut in as_completed(futs):
                out_s, wall, n = fut.result()
                done += 1
                sys.stderr.write(
                    f"\r  [{done}/{len(payload)}] {Path(out_s).name}  "
                    f"T={n}  {wall:.2f}s   "
                )
                sys.stderr.flush()
    sys.stderr.write("\n")
    print(f"  done in {time.perf_counter() - t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--sets",
        default="hard_easy,cross",
        help="comma list: hard_easy,cross",
    )
    args = ap.parse_args()
    sets = {s.strip() for s in args.sets.split(",") if s.strip()}

    if "hard_easy" in sets:
        for dyn in ("chipmunk", "quasistatic"):
            print(f"=== hard_easy {dyn} ===")
            run_jobs(jobs_hard_easy(dyn), args.fps, args.workers)
    if "cross" in sets:
        print("=== cross qs 10x50 ===")
        run_jobs(jobs_cross(), args.fps, args.workers)


if __name__ == "__main__":
    main()
