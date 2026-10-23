#!/usr/bin/env python3
"""QS sweep: 10 init-T × 10 goal-T × 50 agent starts = 5000 problems.

Init T and goal T: evenly spaced on the 50 dino problems ranked worst→best
(first_sol_exp, oat baseline). Agent xy: every problem's start agent.

  OMP_NUM_THREADS=1 /home/stark/ighastar-venv/bin/python \\
    examples/standalone/pusht_agent_grid_qs_sweep.py --workers 8

Skips keys already in the results CSV. Seeds the original diagonal
(agent=init_t=goal_t) from oat when present. Trivial already-at-goal skipped
as success without planning.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from ighastar.scripts.common_utils import BASE_DIR, create_planner

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pusht_cross_qs_sweep import (  # noqa: E402
    CFG_PATH,
    already_at_goal,
    load_oat_baseline,
    pick_starts,
)
from pusht_hyperparam_sweep import run_one  # noqa: E402

REPO = Path(BASE_DIR).parent
CONTENT = REPO / "Content" / "standalone"
PROBLEMS = CONTENT / "pusht_dino_problems.npz"
OUT_CSV = CONTENT / "pusht_agent_grid_qs_10x10x50_results.csv"
OUT_NPZ = CONTENT / "pusht_agent_grid_qs_10x10x50_problems.npz"
TRAJ_DIR = CONTENT / "pusht_agent_grid_qs_10x10x50_traj"
MAX_EXP = 50000
N_T = 10
N_WORKERS = 8

FIELDS = [
    "init_t_idx",
    "goal_t_idx",
    "agent_idx",
    "init_t_rank",
    "goal_t_rank",
    "success",
    "wall_s",
    "expansions",
    "avg_succ_us",
    "exp_per_s",
    "cost",
    "path_len",
    "first_sol_exp",
    "preemptive",
    "traj_path",
    "source",
]

_WORKER: Dict[str, Any] = {}


def make_pair(
    starts: np.ndarray,
    goals: np.ndarray,
    init_t: int,
    goal_t: int,
    agent: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Agent from `agent`, init T from starts[init_t], goal T from goals[goal_t]."""
    start = np.asarray(starts[agent], dtype=np.float32).copy()
    start[2:5] = np.asarray(starts[init_t][2:5], dtype=np.float32)
    goal = np.asarray(goals[goal_t], dtype=np.float32).copy()
    return start, goal


def load_done(path: Path) -> Dict[Tuple[int, int, int], Dict[str, str]]:
    if not path.is_file():
        return {}
    with open(path) as f:
        return {
            (int(r["init_t_idx"]), int(r["goal_t_idx"]), int(r["agent_idx"])): r
            for r in csv.DictReader(f)
        }


def append_row(writer, f, row: Dict[str, Any]) -> None:
    writer.writerow(row)
    f.flush()


def _worker_init(cfg_path: str, max_exp: int) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["experiment_info_default"]["max_expansions"] = int(max_exp)
    _WORKER["cfg"] = cfg
    _WORKER["info"] = cfg["experiment_info_default"]
    _WORKER["planner"] = create_planner(cfg, bidirectional=False)
    _WORKER["n"] = 0


def _worker_plan(job: Tuple[int, int, int, np.ndarray, np.ndarray]):
    it, gt, ag, start, goal = job
    n = _WORKER["n"]
    # ponytail: recreate every 10 — long QS runs can overflow robin_hood.
    if n > 0 and n % 10 == 0:
        _WORKER["planner"] = create_planner(_WORKER["cfg"], bidirectional=False)
    _WORKER["n"] = n + 1
    info = _WORKER["info"]
    metrics, path = run_one(
        _WORKER["planner"], start, goal, info["max_expansions"], info["hysteresis"]
    )
    metrics["exp_per_s"] = (
        metrics["expansions"] / metrics["wall_s"] if metrics["wall_s"] > 0 else 0.0
    )
    return it, gt, ag, metrics, path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--n-t", type=int, default=N_T, help="even sample size for init/goal T")
    ap.add_argument("--max-expansions", type=int, default=MAX_EXP)
    ap.add_argument("--save-traj", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--build-only", action="store_true", help="write npz only, no planning")
    args = ap.parse_args()

    oat = load_oat_baseline()
    t_ids = pick_starts(oat, args.n_t)
    init_rank = {si: r for r, si in enumerate(t_ids)}
    goal_rank = dict(init_rank)  # same evenly spaced pool
    print(f"init/goal T indices (worst→best sample): {t_ids}")
    print(f"agents: all 50  →  {len(t_ids)}×{len(t_ids)}×50 = {len(t_ids)**2 * 50} problems")

    data = np.load(PROBLEMS)
    starts, goals = data["starts"], data["goals"]

    pairs = [
        (it, gt, ag)
        for it in t_ids
        for gt in t_ids
        for ag in range(50)
    ]
    s_arr = np.stack([make_pair(starts, goals, it, gt, ag)[0] for it, gt, ag in pairs])
    g_arr = np.stack([make_pair(starts, goals, it, gt, ag)[1] for it, gt, ag in pairs])
    np.savez_compressed(
        OUT_NPZ,
        starts=s_arr,
        goals=g_arr,
        init_t_idx=np.array([p[0] for p in pairs], dtype=np.int64),
        goal_t_idx=np.array([p[1] for p in pairs], dtype=np.int64),
        agent_idx=np.array([p[2] for p in pairs], dtype=np.int64),
        t_ids=np.array(t_ids, dtype=np.int64),
    )
    print(f"wrote {OUT_NPZ}  ({len(pairs)} problems)")
    if args.build_only:
        return

    done = load_done(args.out)
    oat_seed_rows: List[Dict[str, Any]] = []
    # Original dino problem when agent=init_t=goal_t.
    for ti in t_ids:
        key = (ti, ti, ti)
        if key in done:
            continue
        r = oat[ti]
        row = {
            "init_t_idx": ti,
            "goal_t_idx": ti,
            "agent_idx": ti,
            "init_t_rank": init_rank[ti],
            "goal_t_rank": goal_rank[ti],
            "success": r["success"],
            "wall_s": r["wall_s"],
            "expansions": r["expansions"],
            "avg_succ_us": r["avg_succ_us"],
            "exp_per_s": r.get("exp_per_s", ""),
            "cost": r["cost"],
            "path_len": r["path_len"],
            "first_sol_exp": r["first_sol_exp"],
            "preemptive": r["preemptive"],
            "traj_path": r.get("traj_path", ""),
            "source": "oat",
        }
        done[key] = {k: str(v) for k, v in row.items()}
        oat_seed_rows.append(row)
    print(
        f"already done={len(done) - len(oat_seed_rows)}  "
        f"oat-seeded diagonal={len(oat_seed_rows)}"
    )

    todo = [p for p in pairs if p not in done]
    print(f"to plan: {len(todo)} / {len(pairs)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.save_traj:
        TRAJ_DIR.mkdir(parents=True, exist_ok=True)

    new_file = not args.out.is_file()
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for row in oat_seed_rows:
            append_row(w, f, row)

        still: List[Tuple[int, int, int]] = []
        n_triv = 0
        for it, gt, ag in todo:
            st, go = make_pair(starts, goals, it, gt, ag)
            if already_at_goal(st, go):
                append_row(w, f, {
                    "init_t_idx": it,
                    "goal_t_idx": gt,
                    "agent_idx": ag,
                    "init_t_rank": init_rank[it],
                    "goal_t_rank": goal_rank[gt],
                    "success": 1,
                    "wall_s": 0,
                    "expansions": 0,
                    "avg_succ_us": 0,
                    "exp_per_s": 0,
                    "cost": 0,
                    "path_len": 1,
                    "first_sol_exp": 0,
                    "preemptive": 0,
                    "traj_path": "",
                    "source": "trivial_at_goal",
                })
                n_triv += 1
            else:
                still.append((it, gt, ag))
        print(f"trivial_at_goal={n_triv}  remaining={len(still)}")
        todo = still
        if not todo:
            print("nothing to plan")
            return

        jobs = [
            (it, gt, ag, *make_pair(starts, goals, it, gt, ag))
            for it, gt, ag in todo
        ]
        t0 = time.perf_counter()
        done_n = 0
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(str(CFG_PATH), args.max_expansions),
        ) as ex:
            futs = {ex.submit(_worker_plan, job): job[:3] for job in jobs}
            for fut in as_completed(futs):
                it, gt, ag = futs[fut]
                try:
                    it, gt, ag, metrics, path = fut.result()
                except Exception as e:
                    # Leave key absent so a resume re-plans it (don't advance done_n).
                    print(f"FAIL it{it:02d}gt{gt:02d}ag{ag:02d}: {e}", file=sys.stderr)
                    continue
                traj_path = ""
                if path is not None and args.save_traj:
                    tp = TRAJ_DIR / f"it{it:02d}_gt{gt:02d}_ag{ag:02d}.npy"
                    np.save(tp, path.astype(np.float32))
                    traj_path = str(tp.relative_to(CONTENT))
                append_row(w, f, {
                    "init_t_idx": it,
                    "goal_t_idx": gt,
                    "agent_idx": ag,
                    "init_t_rank": init_rank[it],
                    "goal_t_rank": goal_rank[gt],
                    "success": metrics["success"],
                    "wall_s": metrics["wall_s"],
                    "expansions": metrics["expansions"],
                    "avg_succ_us": metrics["avg_succ_us"],
                    "exp_per_s": metrics["exp_per_s"],
                    "cost": metrics["cost"],
                    "path_len": metrics["path_len"],
                    "first_sol_exp": metrics["first_sol_exp"],
                    "preemptive": metrics["preemptive"],
                    "traj_path": traj_path,
                    "source": "planned",
                })
                done_n += 1
                elapsed = time.perf_counter() - t0
                rate = done_n / elapsed if elapsed > 0 else 0
                eta = (len(todo) - done_n) / rate if rate > 0 else float("nan")
                sys.stderr.write(
                    f"\r[{done_n}/{len(todo)}] ok={metrics['success']} "
                    f"it{it:02d}gt{gt:02d}ag{ag:02d} first={metrics['first_sol_exp']} "
                    f"{rate:.2f}/s ETA {eta / 60:.1f}m   "
                )
                sys.stderr.flush()
        sys.stderr.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
