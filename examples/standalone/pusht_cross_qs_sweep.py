#!/usr/bin/env python3
"""QS sweep: 10 start states × 50 end-T goals = 500 problems.

Starts: evenly spaced on the 50 dino problems ranked worst→best by first_sol_exp
(baseline oat cell). Each start (agent+T) is paired with every problem's goal T.

  OMP_NUM_THREADS=1 /home/stark/ighastar-venv/bin/python \\
    examples/standalone/pusht_cross_qs_sweep.py --workers 8

Skips (start_idx, goal_idx) already in the output CSV, and seeds those from the
oat baseline when start_idx == goal_idx (original paired problems).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import yaml

from ighastar.scripts.common_utils import BASE_DIR, create_planner

# Reuse oat planner call + metrics.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pusht_hyperparam_sweep import run_one  # noqa: E402

REPO = Path(BASE_DIR).parent
CONTENT = REPO / "Content" / "standalone"
CFG_PATH = REPO / "examples" / "standalone" / "Configs" / "pusht_qs.yml"
OAT_CSV = CONTENT / "pusht_sweep_qs_oat_50k_results.csv"
PROBLEMS = CONTENT / "pusht_dino_problems.npz"
OUT_CSV = CONTENT / "pusht_cross_qs_10x50_results.csv"
OUT_NPZ = CONTENT / "pusht_cross_qs_10x50_problems.npz"
TRAJ_DIR = CONTENT / "pusht_cross_qs_10x50_traj"
MAX_EXP = 50000
N_STARTS = 10
N_WORKERS = 8
BASE = dict(
    res_xy="40.0",
    action_radii="200",
    n_directions="24",
    hysteresis="500",
    n_substeps="10",
    qs_support_radius="60.0",
)
FIELDS = [
    "start_idx",
    "goal_idx",
    "rank_worst_to_best",
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
    "source",  # "planned" | "oat"
]

_WORKER: Dict[str, Any] = {}


def load_oat_baseline() -> List[Dict[str, str]]:
    with open(OAT_CSV) as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r["dynamics"] == "quasistatic"
            and all(r[k] == v for k, v in BASE.items())
        ]
    rows.sort(key=lambda r: int(r["problem_idx"]))
    assert len(rows) == 50, len(rows)
    return rows


def pick_starts(rows: List[Dict[str, str]], n: int = N_STARTS) -> List[int]:
    # Worst → best by first_sol_exp; even sample of n indices.
    order = sorted(
        range(50), key=lambda i: float(rows[i]["first_sol_exp"]), reverse=True
    )
    slots = np.linspace(0, 49, n, dtype=int)
    # Dedup if linspace collides (shouldn't for n=10).
    seen: Set[int] = set()
    out: List[int] = []
    for s in slots:
        for k in range(50):
            i = order[(int(s) + k) % 50]
            if i not in seen:
                seen.add(i)
                out.append(i)
                break
    return out


def make_pair(
    starts: np.ndarray, goals: np.ndarray, si: int, gi: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Start = agent+T from problem si; goal T from problem gi (agent unused)."""
    start = np.asarray(starts[si], dtype=np.float32).copy()
    goal = np.asarray(goals[gi], dtype=np.float32).copy()
    # Goal region is block-only; keep gi's agent for bookkeeping.
    return start, goal


def already_at_goal(start: np.ndarray, goal: np.ndarray) -> bool:
    # Matches pusht.h epsilon on block pose (xy 20px, θ 0.349 rad).
    dx = abs(float(start[2] - goal[2]))
    dy = abs(float(start[3] - goal[3]))
    dth = abs(math.atan2(math.sin(start[4] - goal[4]), math.cos(start[4] - goal[4])))
    return dx < 20.0 and dy < 20.0 and dth < 0.349


def load_done(path: Path) -> Dict[Tuple[int, int], Dict[str, str]]:
    if not path.is_file():
        return {}
    with open(path) as f:
        return {
            (int(r["start_idx"]), int(r["goal_idx"])): r for r in csv.DictReader(f)
        }


def _worker_init(cfg_path: str, max_exp: int) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["experiment_info_default"]["max_expansions"] = int(max_exp)
    _WORKER["cfg"] = cfg
    _WORKER["info"] = cfg["experiment_info_default"]
    _WORKER["planner"] = create_planner(cfg, bidirectional=False)
    _WORKER["n"] = 0


def _worker_plan(job: Tuple[int, int, np.ndarray, np.ndarray]):
    si, gi, start, goal = job
    n = _WORKER["n"]
    # ponytail: recreate every 10 — long QS runs can overflow robin_hood.
    if n > 0 and n % 10 == 0:
        _WORKER["planner"] = create_planner(_WORKER["cfg"], bidirectional=False)
    _WORKER["n"] = n + 1
    info = _WORKER["info"]
    metrics, path = run_one(
        _WORKER["planner"], start, goal, info["max_expansions"], info["hysteresis"]
    )
    exp_per_s = (
        metrics["expansions"] / metrics["wall_s"] if metrics["wall_s"] > 0 else 0.0
    )
    metrics["exp_per_s"] = exp_per_s
    return si, gi, metrics, path


def append_row(writer, f, row: Dict[str, Any]) -> None:
    writer.writerow(row)
    f.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--n-starts", type=int, default=N_STARTS)
    ap.add_argument("--max-expansions", type=int, default=MAX_EXP)
    ap.add_argument("--no-traj", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    oat = load_oat_baseline()
    start_ids = pick_starts(oat, args.n_starts)
    rank = {si: r for r, si in enumerate(start_ids)}
    print(f"starts (worst→best sample): {start_ids}")

    data = np.load(PROBLEMS)
    starts, goals = data["starts"], data["goals"]

    # Materialize full 10×50 set for analysis.
    pairs = [(si, gi) for si in start_ids for gi in range(50)]
    s_arr = np.stack([make_pair(starts, goals, si, gi)[0] for si, gi in pairs])
    g_arr = np.stack([make_pair(starts, goals, si, gi)[1] for si, gi in pairs])
    np.savez_compressed(
        OUT_NPZ,
        starts=s_arr,
        goals=g_arr,
        start_idx=np.array([si for si, _ in pairs], dtype=np.int64),
        goal_idx=np.array([gi for _, gi in pairs], dtype=np.int64),
        start_ids=np.array(start_ids, dtype=np.int64),
    )
    print(f"wrote {OUT_NPZ}  ({len(pairs)} problems)")

    done = load_done(args.out)
    # Seed diagonal from oat so we don't re-plan original pairs.
    oat_seed_rows: List[Dict[str, Any]] = []
    for si in start_ids:
        key = (si, si)
        if key in done:
            continue
        r = oat[si]
        row = {
            "start_idx": si,
            "goal_idx": si,
            "rank_worst_to_best": rank[si],
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
    print(f"already done={len(done) - len(oat_seed_rows)}  oat-seeded diagonal={len(oat_seed_rows)}")

    todo = [(si, gi) for si, gi in pairs if (si, gi) not in done]
    print(f"to plan: {len(todo)} / {len(pairs)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_traj:
        TRAJ_DIR.mkdir(parents=True, exist_ok=True)

    new_file = not args.out.is_file()
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for row in oat_seed_rows:
            append_row(w, f, row)

        # Planner returns empty/fail when start T already in goal epsilon — record trivial.
        still: List[Tuple[int, int]] = []
        for si, gi in todo:
            st, go = make_pair(starts, goals, si, gi)
            if already_at_goal(st, go):
                append_row(w, f, {
                    "start_idx": si,
                    "goal_idx": gi,
                    "rank_worst_to_best": rank[si],
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
            else:
                still.append((si, gi))
        todo = still
        print(f"after trivial filter: {len(todo)} to plan")

        if not todo:
            print("nothing to plan")
            return

        jobs = [
            (si, gi, *make_pair(starts, goals, si, gi)) for si, gi in todo
        ]

        t0 = time.perf_counter()
        done_n = 0
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(str(CFG_PATH), args.max_expansions),
        ) as ex:
            futs = {ex.submit(_worker_plan, job): job[:2] for job in jobs}
            for fut in as_completed(futs):
                si, gi = futs[fut]
                try:
                    si, gi, metrics, path = fut.result()
                except Exception as e:
                    # Leave key absent so a resume re-plans it (don't advance done_n).
                    print(f"FAIL s{si:02d}g{gi:02d}: {e}", file=sys.stderr)
                    continue
                traj_path = ""
                if path is not None and not args.no_traj:
                    tp = TRAJ_DIR / f"s{si:02d}_g{gi:02d}.npy"
                    np.save(tp, path.astype(np.float32))
                    traj_path = str(tp.relative_to(CONTENT))
                row = {
                    "start_idx": si,
                    "goal_idx": gi,
                    "rank_worst_to_best": rank[si],
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
                }
                append_row(w, f, row)
                done_n += 1
                elapsed = time.perf_counter() - t0
                rate = done_n / elapsed if elapsed > 0 else 0
                eta = (len(todo) - done_n) / rate if rate > 0 else float("nan")
                sys.stderr.write(
                    f"\r[{done_n}/{len(todo)}] ok={metrics['success']} "
                    f"s{si:02d}g{gi:02d} first={metrics['first_sol_exp']} "
                    f"{rate:.2f}/s ETA {eta / 60:.1f}m   "
                )
                sys.stderr.flush()
        sys.stderr.write("\n")
    print(f"wrote {args.out}")


def _selfcheck() -> None:
    # ponytail: pick_starts uniqueness + already_at_goal epsilon.
    fake = [{"first_sol_exp": str(i)} for i in range(50)]
    ids = pick_starts(fake, 10)
    assert len(ids) == len(set(ids)) == 10
    s = np.zeros(10, dtype=np.float32)
    g = np.zeros(10, dtype=np.float32)
    assert already_at_goal(s, g)
    g[2] = 25.0
    assert not already_at_goal(s, g)
    print("pusht_cross_qs_sweep selfcheck ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        _selfcheck()
    else:
        main()
