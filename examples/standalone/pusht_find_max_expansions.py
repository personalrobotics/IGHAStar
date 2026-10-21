#!/usr/bin/env python3
"""Find the smallest max_expansions (in 5k steps) that solves all dino PushT problems.

Uses best config: res_xy=40, action_radii=[200], max_agent_vel=500.

  OMP_NUM_THREADS=8 /home/stark/ighastar-venv/bin/python \\
    examples/standalone/pusht_find_max_expansions.py
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
import yaml

from ighastar.scripts.common_utils import BASE_DIR, create_planner

# Reuse problem helpers / CONTENT from the sweep module.
import importlib.util

_sweep_path = os.path.join(
    BASE_DIR.parent, "examples", "standalone", "pusht_hyperparam_sweep.py"
)
_spec = importlib.util.spec_from_file_location("pusht_sweep", _sweep_path)
_sweep = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_sweep)

CONTENT = _sweep.CONTENT


def make_cfg(base: Dict[str, Any], max_expansions: int) -> Dict[str, Any]:
    cfg = _sweep.apply_sweep_config(base, dict(_sweep.BASELINE))
    cfg["experiment_info_default"]["max_expansions"] = int(max_expansions)
    return cfg


def try_problem(planner, start, goal, max_expansions: int, hysteresis: int):
    t0 = time.perf_counter()
    ok = bool(
        planner.search(
            torch.as_tensor(start),
            torch.as_tensor(goal),
            torch.zeros(1, dtype=torch.float32),
            max_expansions,
            hysteresis,
            True,
        )
    )
    wall = time.perf_counter() - t0
    prof = planner.get_profiler_info()
    expansions = int(prof[7])
    cost_exp = np.asarray(prof[9], dtype=np.float64)
    sol = np.where(cost_exp < 1e4)[0]
    first_sol = int(sol[0]) if sol.size else -1
    cost = float("nan")
    if ok:
        path = planner.get_best_path().numpy()
        cost = float(path[0, -1])
    return {
        "success": int(ok),
        "wall_s": wall,
        "expansions": expansions,
        "first_sol_exp": first_sol,
        "cost": cost,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-yaml",
        default=os.path.join(
            BASE_DIR.parent, "examples", "standalone", "Configs", "pusht.yml"
        ),
    )
    parser.add_argument(
        "--problems",
        default=os.path.join(CONTENT, "pusht_dino_problems.npz"),
    )
    parser.add_argument("--start-expansions", type=int, default=15000)
    parser.add_argument("--step", type=int, default=5000)
    parser.add_argument("--cap", type=int, default=200000)
    parser.add_argument("--omp-threads", type=int, default=8)
    parser.add_argument(
        "--out",
        default=os.path.join(CONTENT, "pusht_find_max_expansions.csv"),
    )
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)

    with open(args.base_yaml) as f:
        base = yaml.safe_load(f)
    hysteresis = int(base["experiment_info_default"]["hysteresis"])

    problems = _sweep.load_or_make_problems(args.problems, regen=False, n_problems=50)
    n = len(problems["seeds"])
    unsolved: Set[int] = set(range(n))
    results: Dict[int, Dict[str, Any]] = {}

    B = int(args.start_expansions)
    print(
        f"find max_expansions for 100% on {n} problems  "
        f"start={B} step={args.step} cap={args.cap}  "
        f"OMP={os.environ['OMP_NUM_THREADS']}"
    )
    print("config: res=40 radii=[200] vmax=500")

    # Warm-load extension.
    create_planner(make_cfg(base, B), bidirectional=False)

    while unsolved and B <= args.cap:
        cfg = make_cfg(base, B)
        planner = create_planner(cfg, bidirectional=False)
        pending = sorted(unsolved)
        print(f"\n=== budget B={B}  remaining={len(pending)}/{n} ===")
        t_round = time.perf_counter()
        newly: List[int] = []
        for k, pi in enumerate(pending):
            m = try_problem(
                planner,
                problems["starts"][pi],
                problems["goals"][pi],
                B,
                hysteresis,
            )
            if m["success"]:
                results[pi] = {
                    "problem_idx": pi,
                    "dino_seed": int(problems["seeds"][pi]),
                    "budget_solved": B,
                    **m,
                }
                newly.append(pi)
                unsolved.discard(pi)
            status = "OK" if m["success"] else "FAIL"
            print(
                f"  [{k+1}/{len(pending)}] p{pi:02d} {status}  "
                f"first_sol={m['first_sol_exp']}  wall={m['wall_s']:.2f}s  "
                f"left={len(unsolved)}"
            )
        print(
            f"round B={B}: solved +{len(newly)}  "
            f"total {n - len(unsolved)}/{n}  "
            f"wall={time.perf_counter() - t_round:.1f}s"
        )
        if not unsolved:
            break
        B += args.step

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = [
        "problem_idx",
        "dino_seed",
        "budget_solved",
        "success",
        "first_sol_exp",
        "expansions",
        "cost",
        "wall_s",
    ]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for pi in range(n):
            if pi in results:
                w.writerow(results[pi])
            else:
                w.writerow(
                    {
                        "problem_idx": pi,
                        "dino_seed": int(problems["seeds"][pi]),
                        "budget_solved": "",
                        "success": 0,
                        "first_sol_exp": -1,
                        "expansions": "",
                        "cost": "",
                        "wall_s": "",
                    }
                )

    if unsolved:
        print(
            f"\nFAILED 100%: {len(unsolved)} unsolved at cap={args.cap}: "
            f"{sorted(unsolved)}"
        )
        print(f"wrote partial {args.out}")
        raise SystemExit(1)

    firsts = np.array([results[i]["first_sol_exp"] for i in range(n)], dtype=np.float64)
    budgets = np.array([results[i]["budget_solved"] for i in range(n)], dtype=np.int64)
    # Smallest k*step that is >= every first_sol_exp (and matches bump protocol).
    need = int(np.ceil(firsts.max() / args.step) * args.step)
    print(f"\n=== 100% success ===")
    print(f"all {n} solved")
    print(f"max first_sol_exp = {int(firsts.max())}  (problem {int(firsts.argmax())})")
    print(f"median first_sol_exp = {int(np.median(firsts))}")
    print(f"bump-protocol budget that cleared last failures: {int(budgets.max())}")
    print(f"minimal 5k-aligned max_expansions for 100%: {need}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
