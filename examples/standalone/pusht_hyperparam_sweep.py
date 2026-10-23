#!/usr/bin/env python3
"""Sweep PushT IGHAStar hyperparams over a dino_wm-style problem set.

Problem generation matches dino_wm PushTWrapper.sample_random_init_goal_states
(plan_pusht.yaml: seed=99, n_evals=50 → seeds 99*n+1).

  /home/stark/ighastar-venv/bin/python examples/standalone/pusht_hyperparam_sweep.py
  # or 3-way parallel (8 OMP each):
  bash examples/standalone/pusht_hyperparam_sweep_parallel.sh

Outputs under Content/standalone/ by default:
  pusht_dino_problems.npz     — fixed problem set
  pusht_sweep_results.csv     — one row per (config × problem)
  pusht_sweep_traj/           — success paths only:
    res{R}_radii{tag}_vmax{V}/p{idx:03d}.npy   # [T, 11] start→goal, last col = g
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from ighastar.scripts.common_utils import BASE_DIR, create_planner

CONTENT = os.path.join(BASE_DIR.parent, "Content", "standalone")

# dino_wm/conf/plan_pusht.yaml
DINO_SEED = 99
DINO_N_EVALS = 50

# QS OAT around best Chipmunk-ish cell; dynamics/expansions set below.
# Full factorial is huge — one axis at a time from baseline.
BASELINE = {
    "res_xy": 40.0,
    "radii": (200.0,),
    "vmax": 500.0,
    "n_directions": 16,
    "n_substeps": 10,
    "hysteresis": 500,
    "qs_support_radius": 60.0,
}
DYNAMICS = "quasistatic"
MAX_EXPANSIONS_SWEEP = 50000


def build_oat_grid() -> List[Dict[str, Any]]:
    """One-at-a-time configs; baseline included once."""
    seen = set()
    out: List[Dict[str, Any]] = []

    def add(**overrides: Any) -> None:
        c = {**BASELINE, **overrides}
        c["radii"] = tuple(float(r) for r in c["radii"])
        key = (
            c["res_xy"],
            c["radii"],
            c["vmax"],
            c["n_directions"],
            c["n_substeps"],
            c["hysteresis"],
            c["qs_support_radius"],
        )
        if key in seen:
            return
        seen.add(key)
        out.append(c)

    add()
    for res in (20.0, 40.0, 80.0):
        add(res_xy=res)
    for r in (150.0, 200.0, 250.0, 300.0):
        add(radii=(r,))
    for nd in (4, 8, 12, 16, 24):
        add(n_directions=nd)
    for ns in (5, 10, 20):
        add(n_substeps=ns)
    for h in (100, 500, 1000, 2000):
        add(hysteresis=h)
    for R in (40.0, 60.0, 80.0, 100.0):
        add(qs_support_radius=R)
    return out


GRID_ALL = build_oat_grid()


def dino_eval_seeds(seed: int = DINO_SEED, n_evals: int = DINO_N_EVALS) -> List[int]:
    return [seed * n + 1 for n in range(n_evals)]


def sample_dino_init_goal(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror dino_wm PushTWrapper.sample_random_init_goal_states (with_velocity=True).

    Returns two length-7 arrays: [ax, ay, bx, by, angle, avx, avy].
    """
    rs = np.random.RandomState(seed)

    def generate_state() -> np.ndarray:
        return np.array(
            [
                rs.randint(50, 450),
                rs.randint(50, 450),
                rs.randint(100, 400),
                rs.randint(100, 400),
                rs.randn() * 2 * np.pi - np.pi,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

    return generate_state(), generate_state()


def to_planner_state(s7: np.ndarray) -> np.ndarray:
    """7D dino state → 10D planner state (block vel / ω = 0)."""
    s7 = np.asarray(s7, dtype=np.float64).reshape(-1)
    out = np.zeros(10, dtype=np.float32)
    out[:7] = s7[:7]
    return out


def generate_problem_set(
    path: str, seed: int = DINO_SEED, n_evals: int = DINO_N_EVALS
) -> Dict[str, np.ndarray]:
    starts, goals, seeds = [], [], []
    for s in dino_eval_seeds(seed, n_evals):
        init7, goal7 = sample_dino_init_goal(s)
        starts.append(to_planner_state(init7))
        goals.append(to_planner_state(goal7))
        seeds.append(s)
    data = {
        "starts": np.stack(starts),
        "goals": np.stack(goals),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "meta_seed": np.array(seed, dtype=np.int64),
        "meta_n_evals": np.array(n_evals, dtype=np.int64),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"wrote {path}  ({n_evals} problems, dino seed={seed})")
    return data


def load_or_make_problems(path: str, regen: bool, n_problems: int) -> Dict[str, Any]:
    if regen or not os.path.isfile(path):
        data = generate_problem_set(path, n_evals=max(n_problems, DINO_N_EVALS))
    else:
        data = dict(np.load(path))
        print(f"loaded {path}  ({len(data['seeds'])} problems)")
    n = min(n_problems, len(data["seeds"]))
    return {
        "starts": data["starts"][:n],
        "goals": data["goals"][:n],
        "seeds": data["seeds"][:n],
    }


def radii_tag(radii: List[float]) -> str:
    return "+".join(str(int(r)) for r in radii)


def _fmt_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--"
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def print_progress(
    done: int, total: int, t0: float, *, prefix: str = "", width: int = 28
) -> None:
    """Single-line progress bar on stderr."""
    frac = done / total if total else 1.0
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.perf_counter() - t0
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else float("nan")
    sys.stderr.write(
        f"\r{prefix}[{bar}] {done}/{total} ({100 * frac:5.1f}%)  "
        f"{rate:.2f}/s  ETA {_fmt_eta(eta)}   "
    )
    sys.stderr.flush()


def apply_sweep_config(base: Dict[str, Any], cell: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    info = cfg["experiment_info_default"]
    res = list(info["resolution"])
    for i in range(4):
        res[i] = float(cell["res_xy"])
    info["resolution"] = res
    info["hysteresis"] = int(cell["hysteresis"])
    info["max_expansions"] = int(cell.get("max_expansions", MAX_EXPANSIONS_SWEEP))
    ni = info["node_info"]
    ni["action_radii"] = [float(r) for r in cell["radii"]]
    ni["max_agent_vel"] = float(cell["vmax"])
    ni["n_directions"] = int(cell["n_directions"])
    ni["n_substeps"] = int(cell["n_substeps"])
    ni["dynamics"] = cell.get("dynamics", DYNAMICS)
    ni["qs_support_radius"] = float(
        cell.get("qs_support_radius", BASELINE["qs_support_radius"])
    )
    return cfg


def cell_tag(cell: Dict[str, Any]) -> str:
    return (
        f"res={cell['res_xy']:g} radii={radii_tag(cell['radii'])} "
        f"vmax={cell['vmax']:g} ndir={cell['n_directions']} "
        f"nsub={cell['n_substeps']} hyst={cell['hysteresis']} "
        f"R={cell.get('qs_support_radius', BASELINE['qs_support_radius']):g}"
    )


def run_one(
    planner, start: np.ndarray, goal: np.ndarray, max_expansions: int, hysteresis: int
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    t0 = time.perf_counter()
    success = bool(
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
    (
        avg_succ,
        _gct,
        _ovh,
        _gup,
        _sw,
        _ml,
        _qv,
        expansions,
        _elist,
        cost_exp_list,
    ) = planner.get_profiler_info()
    cost = float("nan")
    path_len = 0
    path_sg: Optional[np.ndarray] = None
    if success:
        # get_best_path is goal-first; flip to start→goal for playback.
        path_sg = planner.get_best_path().numpy()[::-1].copy()
        if path_sg.size == 0:
            # Rare: search reports success but best path is empty.
            success = False
            path_sg = None
        else:
            cost = float(path_sg[-1, -1])
            path_len = int(path_sg.shape[0])
    cost_exp = np.asarray(cost_exp_list, dtype=np.float64)
    sol = np.where(cost_exp < 1e4)[0]
    first_sol_exp = int(sol[0]) if sol.size else -1
    metrics = {
        "success": int(success),
        "wall_s": wall,
        "expansions": int(expansions),
        "avg_succ_us": float(avg_succ),
        "cost": cost,
        "path_len": path_len,
        "first_sol_exp": first_sol_exp,
        "preemptive": int(planner.get_preemptive_expansions()),
    }
    return metrics, path_sg


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
        help="problem-set npz (default: Content/standalone/pusht_dino_problems.npz)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(CONTENT, "pusht_sweep_qs_oat_50k_results.csv"),
        help="metrics CSV",
    )
    parser.add_argument(
        "--traj-dir",
        default=os.path.join(CONTENT, "pusht_sweep_qs_oat_50k_traj"),
        help="dir for success trajectories *.npy",
    )
    parser.add_argument(
        "--no-traj",
        action="store_true",
        help="skip saving state trajectories",
    )
    parser.add_argument("--n-problems", type=int, default=DINO_N_EVALS)
    parser.add_argument("--regen-problems", action="store_true")
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=None,
        help="sets OMP_NUM_THREADS (default: keep env, else 4)",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="this worker's shard index in [0, num-shards)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="split the config grid across this many workers",
    )
    args = parser.parse_args()

    if args.omp_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)
    else:
        os.environ.setdefault("OMP_NUM_THREADS", "4")

    if args.num_shards < 1 or not (0 <= args.shard_id < args.num_shards):
        raise SystemExit(
            f"bad shard: shard-id={args.shard_id} num-shards={args.num_shards}"
        )
    if args.num_shards > 1:
        # Disjoint outputs unless user overrode --out / --traj-dir explicitly
        # after defaults; always suffix when sharding so parallel workers don't clash.
        root, ext = os.path.splitext(args.out)
        args.out = f"{root}_shard{args.shard_id}{ext or '.csv'}"
        args.traj_dir = f"{args.traj_dir}_shard{args.shard_id}"

    with open(args.base_yaml) as f:
        base_cfg = yaml.safe_load(f)
    max_expansions = int(MAX_EXPANSIONS_SWEEP)

    problems = load_or_make_problems(args.problems, args.regen_problems, args.n_problems)
    n_prob = len(problems["seeds"])

    grid_all = GRID_ALL
    grid = [c for i, c in enumerate(grid_all) if i % args.num_shards == args.shard_id]
    print(
        f"sweep shard {args.shard_id}/{args.num_shards}: "
        f"{len(grid)}/{len(grid_all)} configs × {n_prob} problems = "
        f"{len(grid) * n_prob} searches"
    )
    print(f"  dynamics={DYNAMICS}  OAT from {cell_tag(BASELINE)}")
    print(
        f"  max_expansions={max_expansions}  "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}  "
        f"out={args.out}"
    )

    # Warm-load extension once (first create_planner compiles).
    create_planner(apply_sweep_config(base_cfg, GRID_ALL[0]), False)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if not args.no_traj:
        os.makedirs(args.traj_dir, exist_ok=True)

    fieldnames = [
        "res_xy",
        "action_radii",
        "max_agent_vel",
        "n_directions",
        "n_substeps",
        "hysteresis",
        "qs_support_radius",
        "dynamics",
        "problem_idx",
        "dino_seed",
        "traj_path",
        "success",
        "wall_s",
        "expansions",
        "avg_succ_us",
        "exp_per_s",
        "cost",
        "path_len",
        "first_sol_exp",
        "preemptive",
    ]
    rows: List[Dict[str, Any]] = []
    n_traj_saved = 0
    n_total = len(grid) * n_prob
    n_done = 0
    t_all = time.perf_counter()

    # Append CSV as we go so Ctrl+C keeps partial results.
    csv_f = open(args.out, "w", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=fieldnames)
    csv_w.writeheader()
    csv_f.flush()

    try:
        for ci, cell in enumerate(grid):
            cfg = apply_sweep_config(base_cfg, cell)
            planner = create_planner(cfg, bidirectional=False)
            rtag = radii_tag(cell["radii"])
            tag = cell_tag(cell)
            traj_subdir = os.path.join(
                args.traj_dir,
                f"res{cell['res_xy']:g}_radii{rtag}_vmax{cell['vmax']:g}"
                f"_nd{cell['n_directions']}_ns{cell['n_substeps']}_h{cell['hysteresis']}",
            )
            if not args.no_traj:
                os.makedirs(traj_subdir, exist_ok=True)
            n_ok = 0
            t_cfg = time.perf_counter()
            for pi in range(n_prob):
                metrics, path_sg = run_one(
                    planner,
                    problems["starts"][pi],
                    problems["goals"][pi],
                    max_expansions,
                    int(cell["hysteresis"]),
                )
                n_ok += metrics["success"]
                wall = float(metrics["wall_s"])
                exp = max(int(metrics["expansions"]), 1)
                metrics["exp_per_s"] = exp / wall if wall > 0 else float("nan")
                traj_rel = ""
                if path_sg is not None and not args.no_traj:
                    traj_path = os.path.join(traj_subdir, f"p{pi:03d}.npy")
                    np.save(traj_path, path_sg.astype(np.float32))
                    traj_rel = os.path.relpath(traj_path, CONTENT)
                    n_traj_saved += 1
                row = {
                    "res_xy": cell["res_xy"],
                    "action_radii": rtag,
                    "max_agent_vel": cell["vmax"],
                    "n_directions": cell["n_directions"],
                    "n_substeps": cell["n_substeps"],
                    "hysteresis": cell["hysteresis"],
                    "qs_support_radius": cell.get(
                        "qs_support_radius", BASELINE["qs_support_radius"]
                    ),
                    "dynamics": DYNAMICS,
                    "problem_idx": pi,
                    "dino_seed": int(problems["seeds"][pi]),
                    "traj_path": traj_rel,
                    **metrics,
                }
                rows.append(row)
                csv_w.writerow(row)
                csv_f.flush()
                n_done += 1
                print_progress(n_done, n_total, t_all, prefix=f"{tag} ")
            dt = time.perf_counter() - t_cfg
            sys.stderr.write("\n")
            print(
                f"[{ci + 1}/{len(grid)}] {tag}  "
                f"success={n_ok}/{n_prob}  wall={dt:.1f}s"
            )
    finally:
        csv_f.close()

    print(f"wrote {args.out}  ({len(rows)} rows)")
    if not args.no_traj:
        print(f"wrote {n_traj_saved} trajectories under {args.traj_dir}")

    # Aggregate success rate / median expansions among successes.
    print("\n=== summary (success rate, median cost|expansions among successes) ===")
    from collections import defaultdict

    buckets: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[
            (
                r["res_xy"],
                r["action_radii"],
                r["max_agent_vel"],
                r["n_directions"],
                r["n_substeps"],
                r["hysteresis"],
                r["qs_support_radius"],
            )
        ].append(r)
    print(
        f"{'res':>5} {'radii':>9} {'nd':>3} {'ns':>3} {'hyst':>5} {'R':>5}  "
        f"{'succ':>8}  {'med_1st':>8}  {'mean_wall':>9}  {'mean_succ_us':>12}  {'mean_exp/s':>10}"
    )
    for key in sorted(
        buckets.keys(), key=lambda k: (k[0], k[1], k[2], k[3], k[4], k[5], k[6])
    ):
        rs = buckets[key]
        ok = [r for r in rs if r["success"]]
        firsts = [r["first_sol_exp"] for r in ok if r["first_sol_exp"] >= 0]
        med_1st = float(np.median(firsts)) if firsts else float("nan")
        mean_wall = float(np.mean([r["wall_s"] for r in rs]))
        mean_succ = float(np.mean([r["avg_succ_us"] for r in rs]))
        mean_eps = float(np.mean([r["exp_per_s"] for r in rs]))
        print(
            f"{key[0]:5g} {key[1]:>9} {key[3]:3d} {key[4]:3d} {key[5]:5d} {key[6]:5g}  "
            f"{len(ok):3d}/{len(rs):<3d}  {med_1st:8.0f}  "
            f"{mean_wall:9.3f}  {mean_succ:12.1f}  {mean_eps:10.0f}"
        )


if __name__ == "__main__":
    main()
