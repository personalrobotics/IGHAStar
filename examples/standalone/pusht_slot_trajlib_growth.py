#!/usr/bin/env python3
"""Grow a per-goal slot trajectory library and plot success and cost by library size.

Thirty goals are spaced from the easiest saved slot problem to the hardest.
Each goal is solved once with an empty library and that solution is the first
trajectory. Ten other valid starts are then searched at 100k expansions. A
solution is appended only when the search succeeds, so the next start sees a
larger library. Curves pool every run that started with the same library size.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ighastar.scripts.common_utils import create_planner

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from pusht_multi_generate_slot_rearrange import free_region  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "slot_cost_curves", REPO / "utils" / "cost_curves.py"
)
_curves = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_curves)
pad_best_cost = _curves.pad_best_cost
success_vs_expansions = _curves.success_vs_expansions

PROBLEMS = REPO / "Content" / "standalone" / "pusht_multi_slot_1T_rearrange.npz"
CONFIG = HERE / "Configs" / "pusht_multi_slot_1T_rearrange.yml"
OUT = REPO / "Content" / "standalone" / "curve_plots" / "slot_trajlib_growth"
N_GOALS = 30
N_DONORS = 10
SEED = 0
COST_INF = 1e4


def in_workspace(start: np.ndarray, side: str) -> bool:
    x0, x1, y0, y1 = free_region(side)
    return x0 <= float(start[2]) <= x1 and y0 <= float(start[3]) <= y1


def first_sol(curve: np.ndarray) -> int:
    hit = np.where(np.asarray(curve) < COST_INF)[0]
    return int(hit[0]) if hit.size else -1


def search(planner, start, goal, world, limit, hyst):
    t0 = time.perf_counter()
    ok = bool(
        planner.search(
            torch.from_numpy(np.asarray(start, dtype=np.float32)),
            torch.from_numpy(np.asarray(goal, dtype=np.float32)),
            torch.from_numpy(np.asarray(world, dtype=np.float32)),
            int(limit),
            int(hyst),
            False,
        )
    )
    path = planner.get_best_path()
    if ok and path.numel() == 0:
        ok = False
    curve = np.asarray(planner.get_profiler_info()[9], dtype=np.float64)
    cost = float("nan")
    if ok:
        cost = float(path.detach().cpu().numpy()[0, -1])
    return ok, curve, cost, time.perf_counter() - t0


def choose_goals(first: np.ndarray) -> np.ndarray:
    order = np.argsort(first)
    idx = np.unique(np.round(np.linspace(0, len(order) - 1, N_GOALS)).astype(int))
    return order[idx]


def donors_for(planner, data, goal_i: int, rng: np.random.RandomState):
    side = str(data["slot_side"][goal_i])
    others = [i for i in range(len(data["starts"])) if i != goal_i and in_workspace(data["starts"][i], side)]
    if not others:
        return []
    states = torch.from_numpy(np.stack([data["starts"][i] for i in others]))
    world = np.asarray(data["obstacles"][goal_i], dtype=np.float32)
    worlds = torch.from_numpy(np.stack([world] * len(others)))
    ok = planner.check_validity_batched(states, worlds).detach().cpu().numpy().astype(bool)
    valid = [others[k] for k in range(len(others)) if ok[k]]
    rng.shuffle(valid)
    return valid[:N_DONORS]


def library_size(planner) -> int:
    return len(planner.export_trajectory_library())


def write_run(goal, donor, n_traj, curve, ok, cost, wall, first, shard_id=0, num_shards=1):
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"g{goal:03d}_d{donor:03d}.pkl" if n_traj else f"g{goal:03d}_d{donor:03d}_n0.pkl"
    path = OUT / name
    with open(path, "wb") as f:
        pickle.dump(
            {
                "best_cost": curve,
                "goal_id": int(goal),
                "donor_id": int(donor),
                "library_size": int(n_traj),
                "success": bool(ok),
                "cost": float(cost),
                "wall_s": float(wall),
                "first_sol": int(first),
            },
            f,
        )
    csv_name = "runs.csv" if num_shards == 1 else f"runs_shard{shard_id}.csv"
    csv_path = OUT / csv_name
    new = not csv_path.is_file()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "goal", "donor", "library_size", "success",
                "first_sol", "cost", "wall_s",
            ],
        )
        if new:
            w.writeheader()
        w.writerow(
            {
                "goal": goal,
                "donor": donor,
                "library_size": n_traj,
                "success": int(ok),
                "first_sol": first,
                "cost": cost,
                "wall_s": f"{wall:.3f}",
            }
        )


def goal_list(data) -> np.ndarray:
    return choose_goals(np.asarray(data["first_sol_expansions"]))


def shard_of(items, shard_id: int, num_shards: int):
    return [item for i, item in enumerate(items) if i % num_shards == shard_id]


def mark_goal(goal: int) -> None:
    dest = OUT / "goals"
    dest.mkdir(parents=True, exist_ok=True)
    np.savez(dest / f"p{int(goal):03d}.npz", goal=np.int32(goal))


def run(shard_id: int = 0, num_shards: int = 1) -> None:
    data = np.load(PROBLEMS, allow_pickle=True)
    goals = goal_list(data)
    mine = shard_of(list(goals), shard_id, num_shards)
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    info = cfg["experiment_info_default"]
    limit = int(info["max_expansions"])
    hyst = int(info["hysteresis"])
    planner = create_planner(cfg, bidirectional=False)
    rng = np.random.RandomState(SEED)
    print(f"goals {len(goals)} mine {len(mine)} limit {limit}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if num_shards == 1:
        for stale in OUT.glob("g*_d*.pkl"):
            if not stale.name.endswith("_n0.pkl"):
                stale.unlink()
        csv_path = OUT / "runs.csv"
        if csv_path.is_file():
            csv_path.unlink()
    for goal in mine:
        for stale in OUT.glob(f"g{int(goal):03d}_d*.pkl"):
            if not stale.name.endswith("_n0.pkl"):
                stale.unlink()
        marker = OUT / "goals" / f"p{int(goal):03d}.npz"
        if marker.is_file():
            marker.unlink()
        picked = donors_for(planner, data, int(goal), rng)
        print(
            f"goal p{int(goal)} side={data['slot_side'][goal]} donors={len(picked)}",
            flush=True,
        )
        planner.import_trajectory_library([])
        ok, _curve, cost, wall = search(
            planner, data["starts"][goal], data["goals"][goal],
            data["obstacles"][goal], limit, hyst,
        )
        print(f"  seed ok={ok} cost={cost:.3f} {wall:.1f}s", flush=True)
        if not ok:
            mark_goal(int(goal))
            continue
        planner.record_solution_trajectory()
        for donor in picked:
            n_traj = library_size(planner)
            ok, curve, cost, wall = search(
                planner, data["starts"][donor], data["goals"][goal],
                data["obstacles"][goal], limit, hyst,
            )
            first = first_sol(curve)
            write_run(
                int(goal), int(donor), n_traj, curve, ok, cost, wall, first,
                shard_id, num_shards,
            )
            if ok:
                planner.record_solution_trajectory()
            print(
                f"  start p{int(donor)} n={n_traj} ok={ok} first={first} "
                f"cost={cost:.3f} {wall:.1f}s",
                flush=True,
            )
        mark_goal(int(goal))


def group_cost(rows, max_exp):
    import scipy.stats as st

    mat = np.stack(rows)
    finite = np.isfinite(mat)
    count = finite.sum(axis=0)
    mean = np.full(max_exp, np.nan)
    ci = np.full(max_exp, np.nan)
    summed = np.where(finite, mat, 0.0).sum(axis=0)
    nz = count > 0
    mean[nz] = summed[nz] / count[nz]
    for n in np.unique(count[count >= 2]):
        cols = count == n
        vals = np.where(finite[:, cols], mat[:, cols], np.nan)
        std = np.nanstd(vals, axis=0, ddof=1)
        half = float(st.t.ppf(0.975, n - 1)) * std / np.sqrt(n)
        ci[cols] = half
    ci[count == 1] = 0.0
    return mean, ci


def baseline_pairs():
    pairs = []
    for path in sorted(OUT.glob("g*_d*.pkl")):
        if path.name.endswith("_n0.pkl"):
            continue
        with open(path, "rb") as f:
            row = pickle.load(f)
        pairs.append((int(row["goal_id"]), int(row["donor_id"])))
    return list(dict.fromkeys(pairs))


def mark_baseline(index: int) -> None:
    dest = OUT / "baseline"
    dest.mkdir(parents=True, exist_ok=True)
    np.savez(dest / f"p{index:03d}.npz", pair=np.int32(index))


def baseline(shard_id: int = 0, num_shards: int = 1) -> None:
    """Same goal/donor pairs, searched with an empty library."""
    data = np.load(PROBLEMS, allow_pickle=True)
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    info = cfg["experiment_info_default"]
    limit = int(info["max_expansions"])
    hyst = int(info["hysteresis"])
    pairs = baseline_pairs()
    mine = [(i, pair) for i, pair in enumerate(pairs) if i % num_shards == shard_id]
    planner = create_planner(cfg, bidirectional=False)
    planner.import_trajectory_library([])
    print(f"baseline pairs {len(pairs)} mine {len(mine)}", flush=True)
    for n_done, (i, (goal, donor)) in enumerate(mine):
        out = OUT / f"g{goal:03d}_d{donor:03d}_n0.pkl"
        if not out.is_file():
            ok, curve, cost, wall = search(
                planner, data["starts"][donor], data["goals"][goal],
                data["obstacles"][goal], limit, hyst,
            )
            first = first_sol(curve)
            write_run(goal, donor, 0, curve, ok, cost, wall, first, shard_id, num_shards)
            print(
                f"  {n_done + 1}/{len(mine)} goal p{goal} start p{donor} "
                f"ok={ok} first={first} cost={cost:.3f} {wall:.1f}s",
                flush=True,
            )
        mark_baseline(i)


def curve_color(size: int, largest: int):
    if size <= 0:
        return "black"
    import matplotlib.pyplot as plt
    # Light red is one trajectory. Dark red is the largest library.
    t = 0.35 + 0.65 * ((size - 1) / max(largest - 1, 1))
    return plt.cm.Reds(t)


def plot() -> None:
    import matplotlib.pyplot as plt

    pkls = sorted(OUT.glob("g*_d*.pkl"))
    if not pkls:
        raise SystemExit(f"no runs in {OUT}")
    loaded = []
    for path in pkls:
        with open(path, "rb") as f:
            loaded.append(pickle.load(f))
    max_exp = max(len(r["best_cost"]) for r in loaded)
    by_size = {}
    for row in loaded:
        by_size.setdefault(int(row["library_size"]), []).append(
            pad_best_cost(np.asarray(row["best_cost"]), max_exp)
        )
    curves = {str(k): by_size[k] for k in sorted(by_size)}
    largest = max(by_size)
    xs = np.arange(max_exp)

    succ = success_vs_expansions(curves, max_exp)
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for name in curves:
        rate, lo, hi = succ[name]
        color = curve_color(int(name), largest)
        ax.plot(xs, rate * 100, color=color, lw=1.5)
        ax.fill_between(xs, lo * 100, hi * 100, color=color, alpha=0.12)
    ax.set_xlabel("Expansions")
    ax.set_ylabel("Success (%)")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(OUT / "success_vs_expansions.png", dpi=150)
    fig.savefig(OUT / "success_vs_expansions.pdf")
    print(f"wrote {OUT / 'success_vs_expansions.png'}", flush=True)

    fig, ax = plt.subplots(figsize=(6.2, 4))
    for name in curves:
        mean, ci = group_cost(curves[name], max_exp)
        mask = np.isfinite(mean) & np.isfinite(ci)
        if not np.any(mask):
            continue
        color = curve_color(int(name), largest)
        ax.plot(xs[mask], mean[mask], color=color, lw=1.5)
        ax.fill_between(
            xs[mask], mean[mask] - ci[mask], mean[mask] + ci[mask],
            color=color, alpha=0.12,
        )
    ax.set_xlabel("Expansions")
    ax.set_ylabel("Cost")
    fig.tight_layout()
    fig.savefig(OUT / "cost_vs_expansions.png", dpi=150)
    fig.savefig(OUT / "cost_vs_expansions.pdf")
    print(f"wrote {OUT / 'cost_vs_expansions.png'}", flush=True)
    for name in curves:
        print(f"  n={name} runs={len(curves[name])}", flush=True)


def main() -> None:
    global CONFIG, PROBLEMS, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--problems", type=Path, default=PROBLEMS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    CONFIG = args.config
    PROBLEMS = args.problems
    OUT = args.out
    if args.num_shards < 1 or not (0 <= args.shard_id < args.num_shards):
        raise SystemExit(
            f"bad shard: shard-id={args.shard_id} num-shards={args.num_shards}"
        )
    if args.dry_run:
        if args.baseline:
            n = len(baseline_pairs())
        else:
            data = np.load(PROBLEMS, allow_pickle=True)
            n = len(goal_list(data))
        print(f"TOTAL {n}", flush=True)
        return
    if args.warmup:
        with open(CONFIG) as f:
            cfg = yaml.safe_load(f)
        create_planner(cfg, bidirectional=False)
        print("warmup ok", flush=True)
        return
    if args.plot_only:
        plot()
        return
    if args.baseline:
        baseline(args.shard_id, args.num_shards)
    else:
        run(args.shard_id, args.num_shards)
    if args.num_shards == 1:
        plot()


if __name__ == "__main__":
    main()
