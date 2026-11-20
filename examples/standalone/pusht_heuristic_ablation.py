#!/usr/bin/env python3
"""Single-tee PushT at 300k expansions, with the block heuristic and with h = 0.

Problems are the saved dino set. They are ordered easiest to hardest by the
saved chipmunk first-solution expansion, then 50 are taken on an even grid.
The saved set has 50, so the grid is the whole set. Nothing is re-sampled.
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import torch
import yaml

from ighastar.scripts.common_utils import create_planner
from pusht_condition_sweep import tree_payload

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROBLEMS = REPO / "Content" / "standalone" / "pusht_dino_problems.npz"
RANK_CSV = REPO / "Content" / "standalone" / "pusht_sweep_chipmunk_oat_50k_results.csv"
CONFIG = HERE / "Configs" / "pusht.yml"
OUT = REPO / "Content" / "standalone" / "curve_plots" / "pusht_heuristic_300k"
N_TAKE = 50
MAX_EXP = 300000
COST_INF = 1e4

_spec = importlib.util.spec_from_file_location(
    "pusht_cost_curves", REPO / "utils" / "cost_curves.py"
)
_curves = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_curves)
pad_best_cost = _curves.pad_best_cost
success_vs_expansions = _curves.success_vs_expansions


def difficulty_order(n: int) -> np.ndarray:
    """Easiest first, using the ndir=16 chipmunk cell that matches pusht.yml."""
    if not RANK_CSV.is_file():
        return np.arange(n)
    rows = [
        r
        for r in csv.DictReader(RANK_CSV.open())
        if r["dynamics"] == "chipmunk"
        and r["res_xy"] == "40.0"
        and r["action_radii"] == "200"
        and r["max_agent_vel"] == "500.0"
        and r["n_directions"] == "16"
        and r["n_substeps"] == "10"
    ]
    rank = {}
    for r in rows:
        i = int(r["problem_idx"])
        first = int(float(r["first_sol_exp"]))
        rank[i] = first if int(r["success"]) and first >= 0 else 10**9
    return np.array(sorted(range(n), key=lambda i: (rank.get(i, 10**9), i)))


def even_sample(order: np.ndarray, n_take: int) -> np.ndarray:
    if len(order) <= n_take:
        return order
    idx = np.unique(np.round(np.linspace(0, len(order) - 1, n_take)).astype(int))
    return order[idx]


def load_cfg(use_heuristic: bool):
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    info = cfg["experiment_info_default"]
    info["max_expansions"] = MAX_EXP
    info["node_info"]["use_heuristic"] = bool(use_heuristic)
    return cfg


def search(planner, start, goal, limit, hyst):
    ok = bool(
        planner.search(
            torch.as_tensor(start, dtype=torch.float32),
            torch.as_tensor(goal, dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32),
            int(limit),
            int(hyst),
            True,
        )
    )
    path = planner.get_best_path()
    if ok and path.numel() == 0:
        ok = False
    curve = np.asarray(planner.get_profiler_info()[9], dtype=np.float64)
    return ok, curve


def run_condition(name: str, use_heuristic: bool, problems, indices) -> None:
    cfg = load_cfg(use_heuristic)
    info = cfg["experiment_info_default"]
    planner = create_planner(cfg, bidirectional=False)
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(indices):
        path = dest / f"p{k:03d}.npz"
        if path.is_file():
            print(f"{name} {k + 1}/{len(indices)} p{int(i)} cached", flush=True)
            continue
        ok, curve = search(
            planner, problems["starts"][i], problems["goals"][i],
            MAX_EXP, int(info["hysteresis"]),
        )
        tree = tree_payload(planner.get_search_tree())
        n_nodes = int(tree["num_nodes"])
        payload = dict(tree)
        payload["best_cost"] = np.asarray(curve, dtype=np.float64)
        payload["problem_idx"] = np.int32(i)
        payload["success"] = np.uint8(ok)
        np.savez(path, **payload)
        del tree, payload
        hit = np.where(curve < COST_INF)[0]
        first = int(hit[0]) if hit.size else -1
        print(
            f"{name} {k + 1}/{len(indices)} p{int(i)} ok={ok} first={first} "
            f"nodes={n_nodes}",
            flush=True,
        )


def mean_cost(rows):
    import scipy.stats as st

    mat = np.stack(rows)
    finite = np.isfinite(mat)
    count = finite.sum(axis=0)
    mean = np.full(mat.shape[1], np.nan)
    ci = np.full(mat.shape[1], np.nan)
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


def plot() -> None:
    import matplotlib.pyplot as plt

    series = (("with", "red"), ("without", "black"))
    loaded = {}
    for name, _ in series:
        rows = []
        for path in sorted((OUT / name).glob("p*.npz")):
            with np.load(path) as saved:
                rows.append(np.asarray(saved["best_cost"]))
        loaded[name] = [pad_best_cost(np.asarray(r), MAX_EXP) for r in rows]
    curves = {name: loaded[name] for name, _ in series}
    succ = success_vs_expansions(curves, MAX_EXP)
    xs = np.arange(MAX_EXP)

    fig, ax = plt.subplots(figsize=(6.2, 4))
    for name, color in series:
        rate, lo, hi = succ[name]
        ax.plot(xs, rate * 100, color=color, lw=1.5, label=name)
        ax.fill_between(xs, lo * 100, hi * 100, color=color, alpha=0.12)
    ax.set_xlabel("Expansions")
    ax.set_ylabel("Success (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "success_vs_expansions.png", dpi=150)
    print(f"wrote {OUT / 'success_vs_expansions.png'}", flush=True)

    fig, ax = plt.subplots(figsize=(6.2, 4))
    for name, color in series:
        mean, ci = mean_cost(curves[name])
        mask = np.isfinite(mean) & np.isfinite(ci)
        ax.plot(xs[mask], mean[mask], color=color, lw=1.5, label=name)
        ax.fill_between(
            xs[mask], mean[mask] - ci[mask], mean[mask] + ci[mask],
            color=color, alpha=0.12,
        )
    ax.set_xlabel("Expansions")
    ax.set_ylabel("Cost")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "cost_vs_expansions.png", dpi=150)
    print(f"wrote {OUT / 'cost_vs_expansions.png'}", flush=True)
    for name, _ in series:
        last = loaded[name]
        n_ok = sum(np.isfinite(c[-1]) for c in last)
        print(f"  {name} {n_ok}/{len(last)}", flush=True)


def main() -> None:
    data = np.load(PROBLEMS)
    order = difficulty_order(len(data["starts"]))
    chosen = even_sample(order, N_TAKE)
    print(f"problems {len(chosen)} of {len(data['starts'])} limit {MAX_EXP}", flush=True)
    run_condition("with", True, data, chosen)
    run_condition("without", False, data, chosen)
    plot()


if __name__ == "__main__":
    main()
