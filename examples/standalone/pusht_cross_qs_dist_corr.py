#!/usr/bin/env python3
"""Fixed-start analysis: goal-T distance vs first_sol_exp (cross QS 10×50).

Within each start, agent–T start distance is constant; only the goal T varies.
  /home/stark/ighastar-venv/bin/python examples/standalone/pusht_cross_qs_dist_corr.py
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from ighastar.scripts.common_utils import BASE_DIR

REPO = Path(BASE_DIR).parent
CONTENT = REPO / "Content" / "standalone"
OUT = CONTENT / "curve_plots" / "cross_qs_10x50"
CSV = CONTENT / "pusht_cross_qs_10x50_results.csv"
DINO = CONTENT / "pusht_dino_problems.npz"
BAR_LEN = 4.0 * 30.0


def screw(s, g, R=BAR_LEN):
    dx, dy = float(g[2] - s[2]), float(g[3] - s[3])
    dth = math.atan2(math.sin(g[4] - s[4]), math.cos(g[4] - s[4]))
    return math.sqrt(dx * dx + dy * dy + (R * dth) ** 2)


def t_xy(s, g):
    return math.hypot(float(g[2] - s[2]), float(g[3] - s[3]))


def t_dth(s, g):
    return abs(math.atan2(math.sin(g[4] - s[4]), math.cos(g[4] - s[4])))


def pusher(s):
    return math.hypot(float(s[0] - s[2]), float(s[1] - s[3]))


def load():
    rows = list(csv.DictReader(open(CSV)))
    dino = np.load(DINO)
    starts, goals = dino["starts"], dino["goals"]
    by = defaultdict(list)
    meta = {}
    for r in rows:
        si, gi = int(r["start_idx"]), int(r["goal_idx"])
        s, g = starts[si], goals[gi]
        by[si].append(
            {
                "gi": gi,
                "exp": float(r["first_sol_exp"]),
                "screw": screw(s, g),
                "txy": t_xy(s, g),
                "dth": t_dth(s, g),
            }
        )
        meta[si] = {
            "rank": int(r["rank_worst_to_best"]),
            "pusher": pusher(s),
        }
    order = sorted(meta, key=lambda si: meta[si]["rank"])
    return by, meta, order


def spearman(xs, ys):
    return float(stats.spearmanr(xs, ys).correlation)


def plot_faceted(by, meta, order, dist_key, xlabel, out_path):
    fig, axes = plt.subplots(2, 5, figsize=(14, 6), constrained_layout=True)
    for ax, si in zip(axes.ravel(), order):
        xs = by[si]
        x = np.array([p[dist_key] for p in xs])
        y = np.array([p["exp"] for p in xs])
        rho = spearman(x, y)
        ax.scatter(x, y, s=18, c="C0", alpha=0.75, edgecolors="none")
        # crude trend
        if len(x) > 2 and np.std(x) > 0:
            m, b = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(xx, m * xx + b, color="C3", lw=1.2, alpha=0.8)
        ax.set_title(
            f"s{si:02d} (rank {meta[si]['rank']}, "
            f"pusher={meta[si]['pusher']:.0f})\nρ={rho:.2f}",
            fontsize=9,
        )
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("first_sol_exp", fontsize=8)
    fig.suptitle(
        f"QS cross: fixed start → 50 goals  |  {xlabel} vs expansions",
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_summary(by, meta, order, out_path):
    keys = [("screw", "T screw"), ("txy", "T xy"), ("dth", "|Δθ|")]
    rhos = {k: [] for k, _ in keys}
    for si in order:
        xs = by[si]
        y = [p["exp"] for p in xs]
        for k, _ in keys:
            rhos[k].append(spearman([p[k] for p in xs], y))

    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    xpos = np.arange(len(order))
    w = 0.25
    for i, (k, lab) in enumerate(keys):
        ax.bar(xpos + (i - 1) * w, rhos[k], width=w, label=lab)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(
        [f"s{si:02d}\nr{meta[si]['rank']}" for si in order], fontsize=8
    )
    ax.set_ylabel("Spearman ρ with first_sol_exp")
    ax.set_ylim(-1, 1)
    ax.legend(fontsize=9)
    ax.set_title("Per-start correlation: goal-T distance ↔ expansions")
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out_path}")

    # Pooled within-start demeaned
    print("\nPooled within-start (demeaned) correlations with first_sol_exp:")
    for k, lab in keys:
        xd, yd = [], []
        for si in order:
            xs = by[si]
            a = np.array([p[k] for p in xs])
            e = np.array([p["exp"] for p in xs])
            xd.append(a - a.mean())
            yd.append(e - e.mean())
        xd, yd = np.concatenate(xd), np.concatenate(yd)
        r = float(stats.pearsonr(xd, yd).statistic)
        rho = float(stats.spearmanr(xd, yd).correlation)
        print(f"  {lab:8s}  Pearson r={r:.3f}  Spearman ρ={rho:.3f}")


def main():
    by, meta, order = load()
    OUT.mkdir(parents=True, exist_ok=True)
    plot_faceted(
        by, meta, order, "screw", "T screw (px)",
        OUT / "fixed_start_screw_vs_exp.png",
    )
    plot_faceted(
        by, meta, order, "txy", "T xy (px)",
        OUT / "fixed_start_txy_vs_exp.png",
    )
    plot_faceted(
        by, meta, order, "dth", "|Δθ| (rad)",
        OUT / "fixed_start_dth_vs_exp.png",
    )
    plot_summary(by, meta, order, OUT / "fixed_start_corr_summary.png")


if __name__ == "__main__":
    main()
