#!/usr/bin/env python3
"""Q2: fixed init-T + goal-T, vary agent start — distance vs expansions.

Uses pusht_agent_grid_qs_10x10x50 (10×10 T pairs × 50 agents).
Within each (init_t, goal_t), T screw/xy is constant; agent–T distance varies.

  /home/stark/ighastar-venv/bin/python examples/standalone/pusht_agent_grid_dist_corr.py
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
OUT = CONTENT / "curve_plots" / "agent_grid_qs_10x10x50"
CSV = CONTENT / "pusht_agent_grid_qs_10x10x50_results.csv"
DINO = CONTENT / "pusht_dino_problems.npz"
BAR_LEN = 4.0 * 30.0


def pusher(agent_xy, t_xy):
    return math.hypot(float(agent_xy[0] - t_xy[0]), float(agent_xy[1] - t_xy[1]))


def screw(s_t, g_t, R=BAR_LEN):
    dx, dy = float(g_t[0] - s_t[0]), float(g_t[1] - s_t[1])
    dth = math.atan2(math.sin(g_t[2] - s_t[2]), math.cos(g_t[2] - s_t[2]))
    return math.sqrt(dx * dx + dy * dy + (R * dth) ** 2)


def spearman(xs, ys):
    return float(stats.spearmanr(xs, ys).correlation)


def load():
    rows = list(csv.DictReader(open(CSV)))
    dino = np.load(DINO)
    starts, goals = dino["starts"], dino["goals"]
    by = defaultdict(list)
    t_meta = {}
    for r in rows:
        it, gt, ag = int(r["init_t_idx"]), int(r["goal_t_idx"]), int(r["agent_idx"])
        init_t = starts[it][2:5]
        goal_t = goals[gt][2:5]
        agent = starts[ag][:2]
        by[(it, gt)].append(
            {
                "ag": ag,
                "exp": float(r["first_sol_exp"]),
                "pusher": pusher(agent, init_t),
                "success": int(r["success"]),
            }
        )
        t_meta[(it, gt)] = {
            "init_rank": int(r["init_t_rank"]),
            "goal_rank": int(r["goal_t_rank"]),
            "screw": screw(init_t, goal_t),
        }
    # Order panels: init_rank major, goal_rank minor.
    order = sorted(
        t_meta.keys(),
        key=lambda k: (t_meta[k]["init_rank"], t_meta[k]["goal_rank"]),
    )
    return by, t_meta, order, starts


def plot_pusher_facets(by, t_meta, order, out_path, max_panels=20):
    # Show a manageable subset: every 5th T-pair along the ordered list, plus ends.
    if len(order) > max_panels:
        idx = np.unique(np.linspace(0, len(order) - 1, max_panels, dtype=int))
        show = [order[i] for i in idx]
    else:
        show = order
    n = len(show)
    ncols = 5
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.6 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, key in zip(axes, show):
        xs = by[key]
        x = np.array([p["pusher"] for p in xs])
        y = np.array([p["exp"] for p in xs])
        rho = spearman(x, y)
        ax.scatter(x, y, s=14, c="C0", alpha=0.7, edgecolors="none")
        if np.std(x) > 0:
            m, b = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 40)
            ax.plot(xx, m * xx + b, color="C3", lw=1.1, alpha=0.85)
        it, gt = key
        m = t_meta[key]
        ax.set_title(
            f"it{it:02d}→gt{gt:02d} (ir{m['init_rank']} gr{m['goal_rank']})\n"
            f"screw={m['screw']:.0f}  ρ={rho:.2f}",
            fontsize=8,
        )
        ax.set_xlabel("agent–T xy (start)", fontsize=8)
        ax.set_ylabel("first_sol_exp", fontsize=8)
    for ax in axes[len(show) :]:
        ax.axis("off")
    fig.suptitle(
        "Q2: fixed init-T + goal-T, vary agent  |  agent–T dist vs expansions",
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_rho_heatmap(by, t_meta, order, out_path):
    # 10×10 grid of Spearman(pusher, exp) keyed by init_rank × goal_rank.
    ranks_i = sorted({t_meta[k]["init_rank"] for k in order})
    ranks_g = sorted({t_meta[k]["goal_rank"] for k in order})
    # Map rank → problem idx for labels.
    init_of_rank = {}
    goal_of_rank = {}
    for (it, gt), m in t_meta.items():
        init_of_rank[m["init_rank"]] = it
        goal_of_rank[m["goal_rank"]] = gt

    mat = np.full((len(ranks_i), len(ranks_g)), np.nan)
    for (it, gt), xs in by.items():
        m = t_meta[(it, gt)]
        ri, rg = m["init_rank"], m["goal_rank"]
        x = [p["pusher"] for p in xs]
        y = [p["exp"] for p in xs]
        mat[ri, rg] = spearman(x, y)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm", origin="upper")
    ax.set_xticks(range(len(ranks_g)))
    ax.set_yticks(range(len(ranks_i)))
    ax.set_xticklabels([f"gt{goal_of_rank[r]:02d}\nr{r}" for r in ranks_g], fontsize=7)
    ax.set_yticklabels([f"it{init_of_rank[r]:02d}\nr{r}" for r in ranks_i], fontsize=7)
    ax.set_xlabel("goal T (rank worst→best)")
    ax.set_ylabel("init T (rank worst→best)")
    ax.set_title("Spearman ρ(agent–T dist, first_sol_exp) per fixed T-pair")
    fig.colorbar(im, ax=ax, fraction=0.046, label="ρ")
    # Annotate
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out_path}")
    return mat


def pooled_stats(by):
    print("\nPooled within-(init_t,goal_t) demeaned correlations:")
    xd, yd = [], []
    rhos = []
    for xs in by.values():
        a = np.array([p["pusher"] for p in xs])
        e = np.array([p["exp"] for p in xs])
        rhos.append(spearman(a, e))
        xd.append(a - a.mean())
        yd.append(e - e.mean())
    xd, yd = np.concatenate(xd), np.concatenate(yd)
    r = float(stats.pearsonr(xd, yd).statistic)
    rho = float(stats.spearmanr(xd, yd).correlation)
    print(f"  agent–T xy  Pearson r={r:.3f}  Spearman ρ={rho:.3f}")
    print(
        f"  per-pair ρ:  median={np.median(rhos):.3f}  "
        f"mean={np.mean(rhos):.3f}  min={np.min(rhos):.3f}  max={np.max(rhos):.3f}"
    )
    return r, rho, rhos


def plot_rho_hist(rhos, out_path):
    fig, ax = plt.subplots(figsize=(4.5, 3), constrained_layout=True)
    ax.hist(rhos, bins=15, color="C0", edgecolor="white")
    ax.axvline(np.median(rhos), color="C3", lw=1.5, label=f"median={np.median(rhos):.2f}")
    ax.set_xlabel("Spearman ρ(agent–T, first_sol_exp)")
    ax.set_ylabel("# of (init_t, goal_t) pairs")
    ax.set_title("Q2: distribution of per-pair correlations (n=100)")
    ax.legend(fontsize=9)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    by, t_meta, order, _ = load()
    assert len(by) == 100, len(by)
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, rhos = pooled_stats(by)
    plot_rho_heatmap(by, t_meta, order, OUT / "q2_pusher_rho_heatmap.png")
    plot_rho_hist(rhos, OUT / "q2_pusher_rho_hist.png")
    plot_pusher_facets(by, t_meta, order, OUT / "q2_pusher_vs_exp_facets.png")


if __name__ == "__main__":
    main()
