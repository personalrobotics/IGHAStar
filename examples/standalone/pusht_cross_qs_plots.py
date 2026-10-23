#!/usr/bin/env python3
"""Success/cost curves + interactive traj HTMLs for the QS 10×50 cross sweep.

  OMP_NUM_THREADS=1 /home/stark/ighastar-venv/bin/python \\
    examples/standalone/pusht_cross_qs_plots.py --workers 8

Outputs under Content/standalone/curve_plots/cross_qs_10x50/:
  success_vs_expansions.{png,pdf}
  cost_vs_expansions.{png,pdf}
  pusht_cross_qs_traj_s{si:02d}.html   # one per start (50 goals)
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import yaml
from matplotlib.colors import Normalize, to_hex

from ighastar.scripts.common_utils import BASE_DIR, create_planner

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pusht_cross_qs_sweep import (  # noqa: E402
    CFG_PATH,
    OUT_CSV,
    OUT_NPZ,
    TRAJ_DIR,
    already_at_goal,
    make_pair,
)
from pusht_hard_easy_scatter import (  # noqa: E402
    agent_circle_xy,
    agent_circles_xy,
    agent_in_t_frame,
    block_in_t_frame,
    hex_rgba,
    place_t,
    plotly_traj_hover_hook,
    t_outline_xy,
    t_polys_body_frame,
)

REPO = Path(BASE_DIR).parent
CONTENT = REPO / "Content" / "standalone"
OUT_DIR = CONTENT / "curve_plots" / "cross_qs_10x50"
PKL_DIR = CONTENT / "curves" / "cross_qs_10x50"
MAX_EXP = 50000
COST_INF = 1e4

_WORKER: Dict[str, Any] = {}


def load_rows() -> List[Dict[str, str]]:
    with open(OUT_CSV) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (int(r["start_idx"]), int(r["goal_idx"])))
    return rows


def pkl_path(si: int, gi: int) -> Path:
    return PKL_DIR / f"s{si:02d}_g{gi:02d}.pkl"


def _worker_init(cfg_path: str, max_exp: int) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["experiment_info_default"]["max_expansions"] = int(max_exp)
    _WORKER["cfg"] = cfg
    _WORKER["info"] = cfg["experiment_info_default"]
    _WORKER["planner"] = create_planner(cfg, bidirectional=False)
    _WORKER["n"] = 0


def _worker_collect(job: Tuple[int, int, np.ndarray, np.ndarray, str]):
    si, gi, start, goal, out_s = job
    out = Path(out_s)
    if already_at_goal(start, goal):
        # Trivial: solved at expansion 0 with cost 0.
        payload = {
            "best_cost": np.array([0.0], dtype=np.float64),
            "wall_s": 0.0,
            "start_idx": si,
            "goal_idx": gi,
            "trivial": True,
        }
        with open(out, "wb") as f:
            pickle.dump(payload, f)
        return si, gi, True, 0.0

    n = _WORKER["n"]
    if n > 0 and n % 10 == 0:
        _WORKER["planner"] = create_planner(_WORKER["cfg"], bidirectional=False)
    _WORKER["n"] = n + 1
    info = _WORKER["info"]
    t0 = time.perf_counter()
    _WORKER["planner"].search(
        torch.as_tensor(start),
        torch.as_tensor(goal),
        torch.zeros(1, dtype=torch.float32),
        info["max_expansions"],
        info["hysteresis"],
        True,
    )
    wall = time.perf_counter() - t0
    cost_exp = np.asarray(
        _WORKER["planner"].get_profiler_info()[9], dtype=np.float64
    )
    with open(out, "wb") as f:
        pickle.dump(
            {
                "best_cost": cost_exp,
                "wall_s": wall,
                "start_idx": si,
                "goal_idx": gi,
            },
            f,
        )
    # Also refresh traj if missing.
    traj = TRAJ_DIR / f"s{si:02d}_g{gi:02d}.npy"
    if not traj.is_file():
        try:
            path = _WORKER["planner"].get_best_path().numpy()[::-1].copy()
            if path.size:
                TRAJ_DIR.mkdir(parents=True, exist_ok=True)
                np.save(traj, path.astype(np.float32))
        except Exception:
            pass
    return si, gi, True, wall


def collect_pkls(workers: int, max_exp: int) -> None:
    PKL_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(OUT_NPZ)
    starts, goals = data["starts"], data["goals"]
    start_idx = data["start_idx"]
    goal_idx = data["goal_idx"]
    jobs = []
    for k in range(len(start_idx)):
        si, gi = int(start_idx[k]), int(goal_idx[k])
        out = pkl_path(si, gi)
        if out.is_file():
            continue
        jobs.append((si, gi, starts[k], goals[k], str(out)))
    print(f"collect pkls: {len(jobs)} missing / {len(start_idx)}")
    if not jobs:
        return
    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(str(CFG_PATH), max_exp),
    ) as ex:
        futs = [ex.submit(_worker_collect, j) for j in jobs]
        for fut in as_completed(futs):
            si, gi, ok, wall = fut.result()
            done += 1
            sys.stderr.write(
                f"\r  pkl [{done}/{len(jobs)}] s{si:02d}g{gi:02d} wall={wall:.2f}s   "
            )
            sys.stderr.flush()
    sys.stderr.write("\n")
    print(f"collect done in {time.perf_counter() - t0:.1f}s")


def pad_best_cost(raw: np.ndarray, max_exp: int) -> np.ndarray:
    out = np.full(max_exp, np.nan, dtype=np.float64)
    n = min(len(raw), max_exp)
    if n == 0:
        return out
    chunk = np.asarray(raw[:n], dtype=np.float64).copy()
    chunk[chunk >= COST_INF] = np.nan
    out[:n] = chunk
    last = chunk[n - 1]
    if np.isfinite(last):
        out[n:] = last
    return out


def load_all_curves(max_exp: int) -> List[np.ndarray]:
    data = np.load(OUT_NPZ)
    start_idx, goal_idx = data["start_idx"], data["goal_idx"]
    curves = []
    for k in range(len(start_idx)):
        si, gi = int(start_idx[k]), int(goal_idx[k])
        with open(pkl_path(si, gi), "rb") as f:
            raw = np.asarray(pickle.load(f)["best_cost"], dtype=np.float64)
        curves.append(pad_best_cost(raw, max_exp))
    return curves


def plot_success_cost(curves: List[np.ndarray], max_exp: int) -> None:
    # Local import of the suite helpers (same policy as utils/cost_curves.py).
    sys.path.insert(0, str(REPO / "utils"))
    from cost_curves import cost_vs_expansions, success_vs_expansions  # type: ignore

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    named = {"cross_qs_10x50": curves}
    labels = {"cross_qs_10x50": "QS 10×50 cross"}
    xs = np.arange(max_exp)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    succ = success_vs_expansions(named, max_exp)
    fig, ax = plt.subplots(figsize=(4, 3))
    rate, lo, hi = succ["cross_qs_10x50"]
    c = colors[0]
    ax.plot(xs, rate * 100, label=labels["cross_qs_10x50"], color=c, lw=2)
    ax.fill_between(xs, lo * 100, hi * 100, color=c, alpha=0.2)
    ax.set_xlabel("Expansions")
    ax.set_ylabel("Success (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "success_vs_expansions.png", dpi=150)
    fig.savefig(OUT_DIR / "success_vs_expansions.pdf")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'success_vs_expansions.png'}")

    cost = cost_vs_expansions(named, max_exp, min_exp_threshold=max_exp)
    fig, ax = plt.subplots(figsize=(4, 3))
    mean, ci = cost["cross_qs_10x50"]
    mask = np.isfinite(mean) & np.isfinite(ci)
    if np.any(mask):
        ax.plot(xs[mask], mean[mask], label=labels["cross_qs_10x50"], color=c, lw=2)
        ax.fill_between(
            xs[mask],
            mean[mask] - ci[mask],
            mean[mask] + ci[mask],
            color=c,
            alpha=0.2,
        )
    ax.set_xlabel("Expansions")
    ax.set_ylabel("Cost")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cost_vs_expansions.png", dpi=150)
    fig.savefig(OUT_DIR / "cost_vs_expansions.pdf")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'cost_vs_expansions.png'}")


def ensure_traj(si: int, gi: int, start: np.ndarray, goal: np.ndarray, planner) -> Optional[np.ndarray]:
    path = TRAJ_DIR / f"s{si:02d}_g{gi:02d}.npy"
    if path.is_file():
        return np.load(path)
    if already_at_goal(start, goal):
        # One-pose traj at start.
        pose = np.asarray(start, dtype=np.float32).reshape(1, -1)
        # path format [T, 11] with g in last col
        row = np.zeros((1, 11), dtype=np.float32)
        row[0, :10] = pose[0, :10]
        np.save(path, row)
        return row
    ok = planner.search(
        torch.as_tensor(start),
        torch.as_tensor(goal),
        torch.zeros(1),
        MAX_EXP,
        500,
        True,
    )
    if not ok:
        return None
    arr = planner.get_best_path().numpy()[::-1].copy().astype(np.float32)
    if arr.size == 0:
        return None
    np.save(path, arr)
    return arr


def plot_traj_html_for_start(
    si: int,
    rows_for_start: List[Dict[str, str]],
    problems_starts: np.ndarray,
    problems_goals: np.ndarray,
    out: Path,
) -> None:
    """Interactive HTML: 50 goals for one start, color by first_sol_exp rank."""
    order = sorted(
        rows_for_start,
        key=lambda r: float(r["first_sol_exp"]) if r["first_sol_exp"] not in ("",) else 1e9,
    )
    rank = {int(r["goal_idx"]): i for i, r in enumerate(order)}
    cmap = plt.cm.coolwarm
    norm = Normalize(0, max(len(order) - 1, 1))
    fig = go.Figure()
    for poly in t_polys_body_frame():
        fig.add_trace(go.Scatter(
            x=np.r_[poly[:, 0], poly[0, 0]],
            y=np.r_[poly[:, 1], poly[0, 1]],
            fill="toself",
            mode="lines",
            line=dict(color="#555", width=1.5),
            fillcolor="rgba(180,180,180,0.7)",
            name="start T",
            hoverinfo="skip",
            showlegend=False,
        ))
    n_base = len(fig.data)
    hover_map = {}
    start = problems_starts[si]
    for r in order:
        gi = int(r["goal_idx"])
        traj_p = TRAJ_DIR / f"s{si:02d}_g{gi:02d}.npy"
        if not traj_p.is_file():
            continue
        path = np.load(traj_p)
        ax, ay = agent_in_t_frame(path, start)
        bx, by, bang = block_in_t_frame(path, start)
        color = to_hex(cmap(norm(rank[gi])))
        exp = float(r["first_sol_exp"])
        tip = (
            f"s{si:02d}→g{gi:02d}<br>rank={rank[gi]}"
            f"<br>first_sol_exp={exp:.0f}<extra></extra>"
        )
        agent_i = len(fig.data)
        fig.add_trace(go.Scatter(
            x=ax, y=ay, mode="lines", line=dict(color=color, width=1.5),
            name=f"g{gi:02d}", hovertemplate=tip, legendgroup=f"g{gi:02d}",
        ))
        path_i = len(fig.data)
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="lines",
            line=dict(color=color, width=2, dash="dot"),
            hoverinfo="skip", showlegend=False, visible=False,
            legendgroup=f"g{gi:02d}",
        ))
        stride = max(len(bx) // 8, 1)
        px, py = (
            t_outline_xy(bx[:-1], by[:-1], bang[:-1], stride)
            if len(bx) > 1
            else ([], [])
        )
        poses_i = len(fig.data)
        fig.add_trace(go.Scatter(
            x=px, y=py, mode="lines", line=dict(color=color, width=1.2),
            hoverinfo="skip", showlegend=False, visible=False,
            legendgroup=f"g{gi:02d}",
        ))
        fx, fy = [], []
        for poly in place_t(float(bx[-1]), float(by[-1]), float(bang[-1])):
            if fx:
                fx.append(None)
                fy.append(None)
            fx.extend(np.r_[poly[:, 0], poly[0, 0]].tolist())
            fy.extend(np.r_[poly[:, 1], poly[0, 1]].tolist())
        final_i = len(fig.data)
        fig.add_trace(go.Scatter(
            x=fx, y=fy, mode="lines", line=dict(color="#000000", width=2.5),
            hoverinfo="skip", showlegend=False, visible=False,
            legendgroup=f"g{gi:02d}",
        ))
        sx, sy = agent_circle_xy(float(ax[0]), float(ay[0]))
        start_i = len(fig.data)
        fig.add_trace(go.Scatter(
            x=sx, y=sy, mode="lines",
            line=dict(color="#000000", width=2),
            fill="toself", fillcolor=hex_rgba(color, 0.3),
            hoverinfo="skip", showlegend=False, visible=False,
            legendgroup=f"g{gi:02d}",
        ))
        agx, agy = agent_circles_xy(ax, ay, stride)
        agent_poses_i = len(fig.data)
        fig.add_trace(go.Scatter(
            x=agx, y=agy, mode="lines", line=dict(color=color, width=1.5),
            hoverinfo="skip", showlegend=False, visible=False,
            legendgroup=f"g{gi:02d}",
        ))
        hover_map[agent_i] = {
            "path": path_i,
            "poses": poses_i,
            "final": final_i,
            "start": start_i,
            "agent": agent_poses_i,
        }

    fig.update_layout(
        title=f"QS cross start s{si:02d} → 50 goals (hover/legend → T path; black=final T)",
        xaxis_title="x in start-T frame (px)",
        yaxis_title="y in start-T frame (px)",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x"),
        width=800, height=800,
        hovermode="closest",
        legend=dict(itemclick=False, itemdoubleclick=False),
    )
    html = out / f"pusht_cross_qs_traj_s{si:02d}.html"
    fig.write_html(html, include_plotlyjs="cdn")
    hook = plotly_traj_hover_hook(hover_map, n_base)
    html.write_text(html.read_text().replace("</body>", hook + "</body>"))
    print(f"wrote {html}")


def make_htmls() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    dino = np.load(CONTENT / "pusht_dino_problems.npz")
    starts, goals = dino["starts"], dino["goals"]

    # Fill any missing trajs (oat diagonal / trivial).
    missing = [
        (int(r["start_idx"]), int(r["goal_idx"]))
        for r in rows
        if not (TRAJ_DIR / f"s{int(r['start_idx']):02d}_g{int(r['goal_idx']):02d}.npy").is_file()
    ]
    if missing:
        print(f"filling {len(missing)} missing trajs…")
        with open(CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        cfg["experiment_info_default"]["max_expansions"] = MAX_EXP
        planner = create_planner(cfg, bidirectional=False)
        for n, (si, gi) in enumerate(missing):
            if n > 0 and n % 10 == 0:
                planner = create_planner(cfg, bidirectional=False)
            st, go = make_pair(starts, goals, si, gi)
            ensure_traj(si, gi, st, go, planner)

    by_start: Dict[int, List[Dict[str, str]]] = {}
    for r in rows:
        by_start.setdefault(int(r["start_idx"]), []).append(r)
    for si in sorted(by_start):
        plot_traj_html_for_start(si, by_start[si], starts, goals, OUT_DIR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-expansions", type=int, default=MAX_EXP)
    ap.add_argument("--skip-collect", action="store_true")
    ap.add_argument("--skip-html", action="store_true")
    args = ap.parse_args()

    if not args.skip_collect:
        collect_pkls(args.workers, args.max_expansions)
    curves = load_all_curves(args.max_expansions)
    print(f"loaded {len(curves)} cost curves")
    plot_success_cost(curves, args.max_expansions)
    if not args.skip_html:
        make_htmls()


if __name__ == "__main__":
    main()
