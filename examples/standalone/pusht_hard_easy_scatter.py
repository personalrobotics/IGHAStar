#!/usr/bin/env python3
"""Best/worst half of dino problems by first_sol_exp → scatters + relative traj plots.

Paths from get_best_path() after a full budget = best (lowest-cost) path, not the
first solution. Color rank still uses first_sol_exp from the oat CSV.

  python pusht_hard_easy_scatter.py
"""
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_hex

from ighastar.scripts.common_utils import create_planner

REPO = Path(__file__).resolve().parents[2]
CONTENT = REPO / "Content" / "standalone"
# Crossbar length of the T (kTeeLength * kTeeScale in pusht_qs.h).
BAR_LEN = 4.0 * 30.0
# gym-pusht / pusht_physics.h kAgentRadius (px).
AGENT_RADIUS = 15.0
K = 25
MAX_EXP = 50000
BASE = dict(res_xy="40.0", action_radii="200", n_directions="24", hysteresis="500",
            n_substeps="10", qs_support_radius="60.0")
CFG = {
    "chipmunk": REPO / "examples" / "standalone" / "Configs" / "pusht_ndir24.yml",
    "quasistatic": REPO / "examples" / "standalone" / "Configs" / "pusht_qs.yml",
}


def load_rows(csv_path, dynamics):
    with open(csv_path) as f:
        rows = [
            r for r in csv.DictReader(f)
            if r["dynamics"] == dynamics and all(r[k] == v for k, v in BASE.items())
        ]
    rows.sort(key=lambda r: int(r["problem_idx"]))
    assert len(rows) == 50, (dynamics, len(rows))
    return rows


def screw(start, goal, R=BAR_LEN):
    dx, dy = float(goal[2] - start[2]), float(goal[3] - start[3])
    dth = math.atan2(math.sin(goal[4] - start[4]), math.cos(goal[4] - start[4]))
    return math.sqrt(dx * dx + dy * dy + (R * dth) ** 2)


def pusher_dist(start):
    return math.hypot(float(start[0] - start[2]), float(start[1] - start[3]))


def t_xy(start, goal):
    return math.hypot(float(goal[2] - start[2]), float(goal[3] - start[3]))


def extremes(rows):
    order = sorted(range(len(rows)), key=lambda i: float(rows[i]["first_sol_exp"]))
    return order[:K], order[-K:]


def scatter(ax, xs, ys, best, worst, xlabel):
    ax.scatter([xs[i] for i in best], [ys[i] for i in best], c="C0", label=f"best {K}", s=36)
    ax.scatter([xs[i] for i in worst], [ys[i] for i in worst], c="C3", label=f"worst {K}", s=36)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("first_sol_exp")
    ax.legend()


def to_t0(x, y, start):
    c, s = math.cos(float(start[4])), math.sin(float(start[4]))
    dx, dy = x - float(start[2]), y - float(start[3])
    return c * dx + s * dy, -s * dx + c * dy


def agent_in_t_frame(path, start):
    # Agent xy in start-T body frame (origin = start block pose, not CoG).
    return to_t0(path[:, 0], path[:, 1], start)


def block_in_t_frame(path, start):
    # Block body-origin xy + relative angle in the start-T body frame.
    # Planner pose[2:4] is body origin (QS / Chipmunk get_state), not CoG.
    bx, by = to_t0(path[:, 2], path[:, 3], start)
    bang = path[:, 4] - float(start[4])
    return bx, by, bang


def t_polys_body_frame():
    # Same local T as pusht_qs.h tee_rects (body origin, bar y∈[0,s]).
    s, L = 30.0, 4.0
    bar = np.array([[-0.5 * L * s, 0], [0.5 * L * s, 0], [0.5 * L * s, s], [-0.5 * L * s, s]])
    stem = np.array([[-0.5 * s, s], [0.5 * s, s], [0.5 * s, L * s], [-0.5 * s, L * s]])
    return bar, stem


def t_polys_cog_frame():
    # Body-local polys shifted so CoG is at the origin (analysis only).
    s, L = 30.0, 4.0
    cog = 0.5 * (np.array([0.0, 0.5 * s]) + np.array([0.0, 0.5 * (s + L * s)]))
    return tuple(p - cog for p in t_polys_body_frame())


def place_t(x, y, ang):
    # Body-local T polys → frame at block body origin (x, y, ang).
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return [poly @ R.T + np.array([x, y]) for poly in t_polys_body_frame()]


def goal_in_t0(start, goal):
    x, y = to_t0(float(goal[2]), float(goal[3]), start)
    return x, y, float(goal[4] - start[4])


def t_outline_xy(bx, by, bang, stride):
    # Subsampled T outlines + always include the last pose. NaN-separated.
    xs, ys = [], []
    n = len(bx)
    idxs = list(range(0, n, max(stride, 1)))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    for k in idxs:
        for poly in place_t(float(bx[k]), float(by[k]), float(bang[k])):
            if xs:
                xs.append(None)
                ys.append(None)
            xs.extend(np.r_[poly[:, 0], poly[0, 0]].tolist())
            ys.extend(np.r_[poly[:, 1], poly[0, 1]].tolist())
    return xs, ys


def hex_rgba(hex_color, alpha=0.25):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def agent_circle_xy(cx, cy, r=AGENT_RADIUS, n=48):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
    return (cx + r * np.cos(t)).tolist(), (cy + r * np.sin(t)).tolist()


def agent_circles_xy(ax, ay, stride, r=AGENT_RADIUS):
    # Subsampled agent disks (true radius in px) + always include last. NaN-separated.
    xs, ys = [], []
    n = len(ax)
    if n == 0:
        return xs, ys
    idxs = list(range(0, n, max(stride, 1)))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    for k in idxs:
        cx, cy = agent_circle_xy(float(ax[k]), float(ay[k]), r)
        if xs:
            xs.append(None)
            ys.append(None)
        xs.extend(cx)
        ys.extend(cy)
    return xs, ys


def load_or_plan_paths(dynamics, starts, goals):
    cache = CONTENT / f"pusht_hard_easy_traj_{dynamics}"
    cache.mkdir(parents=True, exist_ok=True)
    paths = [None] * 50
    missing = [i for i in range(50) if not (cache / f"p{i:03d}.npy").is_file()]
    for i in range(50):
        p = cache / f"p{i:03d}.npy"
        if p.is_file():
            paths[i] = np.load(p)
    if not missing:
        return paths
    with open(CFG[dynamics]) as f:
        configs = yaml.safe_load(f)
    configs["experiment_info_default"]["max_expansions"] = MAX_EXP
    info = configs["experiment_info_default"]
    world = torch.zeros(1)
    planner = None
    for n, i in enumerate(missing):
        # ponytail: recreate every 10 problems — long QS runs can overflow robin_hood.
        if planner is None or n % 10 == 0:
            planner = create_planner(configs, bidirectional=False)
        ok = planner.search(
            torch.tensor(starts[i]), torch.tensor(goals[i]), world,
            info["max_expansions"], info["hysteresis"], True,
        )
        if not ok:
            print(f"  {dynamics} p{i:03d} failed")
            continue
        path = np.flip(planner.get_best_path().numpy(), 0)
        np.save(cache / f"p{i:03d}.npy", path.astype(np.float32))
        paths[i] = path
        print(f"  {dynamics} p{i:03d} path len={len(path)}")
    return paths


def plot_trajs(dynamics, rows, starts, goals, paths, out):
    # Rank 0 = easiest (blue), 49 = hardest (red). Paths are best-cost after budget.
    order = sorted(range(50), key=lambda i: float(rows[i]["first_sol_exp"]))
    rank = {i: r for r, i in enumerate(order)}
    cmap = plt.cm.coolwarm
    norm = Normalize(0, 49)
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    for poly in t_polys_body_frame():
        ax.fill(poly[:, 0], poly[:, 1], color="0.75", zorder=0)
        ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]], color="0.35", lw=1.2, zorder=1)
    for i in order:
        if paths[i] is None:
            continue
        x, y = agent_in_t_frame(paths[i], starts[i])
        ax.plot(x, y, color=cmap(norm(rank[i])), lw=1.0, alpha=0.85, zorder=2)
        gx, gy, ga = goal_in_t0(starts[i], goals[i])
        for poly in place_t(gx, gy, ga):
            ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]],
                    color=cmap(norm(rank[i])), lw=0.8, alpha=0.35, zorder=1)
    ax.plot(0, 0, "k+", ms=10, mew=2, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel("agent x in start-T frame (px)")
    ax.set_ylabel("agent y in start-T frame (px)")
    ax.set_title(f"{dynamics}: best paths (color=first_sol_exp rank)")
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="rank (blue=best, red=worst)")
    fig.savefig(out / f"pusht_hard_easy_traj_{dynamics}.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out / f'pusht_hard_easy_traj_{dynamics}.png'}")


def plotly_traj_hover_hook(hover_map, n_base: int) -> str:
    """Plotly post-load JS: hover/legend pin dims others and reveals T/agent overlays."""
    return f"""
<script>
(function(){{
  var gd = document.querySelector('.plotly-graph-div');
  if (!gd) return;
  var map = {json.dumps({str(k): v for k, v in hover_map.items()})};
  var nBase = {n_base};
  var DIM = 0.12;
  var pinned = null;
  function extras(m){{
    return m ? [m.path, m.poses, m.final, m.start, m.agent] : [];
  }}
  function isFocus(k, i, m){{
    k = Number(k); i = Number(i);
    return k === i || extras(m).indexOf(k) >= 0;
  }}
  function visFor(i){{
    var m = i == null ? null : map[i];
    var show = m ? extras(m) : [];
    return gd.data.map((_,k)=> {{
      if (k < nBase) return true;
      if (map[k] !== undefined) return true;
      return show.indexOf(k) >= 0;
    }});
  }}
  function apply(i){{
    if (i == null) {{
      Plotly.restyle(gd, {{
        'visible': visFor(null),
        'line.width': gd.data.map(()=> 1.5),
        'opacity': gd.data.map(()=> 1),
      }});
      return;
    }}
    var m = map[i];
    var w = gd.data.map((_,k)=> Number(k) === Number(i) ? 5 : (isFocus(k, i, m) ? 2.5 : 1.5));
    var op = gd.data.map((_,k)=> isFocus(k, i, m) ? 1 : DIM);
    Plotly.restyle(gd, {{'visible': visFor(i), 'line.width': w, 'opacity': op}});
  }}
  gd.on('plotly_hover', function(ev){{
    if (pinned !== null) return;
    var i = String(ev.points[0].curveNumber);
    if (!(i in map)) return;
    apply(i);
  }});
  gd.on('plotly_unhover', function(){{
    if (pinned !== null) return;
    apply(null);
  }});
  gd.on('plotly_legendclick', function(ev){{
    var i = String(ev.curveNumber);
    if (!(i in map)) return false;
    pinned = (pinned === i) ? null : i;
    apply(pinned);
    return false;
  }});
}})();
</script>
"""


def plot_trajs_interactive(dynamics, rows, starts, goals, paths, out):
    # Agent trajs always on. On hover: T path + poses + agent disks; final T black.
    order = sorted(range(50), key=lambda i: float(rows[i]["first_sol_exp"]))
    rank = {i: r for r, i in enumerate(order)}
    cmap = plt.cm.coolwarm
    norm = Normalize(0, 49)
    fig = go.Figure()
    for poly in t_polys_body_frame():
        fig.add_trace(go.Scatter(
            x=np.r_[poly[:, 0], poly[0, 0]], y=np.r_[poly[:, 1], poly[0, 1]],
            fill="toself", mode="lines",
            line=dict(color="#555", width=1.5), fillcolor="rgba(180,180,180,0.7)",
            name="start T", hoverinfo="skip", showlegend=False,
        ))
    n_base = len(fig.data)
    # agent_curve -> highlight overlay curve indices.
    hover_map = {}
    for i in order:
        if paths[i] is None:
            continue
        ax, ay = agent_in_t_frame(paths[i], starts[i])
        bx, by, bang = block_in_t_frame(paths[i], starts[i])
        color = to_hex(cmap(norm(rank[i])))
        exp = float(rows[i]["first_sol_exp"])
        tip = f"p{i:03d}<br>rank={rank[i]}<br>first_sol_exp={exp:.0f}<br>best path (after budget)<extra></extra>"
        agent_i = len(fig.data)
        fig.add_trace(go.Scatter(
            x=ax, y=ay, mode="lines", line=dict(color=color, width=1.5),
            name=f"p{i:03d}", hovertemplate=tip, legendgroup=f"p{i:03d}",
        ))
        path_i = len(fig.data)
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="lines", line=dict(color=color, width=2, dash="dot"),
            name=f"p{i:03d} T path", hoverinfo="skip", showlegend=False,
            visible=False, legendgroup=f"p{i:03d}",
        ))
        # Intermediate T poses (skip last — drawn in black separately).
        stride = max(len(bx) // 8, 1)
        px, py = t_outline_xy(bx[:-1], by[:-1], bang[:-1], stride) if len(bx) > 1 else ([], [])
        poses_i = len(fig.data)
        fig.add_trace(go.Scatter(
            x=px, y=py, mode="lines", line=dict(color=color, width=1.2),
            name=f"p{i:03d} T poses", hoverinfo="skip", showlegend=False,
            visible=False, legendgroup=f"p{i:03d}",
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
            name=f"p{i:03d} T final", hoverinfo="skip", showlegend=False,
            visible=False, legendgroup=f"p{i:03d}",
        ))
        # Agent disks at true kAgentRadius=15px (start + subsampled along path).
        sx, sy = agent_circle_xy(float(ax[0]), float(ay[0]))
        start_i = len(fig.data)
        fig.add_trace(go.Scatter(
            x=sx, y=sy, mode="lines",
            line=dict(color="#000000", width=2),
            fill="toself", fillcolor=hex_rgba(color, 0.3),
            name=f"p{i:03d} agent start", hoverinfo="skip", showlegend=False,
            visible=False, legendgroup=f"p{i:03d}",
        ))
        agx, agy = agent_circles_xy(ax, ay, stride)
        agent_poses_i = len(fig.data)
        fig.add_trace(go.Scatter(
            x=agx, y=agy, mode="lines", line=dict(color=color, width=1.5),
            name=f"p{i:03d} agent", hoverinfo="skip", showlegend=False,
            visible=False, legendgroup=f"p{i:03d}",
        ))
        hover_map[agent_i] = {
            "path": path_i,
            "poses": poses_i,
            "final": final_i,
            "start": start_i,
            "agent": agent_poses_i,
        }

    fig.update_layout(
        title=f"{dynamics}: best agent paths (hover/legend → T path + poses; black = final T)",
        xaxis_title="x in start-T frame (px)",
        yaxis_title="y in start-T frame (px)",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x"),
        width=800, height=800,
        hovermode="closest",
        legend=dict(itemclick=False, itemdoubleclick=False),
    )
    html = out / f"pusht_hard_easy_traj_{dynamics}.html"
    fig.write_html(html, include_plotlyjs="cdn")
    hook = plotly_traj_hover_hook(hover_map, n_base)
    html.write_text(html.read_text().replace("</body>", hook + "</body>"))
    print(f"wrote {html}")


def main():
    problems = np.load(CONTENT / "pusht_dino_problems.npz")
    starts, goals = problems["starts"], problems["goals"]
    screws = [screw(starts[i], goals[i]) for i in range(50)]
    pushers = [pusher_dist(starts[i]) for i in range(50)]
    t_xys = [t_xy(starts[i], goals[i]) for i in range(50)]
    screw_plus = [screws[i] + pushers[i] for i in range(50)]
    txy_plus = [t_xys[i] + pushers[i] for i in range(50)]

    runs = [
        ("chipmunk", CONTENT / "pusht_sweep_chipmunk_oat_50k_results.csv"),
        ("quasistatic", CONTENT / "pusht_sweep_qs_oat_50k_results.csv"),
    ]
    out = CONTENT / "curve_plots"
    out.mkdir(parents=True, exist_ok=True)

    for dyn, path in runs:
        rows = load_rows(path, dyn)
        ys = [float(r["first_sol_exp"]) for r in rows]
        best, worst = extremes(rows)
        fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
        scatter(axes[0, 0], screws, ys, best, worst, f"T screw (R={BAR_LEN:g} px)")
        scatter(axes[0, 1], pushers, ys, best, worst, "pusher–T xy (start)")
        scatter(axes[1, 0], screw_plus, ys, best, worst, "T screw + pusher–T xy")
        scatter(axes[1, 1], txy_plus, ys, best, worst, "T xy + pusher–T xy")
        fig.suptitle(f"{dyn}: best/worst {K} by first_sol_exp")
        fig.savefig(out / f"pusht_hard_easy_{dyn}.png", dpi=150)
        plt.close(fig)
        print(f"wrote {out / f'pusht_hard_easy_{dyn}.png'}")

        print(f"planning/caching {dyn} paths…")
        paths = load_or_plan_paths(dyn, starts, goals)
        plot_trajs(dyn, rows, starts, goals, paths, out)
        plot_trajs_interactive(dyn, rows, starts, goals, paths, out)


if __name__ == "__main__":
    main()
