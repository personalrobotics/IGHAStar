#!/usr/bin/env python3
"""Peg-in-slot Push-T problems: insert the stem of a tee into a pocket in a wall.

The pocket is built from wall segments against one edge of the box, so the box edge
is its back. The mouth wall runs across the box, DEPTH px from that edge, and has one
slot SLOT_W px wide. The stem is 30 px wide and 90 px long, so a tee seated with its bar
against the mouth wall has its stem inside the pocket. Slots are always near an edge, and
the open side keeps at least OPEN_MIN px free in front of the mouth wall.

State is the 16-dim 2-tee state: ax ay | t0 x y a | t1 x y a | velocities.
With --tees 1 the second tee is a bystander: its goal is its start pose.

  python3 -u examples/standalone/pusht_multi_generate_slot_rearrange.py --preview 8
  python3 -u examples/standalone/pusht_multi_generate_slot_rearrange.py --n 10 --tees 2
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml

from ighastar.scripts.common_utils import BASE_DIR, create_planner
from pusht_multi_generate_2T1O_clear import (
    BOX_HI,
    BOX_LO,
    CONTENT,
    N_DIMS,
    WORLD_STRIDE,
    _tee_polys,
    cost_expansion,
    pack_state,
    pack_walls,
    self_check,
    wall_segments,
)
from pusht_multi_generate_2T1O_rearrange import (
    TEE_RGB,
)

CONFIGS = {n: os.path.join(BASE_DIR.parent, "examples", "standalone", "Configs",
                           f"pusht_multi_slot_{'1T_' if n == 1 else ''}rearrange.yml")
           for n in (1, 2)}
SLOT_W = 44.0     # wall-center to wall-center. Clear width is 40 with radius-2 walls.
DEPTH = 105.0     # mouth wall to the box edge
BAR = 30.0        # tee bar thickness. Seated bar touches the mouth wall.
WALL_R = 2.0      # must equal the segment radius in pusht_multi_physics.h (add_segment(..., 2))
OPEN_MIN = 160.0  # free band in front of the mouth wall
SIDES = ("top", "bottom", "left", "right")
# stem direction as an angle: top is +y, left is -x, bottom is -y, right is +x
SIDE_ANGLE = {"top": 0.0, "left": np.pi / 2, "bottom": np.pi, "right": -np.pi / 2}


def to_world(side: str, u: float, v: float):
    """u along the edge, v from the box edge into the box."""
    if side == "top":
        return u, BOX_HI - v
    if side == "bottom":
        return u, BOX_LO + v
    if side == "left":
        return BOX_LO + v, u
    return BOX_HI - v, u


def slot_segments(side: str, centers):
    segs, lo, hi = [], BOX_LO, BOX_HI
    edges = sorted(centers)
    cuts = [lo]
    for c in edges:
        cuts += [c - SLOT_W / 2, c + SLOT_W / 2]
    cuts.append(hi)
    for k in range(0, len(cuts), 2):          # mouth wall pieces between slots
        segs.append((to_world(side, cuts[k], DEPTH), to_world(side, cuts[k + 1], DEPTH)))
    for c in edges:                            # two side walls per slot
        for u in (c - SLOT_W / 2, c + SLOT_W / 2):
            segs.append((to_world(side, u, DEPTH), to_world(side, u, 0.0)))
    return segs


def seated_pose(side: str, c: float):
    """Tee pose with the stem in the pocket and the bar against the mouth wall."""
    v = DEPTH + WALL_R + BAR + 0.5   # 0.5 px gap so float rounding cannot overlap; body origin is the outer face of the bar
    x, y = to_world(side, c, v)
    return np.array([x, y, SIDE_ANGLE[side]], dtype=np.float32)


def free_region(side: str):
    """Axis-aligned box in front of the mouth wall, shrunk by a margin."""
    m = 40.0
    lo, hi = BOX_LO + m, BOX_HI - m
    far = DEPTH + OPEN_MIN * 0.5
    if side == "top":
        return lo, hi, lo, BOX_HI - far
    if side == "bottom":
        return lo, hi, BOX_LO + far, hi
    if side == "left":
        return BOX_LO + far, hi, lo, hi
    return lo, BOX_HI - far, lo, hi


def sample_problem(rng, n_tees, near=0.0):
    side = SIDES[rng.randint(4)]
    margin = 70.0
    ids = [0]
    cs = [rng.uniform(BOX_LO + margin, BOX_HI - margin)]
    if n_tees == 2:
        while True:
            c2 = rng.uniform(BOX_LO + margin, BOX_HI - margin)
            if abs(c2 - cs[0]) >= 140.0:
                cs.append(c2)
                break
        ids = [0, 1]
    if rng.rand() < 0.5:
        cs = cs[::-1]
    segs = slot_segments(side, cs)
    x0, x1, y0, y1 = free_region(side)

    mouths = [np.array(to_world(side, c, DEPTH)) for c in cs]

    def tee():
        while True:
            xy = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
            # near > 0: the tee starts within `near` px of the (first) slot mouth
            if near <= 0 or np.hypot(*(xy - mouths[0])) <= near:
                break
        return np.array([xy[0], xy[1], rng.uniform(0, 2 * np.pi)], dtype=np.float32)

    agent = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
    t0, t1 = tee(), tee()
    if n_tees == 1:  # bystander in the corner of the free region farthest from the slot
        slot_xy = np.array(to_world(side, cs[0], DEPTH))
        corners = [(x, y) for x in (x0, x1) for y in (y0, y1)]
        cx, cy = max(corners, key=lambda c: np.hypot(c[0] - slot_xy[0], c[1] - slot_xy[1]))
        # a tee reaches 120 px from its origin, so keep the origin 130 px from the box edges
        cx = cx + 90.0 if cx == x0 else cx - 90.0 if cx == x1 else cx
        cy = cy + 90.0 if cy == y0 else cy - 90.0 if cy == y1 else cy
        t1 = np.array([cx, cy, rng.uniform(0, 2 * np.pi)], dtype=np.float32)
    start = pack_state(agent[0], agent[1], t0, t1)
    if n_tees == 2:
        g0, g1 = seated_pose(side, cs[0]), seated_pose(side, cs[1])
    else:
        g0, g1 = seated_pose(side, cs[0]), t1.copy()
    goal = pack_state(agent[0], agent[1], g0, g1)
    return start, goal, pack_walls(segs), dict(side=side, centers=cs, tees=n_tees)


def render_slot_video(path_goal_first, goal, walls, out_mp4, n_tees, fps=20):
    """Like the rearrange video, but the bystander tee is not drawn when n_tees is 1."""
    from PIL import Image, ImageDraw
    import imageio.v2 as imageio
    rows = np.asarray(path_goal_first, dtype=np.float32)[::-1]
    if rows.shape[0] > 300:
        rows = rows[np.linspace(0, rows.shape[0] - 1, 300).astype(np.int64)]
    H = 512
    pix = lambda x, y: (float(x), H - float(y))  # noqa: E731
    tees = ((2, TEE_RGB[0]), (5, TEE_RGB[1]))[:n_tees]
    frames = []
    for row in rows:
        img = Image.new("RGB", (512, 512), (255, 255, 255))
        dr = ImageDraw.Draw(img)
        box = ((5, 5), (5, 506), (506, 506), (506, 5), (5, 5))
        dr.line([pix(*q) for q in box], fill=(40, 40, 40), width=4)
        for a, b in wall_segments(walls):
            dr.line([pix(*a), pix(*b)], fill=(90, 90, 90), width=6)
        for b, rgb in tees:
            col = tuple(int(c) for c in rgb)
            for poly in _tee_polys(*map(float, goal[b:b + 3])):
                pts = [pix(*q) for q in poly]
                dr.line(pts + [pts[0]], fill=col, width=2)
            for poly in _tee_polys(*map(float, row[b:b + 3])):
                dr.polygon([pix(*q) for q in poly], fill=col)
        ax, ay = float(row[0]), float(row[1])
        dr.ellipse([pix(ax - 15, ay + 15), pix(ax + 15, ay - 15)], fill=(200, 40, 40))
        frames.append(np.asarray(img))
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    imageio.mimsave(out_mp4, frames, fps=fps)
    print(f"wrote {out_mp4} ({len(frames)} frames)", flush=True)


def render_preview(problems, out_png, flags):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle, Rectangle
    n = len(problems)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.8 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, ((s, g, w, meta), ok) in enumerate(zip(problems, flags)):
        ax = axes.ravel()[k]
        ax.axis("on")
        ax.set_xlim(0, 512); ax.set_ylim(0, 512); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.add_patch(Rectangle((BOX_LO, BOX_LO), BOX_HI - BOX_LO, BOX_HI - BOX_LO, fc="#f3f4f2", ec="k", lw=1.5))
        for (a, b) in wall_segments(w):
            ax.plot([a[0], b[0]], [a[1], b[1]], c="#444", lw=3, solid_capstyle="round")
        for t, gt, col in ((s[2:5], g[2:5], "#3f78b8"), (s[5:8], g[5:8], "#d9822b")):
            if meta.get("tees") == 1 and col == "#d9822b":
                continue
            for p in _tee_polys(*map(float, gt)):
                ax.add_patch(Polygon(p, fc="none", ec="#2a9d4a", ls="--", lw=1.5))
            for p in _tee_polys(*map(float, t)):
                ax.add_patch(Polygon(p, fc=col, ec="k", lw=0.8, alpha=0.9))
        ax.add_patch(Circle((s[0], s[1]), 15, fc="#d9534f", ec="k"))
        ax.set_title(f"{meta['side']} slot, valid={ok}", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tees", type=int, default=1, choices=(1, 2))
    ap.add_argument("--max-candidates", type=int, default=200)
    ap.add_argument("--max-expansions", type=int, default=0, help="override the config limit")
    ap.add_argument("--near", type=float, default=0.0, help="tee starts within this many px of the first slot mouth")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--merge", nargs="*", help="merge shard npz files into --out, then render videos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", type=int, default=0, help="draw this many samples and stop")
    ap.add_argument("--preview-png", default=os.path.join(CONTENT, "pusht_multi_slot_preview.png"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--video-dir", default=None)
    args = ap.parse_args()
    if args.near and args.near < DEPTH + OPEN_MIN / 2 - DEPTH + 1:
        ap.error(f"--near below {OPEN_MIN / 2 + 1:.0f} px has no valid tee positions")
    if args.merge:  # tee count comes from the shards, not from --tees
        args.tees = 1 if str(np.load(args.merge[0], allow_pickle=True)["problem_class"]).startswith("slot_1T") else 2
    tag = f"slot_{args.tees}T_rearrange"
    out = args.out or os.path.join(CONTENT, f"pusht_multi_{tag}.npz")
    vdir = args.video_dir or os.path.join(CONTENT, f"pusht_multi_{tag}_videos")

    if args.merge:
        shards = [np.load(f, allow_pickle=True) for f in args.merge]
        keys = ["starts", "goals", "obstacles", "first_sol_expansions", "slot_side"]
        merged = {k: np.concatenate([d[k] for d in shards]) for k in keys}
        merged["paths"] = np.array([x for d in shards for x in d["paths"]], dtype=object)
        merged["cost_exp"] = np.array([x for d in shards for x in d["cost_exp"]], dtype=object)
        merged["slot_centers"] = np.array([x for d in shards for x in d["slot_centers"]], dtype=object)
        merged["problem_class"] = shards[0]["problem_class"]
        merged["tee_rgb"] = shards[0]["tee_rgb"]
        merged["max_expansions"] = shards[0]["max_expansions"]
        np.savez(out, **merged)
        print(f"merged {len(merged['starts'])} problems into {out}", flush=True)
        if not args.no_video:
            for i, pn in enumerate(merged["paths"]):
                render_slot_video(pn, merged["goals"][i], merged["obstacles"][i],
                                  os.path.join(vdir, f"p{i:03d}.mp4"), args.tees)
        return

    with open(CONFIGS[args.tees]) as f:
        cfg = yaml.safe_load(f)
    info = cfg["experiment_info_default"]
    limit = args.max_expansions or int(info["max_expansions"])
    hyst = int(info["hysteresis"])
    planner = create_planner(cfg, bidirectional=False)
    self_check(planner, check_exit=False)
    rng = np.random.RandomState(args.seed)

    def valid(ps):
        st = torch.from_numpy(np.stack([p[0] for p in ps] + [p[1] for p in ps]))
        wd = torch.from_numpy(np.stack([p[2] for p in ps] * 2))
        r = planner.check_validity_batched(st, wd).detach().cpu().numpy().astype(bool)
        return r[: len(ps)], r[len(ps):]

    if args.preview:
        ps = [sample_problem(rng, args.tees, args.near) for _ in range(args.preview)]
        vs, vg = valid(ps)
        render_preview(ps, args.preview_png, [bool(a and b) for a, b in zip(vs, vg)])
        print("start valid", vs.tolist(), "goal valid", vg.tolist())
        print("wrote", args.preview_png)
        return

    starts, goals, obstacles, paths, cost_exps, first_sols, metas = [], [], [], [], [], [], []

    def write():
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out, starts=np.stack(starts), goals=np.stack(goals),
                 obstacles=np.stack(obstacles), paths=np.array(paths, dtype=object),
                 cost_exp=np.array(cost_exps, dtype=object),
                 first_sol_expansions=np.asarray(first_sols, dtype=np.int64),
                 slot_side=np.array([m["side"] for m in metas]),
                 slot_centers=np.array([m["centers"] for m in metas], dtype=object),
                 problem_class=np.array(tag), tee_rgb=TEE_RGB,
                 max_expansions=np.array(limit))

    tried = draws = 0
    # max-candidates counts planner runs. Sampling is also capped so a bad filter cannot spin.
    while tried < args.max_candidates and draws < 50 * args.max_candidates and len(starts) < args.n:
        draws += 32
        ps = [sample_problem(rng, args.tees, args.near) for _ in range(32)]
        vs, vg = valid(ps)
        for i, (s, g, w, meta) in enumerate(ps):
            if not (vs[i] and vg[i]):
                continue
            tried += 1
            ok = bool(planner.search(torch.from_numpy(s), torch.from_numpy(g),
                                     torch.from_numpy(w), limit, hyst, False))
            path = planner.get_best_path()
            first, ce = cost_expansion(planner)
            if ok and path.numel() == 0:
                ok = False
            if ok:
                pn = path.detach().cpu().numpy()
                starts.append(s); goals.append(g); obstacles.append(w)
                paths.append(pn); cost_exps.append(ce); first_sols.append(first); metas.append(meta)
                write()
                if not args.no_video:
                    render_slot_video(pn, g, w, os.path.join(vdir, f"p{len(starts) - 1:03d}.mp4"), args.tees)
            print(f"candidate {tried} side={meta['side']} solved={ok} first={first} kept={len(starts)}/{args.n}",
                  flush=True)
            if tried >= args.max_candidates or len(starts) >= args.n:
                break
    if starts:
        write()
    print(f"wrote {len(starts)} problems to {out}")


if __name__ == "__main__":
    main()
