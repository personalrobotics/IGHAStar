#!/usr/bin/env python3
"""Sample 2T1W2A_clear starts with split agents and keep those the planner solves.

Agent 0 stays in the top half (y >= 256, y is up) and agent 1 in the bottom half.
Agent centers stay at least 50 px apart. The wall is vertical with a centered gap
and both tees start on its left. Both limits are checked by the environment on
the start and along every edge (see Configs/pusht_multi_2T1W2A_split_clear.yml).

State [20]: a0x a0y a1x a1y | t0 x y a | t1 x y a | a0v a1v | t0v t1v.
World tensor: [n, then n segments of x1, y1, x2, y2]. The goal tensor is unused.

  python3 examples/standalone/pusht_multi_generate_2T1W2A_split_clear.py --n 10
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pusht_multi_generate_2T1O_clear import (  # noqa: E402
    BOX_HI,
    BOX_LO,
    CONTENT,
    GAP,
    WORLD_STRIDE,
    _tee_polys,
    cost_expansion,
    gap_wall,
    pack_walls,
    wall_segments,
)
from ighastar.scripts.common_utils import BASE_DIR, create_planner  # noqa: E402

CONFIG = os.path.join(
    BASE_DIR.parent, "examples", "standalone", "Configs",
    "pusht_multi_2T1W2A_split_clear.yml",
)
N_DIMS = 20
Y_SPLIT = 256.0
MIN_SEP = 50.0
AGENT_R = 15.0
TEE_MARGIN = 8.0  # clear of the wall (radius 2) and the box wall


def pack_state(a0, a1, t0, t1) -> np.ndarray:
    s = np.zeros(N_DIMS, dtype=np.float32)
    s[0:2], s[2:4], s[4:7], s[7:10] = a0, a1, t0, t1
    return s


def tee_corners(t) -> np.ndarray:
    return np.array([p for poly in _tee_polys(*map(float, t)) for p in poly])


def sample_tee(rng, wall_x):
    """Tee fully left of the vertical wall and inside the box."""
    for _ in range(200):
        t = np.array(
            [rng.uniform(60.0, wall_x), rng.uniform(60.0, 460.0), rng.uniform(0, 2 * np.pi)],
            dtype=np.float32,
        )
        c = tee_corners(t)
        if (
            c[:, 0].max() <= wall_x - TEE_MARGIN
            and c[:, 0].min() >= BOX_LO + TEE_MARGIN
            and c[:, 1].min() >= BOX_LO + TEE_MARGIN
            and c[:, 1].max() <= BOX_HI - TEE_MARGIN
        ):
            return t
    raise RuntimeError("no tee fits left of the wall")


def sample_candidate(rng):
    wall_x = float(rng.uniform(220.0, 380.0))
    mid = 0.5 * (BOX_LO + BOX_HI)
    world = pack_walls(gap_wall(1.0, wall_x, mid - 0.5 * GAP, mid + 0.5 * GAP))
    t0, t1 = sample_tee(rng, wall_x), sample_tee(rng, wall_x)
    a0 = (rng.uniform(40.0, 470.0), rng.uniform(Y_SPLIT + AGENT_R, 470.0))
    a1 = (rng.uniform(40.0, 470.0), rng.uniform(40.0, Y_SPLIT - AGENT_R))
    return pack_state(a0, a1, t0, t1), world


def self_check(planner) -> None:
    """Agent in the wrong half, agents too close, and a legal start."""
    world = torch.from_numpy(pack_walls(gap_wall(1.0, 400.0, 220.0, 380.0)))
    tees = ((100.0, 150.0, 0.0), (100.0, 320.0, 0.0))
    cases = {
        "ok": pack_state((60, 400), (60, 100), *tees),
        "agent0_low": pack_state((60, 200), (60, 100), *tees),
        "agent1_high": pack_state((60, 400), (60, 300), *tees),
        "too_close": pack_state((300, 270), (300, 240), *tees),
        "just_far": pack_state((300, 290), (300, 240), *tees),
    }
    mask = planner.check_validity_batched(
        torch.from_numpy(np.stack(list(cases.values()))), world
    )
    got = dict(zip(cases, mask.detach().cpu().numpy().astype(bool).tolist()))
    want = {"ok": True, "agent0_low": False, "agent1_high": False,
            "too_close": False, "just_far": True}
    if got != want:
        raise SystemExit(f"split self-check failed: {got}")
    print("split self-check passed")


def render_video(path_goal_first, world_row, out_mp4, fps=20) -> None:
    from PIL import Image, ImageDraw
    import imageio.v2 as imageio

    rows = np.asarray(path_goal_first, dtype=np.float32)[::-1]
    if rows.shape[0] > 300:
        rows = rows[np.linspace(0, rows.shape[0] - 1, 300).astype(np.int64)]
    W, H = 704, 512

    def pix(x, y):
        return (float(x), H - float(y))

    frames = []
    for row in rows:
        img = Image.new("RGB", (W, H), (255, 255, 255))
        dr = ImageDraw.Draw(img)
        dr.rectangle([pix(512, 512), pix(700, 0)], fill=(214, 240, 214))
        dr.line([pix(512, 0), pix(512, 512)], fill=(80, 160, 80), width=2)
        for x in range(5, 507, 16):  # dashed line between the agents' halves
            dr.line([pix(x, Y_SPLIT), pix(x + 8, Y_SPLIT)], fill=(190, 190, 190), width=1)
        for a, b in (((5, 5), (5, 506)), ((5, 5), (506, 5)), ((5, 506), (506, 506))):
            dr.line([pix(*a), pix(*b)], fill=(40, 40, 40), width=4)
        for a, b in wall_segments(world_row):
            dr.line([pix(*a), pix(*b)], fill=(90, 90, 90), width=6)
        for tee, color in ((row[4:7], (70, 130, 180)), (row[7:10], (210, 120, 40))):
            for poly in _tee_polys(*map(float, tee)):
                dr.polygon([pix(px, py) for px, py in poly], fill=color)
        for k, color in enumerate(((200, 40, 40), (130, 60, 170))):
            ax, ay = float(row[2 * k]), float(row[2 * k + 1])
            dr.ellipse(
                [pix(ax - AGENT_R, ay + AGENT_R), pix(ax + AGENT_R, ay - AGENT_R)],
                fill=color,
            )
        frames.append(np.asarray(img))
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    imageio.mimsave(out_mp4, frames, fps=fps)
    print(f"wrote {out_mp4} ({len(frames)} frames)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="solved problems to keep")
    ap.add_argument("--max-candidates", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(CONTENT, "pusht_multi_2T1W2A_split_clear.npz"))
    ap.add_argument("--video-dir", default=os.path.join(CONTENT, "pusht_multi_2T1W2A_split_clear_videos"))
    args = ap.parse_args()

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    info = cfg["experiment_info_default"]
    limit, hyst = int(info["max_expansions"]), int(info["hysteresis"])
    planner = create_planner(cfg, bidirectional=False)
    self_check(planner)

    rng = np.random.RandomState(args.seed)
    keep = {k: [] for k in ("starts", "worlds", "seeds", "paths", "cost_exp", "first")}
    goal = torch.zeros(N_DIMS, dtype=torch.float32)
    planned = draws = 0
    batch = 32
    while planned < args.max_candidates and len(keep["starts"]) < args.n:
        states, worlds, ids = [], [], []
        for _ in range(batch):
            s, w = sample_candidate(rng)
            draws += 1
            states.append(s), worlds.append(w), ids.append(draws)
        st, wt = torch.from_numpy(np.stack(states)), torch.from_numpy(np.stack(worlds))
        free = planner.check_validity_batched(st, wt).detach().cpu().numpy().astype(bool)
        for i in np.flatnonzero(free):
            planned += 1
            ok = bool(planner.search(st[i], goal, wt[i], limit, hyst, True))
            path = planner.get_best_path()
            first, cost_exp = cost_expansion(planner)
            ok = ok and path.numel() > 0
            if ok:
                p = path.detach().cpu().numpy()
                n = len(keep["starts"])
                for k, v in zip(keep, (states[i], worlds[i], ids[i], p, cost_exp, first)):
                    keep[k].append(v)
                render_video(p, worlds[i], os.path.join(args.video_dir, f"p{n:03d}_draw{ids[i]}.mp4"))
            print(f"candidate {planned} draw {ids[i]} solved={ok} first_sol={first} "
                  f"kept={len(keep['starts'])}/{args.n}", flush=True)
            if planned >= args.max_candidates or len(keep["starts"]) >= args.n:
                break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(
        args.out,
        starts=np.stack(keep["starts"]) if keep["starts"] else np.zeros((0, N_DIMS), np.float32),
        obstacles=np.stack(keep["worlds"]) if keep["worlds"] else np.zeros((0, WORLD_STRIDE), np.float32),
        seeds=np.asarray(keep["seeds"], dtype=np.int64),
        paths=np.array(keep["paths"], dtype=object),
        cost_exp=np.array(keep["cost_exp"], dtype=object),
        first_sol_expansions=np.asarray(keep["first"], dtype=np.int64),
        problem_class=np.array("2T1W2A_clear_split"),
        max_expansions=np.array(limit),
    )
    print(f"wrote {len(keep['starts'])} problems to {args.out}")


if __name__ == "__main__":
    main()
