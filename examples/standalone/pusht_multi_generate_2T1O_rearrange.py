#!/usr/bin/env python3
"""Sample 2T1W_rearrange problems and keep those the planner solves.

Each tee has a goal pose. The maze is one wall with a centered gap.
The state stores tee 0 then tee 1. Color is not in the state. Videos paint
tee 0 blue and tee 1 orange, draw the goal pose, and draw a few poses inside
the goal tolerance in a lighter outline of that same color.

  python examples/standalone/pusht_multi_generate_2T1O_rearrange.py --n 10
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml

from ighastar.scripts.common_utils import BASE_DIR, create_planner
from pusht_multi_generate_2T1O_clear import (
    CONTENT,
    N_DIMS,
    WORLD_STRIDE,
    _tee_polys,
    cost_expansion,
    pack_state,
    sample_maze,
    self_check,
    wall_segments,
)

CONFIG = os.path.join(
    BASE_DIR.parent,
    "examples",
    "standalone",
    "Configs",
    "pusht_multi_2T1O_rearrange.yml",
)
# Index is the tee id. The planner never reads these values.
TEE_RGB = np.array([[70, 130, 180], [210, 120, 40]], dtype=np.uint8)
EPS_XY = 20.0
EPS_TH = 0.349
MIN_GOAL_DIST = 100.0


def load_config() -> dict:
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def ang_delta(a: float, b: float) -> float:
    return abs((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def poses_separated(start: np.ndarray, goal: np.ndarray) -> bool:
    """Each tee starts at least MIN_GOAL_DIST from its own goal, and not already in the epsilon box."""
    for b in (2, 5):
        dist = float(np.hypot(start[b] - goal[b], start[b + 1] - goal[b + 1]))
        if dist < MIN_GOAL_DIST:
            return False
        if (
            abs(start[b] - goal[b]) < EPS_XY
            and abs(start[b + 1] - goal[b + 1]) < EPS_XY
            and ang_delta(float(start[b + 2]), float(goal[b + 2])) < EPS_TH
        ):
            return False
    return True


def sample_tee(rng: np.random.RandomState) -> np.ndarray:
    xy = rng.uniform(90.0, 400.0, size=2)
    ang = rng.uniform(0.0, 2.0 * np.pi)
    return np.array([xy[0], xy[1], ang], dtype=np.float32)


def sample_problem(rng: np.random.RandomState):
    agent_s = rng.uniform(40.0, 470.0, size=2)
    agent_g = rng.uniform(40.0, 470.0, size=2)
    start = pack_state(agent_s[0], agent_s[1], sample_tee(rng), sample_tee(rng))
    goal = pack_state(agent_g[0], agent_g[1], sample_tee(rng), sample_tee(rng))
    return start, goal, sample_maze(rng)


def goal_set_poses(tee):
    """Four poses on the inside of the goal box: |dx|,|dy| < 20 px and |dth| < 0.349."""
    x, y, th = (float(v) for v in tee)
    xy = 0.85 * EPS_XY
    ath = 0.85 * EPS_TH
    signs = ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
    return [(x + sx * xy, y + sy * xy, th + st * ath) for sx, sy, st in signs]


def _tint(rgb, toward=255, amount=0.55):
    return tuple(int(c + (toward - c) * amount) for c in rgb)


def render_rearrange_video(path_goal_first, goal, obstacle_xy, out_mp4, fps=20):
    from PIL import Image, ImageDraw
    import imageio.v2 as imageio

    rows = np.asarray(path_goal_first, dtype=np.float32)[::-1]
    if rows.shape[0] > 300:
        idx = np.linspace(0, rows.shape[0] - 1, 300).astype(np.int64)
        rows = rows[idx]
    W = H = 512
    obs = np.asarray(obstacle_xy, dtype=np.float32).reshape(-1)

    def pix(x, y):
        return (float(x), H - float(y))

    frames = []
    for row in rows:
        img = Image.new("RGB", (W, H), (255, 255, 255))
        dr = ImageDraw.Draw(img)
        for a, b in (
            ((5, 5), (5, 506)),
            ((5, 5), (506, 5)),
            ((506, 5), (506, 506)),
            ((5, 506), (506, 506)),
        ):
            dr.line([pix(*a), pix(*b)], fill=(40, 40, 40), width=4)
        if obs.size == 2:
            ox, oy, half = float(obs[0]), float(obs[1]), 15.0
            dr.rectangle(
                [pix(ox - half, oy + half), pix(ox + half, oy - half)],
                fill=(90, 90, 90),
            )
        else:
            for a, b in wall_segments(obs):
                dr.line([pix(*a), pix(*b)], fill=(90, 90, 90), width=6)
        for tee, rgb in ((goal[2:5], TEE_RGB[0]), (goal[5:8], TEE_RGB[1])):
            color = tuple(int(c) for c in rgb)
            pale = _tint(color)
            for sample in goal_set_poses(tee):
                for poly in _tee_polys(*sample):
                    pts = [pix(px, py) for px, py in poly]
                    dr.line(pts + [pts[0]], fill=pale, width=1)
            for poly in _tee_polys(float(tee[0]), float(tee[1]), float(tee[2])):
                pts = [pix(px, py) for px, py in poly]
                dr.line(pts + [pts[0]], fill=color, width=2)
        for tee, rgb in ((row[2:5], TEE_RGB[0]), (row[5:8], TEE_RGB[1])):
            color = tuple(int(c) for c in rgb)
            for poly in _tee_polys(float(tee[0]), float(tee[1]), float(tee[2])):
                dr.polygon([pix(px, py) for px, py in poly], fill=color)
        ax, ay, r = float(row[0]), float(row[1]), 15.0
        dr.ellipse([pix(ax - r, ay + r), pix(ax + r, ay - r)], fill=(200, 40, 40))
        frames.append(np.asarray(img))
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    imageio.mimsave(out_mp4, frames, fps=fps)
    print(f"wrote {out_mp4} ({len(frames)} frames)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default=os.path.join(CONTENT, "pusht_multi_2T1W_rearrange.npz"),
    )
    parser.add_argument(
        "--video-dir",
        default=os.path.join(CONTENT, "pusht_multi_2T1W_rearrange_videos"),
    )
    args = parser.parse_args()

    cfg = load_config()
    info = cfg["experiment_info_default"]
    limit = int(info["max_expansions"])
    hyst = int(info["hysteresis"])
    planner = create_planner(cfg, bidirectional=False)
    self_check(planner, check_exit=False)

    def write_set() -> None:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez(
            args.out,
            starts=np.stack(starts) if starts else np.zeros((0, N_DIMS), np.float32),
            goals=np.stack(goals) if goals else np.zeros((0, N_DIMS), np.float32),
            obstacles=np.stack(obstacles) if obstacles else np.zeros((0, WORLD_STRIDE), np.float32),
            seeds=np.asarray(seeds, dtype=np.int64),
            paths=np.array(paths, dtype=object),
            cost_exp=np.array(cost_exps, dtype=object),
            first_sol_expansions=np.asarray(first_sols, dtype=np.int64),
            problem_class=np.array("2T1W_rearrange"),
            tee_id=np.array([0, 1], dtype=np.int64),
            tee_rgb=TEE_RGB,
            max_expansions=np.array(limit),
        )

    rng = np.random.RandomState(args.seed)
    starts, goals, obstacles, seeds, paths = [], [], [], [], []
    cost_exps, first_sols = [], []
    planned = draws = 0
    batch = 32
    while planned < args.max_candidates and len(starts) < args.n:
        ss, gs, ws, ids = [], [], [], []
        for _ in range(batch):
            s, g, o = sample_problem(rng)
            draws += 1
            ss.append(s)
            gs.append(g)
            ws.append(o)
            ids.append(draws)
        state_t = torch.from_numpy(np.stack(ss + gs))
        world_t = torch.from_numpy(np.stack(ws + ws))
        free = (
            planner.check_validity_batched(state_t, world_t)
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
        )
        for i in range(batch):
            if not (free[i] and free[batch + i] and poses_separated(ss[i], gs[i])):
                continue
            planned += 1
            world = world_t[i]
            goal_t = torch.from_numpy(gs[i])
            ok = bool(
                planner.search(
                    torch.from_numpy(ss[i]),
                    goal_t,
                    world,
                    limit,
                    hyst,
                    False,
                )
            )
            path = planner.get_best_path()
            first_sol, cost_exp = cost_expansion(planner)
            if ok and path.numel() == 0:
                ok = False
            if ok:
                path_np = path.detach().cpu().numpy()
                starts.append(ss[i])
                goals.append(gs[i])
                obstacles.append(ws[i])
                seeds.append(ids[i])
                paths.append(path_np)
                cost_exps.append(cost_exp)
                first_sols.append(first_sol)
                if len(starts) % 10 == 0:
                    write_set()
                    print(f"checkpoint {len(starts)} -> {args.out}")
                if args.video_dir:
                    render_rearrange_video(
                        path_np,
                        gs[i],
                        ws[i],
                        os.path.join(
                            args.video_dir, f"p{len(starts) - 1:03d}_draw{ids[i]}.mp4"
                        ),
                    )
            print(
                f"candidate {planned} draw {ids[i]} solved={ok} "
                f"first_sol={first_sol} path={tuple(path.shape)} "
                f"kept={len(starts)}/{args.n}"
            )
            if planned >= args.max_candidates or len(starts) >= args.n:
                break

    write_set()
    print(f"wrote {len(starts)} problems to {args.out}")


if __name__ == "__main__":
    main()
