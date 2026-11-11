#!/usr/bin/env python3
"""Sample 2T1O_clear starts and keep those solved within the expansion limit.

State [16]: ax ay t0x t0y t0a t1x t1y t1a avx avy t0vx t0vy t0w t1vx t1vy t1w.
World tensor is a maze: [n, then n segments of x1, y1, x2, y2].
A row of length 17 is the old gap encoding and is still accepted.
The goal tensor is unused. Search with ignore_goal_validity=True. A tee is
cleared when every corner is in the exit box. The CoG is checked first.
The bits live on the node.

  python examples/standalone/pusht_multi_generate_2T1O_clear.py --n 1
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml

from ighastar.scripts.common_utils import BASE_DIR, create_planner

CONTENT = os.path.join(BASE_DIR.parent, "Content", "standalone")
CONFIG = os.path.join(
    BASE_DIR.parent, "examples", "standalone", "Configs", "pusht_multi_2T1O_clear.yml"
)
N_DIMS = 16
MAX_WALLS = 32  # pusht_multi_physics.h kMaxWalls
WORLD_STRIDE = 1 + 4 * MAX_WALLS
LEGACY_STRIDE = 1 + 4 * 4
GAP = 160.0
BOX_LO = 5.0
BOX_HI = 506.0


def load_config() -> dict:
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def pack_state(ax, ay, t0, t1) -> np.ndarray:
    s = np.zeros(N_DIMS, dtype=np.float32)
    s[0] = ax
    s[1] = ay
    s[2:5] = t0
    s[5:8] = t1
    return s


def gap_wall(orient, coord, gap_lo, gap_hi):
    """The two solid pieces of one full-span wall. The gap is the opening."""
    g0, g1 = (float(gap_lo), float(gap_hi))
    if g0 > g1:
        g0, g1 = g1, g0
    if float(orient) < 0.5:
        pieces = (
            ((BOX_LO, float(coord)), (g0, float(coord))),
            ((g1, float(coord)), (BOX_HI, float(coord))),
        )
    else:
        pieces = (
            ((float(coord), BOX_LO), (float(coord), g0)),
            ((float(coord), g1), (float(coord), BOX_HI)),
        )
    return [p for p in pieces if abs(p[1][0] - p[0][0]) + abs(p[1][1] - p[0][1]) >= 1.0]


def pack_walls(segments) -> np.ndarray:
    """segments: ((x1, y1), (x2, y2)). Unused slots stay 0."""
    row = np.zeros(WORLD_STRIDE, dtype=np.float32)
    n = min(len(segments), MAX_WALLS)
    row[0] = n
    for i, (a, b) in enumerate(segments[:n]):
        row[1 + 4 * i : 5 + 4 * i] = (a[0], a[1], b[0], b[1])
    return row


def sample_maze(rng: np.random.RandomState, n_walls: int = 1) -> np.ndarray:
    """One wall across the box. The gap is centered so both sides stay connected."""
    del n_walls
    orient = float(rng.randint(0, 2))
    coord = float(rng.uniform(160.0, 360.0))
    mid = 0.5 * (BOX_LO + BOX_HI)
    half = 0.5 * GAP
    return pack_walls(gap_wall(orient, coord, mid - half, mid + half))


def wall_segments(row):
    """Interior segments in world pixels. A length-17 row is the old gap encoding."""
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    n = int(row[0]) if row.size else 0
    if row.size == LEGACY_STRIDE:
        segs = []
        for i in range(max(0, min(n, 4))):
            segs.extend(gap_wall(*row[1 + 4 * i : 5 + 4 * i]))
        return segs
    n = min(max(0, n), max(0, (row.size - 1) // 4))
    segs = []
    for i in range(n):
        x1, y1, x2, y2 = (float(v) for v in row[1 + 4 * i : 5 + 4 * i])
        segs.append(((x1, y1), (x2, y2)))
    return segs


def sample_candidate(rng: np.random.RandomState, n_walls: int = 0):
    agent = rng.uniform(40.0, 470.0, size=2)
    tees = []
    for _ in range(2):
        xy = rng.uniform(90.0, 400.0, size=2)
        ang = rng.uniform(0.0, 2.0 * np.pi)
        tees.append(np.array([xy[0], xy[1], ang], dtype=np.float32))
    state = pack_state(agent[0], agent[1], tees[0], tees[1])
    return state, sample_maze(rng, n_walls)


def _tee_polys(x: float, y: float, ang: float):
    """Local Tee polys from pusht_multi_physics.h, rotated about the body origin."""
    s, L = 30.0, 4.0
    bars = (
        ((-L * s / 2, s), (L * s / 2, s), (L * s / 2, 0), (-L * s / 2, 0)),
        ((-s / 2, s), (-s / 2, L * s), (s / 2, L * s), (s / 2, s)),
    )
    c, sn = np.cos(ang), np.sin(ang)
    out = []
    for poly in bars:
        out.append([(x + c * px - sn * py, y + sn * px + c * py) for px, py in poly])
    return out


def render_clear_video(
    path_goal_first: np.ndarray,
    obstacle_xy: np.ndarray,
    out_mp4: str,
    fps: int = 20,
) -> None:
    """path_goal_first: planner path [T, 17], goal first. Draws the open box."""
    from PIL import Image, ImageDraw
    import imageio.v2 as imageio

    rows = np.asarray(path_goal_first, dtype=np.float32)[::-1]
    if rows.shape[0] > 300:
        idx = np.linspace(0, rows.shape[0] - 1, 300).astype(np.int64)
        rows = rows[idx]
    # World x is 0..700 so the exit strip is on the frame. y is 0..512, up.
    # Width is a multiple of 16 so the mp4 encoder does not rescale the frame.
    W, H = 704, 512

    def pix(x, y):
        return (float(x), H - float(y))

    frames = []
    for row in rows:
        img = Image.new("RGB", (W, H), (255, 255, 255))
        dr = ImageDraw.Draw(img)
        dr.rectangle([pix(512, 512), pix(700, 0)], fill=(214, 240, 214))
        dr.line([pix(512, 0), pix(512, 512)], fill=(80, 160, 80), width=2)
        for a, b in (
            ((5, 5), (5, 506)),
            ((5, 5), (506, 5)),
            ((5, 506), (506, 506)),
        ):
            dr.line([pix(*a), pix(*b)], fill=(40, 40, 40), width=4)
        for a, b in wall_segments(obstacle_xy):
            dr.line([pix(*a), pix(*b)], fill=(90, 90, 90), width=6)
        colors = ((70, 130, 180), (210, 120, 40))
        for tee, color in ((row[2:5], colors[0]), (row[5:8], colors[1])):
            for poly in _tee_polys(float(tee[0]), float(tee[1]), float(tee[2])):
                dr.polygon([pix(px, py) for px, py in poly], fill=color)
        ax, ay = float(row[0]), float(row[1])
        r = 15.0
        dr.ellipse([pix(ax - r, ay + r), pix(ax + r, ay - r)], fill=(200, 40, 40))
        frames.append(np.asarray(img))
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    imageio.mimsave(out_mp4, frames, fps=fps)
    print(f"wrote {out_mp4} ({len(frames)} frames)")


def cost_expansion(planner):
    """Best-cost curve from the last search, and the expansion of the first solution.

    The curve is one sample per completed expansion. A solution is the first
    sample whose cost is below 1e4 (unsolved searches stay at 1e5).
    """
    cost = np.asarray(planner.get_profiler_info()[9], dtype=np.float32)
    hit = np.where(cost < 1e4)[0]
    first = int(hit[0]) if hit.size else -1
    return first, cost


def self_check(planner, check_exit: bool = True) -> None:
    """One separated start is valid. One tee sitting on a wall piece is not.

    check_exit is the clear task: a tee fully in the exit box is already solved,
    and a tee whose CoG is in the box but a corner is not is still a legal start.
    Rearrange closes the right wall and does not use that test.
    """
    world = torch.from_numpy(pack_walls(gap_wall(1.0, 400.0, 220.0, 380.0)))
    free = pack_state(80.0, 256.0, (150.0, 150.0, 0.0), (150.0, 320.0, 0.0))
    bad = pack_state(80.0, 256.0, (400.0, 100.0, 0.0), (150.0, 320.0, 0.0))
    states = torch.from_numpy(np.stack([free, bad]))
    mask = planner.check_validity_batched(states, world)
    ok = mask.detach().cpu().numpy().astype(bool)
    if not (ok[0] and not ok[1]):
        raise SystemExit(f"validity self-check failed: mask={ok.tolist()}")
    legacy = np.zeros(LEGACY_STRIDE, dtype=np.float32)
    legacy[0] = 1
    legacy[1:5] = (1.0, 400.0, 220.0, 380.0)
    legacy_mask = planner.check_validity_batched(states, torch.from_numpy(legacy))
    legacy_ok = legacy_mask.detach().cpu().numpy().astype(bool)
    if not (legacy_ok[0] and not legacy_ok[1]):
        raise SystemExit(f"legacy wall self-check failed: mask={legacy_ok.tolist()}")
    if check_exit:
        # Angle 0, body (600, 200): every corner is in x[512, 700], y[0, 512].
        # Body (512, 200): CoG is on the exit edge, the bar corner is left of it.
        whole = pack_state(80.0, 256.0, (600.0, 200.0, 0.0), (150.0, 320.0, 0.0))
        partial = pack_state(80.0, 256.0, (512.0, 200.0, 0.0), (150.0, 320.0, 0.0))
        exit_mask = planner.check_validity_batched(
            torch.from_numpy(np.stack([whole, partial])), world
        )
        exit_ok = exit_mask.detach().cpu().numpy().astype(bool)
        if exit_ok[0] or not exit_ok[1]:
            raise SystemExit(f"exit self-check failed: mask={exit_ok.tolist()}")
    print("validity self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="verified problems to keep")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=40,
        help="collision-free starts to plan, then stop",
    )
    parser.add_argument(
        "--n-walls",
        type=int,
        default=1,
        help="Unused. The maze is one wall with a centered gap.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default=os.path.join(CONTENT, "pusht_multi_2T1O_clear.npz"),
    )
    parser.add_argument(
        "--video-dir",
        default="",
        help="If set, write one mp4 per solved problem into this directory.",
    )
    args = parser.parse_args()

    cfg = load_config()
    info = cfg["experiment_info_default"]
    limit = int(info["max_expansions"])
    hyst = int(info["hysteresis"])
    planner = create_planner(cfg, bidirectional=False)
    self_check(planner)

    rng = np.random.RandomState(args.seed)
    starts = []
    obstacles = []
    seeds = []
    paths = []
    cost_exps = []
    first_sols = []
    planned = 0
    draws = 0
    goal = torch.zeros(N_DIMS, dtype=torch.float32)
    batch = 32
    while planned < args.max_candidates and len(starts) < args.n:
        states = []
        worlds = []
        draw_ids = []
        for _ in range(batch):
            state, obs = sample_candidate(rng, args.n_walls)
            draws += 1
            states.append(state)
            worlds.append(obs)
            draw_ids.append(draws)
        state_t = torch.from_numpy(np.stack(states))
        world_t = torch.from_numpy(np.stack(worlds))
        mask = planner.check_validity_batched(state_t, world_t)
        free = mask.detach().cpu().numpy().astype(bool)
        for i in range(batch):
            if not free[i]:
                continue
            planned += 1
            world = world_t[i]
            ok = bool(
                planner.search(
                    state_t[i],
                    goal,
                    world,
                    limit,
                    hyst,
                    True,
                )
            )
            path = planner.get_best_path()
            first_sol, cost_exp = cost_expansion(planner)
            if ok and path.numel() == 0:
                ok = False
            if ok:
                path_np = path.detach().cpu().numpy()
                starts.append(states[i])
                obstacles.append(worlds[i])
                seeds.append(draw_ids[i])
                paths.append(path_np)
                cost_exps.append(cost_exp)
                first_sols.append(first_sol)
                if args.video_dir:
                    render_clear_video(
                        path_np,
                        worlds[i],
                        os.path.join(
                            args.video_dir, f"p{len(starts) - 1:03d}_draw{draw_ids[i]}.mp4"
                        ),
                    )
            print(
                f"candidate {planned} draw {draw_ids[i]} solved={ok} "
                f"first_sol={first_sol} path={tuple(path.shape)} "
                f"kept={len(starts)}/{args.n}"
            )
            if planned >= args.max_candidates or len(starts) >= args.n:
                break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(
        args.out,
        starts=np.stack(starts) if starts else np.zeros((0, N_DIMS), np.float32),
        obstacles=np.stack(obstacles) if obstacles else np.zeros((0, WORLD_STRIDE), np.float32),
        seeds=np.asarray(seeds, dtype=np.int64),
        paths=np.array(paths, dtype=object),
        cost_exp=np.array(cost_exps, dtype=object),
        first_sol_expansions=np.asarray(first_sols, dtype=np.int64),
        problem_class=np.array("2T1W_clear"),
        max_expansions=np.array(limit),
    )
    print(f"wrote {len(starts)} problems to {args.out}")


if __name__ == "__main__":
    main()
