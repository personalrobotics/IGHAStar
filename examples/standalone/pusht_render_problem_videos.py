#!/usr/bin/env python3
"""Plan + render mp4 for all dino PushT problems (best config).

  OMP_NUM_THREADS=8 /home/stark/ighastar-venv/bin/python \\
    examples/standalone/pusht_render_problem_videos.py

Writes Content/standalone/pusht_problem_videos/p{idx:03d}_{ok|fail}.mp4
Unsolved problems get a short start→goal teleport clip (green = goal tee).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import torch
import yaml

from ighastar.scripts.common_utils import BASE_DIR, create_planner

_sweep_path = os.path.join(
    BASE_DIR.parent, "examples", "standalone", "pusht_hyperparam_sweep.py"
)
_spec = importlib.util.spec_from_file_location("pusht_sweep", _sweep_path)
_sweep = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_sweep)

CONTENT = _sweep.CONTENT


def unwrap(env):
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e


def load_snapshot_gym(core, state: np.ndarray) -> None:
    s = np.asarray(state, dtype=np.float64).reshape(-1)
    core.agent.velocity = (0.0, 0.0)
    core.block.velocity = (0.0, 0.0)
    core.block.angular_velocity = 0.0
    core.block.angle = float(s[4])
    core.block.position = (float(s[2]), float(s[3]))
    core.agent.position = (float(s[0]), float(s[1]))
    core.space.step(core.dt)


def render_states(states: np.ndarray, goal_pose_bxyth: np.ndarray, out_mp4: str, fps: int) -> None:
    import gymnasium as gym
    import gym_pusht  # noqa: F401
    import imageio.v2 as imageio

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    env.reset(seed=0)
    core = unwrap(env)
    # Green overlay = planner goal block pose (not gym default).
    core.goal_pose = np.array(
        [float(goal_pose_bxyth[0]), float(goal_pose_bxyth[1]), float(goal_pose_bxyth[2])],
        dtype=np.float64,
    )

    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    frames = []
    for row in states:
        load_snapshot_gym(core, row[:10])
        frames.append(np.asarray(env.render()))
    env.close()
    imageio.mimsave(out_mp4, frames, fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-yaml",
        default=os.path.join(
            BASE_DIR.parent, "examples", "standalone", "Configs", "pusht.yml"
        ),
    )
    parser.add_argument(
        "--problems",
        default=os.path.join(CONTENT, "pusht_dino_problems.npz"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(CONTENT, "pusht_qs_best_videos"),
    )
    parser.add_argument("--max-expansions", type=int, default=50000)
    parser.add_argument(
        "--n-directions",
        type=int,
        default=24,
        help="Succ ring directions (best QS OAT cell uses 24)",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--omp-threads", type=int, default=8)
    parser.add_argument("--hold-frames", type=int, default=15, help="fail clip hold per pose")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)

    with open(args.base_yaml) as f:
        base = yaml.safe_load(f)
    cell = dict(_sweep.BASELINE)
    cell["n_directions"] = int(args.n_directions)
    cell["max_expansions"] = int(args.max_expansions)
    cfg = _sweep.apply_sweep_config(base, cell)
    hyst = int(cfg["experiment_info_default"]["hysteresis"])
    # Render teleports gym-pusht to planner states (not action replay / true dyn).

    problems = _sweep.load_or_make_problems(args.problems, regen=False, n_problems=50)
    n = len(problems["seeds"])
    os.makedirs(args.out_dir, exist_ok=True)

    planner = create_planner(cfg, bidirectional=False)
    n_ok = 0
    t_all = time.perf_counter()

    for pi in range(n):
        start = problems["starts"][pi]
        goal = problems["goals"][pi]
        goal_b = goal[2:5]
        t0 = time.perf_counter()
        ok = bool(
            planner.search(
                torch.as_tensor(start),
                torch.as_tensor(goal),
                torch.zeros(1, dtype=torch.float32),
                args.max_expansions,
                hyst,
                True,
            )
        )
        wall = time.perf_counter() - t0
        if ok:
            path_sg = planner.get_best_path().numpy()[::-1].copy()
            tag = "ok"
            n_ok += 1
        else:
            # start hold → goal hold so the hard cases are still visible
            h = max(1, args.hold_frames)
            path_sg = np.vstack(
                [np.tile(start, (h, 1)), np.tile(goal, (h, 1))]
            ).astype(np.float32)
            # pad fake g column for render (uses [:10] only)
            path_sg = np.concatenate(
                [path_sg, np.zeros((path_sg.shape[0], 1), dtype=np.float32)], axis=1
            )
            tag = "fail"

        out = os.path.join(args.out_dir, f"p{pi:03d}_{tag}.mp4")
        render_states(path_sg, goal_b, out, args.fps)
        print(
            f"[{pi+1}/{n}] p{pi:03d} {tag}  T={path_sg.shape[0]}  "
            f"plan={wall:.2f}s  → {os.path.basename(out)}"
        )

    print(
        f"done: {n_ok}/{n} solved  videos in {args.out_dir}  "
        f"wall={time.perf_counter()-t_all:.1f}s"
    )


if __name__ == "__main__":
    main()
