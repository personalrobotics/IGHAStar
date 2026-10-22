#!/usr/bin/env python3
"""PushT IGHAStar smoke test + gym-pusht render of the dense path.

Path includes every physics substep (not just edge endpoints). Frames are
rendered by teleporting gym-pusht to each state and calling env.render().

  cd examples/standalone
  python3 pusht_example.py
  # writes Content/standalone/pusht_plan.mp4
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import yaml

from ighastar.scripts.common_utils import BASE_DIR, create_planner


def unwrap(env):
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e


def load_snapshot_gym(core, state: np.ndarray) -> None:
    """Teleport gym-pusht to a planner state for rendering.

    Zero velocities before the mandatory space.step() so one dt of physics
    doesn't drift the pose (pymunk only refreshes draw transforms on step).
    """
    s = np.asarray(state, dtype=np.float64).reshape(-1)
    core.agent.velocity = (0.0, 0.0)
    core.block.velocity = (0.0, 0.0)
    core.block.angular_velocity = 0.0
    # angle before position (CoG ≠ geometric origin)
    core.block.angle = float(s[4])
    core.block.position = (float(s[2]), float(s[3]))
    core.agent.position = (float(s[0]), float(s[1]))
    core.space.step(core.dt)


def render_path_video(path_sg: np.ndarray, out_mp4: str, fps: int = 20) -> None:
    """path_sg: start→goal dense states [T, 10+] (last col may be g)."""
    import gymnasium as gym
    import gym_pusht  # noqa: F401
    import imageio.v2 as imageio

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    env.reset(seed=0)
    core = unwrap(env)

    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    frames = []
    for i, row in enumerate(path_sg):
        load_snapshot_gym(core, row[:10])
        frame = env.render()
        frames.append(np.asarray(frame))
        if (i + 1) % 20 == 0 or i == 0 or i + 1 == len(path_sg):
            print(f"  frame {i + 1}/{len(path_sg)}")

    env.close()
    imageio.mimsave(out_mp4, frames, fps=fps)
    print(f"wrote {out_mp4}  ({len(frames)} frames @ {fps} fps)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, help="output mp4 path")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    yaml_path = os.path.join(BASE_DIR.parent, "examples", "standalone", "Configs", "pusht.yml")
    with open(yaml_path) as f:
        configs = yaml.safe_load(f)

    info = configs["experiment_info_default"]
    planner = create_planner(configs, bidirectional=False)

    goal_angle = float(np.pi / 4)
    goal = torch.tensor(
        [256.0, 256.0, 256.0, 256.0, goal_angle, 0, 0, 0, 0, 0],
        dtype=torch.float32,
    )
    start = torch.tensor(
        [100.0, 100.0, 200.0, 200.0, 0.5, 0, 0, 0, 0, 0],
        dtype=torch.float32,
    )
    world = torch.zeros(1, dtype=torch.float32)

    t0 = time.perf_counter()
    success = planner.search(
        start,
        goal,
        world,
        info["max_expansions"],
        info["hysteresis"],
        True,
    )
    dt = time.perf_counter() - t0
    print(f"success={success}  wall={dt:.3f}s")

    (
        avg_successor_time,
        avg_goal_check_time,
        avg_overhead_time,
        avg_g_update_time,
        switches,
        max_level_profile,
        Q_v_size,
        expansion_counter,
        expansion_list,
        cost_exp_list,
    ) = planner.get_profiler_info()
    print(f"expansions={expansion_counter}")
    print(f"avg_succ_time={avg_successor_time:.4f} us")
    print(f"avg_goal_check_time={avg_goal_check_time:.4f} us")
    print(f"avg_overhead_time={avg_overhead_time:.4f} us")
    print(f"avg_g_update_time={avg_g_update_time:.4f} us")
    print(f"switches={switches}  max_level={max_level_profile}")
    if switches > 0:
        print(f"avg_Q_v={Q_v_size / switches:.1f}  (sum_Q_v={Q_v_size})")
    print(f"preemptive_expansions={planner.get_preemptive_expansions()}")
    # cost_exp_list[i] = Omega after expansion i; expansion_list is per-switch only.
    cost_exp_arr = np.asarray(cost_exp_list, dtype=np.float64)
    sol = np.where(cost_exp_arr < 1e4)[0]
    if sol.size:
        i0 = int(sol[0])
        print(
            f"first_solution_exp={i0}  "
            f"cost={cost_exp_arr[i0]:.3f}  "
            f"best_cost={cost_exp_arr.min():.3f}"
        )

    if not success:
        raise RuntimeError(
            "pusht_example: search failed "
            "(try raising max_expansions or loosening epsilon)"
        )

    path = planner.get_best_path().numpy()  # goal-first, dense substeps
    path_sg = path[::-1].copy()  # start→goal
    print(f"path states={path_sg.shape[0]}  cost={path[0, -1]:.3f}")
    print("idx | ax      ay      bx      by      th       | g")
    for i, s in enumerate(path_sg):
        print(
            f"{i:3d} | {s[0]:7.2f} {s[1]:7.2f} {s[2]:7.2f} {s[3]:7.2f} "
            f"{s[4]:8.4f} | {s[-1]:.3f}"
        )

    if args.no_video:
        return

    out = args.video or os.path.join(
        BASE_DIR.parent, "Content", "standalone", "pusht_plan.mp4"
    )
    print(f"rendering {path_sg.shape[0]} substep frames…")
    render_path_video(path_sg, out, fps=args.fps)


if __name__ == "__main__":
    main()
