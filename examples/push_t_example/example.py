#!/usr/bin/env python3
"""Closed-loop Push-T: QS IGHA* plans, MPPI tracks in pymunk.

Needs Chipmunk 7 at pusht_ext/Chipmunk2D (or PUSHT_CHIPMUNK_DIR) and gym-pusht.
MPPI is cloned next to this repo on first closed-loop run if it is missing.

  python example.py                  # writes Content/push_t_example/push_t_closed_loop.mp4
  python example.py --live           # same, and a window that updates each step
  python example.py --open-loop      # open_loop_qs.mp4 or open_loop_chipmunk.mp4
  python example.py --dynamics qs    # planner model: qs or chipmunk (default is the yaml value)
  python example.py --no-visuals     # no video; raises if the block misses the goal
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONTENT = REPO / "Content" / "push_t_example"
sys.path.insert(0, str(HERE))

from cost import PushTCost  # noqa: E402
from dynamics import DT, PushTDynamics, _core, _read, hold_action  # noqa: E402
from planner_mp import PushTPlannerMP  # noqa: E402


def _require_gym_pusht():
    try:
        import gym_pusht  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "gym-pusht is required for this example. Install with: pip install gym-pusht"
        ) from e


_require_gym_pusht()
from sampling import Delta_Sampling, MPPI  # noqa: E402


def in_goal(state, goal, epsilon) -> bool:
    d = state[2:5] - goal[2:5]
    d[2] = np.arctan2(np.sin(d[2]), np.cos(d[2]))
    return abs(d[0]) < epsilon[2] and abs(d[1]) < epsilon[3] and abs(d[2]) < epsilon[4]


def path_index(path, t_now) -> int:
    # First row whose stamped sim time is at or after t_now. g is one substep per row.
    i = int(np.searchsorted(path[:, -1], t_now, side="left"))
    return min(i, len(path) - 1)


def ref_window(path, t_now, horizon, stride) -> np.ndarray:
    # Path rows are physics substeps. One MPPI step holds the action for `stride` of them.
    i = path_index(path, t_now)
    idx = np.clip(i + np.arange(horizon) * stride, 0, len(path) - 1)
    return path[idx, :10]


def need_plan(path, t_now, stride) -> bool:
    # ponytail: QS search returns in milliseconds, so a query every idle cycle swaps the
    # reference under MPPI. Hold the path until the clock reaches the last edge.
    if path is None:
        return True
    return path_index(path, t_now) + stride >= len(path)


def _to_px(xy, w, h):
    return np.stack([xy[:, 0] / 512.0 * w, xy[:, 1] / 512.0 * h], axis=1).astype(np.int32)


def draw_overlay(frame, agent_xy, planned_xy, rollouts=None, costs=None, k=0):
    # Frame is RGB. Black: lowest-cost rollout traces. Orange: current plan. Blue: executed path.
    h, w = frame.shape[:2]
    if rollouts is not None:
        order = np.argsort(costs)[:k]
        for i in order:
            cv2.polylines(frame, [_to_px(rollouts[i, :, :2], w, h)], False, (0, 0, 0), 1, cv2.LINE_AA)
    if len(planned_xy) >= 2:
        cv2.polylines(frame, [_to_px(planned_xy, w, h)], False, (255, 140, 0), 2, cv2.LINE_AA)
    if len(agent_xy) >= 2:
        cv2.polylines(frame, [_to_px(np.asarray(agent_xy), w, h)], False, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.yml"))
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--open-loop", action="store_true")
    parser.add_argument("--dynamics", choices=("qs", "chipmunk"))
    parser.add_argument("--video", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    mppi_cfg = cfg["MPPI_config"]
    goal = np.asarray(cfg["goal"], dtype=np.float32)
    epsilon = cfg["Planner_config"]["experiment_info_default"]["epsilon"]

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    env.reset(seed=0)
    core = _core(env)
    # Place the example start. Angle before position (block CoG).
    start = np.asarray(cfg["start"], dtype=np.float32)
    core.block.angle = float(start[4])
    core.block.position = (float(start[2]), float(start[3]))
    core.agent.position = (float(start[0]), float(start[1]))
    core.agent.velocity = (0.0, 0.0)
    core.block.velocity = (0.0, 0.0)
    core.block.angular_velocity = 0.0

    node_info = cfg["Planner_config"]["experiment_info_default"]["node_info"]
    if args.dynamics:
        node_info["dynamics"] = "quasistatic" if args.dynamics == "qs" else "chipmunk"
        print(f"planner dynamics: {node_info['dynamics']}")
    dynamics = costs = sampling = controller = None
    if not args.open_loop:
        dynamics = PushTDynamics(cfg["Dynamics_config"], mppi_cfg, dtype, device)
        costs = PushTCost(cfg["Cost_config"], dtype, device)
        sampling = Delta_Sampling(cfg["Sampling_config"], mppi_cfg, dtype, device)
        # Half-width of the PD-target box, centered on the agent.
        command_radius = float(max(node_info["action_radii"])) * float(mppi_cfg["TIMESTEPS"]) / float(
            cfg["Dynamics_config"]["n_substeps"]
        )
        controller = MPPI(dynamics, costs, sampling, mppi_cfg, device=device, dtype=dtype)

    planner = PushTPlannerMP(cfg["Planner_config"])
    n_sub = int(cfg["Dynamics_config"]["n_substeps"])
    t_world = 0.0
    last_action = start[:2].copy()
    agent_xy = [start[:2].copy()]
    writer = None
    dyn_name = str(node_info.get("dynamics", "chipmunk")).lower()
    open_tag = "qs" if dyn_name in ("quasistatic", "qs", "mason") else "chipmunk"
    default_video = f"open_loop_{open_tag}.mp4" if args.open_loop else "push_t_closed_loop.mp4"
    out = args.video or str(CONTENT / default_video)

    success = False
    try:
        for step in range(int(cfg["step_limit"])):
            state = _read(core)
            if in_goal(state, goal, epsilon):
                success = True
                print(f"goal at step {step}")
                break
            # Open loop plays the whole tape. Past the last stamp, wait. Do not repeat the last command.
            if args.open_loop and (planner.path is None or t_world > planner.path[-1, -1]):
                planner.set_query(state, goal, t_world)
                planner.update()
                time.sleep(0.01)
                continue
            if not args.open_loop and need_plan(planner.path, t_world, n_sub):
                planner.set_query(state, goal, t_world)
            got, _, _ = planner.update()
            if planner.path is None:
                time.sleep(0.05)
                continue
            if got:
                p = planner.path
                print(f"path len={len(p)} block {p[0,2:5]} -> {p[-1,2:5]}")
            if args.open_loop:
                # First row strictly after t_world, so an edge boundary takes the next action.
                i = int(np.searchsorted(planner.path[:, -1], t_world, side="right"))
                last_action = planner.controls[min(i, len(planner.controls) - 1)]
            else:
                ref = ref_window(planner.path, t_world, mppi_cfg["TIMESTEPS"], n_sub)
                costs.set_path(torch.tensor(ref, dtype=dtype))
                ctrl_state = np.concatenate([state, last_action]).astype(np.float32)
                sampling.set_command_box(
                    state[:2],
                    command_radius,
                    cfg["Dynamics_config"]["workspace_lo"],
                    cfg["Dynamics_config"]["workspace_hi"],
                )
                action = controller.forward(
                    torch.tensor(ctrl_state, dtype=dtype, device=device), int(mppi_cfg["n_iter"])
                )
                last_action = action[0].detach().cpu().numpy()
            hold_action(core, last_action, n_sub, float(cfg["Dynamics_config"]["max_agent_vel"]))
            t_world += n_sub * DT
            agent_xy.append(_read(core)[:2].copy())
            if args.live or not args.no_visuals:
                # Clear the action so gym does not paint the red X on the target.
                core._last_action = None
                frame = draw_overlay(
                    env.render(),
                    agent_xy,
                    planner.path[:, :2],
                    None if args.open_loop else dynamics.states[0].detach().cpu().numpy(),
                    None if args.open_loop else sampling.cost_total.detach().cpu().numpy(),
                    int(cfg["trace_k"]),
                )
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if args.live:
                    cv2.imshow("push-t", bgr)
                    cv2.waitKey(1)
                if not args.no_visuals:
                    if writer is None:
                        h, w = frame.shape[:2]
                        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
                        writer = cv2.VideoWriter(
                            out, cv2.VideoWriter_fourcc(*"mp4v"), cfg["fps"], (w, h)
                        )
                        print(f"video -> {out}")
                    writer.write(bgr)
            if step % 20 == 0:
                print(f"step {step} block=({state[2]:.0f},{state[3]:.0f},{state[4]:.2f})")
    finally:
        if writer is not None:
            writer.release()
        if args.live:
            cv2.destroyAllWindows()
        planner.shutdown()
        env.close()
        if dynamics is not None:
            dynamics.env.close()

    if not success:
        raise RuntimeError("push_t closed loop: block did not reach the goal")


if __name__ == "__main__":
    main()
