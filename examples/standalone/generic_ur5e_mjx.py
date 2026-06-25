#!/usr/bin/env python3
"""SKELETON: plan for a 6-DoF UR5e arm using the generic IGHAStar environment
with MuJoCo / MJX for batched forward kinematics and collision checking.

Task: move the end-effector above a target height (goal test: ee_z > Z_THRESH).

This is a runnable scaffold: fill in MODEL_XML_PATH and (optionally) the
collision logic. State = 6 joint angles, control = 6 joint-angle deltas.

The generic environment calls these Python callbacks with batched torch tensors
(shape [B, n_dims] / [B, n_cont]) and expects torch tensors back. We convert
torch <-> jax via numpy. MJX/JAX functions are jit + vmap'd so a whole batch of
configurations is evaluated in one device call, which is exactly what the
preemptive-expansion batching feeds.

Requirements:  pip install mujoco       (default native-FK backend)
               pip install mujoco-mjx jax   (only for FK_BACKEND=mjx)
"""
import os
import time

import numpy as np
import torch
import yaml

from ighastar.scripts.common_utils import BASE_DIR, create_planner


class _FKProfiler:
    """Times the MJX forward-kinematics callback, separating the (JIT-heavy)
    first call from the steady-state rate."""

    def __init__(self) -> None:
        self.first_call_s = None  # includes JAX jit compile
        self.n_calls = 0  # excludes the first call
        self.total_s = 0.0  # excludes the first call
        self.total_states = 0  # states processed in the timed (non-first) calls

    def record(self, dt: float, n_states: int) -> None:
        if self.first_call_s is None:
            self.first_call_s = dt
        else:
            self.n_calls += 1
            self.total_s += dt
            self.total_states += n_states

    def report(self) -> None:
        if self.first_call_s is not None:
            print(f"first FK call: {self.first_call_s * 1e3:.1f} ms "
                  f"(includes JAX JIT compile)")
        if self.n_calls > 0 and self.total_s > 0:
            mean_ms = self.total_s / self.n_calls * 1e3
            print(f"subsequent FK calls: {self.n_calls}, mean {mean_ms:.3f} ms, "
                  f"rate {self.n_calls / self.total_s:.1f} calls/s, "
                  f"{self.total_states / self.total_s:.0f} states/s")
        else:
            print("subsequent FK calls: none recorded")


FK_PROF = _FKProfiler()

# --------------------------------------------------------------------------
# Problem constants -- EDIT THESE
# --------------------------------------------------------------------------
MODEL_XML_PATH = os.environ.get("UR5E_XML", "")  # TODO: path to a UR5e MJCF
EE_BODY_NAME = "wrist_3_link"  # TODO: end-effector body name in the MJCF
Z_THRESH = 0.6  # goal: end-effector height above this (meters)
N_JOINTS = 6
STATE_DIM = N_JOINTS
CONTROL_DIM = N_JOINTS
NUM_CONTROLS = 64  # branching factor K (random joint-delta primitives)
DELTA = 0.15  # max joint-angle step per expansion (radians)
ROLLOUT_T = 8  # sub-steps stored per expansion (full-rollout / smooth playback)

# Joint limits (UR5e); adjust to your MJCF.
JOINT_LOWER = torch.tensor([-2 * np.pi] * N_JOINTS, dtype=torch.float32)
JOINT_UPPER = torch.tensor([2 * np.pi] * N_JOINTS, dtype=torch.float32)


# --------------------------------------------------------------------------
# Forward-kinematics backends. FK_BACKEND selects which one builds the
# ee_pos(qpos_batch[B, n_joints]) -> ee_xyz[B, 3] callable.
#   "mujoco" (default): native MuJoCo mj_kinematics in a loop. For a small arm
#                       on CPU this is microseconds/call -- far faster than
#                       JAX-on-CPU, and needs no jax at all.
#   "mjx":              batched JAX/MJX (only worth it on GPU with big batches).
# --------------------------------------------------------------------------
FK_BACKEND = os.environ.get("FK_BACKEND", "mujoco")


def _build_mujoco():
    import mujoco

    assert MODEL_XML_PATH, "Set UR5E_XML / MODEL_XML_PATH to a UR5e MJCF file."
    mj_model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    mj_data = mujoco.MjData(mj_model)
    ee_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
    nq = mj_model.nq

    def ee_pos(qpos_batch_np):  # [B, n_joints] numpy -> [B, 3] numpy
        b = qpos_batch_np.shape[0]
        n = min(nq, qpos_batch_np.shape[1])
        out = np.empty((b, 3), dtype=np.float32)
        for i in range(b):
            mj_data.qpos[:n] = qpos_batch_np[i, :n]
            mujoco.mj_kinematics(mj_model, mj_data)  # forward kinematics only
            out[i] = mj_data.xpos[ee_body_id]
        return out

    return ee_pos


def _build_mjx():
    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    assert MODEL_XML_PATH, "Set UR5E_XML / MODEL_XML_PATH to a UR5e MJCF file."
    mj_model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    mjx_model = mjx.put_model(mj_model)
    ee_body_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME
    )

    @jax.jit
    @jax.vmap
    def ee_pos_single(qpos):
        data = mjx.make_data(mjx_model)
        data = data.replace(qpos=qpos)
        data = mjx.kinematics(mjx_model, data)  # forward kinematics only
        return data.xpos[ee_body_id]  # [3]

    def ee_pos(qpos_batch_np):  # [B, n_joints] numpy -> [B, 3] numpy
        # np.array (not np.asarray) forces a writable copy, so the downstream
        # torch.from_numpy does not warn about a read-only JAX-backed buffer.
        return np.array(ee_pos_single(jnp.asarray(qpos_batch_np)))

    return ee_pos


_EE_POS = None


def _ee_pos(states_np: np.ndarray) -> np.ndarray:
    global _EE_POS
    if _EE_POS is None:
        _EE_POS = _build_mjx() if FK_BACKEND == "mjx" else _build_mujoco()
    t0 = time.perf_counter()
    out = _EE_POS(states_np)  # blocks until results are on host
    FK_PROF.record(time.perf_counter() - t0, int(states_np.shape[0]))
    return out


# --------------------------------------------------------------------------
# Generic-environment callbacks (torch in / torch out)
# --------------------------------------------------------------------------
def sample_controls() -> torch.Tensor:
    # Random joint-delta primitives in [-DELTA, DELTA]^n_joints. The user owns
    # this choice (could also be a fixed grid of primitives).
    return (torch.rand(NUM_CONTROLS, CONTROL_DIM) * 2.0 - 1.0) * DELTA


def dynamics(states: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
    # Apply the joint-delta as a ROLLOUT_T-step linear interpolation so the
    # planner stores (and the viewer can replay) the full sub-trajectory.
    # Returns [B, ROLLOUT_T, n_joints]; the last sub-step is the node's state.
    nxt = torch.maximum(torch.minimum(states + controls, JOINT_UPPER), JOINT_LOWER)
    alphas = torch.linspace(1.0 / ROLLOUT_T, 1.0, ROLLOUT_T)  # [T]
    diff = nxt - states  # [B, n]
    return states[:, None, :] + alphas[None, :, None] * diff[:, None, :]


def cost(states, controls, next_states):
    # Joint-space path length.
    return torch.norm(next_states - states, dim=1)


def validity(states: torch.Tensor) -> torch.Tensor:
    in_limits = (states >= JOINT_LOWER).all(dim=1) & (states <= JOINT_UPPER).all(dim=1)
    # TODO: add MJX collision checking here (e.g. run mjx.forward + read
    # data.ncon / contact distances) and AND it with in_limits.
    return in_limits.float()


def heuristic(states: torch.Tensor) -> torch.Tensor:
    ee = torch.from_numpy(_ee_pos(states.cpu().numpy())).float()  # [B, 3]
    # Cost-to-go estimate: how far the EE height is below the threshold.
    return torch.clamp(Z_THRESH - ee[:, 2], min=0.0)


def goal_test(state: torch.Tensor) -> bool:
    ee = _ee_pos(state.unsqueeze(0).cpu().numpy())  # [1, 3]
    return bool(ee[0, 2] > Z_THRESH)


def main() -> None:
    config = {
        "experiment_info_default": {
            "state_dim": STATE_DIM,
            "control_dim": CONTROL_DIM,
            "hash_dims": STATE_DIM,
            "num_controls": NUM_CONTROLS,
            "resolution": [0.2] * STATE_DIM,
            "tolerance": [0.05] * STATE_DIM,
            "bounds_lower": JOINT_LOWER.tolist(),
            "bounds_upper": JOINT_UPPER.tolist(),
            "max_level": 4,
            "division_factor": 2.0,
            "max_expansions": 20000,
            "hysteresis": 2000,
            # Preemptive expansion (batched multi-vertex expansion) is OFF for
            # now so the FK timing reflects single-vertex expansions.
            "preemptive_expansion": {
                "enabled": False,
                "min_preemptive": 16,
                "max_preemptive": 64,
            },
            "node_info": {"node_type": "generic", "timesteps": 1},
        },
        "sample_controls_fn": sample_controls,
        "dynamics_fn": dynamics,
        "cost_fn": cost,
        "validity_fn": validity,
        "heuristic_fn": heuristic,
        "goal_test_fn": goal_test,
    }

    planner = create_planner(config, bidirectional=False)

    start = torch.zeros(STATE_DIM, dtype=torch.float32)  # home configuration
    goal = torch.zeros(STATE_DIM, dtype=torch.float32)  # unused by goal_test
    world = torch.zeros(1, dtype=torch.float32)  # unused by generic env

    t0 = time.perf_counter()
    success = planner.search(start, goal, world, 20000, 2000, True)
    search_s = time.perf_counter() - t0
    print("success:", success)
    print(f"FK backend: {FK_BACKEND}")
    print(f"search wall time: {search_s:.3f} s")
    FK_PROF.report()
    if success:
        path = planner.get_best_path().numpy()
        print(f"path: {path.shape[0]} configurations, cost {path[0, -1]:.3f}")
        print(f"preemptive expansions: {planner.get_preemptive_expansions()}")
        # Path is returned goal-first; reverse to start->goal and keep the joint
        # columns (drop the trailing g column).
        qpos_path = path[::-1, :STATE_DIM]
        if os.environ.get("VIEW", "1") != "0":
            visualize_path(qpos_path)
    else:
        print("No path found; try increasing max_expansions / DELTA / NUM_CONTROLS.")


def visualize_path(qpos_path: np.ndarray, step_dt: float = 0.12) -> None:
    """Replay a sequence of joint configurations in the interactive viewer.

    Loops until the viewer window is closed. Set VIEW=0 to skip visualization
    (e.g. on a headless machine).
    """
    import time

    import mujoco
    import mujoco.viewer

    mj_model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    mj_data = mujoco.MjData(mj_model)
    n = min(mj_model.nq, qpos_path.shape[1])
    print(f"Launching viewer: replaying {qpos_path.shape[0]} configurations "
          f"(close the window to exit)...")
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            for q in qpos_path:
                if not viewer.is_running():
                    break
                mj_data.qpos[:n] = q[:n]
                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()
                time.sleep(step_dt)
            time.sleep(0.5)  # pause at the goal before looping


if __name__ == "__main__":
    main()
