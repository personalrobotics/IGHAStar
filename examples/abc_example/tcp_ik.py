"""TCP-pose-delta -> joint-delta conversion for abc's YAM arms, via frax differential IK.

The planner's control for an arm is a rigid motion of the *grasp site* (the point
between the fingertips) in the world frame: ``[dx, dy, dz, rx, ry, rz]`` (metres,
axis-angle radians). Starting from the arm's current joint targets, ``frax.ik``
(differential IK + joint-limit constraints, ElastiQP) is run to convergence on the
moved pose. Being a local method it stays on the current joint branch, so a small
Cartesian move gives a small joint move, with no branch jumps.

A row counts as solved when the converged pose matches the target
(``POS_TOL`` / ``ROT_TOL``). Otherwise (unreachable, or stuck at a joint limit) it is
``ok=False`` and the planner treats the child as invalid; there are no retries
(many controls are sampled per expansion precisely because some will fail). An optional per-edge joint cap can be given (``cap`` finite); by
default only abc's joint limits apply.

The kinematics come from abc's MJCF: ``arm_urdf`` writes a kinematics-only URDF of
one arm (root = ``<side>_arm`` body, tip = ``<side>_link_6``), and the grasp site is the
end-effector offset from link_6. Everything runs on CPU JAX, batched, in-process.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # the GPU belongs to MJWarp
os.environ.setdefault("JAX_ENABLE_X64", "1")
# frax's recommended CPU settings (single-threaded XLA/BLAS); noticeably faster for these small solves
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                      "--xla_cpu_scheduler_type=CPU_SCHEDULER_TYPE_MEMORY_OPTIMIZED")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# Split the batch over several CPU "devices" (frax's recommended way to scale IK on CPU):
# each shard runs its own convergence loop, so rows no longer wait for the slowest row.
N_SHARDS = int(os.environ.get("TCP_IK_SHARDS", min(16, os.cpu_count() or 1)))
if "xla_force_host_platform_device_count" not in os.environ["XLA_FLAGS"]:
    os.environ["XLA_FLAGS"] += f" --xla_force_host_platform_device_count={N_SHARDS}"
import warnings

warnings.filterwarnings("ignore", message="Gimbal lock")

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

POS_TOL = 1e-3  # m
ROT_TOL = 1e-2  # rad
MAX_ITERS = 60
DT_MAX = 0.1  # with gain 10 /s a full step corrects the error in one iteration
GAIN = 10.0


def _pose(p, R) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R).reshape(3, 3)
    T[:3, 3] = p
    return T


def _f(x) -> str:
    return " ".join(f"{float(v):.12g}" for v in np.atleast_1d(x))


def arm_urdf(mjm: mujoco.MjModel, side: str) -> tuple[str, list[str]]:
    """Kinematics-only URDF text of one arm, and its joint names in order."""
    names = [f"{side}_joint{k}" for k in range(1, 7)]
    links = [f"{side}_arm"] + [f"{side}_link_{k}" for k in range(1, 7)]
    body = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    out = ['<?xml version="1.0"?>', f'<robot name="{side}_yam">']
    for ln in links:
        out.append(f'<link name="{ln}"><inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.5"/>'
                   '<inertia ixx="1e-3" iyy="1e-3" izz="1e-3" ixy="0" ixz="0" iyz="0"/></inertial></link>')
    for k, jn in enumerate(names):
        b = body(links[k + 1])
        assert int(mjm.body_parentid[b]) == body(links[k]), "arm bodies are not a plain chain"
        j = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, jn)
        assert int(mjm.jnt_bodyid[j]) == b and not np.any(mjm.jnt_pos[j]), "unexpected joint anchor"
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, mjm.body_quat[b])
        rpy = Rotation.from_matrix(R.reshape(3, 3)).as_euler("xyz")  # URDF rpy: fixed-axis xyz
        lo, hi = mjm.jnt_range[j]
        out.append(
            f'<joint name="{jn}" type="revolute"><parent link="{links[k]}"/><child link="{links[k + 1]}"/>'
            f'<origin xyz="{_f(mjm.body_pos[b])}" rpy="{_f(rpy)}"/>'
            f'<axis xyz="{_f(mjm.jnt_axis[j])}"/>'
            f'<limit lower="{_f(lo)}" upper="{_f(hi)}" effort="10" velocity="3"/></joint>')
    out.append("</robot>")
    return "\n".join(out), names


class _ArmIK:
    def __init__(self, mjm, d, side: str) -> None:
        from frax.core.manipulator import Manipulator
        from frax.ik import DiffIKSolver, EETask, JointLimits

        bid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, f"{side}_arm")
        l6 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, f"{side}_link_6")
        sid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, f"{side}_grasp_site")
        self.T_w_base = _pose(d.xpos[bid], d.xmat[bid])
        self.T_base_w = np.linalg.inv(self.T_w_base)
        T_l6_g = np.linalg.inv(_pose(d.xpos[l6], d.xmat[l6])) @ _pose(d.site_xpos[sid], d.site_xmat[sid])
        text, joints = arm_urdf(mjm, side)
        f = Path(tempfile.mkdtemp()) / f"{side}_yam.urdf"
        f.write_text(text)
        self.robot = Manipulator(str(f), joint_ordering=joints, ee_offset=T_l6_g)
        solver = DiffIKSolver(self.robot, [EETask(self.robot, gain_pos=GAIN, gain_rot=GAIN)],
                              [JointLimits(self.robot)], damping=1e-3)
        lo, hi = jnp.asarray(self.robot.actuated_joint_lower_limits), jnp.asarray(self.robot.actuated_joint_upper_limits)
        robot = self.robot

        def one(q0, pos, rmat):
            state, _ = solver.converge(solver.init(q0), {"ee_pos": pos, "ee_rmat": rmat},
                                       MAX_ITERS, DT_MAX, 1e-5)
            q = jnp.clip(state.q, lo, hi)  # constraints are l1-relaxed: enforce exactly
            T = robot.ee_transform(q)
            perr = jnp.linalg.norm(T[:3, 3] - pos)
            cos = (jnp.trace(T[:3, :3].T @ rmat) - 1.0) / 2.0
            return q, perr, jnp.arccos(jnp.clip(cos, -1.0, 1.0))

        D = len(jax.devices())
        pm = jax.pmap(jax.vmap(one))  # one convergence loop per CPU device

        def solve(q0, pos, rmat):
            """[N,...] -> (q [N,6], pos_err [N], rot_err [N]); N is padded to a multiple of D."""
            n = q0.shape[0]
            pad = (-n) % D
            if pad:
                rep = lambda x: jnp.concatenate([x, jnp.repeat(x[:1], pad, axis=0)])  # noqa: E731
                q0, pos, rmat = rep(q0), rep(pos), rep(rmat)
            split = lambda x: x.reshape((D, -1) + x.shape[1:])  # noqa: E731
            out = pm(split(q0), split(pos), split(rmat))
            return tuple(o.reshape((-1,) + o.shape[2:])[:n] for o in out)

        self.solve = solve
        self.fk = jax.jit(jax.vmap(robot.ee_transform))  # base-frame grasp pose

    def grasp_world(self, q: np.ndarray) -> np.ndarray:
        return self.T_w_base @ np.asarray(self.fk(jnp.asarray(q, float)[None])[0])


def _rotvecs_to_mats(v: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(v).as_matrix()


class TcpIK:
    """frax IK for the arms of one MuJoCo scene (``sides``: 'left' / 'right')."""

    def __init__(self, mjm: mujoco.MjModel, mjd: mujoco.MjData, sides) -> None:
        d = mujoco.MjData(mjm)
        d.qpos[:] = mjd.qpos
        mujoco.mj_forward(mjm, d)
        self.arms = {s: _ArmIK(mjm, d, s) for s in sides}
        self.stats = {"rows": 0, "solved": 0}

    def close(self) -> None:
        pass

    def site_world(self, side: str, q: np.ndarray) -> np.ndarray:
        """Grasp-site poses [N,4,4] in the world frame for joint configs q [N,6]."""
        arm = self.arms[side]
        return arm.T_w_base @ np.asarray(arm.fk(jnp.asarray(np.asarray(q, np.float64))))

    def solve_deltas(self, side: str, q: np.ndarray, deltas: np.ndarray, cap: np.ndarray):
        """q [N,6] current joint targets, deltas [N,6] TCP motions -> (dq [N,6], ok [N]).

        Failed rows get dq = 0 and ok = False; all-zero deltas are trivially ok.
        """
        arm = self.arms[side]
        q = np.asarray(q, np.float64)
        deltas = np.asarray(deltas, np.float64)
        cap = np.asarray(cap, np.float64)
        N = q.shape[0]
        dq, ok = np.zeros((N, 6)), np.zeros(N, bool)
        zero = ~np.any(deltas != 0.0, axis=1)
        ok[zero] = True
        T0 = np.asarray(arm.fk(jnp.asarray(q)))  # [N,4,4] current grasp pose, base frame
        # the delta is a world-frame motion; express it in the base frame (base is fixed)
        Rb = arm.T_base_w[:3, :3]
        # One solve per row, no retries: a pose that cannot be reached (or violates the
        # optional joint cap) is just an invalid control, like any other constraint violation.
        d = deltas
        pos = T0[:, :3, 3] + d[:, :3] @ Rb.T
        Rd = Rb @ _rotvecs_to_mats(d[:, 3:6]) @ Rb.T  # world rotation, in base coordinates
        rmat = Rd @ T0[:, :3, :3]
        qn, perr, rerr = (np.asarray(x) for x in arm.solve(jnp.asarray(q), jnp.asarray(pos), jnp.asarray(rmat)))
        step = qn - q
        good = ~zero & (perr < POS_TOL) & (rerr < ROT_TOL) & np.all(np.abs(step) <= cap, axis=1)
        dq[good], ok[good] = step[good], True
        self.stats["rows"] += int(N)
        self.stats["solved"] += int(ok.sum())
        return dq, ok
