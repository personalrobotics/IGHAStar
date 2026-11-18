"""IGHA* problem for UWLab's OmniReset UR5e peg insertion (Isaac Sim / Isaac Lab 3.0).

UNREVIEWED: written by Claude, not yet read or reviewed by a human. Treat as experimental.

Node state (float32, env-origin-relative coordinates, so every env agrees):

    [ features F=10 (hashed) | grip_cmd (1) | joint_pos (J) | joint_vel (J) | peg pose (7) | peg vel (6) ]

* features: peg position (3), peg long-axis direction (3), hand reference point (3), grip_cmd.
* the hole is kinematic and fixed for one planning problem; it is not part of the node.
* joint_pos/vel hold the 6 arm + 6 gripper joints.

Control (one per edge, ``NCTRL`` values): a delta of the HAND reference point's pose (the gripper's pinch
point) in the robot base frame, in metres and radians per policy step, held for ``steps_per_edge`` policy
steps (each step re-targets "current hand pose + delta"); a gripper command; and a target column used by the
biased rows:

    [dx dy dz [m]  rx ry rz [rad, rotation vector] | grip | tgt_kind tgt_id ]

The OSC controller tracks the arm's wrist_3_link, not the hand point, so each policy step the delta is converted
to the wrist delta that moves the hand point by exactly (dx dy dz) and rotates it by the rotation vector: a pure
rotation turns the hand about the hand point instead of swinging it on the lever arm (see ``_wrist_action``).
    grip: +1 open, -1 close, 0 keep the node's gripper command
    tgt_kind: 0 random row, 1 move toward a point on the peg, 2 move toward the hole

State-dependent rows (kind 1/2) are resolved into an action inside ``dynamics`` from the node's
own state, the same trick the abc example uses for grasp rows.

One policy step is 100 ms (sim dt 1/120 s, decimation 12); an edge is 5 steps = 0.5 s.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from fast_sim import FastPhysics
from rot6d import quat_to_rot6d

NCTRL = 9


@dataclass
class PegConfig:
    steps_per_edge: int = 5  # policy steps (0.1 s each) per IGHA* edge
    num_controls: int = 64
    n_peg_rows: int = 8  # of num_controls: move the hand toward a random point on the peg
    n_hole_rows: int = 8  # ... toward the hole
    peg_grasped_dist: float = 0.06  # gripper closed and hand this close to the peg: counts as holding it
    # Sampling range of the hand-pose delta per policy step: uniform in [-max, +max] per axis. Scalar or 3 values.
    # xyz in metres, rot in radians (rotation vector components in the base frame, as the OSC action uses). An edge
    # holds one delta for steps_per_edge policy steps. These also set the heuristic's speed bound (xyz).
    max_delta_xyz: Any = 0.1
    max_delta_rot: Any = 0.1
    hole_jitter: float = 0.01  # lateral jitter [m] of the hole target
    hole_height: float = 0.04  # target heights above the assembled pose are U(0, hole_height)
    # Fraction of the wrist lever-arm compensation applied with a rotation command (see _wrist_action). None = derive it
    # from the OSC gains (realised fraction per policy step ~ Kp/Kd*dt, each capped at 1); a number overrides that.
    lever_gain: Any = None
    # How an edge's per-step hand delta reaches the OSC. "relative": each policy step commands "current pose + delta"
    # (the OSC re-anchors on wherever the arm got to, so lag is never made up). "absolute": the planner computes the
    # T pose targets of the edge up front (start pose + k * delta, k = 1..T, exact hand-point geometry) and each step
    # sends the action that makes the OSC's target exactly the k-th one; lag carries over to the next step's error.
    osc_targets: str = "relative"
    max_joint_vel: float = 20.0  # validity
    # validity: an edge is invalid if the peg's acceleration between consecutive policy steps exceeds this many g.
    # Free fall is 1 g; ordinary edges stayed below 0.6 g in a replay of 512 edges, blow-ups (>0.2 m peg step) were all above it.
    max_peg_accel_g: float = 1.1
    fast_step: bool = True  # lean policy step + CUDA-graphed OSC controller (fast_sim.py); False = stock env.step
    edge_graph: bool = True  # Newton with step_graph: replay the whole edge as one CUDA graph
    log_tree: bool = False  # record every expansion (edge paths, link poses) for viser_peg_tree.py
    table_z_margin: float = 0.05
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        known = {k: d.pop(k) for k in list(d) if k in cls.__dataclass_fields__}
        return cls(**known)


OSC_KEYS = ("motion_stiffness", "motion_damping_ratio", "torque_limit")
# Split keys: (field, 3-value block). kp/damping are task space (x y z | rx ry rz); torque limit is per joint
# (shoulder_pan shoulder_lift elbow | wrist_1 wrist_2 wrist_3), clamped after tau = J^T F.
OSC_SPLIT_KEYS = {
    "kp_translation": ("motion_stiffness", 0), "kp_rotation": ("motion_stiffness", 1),
    "damping_ratio_translation": ("motion_damping_ratio", 0), "damping_ratio_rotation": ("motion_damping_ratio", 1),
    "torque_limit_base_joints": ("torque_limit", 0), "torque_limit_wrist_joints": ("torque_limit", 1),
}


def apply_osc_config(env_cfg, osc: dict | None) -> None:
    """Override the arm OSC action term's gains from the config's ``osc`` section (before gym.make).
    Either the split keys in OSC_SPLIT_KEYS (one scalar each, or 3 values for that block), or the raw 6-value term
    fields motion_stiffness / motion_damping_ratio / torque_limit (or one scalar). Unset values keep the task's."""
    if not osc:
        return
    bad = set(osc) - set(OSC_KEYS) - set(OSC_SPLIT_KEYS)
    if bad:
        raise KeyError(f"unknown osc keys {sorted(bad)}; allowed {sorted(OSC_SPLIT_KEYS)} or {OSC_KEYS}")
    arm = env_cfg.actions.arm
    for k, v in osc.items():
        if k in OSC_KEYS:
            v = [float(v)] * 6 if isinstance(v, (int, float)) else [float(x) for x in v]
            if len(v) != 6:
                raise ValueError(f"osc.{k} needs 6 values, got {len(v)}")
            setattr(arm, k, tuple(v))
    for k, v in osc.items():
        if k in OSC_SPLIT_KEYS:
            field, blk = OSC_SPLIT_KEYS[k]
            v = [float(v)] * 3 if isinstance(v, (int, float)) else [float(x) for x in v]
            if len(v) != 3:
                raise ValueError(f"osc.{k} needs one value or 3, got {len(v)}")
            cur = list(getattr(arm, field))
            cur[3 * blk:3 * blk + 3] = v
            setattr(arm, field, tuple(cur))


# OmniReset's training domain randomization (BaseEventCfg): per-env friction (robot, peg, hole, table) and mass at
# startup, gripper actuator gains at every reset. Left on, every Isaac env of the planner has different physics, so an
# edge's outcome depends on which env slot it lands in (and no single CPU model can match the GPU rows).
RANDOMIZATION_EVENTS = ("robot_material", "insertive_object_material", "receptive_object_material", "table_material",
                        "randomize_robot_mass", "randomize_insertive_object_mass", "randomize_receptive_object_mass",
                        "randomize_table_mass", "randomize_gripper_actuator_parameters")


def disable_domain_randomization(env_cfg) -> list[str]:
    """Turn OmniReset's material / mass / gripper-gain randomization events off (nominal physics in every env)."""
    off = []
    for name in RANDOMIZATION_EVENTS:
        if getattr(env_cfg.events, name, None) is not None:
            setattr(env_cfg.events, name, None)
            off.append(name)
    return off


ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


def apply_arm_sysid(env, enabled: bool, friction_scale: float = 1.0) -> None:
    """Write the robot's nominal sysid joint dynamics (armature, static/dynamic/viscous friction, from metadata.yaml
    next to its USD) to all envs, as UWLab's randomize_arm_from_sysid_fixed does at scale 1.0 (no randomisation, no
    motor delay). These are what the stiff eval OSC gains were matched to. friction_scale multiplies the static, dynamic
    and viscous friction (not the armature), standing in for the UR's own friction compensation. Call after env.reset()."""
    if not enabled:
        return
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils as ou

    robot = env.scene["robot"]
    sysid = ou.read_metadata_from_usd_directory(robot.cfg.spawn.usd_path)["sysid"]
    ids = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
    N, dev = env.num_envs, env.device

    def rows(v):
        return torch.tensor(v, device=dev, dtype=torch.float32)[None].expand(N, -1).contiguous()

    st = rows(sysid["static_friction"]) * friction_scale
    robot.write_joint_armature_to_sim_index(armature=rows(sysid["armature"]), joint_ids=ids)
    robot.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=st,
                                                        joint_dynamic_friction_coeff=st * rows(sysid["dynamic_ratio"]),
                                                        joint_viscous_friction_coeff=rows(sysid["viscous_friction"]) * friction_scale,
                                                        joint_ids=ids)
    print(f"[peg] arm sysid applied: armature {sysid['armature']}, friction scale {friction_scale} of static "
          f"{sysid['static_friction']}, viscous {sysid['viscous_friction']}")


def _quat_axis(q: torch.Tensor, axis: int) -> torch.Tensor:
    """Column ``axis`` of the rotation matrix of quaternions q [...,4] (xyzw)."""
    from isaaclab.utils import math as mu

    e = torch.zeros(q.shape[:-1] + (3,), device=q.device, dtype=q.dtype)
    e[..., axis] = 1.0
    return mu.quat_apply(q, e)


class PegProblem:
    # hashed state: hand pose (position + 6D rotation = first two columns of the hand's rotation matrix),
    # peg position, peg z-axis, gripper command. The node state continues with joint/peg-pose data used only
    # to restore the simulator exactly (the peg pose stays a quaternion: that is Isaac's interface).
    feature_names = (
        "hand_x", "hand_y", "hand_z", "hand_r1", "hand_r2", "hand_r3", "hand_r4", "hand_r5", "hand_r6",
        "peg_x", "peg_y", "peg_z", "peg_axis_x", "peg_axis_y", "peg_axis_z", "grip",
    )
    i_hand, i_rot6, i_peg, i_axis, i_grip = slice(0, 3), slice(3, 9), slice(9, 12), slice(12, 15), 15

    def __init__(self, env, cfg: PegConfig, task_cfg: dict[str, Any] | None = None) -> None:
        from isaaclab.utils import math as mu  # noqa: F401  (import after the app started)

        self.env, self.cfg, self.task_cfg = env, cfg, dict(task_cfg or {})
        self.device = env.device
        sc = env.scene
        self.scene = sc
        self.robot, self.peg, self.hole = sc["robot"], sc["insertive_object"], sc["receptive_object"]
        self.N = env.num_envs
        self.T = cfg.steps_per_edge
        self.J = self.robot.num_joints
        self.F = len(self.feature_names)
        self.o_grip = self.F
        self.o_q = self.o_grip + 1
        self.o_qd = self.o_q + self.J
        self.o_peg = self.o_qd + self.J
        self.o_pegv = self.o_peg + 7
        self.n = self.o_pegv + 6
        self.edge_dt = self.T * env.step_dt
        self.gen = torch.Generator().manual_seed(cfg.seed)

        arm = env.action_manager.get_term("arm")
        self.arm = arm
        self.scale = arm._scale.reshape(-1, 6)[0].clone()  # [6] env action units -> metres / radians (only to build raw env actions)
        self.d_xyz = self._per_axis(cfg.max_delta_xyz, "max_delta_xyz")
        self.d_rot = self._per_axis(cfg.max_delta_rot, "max_delta_rot")
        self.hand_body = self.robot.find_bodies("robotiq_base_link")[0][0]
        ee_func = env.reward_manager.get_term_cfg("ee_asset_distance").func
        self.ee_offset = ee_func.root_asset_offset
        prog = env.reward_manager.get_term_cfg("progress_context").func
        self.prog = prog
        self.fast = FastPhysics(env) if cfg.fast_step else None
        self.newton = type(env.cfg.sim.physics).__name__ == "NewtonCfg"
        if self.newton and self.fast is not None and not getattr(env.cfg.sim.physics, "use_cuda_graph", True):
            # Newton's own per-physics-step graph is off: replay one graph per policy step instead (fast_sim.py)
            from fast_sim import NewtonGraphedStep
            self.fast = NewtonGraphedStep(self.fast)
        self.edge_graph = None  # set at the end of __init__ (needs self.n etc.)
        self.peg_offset, self.hole_offset = prog.insertive_asset_offset, prog.receptive_asset_offset
        cmd = env.command_manager.get_term("task_command")
        self.pos_thr = float(cmd.success_position_threshold)
        self.rot_thr = float(cmd.success_orientation_threshold)
        self.env_origins = sc.env_origins.clone()
        self.lever = self._hand_lever()
        # The OSC tracks "current + delta" with a PD law, so each policy step it realises only kp/kd*step_dt of the
        # commanded translation (0.236 here) but of the commanded rotation a different fraction (0.087 here). The
        # lever-arm translation that cancels the hand point's swing is therefore scaled by realised_rot/realised_xyz;
        # with the full lever the wrist would translate 2.7x more than the rotation turns it and the hand would
        # swing the other way.
        # With stiff gains Kp/Kd*dt exceeds 1 (the step is fully realised), hence the cap.
        kp, kd = arm._kp[0].clone(), arm._kd[0].clone()
        rho = (kp / kd * env.step_dt).clamp(max=1.0)
        if float(rho[:3].max() - rho[:3].min()) > 1e-6 or float(rho[3:].max() - rho[3:].min()) > 1e-6:
            raise RuntimeError(f"per-axis OSC realised fractions differ ({rho.tolist()}); lever compensation assumes uniform ones")
        self.lever_gain = float(rho[3] / rho[0]) if self.cfg.lever_gain is None else float(self.cfg.lever_gain)
        print(f"[peg] OSC Kp {[round(float(x), 1) for x in kp]} Kd {[round(float(x), 2) for x in kd]} "
              f"torque limit {[round(float(x), 1) for x in arm._torque_max.flatten()[:6]] if hasattr(arm, '_torque_max') else '?'}; "
              f"lever_gain {self.lever_gain:.3f}" + (" (from gains)" if self.cfg.lever_gain is None else " (config)"))
        self.joint_lo = self.robot.data.soft_joint_pos_limits.torch[0, :, 0].clone() if hasattr(
            self.robot.data, "soft_joint_pos_limits") else None

        # template of the whole scene state (static entities stay as they are)
        self.template = {k: {n: {f: v.clone() for f, v in d.items()} for n, d in grp.items()}
                         for k, grp in sc.get_state(is_relative=True).items()}
        self.base_quat = self.template["articulation"]["robot"]["root_pose"][0, 3:7].clone()
        self.table_top_z = float(self.template["rigid_object"]["table"]["root_pose"][0, 2]) + 0.0
        self.peg_local_box = self._peg_local_box()
        self.hole_pose = None  # [7] set by set_start
        self.start = None
        # Speed bounds for the heuristic come from the configured hand-delta range: the hand target is at most
        # |max_delta| from the current pose each policy step, so the speed is bounded by |max_delta| / step_dt
        # (m/s, rad/s); per edge that is times edge_dt. The run logs the largest observed step against it.
        self.max_hand_speed = float(torch.linalg.norm(self.d_xyz)) / env.step_dt  # m/s
        self.max_rot_speed = float(torch.linalg.norm(self.d_rot)) / env.step_dt  # rad/s
        self.max_hand_step = self.max_hand_speed * self.edge_dt  # m / edge
        self.max_object_step = self.max_hand_step
        self.max_rot_step = self.max_rot_speed * self.edge_dt  # rad / edge
        self.profile = {"dyn_calls": 0, "dyn_rows": 0, "dyn_time": 0.0, "max_peg_step": 0.0,
                        "max_hand_step": 0.0, "bad_rows": 0}
        self.flung = []  # (state[n], control[NCTRL], peg_step, bad) of edges with peg step > 0.1 m or bad; for contact replays
        self.normal = []  # random ~2% sample of ordinary edges (same fields): baseline for contact replays
        self._tree = {k: [] for k in ("parent_call", "parent_child_id", "parent_feat", "child_parent", "child_feat",
                                      "child_peg", "child_bp", "child_valid", "child_cost", "child_h", "child_u")}
        self._tree_index: dict[int, int] = {}
        self._n_children = 0
        # closest approach of the peg to the goal over all valid simulated edge end states (= search nodes):
        # [peg-to-hole distance m, |roll|+|pitch| rad] of the nearest by distance, of the nearest by angle, and of the
        # nearest by distance among states already inside the angle threshold; plus best-so-far distance per call.
        self.closest = {"dist": (float("inf"), float("nan"), -1), "angle": (float("nan"), float("inf"), -1),
                        "dist_in_angle": (float("inf"), float("nan"), -1)}
        self.closest_state = {}
        self.closest_curve = []
        self.start_bp = None  # [n_bodies,7] link poses (env frame) of the start node, set by set_start
        self.diag = []  # per simulated edge: [peak |qd|/limit over robot joints, peg step (m), bad(1/0), peak peg speed m/s]
        if self.newton and self.fast is not None and type(self.fast).__name__ == "NewtonGraphedStep" and cfg.edge_graph:
            from fast_sim import NewtonGraphedEdge
            self.edge_graph = NewtonGraphedEdge(self, self.fast)  # whole edge as one CUDA graph (fast_sim.py)

    # ------------------------------------------------------------------ hand-point deltas
    def _per_axis(self, v, name: str) -> torch.Tensor:
        t = torch.as_tensor(v, dtype=torch.float32).flatten()
        if t.numel() == 1:
            t = t.expand(3).clone()
        if t.numel() != 3 or not bool((t > 0).all()):
            raise ValueError(f"'{name}' must be one positive number or three, got {v}")
        return t.to(self.device)

    def _hand_lever(self) -> torch.Tensor:
        """Hand reference point relative to the OSC-controlled body (wrist_3_link), in that body's frame [3]."""
        from isaaclab.utils import math as mu

        d = self.robot.data
        idx = self.arm._ee_body_idx
        w_pos = d.body_link_pos_w.torch[:, idx].view(-1, 3)
        w_quat = d.body_link_quat_w.torch[:, idx].view(-1, 4)
        h_pos, _ = self._hand_pose()
        lever = mu.quat_apply_inverse(w_quat, h_pos + self.env_origins - w_pos)
        spread = float((lever - lever[0]).abs().max())
        if spread > 1e-4:  # rigid link offset: must agree across envs
            raise RuntimeError(f"hand point is not rigidly attached to the controlled body (spread {spread:.2e} m)")
        return lever[0].clone()

    def _wrist_action(self, a: torch.Tensor, t: int | None = None) -> torch.Tensor:
        """Hand-delta actions a [B,7] at policy step t of the edge -> raw env actions [B,7] (see osc_targets)."""
        if self.cfg.osc_targets == "absolute":
            return self._wrist_action_absolute(a, t)
        return self._wrist_action_relative(a)

    def _wrist_action_absolute(self, a: torch.Tensor, t: int) -> torch.Tensor:
        """Precomputed targets. At t == 0 the edge's start pose (controlled body, base frame) is stored; the target of
        step t is the hand point moved by (t+1)*dp and the hand rotated by (t+1)*rot (rotation vector, base frame, on
        the left), converted to the wrist pose through the rigid lever. The returned action is target - current pose
        (rotation: axis-angle of q_target q_current^-1), which the OSC adds back to the same current pose, so its desired
        pose is exactly the target. Pure tensor ops (CUDA-graph safe); t is a Python int (the edge loop is unrolled)."""
        from isaaclab.utils import math as mu

        if t is None:
            raise ValueError("osc_targets=absolute needs the policy-step index t")
        p, q = self.arm._get_ee_pose_root_frame()
        B = a.shape[0]
        lev = self.lever[None].expand(B, -1)
        if t == 0:
            self._edge_p0 = p + mu.quat_apply(q, lev)  # hand point at the edge start
            self._edge_q0 = q
        k = float(t + 1)
        rot = a[:, 3:6] * k
        ang = rot.norm(dim=1, keepdim=True)
        axis = torch.where(ang > 1e-6, rot / torch.where(ang > 1e-6, ang, torch.ones_like(ang)), torch.zeros_like(rot))
        dq = torch.cat([axis * torch.sin(ang / 2), torch.cos(ang / 2)], dim=1)
        q_t = mu.quat_mul(dq, self._edge_q0)
        p_t = self._edge_p0 + k * a[:, :3] - mu.quat_apply(q_t, lev)  # wrist position for the target hand pose
        d_rot = mu.axis_angle_from_quat(mu.quat_mul(q_t, mu.quat_inv(q)))
        return torch.cat([(p_t - p) / self.scale[:3], d_rot / self.scale[3:6], a[:, 6:7]], dim=1)

    def _wrist_action_relative(self, a: torch.Tensor) -> torch.Tensor:
        """Hand-point deltas a [B,7] (dx dy dz [m], rotation vector [rad], grip; base frame) -> raw env actions [B,7].

        The OSC target is "controlled-body pose + delta", then rotation-vector delta composed on the left. To move the
        hand point p_h = p_w + R_w l by dp and rotate it by Rd (hand and wrist rotate together), the wrist must go
        to p_h + dp - Rd R_w l, i.e. a wrist translation of dp + R_w l - Rd R_w l; the lever part is scaled by
        ``lever_gain`` (realised-rotation / realised-translation fraction of the controller, see __init__). Uses the
        live pose, so call once per policy step. Raw action = delta / env scale (no clipping in this env's action term).
        """
        from isaaclab.utils import math as mu

        _, q = self.arm._get_ee_pose_root_frame()  # controlled body in the base frame
        r = mu.quat_apply(q, self.lever[None].expand(a.shape[0], -1))
        rot = a[:, 3:6]
        ang = rot.norm(dim=1, keepdim=True)
        axis = torch.where(ang > 1e-6, rot / torch.where(ang > 1e-6, ang, torch.ones_like(ang)), torch.zeros_like(rot))
        dq = torch.cat([axis * torch.sin(ang / 2), torch.cos(ang / 2)], dim=1)  # same construction as the action term
        dp_wrist = a[:, :3] + self.lever_gain * (r - mu.quat_apply(dq, r))
        return torch.cat([dp_wrist / self.scale[:3], rot / self.scale[3:6], a[:, 6:7]], dim=1)

    # ------------------------------------------------------------------ geometry
    def _peg_local_box(self) -> torch.Tensor:
        """Peg bounding box [2,3] in its own frame (from its USD prim)."""
        try:
            import omni.usd
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath("/World/envs/env_0/InsertiveObject")
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                                                              UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide])
            rng = cache.ComputeRelativeBound(prim, prim).ComputeAlignedRange()
            lo, hi = np.array(rng.GetMin(), float), np.array(rng.GetMax(), float)
            box = torch.tensor(np.stack([lo, hi]), dtype=torch.float32)
            if torch.isfinite(box).all() and (box[1] > box[0]).all():
                return box
        except Exception as e:  # noqa: BLE001
            print(f"[peg_problem] peg bounding box from USD failed ({e}); using a 2 cm cube")
        return torch.tensor([[-0.01] * 3, [0.01] * 3])

    # ------------------------------------------------------------------ state <-> sim
    def _read_states(self, grip_cmd: torch.Tensor) -> torch.Tensor:
        """Pack the sim's current state for all N envs -> [N,n] (device)."""
        st = self.scene.get_state(is_relative=True)
        jp = st["articulation"]["robot"]["joint_position"]
        jv = st["articulation"]["robot"]["joint_velocity"]
        pp = st["rigid_object"]["insertive_object"]["root_pose"]
        pv = st["rigid_object"]["insertive_object"]["root_velocity"]
        hand, hand_q = self._hand_pose()
        feats = torch.cat([hand, quat_to_rot6d(hand_q), pp[:, :3], _quat_axis(pp[:, 3:7], 2), grip_cmd[:, None]], dim=1)
        return torch.cat([feats, grip_cmd[:, None], jp, jv, pp, pv], dim=1)

    def _body_poses(self) -> torch.Tensor:
        """Link poses [N,n_bodies,7] (pos, quat xyzw) in each env's frame, for visualisation."""
        d = self.robot.data
        pos = d.body_link_pos_w.torch - self.env_origins[:, None]
        return torch.cat([pos, d.body_link_quat_w.torch], dim=-1)

    def _hand_pose(self):
        """Hand reference frame (the gripper's pinch point), env-relative: position [N,3], quaternion [N,4] xyzw."""
        d = self.robot.data
        from isaaclab.utils import math as mu

        bpos = d.body_link_pos_w.torch[:, self.hand_body].view(-1, 3)
        if getattr(self, "_ee_off", None) is None:  # Offset.combine builds its tensors from Python lists every call
            self._ee_off = (torch.tensor(self.ee_offset.pos, device=bpos.device, dtype=bpos.dtype),  # (a host copy,
                            torch.tensor(self.ee_offset.quat, device=bpos.device, dtype=bpos.dtype))  # not graphable)
        n = bpos.shape[0]
        pos, quat = mu.combine_frame_transforms(bpos, d.body_link_quat_w.torch[:, self.hand_body].view(-1, 4),
                                                self._ee_off[0].expand(n, -1), self._ee_off[1].expand(n, -1))
        return pos - self.env_origins, quat

    def _write_states(self, s: torch.Tensor) -> None:
        """Write node states s [N,n] (device) into the N envs."""
        st = {k: {n: {f: v.clone() for f, v in d.items()} for n, d in grp.items()}
              for k, grp in self.template.items()}
        st["articulation"]["robot"]["joint_position"] = s[:, self.o_q:self.o_q + self.J].contiguous()
        st["articulation"]["robot"]["joint_velocity"] = s[:, self.o_qd:self.o_qd + self.J].contiguous()
        st["rigid_object"]["insertive_object"]["root_pose"] = s[:, self.o_peg:self.o_peg + 7].contiguous()
        st["rigid_object"]["insertive_object"]["root_velocity"] = s[:, self.o_pegv:self.o_pegv + 6].contiguous()
        st["rigid_object"]["receptive_object"]["root_pose"] = self.hole_pose[None].expand(s.shape[0], -1).contiguous()
        self.scene.reset_to(st, env_ids=None, is_relative=True)
        self.env.scene.write_data_to_sim()
        self.env.sim.forward() if hasattr(self.env.sim, "forward") else None
        self.scene.update(self.env.physics_dt)
        if self.newton:  # MuJoCo-Warp keeps last step's body velocities; recompute them from the written state
            import newton_robot_usd
            newton_robot_usd.refresh_velocity_kinematics()

    def set_start(self, scene_state: dict, env_idx: int = 0) -> torch.Tensor:
        """Take env ``env_idx``'s current scene state as the start node; remember the hole pose."""
        st = self.scene.get_state(is_relative=True)
        self.hole_pose = st["rigid_object"]["receptive_object"]["root_pose"][env_idx].clone()
        grip = self.robot.data.joint_pos.torch[env_idx, 6]
        cmd = torch.where(grip > 0.4, -1.0, 1.0).reshape(1)  # closed finger -> last command was "close"
        s = self._read_states(cmd.expand(self.N).clone())[env_idx:env_idx + 1]
        self.start = s[0].cpu()
        self.start_bp = self._body_poses()[env_idx].cpu().clone()
        return self.start

    # ------------------------------------------------------------------ success / heuristic
    def alignment(self, peg_pose: torch.Tensor):
        """OmniReset's ProgressContext quantities for peg poses [B,7] (env-relative)."""
        from isaaclab.utils import math as mu

        dev = peg_pose.device
        hp = self.hole_pose.to(dev)[None].expand(peg_pose.shape[0], -1)
        pa_p, pa_q = self.peg_offset.combine(peg_pose[:, :3], peg_pose[:, 3:7])
        ha_p, ha_q = self.hole_offset.combine(hp[:, :3], hp[:, 3:7])
        rel_p, rel_q = mu.subtract_frame_transforms(ha_p, ha_q, pa_p, pa_q)
        ex, ey, _ = mu.euler_xyz_from_quat(rel_q)
        euler_xy = mu.wrap_to_pi(ex).abs() + mu.wrap_to_pi(ey).abs()
        return torch.norm(rel_p, dim=1), euler_xy

    def goal_mask(self, states: torch.Tensor) -> torch.Tensor:
        dist, ang = self.alignment(states[:, self.o_peg:self.o_peg + 7].to(self.device, torch.float32))
        return ((dist < self.pos_thr) & (ang < self.rot_thr)).cpu()

    def goal_test(self, state: torch.Tensor) -> bool:
        return bool(self.goal_mask(state[None])[0])

    def heuristic(self, states: torch.Tensor) -> torch.Tensor:
        """Admissible: remaining position gap / (TCP-speed bound per edge) * edge_dt.

        Where the peg is already moving faster than the bound (flung), its own speed
        per edge replaces the bound.
        """
        dist, ang = self.alignment(states[:, self.o_peg:self.o_peg + 7].to(self.device, torch.float32))
        gap = (dist - self.pos_thr).clamp(min=0.0).cpu()
        v = states[:, self.o_pegv:self.o_pegv + 3].norm(dim=1) * self.edge_dt
        h_pos = gap / torch.clamp(v, min=self.max_object_step) * self.edge_dt
        # Orientation: success needs |roll|+|pitch| (peg in the hole frame) under rot_thr. The hand
        # rotates at most max_rot_step rad per edge, and |roll|+|pitch| changes by at most
        # ~2x the rotation angle, so this many edges are needed at least. (A peg that tips over
        # under gravity or a contact flick can rotate faster; not covered, as for the flung-peg
        # position case it is checked empirically.)
        gap_r = (ang - self.rot_thr).clamp(min=0.0).cpu()
        h_rot = gap_r / (2.0 * self.max_rot_step) * self.edge_dt
        return torch.maximum(h_pos, h_rot)

    # ------------------------------------------------------------------ controls
    def sample_controls(self) -> torch.Tensor:
        """[K, NCTRL]: 6 hand-delta dims uniform in [-max_delta_xyz, +max_delta_xyz] (m) / [-max_delta_rot, +max_delta_rot]
        (rad) and the gripper command uniform in [-1,1] (gripper < 0 closes); the last
        n_peg_rows + n_hole_rows rows replace their translation by a step toward a random point on the peg / toward
        the hole (kind 1 / 2, seeded by column 8); their rotation and gripper stay as sampled."""
        c, g, K = self.cfg, self.gen, self.cfg.num_controls
        u = torch.zeros(K, NCTRL)
        u[:, :7] = torch.rand(K, 7, generator=g) * 2 - 1
        u[:, :3] *= self.d_xyz.cpu()
        u[:, 3:6] *= self.d_rot.cpu()
        n_p, n_h = c.n_peg_rows, c.n_hole_rows
        u[K - n_p - n_h:K - n_h, 7] = 1.0
        u[K - n_h:, 7] = 2.0
        u[K - n_p - n_h:, 8] = torch.randint(0, 2 ** 30, (n_p + n_h,), generator=g).float()  # seed for the point
        return u

    def _resolve(self, s: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Node states s [B,n], controls u [B,NCTRL] (device) -> hand-delta actions [B,7] (m, rad, grip), grip cmd [B]."""
        from isaaclab.utils import math as mu

        B = s.shape[0]
        a = torch.zeros(B, 7, device=self.device)
        a[:, :6] = u[:, :6]
        grip_prev = s[:, self.o_grip]
        cmd = torch.where(u[:, 6] < 0, -torch.ones_like(u[:, 6]), torch.ones_like(u[:, 6]))
        kind = u[:, 7].long()
        peg_pose, hand = s[:, self.o_peg:self.o_peg + 7], s[:, self.i_hand]
        if (kind > 0).any():
            gen = torch.Generator(device="cpu")
            # deterministic per-row point from the row's own seed column
            rnd = torch.stack([torch.rand(3, generator=gen.manual_seed(int(x))) for x in u[:, 8].long().cpu()]
                              ).to(self.device) if B else torch.zeros(0, 3, device=self.device)
            box = self.peg_local_box.to(self.device)
            # target 1: random point of the peg's bounding box, in world (env-relative) coordinates
            p_local = box[0] + rnd * (box[1] - box[0])
            p_peg = mu.quat_apply(peg_pose[:, 3:7], p_local) + peg_pose[:, :3]
            # target 2: the hole's assembled pose, jittered laterally and lifted by a random height
            hp = self.hole_pose.to(self.device)[None].expand(B, -1)
            h_pos, h_quat = self.hole_offset.combine(hp[:, :3], hp[:, 3:7])
            jit = (rnd[:, :2] - 0.5) * 2 * self.cfg.hole_jitter
            p_hole = h_pos + torch.cat([jit, rnd[:, 2:3] * self.cfg.hole_height], dim=1)
            # a peg held in the gripper moves with the hand: steer the peg's alignment point instead
            pa_pos, pa_quat = self.peg_offset.combine(peg_pose[:, :3], peg_pose[:, 3:7])
            held = (grip_prev < 0) & ((hand - peg_pose[:, :3]).norm(dim=1) < self.cfg.peg_grasped_dist)
            ref = torch.where(held[:, None], pa_pos, hand)
            target = torch.where((kind == 1)[:, None], p_peg, p_hole)
            delta_w = target - ref
            self._dbg = {"target": target, "ref": ref, "delta_w": delta_w, "kind": kind}
            # world -> robot base frame (the OSC action's frame); the base is fixed
            qb = self.base_quat.to(self.device)[None].expand(B, -1)
            delta_b = mu.quat_apply_inverse(qb, delta_w)
            # commanded hand delta per policy step toward the target, limited to the configured range
            dxyz = self.d_xyz[None]
            act = torch.maximum(torch.minimum(delta_b / self.T, dxyz), -dxyz)
            m = kind > 0
            a[m, :3] = act[m]
        a[:, 6] = cmd  # <0 closes, >=0 opens (Isaac Lab binary action)
        return a, cmd

    # ------------------------------------------------------------------ dynamics
    def _simulate_edge(self, act: torch.Tensor, cmd: torch.Tensor):
        """From the written states: T policy steps of hand-delta actions act [N,7] -> (traj [N,T,n], link poses at the
        last step [N,n_bodies,7] or None). One CUDA graph replay on Newton (edge_graph), else step by step."""
        if self.edge_graph is not None:
            return self.edge_graph.run(act, cmd)
        traj, bp = [], None
        for t in range(self.T):
            (self.fast.step if self.fast is not None else self.env.step)(self._wrist_action(act, t))
            traj.append(self._read_states(cmd))
            if self.cfg.log_tree and t == self.T - 1:
                bp = self._body_poses()
        return torch.stack(traj, dim=1), bp

    @torch.no_grad()
    def dynamics(self, states: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        """states [B,n], controls [B,NCTRL] (cpu) -> [B,T,n] (cpu); invalid rows are all-NaN."""
        t0 = time.perf_counter()
        B = states.shape[0]
        out = torch.empty(B, self.T, self.n)
        for c0 in range(0, B, self.N):
            c1 = min(B, c0 + self.N)
            nb = c1 - c0
            s = states[c0:c1].to(self.device, torch.float32)
            u = controls[c0:c1].to(self.device, torch.float32)
            if nb < self.N:  # pad with copies of the first row
                pad = self.N - nb
                s = torch.cat([s, s[:1].expand(pad, -1)])
                u = torch.cat([u, u[:1].expand(pad, -1)])
            act, cmd = self._resolve(s, u)
            self._write_states(s)
            traj, bp = self._simulate_edge(act, cmd)  # [N,T,n] (device), link poses at the last step or None
            lim = self.robot.data.joint_vel_limits.torch
            qd = traj[:, :, self.o_qd:self.o_qd + self.J].abs()
            ratio = torch.nan_to_num((qd / lim[:, None]).amax(dim=2), nan=1e9).amax(dim=1)
            bad = (~torch.isfinite(traj).all(dim=2)).any(dim=1) | (qd.amax(dim=2) > self.cfg.max_joint_vel).any(dim=1)
            v = traj[:, :, self.o_pegv:self.o_pegv + 3]
            prev_v = torch.cat([s[:, None, self.o_pegv:self.o_pegv + 3], v[:, :-1]], dim=1)
            a_max = self.cfg.max_peg_accel_g * 9.81
            acc_bad = (~((v - prev_v).norm(dim=2) / self.env.step_dt <= a_max)).any(dim=1)  # NaN also counts
            out[c0:c1] = traj[:nb].cpu()
            self.profile["accel_rows"] = self.profile.get("accel_rows", 0) + int((acc_bad & ~bad)[:nb].sum())
            bad = (bad | acc_bad)[:nb].cpu()
            pstep = torch.nan_to_num((out[c0:c1, -1, self.o_peg:self.o_peg + 3] - states[c0:c1, self.o_peg:self.o_peg + 3]).norm(dim=1), nan=1e9)
            self.diag.append(torch.stack([ratio[:nb].cpu(), pstep, bad.float()], dim=1))
            sel = ((pstep > 0.1) | bad).nonzero().flatten()
            if len(self.normal) < 1500:
                ok_rows = ((pstep <= 0.1) & ~bad & (torch.rand(nb) < 0.02)).nonzero().flatten()
                for i in ok_rows.tolist():
                    self.normal.append((states[c0 + i].clone(), controls[c0 + i].clone(), float(pstep[i]), False))
            if len(self.flung) < 3000:
                for i in sel.tolist():
                    self.flung.append((states[c0 + i].clone(), controls[c0 + i].clone(), float(pstep[i]), bool(bad[i])))
            self._track_closest(out[c0:c1, -1], bad)
            raw = out[c0:c1].clone() if self.cfg.log_tree else None  # keep blown-up edges for the viewer
            out[c0:c1][bad] = float("nan")
            if self.cfg.log_tree:
                self._log_expansion(states[c0:c1], controls[c0:c1], out[c0:c1], raw, bp[:nb].cpu())
            self.profile["bad_rows"] += int(bad.sum())
            hs = (out[c0:c1, -1, self.i_hand] - states[c0:c1, self.i_hand]).norm(dim=1)
            hs = hs[torch.isfinite(hs)]
            if hs.numel():
                self.profile["max_hand_step"] = max(self.profile["max_hand_step"], float(hs.max()))
            step = (out[c0:c1, -1, self.o_peg:self.o_peg + 3] - states[c0:c1, self.o_peg:self.o_peg + 3]).norm(dim=1)
            step = step[torch.isfinite(step)]
            if step.numel():
                self.profile["max_peg_step"] = max(self.profile["max_peg_step"], float(step.max()))
        self.profile["dyn_calls"] += 1
        self.profile["dyn_rows"] += B
        self.profile["dyn_time"] += time.perf_counter() - t0
        return out

    def _track_closest(self, fin: torch.Tensor, bad: torch.Tensor) -> None:
        """Update the closest-approach record from end states fin [b,n] (cpu, before NaN masking) and bad [b]."""
        ok = (~bad) & torch.isfinite(fin).all(dim=1)
        call = self.profile["dyn_calls"] + 1
        if ok.any():
            rows = fin[ok]
            d, a = (x.cpu() for x in self.alignment(rows[:, self.o_peg:self.o_peg + 7].to(self.device, torch.float32)))
            for key, score in (("dist", d), ("angle", a), ("dist_in_angle", torch.where(a < self.rot_thr, d, torch.inf))):
                i = int(torch.argmin(score))
                best = self.closest[key][0 if key != "angle" else 1]
                if float(score[i]) < best:
                    self.closest[key] = (float(d[i]), float(a[i]), call)
                    self.closest_state[key] = rows[i].clone()
        self.closest_curve.append(self.closest["dist"][0])

    def closest_summary(self) -> dict:
        return {f"closest_{k}": {"dist_m": v[0], "angle_rad": v[1], "call": v[2]} for k, v in self.closest.items()}

    # ------------------------------------------------------------------ tree log (for viser_peg_tree.py)
    @staticmethod
    def _key(row: torch.Tensor) -> int:
        return hash(row.contiguous().numpy().tobytes())

    def _log_expansion(self, states, controls, out, raw, bp) -> None:
        """One batched expansion: parents (rows are parent-major, K controls each) and their children's paths."""
        t = self._tree
        K = self.cfg.num_controls if states.shape[0] % self.cfg.num_controls == 0 else 1
        parents = states[::K]
        call = self.profile["dyn_calls"] + 1
        p0 = sum(len(a) for a in t["parent_feat"])
        for row in parents:
            t["parent_child_id"].append(self._tree_index.get(self._key(row), -1))
            t["parent_call"].append(call)
        t["parent_feat"].append(parents[:, :self.F].numpy().copy())
        fin = out[:, -1]
        valid = self.validity(fin) > 0
        B = out.shape[0]
        t["child_parent"].append(p0 + torch.arange(B).numpy() // K)
        t["child_feat"].append(raw[:, :, :self.F].numpy().copy())
        t["child_peg"].append(raw[:, :, self.o_peg:self.o_peg + 7].numpy().copy())
        t["child_bp"].append(bp.numpy().astype(np.float16))
        t["child_valid"].append(valid.numpy())
        t["child_u"].append(controls.numpy().astype(np.float32).copy())  # exact controls, so a plan can be replayed
        t["child_cost"].append(self.cost(states, controls, fin).numpy())
        t["child_h"].append(self.heuristic(fin).numpy())
        for i in range(B):
            if valid[i]:
                self._tree_index[self._key(fin[i])] = self._n_children + i
        self._n_children += B

    def tree_arrays(self) -> dict:
        t = self._tree
        if not t["parent_call"]:
            return {}
        out = {k: np.concatenate(v) for k, v in t.items() if k not in ("parent_call", "parent_child_id")}
        out["parent_call"] = np.asarray(t["parent_call"], np.int32)
        out["parent_child_id"] = np.asarray(t["parent_child_id"], np.int64)
        return out

    # ------------------------------------------------------------------ planner callbacks
    def cost(self, states, controls, next_states):
        return torch.full((states.shape[0],), self.edge_dt)

    def validity(self, states):
        ok = torch.isfinite(states).all(dim=1)
        z = states[:, self.o_peg + 2]
        ok &= z > (self.table_top_z - self.cfg.table_z_margin)
        return ok.float()

    def planner_config(self, planner_cfg: dict[str, Any]) -> dict[str, Any]:
        info = dict(planner_cfg)

        def hashed(key):
            v = info.get(key, self.task_cfg.get(key))
            if v is None:
                raise KeyError(f"no '{key}' in planner or task config")
            v = [float(x) for x in v]
            if len(v) != self.F:
                raise ValueError(f"'{key}' needs {self.F} values ({', '.join(self.feature_names)}), got {len(v)}")
            return v + [1.0] * (self.n - self.F)

        info.update(state_dim=self.n, control_dim=NCTRL, hash_dims=self.F, num_controls=self.cfg.num_controls,
                    resolution=hashed("resolution"), tolerance=hashed("tolerance"))
        info["node_info"] = {"node_type": "generic", "timesteps": self.T}
        return {
            "experiment_info_default": info,
            "sample_controls_fn": self.sample_controls,
            "dynamics_fn": self.dynamics,
            "cost_fn": self.cost,
            "validity_fn": self.validity,
            "heuristic_fn": self.heuristic,
            "goal_test_fn": self.goal_test,
            "goal_mask_fn": self.goal_mask,  # batched goal test for the post-search pass
        }
