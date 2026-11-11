"""IGHA* problem for UWLab's OmniReset UR5e peg insertion (Isaac Sim / Isaac Lab 3.0).

UNREVIEWED: written by Claude, not yet read or reviewed by a human. Treat as experimental.

Node state (float32, env-origin-relative coordinates, so every env agrees):

    [ features F=10 (hashed) | grip_cmd (1) | joint_pos (J) | joint_vel (J) | peg pose (7) | peg vel (6) ]

* features: peg position (3), peg long-axis direction (3), hand reference point (3), grip_cmd.
* the hole is kinematic and fixed for one planning problem; it is not part of the node.
* joint_pos/vel hold the 6 arm + 6 gripper joints.

Control (one per edge, ``NCTRL`` values): the hand pose delta in the OSC action's own units
([-1, 1] per axis, multiplied by the env's action scale and held for ``steps_per_edge`` policy
steps), a gripper command, and a target column used by the biased rows:

    [ax ay az  arx ary arz | grip | tgt_kind tgt_id ]
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

NCTRL = 9


@dataclass
class PegConfig:
    steps_per_edge: int = 5  # policy steps (0.1 s each) per IGHA* edge
    num_controls: int = 64
    n_peg_rows: int = 8  # of num_controls: move the hand toward a random point on the peg
    n_hole_rows: int = 8  # ... toward the hole
    rot_prob: float = 0.3  # fraction of random rows that also rotate
    grip_toggle_prob: float = 0.3
    peg_grasp_dist: float = 0.03  # hand within this of the chosen peg point: close the gripper
    peg_grasped_dist: float = 0.06  # gripper closed and hand this close to the peg: counts as holding it
    hole_jitter: float = 0.01  # lateral jitter [m] of the hole target
    hole_height: float = 0.04  # target heights above the assembled pose are U(0, hole_height)
    joint_motion_weight: float = 0.05
    max_joint_vel: float = 20.0  # validity
    # validity: an edge is invalid if the peg's acceleration between consecutive policy steps exceeds this many g.
    # Free fall is 1 g; ordinary edges stayed below 0.6 g in a replay of 512 edges, blow-ups (>0.2 m peg step) were all above it.
    max_peg_accel_g: float = 1.1
    log_tree: bool = False  # record every expansion (edge paths, link poses) for viser_peg_tree.py
    table_z_margin: float = 0.05
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        known = {k: d.pop(k) for k in list(d) if k in cls.__dataclass_fields__}
        return cls(**known)


def _quat_axis(q: torch.Tensor, axis: int) -> torch.Tensor:
    """Column ``axis`` of the rotation matrix of quaternions q [...,4] (xyzw)."""
    from isaaclab.utils import math as mu

    e = torch.zeros(q.shape[:-1] + (3,), device=q.device, dtype=q.dtype)
    e[..., axis] = 1.0
    return mu.quat_apply(q, e)


class PegProblem:
    feature_names = (
        "peg_x", "peg_y", "peg_z", "ax_x", "ax_y", "ax_z", "hand_x", "hand_y", "hand_z", "grip",
    )

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
        self.scale = arm._scale.clone()  # [6] metres / radians per unit action per policy step
        self.hand_body = self.robot.find_bodies("robotiq_base_link")[0][0]
        ee_func = env.reward_manager.get_term_cfg("ee_asset_distance").func
        self.ee_offset = ee_func.root_asset_offset
        prog = env.reward_manager.get_term_cfg("progress_context").func
        self.prog = prog
        self.peg_offset, self.hole_offset = prog.insertive_asset_offset, prog.receptive_asset_offset
        cmd = env.command_manager.get_term("task_command")
        self.pos_thr = float(cmd.success_position_threshold)
        self.rot_thr = float(cmd.success_orientation_threshold)
        self.env_origins = sc.env_origins.clone()
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
        # The OSC re-targets "current pose + delta" every policy step and tracks it with a PD law, so
        # in steady state the pose moves at v = Kp*delta/Kd, i.e. only Kp/Kd*step_dt of each commanded
        # delta is realised per step (about 0.24 for the translation gains of the default env).
        # Bounds from that, no margin; the largest observed hand step is logged against it.
        kp, kd = arm._kp[0].clone(), arm._kd[0].clone()
        self.realised = kp / kd * env.step_dt  # fraction of the commanded delta realised per step, [6]
        self.max_hand_step = float(self.T * torch.linalg.norm(self.scale[:3] * self.realised[:3]))  # m / edge
        self.max_object_step = self.max_hand_step
        self.max_rot_step = float(self.T * torch.linalg.norm(self.scale[3:] * self.realised[3:]))  # rad / edge
        self.profile = {"dyn_calls": 0, "dyn_rows": 0, "dyn_time": 0.0, "max_peg_step": 0.0,
                        "max_hand_step": 0.0, "bad_rows": 0}
        self.flung = []  # (state[n], control[NCTRL], peg_step, bad) of edges with peg step > 0.1 m or bad; for contact replays
        self.normal = []  # random ~2% sample of ordinary edges (same fields): baseline for contact replays
        self._tree = {k: [] for k in ("parent_call", "parent_child_id", "parent_feat", "child_parent", "child_feat",
                                      "child_peg", "child_bp", "child_valid", "child_cost", "child_h")}
        self._tree_index: dict[int, int] = {}
        self._n_children = 0
        self.start_bp = None  # [n_bodies,7] link poses (env frame) of the start node, set by set_start
        self.diag = []  # per simulated edge: [peak |qd|/limit over robot joints, peg step (m), bad(1/0), peak peg speed m/s]

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
        hand = self._hand_point()
        feats = torch.cat([pp[:, :3], _quat_axis(pp[:, 3:7], 2), hand, grip_cmd[:, None]], dim=1)
        return torch.cat([feats, grip_cmd[:, None], jp, jv, pp, pv], dim=1)

    def _body_poses(self) -> torch.Tensor:
        """Link poses [N,n_bodies,7] (pos, quat xyzw) in each env's frame, for visualisation."""
        d = self.robot.data
        pos = d.body_link_pos_w.torch - self.env_origins[:, None]
        return torch.cat([pos, d.body_link_quat_w.torch], dim=-1)

    def _hand_point(self) -> torch.Tensor:
        """Hand reference point (the gripper's pinch point), env-relative [N,3]."""
        d = self.robot.data
        pos, _ = self.ee_offset.combine(d.body_link_pos_w.torch[:, self.hand_body].view(-1, 3),
                                        d.body_link_quat_w.torch[:, self.hand_body].view(-1, 4))
        return pos - self.env_origins

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
        c, g, K = self.cfg, self.gen, self.cfg.num_controls
        u = torch.zeros(K, NCTRL)
        d = torch.randn(K, 3, generator=g)
        d /= d.norm(dim=1, keepdim=True)
        mag = torch.where(torch.rand(K, generator=g) < 0.5, torch.ones(K), 0.25 + 0.75 * torch.rand(K, generator=g))
        u[:, :3] = (d * mag[:, None]).clamp(-1, 1)
        r = torch.randn(K, 3, generator=g)
        r = r / r.norm(dim=1, keepdim=True) * torch.rand(K, 1, generator=g)
        u[:, 3:6] = (r * (torch.rand(K, 1, generator=g) < c.rot_prob)).clamp(-1, 1)
        tog = torch.rand(K, generator=g) < c.grip_toggle_prob
        u[:, 6] = torch.where(tog, torch.sign(torch.rand(K, generator=g) - 0.5), torch.zeros(K))
        u[0] = 0.0  # row 0: stand still, keep the gripper
        n_p, n_h = c.n_peg_rows, c.n_hole_rows
        u[K - n_p - n_h:K - n_h, 7] = 1.0
        u[K - n_h:, 7] = 2.0
        u[K - n_p - n_h:, 8] = torch.randint(0, 2 ** 30, (n_p + n_h,), generator=g).float()  # seed for the point
        return u

    def _resolve(self, s: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Node states s [B,n], controls u [B,NCTRL] (device) -> env actions [B,7], grip cmd [B]."""
        from isaaclab.utils import math as mu

        B = s.shape[0]
        a = torch.zeros(B, 7, device=self.device)
        a[:, :6] = u[:, :6]
        grip_prev = s[:, self.o_grip]
        cmd = torch.where(u[:, 6] == 0, grip_prev, u[:, 6])
        kind = u[:, 7].long()
        peg_pose, hand = s[:, self.o_peg:self.o_peg + 7], s[:, 6:9]
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
            step = self.scale[:3] * self.T
            act = (delta_b / step).clamp(-1, 1)
            m = kind > 0
            a[m, :3] = act[m]
            a[m, 3:6] = 0.0
            # peg rows: close the gripper once the hand sits on the chosen point, otherwise open it
            near = delta_w.norm(dim=1) < self.cfg.peg_grasp_dist
            peg_cmd = torch.where(near, -torch.ones_like(cmd), torch.ones_like(cmd))
            cmd = torch.where(kind == 1, peg_cmd, cmd)
        a[:, 6] = cmd  # <0 closes, >=0 opens (Isaac Lab binary action)
        return a, cmd

    # ------------------------------------------------------------------ dynamics
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
            bad = torch.zeros(self.N, dtype=torch.bool, device=self.device)
            lim = self.robot.data.joint_vel_limits.torch
            acc_bad = torch.zeros(self.N, dtype=torch.bool, device=self.device)
            prev_v = s[:, self.o_pegv:self.o_pegv + 3]
            a_max = self.cfg.max_peg_accel_g * 9.81
            ratio = torch.zeros(self.N, device=self.device)
            bp = None
            for t in range(self.T):
                self.env.step(act)
                cur = self._read_states(cmd)
                if self.cfg.log_tree and t == self.T - 1:
                    bp = self._body_poses()
                ratio = torch.maximum(ratio, torch.nan_to_num((cur[:, self.o_qd:self.o_qd + self.J].abs() / lim).amax(dim=1), nan=1e9))
                bad |= ~torch.isfinite(cur).all(dim=1)
                bad |= cur[:, self.o_qd:self.o_qd + self.J].abs().amax(dim=1) > self.cfg.max_joint_vel
                v = cur[:, self.o_pegv:self.o_pegv + 3]
                acc_bad |= ~((v - prev_v).norm(dim=1) / self.env.step_dt <= a_max)  # NaN also counts
                prev_v = v
                out[c0:c1, t] = cur[:nb].cpu()
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
            raw = out[c0:c1].clone() if self.cfg.log_tree else None  # keep blown-up edges for the viewer
            out[c0:c1][bad] = float("nan")
            if self.cfg.log_tree:
                self._log_expansion(states[c0:c1], controls[c0:c1], out[c0:c1], raw, bp[:nb].cpu())
            self.profile["bad_rows"] += int(bad.sum())
            hs = (out[c0:c1, -1, 6:9] - states[c0:c1, 6:9]).norm(dim=1)
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
        dj = (next_states[:, self.o_q:self.o_q + 6] - states[:, self.o_q:self.o_q + 6]).norm(dim=1)
        return self.edge_dt + self.cfg.joint_motion_weight * dj

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
        }
