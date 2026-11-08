"""abc_sim tasks as IGHA* ``generic`` environments (Python callbacks).

Nothing here touches the IGHA* search code: a problem object only produces the
six callbacks described in ``ighastar/src/Environments/GENERIC.md`` plus the
config keys ``create_planner`` needs.

State vector (``n = F + 14 + nq + nv``)::

    [ features (F, hashed) | ctrl targets (14, policy space) | qpos (nq) | qvel (nv) ]

* ``features`` are the task-relevant low-dimensional quantities the planner's
  grid hashes on (the leading ``hash_dims``). They are recomputed from the sim
  after every rollout, so they are always consistent with ``qpos``.
* ``ctrl`` is the robot's current position-actuator target in abc's 14-D policy
  space ``[left j1..j6, left grip, right j1..j6, right grip]`` (grippers in
  [0, 1]). It is part of the state because the actions are deltas on it.
* ``qpos`` / ``qvel`` are the full MuJoCo state, so a node can be re-simulated
  exactly from its vector alone.

Control (14-D): a delta on the policy-space target. One edge ramps the target
linearly from ``ctrl`` to ``clip(ctrl + delta)`` over ``ctrl_steps_per_edge``
30 Hz control periods, simulated in MJWarp. Every control period is stored as an
intermediate state, so the planner's dense path is directly a 30 Hz
(observation, action) episode in abc's own action convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
import torch

from warp_rollout import WarpBatchRollout

ARM_SLICES = {"left": slice(0, 7), "right": slice(7, 14)}
GRIPPER_POLICY_IDX = (6, 13)
TCP_SITES = ("left_tcp_site", "right_tcp_site")


@dataclass
class ProblemConfig:
    ctrl_steps_per_edge: int = 10
    num_controls: int = 64
    joint_delta: list[float] = field(default_factory=lambda: [0.2, 0.2, 0.2, 0.3, 0.3, 0.3])
    gripper_delta: float = 1.0
    include_zero_control: bool = True
    active_arms: str = "auto"  # auto | left | right | both
    settle_ctrl_steps: int = 15
    rollout_capacities: list[int] = field(default_factory=lambda: [64, 256, 1024])
    nconmax: int = 128
    njmax: int = 1200
    # cost = edge duration [s] + joint_motion_weight * |delta joints|
    joint_motion_weight: float = 0.05
    # validity
    table_z: float = 0.75
    min_tcp_height: float = 0.01
    max_qvel: float = 50.0
    object_xy_bounds: list[float] = field(default_factory=lambda: [0.2, 1.0, -0.65, 0.65])
    disturb_tol: float = 0.05
    speed_bound_margin: float = 1.5
    log_tree: bool = False  # record every expansion for visualization
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ProblemConfig":
        d = dict(d or {})
        known = {k: d.pop(k) for k in list(d) if k in cls.__dataclass_fields__}
        return cls(**known)


class AbcGenericProblem:
    """Task-independent part: state layout, MJWarp dynamics, generic validity."""

    feature_names: tuple[str, ...] = ()

    def __init__(self, env, cfg: ProblemConfig, task_cfg: dict[str, Any]) -> None:
        from abc_sim.env import _GRIPPER_CTRL_MAX

        self.env = env
        self.cfg = cfg
        self.task_cfg = dict(task_cfg or {})
        self.mjm: mujoco.MjModel = env.model
        self.mjd: mujoco.MjData = env.data
        self.nq, self.nv, self.nu = self.mjm.nq, self.mjm.nv, self.mjm.nu
        self.F = len(self.feature_names)
        self.o_ctrl = self.F
        self.o_qpos = self.F + 14
        self.o_qvel = self.o_qpos + self.nq
        self.n = self.o_qvel + self.nv
        self.dt_ctrl = float(env._physics_dt * env._control_decimation)
        self.edge_dt = self.dt_ctrl * cfg.ctrl_steps_per_edge

        # policy space <-> actuator space
        self.ctrl_indices = torch.as_tensor(list(env._ctrl_indices), dtype=torch.long)
        scale = torch.ones(14)
        scale[list(GRIPPER_POLICY_IDX)] = float(_GRIPPER_CTRL_MAX)
        self.policy_scale = scale
        lo = torch.as_tensor(self.mjm.actuator_ctrlrange[:, 0], dtype=torch.float32)
        hi = torch.as_tensor(self.mjm.actuator_ctrlrange[:, 1], dtype=torch.float32)
        self.policy_lo = lo[self.ctrl_indices] / scale
        self.policy_hi = hi[self.ctrl_indices] / scale
        self.policy_lo[list(GRIPPER_POLICY_IDX)] = 0.0
        self.policy_hi[list(GRIPPER_POLICY_IDX)] = 1.0

        self.site_ids = [
            mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_SITE, s) for s in TCP_SITES
        ]
        if min(self.site_ids) < 0:
            raise ValueError(f"missing TCP sites {TCP_SITES}")
        self.free_qpos_adrs = [
            int(self.mjm.jnt_qposadr[j])
            for j in range(self.mjm.njnt)
            if self.mjm.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ]

        self.rollout = WarpBatchRollout(
            self.mjm, self.mjd,
            capacities=cfg.rollout_capacities,
            substeps=env._control_decimation,
            site_ids=self.site_ids,
            nconmax=cfg.nconmax, njmax=cfg.njmax,
        )
        self.dev = self.rollout.device
        self._gen = torch.Generator().manual_seed(int(cfg.seed))
        self.profile = {"dyn_calls": 0, "dyn_rows": 0, "dyn_time": 0.0, "overflows": 0}
        self._tree = {k: [] for k in (
            "parent_feat", "parent_qpos", "parent_call", "parent_child_id",
            "child_parent", "child_feat", "child_qpos", "child_valid", "child_cost", "child_h")}
        self._tree_index: dict[int, int] = {}  # hash(final state bytes) -> child id
        self._n_children = 0

        self.active_arms = self._resolve_active_arms()
        mask = torch.zeros(14)
        for arm in self.active_arms:
            mask[ARM_SLICES[arm]] = 1.0
        per_arm = torch.tensor(list(cfg.joint_delta) + [cfg.gripper_delta], dtype=torch.float32)
        self.delta_scale = torch.cat([per_arm, per_arm]) * mask
        self.max_arm_step = self._max_arm_point_step()
        self.start = self._make_start_state()

    def _max_arm_point_step(self) -> float:
        """Upper bound [m] on how far any point of an active arm moves in one edge.

        A rotation by dq about joint i moves a point at most |dq| * r, with r the
        point's distance from joint i's anchor. Consecutive joint anchors are a
        fixed distance apart in any configuration, so r_i <= sum of the
        anchor-to-anchor link lengths below joint i + the hand's radius about
        the last anchor (all geoms of the wrist subtree, fingers fully open).
        The finger slides add gripper_delta * finger travel. The per-edge
        commanded |dq| is joint_delta; ``speed_bound_margin`` covers PD
        overshoot of the commanded ramp (checked empirically during search).
        """
        m, d = self.mjm, mujoco.MjData(self.mjm)
        d.qpos[:] = self.mjd.qpos
        mujoco.mj_forward(m, d)
        from abc_sim.env import _GRIPPER_CTRL_MAX

        steps = []
        for arm in self.active_arms:
            jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_joint{k}") for k in range(1, 7)]
            anchors = [np.array(d.xanchor[j]) for j in jids]
            wrist_body = int(m.jnt_bodyid[jids[-1]])
            hand_r = 0.0
            for g in range(m.ngeom):
                b = int(m.geom_bodyid[g])
                while b not in (0, wrist_body):
                    b = int(m.body_parentid[b])
                if b == wrist_body:
                    hand_r = max(hand_r, float(np.linalg.norm(d.geom_xpos[g] - anchors[-1]) + m.geom_rbound[g]))
            hand_r += float(_GRIPPER_CTRL_MAX)  # finger slide travel
            links = [float(np.linalg.norm(anchors[k + 1] - anchors[k])) for k in range(5)]
            radii = [sum(links[i:]) + hand_r for i in range(6)]
            step = sum(dq * r for dq, r in zip(self.cfg.joint_delta, radii))
            step += self.cfg.gripper_delta * float(_GRIPPER_CTRL_MAX)
            steps.append(step)
        # Two independently moving arms could both push the object.
        return float(self.cfg.speed_bound_margin * sum(steps))

    # ---------------------------------------------------------------- hooks
    def _resolve_active_arms(self) -> tuple[str, ...]:
        a = self.cfg.active_arms
        if a == "both":
            return ("left", "right")
        if a in ARM_SLICES:
            return (a,)
        raise ValueError(f"active_arms={a!r} not understood for this task")

    def features(self, qpos: torch.Tensor, tcp: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
        """qpos [..., nq], tcp [..., 2, 3], ctrl [..., 14] -> [..., F]."""
        raise NotImplementedError

    def task_valid(self, states: torch.Tensor) -> torch.Tensor:
        return torch.ones(states.shape[0], dtype=torch.bool)

    def heuristic(self, states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def goal_mask(self, states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # ---------------------------------------------------------------- layout
    def ctrl(self, s: torch.Tensor) -> torch.Tensor:
        return s[..., self.o_ctrl:self.o_ctrl + 14]

    def qpos(self, s: torch.Tensor) -> torch.Tensor:
        return s[..., self.o_qpos:self.o_qpos + self.nq]

    def qvel(self, s: torch.Tensor) -> torch.Tensor:
        return s[..., self.o_qvel:self.o_qvel + self.nv]

    def policy_to_actuator(self, ctrl_policy: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(ctrl_policy.shape[:-1] + (self.nu,), device=ctrl_policy.device)
        out[..., self.ctrl_indices.to(ctrl_policy.device)] = ctrl_policy * self.policy_scale.to(ctrl_policy.device)
        return out

    def pack(self, qpos, qvel, tcp, ctrl) -> torch.Tensor:
        return torch.cat([self.features(qpos, tcp, ctrl), ctrl, qpos, qvel], dim=-1)

    def _make_start_state(self) -> torch.Tensor:
        """Reset state of the abc env, settled in MJWarp with the arms holding pose."""
        qpos0 = torch.as_tensor(self.mjd.qpos, dtype=torch.float32)[None]
        qvel0 = torch.zeros(1, self.nv)
        ctrl_policy = torch.as_tensor(self.env.get_state(), dtype=torch.float32)[None]
        ctrl_policy = torch.minimum(torch.maximum(ctrl_policy, self.policy_lo), self.policy_hi)
        act = self.policy_to_actuator(ctrl_policy)
        r = self.rollout.rollout(qpos0, qvel0, act, act, self.cfg.settle_ctrl_steps)
        q, v, sx = r.qpos[:, -1].cpu(), r.qvel[:, -1].cpu(), r.site_xpos[:, -1].cpu()
        return self.pack(q, v, sx, ctrl_policy)[0]

    # ------------------------------------------------------------ callbacks
    def sample_controls(self) -> torch.Tensor:
        K = self.cfg.num_controls
        u = torch.rand(K, 14, generator=self._gen) * 2.0 - 1.0
        u = u * self.delta_scale
        if self.cfg.include_zero_control:
            u[0] = 0.0
        return u

    def dynamics(self, states: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        import time

        t0 = time.perf_counter()
        T = self.cfg.ctrl_steps_per_edge
        s = states.to(self.dev, torch.float32)
        u = controls.to(self.dev, torch.float32)
        c0 = self.ctrl(s)
        lo, hi = self.policy_lo.to(self.dev), self.policy_hi.to(self.dev)
        c1 = torch.minimum(torch.maximum(c0 + u, lo), hi)
        r = self.rollout.rollout(
            self.qpos(s), self.qvel(s), self.policy_to_actuator(c0),
            self.policy_to_actuator(c1), T,
        )
        if r.overflow:
            self.profile["overflows"] += 1
        alphas = torch.arange(1, T + 1, device=self.dev, dtype=torch.float32) / T
        ctrl_seq = c0[:, None] + alphas[None, :, None] * (c1 - c0)[:, None]
        out = self.pack(r.qpos, r.qvel, r.site_xpos, ctrl_seq)  # [B, T, n]
        out = out.cpu().contiguous()
        if r.overflow:
            # A rollout whose contact buffer overflowed is not trustworthy.
            out[..., self.o_qvel] = float("nan")
        self.profile["dyn_calls"] += 1
        self.profile["dyn_rows"] += int(states.shape[0])
        self.profile["dyn_time"] += time.perf_counter() - t0
        if self.cfg.log_tree:
            self._log_expansion(states, controls, out)
        return out

    # ------------------------------------------------------------ tree log
    @staticmethod
    def _key(row: torch.Tensor) -> int:
        return hash(row.contiguous().numpy().tobytes())

    def _log_expansion(self, states, controls, out) -> None:
        """Record one batched expansion: parents, their children's rollouts, verdicts.

        Rows are packed parent-major (row i*K + k), so parents are states[::K].
        A parent is linked to the child (from an earlier expansion) whose final
        state it is, which makes the log a tree that the viewer can walk.
        """
        t = self._tree
        K = self.cfg.num_controls if states.shape[0] % self.cfg.num_controls == 0 else 1
        parents = states[::K]
        call = self.profile["dyn_calls"]
        p0 = len(t["parent_call"])
        for row in parents:
            t["parent_child_id"].append(self._tree_index.get(self._key(row), -1))
            t["parent_call"].append(call)
        t["parent_feat"].append(parents[:, : self.F].numpy().copy())
        t["parent_qpos"].append(self.qpos(parents).numpy().copy())
        fin = out[:, -1]
        valid = self.validity(fin) > 0
        cost = self.cost(states, controls, fin)
        h = self.heuristic(fin)
        B = out.shape[0]
        t["child_parent"].append(p0 + torch.arange(B).numpy() // K)
        t["child_feat"].append(out[:, :, : self.F].numpy().copy())
        t["child_qpos"].append(self.qpos(fin).numpy().copy())
        t["child_valid"].append(valid.numpy())
        t["child_cost"].append(cost.numpy())
        t["child_h"].append(h.numpy())
        for i in range(B):
            if valid[i]:
                self._tree_index[self._key(fin[i])] = self._n_children + i
        self._n_children += B

    def tree_arrays(self) -> dict[str, np.ndarray]:
        t = self._tree
        if not t["parent_call"]:
            return {}
        out = {k: np.concatenate(v) for k, v in t.items()
               if k not in ("parent_call", "parent_child_id")}
        out["parent_call"] = np.asarray(t["parent_call"], np.int32)
        out["parent_child_id"] = np.asarray(t["parent_child_id"], np.int64)
        return out

    def child_ids_of(self, states: torch.Tensor) -> list[int]:
        return [self._tree_index.get(self._key(r), -1) for r in states]

    def cost(self, states, controls, next_states) -> torch.Tensor:
        dj = self.ctrl(next_states) - self.ctrl(states)
        dj = dj.clone()
        dj[:, list(GRIPPER_POLICY_IDX)] = 0.0
        return self.edge_dt + self.cfg.joint_motion_weight * dj.norm(dim=1)

    def validity(self, states: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        finite = torch.isfinite(states).all(dim=1)
        qv_ok = self.qvel(states).abs().amax(dim=1) < c.max_qvel
        q = self.qpos(states)
        ok = finite & qv_ok
        x0, x1, y0, y1 = c.object_xy_bounds
        for adr in self.free_qpos_adrs:
            p = q[:, adr:adr + 3]
            ok &= (p[:, 2] > c.table_z - 0.05) & (p[:, 0] > x0) & (p[:, 0] < x1)
            ok &= (p[:, 1] > y0) & (p[:, 1] < y1)
        ok &= self.task_valid(states)
        return ok.float()

    def heuristic_fn(self, states: torch.Tensor) -> torch.Tensor:
        return self.heuristic(states)

    def goal_test(self, state: torch.Tensor) -> bool:
        return bool(self.goal_mask(state[None])[0])

    # --------------------------------------------------------------- config
    def planner_config(self, planner_cfg: dict[str, Any]) -> dict[str, Any]:
        info = dict(planner_cfg)

        def hashed(key: str) -> list[float]:
            # planner-level value (YAML `planner:` or CLI) wins over the task's.
            # Give one value per hashed feature (F); a full n_dims list is also
            # accepted, but only its first F entries affect the search.
            v = info.get(key, self.task_cfg.get(key))
            if v is None:
                raise KeyError(f"no '{key}' in planner or task config")
            v = [float(x) for x in v]
            if len(v) == self.n:
                v = v[: self.F]
            if len(v) != self.F:
                raise ValueError(
                    f"'{key}' needs {self.F} values (hashed features "
                    f"{', '.join(self.feature_names)}), got {len(v)}")
            return v + [1.0] * (self.n - self.F)  # non-hashed dims: unused filler

        res, tol = hashed("resolution"), hashed("tolerance")
        info.update(
            state_dim=self.n,
            control_dim=14,
            hash_dims=self.F,
            num_controls=self.cfg.num_controls,
            resolution=res,
            tolerance=tol,
        )
        info["node_info"] = {"node_type": "generic", "timesteps": self.cfg.ctrl_steps_per_edge}
        return {
            "experiment_info_default": info,
            "sample_controls_fn": self.sample_controls,
            "dynamics_fn": self.dynamics,
            "cost_fn": self.cost,
            "validity_fn": self.validity,
            "heuristic_fn": self.heuristic_fn,
            "goal_test_fn": self.goal_test,
        }


class PutRelativeProblem(AbcGenericProblem):
    """abc ``put_relative``: put the mover object <direction> of the reference.

    Goal test = abc's ``PutRelativeEvaluator`` (directional cone around the live
    reference, mover resting no higher than spawn + 3 cm) plus the mover being
    nearly at rest, so the final frame is a settled scene.
    """

    feature_names = (
        "mover_x", "mover_y", "mover_z",
        "ltcp_x", "ltcp_y", "ltcp_z",
        "rtcp_x", "rtcp_y", "rtcp_z",
        "lgrip", "rgrip",
    )

    def __init__(self, env, cfg: ProblemConfig, task_cfg: dict[str, Any]) -> None:
        ev = env._task_evaluator
        goal = getattr(ev, "_goal", None)
        if goal is None:
            raise RuntimeError("put_relative evaluator was not configured by reset()")
        self.goal = goal
        self.mover_adr = int(goal["mover_adr"])
        self.ref_adr = int(goal["reference_adr"])
        self.axis = torch.as_tensor(goal["axis"], dtype=torch.float32)
        self.perp_axis = torch.tensor([-float(self.axis[1]), float(self.axis[0])])
        self.spawn_z = goal["spawn_z"]
        self.along_min = float(ev.along_min_m)
        self.along_max = float(ev.along_max_m)
        self.perp_max = float(ev.perp_max_m)
        self.placed_tol = float(ev.placed_z_tolerance_m)
        super().__init__(env, cfg, task_cfg)
        q0 = self.qpos(self.start)
        self.others = [a for a in self.free_qpos_adrs if a != self.mover_adr]
        self.others_xy0 = torch.stack([q0[a:a + 2] for a in self.others]) if self.others else None
        self.mover_start = q0[self.mover_adr:self.mover_adr + 3].clone()
        self.ref_start_xy = q0[self.ref_adr:self.ref_adr + 2].clone()
        # Objects only move when the robot moves them, so they cannot travel
        # farther per edge than the fastest point of the arm.
        self.max_object_step = self.max_arm_step
        self.profile["max_mover_step"] = 0.0

    def dynamics(self, states, controls):
        out = super().dynamics(states, controls)
        a = self.o_qpos + self.mover_adr
        step = (out[:, -1, a:a + 3] - states[:, a:a + 3]).norm(dim=1)
        step = step[torch.isfinite(step)]
        if step.numel():
            self.profile["max_mover_step"] = max(self.profile["max_mover_step"], float(step.max()))
        # Diagnostics: closest approach of an active TCP to the mover.
        fin = out[:, -1]
        mover = fin[:, 0:3]
        tcp = fin[:, 3:9].view(-1, 2, 3)
        idx = [0 if a == "left" else 1 for a in self.active_arms]
        dmin = (tcp[:, idx] - mover[:, None]).norm(dim=2).amin(1)
        dmin = dmin[torch.isfinite(dmin)]
        if dmin.numel():
            self.profile["min_tcp_mover_dist"] = min(
                self.profile.get("min_tcp_mover_dist", float("inf")), float(dmin.min()))
        return out

    def _resolve_active_arms(self) -> tuple[str, ...]:
        if self.cfg.active_arms != "auto":
            return super()._resolve_active_arms()
        # The left arm is mounted at y=+0.3, the right one at y=-0.3: use the
        # arm on the mover's side of the table.
        y = float(self.mjd.qpos[self.mover_adr + 1])
        return ("left",) if y >= 0.0 else ("right",)

    def features(self, qpos, tcp, ctrl):
        mover = qpos[..., self.mover_adr:self.mover_adr + 3]
        return torch.cat(
            [mover, tcp[..., 0, :], tcp[..., 1, :], ctrl[..., 6:7], ctrl[..., 13:14]], dim=-1
        )

    def _cone(self, states):
        q = self.qpos(states)
        mover = q[:, self.mover_adr:self.mover_adr + 3]
        ref = q[:, self.ref_adr:self.ref_adr + 3]
        rel = mover[:, :2] - ref[:, :2]
        along = rel @ self.axis
        perp = (rel @ self.perp_axis).abs()
        return mover, ref, along, perp

    def goal_mask(self, states):
        mover, _, along, perp = self._cone(states)
        ok = (along >= self.along_min) & (along <= self.along_max) & (perp <= self.perp_max)
        if self.spawn_z is not None:
            ok &= mover[:, 2] <= self.spawn_z + self.placed_tol
        v = self.qvel(states)
        # free joint qvel layout: [lin(3), ang(3)] at its dof address
        dof = self._mover_dof()
        ok &= v[:, dof:dof + 3].norm(dim=1) < float(self.task_cfg.get("goal_max_speed", 0.05))
        return ok

    def _mover_dof(self) -> int:
        if not hasattr(self, "_mover_dof_adr"):
            j = next(
                j for j in range(self.mjm.njnt)
                if self.mjm.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                and int(self.mjm.jnt_qposadr[j]) == self.mover_adr
            )
            self._mover_dof_adr = int(self.mjm.jnt_dofadr[j])
        return self._mover_dof_adr

    def goal_distance(self, states):
        """Lower bound on how far the mover still has to travel to reach the goal.

        Exact distance from the mover to the goal region (the evaluator's cone
        around the reference, capped at the placed height), minus the most the
        reference itself could still move toward it: the reference must stay
        within ``disturb_tol`` of its start, so from where it is now it can move
        at most ``disturb_tol + |ref - ref_start|``.
        """
        mover, ref, along, perp = self._cone(states)
        gap = torch.hypot(
            along - along.clamp(self.along_min, self.along_max),
            perp - perp.clamp(max=self.perp_max),
        )
        lift = torch.zeros_like(gap)
        if self.spawn_z is not None:
            lift = (mover[:, 2] - (self.spawn_z + self.placed_tol)).clamp(min=0.0)
        dist = torch.hypot(gap, lift)
        ref_slack = self.cfg.disturb_tol + (ref[:, :2] - self.ref_start_xy).norm(dim=1)
        if self.cfg.disturb_tol <= 0:
            ref_slack = torch.full_like(dist, float("inf"))
        return (dist - ref_slack).clamp(min=0.0)

    def heuristic(self, states):
        # Admissible: every edge costs >= edge_dt and the mover can move at most
        # max_object_step metres per edge, so the remaining cost is at least
        # (distance / max_object_step) * edge_dt.
        return self.goal_distance(states) / self.max_object_step * self.edge_dt

    def task_valid(self, states):
        ok = torch.ones(states.shape[0], dtype=torch.bool)
        q = self.qpos(states)
        c = self.cfg
        tcp = states[:, 3:9].view(-1, 2, 3)
        ok &= (tcp[:, :, 2] > c.table_z + c.min_tcp_height).all(dim=1)
        if self.others_xy0 is not None and c.disturb_tol > 0:
            xy = torch.stack([q[:, a:a + 2] for a in self.others], 1)
            ok &= ((xy - self.others_xy0[None]).norm(dim=2) < c.disturb_tol).all(dim=1)
        return ok


PROBLEMS = {"put_relative": PutRelativeProblem}
