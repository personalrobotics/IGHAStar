"""Lean policy step for planning: same physics loop as ManagerBasedEnv.step, minus the RL managers, with the UWLab OSC
controller's torque computation captured in a CUDA graph. UNREVIEWED.

What the stock path does per policy step (env.step): process_action once, then ``decimation`` times
[action_manager.apply_action -> scene.write_data_to_sim -> sim.step -> scene.update], then the event/observation/
reward/termination/command managers. The planner needs none of the managers (the task has only startup/reset events, no
interval events; goal and features are computed from the scene), so this skips them.

The controller (RelCartesianOSCAction.apply_actions) is tau = clamp(J(q)^T (Kp * pose_err + Kd * (-J(q) qd))) with the
analytical Jacobian, recomputed from the current joint state at every physics step. It is eager PyTorch (hundreds of tiny
kernels and a CPU->GPU copy of the calibration constants per call). Here the same math runs on constants kept on the GPU
and the whole thing is replayed as one CUDA graph; inputs are copied into static buffers first, so any lazy refresh of the
simulator's data buffers still happens outside the graph. PhysX itself (sim.step) is not captured.
"""
from __future__ import annotations

import torch
from isaaclab.utils import math as math_utils
from uwlab_assets.robots.ur5e_robotiq_gripper import kinematics as K


class FastPhysics:
    def __init__(self, env, use_graph: bool = True):
        self.env = env
        self.dev = env.device
        self.term = next(t for t in env.action_manager._terms.values() if type(t).__name__ == "RelCartesianOSCAction")
        self.others = [t for t in env.action_manager._terms.values() if t is not self.term]
        self.asset = self.term._asset
        self.ids = self.term._joint_ids
        self.ee = self.term._ee_body_idx
        N = env.num_envs
        d = self.dev
        cal = K._load_calibration()
        xyz, rpy = cal["joints_xyz"].to(d), cal["joints_rpy"].to(d)
        fixed = torch.zeros(6, 4, 4, device=d)
        for i in range(6):
            fixed[i] = torch.eye(4, device=d)
            fixed[i, :3, :3] = K.rpy_to_matrix_torch(rpy[i])
            fixed[i, :3, 3] = xyz[i]
        self.fixed = fixed
        self.R180 = K.R_180Z.to(d)
        f = dict(device=d, dtype=torch.float32)
        z = lambda *s: torch.zeros(*s, **f)  # noqa: E731
        self.q, self.qd = z(N, 6), z(N, 6)
        self.ee_pos_w, self.ee_quat_w = z(N, 3), z(N, 4)
        self.root_pos_w, self.root_quat_w = z(N, 3), z(N, 4)
        self.des_pos, self.des_quat = z(N, 3), z(N, 4)
        self.kp, self.kd = z(N, 6), z(N, 6)
        self.tmax = z(6)
        self.tau = z(N, 6)
        self.graph = None
        self._load_static()
        self._copy_policy_inputs()
        if use_graph:
            self._capture()

    # ------------------------------------------------------------------ the controller math (same as UWLab's)
    def jacobian(self, q: torch.Tensor) -> torch.Tensor:
        """Analytical Jacobian [N,6,6] (linear; angular) in base_link, same convention as compute_jacobian_analytical."""
        N = q.shape[0]
        c, s = torch.cos(q), torch.sin(q)
        rot = torch.zeros(N, 6, 4, 4, device=q.device, dtype=q.dtype)
        rot[..., 0, 0], rot[..., 0, 1], rot[..., 1, 0], rot[..., 1, 1] = c, -s, s, c
        rot[..., 2, 2] = 1.0
        rot[..., 3, 3] = 1.0
        T = torch.eye(4, device=q.device, dtype=q.dtype).expand(N, 4, 4)
        zs, ps = [], []
        for i in range(6):
            Tf = T @ self.fixed[i]
            zs.append(Tf[:, :3, 2])
            ps.append(Tf[:, :3, 3])
            T = Tf @ rot[:, i]
        p_ee = T[:, :3, 3]
        lin = torch.stack([torch.cross(zs[i], p_ee - ps[i], dim=1) for i in range(6)], dim=2)  # [N,3,6]
        ang = torch.stack(zs, dim=2)
        return torch.cat([self.R180 @ lin, self.R180 @ ang], dim=1)

    def torques(self) -> torch.Tensor:
        J = self.jacobian(self.q)
        ee_vel = torch.bmm(J, self.qd.unsqueeze(-1)).squeeze(-1)
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(self.root_pos_w, self.root_quat_w, self.ee_pos_w, self.ee_quat_w)
        pos_error = self.des_pos - ee_pos_b
        quat_error = math_utils.quat_mul(self.des_quat, math_utils.quat_inv(ee_quat_b))
        pose_error = torch.cat([pos_error, math_utils.axis_angle_from_quat(quat_error)], dim=-1)
        task_force = self.kp * pose_error + self.kd * (-ee_vel)
        tau = torch.bmm(J.transpose(-1, -2), task_force.unsqueeze(-1)).squeeze(-1)
        return torch.clamp(tau, -self.tmax, self.tmax)

    # ------------------------------------------------------------------ buffers and graph
    def _load_static(self):
        self.kp.copy_(self.term._kp)
        self.kd.copy_(self.term._kd)
        self.tmax.copy_(self.term._torque_max)

    def _copy_policy_inputs(self):
        """Per policy step: the desired pose set by process_actions (and gains, in case they were re-randomised)."""
        self.des_pos.copy_(self.term._ee_pos_des)
        self.des_quat.copy_(self.term._ee_quat_des)
        self._load_static()

    def _copy_state_inputs(self):
        d = self.asset.data
        self.q.copy_(d.joint_pos.torch[:, self.ids])
        self.qd.copy_(d.joint_vel.torch[:, self.ids])
        self.ee_pos_w.copy_(d.body_pos_w.torch[:, self.ee])
        self.ee_quat_w.copy_(d.body_quat_w.torch[:, self.ee])
        self.root_pos_w.copy_(d.root_pos_w.torch)
        self.root_quat_w.copy_(d.root_quat_w.torch)

    def _capture(self):
        self._copy_state_inputs()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.tau.copy_(self.torques())
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.tau.copy_(self.torques())
        self.graph = g

    def compute_torques(self) -> torch.Tensor:
        self._copy_state_inputs()
        if self.graph is not None:
            self.graph.replay()
        else:
            self.tau.copy_(self.torques())
        return self.tau

    # ------------------------------------------------------------------ stepping
    def step(self, action: torch.Tensor) -> None:
        """One policy step (decimation physics steps) without the RL managers."""
        env = self.env
        env.action_manager.process_action(action.to(self.dev))
        self._copy_policy_inputs()
        for _ in range(env.cfg.decimation):
            self.asset.set_joint_effort_target(self.compute_torques(), joint_ids=self.ids)
            for t in self.others:
                t.apply_actions()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)


class NewtonGraphedStep:
    """Newton only: one CUDA graph per policy step (the whole decimation loop: OSC torques, gripper term, Isaac Lab
    write/read, Newton collision + solver). UNREVIEWED.

    Requires Newton's own per-step graph OFF (NewtonCfg.use_cuda_graph=False; Newton's kernels are recorded into this
    graph instead). process_action (once per policy step) stays eager. The first call runs eagerly (loads kernels), the
    second captures, every call after that replays.

    Tensors the graph reads must keep their address: the gripper term's command is pinned (_pin_term_buffers).
    Replay does not run the Python side of the loop, so Isaac Lab's lazy data buffers still carry the timestamps of the
    capture; one eager scene.update() after each replay marks them stale so later reads recompute from the sim.
    Newton ping-pongs two state buffers; the graph bakes in their addresses, so the decimation count must be even (the
    buffers are back where they started after the loop) -- checked after capture."""

    def __init__(self, fast: FastPhysics):
        self.inner = fast
        self.env = fast.env
        ids = list(fast.ids) if not isinstance(fast.ids, slice) else None
        contiguous = ids is not None and ids == list(range(ids[0], ids[-1] + 1))
        # a Python-list / CPU index would copy from the host inside the graph (not capturable)
        self.ids = slice(ids[0], ids[-1] + 1) if contiguous else torch.as_tensor(fast.ids, device=fast.dev)
        self.graph = None
        self.calls = 0
        self._pinned = {}

    def __getattr__(self, name):  # everything else (compute_torques, asset, ...) as FastPhysics
        return getattr(self.inner, name)

    def _loop(self):
        env, fp, sl = self.env, self.inner, self.ids
        for _ in range(env.cfg.decimation):
            d = fp.asset.data
            fp.q.copy_(d.joint_pos.torch[:, sl])
            fp.qd.copy_(d.joint_vel.torch[:, sl])
            fp.ee_pos_w.copy_(d.body_pos_w.torch[:, fp.ee])
            fp.ee_quat_w.copy_(d.body_quat_w.torch[:, fp.ee])
            fp.root_pos_w.copy_(d.root_pos_w.torch)
            fp.root_quat_w.copy_(d.root_quat_w.torch)
            fp.tau.copy_(fp.torques())
            fp.asset.set_joint_effort_target(fp.tau, joint_ids=sl)
            for t in fp.others:
                t.apply_actions()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)

    def _capture(self):
        import warp as wp
        from isaaclab_newton.physics import NewtonManager as NM

        s0, s1 = NM._state_0, getattr(NM, "_state_1", None)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            ws = wp.stream_from_torch(s)
            with wp.ScopedStream(ws, sync_enter=False):
                wp.capture_begin(stream=ws, external=True, capture_mode=wp.CaptureMode.GLOBAL)
                try:
                    self._loop()
                finally:
                    wp.capture_end(stream=ws)
        torch.cuda.current_stream().wait_stream(s)
        if NM._state_0 is not s0 or getattr(NM, "_state_1", None) is not s1:
            raise RuntimeError("Newton state buffers did not return to their starting slots after one policy step "
                               "(odd decimation x substeps?); a replayed graph would write the wrong buffer")
        self.graph = g

    def _pin_term_buffers(self):
        """Other action terms (the gripper's BinaryJointAction) REASSIGN ``_processed_actions`` in process_actions
        (torch.where -> new tensor); a captured apply_actions would keep reading the capture-time tensor, i.e. a frozen
        gripper command. Copy each step's value into one persistent tensor and point the term at it."""
        for t in self.inner.others:
            cur = getattr(t, "_processed_actions", None)
            if cur is None:
                continue
            pinned = self._pinned.get(id(t))
            if pinned is None:
                pinned = self._pinned[id(t)] = cur.clone()
            elif cur is not pinned:
                pinned.copy_(cur)
            t._processed_actions = pinned

    def step(self, action: torch.Tensor) -> None:
        env, fp = self.env, self.inner
        env.action_manager.process_action(action.to(fp.dev))
        fp._copy_policy_inputs()
        self._pin_term_buffers()
        self.calls += 1
        if self.calls == 1:  # eager once: loads every kernel and allocates every buffer before capture
            self._loop()
            return
        if self.graph is None:
            self._capture()
        self.graph.replay()
        env.scene.update(dt=env.physics_dt)  # mark Isaac Lab's lazy buffers stale (see class docstring)


class NewtonGraphedEdge:
    """Newton only: the WHOLE edge (T policy steps: hand-delta -> wrist action, process_action, decimation physics
    steps, state read-back) as one CUDA graph. UNREVIEWED.

    Inputs are static GPU buffers (``act`` [N,7] hand-delta actions, ``cmd`` [N] gripper command) that run() copies
    into; outputs are static buffers ``traj`` [N,T,n] (and ``bp`` [N,n_bodies,7] link poses at the last step when the
    problem logs the tree). Same first-call-eager / second-call-capture scheme and the same post-replay
    scene.update() as NewtonGraphedStep (whose physics loop it reuses). The state write before the edge stays eager."""

    def __init__(self, prob, step: "NewtonGraphedStep"):
        self.prob, self.step, self.env = prob, step, prob.env
        N, d = prob.N, prob.device
        self.act = torch.zeros(N, 7, device=d)
        self.cmd = torch.zeros(N, device=d)
        self.traj = torch.zeros(N, prob.T, prob.n, device=d)
        self.bp = None
        self.graph = None
        self.calls = 0

    def _body(self):
        prob, env, fp = self.prob, self.env, self.step.inner
        for t in range(prob.T):
            env.action_manager.process_action(prob._wrist_action(self.act, t))
            fp._copy_policy_inputs()
            self.step._loop()
            self.traj[:, t].copy_(prob._read_states(self.cmd))
        if prob.cfg.log_tree:
            b = prob._body_poses()
            if self.bp is None:
                self.bp = torch.zeros_like(b)
            self.bp.copy_(b)

    def _capture(self):
        import warp as wp
        from isaaclab_newton.physics import NewtonManager as NM

        s0, s1 = NM._state_0, getattr(NM, "_state_1", None)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            ws = wp.stream_from_torch(s)
            with wp.ScopedStream(ws, sync_enter=False):
                wp.capture_begin(stream=ws, external=True, capture_mode=wp.CaptureMode.GLOBAL)
                try:
                    self._body()
                finally:
                    wp.capture_end(stream=ws)
        torch.cuda.current_stream().wait_stream(s)
        if NM._state_0 is not s0 or getattr(NM, "_state_1", None) is not s1:
            raise RuntimeError("Newton state buffers did not return to their starting slots over one edge")
        self.graph = g

    def run(self, act: torch.Tensor, cmd: torch.Tensor):
        """act [N,7], cmd [N] (device) -> (traj [N,T,n], bp or None), both static buffers (copy before the next run)."""
        self.act.copy_(act)
        self.cmd.copy_(cmd)
        self.calls += 1
        if self.calls == 1:
            self._body()
        else:
            if self.graph is None:
                self._capture()
            self.graph.replay()
            self.env.scene.update(dt=self.env.physics_dt)  # mark Isaac Lab's lazy buffers stale
        return self.traj, self.bp
