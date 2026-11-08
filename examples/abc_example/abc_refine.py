"""UNREVIEWED: written by Claude, not yet read or reviewed by a human. Treat as experimental.

abc_sim adapter for ``segment_mppi``: polish a coarse-step plan under abc's 2 ms physics.

* ``CpuAbcFine``: Dynamics. One control = one 14-D policy-space action held for a control
  period (abc's convention: ``ctrl`` constant over ``control_decimation`` physics steps).
  Rolled out with MuJoCo's multithreaded ``rollout`` on the env's own model, i.e. exactly
  the physics (2 ms, float64) that abc's evaluator replay uses.
* ``KnotCost``: Costs. Squared, sigma-normalised error of the rollout state against the
  coarse plan's knot: robot joints, and every free object's position, orientation and
  velocity. State layout is ``[time, qpos, qvel]`` (MuJoCo's full-physics state).
* ``polish``: wires them to ``SegmentMPPI`` and refines a plan loaded from the
  ``<task>_seed<k>.npz`` that ``plan_abc.py`` writes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import torch
from mujoco import rollout as mj_rollout

from segment_mppi import GaussianSampling, SegmentMPPI, refine_trajectory

GRIPPER_POLICY_IDX = (6, 13)


class PlanData:
    """A plan episode as written by plan_abc.py."""

    def __init__(self, path: str | Path):
        z = np.load(path, allow_pickle=False)
        self.rec = json.loads(str(z["record"]))
        self.qpos, self.qvel = z["qpos"].astype(np.float64), z["qvel"].astype(np.float64)
        self.actions = z["action"].astype(np.float64)  # [N,14] policy space
        self.seed, self.task = int(self.rec["seed"]), self.rec["task"]
        self.arms = list(self.rec["active_arms"])


def build_env(plan: PlanData, env=None):
    """abc env at the plan's seed. Pass the previous env back in to reuse it: make_env can
    only be called once per process for the randomized tasks, reset() is the way to re-seed."""
    import abc_sim

    if env is None:
        spec = abc_sim.get_task_spec(plan.task)
        env = abc_sim.make_env(task=spec.env_task, prompt=spec.prompt, render_cameras=False)
    env.reset(seed=plan.seed, randomize=True)
    return env


def policy_to_ctrl(env, a: np.ndarray) -> np.ndarray:
    """[..., 14] policy-space actions -> [..., nu] actuator controls (as env._step_single)."""
    from abc_sim.env import _GRIPPER_CTRL_MAX

    out = np.zeros(a.shape[:-1] + (env.model.nu,))
    for i in range(a.shape[-1]):
        out[..., env._ctrl_indices[i]] = a[..., i] * (_GRIPPER_CTRL_MAX if i in GRIPPER_POLICY_IDX else 1.0)
    return out


class CpuAbcFine(torch.nn.Module):
    """Dynamics: K controls x T periods rolled out in abc's CPU MuJoCo (multithreaded)."""

    def __init__(self, env, nthread: int = 16):
        super().__init__()
        self.env, self.m = env, env.model
        self.nq, self.nv = self.m.nq, self.m.nv
        self.sub = int(env._control_decimation)
        assert self.m.na == 0
        self.datas = [mujoco.MjData(self.m) for _ in range(nthread)]

    def _roll(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """x [N,NX], u [N,14] -> state after one control period [N,NX]."""
        ctrl = np.repeat(policy_to_ctrl(self.env, u)[:, None], self.sub, axis=1)
        st, _ = mj_rollout.rollout(self.m, self.datas, x, ctrl)
        return st[:, -1]

    def forward(self, states, controls):
        M, K, T, NX = states.shape
        x = states[:, :, 0].reshape(M * K, NX).numpy()
        u = controls.reshape(M * K, T, -1).numpy()
        out = np.zeros((M * K, T, NX))
        for t in range(T):
            x = self._roll(x, u[:, t])
            out[:, t] = x
        return torch.from_numpy(out.reshape(M, K, T, NX))

    def step(self, x, u):
        return torch.from_numpy(self._roll(x.view(1, -1).numpy(), u.view(1, -1).numpy())[0])


class KnotCost(torch.nn.Module):
    """Costs: sigma-normalised distance of each rollout state to the coarse knot."""

    def __init__(self, model: mujoco.MjModel, sigma_joint=0.02, sigma_slide=0.003, sigma_joint_vel=0.3,
                 sigma_obj_pos=0.004, sigma_obj_rot=0.1, sigma_obj_vel=0.1, sigma_obj_angvel=1.0):
        super().__init__()
        nq = model.nq
        self.nq = nq
        qi, qs, vi, vs = [], [], [], []
        self.obj_pos, self.obj_quat, self.obj_lin, self.obj_ang = [], [], [], []
        for j in range(model.njnt):
            qa, da = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.obj_pos.append(list(range(qa, qa + 3)))
                self.obj_quat.append(list(range(qa + 3, qa + 7)))
                self.obj_lin.append(list(range(da, da + 3)))
                self.obj_ang.append(list(range(da + 3, da + 6)))
            elif model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                qs.append(qa), vs.append(da)
            else:
                qi.append(qa), vi.append(da)
        t = lambda l: torch.as_tensor(l, dtype=torch.long)  # noqa: E731
        self.q_hinge, self.q_slide, self.v_joint = t(qi), t(qs), t(vi + vs)
        self.obj_pos, self.obj_quat = t(self.obj_pos), t(self.obj_quat)
        self.obj_lin, self.obj_ang = t(self.obj_lin), t(self.obj_ang)
        self.s = dict(joint=sigma_joint, slide=sigma_slide, jvel=sigma_joint_vel, pos=sigma_obj_pos,
                      rot=sigma_obj_rot, vel=sigma_obj_vel, ang=sigma_obj_angvel)
        self.targets = None

    def set_targets(self, tgt):  # [T,NX]
        self.targets = tgt

    def forward(self, states, controls=None):
        nq, s, tg = self.nq, self.s, self.targets
        T = states.shape[2]
        x, g = states[0], tg[:T].unsqueeze(0)  # x [K,T,NX], g [1,T,NX]
        q, gq = x[..., 1:1 + nq], g[..., 1:1 + nq]
        v, gv = x[..., 1 + nq:], g[..., 1 + nq:]
        c = ((q[..., self.q_hinge] - gq[..., self.q_hinge]) / s["joint"]).pow(2).sum(-1)
        c = c + ((q[..., self.q_slide] - gq[..., self.q_slide]) / s["slide"]).pow(2).sum(-1)
        c = c + ((v[..., self.v_joint] - gv[..., self.v_joint]) / s["jvel"]).pow(2).sum(-1)
        if len(self.obj_pos):
            c = c + ((q[..., self.obj_pos] - gq[..., self.obj_pos]) / s["pos"]).pow(2).sum((-1, -2))
            dot = (q[..., self.obj_quat] * gq[..., self.obj_quat]).sum(-1).abs().clamp(max=1.0)
            c = c + ((2 * torch.acos(dot)) / s["rot"]).pow(2).sum(-1)
            c = c + ((v[..., self.obj_lin] - gv[..., self.obj_lin]) / s["vel"]).pow(2).sum((-1, -2))
            c = c + ((v[..., self.obj_ang] - gv[..., self.obj_ang]) / s["ang"]).pow(2).sum((-1, -2))
        return c.sum(-1)  # [K], summed over the horizon


def policy_bounds(env):
    from abc_sim.env import _GRIPPER_CTRL_MAX

    m = env.model
    idx = list(env._ctrl_indices)
    lo = m.actuator_ctrlrange[idx, 0].copy()
    hi = m.actuator_ctrlrange[idx, 1].copy()
    for i in GRIPPER_POLICY_IDX:
        lo[i], hi[i] = 0.0, 1.0
    return lo, hi


def polish(plan: PlanData, env, *, horizon=1, K=256, iters=4, anneal=0.7, sigma_arm=0.03, sigma_grip=0.05,
           nthread=16, seed=0, adaptive_scale=0.3, cost_kwargs=None, log=None):
    """Refine ``plan`` under the env's physics. Returns (actions [N,14], fine qpos [N+1,nq], info)."""
    dyn = CpuAbcFine(env, nthread=nthread)
    cost = KnotCost(env.model, **(cost_kwargs or {}))
    sig = np.zeros(14)
    for arm in plan.arms:
        a = 0 if arm == "left" else 7
        sig[a:a + 6], sig[a + 6] = sigma_arm, sigma_grip
    lo, hi = policy_bounds(env)
    sampling = GaussianSampling(sig, lo, hi, K=K, T=horizon, seed=seed, adaptive_scale=adaptive_scale)
    mppi = SegmentMPPI(dyn, cost, sampling, K=K, T=horizon)
    N = len(plan.actions)
    ref = torch.from_numpy(np.concatenate([np.zeros((N + 1, 1)), plan.qpos, plan.qvel], axis=1))
    U, xs, info = refine_trajectory(mppi, ref[0].clone(), ref, torch.from_numpy(plan.actions), horizon=horizon,
                                    num_iters=iters, anneal=anneal, log=log)
    return U.numpy(), xs[:, 1:1 + env.model.nq].numpy(), info


def replay_cpu(env, plan: PlanData, actions: np.ndarray) -> np.ndarray:
    """Open-loop replay in abc's CPU env from the plan's start state; qpos [N+1,nq]."""
    d = env.data
    d.qpos[:] = plan.qpos[0]
    d.qvel[:] = plan.qvel[0]
    d.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(env.model, d)
    out = [d.qpos.copy()]
    for a in actions:
        env.step(np.asarray(a), render_obs=False)
        out.append(d.qpos.copy())
    return np.stack(out)


def evaluate(env, qpos_seq: np.ndarray):
    """abc's evaluator per frame -> (success final, success any)."""
    ev = env._task_evaluator
    ev.reset(nworld=len(qpos_seq))
    ok = np.asarray(ev.evaluate_qpos_batch(np.asarray(qpos_seq, np.float32)).success, bool)
    return bool(ok[-1]), bool(ok.any())
