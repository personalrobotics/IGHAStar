"""Time one 17-substep control period of the MJWarp scene under solver/integrator variants.

Each variant edits a copy of the MuJoCo model's ``opt`` before MJWarp builds its
model, then times graph replays at NW worlds with a random action ramp (so
contacts are exercised). Also reports how far each variant's end state drifts
from the baseline's (same start, same controls), as a fidelity sanity check.

    NW=64 python tools/speed_sweep.py
"""
import copy
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import torch
import warp as wp
import yaml

EX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EX))
import abc_sim  # noqa: E402
from abc_problem import PROBLEMS, ProblemConfig  # noqa: E402
from abc_sim.rendering.replay.renderer import WarpReplayRuntime  # noqa: E402

NW = int(os.environ.get("NW", 64))
cfg = yaml.safe_load(open(EX / "config.yml"))
spec = abc_sim.get_task_spec("put_relative")
env = abc_sim.make_env(task=spec.env_task, prompt=spec.prompt, render_cameras=False)
env.reset(seed=13, randomize=True)
prob = PROBLEMS["put_relative"](
    env, ProblemConfig.from_dict({**cfg["problem"], "seed": 13, "rollout_capacities": [NW]}),
    cfg["task"]["put_relative"])
mjm0, mjd = env.model, env.data
DB = mujoco.mjtDisableBit
q0 = prob.qpos(prob.start[None]).cuda().expand(NW, -1).contiguous()
v0 = prob.qvel(prob.start[None]).cuda().expand(NW, -1).contiguous()
c0 = prob.policy_to_actuator(prob.ctrl(prob.start[None]).cuda())
g = torch.Generator().manual_seed(0)
du = (torch.rand(NW, 14, generator=g) * 2 - 1) * prob.delta_scale
c1 = prob.policy_to_actuator(torch.minimum(torch.maximum(prob.ctrl(prob.start[None]) + du, prob.policy_lo), prob.policy_hi).cuda())
c0 = c0.expand(NW, -1)


def run(name, edit=None, substeps=17, periods=10, graph_conditional=None):
    m = copy.copy(mjm0)
    if edit:
        edit(m)
    rt = WarpReplayRuntime(m, mjd, nworld=NW, nconmax=128, njmax=1200)
    rt.sync_from_mujoco()
    if graph_conditional is not None:
        rt.m_warp.opt.graph_conditional = graph_conditional
    d = rt.d_warp
    qp, qv, ct = wp.to_torch(d.qpos), wp.to_torch(d.qvel), wp.to_torch(d.ctrl)
    qp.copy_(q0); qv.copy_(v0)
    for n in ("qacc_warmstart", "qacc", "act", "time"):
        a = getattr(d, n, None)
        if a is not None and a.size:
            wp.to_torch(a).zero_()

    def edge():
        for t in range(periods):
            ct.copy_(c0 + (t + 1) / periods * (c1 - c0))
            rt.step(nstep=substeps)

    edge(); wp.synchronize()  # capture
    qp.copy_(q0); qv.copy_(v0)
    wp.synchronize()
    t0 = time.perf_counter()
    edge(); wp.synchronize()
    dt = time.perf_counter() - t0
    end = qp.clone()
    rt.close()
    return dt, end


results = {}
base_dt, base_q = run("base")
print(f"NW={NW}  one edge = 10 periods x 17 substeps")
print(f"{'variant':44s} {'ms/edge':>8s} {'speedup':>8s} {'max|dq| vs base':>16s}")


def report(name, dt, end):
    err = (end - base_q).abs().nan_to_num(1e9).max().item()
    print(f"{name:44s} {1e3 * dt:8.1f} {base_dt / dt:8.2f}x {err:16.2e}")


report("baseline", base_dt, base_q)
variants = [
    ("integrator=EULER", lambda m: setattr(m.opt, "integrator", mujoco.mjtIntegrator.mjINT_EULER), {}),
    ("iterations=3", lambda m: setattr(m.opt, "iterations", 3), {}),
    ("iterations=3, no graph conditional", lambda m: setattr(m.opt, "iterations", 3), {"graph_conditional": False}),
    ("iterations=1, no graph conditional", lambda m: setattr(m.opt, "iterations", 1), {"graph_conditional": False}),
    ("ls_iterations=10", lambda m: setattr(m.opt, "ls_iterations", 10), {}),
    ("ls_iterations=10, iterations=3, no cond", lambda m: (setattr(m.opt, "ls_iterations", 10), setattr(m.opt, "iterations", 3)), {"graph_conditional": False}),
    ("disable EULERDAMP|FILTERPARENT", lambda m: setattr(m.opt, "disableflags", int(DB.mjDSBL_EULERDAMP | DB.mjDSBL_FILTERPARENT)), {}),
    ("jacobian=dense", lambda m: setattr(m.opt, "jacobian", mujoco.mjtJacobian.mjJAC_DENSE), {}),
    ("solver=CG", lambda m: setattr(m.opt, "solver", mujoco.mjtSolver.mjSOL_CG), {}),
    ("cone=elliptic", lambda m: setattr(m.opt, "cone", mujoco.mjtCone.mjCONE_ELLIPTIC), {}),
    ("dt 2ms -> 3.4ms, 10 substeps (1.7x fewer)", lambda m: setattr(m.opt, "timestep", 0.0034), {"substeps": 10}),
    ("dt 2ms -> 4.25ms, 8 substeps (2.1x fewer)", lambda m: setattr(m.opt, "timestep", 0.00425), {"substeps": 8}),
    ("dt 2ms -> 6.8ms, 5 substeps (3.4x fewer)", lambda m: setattr(m.opt, "timestep", 0.0068), {"substeps": 5}),
    ("dt 2ms -> 8.5ms, 4 substeps (4.2x fewer)", lambda m: setattr(m.opt, "timestep", 0.0085), {"substeps": 4}),
]
only = os.environ.get("ONLY")
for name, edit, kw in variants:
    if only and only not in name:
        continue
    try:
        dt, end = run(name, edit, **kw)
        report(name, dt, end)
    except Exception as e:  # noqa: BLE001
        print(f"{name:44s} FAILED: {type(e).__name__}: {str(e)[:80]}")
