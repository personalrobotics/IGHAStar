"""Expansion rate of the same edge rollout on CPU (mujoco.rollout, multithreaded).

One edge = 10 control periods x 17 substeps = 170 physics steps per world, K worlds.
    NT=16 python tools/cpu_rollout_bench.py
"""
import os, sys, time, copy
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
from mujoco import rollout
import numpy as np
import torch, yaml

EX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EX))
import abc_sim  # noqa: E402
from abc_problem import PROBLEMS, ProblemConfig  # noqa: E402

cfg = yaml.safe_load(open(EX / "config.yml"))
spec = abc_sim.get_task_spec("put_relative")
env = abc_sim.make_env(task=spec.env_task, prompt=spec.prompt, render_cameras=False)
env.reset(seed=13, randomize=True)
prob = PROBLEMS["put_relative"](env, ProblemConfig.from_dict({**cfg["problem"], "seed": 13}),
                                cfg["task"]["put_relative"])
m = env.model
K, T, SUB = 64, 10, 17
nstep = T * SUB
s = prob.start[None]
u = prob.sample_controls()
c0 = prob.ctrl(s).expand(K, -1)
c1 = torch.minimum(torch.maximum(c0 + u, prob.policy_lo), prob.policy_hi)
a0, a1 = prob.policy_to_actuator(c0).numpy(), prob.policy_to_actuator(c1).numpy()
alpha = (np.arange(nstep) // SUB + 1) / T  # ramp per control period, as in the GPU path
ctrl = a0[:, None] + alpha[None, :, None] * (a1 - a0)[:, None]
q, v = prob.qpos(s).numpy(), prob.qvel(s).numpy()
x0 = np.concatenate([[0.0], q[0], v[0]])  # [time, qpos, qvel] (no act state)
assert m.na == 0
x0 = np.tile(x0, (K, 1))
print(f"model dt={m.opt.timestep}, nstep/edge={nstep}, K={K}, cpu cores={os.cpu_count()}")
for nt in [int(x) for x in os.environ.get("NT", "1,8,16,32").split(",")]:
    ds = [mujoco.MjData(m) for _ in range(nt)]
    rollout.rollout(m, ds, x0, ctrl)  # warm
    t0 = time.perf_counter(); n = 3
    for _ in range(n):
        st, _ = rollout.rollout(m, ds, x0, ctrl)
    dt = (time.perf_counter() - t0) / n
    print(f"threads={nt:3d}: {1e3 * dt:7.1f} ms per {K}-control expansion -> {1 / dt:5.2f} expansions/s")
