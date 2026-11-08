"""Where does an IGHA* expansion on abc put_relative spend its time?

[A] runs a short search with every callback timed (plus the GPU rollout, synced).
[B] times MJWarp graph launches at each rollout capacity, then each stage of
    one physics substep at 64 worlds (ungraphed, so CPU launch overhead is
    included; use nsys_step.py for GPU-side kernel times).

    NEXP=150 python tools/profile_expansion.py
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import torch
import yaml

EX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EX))
import abc_sim  # noqa: E402
from abc_problem import PROBLEMS, ProblemConfig  # noqa: E402
from ighastar.scripts.common_utils import create_planner  # noqa: E402

SEED, NEXP = int(os.environ.get("SEED", 13)), int(os.environ.get("NEXP", 150))
cfg = yaml.safe_load(open(EX / "config.yml"))
spec = abc_sim.get_task_spec("put_relative")
env = abc_sim.make_env(task=spec.env_task, prompt=spec.prompt, render_cameras=False)
env.reset(seed=SEED, randomize=True)
pcfg = ProblemConfig.from_dict({**cfg["problem"], "seed": SEED})
prob = PROBLEMS["put_relative"](env, pcfg, cfg["task"]["put_relative"])
m = prob.mjm
print(f"model: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} nmesh={m.nmesh} "
      f"integrator={m.opt.integrator} solver={m.opt.solver} iters={m.opt.iterations} "
      f"ls_iters={m.opt.ls_iterations} cone={m.opt.cone} timestep={m.opt.timestep}")

# ---------------- A: search-level breakdown via wrapped callbacks
configs = prob.planner_config(dict(cfg["planner"]))
T = {}


def wrap(name, fn):
    def f(*a):
        t0 = time.perf_counter()
        r = fn(*a)
        T.setdefault(name, [0.0, 0])
        T[name][0] += time.perf_counter() - t0
        T[name][1] += 1
        return r
    return f


info = configs["experiment_info_default"]
for k in [k for k in configs if k.endswith("_fn")]:
    configs[k] = wrap(k, configs[k])
info["max_expansions"] = NEXP
planner = create_planner(configs, bidirectional=False)

orig_rollout = prob.rollout.rollout


def timed_rollout(*a):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = orig_rollout(*a)
    torch.cuda.synchronize()
    T.setdefault("  rollout(GPU, synced)", [0.0, 0])
    T["  rollout(GPU, synced)"][0] += time.perf_counter() - t0
    T["  rollout(GPU, synced)"][1] += 1
    return r


prob.rollout.rollout = timed_rollout

t0 = time.perf_counter()
planner.search(prob.start, torch.zeros(prob.n), torch.zeros(1), NEXP, int(info["hysteresis"]), True)
wall = time.perf_counter() - t0
exp = planner.get_profiler_info()[7]
print(f"\n[A] search: {exp} expansions in {wall:.1f}s = {exp / wall:.1f} exp/s")
cbsum = sum(v[0] for k, v in T.items() if not k.startswith(" "))
for k, (s, n) in sorted(T.items(), key=lambda kv: -kv[1][0]):
    print(f"   {k:28s} {s:7.2f}s  {100 * s / wall:5.1f}%  calls={n:6d}  {1000 * s / max(n, 1):8.2f} ms/call")
print(f"   {'planner (C++ + marshaling)':28s} {wall - cbsum:7.2f}s  {100 * (wall - cbsum) / wall:5.1f}%")

# ---------------- B: physics-level breakdown
import mujoco_warp as mjw  # noqa: E402
import warp as wp  # noqa: E402
from mujoco_warp._src import collision_driver, solver  # noqa: E402
from mujoco_warp._src import forward as F  # noqa: E402

sub = env._control_decimation
print(f"\n[B] MJWarp, {sub} substeps (one control period) per graph launch")
for cap in prob.rollout.capacities:
    rt = prob.rollout.runtimes[cap]
    d, v = prob.rollout._views(cap)
    v["qpos"].copy_(prob.qpos(prob.start[None]).cuda().expand(cap, -1))
    v["qvel"].zero_()
    for _ in range(3):
        rt.step(nstep=sub)  # capture + warm up
    wp.synchronize()
    n = 30
    t0 = time.perf_counter()
    for _ in range(n):
        rt.step(nstep=sub)
    wp.synchronize()
    dt = (time.perf_counter() - t0) / n
    print(f"   nworld={cap:5d}: {1000 * dt:7.2f} ms/control period -> {cap / dt:9.0f} world-ctrl-steps/s")

cap = 64
rt = prob.rollout.runtimes[cap]
mw, dw = rt.m_warp, rt.d_warp
stages = {
    "fwd_position (kin+collision+constraints)": lambda: F.fwd_position(mw, dw, factorize=False),
    "  collision only": lambda: collision_driver.collision(mw, dw),
    "fwd_velocity": lambda: F.fwd_velocity(mw, dw),
    "fwd_actuation": lambda: F.fwd_actuation(mw, dw),
    "fwd_acceleration": lambda: F.fwd_acceleration(mw, dw, factorize=True),
    "solver.solve": lambda: solver.solve(mw, dw),
    "full mjw.step (no graph)": lambda: mjw.step(mw, dw),
}
for _ in range(5):
    mjw.step(mw, dw)
wp.synchronize()
print("\n   stage timing at nworld=64, per physics substep (ungraphed, launch overhead included):")
for name, fn in stages.items():
    fn()
    wp.synchronize()
    n = 40
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    wp.synchronize()
    print(f"   {name:42s} {1e3 * (time.perf_counter() - t0) / n:7.3f} ms")
nacon = wp.to_torch(dw.nacon).cpu().numpy()
nefc = wp.to_torch(dw.nefc).cpu().numpy()
print(f"   contacts (total over worlds)={int(nacon.sum())}  nefc/world mean={nefc.mean():.0f} max={nefc.max()}")
if hasattr(dw, "solver_niter"):
    it = wp.to_torch(dw.solver_niter).cpu().numpy()
    print(f"   solver iterations used: mean={it.mean():.1f} max={it.max()}")
