"""Frame, timing and branch-continuity check of tcp_ik.TcpIK (frax) against MuJoCo.

    python tools/tcp_ik_check.py
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

EX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EX))
import abc_sim  # noqa: E402
from tcp_ik import TcpIK  # noqa: E402

spec = abc_sim.get_task_spec("put_relative")
env = abc_sim.make_env(task=spec.env_task, prompt=spec.prompt, render_cameras=False)
env.reset(seed=13, randomize=True)
m, d0 = env.model, env.data
rng = np.random.default_rng(0)
ik = TcpIK(m, d0, ["left", "right"])
for side in ("left", "right"):
    arm = ik.arms[side]
    jadr = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint{k}")]) for k in range(1, 7)]
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_grasp_site")
    q_home = d0.qpos[jadr].copy()
    err = 0.0
    for _ in range(20):
        q = q_home + rng.uniform(-0.4, 0.4, 6)
        d = mujoco.MjData(m); d.qpos[:] = d0.qpos; d.qpos[jadr] = q; mujoco.mj_kinematics(m, d)
        T = np.eye(4); T[:3, :3] = d.site_xmat[sid].reshape(3, 3); T[:3, 3] = d.site_xpos[sid]
        err = max(err, float(np.abs(arm.grasp_world(q) - T).max()))
    print(f"{side}: max |FK(grasp pose) - MuJoCo| over 20 configs = {err:.2e}")
    N = 64
    qs = np.tile(q_home, (N, 1))
    inf = np.full(6, np.inf)
    for step in (0.04, 0.15):
        deltas = np.zeros((N, 6)); v = rng.normal(size=(N, 3)); deltas[:, :3] = step * v / np.linalg.norm(v, axis=1, keepdims=True)
        deltas[N // 2:, 3:] = rng.normal(size=(N - N // 2, 3)) * 0.1   # second half also rotates
        ik.solve_deltas(side, qs, deltas, inf)  # compile / warm up
        t = time.perf_counter(); dq, ok = ik.solve_deltas(side, qs, deltas, inf); dt = time.perf_counter() - t
        got = np.array([np.linalg.norm(arm.grasp_world(q_home + dq[i])[:3, 3] - arm.grasp_world(q_home)[:3, 3]) for i in np.flatnonzero(ok)])
        print(f"   step {100 * step:4.0f} cm: {1e3 * dt:6.1f} ms/64 rows; solved {ok.sum()}/{N}; max|dq| median {np.median(np.abs(dq[ok]).max(1)):.2f} "
              f"p90 {np.quantile(np.abs(dq[ok]).max(1), .9):.2f} max {np.abs(dq[ok]).max():.2f} rad; realized TCP travel median {100 * np.median(got):.1f} cm")
