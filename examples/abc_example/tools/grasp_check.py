"""Grasp pool for the mover object, and a physical check that the poses really grasp.

    python tools/grasp_check.py [seed]

1. prints the pool size and what the grasps look like,
2. scripted reach: repeatedly move the grasp site toward one grasp pose (<= tcp_step per
   edge), close the gripper, lift 10 cm, and report whether the mover came along.
"""
import os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import torch, yaml

EX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EX))
import abc_sim  # noqa: E402
from abc_problem import PROBLEMS, ProblemConfig  # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 13
cfg = yaml.safe_load(open(EX / "config.yml"))
spec = abc_sim.get_task_spec("put_relative")
env = abc_sim.make_env(task=spec.env_task, prompt=spec.prompt, render_cameras=False)
env.reset(seed=seed, randomize=True)
p = PROBLEMS["put_relative"](env, ProblemConfig.from_dict({**cfg["problem"], "seed": seed, "sampler": "tcp",
                             "tcp_site": "grasp"}), cfg["task"]["put_relative"])
G = p.grasp_pool
print(f"seed {seed}: mover {p.goal['mover']}, grasp pool {len(G)}")
if len(G) == 0:
    sys.exit()
print("approach dirs in object frame (mean, std):", G[:, :3, 2].mean(0).round(2), G[:, :3, 2].std(0).round(2))

from abc_problem import ARM_SLICES  # noqa: E402
arm = p.active_arms[0]; a = ARM_SLICES[arm].start
adr = p.grasp_adr
NTRY = int(os.environ.get("NTRY", 24))
rng = np.random.default_rng(0)
ks = rng.choice(len(G), min(NTRY, len(G)), replace=False)


def edge(S, k=None, delta=None, grip=None):
    u = torch.zeros(1, p.control_dim)
    if k is not None:
        u[0, 14] = k + 1
    if delta is not None:
        u[0, a:a + 6] = torch.as_tensor(delta, dtype=torch.float32)
    if grip is not None:
        u[0, a + 6] = grip
    out = p.dynamics(S, u)
    return out[:, -1], bool(torch.isfinite(out[:, -1]).all())


res = []
for k in ks:
    S, valid = p.start[None], True
    reached_at = None
    for i in range(12):
        S, valid = edge(S, k=k)
        if not valid:
            break
        gnow = float(p.ctrl(S)[0, a + 6])
        if gnow < 0.05:  # gripper commanded closed: the edge after reaching
            reached_at = i
            break
    if not valid or reached_at is None:
        res.append((k, "unreachable" if not valid else "did not reach in 12 edges", None, None))
        continue
    obj0 = p.qpos(S)[0, adr:adr + 3].clone()
    for _ in range(3):  # lift 5 cm per edge, gripper held closed
        S, valid = edge(S, delta=[0, 0, 0.05, 0, 0, 0], grip=0.0)
        if not valid:
            break
    obj1 = p.qpos(S)[0, adr:adr + 3]
    rise = float(obj1[2] - obj0[2])
    res.append((k, "reached" if valid else "lift IK failed", reached_at + 1, rise))
n = len(res)
print(f"{n} grasps tried: " + ", ".join(f"{lab}: {sum(r[1] == lab for r in res)}" for lab in sorted({r[1] for r in res})))
lifted = [r for r in res if r[3] is not None and r[3] > 0.10]
print(f"mover lifted > 10 cm of the 15 cm commanded lift: {len(lifted)}/{n}; rises [cm]: "
      + " ".join(f"{100 * r[3]:.0f}" for r in res if r[3] is not None))
print("edges to reach (those that did):", [r[2] for r in res if r[2]])
p.close()
