"""How far does the grasp site really move in one 0.34 s edge when commanded a big TCP step?

IK with joint limits only (no per-edge cap); commands +-x/y/z moves of increasing size from
the settled start pose, simulates one edge in MJWarp, reports realized grasp-site travel.
    python tools/tcp_reach.py [seed]
"""
import os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, yaml
EX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EX))
import abc_sim  # noqa: E402
from abc_problem import PROBLEMS, ProblemConfig, ARM_SLICES  # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 13
cfg = yaml.safe_load(open(EX / "config.yml"))
spec = abc_sim.get_task_spec("put_relative")
env = abc_sim.make_env(task=spec.env_task, prompt=spec.prompt, render_cameras=False)
env.reset(seed=seed, randomize=True)
p = PROBLEMS["put_relative"](env, ProblemConfig.from_dict({**cfg["problem"], "seed": seed, "sampler": "tcp",
                             "tcp_site": "grasp", "tcp_axis_moves": False}), cfg["task"]["put_relative"])
arm = p.active_arms[0]; a = ARM_SLICES[arm].start
ti = 0 if arm == "left" else 1
print(f"seed {seed} arm {arm}; grasp-site travel after ONE 0.34 s edge (joint limits only)")
print(f"{'cmd [cm]':>8s} {'IK ok':>6s} {'median dq':>10s} {'max dq':>8s} {'realized median [cm]':>21s} {'min':>6s}")
ax = torch.cat([torch.eye(3), -torch.eye(3)])
for step in (0.04, 0.08, 0.12, 0.20, 0.30, 0.40):
    u = torch.zeros(6, 14); u[:, a:a + 3] = ax * step
    S = p.start[None].repeat(6, 1)
    uj, ok = p.tcp_to_joint(S, u)
    out = p.dynamics_joint(S, uj)
    tcp0 = S[:, 3 + 3 * ti:6 + 3 * ti]; tcp1 = out[:, -1, 3 + 3 * ti:6 + 3 * ti]
    got = (tcp1 - tcp0).norm(dim=1)[ok]
    dq = uj[:, a:a + 6].abs()[ok]
    print(f"{100 * step:8.0f} {int(ok.sum()):>4d}/6 {dq.median().item():10.2f} {dq.max().item() if len(dq) else 0:8.2f} {100 * got.median().item() if len(got) else float('nan'):21.1f} {100 * got.min().item() if len(got) else float('nan'):6.1f}")
p.close()
