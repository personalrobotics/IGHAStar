"""Does one big CUDA graph (170 substeps) beat ten 17-substep graphs? (It doesn't.)

Also times one full ``dynamics()`` call, so the difference to pure stepping is
the torch glue (ctrl ramp, recording, overflow check, pack, .cpu()).
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

cfg = yaml.safe_load(open(EX / "config.yml"))
spec = abc_sim.get_task_spec("put_relative")
env = abc_sim.make_env(task=spec.env_task, prompt=spec.prompt, render_cameras=False)
env.reset(seed=13, randomize=True)
prob = PROBLEMS["put_relative"](
    env, ProblemConfig.from_dict({**cfg["problem"], "seed": 13}), cfg["task"]["put_relative"])
s = prob.start[None].repeat(prob.cfg.num_controls, 1)
u = prob.sample_controls()


def t(fn, n=10):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0) / n


for cap in (64, 1024):
    rt = prob.rollout.runtimes[cap]
    a = t(lambda: [rt.step(nstep=17) for _ in range(10)])
    b = t(lambda: rt.step(nstep=170))
    print(f"nworld={cap}: 10 graphs x 17 substeps {a:6.1f} ms | 1 graph x 170 substeps {b:6.1f} ms")
print(f"full dynamics() call ({len(s)} rows, incl. ctrl ramp, recording, pack, .cpu()): "
      f"{t(lambda: prob.dynamics(s, u)):.1f} ms")
