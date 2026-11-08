"""GPU kernel breakdown of one MJWarp control period (17 substeps, 64 worlds).

    nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node -t cuda \
        -o step64 -f true python tools/nsys_step.py
    nsys stats -r cuda_gpu_kern_sum step64.nsys-rep
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import torch
import yaml

EX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EX))
import abc_sim  # noqa: E402
import warp as wp  # noqa: E402
from abc_problem import PROBLEMS, ProblemConfig  # noqa: E402

cfg = yaml.safe_load(open(EX / "config.yml"))
spec = abc_sim.get_task_spec("put_relative")
env = abc_sim.make_env(task=spec.env_task, prompt=spec.prompt, render_cameras=False)
env.reset(seed=13, randomize=True)
prob = PROBLEMS["put_relative"](
    env, ProblemConfig.from_dict({**cfg["problem"], "seed": 13}), cfg["task"]["put_relative"])
rt = prob.rollout.runtimes[64]
rt.step(nstep=17)
wp.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for _ in range(20):
    rt.step(nstep=17)
wp.synchronize()
torch.cuda.cudart().cudaProfilerStop()
