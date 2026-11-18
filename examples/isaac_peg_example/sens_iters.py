"""Edge outcomes for a fixed set of (node, control) pairs, to compare solver settings across processes. UNREVIEWED.

Writes out[start] = [N, T, n] (invalid rows are NaN) for each start state. Controls are identical across runs (same seed, same
start state). Compare runs with sens_compare.py.

    python sens_iters.py --robot-iters 8 --out r8.npz
"""
import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

ap = argparse.ArgumentParser()
ap.add_argument("--num-envs", type=int, default=64)
ap.add_argument("--robot-iters", type=int, default=None, help="DIAGNOSTIC ONLY: override the robot solver position iterations")
ap.add_argument("--starts", nargs="+", default=["ObjectPartiallyAssembledEEGrasped:3261", "ObjectAnywhereEEAnywhere:0"])
ap.add_argument("--batches", type=int, default=4, help="distinct control batches per start state")
ap.add_argument("--out", required=True)
ap.add_argument("--config", default=str(HERE / "config.yml"))
AppLauncher.add_app_launcher_args(ap)
args, hydra_args = ap.parse_known_args()
for ov in ("env.scene.insertive_object=peg", "env.scene.receptive_object=peghole"):
    if not any(h.startswith(ov.split("=")[0] + "=") for h in hydra_args):
        hydra_args.append(ov)
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import uwlab_tasks  # noqa: E402,F401
import yaml  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from diag_contacts_lib import load_start_states, put_start  # noqa: E402
from peg_problem import PegConfig, PegProblem  # noqa: E402

TASK = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0"


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    cfg = yaml.safe_load(open(args.config))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = 1.0e6
    env_cfg.terminations.time_out = None
    env_cfg.terminations.abnormal_robot = None
    if args.robot_iters is not None:
        env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = args.robot_iters
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()
    res = {}
    for spec in args.starts:
        rt, idx = spec.split(":")
        ds, M = load_start_states(env, rt)
        put_start(env, ds, int(idx) % M)
        prob = PegProblem(env, PegConfig.from_dict({**cfg.get("problem", {}), "num_controls": args.num_envs, "seed": 0}), cfg.get("task"))
        start = prob.set_start(None, 0)
        outs, ctrls = [], []
        for _ in range(args.batches):
            U = prob.sample_controls()[:prob.N]
            outs.append(prob.dynamics(start[None].repeat(prob.N, 1), U).numpy())
            ctrls.append(U.numpy())
        res[spec] = np.concatenate(outs)
        res[spec + "|u"] = np.concatenate(ctrls)
        res[spec + "|start"] = start.numpy()
        print(f"{spec}: {len(res[spec])} edges, invalid {int(np.isnan(res[spec]).any(axis=(1, 2)).sum())}")
    res["F"] = np.asarray(prob.F)
    res["feature_names"] = np.asarray(prob.feature_names)
    res["robot_iters"] = np.asarray(-1 if args.robot_iters is None else args.robot_iters)
    np.savez_compressed(args.out, **res)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
