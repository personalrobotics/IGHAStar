"""Where does one IGHA* expansion's time go? (rendering check + physics vs Python split). UNREVIEWED diagnostic.

Runs prob.dynamics on a batch of N identical start nodes with random controls and times each stage with CUDA syncs.

    python profile_dyn.py --reset-type ObjectPartiallyAssembledEEGrasped --start-idx 3261 --num-envs 64
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

ap = argparse.ArgumentParser()
ap.add_argument("--num-envs", type=int, default=64)
ap.add_argument("--reset-type", default="ObjectPartiallyAssembledEEGrasped")
ap.add_argument("--start-idx", type=int, default=3261)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--robot-iters", type=int, default=None, help="DIAGNOSTIC ONLY: override the robot articulation solver position iterations")
ap.add_argument("--obj-iters", type=int, default=None, help="DIAGNOSTIC ONLY: override peg/hole rigid-body solver position iterations")
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
import torch  # noqa: E402
import uwlab_tasks  # noqa: E402,F401
import yaml  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from peg_problem import PegConfig, PegProblem  # noqa: E402
from diag_contacts_lib import load_start_states, put_start  # noqa: E402

TASK = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0"
T_ACC = defaultdict(float)
N_ACC = defaultdict(int)


def timed(obj, name, label=None):
    """Wrap obj.name so its wall time (GPU-synced) accumulates under label."""
    fn = getattr(obj, name, None)
    if fn is None:
        return
    label = label or f"{type(obj).__name__}.{name}"

    def w(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize(); T_ACC[label] += time.perf_counter() - t; N_ACC[label] += 1
        return r

    setattr(obj, name, w)


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    cfg = yaml.safe_load(open(args.config))
    env_cfg.scene.num_envs = args.num_envs
    if args.robot_iters is not None:
        env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = args.robot_iters
    if args.obj_iters is not None:
        for nm in ("insertive_object", "receptive_object"):
            getattr(env_cfg.scene, nm).spawn.rigid_props.solver_position_iteration_count = args.obj_iters
    env_cfg.episode_length_s = 1.0e6
    env_cfg.terminations.time_out = None
    env_cfg.terminations.abnormal_robot = None
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()
    sim = env.sim
    print("\n=== rendering ===")
    print("launcher: visualizer =", getattr(args, "visualizer", None), "| experience =", getattr(args, "experience", None))
    print("sim.has_gui:", sim.has_gui, "| sim.is_rendering:", sim.is_rendering, "| has_active_visualizers:", sim.has_active_visualizers(),
          "| env.has_rtx_sensors:", env.has_rtx_sensors, "| render_interval:", env.cfg.sim.render_interval)
    print("scene sensors:", list(env.scene.sensors.keys()), "| physics handles decimation:", getattr(env, "_physics_handles_decimation", "n/a"))
    print("env class:", type(env).__name__, "| decimation:", env.cfg.decimation, "| physics dt:", env.physics_dt)
    ds, M = load_start_states(env, args.reset_type)
    put_start(env, ds, args.start_idx % M)
    prob = PegProblem(env, PegConfig.from_dict({**cfg.get("problem", {}), "num_controls": args.num_envs}), cfg.get("task"))
    start = prob.set_start(None, 0)
    N = prob.N
    S = start[None].repeat(N, 1)
    U = prob.sample_controls()[:N]

    for name in ("process_action", "apply_action"):
        timed(env.action_manager, name, f"action_manager.{name}")
    timed(env.scene, "write_data_to_sim", "scene.write_data_to_sim")
    timed(env.scene, "update", "scene.update")
    timed(env.sim, "step", "sim.step (PhysX)")
    timed(env.sim, "render", "sim.render")
    for mgr in ("observation_manager", "reward_manager", "termination_manager", "event_manager", "command_manager", "curriculum_manager"):
        m = getattr(env, mgr, None)
        if m is not None:
            timed(m, "compute", f"{mgr}.compute")
            timed(m, "apply", f"{mgr}.apply")
    timed(env, "step", "env.step (total)")
    for name in ("_resolve", "_write_states", "_read_states", "validity"):
        timed(prob, name, f"prob.{name}")

    prob.dynamics(S, U)  # warm-up
    T_ACC.clear(); N_ACC.clear()
    timed(prob, "dynamics", "prob.dynamics (total)")
    torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStart(); t0 = time.perf_counter()
    for _ in range(args.reps):
        prob.dynamics(S, U)
    torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize(); wall = (time.perf_counter() - t0) / args.reps
    tot = T_ACC["prob.dynamics (total)"] / args.reps
    print(f"\n=== one batch: {N} envs x {prob.T} policy steps ({prob.T * env.cfg.decimation} physics steps = {prob.edge_dt:.2f}s sim) ===")
    print(f"wall per dynamics call {wall:.3f} s   (instrumented total {tot:.3f} s; syncs add some overhead)")
    for k, v in sorted(T_ACC.items(), key=lambda kv: -kv[1]):
        print(f"  {k:42s} {v / args.reps * 1e3:9.1f} ms/call  x{N_ACC[k] // args.reps:3d} calls  = {100 * v / args.reps / tot:5.1f}% of dynamics")
    ph = T_ACC["sim.step (PhysX)"] / args.reps
    es = T_ACC["env.step (total)"] / args.reps
    print(f"\nphysics (sim.step): {ph:.3f} s = {100 * ph / tot:.0f}% | env.step minus sim.step (managers, buffers): {es - ph:.3f} s = "
          f"{100 * (es - ph) / tot:.0f}% | rest of dynamics (my Python: resolve/write/read/validity/checks): {tot - es:.3f} s = {100 * (tot - es) / tot:.0f}%")
    print(f"per physics step: {ph / (prob.T * env.cfg.decimation) * 1e3:.2f} ms  (real time would be {env.physics_dt * 1e3:.2f} ms)")
    # pure PhysX: 60 sim.step calls with no controller, no managers, no Python in between
    sim_step = type(env.sim).step
    env.scene.write_data_to_sim()
    for _ in range(5):
        sim_step(env.sim, render=False)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(60):
        sim_step(env.sim, render=False)
    torch.cuda.synchronize(); pure = (time.perf_counter() - t) / 60
    print(f"pure sim.step loop (no controller/managers, whatever efforts are already set): {pure * 1e3:.2f} ms/step")
    print(f"SUMMARY envs={N} robot_iters={args.robot_iters} obj_iters={args.obj_iters} start={args.start_idx}: "
          f"call {wall * 1e3:.0f} ms | physx {ph * 1e3:.0f} ms ({ph / 60 * 1e3:.2f}/step) | controller {T_ACC['action_manager.apply_action'] / args.reps * 1e3:.0f} ms | pure physx {pure * 1e3:.2f} ms/step")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
