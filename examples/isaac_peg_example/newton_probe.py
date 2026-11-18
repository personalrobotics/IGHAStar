"""Run the planner's edge simulation on Newton (MuJoCo-Warp) instead of PhysX and time it. UNREVIEWED, exploratory.

    python newton_probe.py --num-envs 64 [--backend newton|physx]
"""
import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
ap = argparse.ArgumentParser()
ap.add_argument("--num-envs", type=int, default=64)
ap.add_argument("--backend", default="newton", choices=["newton", "physx"])
ap.add_argument("--reset-type", default="ObjectPartiallyAssembledEEGrasped")
ap.add_argument("--start-idx", type=int, default=3261)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--robot-usd", default=None, help="replacement robot USD (copy with the gripper loop joints' bodies swapped)")
ap.add_argument("--no-gravity", action="store_true", help="DIAGNOSTIC: zero gravity for everything (the robot asset has disable_gravity=True, a per-body PhysX flag)")
ap.add_argument("--substeps", type=int, default=1)
ap.add_argument("--armature", type=float, default=None, help="DIAGNOSTIC: arm joint armature [kg m^2]")
ap.add_argument("--no-self-collision", action="store_true", help="DIAGNOSTIC: disable robot self collisions")
ap.add_argument("--grip", type=float, default=1.0, help="gripper command for the zero-motion test steps (<0 close, >=0 open)")
ap.add_argument("--trace", type=int, default=0, help="print gripper joint q/qd for this many physics steps of a zero-motion policy step")
ap.add_argument("--grip-armature", type=float, default=None, help="DIAGNOSTIC: armature of the driven finger joint [kg m^2]")
ap.add_argument("--newton-contacts", action="store_true", help="use Newton's collision pipeline (mesh SDF) instead of MuJoCo contacts")
ap.add_argument("--save", default=None, help="save start node, controls and edge outcomes (for compare_backends.py)")
ap.add_argument("--auto-usd", action="store_true", help="build the Newton robot USD for this start state (newton_robot_usd.py)")
ap.add_argument("--njmax", type=int, default=300)
ap.add_argument("--nconmax", type=int, default=100)
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

from diag_contacts_lib import load_start_states, put_start  # noqa: E402
from peg_problem import PegConfig, PegProblem  # noqa: E402

GRIP = None
TASK = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0"


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    global GRIP
    GRIP = args.grip
    cfg = yaml.safe_load(open(args.config))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = 1.0e6
    env_cfg.terminations.time_out = None
    env_cfg.terminations.abnormal_robot = None
    if args.backend == "newton":
        from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

        kw = {}
        if args.newton_contacts:
            from isaaclab_newton.physics import NewtonCollisionPipelineCfg
            kw["collision_cfg"] = NewtonCollisionPipelineCfg()
        env_cfg.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(integrator="implicitfast", njmax=args.njmax, nconmax=args.nconmax,
                                       impratio=10.0, cone="elliptic", use_mujoco_contacts=not args.newton_contacts),
            num_substeps=args.substeps, **kw)
    if args.no_gravity:
        env_cfg.sim.gravity = (0.0, 0.0, 0.0)
    if args.armature is not None:
        env_cfg.scene.robot.actuators["arm"].armature = args.armature
    if args.no_self_collision:
        env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    if args.grip_armature is not None:
        env_cfg.scene.robot.actuators["gripper"].armature = args.grip_armature
    if args.auto_usd:
        import torch as _t
        from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
        from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils as ou

        import newton_robot_usd
        pair = ou.compute_pair_dir(env_cfg.scene.insertive_object.spawn.usd_path, env_cfg.scene.receptive_object.spawn.usd_path)
        data = _t.load(ou.safe_retrieve_file_path(f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset_isaaclab3/Resets/{pair}/resets_{args.reset_type}.pt"))
        rp = data["initial_state"]["articulation"]["robot"]["root_pose"][args.start_idx % len(data["initial_state"]["articulation"]["robot"]["root_pose"])]
        args.robot_usd = newton_robot_usd.build(str(Path(__file__).resolve().parent / "outputs" / "newton_usd" / f"auto_{args.reset_type}_{args.start_idx}"), rp[:3].tolist())
        print("[probe] built Newton robot USD with base at", rp[:3].tolist(), "->", args.robot_usd)
    if args.robot_usd:
        env_cfg.scene.robot.spawn.usd_path = args.robot_usd
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()
    print(f"[probe] backend={args.backend} physics cfg = {type(env.cfg.sim.physics).__name__}, dt={env.physics_dt}, decimation={env.cfg.decimation}")
    ds, M = load_start_states(env, args.reset_type)
    put_start(env, ds, args.start_idx % M)
    rp = ds["initial_state"]["articulation"]["robot"]["root_pose"]
    print(f"[probe] dataset robot root pose over {len(rp)} start states: min {rp.min(0).values.cpu().numpy().round(4)} max {rp.max(0).values.cpu().numpy().round(4)}")
    prob = PegProblem(env, PegConfig.from_dict({**cfg.get("problem", {}), "num_controls": args.num_envs}), cfg.get("task"))
    start = prob.set_start(None, 0)
    N = prob.N
    S = start[None].repeat(N, 1)
    U = prob.sample_controls()[:N]
    print(f"[probe] start node: hand {S[0, prob.i_hand].numpy().round(3)} peg {S[0, prob.o_peg:prob.o_peg + 3].numpy().round(3)} "
          f"|qd| max {float(S[0, prob.o_qd:prob.o_qd + prob.J].abs().max()):.3f}")
    prob._write_states(S.to(env.device))
    back = prob._read_states(torch.ones(N, device=env.device)).cpu()
    print(f"[probe] write->read round trip: hand err {float((back[:, prob.i_hand] - S[:, prob.i_hand]).abs().max()):.2e}, "
          f"peg err {float((back[:, prob.o_peg:prob.o_peg + 7] - S[:, prob.o_peg:prob.o_peg + 7]).abs().max()):.2e}")
    print("[probe] initial joint q: " + ", ".join(f"{n} {float(v):+.3f}" for n, v in zip(prob.robot.joint_names, S[0, prob.o_q:prob.o_q + prob.J])))
    try:
        from isaaclab_newton.physics import NewtonManager
        m = NewtonManager._model
        for nm in ("constraint_mimic_joint0", "constraint_mimic_joint1", "constraint_mimic_coef0", "constraint_mimic_coef1", "constraint_mimic_enabled"):
            if hasattr(m, nm):
                v = getattr(m, nm); print(f"[probe] newton {nm}: {v.numpy()[:12] if v is not None else None}")
        print("[probe] newton joint_label:", [str(x).split('/')[-1] for x in m.joint_label[:20]] if hasattr(m, "joint_label") else "n/a")
    except Exception as e:  # noqa: BLE001
        print("[probe] newton model inspect failed:", e)
    def loop_residual(tag):
        from isaaclab.utils import math as mu
        d = prob.robot.data; names = list(prob.robot.body_names)
        pos, quat = d.body_link_pos_w.torch[0], d.body_link_quat_w.torch[0]
        def pt(b, lp):
            i = names.index(b); return pos[i] + mu.quat_apply(quat[i:i + 1], torch.tensor([lp], device=pos.device, dtype=pos.dtype))[0]
        out = []
        for side, sy in (("left", 1.0), ("right", -1.0)):
            r1 = (pt(f"{side}_outer_finger", [0.0981, sy * 0.0678, -0.0133]) - pt(f"{side}_inner_finger", [0.0981, sy * 0.0678, -0.0133])).norm()
            r2 = (pt(f"{side}_inner_finger", [0.1046, sy * 0.0499, 0.0]) - pt(f"{side}_inner_knuckle", [0.1046, sy * 0.0499, 0.0])).norm()
            r3 = (pt("robotiq_base_link", [0.0615, sy * 0.0128, 0.0]) - pt(f"{side}_inner_knuckle", [0.0612, sy * 0.0127, 0.0])).norm()
            out.append(f"{side}: outer_finger-inner_finger pivot {float(r1) * 1e3:.2f} mm, inner_finger-inner_knuckle pivot {float(r2) * 1e3:.2f} mm, base-inner_knuckle pivot {float(r3) * 1e3:.2f} mm")
        print(f"[probe] loop residuals {tag}: " + " | ".join(out))
    rd = prob.robot.data
    print("[probe] robot root pose (world) pos", rd.root_pos_w.torch[0].cpu().numpy().round(4), "quat xyzw", rd.root_quat_w.torch[0].cpu().numpy().round(4),
          "| template root_pose", prob.template["articulation"]["robot"]["root_pose"][0].cpu().numpy().round(4),
          "| base_link body pos", rd.body_link_pos_w.torch[0, 0].cpu().numpy().round(4), "quat", rd.body_link_quat_w.torch[0, 0].cpu().numpy().round(4),
          "| env origin", prob.env_origins[0].cpu().numpy())
    try:
        rp0 = prob.template["articulation"]["robot"]["root_pose"].clone().to(env.device)
        rp0[:, :3] += prob.env_origins
        prob.robot.write_root_link_pose_to_sim_index(root_pose=rp0)
        env.sim.forward() if hasattr(env.sim, "forward") else None
        env.scene.update(env.physics_dt)
        print("[probe] after explicit root write: base_link body pos", prob.robot.data.body_link_pos_w.torch[0, 0].cpu().numpy().round(4) - prob.env_origins[0].cpu().numpy(), "(wanted", prob.template["articulation"]["robot"]["root_pose"][0, :3].cpu().numpy().round(4), ")")
    except Exception as e:  # noqa: BLE001
        print("[probe] explicit root write failed:", e)
    loop_residual("after state write")
    import itertools
    jn = list(prob.robot.joint_names)
    cols = [jn.index(n) for n in ("right_inner_knuckle_joint", "right_inner_finger_knuckle_joint", "right_outer_knuckle_joint")]
    for sg in itertools.product((1, -1), repeat=3):
        S2 = S.clone().to(env.device)
        for c, sgn in zip(cols, sg):
            S2[:, prob.o_q + c] *= sgn
        prob._write_states(S2)
        print("[probe] right-side sign flips (inner_knuckle, finger_knuckle, outer_knuckle) =", sg, end=" -> ")
        loop_residual("")
    prob._write_states(S.to(env.device))
    d0 = prob.robot.data
    for nm in ("joint_armature", "joint_friction_coeff", "joint_damping", "joint_stiffness"):
        try:
            v = getattr(d0, nm); v = v.torch if hasattr(v, "torch") else v
            print(f"[probe] {nm} (env0, arm 6 joints): {v[0, :6].cpu().numpy().round(4)}")
        except Exception as e:  # noqa: BLE001
            print(f"[probe] {nm}: {e}")
    try:
        d = prob.robot.data
        m = d.body_mass.torch[0].cpu().numpy() if hasattr(d.body_mass, "torch") else d.body_mass[0].cpu().numpy()
        print("[probe] robot body masses (env0):", {n: round(float(x), 4) for n, x in zip(prob.robot.body_names, m)})
    except Exception as e:  # noqa: BLE001
        print("[probe] mass read failed:", e)
    try:
        inr = d.body_inertia.torch[0].cpu().numpy().reshape(len(prob.robot.body_names), -1)
        for n, row in zip(prob.robot.body_names, inr):
            if "robotiq" in n or "knuckle" in n or "finger" in n:
                print(f"[probe] inertia {n:22s} {row.round(8)}")
    except Exception as e:  # noqa: BLE001
        print("[probe] inertia read failed:", e)
    # physics only, no controller: does a bare sim step stay finite?
    for k in range(3):
        env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(dt=env.physics_dt)
    cur0 = prob._read_states(torch.ones(N, device=env.device)).cpu()
    print(f"[probe] bare sim steps (zero effort): finite {bool(torch.isfinite(cur0).all())}")
    prob._write_states(S.to(env.device))
    if args.trace:
        prob._write_states(S.to(env.device))
        fp = prob.fast; env.action_manager.process_action(prob._wrist_action(torch.cat([torch.zeros(N, 6, device=env.device), torch.full((N, 1), GRIP, device=env.device)], 1)))
        fp._copy_policy_inputs()
        jn2 = list(prob.robot.joint_names)
        for k in range(int(args.trace)):
            fp.asset.set_joint_effort_target(fp.compute_torques(), joint_ids=fp.ids)
            for t in fp.others:
                t.apply_actions()
            env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(dt=env.physics_dt)
            q = prob.robot.data.joint_pos.torch[0].cpu(); qd = prob.robot.data.joint_vel.torch[0].cpu()
            print(f"[trace] step {k:2d}: " + " ".join(f"{n.replace('_joint','')[:14]} {float(q[i]):+.3f}/{float(qd[i]):+.1f}" for i, n in enumerate(jn2[6:], 6)))
        prob._write_states(S.to(env.device))
    # one zero-action policy step: does anything blow up?
    a0 = torch.zeros(N, 7, device=env.device); a0[:, 6] = GRIP
    for k in range(3):
        prob.fast.step(prob._wrist_action(a0)) if prob.fast is not None else env.step(prob._wrist_action(a0))
        cur = prob._read_states(torch.ones(N, device=env.device)).cpu()
        print(f"[probe] zero-action step {k}: finite {bool(torch.isfinite(cur).all())}, hand drift {float((cur[:, prob.i_hand] - S[:, prob.i_hand]).norm(dim=1).max()) * 1e3:.2f} mm, "
              f"peg drift {float((cur[:, prob.o_peg:prob.o_peg + 3] - S[:, prob.o_peg:prob.o_peg + 3]).norm(dim=1).max()) * 1e3:.2f} mm, "
              f"|qd| max {float(cur[:, prob.o_qd:prob.o_qd + prob.J].abs().max()):.2f}")
    prob._write_states(S.to(env.device))
    a0 = torch.zeros(N, 7, device=env.device); a0[:, 6] = GRIP
    prob.fast.step(prob._wrist_action(a0))
    loop_residual("after one zero-action policy step")
    cur = prob._read_states(torch.ones(N, device=env.device)).cpu()
    qd = cur[0, prob.o_qd:prob.o_qd + prob.J]; dq = cur[0, prob.o_q:prob.o_q + prob.J] - S[0, prob.o_q:prob.o_q + prob.J]
    print("[probe] per-joint after one zero-action policy step (env0): " + ", ".join(f"{n}: qd {float(v):+.2f} dq {float(x):+.3f}" for n, v, x in zip(prob.robot.joint_names, qd, dq)))
    out = prob.dynamics(S, U)  # warm-up
    print(f"[probe] bad rows {prob.profile['bad_rows']}/{N}, peg-accel rows {prob.profile.get('accel_rows', 0)}")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(args.reps):
        out = prob.dynamics(S, U)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / args.reps
    if args.save:
        import numpy as np
        np.savez_compressed(args.save, S=S.numpy(), U=U.numpy(), out=out.numpy(), F=prob.F, o_peg=prob.o_peg, names=np.asarray(prob.feature_names))
    ok = torch.isfinite(out[:, -1]).all(dim=1)
    if not ok.any():
        print(f"[probe] RESULT backend={args.backend} envs={N}: every edge invalid; one dynamics call {dt * 1e3:.0f} ms -> {1 / dt:.2f} expansions/s")
        env.close(); return
    hs = (out[:, -1, prob.i_hand] - S[:, prob.i_hand]).norm(dim=1)[ok]
    ps = (out[:, -1, prob.o_peg:prob.o_peg + 3] - S[:, prob.o_peg:prob.o_peg + 3]).norm(dim=1)[ok]
    print(f"[probe] RESULT backend={args.backend} envs={N}: one dynamics call (one expansion of {N} edges) = {dt * 1e3:.0f} ms -> {1 / dt:.2f} expansions/s; "
          f"valid {int(ok.sum())}/{N}; hand step median {float(hs.median()) * 1e3:.0f} mm max {float(hs.max()) * 1e3:.0f} mm; peg step median {float(ps.median()) * 1e3:.1f} mm max {float(ps.max()) * 1e3:.0f} mm")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
