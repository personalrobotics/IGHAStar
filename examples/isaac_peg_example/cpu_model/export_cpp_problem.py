"""Reference data for the C++ MuJoCo peg environment (peg_mujoco.h). Saves cpp_problem.npz:
  constants      everything PegProblem uses (offsets, thresholds, OSC gains, frames, sampler ranges), as arrays
  pairs          Isaac node states (PegProblem layout) and the MuJoCo-Warp qpos/qvel of the same world right after the
                 state write, for start + set A + set B states  -> checks the Isaac <-> MuJoCo state conversion
  python refs    features / heuristic / goal / validity of those states, and GPU dynamics outputs (Isaac layout)
UNREVIEWED.   python cpu_model/export_cpp_problem.py --out <dir> [--start-idx 3261]  (or build_cpu_model.sh)
"""
import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

EX = Path(__file__).resolve().parent.parent  # examples/isaac_peg_example
sys.path.insert(0, str(EX))
sys.path.insert(0, str(EX.parent.parent))

ap = argparse.ArgumentParser()
ap.add_argument("--reset-type", default="ObjectPartiallyAssembledEEGrasped")
ap.add_argument("--start-idx", type=int, default=3261)
ap.add_argument("--config", default=str(EX / "config.yml"))
ap.add_argument("--out", required=True, help="output directory (shared by the cpu_model scripts)")
AppLauncher.add_app_launcher_args(ap)
args, hydra_args = ap.parse_known_args()
for ov in ("env.scene.insertive_object=peg", "env.scene.receptive_object=peghole"):
    if not any(h.startswith(ov.split("=")[0] + "=") for h in hydra_args):
        hydra_args.append(ov)
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import uwlab_tasks  # noqa: E402,F401
import yaml  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import newton_robot_usd  # noqa: E402
from diag_contacts_lib import load_start_states, put_start  # noqa: E402
from peg_problem import PegConfig, PegProblem, apply_arm_sysid, apply_osc_config  # noqa: E402

TASK = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0"
NE = 64


def npy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    cfg = yaml.safe_load(open(args.config))
    env_cfg.scene.num_envs = NE
    env_cfg.episode_length_s = 1.0e6
    env_cfg.terminations.time_out = None
    env_cfg.terminations.abnormal_robot = None
    apply_osc_config(env_cfg, cfg.get("osc"))
    from peg_problem import disable_domain_randomization
    print("[cpu_model] randomization off:", disable_domain_randomization(env_cfg))
    newton_robot_usd.configure_newton(env_cfg, args.reset_type, args.start_idx, OUT / "usd")
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()
    ds, M = load_start_states(env, args.reset_type)
    put_start(env, ds, args.start_idx % M)
    apply_arm_sysid(env, bool(cfg.get("arm_sysid", False)), float(cfg.get("arm_friction_scale", 1.0)))
    pc = {**cfg.get("problem", {}), "num_controls": 64, "seed": 0, "edge_graph": False}
    prob = PegProblem(env, PegConfig.from_dict(pc), cfg.get("task"))
    from isaaclab_newton.physics import NewtonManager as NM
    dw = NM._solver.mjw_data

    caps = []
    ow = prob._write_states

    def wr(s):
        ow(s)
        caps.append((s.detach().cpu().numpy().copy(), dw.qpos.numpy().copy(), dw.qvel.numpy().copy()))
    prob._write_states = wr

    s0 = prob.set_start(None, 0)
    UA = prob.sample_controls()[:NE]
    trA = prob.dynamics(s0[None].repeat(NE, 1), UA)
    okA = torch.isfinite(trA[:, -1]).all(1)
    SB = trA[:, -1].clone(); SB[~okA] = s0
    UB = prob.sample_controls()[:NE]
    trB = prob.dynamics(SB, UB)
    okB = torch.isfinite(trB[:, -1]).all(1)
    SC = trB[:, -1].clone(); SC[~okB] = s0
    prob.dynamics(SC, prob.sample_controls()[:NE])  # only to capture the write of SC
    prob._write_states = ow

    st = env.scene.get_state(is_relative=True)
    S_all = torch.cat([s0[None].repeat(NE, 1), SB, SC])  # states written in the three calls
    feats = S_all[:, :prob.F]
    pegp = S_all[:, prob.o_peg:prob.o_peg + 7].to(prob.device)
    dist, ang = prob.alignment(pegp)
    rec = dict(
        isaac_states=np.stack([c[0] for c in caps]), mj_qpos=np.stack([c[1] for c in caps]), mj_qvel=np.stack([c[2] for c in caps]),
        U_A=npy(UA), U_B=npy(UB), traj_A=npy(trA), traj_B=npy(trB), S_B=npy(SB), start=npy(s0),
        ref_states=npy(S_all), ref_feats=npy(feats), ref_h=npy(prob.heuristic(S_all)), ref_goal=npy(prob.goal_mask(S_all)),
        ref_valid=npy(prob.validity(S_all)), ref_dist=npy(dist), ref_ang=npy(ang),
        env_origins=npy(prob.env_origins), hole_pose=npy(prob.hole_pose),
        robot_root_pose=npy(st["articulation"]["robot"]["root_pose"][0]),
        rigid_names=np.array(list(st["rigid_object"].keys())),
        rigid_root_poses=np.stack([npy(st["rigid_object"][k]["root_pose"][0]) for k in st["rigid_object"]]),
        ee_offset_pos=np.array(prob.ee_offset.pos), ee_offset_quat=np.array(prob.ee_offset.quat),
        peg_offset_pos=np.array(prob.peg_offset.pos), peg_offset_quat=np.array(prob.peg_offset.quat),
        hole_offset_pos=np.array(prob.hole_offset.pos), hole_offset_quat=np.array(prob.hole_offset.quat),
        peg_local_box=npy(prob.peg_local_box), table_top_z=prob.table_top_z, pos_thr=prob.pos_thr, rot_thr=prob.rot_thr,
        base_quat=npy(prob.base_quat), d_xyz=npy(prob.d_xyz), d_rot=npy(prob.d_rot), lever=npy(prob.lever),
        kp=npy(prob.arm._kp[0]), kd=npy(prob.arm._kd[0]), tau_max=npy(prob.arm._torque_max.reshape(-1, 6)[0]),
        T=prob.T, step_dt=env.step_dt, physics_dt=env.physics_dt, decimation=env.cfg.decimation, edge_dt=prob.edge_dt,
        max_object_step=prob.max_object_step, max_rot_step=prob.max_rot_step,
        n_peg_rows=prob.cfg.n_peg_rows, n_hole_rows=prob.cfg.n_hole_rows, num_controls=prob.cfg.num_controls,
        peg_grasped_dist=prob.cfg.peg_grasped_dist, hole_jitter=prob.cfg.hole_jitter, hole_height=prob.cfg.hole_height,
        max_joint_vel=prob.cfg.max_joint_vel, max_peg_accel_g=prob.cfg.max_peg_accel_g, table_z_margin=prob.cfg.table_z_margin,
        osc_targets=prob.cfg.osc_targets, joint_names=np.array(list(prob.robot.joint_names)),
        body_names=np.array(list(prob.robot.body_names)), F=prob.F, n=prob.n, o_grip=prob.o_grip, o_q=prob.o_q,
        o_qd=prob.o_qd, o_peg=prob.o_peg, o_pegv=prob.o_pegv, J=prob.J,
        peg_com_local=npy(prob.peg.data.body_com_pos_b.torch[0].reshape(-1)) if hasattr(prob.peg.data, "body_com_pos_b") else np.zeros(3),
    )
    np.savez(OUT / "cpp_problem.npz", **rec)
    print("[exp] saved", OUT / "cpp_problem.npz", {k: np.shape(v) for k, v in rec.items() if np.size(v) > 1})
    print("[exp] rigid objects", list(st["rigid_object"].keys()), "osc_targets", prob.cfg.osc_targets)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
