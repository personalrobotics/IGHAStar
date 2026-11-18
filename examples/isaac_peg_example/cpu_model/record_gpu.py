"""GPU (Newton / MuJoCo-Warp) reference edges for the CPU MuJoCo prototype. Saves:
  cpu_scene.mjb   Newton's single-world MuJoCo model with world-0 sysid dof params, timestep 1/120, and per-geom
                  contype/conaffinity bitmasks that reproduce Newton's explicit contact-pair list exactly
  gpu_edges.npz   per edge: MuJoCo qpos/qvel after the state write, the resolved hand-delta action (base frame) and
                  gripper command, MuJoCo qpos/qvel after each policy step; plus the OSC / gripper / frame constants
UNREVIEWED.   python cpu_model/record_gpu.py --out <dir> [--start-idx 3261]  (or build_cpu_model.sh)
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

import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import uwlab_tasks  # noqa: E402,F401
import yaml  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import newton_robot_usd  # noqa: E402
from diag_contacts_lib import load_start_states, put_start  # noqa: E402
from peg_problem import NCTRL, PegConfig, PegProblem, apply_arm_sysid, apply_osc_config  # noqa: E402

TASK = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0"
NE = 64


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
    print(f"[rec] osc_targets {prob.cfg.osc_targets} N {prob.N} T {prob.T} decimation {env.cfg.decimation} dt {env.physics_dt}")
    from isaaclab_newton.physics import NewtonManager as NM
    sol = NM._solver
    m, mw, dw = sol.mj_model, sol.mjw_model, sol.mjw_data
    nq, nv = m.nq, m.nv

    # ---- hooks: MuJoCo state after the write, after every policy step; the resolved action
    rec = {"q0": [], "v0": [], "act": [], "cmd": [], "qs": [], "vs": []}
    cur = {}
    ow, orr, osim = prob._write_states, prob._read_states, prob._simulate_edge

    def wr(s):
        ow(s)
        cur["q0"], cur["v0"] = dw.qpos.numpy().copy(), dw.qvel.numpy().copy()
        cur["qs"], cur["vs"] = [], []

    def rd(cmd):
        out = orr(cmd)
        if "qs" in cur:
            cur["qs"].append(dw.qpos.numpy().copy()); cur["vs"].append(dw.qvel.numpy().copy())
        return out

    def simu(act, cmd):
        cur["act"], cur["cmd"] = act.cpu().numpy().copy(), cmd.cpu().numpy().copy()
        r = osim(act, cmd)
        for k in ("q0", "v0", "act", "cmd"):
            rec[k].append(cur[k])
        rec["qs"].append(np.stack(cur["qs"], 1)); rec["vs"].append(np.stack(cur["vs"], 1))
        return r
    prob._write_states, prob._read_states, prob._simulate_edge = wr, rd, simu

    s0 = prob.set_start(None, 0)
    B = NE
    # A: sampler controls from the start
    UA = prob.sample_controls()[:B]
    t0 = time.perf_counter()
    trA = prob.dynamics(s0[None].repeat(B, 1), UA)
    print(f"[rec] edge set A (sampler from start) {1e3 * (time.perf_counter() - t0):.0f} ms")
    # B: sampler controls from A's end states
    okA = torch.isfinite(trA[:, -1]).all(1)
    SB = trA[:, -1].clone(); SB[~okA] = s0
    UB = prob.sample_controls()[:B]
    prob.dynamics(SB, UB)
    # C: directional, gripper closed: +-x,y,z translation 20 mm/step, +-rx,ry,rz 0.1 rad/step (base frame), hold
    rows = []
    for k in range(6):
        for sg in (1, -1):
            u = torch.zeros(NCTRL); u[6] = -1.0
            u[k] = sg * (0.02 if k < 3 else 0.1)
            rows.append(u)
    u = torch.zeros(NCTRL); u[6] = -1.0; rows.append(u)
    UC = torch.stack(rows); UC = torch.cat([UC, UC[:1].repeat(B - len(rows), 1)])
    prob.dynamics(s0[None].repeat(B, 1), UC)
    # timing of the same call (eager per-step graph path)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(3):
        prob.dynamics(s0[None].repeat(B, 1), UA)
    torch.cuda.synchronize()
    print(f"[rec] GPU dynamics call (per-step graph, {B} envs): {(time.perf_counter() - t0) / 3 * 1e3:.1f} ms")
    # ---- model: sysid dof params, timestep, contact pairs exactly as Newton's list (world 0)
    def w0(arr, n):  # noqa: E306
        a = arr.numpy()
        return a.reshape(-1, n)[0] if a.size > n and a.size % n == 0 else a.reshape(-1)[:n]
    # copy every MuJoCo-Warp model field that is per-world (or global) and shaped like the template's field
    changed = []
    for name in dir(m):
        if name.startswith("_") or not hasattr(mw, name):
            continue
        tv = getattr(m, name)
        if not isinstance(tv, np.ndarray) or tv.size == 0 or not np.issubdtype(tv.dtype, np.number):
            continue
        try:
            a = getattr(mw, name).numpy()
        except Exception:  # noqa: BLE001
            continue
        if a.size == tv.size:
            src = a.reshape(tv.shape)
        elif a.size % tv.size == 0 and a.shape[0] == a.size // tv.size:
            src = a.reshape(a.shape[0], *tv.shape)[0]
        else:
            continue
        src = src.astype(tv.dtype)
        if not np.array_equal(src, tv) and not (np.issubdtype(tv.dtype, np.floating) and np.allclose(src, tv, rtol=1e-6, atol=1e-9)):
            diff = float(np.abs(src.astype(float) - tv.astype(float)).max()) if src.shape == tv.shape else -1
            changed.append(f"{name}({diff:.3g})")
            tv[...] = src
    print(f"[rec] copied from mjw world 0 (field(max abs change)): {' '.join(changed)}")
    m.opt.timestep = env.physics_dt
    pairs = NM._model.shape_contact_pairs.numpy()
    g2s = sol.mjc_geom_to_newton_shape.numpy()[0]  # [ngeom] newton shape of world 0 (-1 = none)
    s2g = {int(s): g for g, s in enumerate(g2s) if s >= 0}
    allowed = set()
    for a, b in pairs:
        if int(a) in s2g and int(b) in s2g:
            ga, gb = s2g[int(a)], s2g[int(b)]
            allowed.add((min(ga, gb), max(ga, gb)))
    assert m.ngeom <= 31, m.ngeom
    ct = np.zeros(m.ngeom, np.int64); ca = np.zeros(m.ngeom, np.int64)
    for g in range(m.ngeom):
        ct[g] = 1 << g
    for ga, gb in allowed:
        ca[ga] |= 1 << gb; ca[gb] |= 1 << ga
    m.geom_contype[:] = ct; m.geom_conaffinity[:] = ca
    bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b).replace("_World_envs_env_0_", "")  # noqa: E731
    print(f"[rec] Newton contact pairs (world 0 geoms): {len(allowed)}")
    for ga, gb in sorted(allowed):
        print(f"[rec]   pair {ga}:{bname(m.geom_bodyid[ga])}[{m.geom_type[ga]}] - {gb}:{bname(m.geom_bodyid[gb])}[{m.geom_type[gb]}]")
    print(f"[rec] geom margin {np.unique(np.round(m.geom_margin, 4)).tolist()} gap {np.unique(np.round(m.geom_gap, 4)).tolist()} "
          f"solref {np.unique(np.round(m.geom_solref, 4), axis=0).tolist()} solimp {np.unique(np.round(m.geom_solimp, 4), axis=0).tolist()} "
          f"friction {np.unique(np.round(m.geom_friction, 3), axis=0).tolist()} condim {np.unique(m.geom_condim).tolist()}")
    print(f"[rec] mjw margins {np.unique(np.round(w0(mw.geom_margin, m.ngeom), 4)).tolist()} "
          f"mjw geom_friction {np.unique(np.round(mw.geom_friction.numpy().reshape(-1, 3), 3), axis=0).tolist()}")
    print(f"[rec] opt: solver {m.opt.solver} iter {m.opt.iterations} ls {m.opt.ls_iterations} tol {m.opt.tolerance} "
          f"cone {m.opt.cone} impratio {m.opt.impratio} integrator {m.opt.integrator}")
    nm = NM._model
    for a in dir(nm):
        if ("margin" in a or "gap" in a or "thickness" in a) and not a.startswith("_"):
            v = getattr(nm, a)
            try:
                print(f"[rec] newton model {a}: {np.unique(np.round(v.numpy(), 5)).tolist()[:8]}")
            except Exception:  # noqa: BLE001
                print(f"[rec] newton model {a}: {v}")
    for fld in ("geom_friction", "body_mass", "actuator_gainprm", "actuator_biasprm", "dof_frictionloss", "dof_damping", "geom_solref"):
        a = getattr(mw, fld).numpy()
        if a.ndim >= 2 and a.shape[0] > 1:
            print(f"[cpu_model] per-world spread of {fld}: max std across worlds {float(a.std(0).max()):.3g} (shape {a.shape})")
        else:
            print(f"[cpu_model] {fld}: shared across worlds (shape {a.shape})")
    mujoco.mj_saveModel(m, str(OUT / "cpu_scene.mjb"))

    for k in rec:
        rec[k] = np.concatenate(rec[k][:3], 0)  # sets A, B, C only

    arm = prob.arm
    root_q = prob.robot.data.root_quat_w.torch[0].cpu().numpy()  # xyzw
    root_p = (prob.robot.data.root_pos_w.torch[0]).cpu().numpy()
    wrist = arm._ee_body_idx if hasattr(arm, "_ee_body_idx") else None
    wname = prob.robot.body_names[wrist] if wrist is not None else "wrist_3_link"
    print(f"[rec] OSC body {wname}; kp {arm._kp[0].cpu().numpy().tolist()} kd {np.round(arm._kd[0].cpu().numpy(), 2).tolist()} "
          f"tau_max {arm._torque_max.reshape(-1, 6)[0].cpu().numpy().tolist()} scale {prob.scale.cpu().numpy().tolist()} lever {prob.lever.cpu().numpy().tolist()}")
    ew = prob.robot.data.body_pos_w.torch[0, wrist].cpu().numpy()
    print(f"[rec] Isaac env0 root pos {np.round(root_p, 4).tolist()} quat(xyzw) {np.round(root_q, 4).tolist()}; wrist world pos now {np.round(ew, 4).tolist()}")
    gterm = env.action_manager.get_term("gripper")
    print(f"[rec] gripper term {type(gterm).__name__} open {getattr(gterm, '_open_command', None)} close {getattr(gterm, '_close_command', None)}")
    jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    free_body = [bname(m.jnt_bodyid[j]) for j in range(m.njnt) if m.jnt_type[j] == 0]
    print(f"[rec] free joints bodies {free_body}")
    np.savez(OUT / "gpu_edges.npz", **rec, kp=arm._kp[0].cpu().numpy(), kd=arm._kd[0].cpu().numpy(),
             tau_max=arm._torque_max.reshape(-1, 6)[0].cpu().numpy(), lever=prob.lever.cpu().numpy(),
             root_quat_xyzw=root_q, root_pos=root_p, wrist_name=f"_World_envs_env_0_Robot_{wname}",
             decimation=env.cfg.decimation, dt=env.physics_dt, T=prob.T, grip_open=0.0, grip_close=0.785398,
             jnt_names=np.array(jn), free_bodies=np.array(free_body), max_joint_vel=prob.cfg.max_joint_vel,
             max_peg_accel_g=prob.cfg.max_peg_accel_g, ngpu=np.array([B, B, B]))
    print("[rec] saved", OUT / "gpu_edges.npz", {k: v.shape for k, v in rec.items()})
    env.close()


if __name__ == "__main__":
    main()
    app.close()
