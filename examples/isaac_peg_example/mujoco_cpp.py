"""Planner config for the C++ MuJoCo peg environment (cpp/peg_mujoco.h on ighastar's mujoco_env.h).

UNREVIEWED: written by Claude, not yet read or reviewed by a human.

The search then runs entirely in C++ on CPU MuJoCo (no Isaac, no Python callbacks). What it needs from the Isaac side
is a set of constants (frames, offsets, thresholds, OSC gains, sampler ranges), collected by ``constants_from_problem``
from a live PegProblem, or loaded from an .npz written earlier with the same keys; and a CPU MuJoCo model of the scene
with the static bodies (hole, table, robot support) and the robot base as mocap bodies.

    configs = planner_config(consts, planner_cfg, task_cfg, model_path)
    planner = create_planner(configs)            # node_type "mujoco", header peg_mujoco.h
    planner.search(start_state, ...)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
NCTRL = 9
FEATURE_NAMES = (
    "hand_x", "hand_y", "hand_z", "hand_r1", "hand_r2", "hand_r3", "hand_r4", "hand_r5", "hand_r6",
    "peg_x", "peg_y", "peg_z", "peg_axis_x", "peg_axis_y", "peg_axis_z", "grip",
)
PFX = "_World_envs_env_0_"
# Isaac scene entity -> MuJoCo mocap body (the robot base is the articulation root)
MOCAP_BODIES = {"robot": "Robot_base_link", "receptive_object": "ReceptiveObject", "table": "Table",
                "ur5_metal_support": "UR5MetalSupport"}


def _np(x):
    try:
        import torch

        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def constants_from_problem(prob, env) -> dict[str, Any]:
    """Everything peg_mujoco.h needs, from a live PegProblem (after set_start) and its Isaac env."""
    st = env.scene.get_state(is_relative=True)
    c = prob.cfg
    return dict(
        env_origins=_np(prob.env_origins), hole_pose=_np(prob.hole_pose),
        robot_root_pose=_np(st["articulation"]["robot"]["root_pose"][0]),
        rigid_names=np.array(list(st["rigid_object"].keys())),
        rigid_root_poses=np.stack([_np(st["rigid_object"][k]["root_pose"][0]) for k in st["rigid_object"]]),
        ee_offset_pos=np.array(prob.ee_offset.pos), ee_offset_quat=np.array(prob.ee_offset.quat),
        peg_offset_pos=np.array(prob.peg_offset.pos), peg_offset_quat=np.array(prob.peg_offset.quat),
        hole_offset_pos=np.array(prob.hole_offset.pos), hole_offset_quat=np.array(prob.hole_offset.quat),
        peg_local_box=_np(prob.peg_local_box), table_top_z=prob.table_top_z, pos_thr=prob.pos_thr, rot_thr=prob.rot_thr,
        base_quat=_np(prob.base_quat), d_xyz=_np(prob.d_xyz), d_rot=_np(prob.d_rot), lever=_np(prob.lever),
        kp=_np(prob.arm._kp[0]), kd=_np(prob.arm._kd[0]), tau_max=_np(prob.arm._torque_max.reshape(-1, 6)[0]),
        T=prob.T, step_dt=env.step_dt, physics_dt=env.physics_dt, decimation=env.cfg.decimation, edge_dt=prob.edge_dt,
        max_object_step=prob.max_object_step, max_rot_step=prob.max_rot_step,
        n_peg_rows=c.n_peg_rows, n_hole_rows=c.n_hole_rows, num_controls=c.num_controls,
        peg_grasped_dist=c.peg_grasped_dist, hole_jitter=c.hole_jitter, hole_height=c.hole_height,
        max_joint_vel=c.max_joint_vel, max_peg_accel_g=c.max_peg_accel_g, table_z_margin=c.table_z_margin,
        osc_targets=c.osc_targets, joint_names=np.array(list(prob.robot.joint_names)),
        F=prob.F, n=prob.n, o_grip=prob.o_grip, o_q=prob.o_q, o_qd=prob.o_qd, o_peg=prob.o_peg, o_pegv=prob.o_pegv,
        J=prob.J,
    )


def mujoco_params(c: dict, model_path: str, nthread: int = 0, seed: int = 0) -> dict[str, Any]:
    """configs["mujoco"] for peg_mujoco.h: model indices by name + the constants as plain lists/numbers."""
    import mujoco

    if str(c["osc_targets"]) != "absolute":
        raise ValueError("peg_mujoco.h implements osc_targets: absolute only")
    m = mujoco.MjModel.from_binary_path(model_path) if str(model_path).endswith(".mjb") else mujoco.MjModel.from_xml_path(model_path)
    OB, OJ = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT

    def body(name):
        i = mujoco.mj_name2id(m, OB, PFX + name)
        if i < 0:
            raise KeyError(f"body {PFX + name} not in {model_path}")
        return i

    jq, jd = [], []
    for jn in [str(x) for x in c["joint_names"]]:
        j = mujoco.mj_name2id(m, OJ, PFX + "Robot_" + jn)
        if j < 0:
            raise KeyError(f"joint {jn} not in {model_path}")
        jq.append(int(m.jnt_qposadr[j])); jd.append(int(m.jnt_dofadr[j]))
    peg = body("InsertiveObject")
    pj = [j for j in range(m.njnt) if m.jnt_bodyid[j] == peg and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if len(pj) != 1:
        raise ValueError("the peg must have exactly one free joint")
    poses = {"robot": np.asarray(c["robot_root_pose"], float)}
    for name, p in zip([str(x) for x in c["rigid_names"]], np.asarray(c["rigid_root_poses"], float)):
        poses[name] = p
    poses["receptive_object"] = np.asarray(c["hole_pose"], float)
    mocap_ids, mocap_poses = [], []
    for ent, bname in MOCAP_BODIES.items():
        b = body(bname)
        if m.body_mocapid[b] < 0:
            raise ValueError(f"{bname} must be a mocap body in the CPU model")
        mocap_ids.append(int(m.body_mocapid[b])); mocap_poses += poses[ent].tolist()
    lst = lambda k: [float(x) for x in np.asarray(c[k], float).reshape(-1)]  # noqa: E731
    return dict(
        model=str(model_path), nthread=int(nthread), seed=int(seed),
        F=int(c["F"]), o_grip=int(c["o_grip"]), o_q=int(c["o_q"]), o_qd=int(c["o_qd"]), o_peg=int(c["o_peg"]),
        o_pegv=int(c["o_pegv"]), n=int(c["n"]), J=int(c["J"]), joint_qadr=jq, joint_dadr=jd,
        peg_qadr=int(m.jnt_qposadr[pj[0]]), peg_dadr=int(m.jnt_dofadr[pj[0]]),
        wrist_body=body("Robot_wrist_3_link"), hand_body=body("Robot_robotiq_base_link"),
        origin=lst("env_origins")[:3], base_pose=lst("robot_root_pose"), hole_pose=lst("hole_pose"),
        ee_offset_pos=lst("ee_offset_pos"), ee_offset_quat=lst("ee_offset_quat"),
        peg_offset_pos=lst("peg_offset_pos"), peg_offset_quat=lst("peg_offset_quat"),
        hole_offset_pos=lst("hole_offset_pos"), hole_offset_quat=lst("hole_offset_quat"),
        peg_local_box=lst("peg_local_box"), mocap_ids=mocap_ids, mocap_poses=mocap_poses,
        kp=lst("kp"), kd=lst("kd"), tau_max=lst("tau_max"), lever=lst("lever"),
        T=int(c["T"]), decimation=int(c["decimation"]), step_dt=float(c["step_dt"]), edge_dt=float(c["edge_dt"]),
        grip_open=0.0, grip_close=0.785398, d_xyz=lst("d_xyz"), d_rot=lst("d_rot"),
        num_controls=int(c["num_controls"]), n_peg_rows=int(c["n_peg_rows"]), n_hole_rows=int(c["n_hole_rows"]),
        peg_grasped_dist=float(c["peg_grasped_dist"]), hole_jitter=float(c["hole_jitter"]),
        hole_height=float(c["hole_height"]), max_joint_vel=float(c["max_joint_vel"]),
        max_peg_accel_g=float(c["max_peg_accel_g"]), table_top_z=float(c["table_top_z"]),
        table_z_margin=float(c["table_z_margin"]), pos_thr=float(c["pos_thr"]), rot_thr=float(c["rot_thr"]),
        max_object_step=float(c["max_object_step"]), max_rot_step=float(c["max_rot_step"]),
    )


def planner_config(c: dict, planner_cfg: dict, task_cfg: dict | None, model_path: str, nthread: int = 0,
                   seed: int = 0) -> dict[str, Any]:
    """Same experiment_info as PegProblem.planner_config, node_type "mujoco" (peg_mujoco.h)."""
    info = dict(planner_cfg)
    task_cfg = dict(task_cfg or {})
    F, n = int(c["F"]), int(c["n"])

    def hashed(key):
        v = info.get(key, task_cfg.get(key))
        if v is None:
            raise KeyError(f"no '{key}' in planner or task config")
        v = [float(x) for x in v]
        if len(v) != F:
            raise ValueError(f"'{key}' needs {F} values ({', '.join(FEATURE_NAMES)}), got {len(v)}")
        return v + [1.0] * (n - F)

    info.update(state_dim=n, control_dim=NCTRL, hash_dims=F, num_controls=int(c["num_controls"]),
                resolution=hashed("resolution"), tolerance=hashed("tolerance"))
    info["node_info"] = {"node_type": "mujoco", "timesteps": int(c["T"]), "env_header": "peg_mujoco.h",
                         "include_dirs": [str(HERE / "cpp")]}
    return {"experiment_info_default": info, "mujoco": mujoco_params(c, model_path, nthread, seed)}


def build_test_module(n: int = 54, hash_dims: int = 16):
    """Python bindings of the peg env's rollout / encode / heuristic / goal for tests (cpp/peg_mujoco_bind.cpp)."""
    import sys

    sys.path.insert(0, str(HERE.parent.parent))
    from ighastar.scripts.common_utils import BASE_DIR, CXX_STD_FLAG, _ext_load, env_header_flag, mujoco_build_flags

    mj_inc, mj_ld = mujoco_build_flags()
    src = BASE_DIR / "src"
    return _ext_load(
        name=f"peg_mujoco_bind_{n}_{hash_dims}",
        sources=[str(HERE / "cpp" / "peg_mujoco_bind.cpp")],
        extra_include_paths=[str(src / "Environments" / "include"), str(src / "utils"), str(src), str(HERE / "cpp"), mj_inc],
        extra_cflags=[CXX_STD_FLAG, "-O3", "-march=native", "-Wno-attributes", f"-DN_DIMS={n}", f"-DN_CONT={NCTRL}",
                      f"-DHASH_DIMS={hash_dims}"],
        extra_ldflags=mj_ld, verbose=False,
    )
