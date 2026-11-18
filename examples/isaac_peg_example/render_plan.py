"""Render a planned trajectory in Isaac Sim by STATE replay (no physics, no controls): every frame writes the planner's
node state (arm + gripper joints, peg pose; hole, table, support and robot base from the problem constants) into one env
of OmniReset's RGB data-collection task and renders its cameras (front, side, wrist; the cameras the visuo-motor policy
is distilled on).

UNREVIEWED: written by Claude, not yet read or reviewed by a human.

    python render_plan.py --plan <plan.npz> --consts <problem.npz> --out <dir> [--interp 3] [--fps 30]

--plan: an .npz with "states" [N, 54] in PegProblem's node layout (plan_peg.py / plan_peg_cpp.py output; one state per
policy step). --consts: hole / robot root / table / support poses (mujoco_cpp.constants_from_problem keys).
Writes <out>/<camera>.mp4, <out>/mosaic.mp4 and <out>/frames.npz (uint8 images per camera + the replayed states).
"""
import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--consts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--interp", type=int, default=1, help="frames per policy step (>1: interpolate between node states)")
ap.add_argument("--fps", type=float, default=None, help="video frame rate (default: real time = interp / 0.1 s)")
ap.add_argument("--task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DataCollection-v0")
ap.add_argument("--reset-type", default="ObjectPartiallyAssembledEEGrasped")
ap.add_argument("--warmup", type=int, default=2, help="renders per frame (the RTX pipeline lags a frame behind)")
ap.add_argument("--res-scale", type=float, default=1.0, help="camera resolution multiplier (1 = the policy's 320x240)")
ap.add_argument("--camera-randomization", action="store_true",
                help="keep the task's per-reset camera pose / focal-length randomization (default: nominal cameras)")
ap.add_argument("--sync-dt", type=float, default=1e-5, help="physics dt of the per-frame sync step [s]")
AppLauncher.add_app_launcher_args(ap)
args, hydra_args = ap.parse_known_args()
args.enable_cameras = True
for ov in ("env.scene.insertive_object=peg", "env.scene.receptive_object=peghole"):
    if not any(h.startswith(ov.split("=")[0] + "=") for h in hydra_args):
        hydra_args.append(ov)
if not any("reset_from_reset_states.params.reset_types" in h for h in hydra_args):
    hydra_args += [f"env.events.reset_from_reset_states.params.reset_types=[{args.reset_type}]",
                   "env.events.reset_from_reset_states.params.probs=[1.0]"]
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import uwlab_tasks  # noqa: E402,F401
from uwlab_tasks.utils.hydra import hydra_task_config  # noqa: E402

# Planning-asset joint value -> render-asset joint value. newton_robot_usd.py swaps body0/body1 of the right four-bar
# closing joint (Newton refuses PhysX's reversed joint), which flips that joint's angle sign; the left one is re-framed to
# keep it. Checked against PhysX's grasped reset states (finger 0.78: right_inner_finger_knuckle -0.78 in PhysX, +0.78 in
# the planning model with the stock mimic sign).
PLAN_TO_RENDER_SIGN = {"right_inner_finger_knuckle_joint": -1.0}

# PegProblem node layout
O_Q, O_PEG, J = 17, 41, 12


def slerp(q0, q1, a):
    """xyzw quaternions, a in [0, 1]."""
    q0, q1 = q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)


def frames_from_states(S, k):
    """Node states [N, n] -> per-frame (joint_pos [12], peg pose [7]); k frames per policy step."""
    out = [(S[0, O_Q:O_Q + J], S[0, O_PEG:O_PEG + 7])]
    for i in range(1, len(S)):
        for j in range(1, k + 1):
            a = j / k
            q = (1 - a) * S[i - 1, O_Q:O_Q + J] + a * S[i, O_Q:O_Q + J]
            p = (1 - a) * S[i - 1, O_PEG:O_PEG + 3] + a * S[i, O_PEG:O_PEG + 3]
            out.append((q, np.concatenate([p, slerp(S[i - 1, O_PEG + 3:O_PEG + 7], S[i, O_PEG + 3:O_PEG + 7], a)])))
    return out


def strip_mimic_joints(src: str, dst: Path) -> str:
    """Write a USD layer over the robot asset with PhysX mimic joints removed and return its path. The asset couples the
    Robotiq linkage with hard mimic constraints (q_j = -gearing * q_finger); the planner's model (MuJoCo / Newton)
    has soft equalities instead, so a planned gripper state is not exactly on PhysX's constraint manifold and the
    correction during the sync step kicks the whole articulation (arm joints by up to 0.09 rad). Without mimic every
    joint just takes the written value. The asset's metadata.yaml (sysid, read by the task's events from the USD's
    directory) is copied alongside."""
    import shutil

    from isaaclab.utils.assets import retrieve_file_path
    from pxr import Sdf, Usd

    dst.parent.mkdir(parents=True, exist_ok=True)
    meta = src.rsplit("/", 1)[0] + "/metadata.yaml"
    try:
        shutil.copy(retrieve_file_path(meta), dst.parent / "metadata.yaml")
    except FileNotFoundError:
        pass

    src_stage = Usd.Stage.Open(src)
    layer = Sdf.Layer.CreateNew(str(dst))
    layer.subLayerPaths.append(src_stage.GetRootLayer().identifier)
    layer.defaultPrim = src_stage.GetDefaultPrim().GetName()
    n = 0
    for prim in src_stage.Traverse():
        ops = prim.GetMetadata("apiSchemas")
        if ops is None:
            continue
        if prim.GetAttribute("physics:excludeFromArticulation").Get():  # four-bar loop closures: the plan sets every joint
            spec = Sdf.CreatePrimInLayer(layer, prim.GetPath())
            spec.specifier = Sdf.SpecifierOver
            attr = Sdf.AttributeSpec(spec, "physics:jointEnabled", Sdf.ValueTypeNames.Bool)
            attr.default = False
            n += 1
            continue
        items = list(ops.GetAddedOrExplicitItems())
        keep = [x for x in items if not x.startswith("PhysxMimicJointAPI")]
        if len(keep) != len(items):
            spec = Sdf.CreatePrimInLayer(layer, prim.GetPath())
            spec.specifier = Sdf.SpecifierOver
            spec.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(keep))
            n += 1
    layer.Save()
    print(f"[render] robot asset without mimic / loop-closure joints ({n} changed): {dst}")
    return str(dst)


def render_only_physics(env_cfg, out: Path):
    """Physics in this script only carries the written state to the renderer (one tiny step per frame, see main): no
    gravity, no collisions (the planner's peg / finger / hole contacts would be resolved and push bodies apart), no
    mimic constraints."""
    import isaaclab.sim as sim_utils

    env_cfg.sim.gravity = (0.0, 0.0, 0.0)
    for name in dir(env_cfg.scene):
        spawn = getattr(getattr(env_cfg.scene, name, None), "spawn", None)
        if spawn is not None and hasattr(spawn, "collision_props"):
            spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    robot_spawn = env_cfg.scene.robot.spawn
    robot_spawn.usd_path = strip_mimic_joints(robot_spawn.usd_path, out / "robot_asset" / "robot_render.usda")


CAMERA_POSE_EVENTS = ("randomize_front_camera", "randomize_side_camera", "randomize_wrist_camera")


def fix_camera_events(env_cfg, keep: bool):
    """UWLab's randomize_tiled_cameras reads base_rotation as (w, x, y, z) and writes it to the camera prim, but the task
    passes the TiledCameraCfg offset quaternions, which Isaac Lab 3.0 stores as (x, y, z, w): every reset then turns the
    cameras away from the workspace. keep=False drops the camera pose / focal-length events (the cameras stay at their
    configured offsets); keep=True reorders base_rotation so the randomization is centred on the configured pose."""
    ev = env_cfg.events
    for name in CAMERA_POSE_EVENTS:
        term = getattr(ev, name, None)
        if term is None:
            continue
        if keep:
            x, y, z, w = term.params["base_rotation"]
            term.params["base_rotation"] = (w, x, y, z)
        else:
            setattr(ev, name, None)
            setattr(ev, name + "_focal_length", None)


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fix_camera_events(env_cfg, keep=args.camera_randomization)
    policy_dt = env_cfg.sim.dt * env_cfg.decimation  # video real-time rate, before the sync dt override below
    env_cfg.sim.dt = args.sync_dt
    render_only_physics(env_cfg, out)
    if args.res_scale != 1:  # same field of view, more pixels (the policy's cameras are 320x240)
        for k in ("front_camera", "side_camera", "wrist_camera"):
            cam = getattr(env_cfg.scene, k, None)
            if cam is not None:
                cam.width, cam.height = int(cam.width * args.res_scale), int(cam.height * args.res_scale)
    env_cfg.episode_length_s = 1.0e6
    env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "abnormal_robot"):
        env_cfg.terminations.abnormal_robot = None
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()
    sc = env.scene
    c = np.load(args.consts)
    S = np.load(args.plan)["states"].astype(np.float32)
    jn = [str(x) for x in c["joint_names"]]
    robot = sc["robot"]
    # the plan's joint order is the planning asset's (Newton's patched USD); the render asset may order them differently
    missing = set(robot.joint_names) ^ set(jn)
    if missing:
        raise RuntimeError(f"joint names differ between the plan and the render asset: {sorted(missing)}")
    perm = [jn.index(n) for n in robot.joint_names]  # render joint i <- plan joint perm[i]
    sign = np.array([PLAN_TO_RENDER_SIGN.get(n, 1.0) for n in robot.joint_names], np.float32)
    finger_bodies = [robot.body_names.index(n) for n in ("robotiq_base_link", "left_inner_finger", "right_inner_finger")]
    body_log = []
    if perm != list(range(len(jn))):
        print(f"[render] joint order remapped by name: {list(robot.joint_names)}")
    fixed = {"receptive_object": c["hole_pose"]}
    for name, pose in zip([str(x) for x in c["rigid_names"]], c["rigid_root_poses"]):
        if name not in ("insertive_object", "receptive_object"):
            fixed[name] = pose
    cams = [k for k, s in sc.sensors.items() if hasattr(s, "data") and hasattr(s.data, "output")]
    print(f"[render] {len(S)} node states, {args.interp} frames per step; cameras {cams}")
    dev = env.device
    T = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=dev)[None]  # noqa: E731
    imgs = {k: [] for k in cams}
    drift = []  # per frame: max joint error [rad], peg position error [m] of the rendered state vs the plan
    for q, peg in frames_from_states(S, args.interp):
        st = sc.get_state(is_relative=True)
        r = st["articulation"]["robot"]
        r["root_pose"] = T(c["robot_root_pose"]); r["root_velocity"] = torch.zeros_like(r["root_velocity"])
        r["joint_position"] = T(np.asarray(q)[perm] * sign); r["joint_velocity"] = torch.zeros_like(r["joint_velocity"])
        ro = st["rigid_object"]
        ro["insertive_object"]["root_pose"] = T(peg)
        for name, pose in fixed.items():
            if name in ro:
                ro[name]["root_pose"] = T(pose)
        for name in ro:
            ro[name]["root_velocity"] = torch.zeros_like(ro[name]["root_velocity"])
        sc.reset_to(st, env_ids=None, is_relative=True)
        robot.set_joint_position_target(r["joint_position"])  # the gripper drive holds the written pose
        robot.set_joint_effort_target(torch.zeros_like(r["joint_position"]))
        sc.write_data_to_sim()
        # The renderer reads body transforms from fabric, which PhysX only refreshes on a step (sim.forward() and a
        # PhysX -> USD push both left the default pose on screen). One step of sync_dt with nothing acting moves nothing.
        env.sim.step(render=False)
        sc.update(env.physics_dt)
        drift.append([(robot.data.joint_pos.torch[0] - r["joint_position"][0]).abs().cpu().numpy(),
                      float((sc["insertive_object"].data.root_pose_w.torch[0, :3] - sc.env_origins[0]
                             - ro["insertive_object"]["root_pose"][0, :3]).norm())])
        body_log.append(robot.data.body_link_pose_w.torch[0, finger_bodies].cpu().numpy())
        for _ in range(args.warmup):
            env.sim.render()
            sc.update(env.physics_dt)
        for k in cams:
            imgs[k].append(sc.sensors[k].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
    dj, dp = np.stack([x[0] for x in drift]).max(0), max(x[1] for x in drift)
    print(f"[render] rendered vs planned state: peg max {dp * 1e3:.4f} mm, joint max [rad] "
          + ", ".join(f"{n} {e:.1e}" for n, e in zip(robot.joint_names, dj)))
    fps = args.fps or args.interp / policy_dt
    for k in cams:
        imageio.mimwrite(out / f"{k}.mp4", imgs[k], fps=fps, macro_block_size=1)
    if cams:
        mosaic = [np.concatenate([imgs[k][i] for k in cams], axis=1) for i in range(len(imgs[cams[0]]))]
        imageio.mimwrite(out / "mosaic.mp4", mosaic, fps=fps, macro_block_size=1)
        imageio.imwrite(out / "first.png", mosaic[0]); imageio.imwrite(out / "last.png", mosaic[-1])
    np.savez_compressed(out / "frames.npz", states=S, finger_body_poses=np.stack(body_log), **{k: np.stack(v) for k, v in imgs.items()})
    print(f"[render] wrote {len(imgs[cams[0]]) if cams else 0} frames per camera to {out}")
    env.close()


main()
app.close()
