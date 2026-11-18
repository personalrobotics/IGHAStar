"""Newton-ready copy of the OmniReset UR5e + Robotiq 2F-85 robot USD. UNREVIEWED.

The stock asset is authored for PhysX. Newton / MuJoCo-Warp needs these changes (all done on a COPY; the original is
untouched):

1. The loop-closing joints ``{left,right}_inner_finger_knuckle_joint`` have body0 = child (a "reversed" joint, which
   Newton refuses to import). Swap body0/body1 and their local frames; the left joint also turns both frames 180 deg
   about x so the joint angle keeps the PhysX sign (empirically both sides then reproduce the PhysX start states: the
   four-bar pivots close to 0.1 mm).
2. PhysX mimic joints on the four-bar followers are dropped: the loop (an excluded revolute that Newton turns into
   equality constraints) already fixes them. One mimic stays, right_outer_knuckle = +finger_joint, the stock gearing -1
   (PhysX: q + gearing * q_ref = offset; Newton's coef is -gearing). PhysX's grasped reset states agree (finger 0.78 ->
   right_outer +0.78); the inner knuckles are the ones with right = -left. (An earlier version set gearing +1, i.e.
   right_outer = -finger: that fights right_outer's [0, 75 deg] limit, the finger stalls at 0.66 of 0.78 and the right
   pad swings across the gripper instead of mirroring the left one.)
3. The outer knuckles carry no collision shapes, so Newton computes zero mass for them (NaN): give them 10 g and a tiny
   inertia.
4. Newton ignores PhysX's per-body ``disable_gravity``; the robot is weightless in the PhysX task, so set MuJoCo's
   ``mjc:gravcomp`` = 1 on every link.
5. base_link is authored rotated 180 deg about z; PhysX overrides that with the scene's root pose (identity), Newton keeps
   it. Set it to identity, and place the (fixed) base at the start state's root translation, because Newton cannot move
   a fixed base at run time (the dataset's robot base position is randomised by a few cm per start state).

Also needed at run time (done by the caller, not the USD): armature on the arm joints (explicit OSC torque is unstable in
MuJoCo with zero armature) and on the driven finger joint (explicit position drive on tiny link inertias), Newton's
own collision pipeline (MuJoCo's convex hulls cannot represent the concave hole).
"""
from __future__ import annotations

import shutil
from pathlib import Path

SRC = ("/tmp/https/huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/83860532010b2737aa80d6e8621235ee554186f0/Robots/"
       "UniversalRobots/Ur5e2f85RobotiqGripperCalibrated")
USD_NAME = "ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd"


def build(out_dir: str | Path, base_xyz=(0.0, 0.0, 0.0), src_dir: str | Path = SRC) -> str:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(src_dir) / "metadata.yaml", out_dir / "metadata.yaml")
    st = Usd.Stage.Open(str(Path(src_dir) / USD_NAME))
    rx = Gf.Quatf(0, 1, 0, 0)  # 180 deg about x
    for p in st.Traverse():
        n = p.GetName()
        if n in ("left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint"):
            j = UsdPhysics.Joint(p)
            b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
            p0, r0 = j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()
            p1, r1 = j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get()
            j.GetBody0Rel().SetTargets(b1)
            j.GetBody1Rel().SetTargets(b0)
            f = rx if n.startswith("left") else Gf.Quatf(1, 0, 0, 0)
            j.GetLocalPos0Attr().Set(p1)
            j.GetLocalRot0Attr().Set(r1 * f)
            j.GetLocalPos1Attr().Set(p0)
            j.GetLocalRot1Attr().Set(r0 * f)
        if n in ("left_inner_knuckle_joint", "left_inner_finger_knuckle_joint", "right_inner_knuckle_joint",
                 "right_inner_finger_knuckle_joint"):
            p.SetMetadata("apiSchemas", Sdf.TokenListOp())
        if n in ("left_outer_knuckle", "right_outer_knuckle"):
            m = UsdPhysics.MassAPI.Apply(p)
            m.CreateMassAttr(0.01)
            m.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-6, 1e-6, 1e-6))
            m.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            p.CreateAttribute("mjc:gravcomp", Sdf.ValueTypeNames.Float).Set(1.0)
        if n == "base_link" and p.GetParent().GetName() == "ur5e_robotiq_gripper_d415_mount":
            for op in UsdGeom.Xformable(p).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    op.Set(Gf.Quatd(1, 0, 0, 0))
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(*[float(v) for v in base_xyz]))
    out = out_dir / USD_NAME
    st.Export(str(out))
    return str(out)


# Run-time settings validated against PhysX in newton_probe.py / compare_backends.py (start 3261 and a free-space start):
ARM_ARMATURE = 0.02     # kg m^2; 0.005 is unstable in the wrist, 0.02-0.1 give the same edge outcomes within ~1 mm
FINGER_ARMATURE = 0.02  # kg m^2 on the driven finger joint


PEG_BOX_SIZE = (0.03, 0.03, 0.06)  # m; the stock peg collider is an 8-vertex box mesh of exactly this size
STATIC_SHAPE_KEYS = ("/Table/", "/UR5MetalSupport/", "/ReceptiveObject/", "/GroundPlane/")  # kinematic / static in the task


def build_peg_box(src_usd: str, out_dir: str | Path) -> str:
    """Copy of the peg USD whose collider is a primitive box instead of a 12-triangle mesh with SDF collision.

    Newton collides two meshes with signed-distance fields (the single most expensive kernel, ~60% of a physics step
    here); a box against the hole's mesh is a BVH triangle query + convex-vs-triangle tests instead. The geometry is
    the same box (8-vertex mesh, 30 x 30 x 60 mm). Kept in a folder named ``Peg`` with the asset's metadata.yaml next
    to it, because the task finds its datasets and keypoint offsets from the folder name and that file."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils as ou

    out = Path(out_dir) / "Peg"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "peg.usd"
    shutil.copy(ou.safe_retrieve_file_path(src_usd), dst)
    shutil.copy(ou.safe_retrieve_file_path(src_usd.rsplit("/", 1)[0] + "/metadata.yaml"), out / "metadata.yaml")
    st = Usd.Stage.Open(str(dst))
    mesh = st.GetPrimAtPath("/peg/collisions/peg")
    pts = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), float)
    xf = UsdGeom.Xformable(mesh).GetLocalTransformation()
    lo, hi = pts.min(0), pts.max(0)
    if len(pts) != 8:
        raise RuntimeError(f"peg collider is not an 8-vertex box ({len(pts)} vertices); refusing to replace it by a box")
    ctr = xf.Transform(Gf.Vec3d(*((lo + hi) / 2)))
    size = [float(np.linalg.norm(np.asarray(xf.TransformDir(Gf.Vec3d(*v))))) for v in np.diag(hi - lo)]
    if np.abs(np.asarray(size) - np.asarray(PEG_BOX_SIZE)).max() > 1e-6:
        raise RuntimeError(f"peg collider size {size} != expected {PEG_BOX_SIZE}")
    cube = UsdGeom.Cube.Define(st, "/peg/collisions/peg_box")
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(ctr)
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    for rel in mesh.GetRelationships():  # material bindings (visual and physics)
        if rel.GetName().startswith("material:binding"):
            cube.GetPrim().CreateRelationship(rel.GetName()).SetTargets(rel.GetTargets())
    mesh.SetActive(False)
    st.GetRootLayer().Save()
    return str(dst)


# Peg-hole collider as convex pieces (mesh units mm; measured from the stock 4884-triangle mesh by slicing it):
# plate 69.85 square, z -17.7033..-15.1633 (= cavity floor); walls between the 32 mm square cavity and the 52 mm square
# body up to z 22.9367; 45 deg entry chamfer starting at z 21.540 (half-width 16 + (z - 21.540), 17.397 at the top).
HOLE_MM = dict(plate_half=34.925, z_bot=-17.7033, z_floor=-15.1633, cav_half=16.0, body_half=26.0, z_top=22.9367,
               z_chamfer=21.540)


def hole_pieces_mm():
    """[(name, vertices [k,3] mm)] convex pieces whose union is the hole: one plate box and four wall prisms (each a
    rectangle with the chamfer cut off its top inner corner, extruded over the full body width; they overlap at the
    corners, which reproduces the square opening and the chamfer exactly). The outer rounds/fillets (never touched by
    the peg) become sharp edges: at most ~2.4 mm off at the body's top outer edge, ~7 mm fillet into the plate."""
    import numpy as np

    h = HOLE_MM
    c = h["z_top"] - h["z_chamfer"]
    pieces = []
    px = h["plate_half"]
    pieces.append(("plate", np.array([[x, y, z] for x in (-px, px) for y in (-px, px) for z in (h["z_bot"], h["z_floor"])])))
    sec = [(h["cav_half"], h["z_floor"]), (h["body_half"], h["z_floor"]), (h["body_half"], h["z_top"]),
           (h["cav_half"] + c, h["z_top"]), (h["cav_half"], h["z_chamfer"])]  # (outward coordinate u, z)
    b = h["body_half"]
    for name, axis, sign in (("wall_px", 0, 1), ("wall_nx", 0, -1), ("wall_py", 1, 1), ("wall_ny", 1, -1)):
        v = []
        for u, z in sec:
            for w in (-b, b):
                p = [0.0, 0.0, z]
                p[axis] = sign * u
                p[1 - axis] = w
                v.append(p)
        pieces.append((name, np.array(v)))
    return pieces


def build_hole_pieces(src_usd: str, out_dir: str | Path) -> str:
    """Copy of the peg-hole USD whose collider is the five convex pieces of hole_pieces_mm() instead of the
    4884-triangle mesh. Checked against the source mesh before writing (see check_hole_pieces). Kept in a folder named
    ``PegHole`` with the asset's metadata.yaml (the task finds datasets / keypoint offsets from these)."""
    import numpy as np
    import trimesh
    from pxr import Usd, UsdGeom, UsdPhysics, Vt
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils as ou

    out = Path(out_dir) / "PegHole"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "peg_hole.usd"
    if (out / "checked.txt").exists() and dst.exists():  # built and checked before (the check takes ~13 s)
        return str(dst)
    shutil.copy(ou.safe_retrieve_file_path(src_usd), dst)
    shutil.copy(ou.safe_retrieve_file_path(src_usd.rsplit("/", 1)[0] + "/metadata.yaml"), out / "metadata.yaml")
    st = Usd.Stage.Open(str(dst))
    mesh = st.GetPrimAtPath("/hole/collisions/hole")
    m = UsdGeom.Mesh(mesh)
    P = np.asarray(m.GetPointsAttr().Get(), float)
    report = check_hole_pieces(P, np.asarray(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3))
    if report["cavity_void_inside_pieces"] > 0 or report["cavity_wall_max_dev_mm"] > 0.02:
        raise RuntimeError(f"convex hole pieces do not match the source mesh: {report}")
    scale = 1e-3  # mesh units -> m (the stock mesh prim carries scale 0.001; the pieces are authored in metres)
    for name, v in hole_pieces_mm():
        hull = trimesh.Trimesh(v * scale, process=True).convex_hull
        g = UsdGeom.Mesh.Define(st, f"/hole/collisions/{name}")
        g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(hull.vertices.astype(np.float32)))
        g.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(hull.faces)))
        g.CreateFaceVertexIndicesAttr(Vt.IntArray(hull.faces.flatten().tolist()))
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(g.GetPrim()).CreateApproximationAttr("convexHull")
        UsdGeom.Imageable(g).CreatePurposeAttr("guide")
        for rel in mesh.GetRelationships():
            if rel.GetName().startswith("material:binding"):
                g.GetPrim().CreateRelationship(rel.GetName()).SetTargets(rel.GetTargets())
    mesh.SetActive(False)
    st.GetRootLayer().Save()
    (out / "checked.txt").write_text(str(report) + "\n")
    print(f"[newton_robot_usd] hole collider -> 5 convex pieces; check vs source mesh: {report}")
    return str(dst)


def check_hole_pieces(P, I, n=20000, seed=0) -> dict:
    """Compare the convex pieces with the source mesh (mm). Contact-relevant region = the cavity incl. chamfer.
    - cavity_void_inside_pieces: points sampled in the source's empty cavity that fall inside a piece (must be 0)
    - cavity_wall_max_dev_mm: for points on the source's cavity surfaces (walls, floor, chamfer), distance to the
      pieces' union surface
    - material_missing_frac: points in the source's solid that no piece covers (outer rounds/fillets show up here)
    - extra_material_frac: points inside some piece but outside the source solid (sharp outer edges)"""
    import numpy as np
    import trimesh

    src = trimesh.Trimesh(P, I, process=True)
    hulls = [trimesh.Trimesh(v, process=True).convex_hull for _, v in hole_pieces_mm()]
    rng = np.random.default_rng(seed)
    h = HOLE_MM
    inside_any = lambda q: np.any([hh.contains(q) for hh in hulls], axis=0)  # noqa: E731
    # empty cavity samples (strictly inside the opening, between floor and top)
    q = np.column_stack([rng.uniform(-h["cav_half"] - 1.4, h["cav_half"] + 1.4, (n, 2)), rng.uniform(h["z_floor"], h["z_top"], n)])
    void = ~src.contains(q)
    void_bad = int(inside_any(q[void]).sum())
    # source cavity surface samples
    pts, fi = trimesh.sample.sample_surface(src, n * 2, seed=seed)
    cav = (np.abs(pts[:, :2]).max(1) < h["cav_half"] + 1.45) & (pts[:, 2] > h["z_floor"] - 0.01)
    d = np.min([np.abs(trimesh.proximity.signed_distance(hh, pts[cav])) for hh in hulls], axis=0)
    # whole-body volume comparison
    lo, hi = src.bounds
    qv = rng.uniform(lo, hi, (n, 3))
    in_src, in_p = src.contains(qv), inside_any(qv)
    return {"cavity_void_samples": int(void.sum()), "cavity_void_inside_pieces": void_bad,
            "cavity_surface_samples": int(cav.sum()), "cavity_wall_max_dev_mm": round(float(d.max()), 4),
            "cavity_wall_p99_dev_mm": round(float(np.percentile(d, 99)), 4),
            "material_missing_frac": round(float((in_src & ~in_p).sum() / max(in_src.sum(), 1)), 4),
            "extra_material_frac": round(float((in_p & ~in_src).sum() / max(in_src.sum(), 1)), 4)}


def install_static_pair_filter(keys=STATIC_SHAPE_KEYS) -> None:
    """Drop collision pairs whose two shapes are both static/kinematic (table, robot plate, hole, ground plane): they
    can never produce a force, but e.g. hole-vs-table was ~3000 triangle tests per env per physics step. Patches the
    Newton manager so the explicit broad-phase pair list is filtered before the collision pipeline is built."""
    import numpy as np
    import warp as wp
    from isaaclab_newton.physics import mjwarp_manager, newton_manager

    for C in (newton_manager.NewtonManager, mjwarp_manager.NewtonMJWarpManager):
        f = C.__dict__.get("_initialize_contacts")
        if f is None or getattr(f.__func__, "_static_filter", False):
            continue
        orig = f.__func__

        def init(cls, _orig=orig):
            m = cls._model
            if m is not None and getattr(m, "shape_contact_pairs", None) is not None and not getattr(m, "_static_filtered", False):
                pairs = m.shape_contact_pairs.numpy()
                static = np.array([any(k in str(lb) for k in keys) for lb in m.shape_label])
                keep = ~(static[pairs[:, 0]] & static[pairs[:, 1]])
                m.shape_contact_pairs = wp.array(pairs[keep], dtype=wp.vec2i, device=m.device)
                m.shape_contact_pair_count = int(keep.sum())
                m._static_filtered = True
                print(f"[newton_robot_usd] static pair filter: {len(pairs)} -> {int(keep.sum())} collision pairs")
            _orig(cls)

        init._static_filter = True
        setattr(C, "_initialize_contacts", classmethod(init))


def configure_newton(env_cfg, reset_type: str, start_idx: int, work_dir: str | Path, peg_box: bool = True,
                     filter_static: bool = True, hole_pieces: bool = True, step_graph: bool = True) -> str:
    """Switch an OmniReset env cfg from PhysX to Newton (MuJoCo-Warp) for ONE start state; returns the robot USD path.
    peg_box / filter_static / hole_pieces: collision speed-ups (see build_peg_box / install_static_pair_filter /
    build_hole_pieces). step_graph: turn Newton's per-physics-step CUDA graph off so PegProblem replays one graph per
    policy step instead (fast_sim.NewtonGraphedStep)."""
    import torch
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
    from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils as ou

    pair = ou.compute_pair_dir(env_cfg.scene.insertive_object.spawn.usd_path, env_cfg.scene.receptive_object.spawn.usd_path)
    if peg_box:
        env_cfg.scene.insertive_object.spawn.usd_path = build_peg_box(env_cfg.scene.insertive_object.spawn.usd_path,
                                                                      Path(work_dir) / "peg_box")
    if hole_pieces:
        env_cfg.scene.receptive_object.spawn.usd_path = build_hole_pieces(env_cfg.scene.receptive_object.spawn.usd_path,
                                                                          Path(work_dir) / "hole_pieces")
    if filter_static:
        install_static_pair_filter()
    data = torch.load(ou.safe_retrieve_file_path(f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset_isaaclab3/Resets/{pair}/resets_{reset_type}.pt"))
    roots = data["initial_state"]["articulation"]["robot"]["root_pose"]
    base = roots[start_idx % len(roots)][:3].tolist()
    usd = build(Path(work_dir) / f"newton_robot_{reset_type}_{start_idx % len(roots)}", base)
    env_cfg.scene.robot.spawn.usd_path = usd
    env_cfg.scene.robot.actuators["arm"].armature = ARM_ARMATURE
    env_cfg.scene.robot.actuators["gripper"].armature = FINGER_ARMATURE
    env_cfg.sim.physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(integrator="implicitfast", njmax=300, nconmax=100, impratio=10.0, cone="elliptic",
                                   use_mujoco_contacts=False),
        num_substeps=1, collision_cfg=NewtonCollisionPipelineCfg(), use_cuda_graph=not step_graph)
    return usd


def refresh_velocity_kinematics() -> None:
    """Recompute MuJoCo-Warp's body velocities (cvel, cdof_dot) from the CURRENT Newton joint state. Call after writing
    a state into the sim, before stepping.

    MuJoCo-Warp builds the constraints (fwd_position -> make_constraint) before it recomputes cvel / cdof_dot from qvel
    (fwd_velocity -> com_vel), so each step's constraints use the previous step's values. Within one edge that is a
    fixed one-step lag; after a state write it drags in the velocities of whatever ran before, and the same node +
    control gives a different child (measured: 1-step joint velocity differs by ~1e-4, the edge end by mm to cm).
    Restoring exactly these two arrays made the step bit-identical.
    """
    import warp as wp
    from isaaclab_newton.physics import NewtonManager as NM
    from mujoco_warp._src import smooth

    sv = NM._solver
    m, d = sv.mjw_model, sv.mjw_data
    with wp.ScopedDevice(NM._model.device), sv._scoped_mujoco_warp_execution():
        sv._update_mjc_data(d, NM._model, NM._state_0)
        smooth.kinematics(m, d)
        smooth.com_pos(m, d)
        smooth.com_vel(m, d)
