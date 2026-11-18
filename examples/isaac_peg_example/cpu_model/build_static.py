"""Newton's exported MJCF -> CPU model with the hole, table and robot support as MOCAP bodies (no dofs), every
physical parameter copied by name from cpu_scene.mjb (Newton/MuJoCo-Warp world-0 values), Newton's contact-pair list
re-applied as per-geom contype bits, geom_gap zeroed. Writes static_scene.mjb + static_map.npz. UNREVIEWED.
"""
import argparse

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True, help="directory with cpu_scene.mjb + newton_scene.xml; outputs are written there")
ap.add_argument("--box-gripper", action="store_true", help="replace the gripper's mesh colliders by fitted oriented boxes "
                "(measured: changes peg outcomes significantly, only ~8%% faster)")
ap.add_argument("--pairs", action="store_true", help="Newton/MJWarp per-pair contact params as explicit <pair>s "
                "(measured: no better GPU agreement, ~30%% slower)")
args = ap.parse_args()
BOX, PAIRS = args.box_gripper, args.pairs
OUT = ("static_box" if BOX else "static_scene") + ("_pairs" if PAIRS else "")

STATIC = ("ReceptiveObject", "Table", "UR5MetalSupport")
PFX = "_World_envs_env_0_"
SRC = DST = args.dir

old = mujoco.MjModel.from_binary_path(f"{SRC}/cpu_scene.mjb")
spec = mujoco.MjSpec.from_file(f"{SRC}/newton_scene.xml")
for b in STATIC:
    body = spec.body(PFX + b)
    for j in list(body.joints):
        spec.delete(j)
    body.mocap = True
# inertial properties go into the spec BEFORE compiling: MuJoCo builds each body's midphase BVH in the body's inertial
# frame, so copying body_ipos/iquat into a compiled model leaves a stale BVH that culls real contacts
for i in range(1, old.nbody):
    bn = mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_BODY, i)
    sb = spec.body(bn)
    if sb is None:
        continue
    sb.explicitinertial = True
    sb.mass = float(old.body_mass[i]); sb.ipos = old.body_ipos[i].copy(); sb.iquat = old.body_iquat[i].copy()
    sb.inertia = np.maximum(old.body_inertia[i], 1e-12).copy()
for ex in list(spec.excludes):  # Newton writes its collision filter as <exclude>s; MuJoCo-C would apply them on top
    spec.delete(ex)               # of the contype bits below (which already encode Newton's exact pair list)
new = spec.compile()
replaced = set()
if BOX:
    def obb(V):
        """Smallest of: geom-frame AABB, PCA-axis box. -> center, half extents, rotation (columns = box axes)."""
        best = None
        for R in (np.eye(3), np.linalg.svd(V - V.mean(0), full_matrices=False)[2].T):
            if np.linalg.det(R) < 0:
                R = R.copy(); R[:, 2] *= -1
            P = V @ R
            lo, hi = P.min(0), P.max(0)
            vol = np.prod(hi - lo)
            if best is None or vol < best[0]:
                best = (vol, R @ ((lo + hi) / 2), (hi - lo) / 2, R)
        return best[1:]
    for b in range(1, new.nbody):
        bn = mujoco.mj_id2name(new, mujoco.mjtObj.mjOBJ_BODY, b)
        if "Robot" not in bn:
            continue
        gs = [g for g in range(new.ngeom) if new.geom_bodyid[g] == b]
        sgs = list(spec.body(bn).geoms)
        sgs = [x for x in sgs if x.contype or x.conaffinity or True]
        assert len(sgs) == len(gs), (bn, len(sgs), len(gs))
        for g, sg in zip(gs, sgs):
            if new.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            a, n = new.mesh_vertadr[new.geom_dataid[g]], new.mesh_vertnum[new.geom_dataid[g]]
            V = new.mesh_vert[a:a + n].astype(float)
            c, h, R = obb(V)
            Rg = np.zeros(9); mujoco.mju_quat2Mat(Rg, new.geom_quat[g]); Rg = Rg.reshape(3, 3)
            qb = np.zeros(4); mujoco.mju_mat2Quat(qb, R.flatten()); q = np.zeros(4); mujoco.mju_mulQuat(q, new.geom_quat[g], qb)
            hull = ConvexHull(V).volume
            print(f"  box for {bn[-22:]:22s} geom {g}: half {np.round(h * 1e3, 1).tolist()} mm, hull/box volume {hull / np.prod(2 * h):.2f}")
            sg.type = mujoco.mjtGeom.mjGEOM_BOX
            sg.meshname = ""
            sg.size = h
            sg.pos = new.geom_pos[g] + Rg @ c
            sg.quat = q
            replaced.add(g)
    new = spec.compile()
print(f"old nq {old.nq} nv {old.nv} nbody {old.nbody} ngeom {old.ngeom} | new nq {new.nq} nv {new.nv} nbody {new.nbody} "
      f"ngeom {new.ngeom} nmocap {new.nmocap} neq {new.neq} nu {new.nu}")

name = lambda m, t, i: mujoco.mj_id2name(m, t, i)  # noqa: E731
OB, OJ, OG = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_GEOM
bmap = {}
for i in range(new.nbody):
    j = mujoco.mj_name2id(old, OB, name(new, OB, i))
    assert j >= 0 or i == 0, name(new, OB, i)
    bmap[i] = max(j, 0)
jmap = {}
for i in range(new.njnt):
    j = mujoco.mj_name2id(old, OJ, name(new, OJ, i))
    assert j >= 0, name(new, OJ, i)
    jmap[i] = j
# geoms: same body, same order within the body
gmap = {}
for b in range(new.nbody):
    gn = [g for g in range(new.ngeom) if new.geom_bodyid[g] == b]
    go = [g for g in range(old.ngeom) if old.geom_bodyid[g] == bmap[b]]
    assert len(gn) == len(go), (name(new, OB, b), len(gn), len(go))
    for a, c in zip(gn, go):
        assert new.geom_type[a] == old.geom_type[c] or a in replaced
        gmap[a] = c
# --- copy parameters
for f in ("body_gravcomp",):
    for i, j in bmap.items():
        getattr(new, f)[i] = getattr(old, f)[j]
for i, j in jmap.items():
    na = {0: 7, 1: 4, 2: 1, 3: 1}[new.jnt_type[i]]
    nd = {0: 6, 1: 3, 2: 1, 3: 1}[new.jnt_type[i]]
    for f in ("jnt_range", "jnt_limited", "jnt_solref", "jnt_solimp", "jnt_margin", "jnt_stiffness", "jnt_actfrcrange",
              "jnt_actfrclimited", "jnt_actgravcomp"):
        if hasattr(new, f):
            getattr(new, f)[i] = getattr(old, f)[j]
    a, c = new.jnt_dofadr[i], old.jnt_dofadr[j]
    for f in ("dof_armature", "dof_frictionloss", "dof_damping", "dof_solref", "dof_solimp"):
        getattr(new, f)[a:a + nd] = getattr(old, f)[c:c + nd]
    qa, qc = new.jnt_qposadr[i], old.jnt_qposadr[j]
    new.qpos0[qa:qa + na] = old.qpos0[qc:qc + na]
for i, j in gmap.items():
    for f in (("geom_size", "geom_pos", "geom_quat") if i not in replaced else ()) + ("geom_friction", "geom_solref", "geom_solimp", "geom_solmix",
              "geom_margin", "geom_condim", "geom_priority"):
        getattr(new, f)[i] = getattr(old, f)[j]
new.geom_gap[:] = 0.0
for f in ("actuator_gainprm", "actuator_biasprm", "actuator_dynprm", "actuator_gear", "actuator_ctrlrange",
          "actuator_ctrllimited", "actuator_forcerange", "actuator_forcelimited", "actuator_trntype", "actuator_gaintype",
          "actuator_biastype"):
    getattr(new, f)[:] = getattr(old, f)
for f in ("eq_data", "eq_solref", "eq_solimp", "eq_active0"):
    getattr(new, f)[:] = getattr(old, f)
for f in ("timestep", "iterations", "ls_iterations", "tolerance", "ls_tolerance", "cone", "impratio", "integrator", "solver",
          "noslip_iterations", "ccd_iterations", "disableflags", "enableflags"):
    setattr(new.opt, f, getattr(old.opt, f))
new.opt.gravity[:] = old.opt.gravity
new.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)  # BVH is valid now (inertials set pre-compile)
# body_pos/quat of the robot chain from the old model (calibrated), by name
for i, j in bmap.items():
    if i and new.body_mocapid[i] < 0:
        new.body_pos[i], new.body_quat[i] = old.body_pos[j], old.body_quat[j]
# --- contact pairs: old contype bits are one bit per OLD geom index; re-encode for the new indices
inv = {c: a for a, c in gmap.items()}
allowed = set()
for c in range(old.ngeom):
    for c2 in range(old.ngeom):
        if c < c2 and ((old.geom_conaffinity[c] >> c2) & 1):
            allowed.add((inv[c], inv[c2]))
new.geom_contype[:] = [1 << g for g in range(new.ngeom)]
ca = np.zeros(new.ngeom, np.int64)
for a, b in allowed:
    ca[a] |= 1 << b; ca[b] |= 1 << a
new.geom_conaffinity[:] = ca
print(f"contact pairs re-encoded: {len(allowed)}")
for i, j in bmap.items():
    if i:
        assert np.allclose(new.body_mass[i], old.body_mass[j]) and np.allclose(new.body_inertia[i], np.maximum(old.body_inertia[j], 1e-12), rtol=1e-5), name(new, OB, i)
if PAIRS:
    # MJWarp with Newton contacts (nominal physics): friction = max of the two geoms and solref/solimp per geom, exactly
    # as MuJoCo-C combines them; but the friction rows get solreffriction = (2 / (kf invw ...), 1), which refsafe clamps
    # to 2*timestep for every pair here (MuJoCo-C dynamic contacts would use solref 0.02 instead). Only a <pair> can
    # carry solreffriction in MuJoCo-C, so every allowed pair becomes an explicit pair and dynamic collisions go off.
    gname = {}
    sgeoms = [None] * new.ngeom
    for b in range(new.nbody):
        bn = mujoco.mj_id2name(new, OB, b)
        sb = spec.worldbody if b == 0 else spec.body(bn)
        gs = [g for g in range(new.ngeom) if new.geom_bodyid[g] == b]
        for g, sg in zip(gs, list(sb.geoms)):
            if not sg.name:
                sg.name = f"g{g}"
            gname[g] = sg.name
    for a, b in sorted(allowed):
        fr = max(new.geom_friction[a, 0], new.geom_friction[b, 0])
        spec.add_pair(geomname1=gname[a], geomname2=gname[b], condim=int(max(new.geom_condim[a], new.geom_condim[b])),
                      friction=[fr, fr, 0.005, 0.0001, 0.0001], solref=new.geom_solref[a].tolist(),
                      solreffriction=[2.0 * new.opt.timestep, 1.0], solimp=new.geom_solimp[a].tolist(),
                      margin=0.0, gap=0.0)
    np2 = spec.compile()
    skip = {"geom_contype", "geom_conaffinity"}
    for f in dir(new):
        if f.startswith("_") or f.startswith("pair_") or f in skip:
            continue
        a = getattr(new, f)
        if isinstance(a, np.ndarray) and a.size and hasattr(np2, f) and getattr(np2, f).shape == a.shape:
            try:
                getattr(np2, f)[...] = a
            except (ValueError, TypeError):
                pass
    for f in ("timestep", "iterations", "ls_iterations", "tolerance", "ls_tolerance", "cone", "impratio", "integrator",
              "solver", "noslip_iterations", "ccd_iterations", "disableflags", "enableflags"):
        setattr(np2.opt, f, getattr(new.opt, f))
    np2.opt.gravity[:] = new.opt.gravity
    np2.geom_contype[:] = 0
    np2.geom_conaffinity[:] = 0
    print(f"explicit pairs: {np2.npair}; solreffriction {np2.pair_solreffriction[0].tolist()}; "
          f"friction range {np2.pair_friction[:, 0].min():.3f}-{np2.pair_friction[:, 0].max():.3f}")
    new = np2
mujoco.mj_saveModel(new, f"{DST}/{OUT}.mjb")
# --- index map old qpos/qvel -> new; mocap ids of the static bodies (pose comes from the old free-joint qpos)
qidx, vidx = [], []
for i, j in jmap.items():
    na = {0: 7, 1: 4, 2: 1, 3: 1}[new.jnt_type[i]]; nd = {0: 6, 1: 3, 2: 1, 3: 1}[new.jnt_type[i]]
    assert new.jnt_qposadr[i] == len(qidx)
    qidx += list(range(old.jnt_qposadr[j], old.jnt_qposadr[j] + na))
    vidx += list(range(old.jnt_dofadr[j], old.jnt_dofadr[j] + nd))
moc = []
for b in STATIC:
    bi = mujoco.mj_name2id(new, OB, PFX + b)
    jo = [j for j in range(old.njnt) if old.jnt_bodyid[j] == mujoco.mj_name2id(old, OB, PFX + b)][0]
    moc.append((new.body_mocapid[bi], old.jnt_qposadr[jo]))
np.savez(f"{DST}/static_map.npz", qidx=np.array(qidx), vidx=np.array(vidx), mocap=np.array(moc))
print("qidx", qidx, "\nmocap (id, old qadr)", moc)
