"""Antipodal grasp candidates for one object, from its collision geometry.

Used by the TCP-delta sampler to "beamform" part of the action set toward sensible
reaches: each candidate is a grasp-site pose in the *object* frame. At search time
the node's own object pose turns it into a world target, and IK decides whether the
arm can get there.

Sampling (parallel-jaw, YAM gripper):
  1. sample points on the surface (area weighted) with outward normals;
  2. shoot a ray inward along -normal, take the far hit; keep the pair if the opening
     is within the gripper's usable width and the far normal opposes the near one
     within a friction cone (antipodal);
  3. the closing axis is the line through the pair; for each roll about it, the
     approach direction is perpendicular to it, giving a full grasp-site rotation;
  4. drop grasps whose approach points up (hand cannot come from below the object),
     or whose pad centre is at the table.

The geometry of the gripper in the grasp-site frame was measured from abc's MJCF
(tools/grasp_check.py prints it): the site's +z is the approach direction, the
fingers close along the site's y axis, fully open gap 0.0946 m, and the pad contact
patch is centred at ``PAD_CENTER_IN_SITE`` (the site itself sits at the fingertips).
"""

from __future__ import annotations

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

MAX_OPENING = 0.0946  # m, finger gap with the gripper fully open
USABLE_WIDTH = (0.010, 0.085)  # object thickness range we try to grasp
PAD_CENTER_IN_SITE = np.array([0.018, 0.0, -0.030])  # where the object's grasp centre goes
FRICTION_ANGLE = np.deg2rad(30.0)
# Site frame: approach = +z, closing axis = +y, third axis x = y cross z.
_E_Z, _E_Y = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
_M_SRC = np.stack([_E_Z, _E_Y, np.cross(_E_Z, _E_Y)], axis=1)


def geom_trimesh(m: mujoco.MjModel, g: int) -> trimesh.Trimesh:
    """One MuJoCo geom as a trimesh, in its body's frame."""
    t, s = int(m.geom_type[g]), m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        tm = trimesh.creation.box(extents=2 * s)
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        tm = trimesh.creation.icosphere(radius=float(s[0]))
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        tm = trimesh.creation.capsule(radius=float(s[0]), height=float(2 * s[1]))
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        tm = trimesh.creation.cylinder(radius=float(s[0]), height=float(2 * s[1]))
    elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        tm = trimesh.creation.icosphere(radius=1.0)
        tm.apply_scale(s)
    elif t == mujoco.mjtGeom.mjGEOM_MESH:
        d = int(m.geom_dataid[g])
        v = m.mesh_vert[m.mesh_vertadr[d]:m.mesh_vertadr[d] + m.mesh_vertnum[d]]
        f = m.mesh_face[m.mesh_faceadr[d]:m.mesh_faceadr[d] + m.mesh_facenum[d]]
        tm = trimesh.Trimesh(vertices=v.copy(), faces=f.copy(), process=False)
    else:
        raise ValueError(f"unsupported geom type {t}")
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(m.geom_quat[g][[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = m.geom_pos[g]
    tm.apply_transform(T)
    return tm


def body_collision_mesh(m: mujoco.MjModel, body: int) -> trimesh.Trimesh:
    """Union of the body's colliding geoms (what the physics sees), in the body frame.
    Falls back to every geom of the body (e.g. the visual mesh) if none collide."""
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body]
    coll = [g for g in geoms if m.geom_contype[g] > 0 or m.geom_conaffinity[g] > 0]
    parts = [geom_trimesh(m, g) for g in (coll or geoms)]
    return trimesh.util.concatenate(parts)


def first_hit(tris: np.ndarray, origins: np.ndarray, dirs: np.ndarray, chunk: int = 20_000_000):
    """Moller-Trumbore, brute force: nearest hit with t > 0 per ray.

    tris [F,3,3], origins/dirs [R,3] -> (t [R] (inf = miss), tri index [R]).
    Chunked over rays so F x R stays under ``chunk`` elements (no rtree needed).
    """
    v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    R, F = len(origins), len(tris)
    t_out, i_out = np.full(R, np.inf), np.full(R, -1)
    step = max(1, chunk // max(F, 1))
    for s0 in range(0, R, step):
        o, d = origins[s0:s0 + step, None], dirs[s0:s0 + step, None]  # [r,1,3]
        pvec = np.cross(d, e2[None])
        det = np.einsum("rfk,fk->rf", pvec, e1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tvec = o - v0[None]
        u = np.einsum("rfk,rfk->rf", tvec, pvec) * inv
        qvec = np.cross(tvec, e1[None])
        v = np.einsum("rk,rfk->rf", dirs[s0:s0 + step], qvec) * inv
        t = np.einsum("fk,rfk->rf", e2, qvec) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
        t = np.where(hit, t, np.inf)
        i = t.argmin(1)
        t_out[s0:s0 + step] = t[np.arange(len(i)), i]
        i_out[s0:s0 + step] = i
    return t_out, i_out


def sample_grasps(mesh: trimesh.Trimesh, R_world_obj: np.ndarray, table_clearance: float,
                  n_surface: int = 4000, n_rolls: int = 8, n_pool: int = 256,
                  seed: int = 0) -> np.ndarray:
    """Antipodal grasp-site poses T_obj_site [N,4,4] (float64), at most ``n_pool``.

    ``R_world_obj`` is the object's rotation at spawn: it only decides which
    approach directions are allowed (not from below) and which grasps would put the
    pad centre under ``table_clearance`` [m above the object's own lowest point].
    """
    rng = np.random.default_rng(seed)
    pts, fi = trimesh.sample.sample_surface(mesh, n_surface, seed=seed)
    nrm = mesh.face_normals[fi]
    d_in = -nrm
    t, it = first_hit(mesh.triangles, pts + 1e-5 * d_in, d_in)
    ir = np.flatnonzero(np.isfinite(t))
    if len(ir) == 0:
        return np.zeros((0, 4, 4))
    loc, it = pts[ir] + (1e-5 + t[ir, None]) * d_in[ir], it[ir]
    p, q = pts[ir], loc
    width = np.linalg.norm(q - p, axis=1)
    d = d_in[ir]
    ok = (width >= USABLE_WIDTH[0]) & (width <= USABLE_WIDTH[1])
    ok &= np.einsum("ij,ij->i", mesh.face_normals[it], d) >= np.cos(FRICTION_ANGLE)
    p, q, d = p[ok], q[ok], d[ok]
    if len(p) == 0:
        return np.zeros((0, 4, 4))
    center = 0.5 * (p + q)
    lowest = (R_world_obj @ mesh.vertices.T).T[:, 2].min()
    out = []
    for c, dd in zip(center, d):
        e1 = np.cross(dd, [1.0, 0.0, 0.0] if abs(dd[0]) < 0.9 else [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(dd, e1)
        for sign in (1.0, -1.0):  # which jaw is which
            for k in range(n_rolls):
                phi = 2 * np.pi * (k + rng.uniform()) / n_rolls
                a = np.cos(phi) * e1 + np.sin(phi) * e2  # approach, perpendicular to the closing line
                if (R_world_obj @ a)[2] > -0.2:  # the hand needs a downward component
                    continue
                if (R_world_obj @ c)[2] - lowest < table_clearance:
                    continue
                y = sign * dd
                R = np.stack([a, y, np.cross(a, y)], axis=1) @ _M_SRC.T  # site frame in object frame
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = c - R @ PAD_CENTER_IN_SITE
                out.append(T)
    if not out:
        return np.zeros((0, 4, 4))
    out = np.stack(out)
    if len(out) > n_pool:
        out = out[rng.choice(len(out), n_pool, replace=False)]
    return out
