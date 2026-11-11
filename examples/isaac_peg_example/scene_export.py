"""Export the visual meshes of the peg scene (robot links, peg, hole) for viser_peg_tree.py. UNREVIEWED.

Run inside the Isaac process after the env exists. Every mesh under a ``visuals`` prim is stored in the frame of the
body that owns it (robot link / InsertiveObject / ReceptiveObject), so the viewer only needs body poses.
"""
from __future__ import annotations

import numpy as np


def export_scene_meshes(robot_body_names: list[str], path: str, env_prim: str = "/World/envs/env_0") -> dict:
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(env_prim)
    cache = UsdGeom.XformCache()
    owners = set(robot_body_names) | {"InsertiveObject", "ReceptiveObject"}
    out: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or "/visuals" not in str(prim.GetPath()):
            continue
        owner = prim
        while owner and owner.GetName() not in owners:
            owner = owner.GetParent()
        if not owner or not owner.IsValid():
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts_attr = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if len(pts) == 0 or len(counts_attr) == 0:
            continue
        # fan triangulation
        tris, k = [], 0
        for c in counts_attr:
            for j in range(1, c - 1):
                tris.append((idx[k], idx[k + j], idx[k + j + 1]))
            k += c
        faces = np.asarray(tris, np.int32)
        # mesh -> owner transform (USD row-vector convention)
        m = np.asarray(cache.GetLocalToWorldTransform(prim)) @ np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(owner)))
        verts = (np.c_[pts, np.ones(len(pts))] @ m)[:, :3].astype(np.float32)
        rgb = np.array([180, 180, 190], np.uint8)
        try:
            dc = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
            if dc is not None and len(dc):
                rgb = (np.clip(np.asarray(dc[0], float), 0, 1) * 255).astype(np.uint8)
        except Exception:
            pass
        name = owner.GetName()
        i = counts.get(name, 0)
        counts[name] = i + 1
        out[f"{name}|{i}|v"], out[f"{name}|{i}|f"], out[f"{name}|{i}|rgb"] = verts, faces, rgb
    np.savez_compressed(path, **out)
    return {k: v for k, v in counts.items()}
