#!/usr/bin/env python3
"""Scrub an IGHA* search tree of the UWLab/OmniReset UR5e peg task in Viser. UNREVIEWED.

Reads ``peg_seed<k>_tree.npz`` and ``scene_meshes.npz`` that ``plan_peg.py --save-tree`` writes (no Isaac needed to
view them) and shows:

* the robot, peg and hole as the simulator's own visual meshes, posed at any expanded node,
* every simulated edge as the hand point's path (one segment per policy step), coloured by expansion order
  (dark = early, yellow = late); edges rejected as invalid (blow-ups, over-acceleration) are red,
* peg paths (orange) for edges that moved the peg,
* the goal: sphere of the success position threshold around the hole's alignment point (green),
* a branch (root -> node) in black that can be played back: the solution, lowest-h node, closest hand to peg,
  most peg motion, or deepest. Arm links are interpolated between edge ends (poses are logged once per edge);
  the peg follows its exact per-step poses.

    python viser_peg_tree.py --tree .../peg_seed0_tree.npz --port 8090
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import viser


def viridis(x: np.ndarray) -> np.ndarray:
    stops = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], float)
    x = np.clip(x, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[:, None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    """q0, q1 [..., 4] (xyzw), a in [0,1]."""
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    d = np.abs(d).clip(0, 1)
    th = np.arccos(d)
    s = np.sin(th)
    w0 = np.where(s > 1e-6, np.sin((1 - a) * th) / np.maximum(s, 1e-9), 1 - a)
    w1 = np.where(s > 1e-6, np.sin(a * th) / np.maximum(s, 1e-9), a)
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--meshes", default=None, help="scene_meshes.npz (default: next to the tree file)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-segments", type=int, default=400_000)
    args = ap.parse_args()

    z = np.load(args.tree, allow_pickle=False)
    meshes = np.load(args.meshes or str(Path(args.tree).with_name("scene_meshes.npz")), allow_pickle=False)
    T = int(z["steps_per_edge"])
    step_dt = float(z["step_dt"])
    names = [str(n) for n in z["body_names"]]
    pf, pcall, pcid = z["parent_feat"], z["parent_call"], z["parent_child_id"]
    cpar, cf, cpeg, cbp = z["child_parent"], z["child_feat"], z["child_peg"], z["child_bp"].astype(np.float32)
    cvalid, ch, ccost = z["child_valid"].astype(bool), z["child_h"], z["child_cost"]
    sol = z["solution_child_ids"]
    start_bp, start_peg = z["start_bp"], z["start_peg"]
    E, C = len(pcall), len(cpar)
    ccall = pcall[cpar]
    print(f"peg seed {int(z['seed'])} (start state {int(z['start_idx'])}, {z['reset_type']}): {E} expansions, "
          f"{C} edges ({int(cvalid.sum())} valid)")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")

    # ---- scene meshes: one frame per body, meshes in the body frame
    frames = {}
    for key in sorted({k.rsplit("|", 1)[0] for k in meshes.files}):
        owner = key.split("|")[0]
        if owner not in frames:
            if owner == "ReceptiveObject":
                hp = z["hole_pose"]
                frames[owner] = server.scene.add_frame(f"/{owner}", show_axes=False, position=tuple(hp[:3]),
                                                       wxyz=tuple(xyzw_to_wxyz(hp[3:7])))
            else:
                frames[owner] = server.scene.add_frame(f"/{owner}", show_axes=False)
        col = tuple(int(c) for c in meshes[f"{key}|rgb"])
        if owner == "InsertiveObject":
            col = (230, 90, 60)
        elif owner == "ReceptiveObject":
            col = (70, 110, 200)
        server.scene.add_mesh_simple(f"/{owner}/{key.split('|')[1]}", meshes[f"{key}|v"], meshes[f"{key}|f"],
                                     color=col, flat_shading=False)
    hole_z = float(z["hole_pose"][2])
    server.scene.add_grid("/floor", width=1.6, height=1.2, position=(0.45, 0.0, hole_z), plane="xy",
                          cell_size=0.1, cell_color=(120, 120, 120), section_color=(90, 90, 90))
    gp = z["goal_pos"]
    server.scene.add_icosphere("/goal", radius=float(z["pos_thr"]), color=(40, 200, 60), position=tuple(gp), opacity=0.5)

    def set_pose(bp: np.ndarray, peg: np.ndarray) -> None:
        for i, n in enumerate(names):
            if n in frames:
                frames[n].position = tuple(bp[i, :3])
                frames[n].wxyz = tuple(xyzw_to_wxyz(bp[i, 3:7]))
        frames["InsertiveObject"].position = tuple(peg[:3])
        frames["InsertiveObject"].wxyz = tuple(xyzw_to_wxyz(peg[3:7]))

    def node_pose(cid: int):
        return (start_bp, start_peg) if cid < 0 else (cbp[cid], cpeg[cid, -1])

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.position = (gp[0] - 0.55, gp[1] - 0.55, gp[2] + 0.45)
        client.camera.look_at = tuple(gp)

    # ---- edges: parent point + T per-step points
    ok_edge = np.isfinite(cf).all(axis=(1, 2)) & (np.abs(cf).max(axis=(1, 2)) < 5.0)

    def edge_points(sl):
        p0 = pf[cpar][:, sl][:, None]
        pts = np.concatenate([p0, cf[:, :, sl]], 1)  # [C,T+1,3]
        return np.stack([pts[:, :-1], pts[:, 1:]], 2)  # [C,T,2,3]

    hand_segs = edge_points(slice(6, 9))
    peg_segs = edge_points(slice(0, 3))
    peg_moved = ok_edge & (np.linalg.norm(cf[:, -1, 0:3] - pf[cpar][:, 0:3], axis=1) > 0.003)
    order_rgb = viridis(ccall / max(E, 1))
    child_up = pcid[cpar]

    def branch(cid: int) -> list[int]:
        out = []
        while cid >= 0 and len(out) < 10_000:
            out.append(int(cid))
            cid = int(child_up[cid])
        return out[::-1]

    with server.gui.add_folder("Tree"):
        info = server.gui.add_markdown("")
        s_exp = server.gui.add_slider("expansions shown", 1, max(E, 1), 1, max(E, 1))
        cb_invalid = server.gui.add_checkbox("show invalid edges", False)
        cb_peg = server.gui.add_checkbox("show peg paths", True)
        s_thick = server.gui.add_slider("line width (mm)", 0.5, 6.0, 0.5, 1.5)
    with server.gui.add_folder("Robot"):
        s_node = server.gui.add_slider("pose at expansion #", 0, max(E - 1, 0), 1, 0)
        cb_play = server.gui.add_checkbox("play expansions", False)
        s_fps = server.gui.add_slider("play rate (fps)", 1, 60, 1, 10)
        node_md = server.gui.add_markdown("")
    with server.gui.add_folder("Branch"):
        choices = ["lowest h", "closest hand to peg", "most peg motion", "deepest"]
        if len(sol):
            choices.insert(0, "solution")
        dd = server.gui.add_dropdown("branch to", choices, initial_value=choices[0])
        b_play = server.gui.add_button("play branch on robot")
        branch_md = server.gui.add_markdown("")

    def redraw(_=None):
        k = int(s_exp.value)
        sel = (ccall <= k) & ok_edge
        if not cb_invalid.value:
            sel &= cvalid
        idx = np.nonzero(sel)[0]
        stride = max(1, int(np.ceil(len(idx) * T / args.max_segments)))
        idx = idx[::stride]
        w = float(s_thick.value) / 1000.0
        segs = hand_segs[idx].reshape(-1, 2, 3)
        col = np.where(cvalid[idx, None], order_rgb[idx], np.array([[220, 30, 30]], np.uint8))
        col = np.repeat(col[:, None, None, :], T, 1).repeat(2, 2).reshape(-1, 2, 3)
        server.scene.add_line_segments("/tree/hand", segs, col, thickness=w)
        mi = idx[peg_moved[idx]]
        if cb_peg.value and len(mi):
            server.scene.add_line_segments("/tree/peg", peg_segs[mi].reshape(-1, 2, 3), (255, 140, 0), thickness=1.5 * w)
        else:
            server.scene.remove_by_name("/tree/peg")
        info.content = (f"{E} expansions, {C} edges ({int(cvalid.sum())} valid, {int((~cvalid).sum())} invalid)  \n"
                        f"showing {len(idx)} edges up to expansion {k}" + (f" (every {stride}th)" if stride > 1 else ""))

    def show_node(_=None):
        k = int(s_node.value)
        set_pose(*node_pose(int(pcid[k])))
        f = pf[k]
        d = np.linalg.norm(f[6:9] - f[0:3])
        node_md.content = (f"expansion {k}: peg ({f[0]:.3f}, {f[1]:.3f}, {f[2]:.3f}), hand ({f[6]:.3f}, {f[7]:.3f}, {f[8]:.3f})  \n"
                           f"|hand - peg| = {d * 100:.1f} cm, |peg - goal| = {np.linalg.norm(f[0:3] - gp) * 100:.1f} cm")

    def pick_branch() -> int:
        v = np.nonzero(cvalid & ok_edge)[0]
        name = dd.value
        if name == "solution":
            return int(sol[-1])
        if name == "lowest h":
            return int(v[np.argmin(ch[v])])
        if name == "most peg motion":
            mv = np.linalg.norm(cf[v, -1, 0:3] - start_peg[:3], axis=1)
            return int(v[np.argmax(mv)])
        if name == "deepest":
            ex = v[np.isin(v, pcid)] if np.isin(v, pcid).any() else v
            return int(max(ex, key=lambda c: len(branch(int(c)))))
        dist = np.linalg.norm(cf[v, -1, 6:9] - cf[v, -1, 0:3], axis=1)
        return int(v[np.argmin(dist)])

    def draw_branch(ids):
        pts = []
        for cid in ids:
            seq = np.concatenate([pf[cpar[cid]][6:9][None], cf[cid, :, 6:9]], 0)
            pts.append(np.stack([seq[:-1], seq[1:]], 1))
        if pts:
            server.scene.add_line_segments("/branch", np.concatenate(pts), (0, 0, 0), thickness=0.006)

    def on_branch(_=None):
        ids = branch(pick_branch())
        draw_branch(ids)
        last = ids[-1]
        branch_md.content = (f"{len(ids)} edges = {len(ids) * float(z['edge_dt']):.2f} s, cost {float(np.sum(ccost[ids])):.2f}, "
                             f"h at end {ch[last]:.3f}")
        return ids

    for h in (s_exp, cb_invalid, cb_peg, s_thick):
        h.on_update(redraw)
    s_node.on_update(show_node)
    dd.on_update(on_branch)

    @b_play.on_click
    def _(_):
        ids = on_branch()

        def run():
            for cid in ids:
                bp0, _ = node_pose(int(child_up[cid]))
                bp1, _ = node_pose(cid)
                for s in range(1, T + 1):
                    a = s / T
                    bp = np.concatenate([bp0[:, :3] * (1 - a) + bp1[:, :3] * a, slerp(bp0[:, 3:7], bp1[:, 3:7], a)], axis=1)
                    set_pose(bp, cpeg[cid, s - 1])
                    time.sleep(step_dt)
        threading.Thread(target=run, daemon=True).start()

    redraw()
    show_node()
    on_branch()
    print(f"open http://localhost:{args.port}")
    while True:
        if cb_play.value and E > 1:
            s_node.value = (int(s_node.value) + 1) % E
            time.sleep(1.0 / float(s_fps.value))
        else:
            time.sleep(0.1)


if __name__ == "__main__":
    main()
