#!/usr/bin/env python3
"""Scrub an IGHA* search tree over an abc_sim scene in Viser.

Reads the ``<task>_seed<k>_tree.npz`` that ``plan_abc.py --save-tree`` writes,
rebuilds the same abc scene (task + seed), and shows:

* the MuJoCo scene (mjviser), posed at any expanded node,
* every simulated edge as the TCP's 30 Hz path (one segment per control period),
  coloured by expansion order (dark = early, yellow = late); invalid edges red,
* mover-object paths (orange) for edges that moved it,
* abc's goal region for the mover (green box),
* a branch (root -> node) in black that can be played back on the robot:
  the solution, the node closest to the mover, the lowest-h node, or the deepest.

    python viser_abc_tree.py --tree /path/put_relative_seed13_tree.npz --port 8090
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import viser

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import abc_sim  # noqa: E402
from mjviser import ViserMujocoScene  # noqa: E402

TCP_SLICES = {"left": slice(3, 6), "right": slice(6, 9)}


def viridis(x: np.ndarray) -> np.ndarray:
    """Small viridis approximation, x in [0,1] -> uint8 RGB."""
    stops = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], float)
    x = np.clip(x, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[:, None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-segments", type=int, default=400_000)
    args = ap.parse_args()

    z = np.load(args.tree, allow_pickle=False)
    task, seed = str(z["task"]), int(z["seed"])
    arms = [str(a) for a in z["active_arms"]]
    T = int(z["ctrl_steps_per_edge"])
    pf, pq = z["parent_feat"], z["parent_qpos"]
    pcall, pcid = z["parent_call"], z["parent_child_id"]
    cpar, cf, cq = z["child_parent"], z["child_feat"], z["child_qpos"]
    cvalid, ch = z["child_valid"].astype(bool), z["child_h"]
    ccost = z["child_cost"]
    sol = z["solution_child_ids"]
    E, C = len(pcall), len(cpar)
    ccall = pcall[cpar]  # expansion index of each child's edge
    print(f"{task} seed {seed}: '{z['prompt']}'  {E} expansions, {C} edges, arms={arms}")

    # Scene: same randomized world as the search.
    env = abc_sim.make_env(task=task, render_cameras=False)
    env.reset(seed=seed, randomize=True)
    model = env.model
    data = mujoco.MjData(model)

    def pose(qpos):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        scene.update_from_mjdata(data)

    server = viser.ViserServer(port=args.port)
    scene = ViserMujocoScene(server, model, num_envs=1)
    scene.camera_tracking_enabled = False
    server.scene.remove_by_name("/fixed_bodies/world/table_plane")
    pose(z["start_qpos"])

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # Behind and above the robot bases, looking at the table.
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.position = (-0.45, 0.0, 1.65)
        client.camera.look_at = (0.55, 0.0, 0.85)

    # Goal region (put_relative evaluator cone around the reference).
    ev = env._task_evaluator
    goal = getattr(ev, "_goal", None)
    if goal is not None:
        ref = z["start_qpos"][int(z["ref_adr"]):int(z["ref_adr"]) + 3]
        ax = np.asarray(goal["axis"], float)
        c = 0.5 * (ev.along_min_m + ev.along_max_m)
        yaw = np.arctan2(ax[1], ax[0])
        server.scene.add_box(
            "/goal_region", color=(40, 200, 60),
            dimensions=(ev.along_max_m - ev.along_min_m, 2 * ev.perp_max_m, 0.004),
            position=(ref[0] + ax[0] * c, ref[1] + ax[1] * c, 0.752),
            wxyz=(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)), opacity=0.4,
        )

    # Edge polylines: parent point followed by the T recorded control periods.
    def edge_points(sl):
        p0 = pf[cpar][:, sl][:, None]  # [C,1,3]
        pts = np.concatenate([p0, cf[:, :, sl]], 1)  # [C,T+1,3]
        return np.stack([pts[:, :-1], pts[:, 1:]], 2)  # [C,T,2,3]

    tcp_segs = {a: edge_points(TCP_SLICES[a]) for a in arms}
    mover_segs = edge_points(slice(0, 3))
    mover_moved = np.linalg.norm(cf[:, -1, 0:3] - pf[cpar][:, 0:3], axis=1) > 0.003
    order_rgb = viridis(ccall / max(E, 1))

    # Tree structure: child -> parent child id (or -1 for the root).
    child_up = pcid[cpar]

    def branch(cid: int) -> list[int]:
        out = []
        while cid >= 0 and len(out) < 10_000:
            out.append(int(cid))
            cid = int(child_up[cid])
        return out[::-1]

    with server.gui.add_folder("Tree"):
        info = server.gui.add_markdown(
            f"**{z['prompt']}**  \n{E} expansions, {C} edges ({int(cvalid.sum())} valid), "
            f"edge = {T} x {float(z['edge_dt']) / T * 1000:.0f} ms")
        s_exp = server.gui.add_slider("expansions shown", 1, max(E, 1), 1, max(E, 1))
        cb_invalid = server.gui.add_checkbox("show invalid edges", False)
        cb_mover = server.gui.add_checkbox("show mover paths", True)
        s_thick = server.gui.add_slider("line width (mm)", 0.5, 6.0, 0.5, 1.5)
    with server.gui.add_folder("Robot"):
        s_node = server.gui.add_slider("pose at expansion #", 0, max(E - 1, 0), 1, 0)
        cb_play = server.gui.add_checkbox("play expansions", False)
        s_fps = server.gui.add_slider("play rate (fps)", 1, 60, 1, 10)
        node_md = server.gui.add_markdown("")
    with server.gui.add_folder("Branch"):
        choices = ["closest TCP to mover", "lowest h", "deepest", "highest mover motion"]
        if len(sol):
            choices.insert(0, "solution")
        dd = server.gui.add_dropdown("branch to", choices, initial_value=choices[0])
        b_play = server.gui.add_button("play branch on robot")
        branch_md = server.gui.add_markdown("")

    def redraw(_=None):
        k = int(s_exp.value)
        sel = ccall <= k
        if not cb_invalid.value:
            sel &= cvalid
        idx = np.nonzero(sel)[0]
        stride = max(1, int(np.ceil(len(idx) * T / args.max_segments)))
        idx = idx[::stride]
        w = float(s_thick.value) / 1000.0
        for a in arms:
            segs = tcp_segs[a][idx].reshape(-1, 2, 3)
            col = np.where(cvalid[idx, None], order_rgb[idx], np.array([[220, 30, 30]], np.uint8))
            col = np.repeat(col[:, None, None, :], T, 1).repeat(2, 2).reshape(-1, 2, 3)
            server.scene.add_line_segments(f"/tree/tcp_{a}", segs, col, thickness=w)
        mi = idx[mover_moved[idx]]
        if cb_mover.value and len(mi):
            server.scene.add_line_segments(
                "/tree/mover", mover_segs[mi].reshape(-1, 2, 3), (255, 140, 0), thickness=1.5 * w)
        else:
            server.scene.remove_by_name("/tree/mover")
        info.content = (f"**{z['prompt']}**  \nshowing {len(idx)} edges up to expansion {k}"
                        + (f" (every {stride}th)" if stride > 1 else ""))

    def show_node(_=None):
        k = int(s_node.value)
        pose(pq[k])
        f = pf[k]
        tcp = " ".join(f"{a} tcp ({f[TCP_SLICES[a]][0]:.2f}, {f[TCP_SLICES[a]][1]:.2f}, {f[TCP_SLICES[a]][2]:.2f})" for a in arms)
        d = min(np.linalg.norm(f[TCP_SLICES[a]] - f[0:3]) for a in arms)
        node_md.content = f"expansion {k}: {tcp}  \n|tcp - mover| = {d * 100:.1f} cm"

    def pick_branch() -> int:
        v = np.nonzero(cvalid)[0]
        name = dd.value
        if name == "solution":
            return int(sol[-1])
        if name == "lowest h":
            return int(v[np.argmin(ch[v])])
        if name == "highest mover motion":
            mv = np.linalg.norm(cf[v, -1, 0:3] - z["start_qpos"][int(z["mover_adr"]):int(z["mover_adr"]) + 3], axis=1)
            return int(v[np.argmax(mv)])
        if name == "deepest":
            ex = v[np.isin(v, pcid)] if np.isin(v, pcid).any() else v
            return int(max(ex, key=lambda c: len(branch(int(c)))))
        dist = np.min(np.stack([np.linalg.norm(cf[v, -1, TCP_SLICES[a]] - cf[v, -1, 0:3], axis=1) for a in arms]), 0)
        return int(v[np.argmin(dist)])

    def draw_branch(ids):
        pts = []
        for cid in ids:
            p = pf[cpar[cid]]
            for a in arms:
                seq = np.concatenate([p[TCP_SLICES[a]][None], cf[cid, :, TCP_SLICES[a]]], 0)
                pts.append(np.stack([seq[:-1], seq[1:]], 1))
        if pts:
            server.scene.add_line_segments("/branch", np.concatenate(pts), (0, 0, 0), thickness=0.006)

    def on_branch(_=None):
        ids = branch(pick_branch())
        draw_branch(ids)
        g = float(np.sum(ccost[ids]))
        last = ids[-1]
        branch_md.content = (f"{len(ids)} edges = {len(ids) * float(z['edge_dt']):.2f} s, cost {g:.2f}, "
                             f"h at end {ch[last]:.3f}")
        return ids

    for h in (s_exp, cb_invalid, cb_mover, s_thick):
        h.on_update(redraw)
    s_node.on_update(show_node)
    dd.on_update(on_branch)

    @b_play.on_click
    def _(_):
        ids = on_branch()

        def run():
            pose(pq[cpar[ids[0]]])
            for cid in ids:
                time.sleep(float(z["edge_dt"]))
                pose(cq[cid])
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
