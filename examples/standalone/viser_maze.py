#!/usr/bin/env python3
"""Hand-draw a maze and push the two tees with two agents.

The box is the closed 512 workspace. The task name is {N}T{M}W{K}A:
N tees, M interior segments, K agents. This viewer is 2T{M}W2A.
Drag in Wall mode to add a segment. Tee mode: the press sets the center of
mass, the drag points the stem. Agent mode places that pusher. Push mode
drives that pusher toward the click. The other pusher holds. Save writes
the segment row the search reads.

    python3 examples/standalone/viser_maze.py
"""
from __future__ import annotations

import os
import sys
import threading

import numpy as np
import torch
import viser
import yaml

from ighastar.scripts.common_utils import create_planner
from pusht_multi_generate_2T1O_clear import CONTENT, pack_walls
from viser_street_tee import hit_map, use_patched_client

BOX = 512.0
SCENE = 10.0
# Same local polygons as pusht_multi_physics.h. The pose is the body origin.
# The center of mass is the unweighted average of the two rectangle centroids.
COG = np.array([0.0, 45.0])
BAR = np.array([[-60, 0], [60, 0], [60, 30], [-60, 30]], float)
STEM = np.array([[-15, 30], [-15, 120], [15, 120], [15, 30]], float)
AGENT_R = 15.0
N_AGENTS = 2
# State: a0 a1 | t0 t1 | velocities. Tee i origin is at 4 + 3*i. Velocities at 10.
TEE0 = 4
VEL0 = 10
TEE_COLOR = ((70, 130, 180), (210, 120, 40))
AGENT_COLOR = ((200, 40, 40), (120, 70, 190))
CONFIG = os.path.join(
    os.path.dirname(__file__), "Configs", "pusht_multi_2T0W2A.yml"
)
SAVE_PATH = os.path.join(CONTENT, "maze_hand.npz")


def to_vis(x: float, y: float, z: float = 0.02) -> np.ndarray:
    s = SCENE / BOX
    return np.array([(x - BOX / 2) * s, (y - BOX / 2) * s, z], np.float32)


def to_chip(hit: np.ndarray) -> np.ndarray:
    s = BOX / SCENE
    return np.array([hit[0] * s + BOX / 2, hit[1] * s + BOX / 2])


def rot(yaw: float, local: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]])


def origin_of(com: np.ndarray, yaw: float) -> np.ndarray:
    return com - rot(yaw, COG)


def com_of(origin: np.ndarray, yaw: float) -> np.ndarray:
    return origin + rot(yaw, COG)


def tee_polys(origin: np.ndarray, yaw: float) -> list[np.ndarray]:
    out = []
    for poly in (BAR, STEM):
        out.append(origin + np.stack([rot(yaw, p) for p in poly]))
    return out


def mesh_of(polys: list[np.ndarray], z: float):
    verts = []
    faces = []
    for poly in polys:
        base = len(verts)
        for x, y in poly:
            verts.append(to_vis(float(x), float(y), z))
        faces.append([base, base + 1, base + 2])
        faces.append([base, base + 2, base + 3])
    return np.asarray(verts, np.float32), np.asarray(faces, np.uint32)


def disk(x: float, y: float, radius: float, z: float, n: int = 20):
    verts = [to_vis(x, y, z)]
    for i in range(n):
        a = 2 * np.pi * i / n
        verts.append(to_vis(x + radius * np.cos(a), y + radius * np.sin(a), z))
    faces = [[0, 1 + i, 1 + (i + 1) % n] for i in range(n)]
    return np.asarray(verts, np.float32), np.asarray(faces, np.uint32)


def home_state() -> np.ndarray:
    # Origins, not centers of mass. Angle 0 leaves the stem along +y.
    # Agents at x=80, separated in y. Tees sit to their right.
    s = np.zeros(4 * N_AGENTS + 6 * 2, np.float32)
    s[0], s[1] = 50.0, 340.0
    s[2], s[3] = 50.0, 170.0
    s[TEE0 : TEE0 + 3] = (150.0, 340.0, 0.0)
    s[TEE0 + 3 : TEE0 + 6] = (360.0, 170.0, 0.0)
    return s


def hold_target(state: np.ndarray, agent: int, xy: np.ndarray) -> np.ndarray:
    """One xy per agent. The chosen agent goes to xy. The other holds."""
    target = np.array(state[: 2 * N_AGENTS], np.float32).copy()
    target[2 * agent : 2 * agent + 2] = xy
    return target


def load_kernel():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    planner = create_planner(cfg, bidirectional=False)
    return sys.modules[type(planner).__module__], planner, cfg


def probe(mod) -> None:
    """Agent 0 moves. Agent 1 holds. A segment stops agent 0."""
    state = home_state()
    empty = pack_walls([])
    free_t = hold_target(state, 0, np.array([300.0, 340.0]))
    free = mod.step(
        torch.from_numpy(state), torch.from_numpy(empty), torch.from_numpy(free_t), 30, True, 500.0
    )
    if float(free[0]) <= float(state[0]) + 10:
        raise SystemExit(f"step did not move agent 0: x {state[0]} -> {float(free[0])}")
    if abs(float(free[2]) - float(state[2])) > 5.0:
        raise SystemExit(f"agent 1 did not hold: x {state[2]} -> {float(free[2])}")
    blocked = pack_walls([((200.0, 5.0), (200.0, 506.0))])
    hit_t = hold_target(state, 0, np.array([400.0, 340.0]))
    hit = mod.step(
        torch.from_numpy(state), torch.from_numpy(blocked), torch.from_numpy(hit_t), 80, True, 500.0
    )
    if float(hit[0]) >= 200.0 - AGENT_R:
        raise SystemExit(f"agent 0 crossed the segment: x={float(hit[0])}")
    print(
        f"step check passed (free x={float(free[0]):.1f}, held x={float(free[2]):.1f}, blocked x={float(hit[0]):.1f})",
        flush=True,
    )


def main() -> None:
    print("loading Chipmunk stepper", flush=True)
    mod, planner, cfg = load_kernel()
    probe(mod)
    vel = float(cfg["experiment_info_default"]["node_info"]["max_agent_vel"])

    use_patched_client()
    server = viser.ViserServer(port=8087, label="Maze")
    server.scene.set_up_direction("+z")
    server.gui.add_markdown(
        "Wall: drag a segment. Tee: press sets the center of mass, drag points the stem. "
        "Agent: click to place that pusher. Push: click where that pusher should go. "
        "The other pusher holds."
    )
    mode = server.gui.add_dropdown(
        "Mode",
        ("Wall", "Tee 0", "Tee 1", "Agent 0", "Agent 1", "Push 0", "Push 1"),
        initial_value="Wall",
    )
    burst = server.gui.add_slider("Push steps", min=1, max=100, step=1, initial_value=25)
    status = server.gui.add_text("Status", initial_value="", disabled=True)
    undo = server.gui.add_button("Undo wall")
    clear = server.gui.add_button("Clear walls")
    reset = server.gui.add_button("Reset push")
    again = server.gui.add_button("Push again")
    save = server.gui.add_button("Save")

    lock = threading.Lock()
    sim = home_state()
    rest = sim.copy()
    walls: list[tuple[tuple[float, float], tuple[float, float]]] = []
    target = {"agent": 0, "xy": None}
    note = {"text": ""}
    handles: dict = {}

    def world_row() -> np.ndarray:
        return pack_walls(walls)

    def redraw() -> None:
        if "floor" not in handles:
            gray = np.full((8, 8, 3), 232, np.uint8)
            handles["floor"] = server.scene.add_image(
                "/floor", gray, render_width=SCENE, render_height=SCENE
            )
            box = [(5, 5), (506, 5), (506, 506), (5, 506), (5, 5)]
            segs = np.stack(
                [
                    [to_vis(*a, 0.03), to_vis(*b, 0.03)]
                    for a, b in zip(box, box[1:])
                ]
            ).astype(np.float32)
            handles["box"] = server.scene.add_line_segments(
                "/box", segs, colors=(30, 30, 30), line_width=4
            )
        if "walls" in handles:
            handles["walls"].remove()
            del handles["walls"]
        if walls:
            segs = np.stack(
                [[to_vis(*a, 0.04), to_vis(*b, 0.04)] for a, b in walls]
            ).astype(np.float32)
            handles["walls"] = server.scene.add_line_segments(
                "/walls", segs, colors=(20, 20, 20), line_width=6
            )
        for i, color in enumerate(TEE_COLOR):
            b = TEE0 + 3 * i
            origin = sim[b : b + 2]
            yaw = float(sim[b + 2])
            verts, faces = mesh_of(tee_polys(origin, yaw), 0.05 + 0.01 * i)
            name = f"/tee{i}"
            if name not in handles:
                handles[name] = server.scene.add_mesh_simple(
                    name, verts, faces, color=color, side="double"
                )
            else:
                handles[name].vertices = verts
        for i, color in enumerate(AGENT_COLOR):
            vx, vy = disk(float(sim[2 * i]), float(sim[2 * i + 1]), AGENT_R, 0.08)
            name = f"/agent{i}"
            if name not in handles:
                handles[name] = server.scene.add_mesh_simple(
                    name, vx, vy, color=color, side="double"
                )
            else:
                handles[name].vertices = vx
        if target["xy"] is None:
            if "/aim" in handles:
                handles["/aim"].remove()
                del handles["/aim"]
        else:
            agent = int(target["agent"])
            seg = np.stack(
                [
                    to_vis(float(sim[2 * agent]), float(sim[2 * agent + 1]), 0.09),
                    to_vis(*target["xy"], 0.09),
                ]
            )[None].astype(np.float32)
            if "/aim" not in handles:
                handles["/aim"] = server.scene.add_line_segments(
                    "/aim", seg, colors=(40, 90, 200), line_width=2
                )
            else:
                handles["/aim"].points = seg
        overlap = ""
        try:
            ok = planner.check_validity_batched(
                torch.from_numpy(sim[None]), torch.from_numpy(world_row())
            )
            if not bool(ok[0]):
                overlap = "  overlap"
        except Exception:
            overlap = ""
        status.value = (
            f"walls {len(walls)}  "
            f"a0 {sim[0]:.0f},{sim[1]:.0f}  a1 {sim[2]:.0f},{sim[3]:.0f}"
            f"{overlap}{note['text']}"
        )

    def push_toward(agent: int, xy: np.ndarray) -> None:
        target["agent"] = agent
        target["xy"] = np.clip(xy, 20.0, 492.0)
        cmd = hold_target(sim, agent, target["xy"])
        out = mod.step(
            torch.from_numpy(sim),
            torch.from_numpy(world_row()),
            torch.from_numpy(cmd),
            int(burst.value),
            True,
            vel,
        )
        n = sim.size
        sim[:] = out[:n].detach().cpu().numpy()
        note["text"] = f"  pen {float(out[n]):.2f}  force {float(out[n + 1]):.0f}"

    redraw()

    @server.on_client_connect
    def _(client) -> None:
        placed = {"ok": False}

        @client.camera.on_update
        def _(_camera) -> None:
            if placed["ok"]:
                return
            placed["ok"] = True
            client.camera.far = 100.0
            client.camera.position = (0.0, 0.0, 7.2)
            client.camera.look_at = (0.0, 0.0, 0.0)
            client.camera.up_direction = (0.0, 1.0, 0.0)

    @server.scene.on_pointer_event("rect-select")
    def _(event: viser.ScenePointerEvent) -> None:
        hits = []
        for sx, sy in event.screen_pos:
            hit = hit_map(event.client.camera, sx, sy)
            if hit is None:
                return
            hits.append(to_chip(hit))
        a, b = hits[0], hits[1]
        long = float(np.hypot(*(b - a))) > 8.0
        with lock:
            choice = mode.value
            if choice == "Wall":
                if long:
                    walls.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
            elif choice in ("Tee 0", "Tee 1"):
                i = 0 if choice == "Tee 0" else 1
                bse = TEE0 + 3 * i
                yaw = float(sim[bse + 2])
                if long:
                    d = b - a
                    yaw = float(np.arctan2(-d[0], d[1]))
                origin = origin_of(np.clip(a, 40.0, 472.0), yaw)
                sim[bse : bse + 3] = (origin[0], origin[1], yaw)
                sim[VEL0:] = 0
                rest[:] = sim
            elif choice in ("Agent 0", "Agent 1"):
                i = 0 if choice == "Agent 0" else 1
                place = np.clip(a, 30.0, 482.0)
                sim[2 * i], sim[2 * i + 1] = float(place[0]), float(place[1])
                sim[VEL0:] = 0
                rest[:] = sim
            elif choice in ("Push 0", "Push 1"):
                push_toward(0 if choice == "Push 0" else 1, a)
            redraw()

    @undo.on_click
    def _(_event) -> None:
        with lock:
            if walls:
                walls.pop()
            redraw()

    @clear.on_click
    def _(_event) -> None:
        with lock:
            walls.clear()
            redraw()

    @reset.on_click
    def _(_event) -> None:
        with lock:
            sim[:] = rest
            target["xy"] = None
            note["text"] = ""
            redraw()

    @again.on_click
    def _(_event) -> None:
        with lock:
            if target["xy"] is not None:
                push_toward(int(target["agent"]), target["xy"])
            redraw()

    @save.on_click
    def _(_event) -> None:
        with lock:
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            posed = sim.copy()
            posed[VEL0:] = 0
            name = f"2T{len(walls)}W2A"
            np.savez(
                SAVE_PATH,
                state=posed.astype(np.float32),
                world=world_row(),
                close_right=np.array(True),
                problem_class=np.array(name),
            )
            print(f"wrote {SAVE_PATH} {name}", flush=True)
            note["text"] = "  saved"
            redraw()

    print("viser http://localhost:8087", flush=True)
    try:
        while True:
            import time

            time.sleep(1.0)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
