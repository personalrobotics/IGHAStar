#!/usr/bin/env python3
"""Drag a T around a 1024 street map.

Click sets the center of mass. The drag points the stem. The bar-length slider
is the T's narrowest width, in pixels. Maps are the street PNGs; black is wall.

    python3 examples/standalone/viser_street_tee.py
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import viser
import viser.infra
from PIL import Image, ImageDraw

MAP_DIR = Path(__file__).resolve().parent / "Maps" / "street-png"
MAP_PX = 1024
# The map is 1024 px. Draw it 10 units across so the default camera can see it.
SCENE = 10.0
# Native Push-T, bar length 120. CoG is the average of the two rectangle centroids.
COG_LOCAL = np.array([0.0, 45.0])
BAR = np.array([[-60, 0], [60, 0], [60, 30], [-60, 30]], float)
STEM = np.array([[-15, 30], [-15, 120], [15, 120], [15, 30]], float)


def maps_1024() -> list[str]:
    return sorted(p.name for p in MAP_DIR.glob("*_1024.png"))


def load_map(name: str) -> tuple[np.ndarray, np.ndarray]:
    im = Image.open(MAP_DIR / name).convert("RGB").resize((MAP_PX, MAP_PX), Image.NEAREST)
    # Viser puts the top of the texture at -Y. Flip so the file's top row is +Y,
    # which is up on the screen.
    raw = np.asarray(im)
    # Collision uses the file's top-down pixels. Only the texture is flipped,
    # because Viser puts the top of an image at -Y.
    wall = raw[:, :, 0] < 10
    return np.flipud(raw), wall


def patched_client_root() -> Path:
    """Serve a viser client that reports the drag in order, not as a box.

    This viser build only has click and rect-select. Rect-select keeps the
    drag, then sends the box corners, which drops the direction. The patched
    client sends the press point and the release point, and draws that segment.
    """
    src = Path(viser.__file__).resolve().parent / "client" / "build"
    dst = Path("/tmp/viser_street_tee_client")
    marker = dst / ".drag_patch"
    if marker.exists():
        return dst
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    assets = list((dst / "assets").glob("index-*.js"))
    if len(assets) != 1:
        raise SystemExit(f"expected one viser bundle, found {assets}")
    js = assets[0]
    text = js.read_text()
    old_box = (
        "xt.fillRect(ft.dragStart[0],ft.dragStart[1],"
        "ft.dragEnd[0]-ft.dragStart[0],ft.dragEnd[1]-ft.dragStart[1])"
    )
    new_line = (
        "xt.moveTo(ft.dragStart[0],ft.dragStart[1]),"
        "xt.lineTo(ft.dragEnd[0],ft.dragEnd[1]),xt.lineWidth=2"
    )
    old_msg = "screen_pos:[[xt,vt],[yt,St]]"
    new_msg = "screen_pos:[[pt.x,pt.y],[mt.x,mt.y]]"
    if old_box not in text or old_msg not in text:
        raise SystemExit("viser client bundle does not match the drag patch")
    js.write_text(text.replace(old_box, new_line, 1).replace(old_msg, new_msg, 1))
    marker.write_text("ok\n")
    return dst


def use_patched_client() -> None:
    root = patched_client_root()
    original = viser.infra.WebsockServer.__init__

    def init(self, *args, **kwargs):
        kwargs["http_server_root"] = root
        original(self, *args, **kwargs)

    viser.infra.WebsockServer.__init__ = init


def quat_to_R(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def hit_map(camera, sx: float, sy: float):
    """Ray through an OpenCV screen point, hit on the map plane z=0."""
    half = np.tan(float(camera.fov) * 0.5)
    d_cam = np.array(
        [(sx - 0.5) * 2 * half * float(camera.aspect), (sy - 0.5) * 2 * half, 1.0]
    )
    d_cam /= np.linalg.norm(d_cam)
    direction = quat_to_R(camera.wxyz) @ d_cam
    origin = np.asarray(camera.position, float)
    if abs(direction[2]) < 1e-8:
        return None
    t = -origin[2] / direction[2]
    if t <= 0:
        return None
    return origin + t * direction


def px_to_world(px: np.ndarray, z: float = 0.0) -> np.ndarray:
    s = SCENE / MAP_PX
    return np.array([(px[0] - MAP_PX / 2) * s, (MAP_PX / 2 - px[1]) * s, z])


def world_to_px(hit: np.ndarray) -> np.ndarray:
    s = MAP_PX / SCENE
    return np.array([hit[0] * s + MAP_PX / 2, MAP_PX / 2 - hit[1] * s])


def tee_polygons(bar: float, yaw: float, com: np.ndarray) -> list[np.ndarray]:
    k = bar / 120.0
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    out = []
    for poly in (BAR, STEM):
        local = poly * k - COG_LOCAL * k
        out.append((local @ rot.T) + com)
    return out


def mesh_from_polys(polys: list[np.ndarray]):
    verts = []
    faces = []
    for poly in polys:
        base = len(verts)
        for x, y in poly:
            verts.append(px_to_world(np.array([x, y]), 0.02))
        faces.append([base, base + 1, base + 2])
        faces.append([base, base + 2, base + 3])
    return np.asarray(verts, np.float32), np.asarray(faces, np.uint32)


def overlap_px(wall: np.ndarray, polys: list[np.ndarray]) -> int:
    mask = Image.new("L", (MAP_PX, MAP_PX), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polys:
        draw.polygon([tuple(p) for p in poly], fill=255)
    return int(((np.asarray(mask) > 0) & wall).sum())


def main() -> None:
    use_patched_client()
    names = maps_1024()
    if "Berlin_0_1024.png" not in names:
        raise SystemExit(f"no 1024 maps in {MAP_DIR}")

    server = viser.ViserServer(port=8086, label="Street T")
    server.scene.set_up_direction("+z")
    server.gui.add_markdown(
        "Drag on the map. The press sets the center of mass. "
        "The drag points the stem. Red means the T is on a wall."
    )
    map_dd = server.gui.add_dropdown("Map", names, initial_value="Berlin_0_1024.png")
    size = server.gui.add_slider("Bar (px)", min=10, max=140, step=1, initial_value=60)
    pose_txt = server.gui.add_text("Pose", initial_value="", disabled=True)

    state = {"com": np.array([512.0, 512.0]), "yaw": 0.0, "wall": None, "rgb": None}
    handles = {"map": None, "tee": None, "stem": None}

    def show_map(name: str) -> None:
        rgb, wall = load_map(name)
        state["rgb"] = rgb
        state["wall"] = wall
        if handles["map"] is not None:
            handles["map"].remove()
        handles["map"] = server.scene.add_image(
            "/map",
            rgb,
            render_width=SCENE,
            render_height=SCENE,
            format="jpeg",
            jpeg_quality=90,
        )

    def show_tee() -> None:
        bar = float(size.value)
        com = state["com"]
        yaw = state["yaw"]
        polys = tee_polygons(bar, yaw, com)
        hit = overlap_px(state["wall"], polys)
        verts, faces = mesh_from_polys(polys)
        color = (200, 50, 40) if hit else (210, 120, 40)
        if handles["tee"] is None:
            handles["tee"] = server.scene.add_mesh_simple(
                "/tee", verts, faces, color=color, side="double"
            )
        else:
            handles["tee"].vertices = verts
            handles["tee"].color = color
        c, s = np.cos(yaw), np.sin(yaw)
        tip_px = com + rot_stem(c, s, bar * 0.45)
        a = px_to_world(com, 0.03)
        b = px_to_world(tip_px, 0.03)
        seg = np.stack([a, b])[None].astype(np.float32)
        if handles["stem"] is None:
            handles["stem"] = server.scene.add_line_segments(
                "/stem", seg, colors=(20, 20, 20), line_width=3
            )
        else:
            handles["stem"].points = seg
        pose_txt.value = (
            f"x {com[0]:.0f}  y {com[1]:.0f}  yaw {yaw:.2f}  wall px {hit}"
        )

    show_map(map_dd.value)
    show_tee()

    @map_dd.on_update
    def _(_event) -> None:
        show_map(map_dd.value)
        show_tee()

    @size.on_update
    def _(_event) -> None:
        show_tee()

    @server.on_client_connect
    def _(client) -> None:
        # The first camera message has to arrive before position can be set.
        # The default camera sits a few units from the origin, inside a 1024px map.
        placed = {"ok": False}

        @client.camera.on_update
        def _(_camera) -> None:
            if placed["ok"]:
                return
            placed["ok"] = True
            try:
                client.camera.far = 100.0
                client.camera.position = (0.0, 0.0, 7.2)
                client.camera.look_at = (0.0, 0.0, 0.0)
                client.camera.up_direction = (0.0, 1.0, 0.0)
            except Exception:
                import traceback

                traceback.print_exc()

    @server.scene.on_pointer_event("rect-select")
    def _(event: viser.ScenePointerEvent) -> None:
        hits = []
        for sx, sy in event.screen_pos:
            hit = hit_map(event.client.camera, sx, sy)
            if hit is None:
                return
            hits.append(hit)
        state["com"] = np.clip(world_to_px(hits[0]), 0, MAP_PX - 1)
        delta = hits[1] - hits[0]
        dpx = np.array([delta[0], -delta[1]]) * (MAP_PX / SCENE)
        if np.hypot(dpx[0], dpx[1]) > 4:
            state["yaw"] = float(-np.arctan2(dpx[0], dpx[1]))
        show_tee()

    print("viser http://localhost:8086")
    try:
        while True:
            import time

            time.sleep(1.0)
    except KeyboardInterrupt:
        return


def rot_stem(c: float, s: float, length: float) -> np.ndarray:
    """Pixel offset of the stem tip. yaw is carried by c=cos, s=sin."""
    return np.array([-s, c]) * length


if __name__ == "__main__":
    main()
