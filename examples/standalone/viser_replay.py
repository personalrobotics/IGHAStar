#!/usr/bin/env python3
"""Replay a planned trajectory over the terrain in Viser.

Works for both terrain representations, so the same trajectory can be replayed
over an elevation map and over an ESDF and the resulting roll, pitch and ride
height compared directly.  The terrain heights and the vehicle attitude are
computed here with a NumPy mirror of get_footprint_z() in kinodynamic.cu, which
doubles as a check that the GPU query is behaving.

Expects a trajectory as returned by ``get_best_path()``, i.e. goal-first.

    python3 viser_replay.py -c Configs/kinodynamic_example.yml \
        --path ../../Content/standalone/race-2_kinodynamic_esdf_IGHAStar_path.npy
"""

import argparse
import math
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import get_map, get_map_type, resolve_map_dir, terrain_layers

STANDALONE_DIR = os.path.dirname(os.path.abspath(__file__))
OBSTACLE_COLOR = np.array([255, 105, 180], dtype=np.uint8)


def sample_height(elevation: np.ndarray, x: Any, y: Any, map_res: float) -> np.ndarray:
    """Terrain height lookup with the same truncation the CUDA query uses."""
    ny, nx = elevation.shape
    res_inv = np.float32(1.0 / map_res)
    ix = np.clip((np.asarray(x, dtype=np.float32) * res_inv).astype(np.int32), 0, nx - 1)
    iy = np.clip((np.asarray(y, dtype=np.float32) * res_inv).astype(np.int32), 0, ny - 1)
    return elevation[iy, ix]


def footprint_attitude(
    elevation: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    yaw: np.ndarray,
    car_l2: float,
    car_w2: float,
    map_res: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ride height, roll and pitch; mirrors get_footprint_z() in kinodynamic.cu."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    fl_z = sample_height(
        elevation, car_l2 * cy - car_w2 * sy + x, car_l2 * sy + car_w2 * cy + y, map_res
    )
    fr_z = sample_height(
        elevation, car_l2 * cy + car_w2 * sy + x, car_l2 * sy - car_w2 * cy + y, map_res
    )
    bl_z = sample_height(
        elevation, -car_l2 * cy - car_w2 * sy + x, -car_l2 * sy + car_w2 * cy + y, map_res
    )
    br_z = sample_height(
        elevation, -car_l2 * cy + car_w2 * sy + x, -car_l2 * sy - car_w2 * cy + y, map_res
    )
    z = sample_height(elevation, x, y, map_res)
    roll = np.arctan2((fl_z + bl_z) - (fr_z + br_z), 4 * car_w2)
    pitch = np.arctan2((bl_z + br_z) - (fl_z + fr_z), 4 * car_l2)
    return z, roll, pitch


def _quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """wxyz quaternion for the intrinsic Rz(yaw) Ry(pitch) Rx(roll) attitude."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _terrain_colors(costmap: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    """Grayscale elevation shading, with obstacles picked out in pink."""
    span = max(float(elevation.max() - elevation.min()), 1e-6)
    shade = ((elevation - elevation.min()) / span * 255).astype(np.uint8)
    colors = np.repeat(shade[:, :, None], 3, axis=2)
    colors[costmap <= 250.0] = OBSTACLE_COLOR
    return colors


def add_terrain(
    server: Any,
    costmap: np.ndarray,
    elevation: np.ndarray,
    map_res: float,
    stride: int = 2,
) -> None:
    """Terrain surface as a coloured mesh, falling back to a point cloud."""
    heights = elevation[::stride, ::stride]
    colors = _terrain_colors(costmap, elevation)[::stride, ::stride]
    rows, cols = heights.shape
    grid_y, grid_x = np.indices(heights.shape)
    vertices = np.stack(
        [
            grid_x.ravel() * stride * map_res,
            grid_y.ravel() * stride * map_res,
            heights.ravel(),
        ],
        axis=-1,
    ).astype(np.float32)
    vertex_colors = colors.reshape(-1, 3)

    try:
        import trimesh
    except ImportError:
        server.scene.add_point_cloud(
            "/terrain",
            points=vertices,
            colors=vertex_colors,
            point_size=map_res * stride,
        )
        return

    corner = (np.arange(rows - 1)[:, None] * cols + np.arange(cols - 1)[None, :]).ravel()
    faces = np.concatenate(
        [
            np.stack([corner, corner + 1, corner + cols], axis=-1),
            np.stack([corner + 1, corner + cols + 1, corner + cols], axis=-1),
        ]
    )
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False
    )
    server.scene.add_mesh_trimesh("/terrain", mesh)


def replay_trajectory(
    path: np.ndarray,
    bitmap: Any,
    node_info: Dict[str, Any],
    stride: int = 2,
    port: int = 8080,
    fps: float = 20.0,
    block: bool = True,
) -> Any:
    """Drive the vehicle along ``path`` over the terrain in ``bitmap``."""
    import viser

    map_res = float(node_info["map_res"])
    car_l2 = float(node_info["length"]) / 2
    car_w2 = float(node_info["width"]) / 2
    car_h = float(node_info.get("height", 0.8))
    map_type = get_map_type(node_info)

    # get_best_path() returns states goal-first; reverse to start->goal so the
    # replay runs forwards in time.
    path = np.asarray(path)[::-1]

    costmap, elevation = terrain_layers(bitmap, node_info)
    x, y, yaw = path[:, 0], path[:, 1], path[:, 2]
    velocity = path[:, 3]
    z, roll, pitch = footprint_attitude(
        elevation, x, y, yaw, car_l2, car_w2, map_res
    )
    n_states = len(path)

    server = viser.ViserServer(port=port)
    add_terrain(server, costmap, elevation, map_res, stride=stride)
    server.scene.add_spline_catmull_rom(
        "/trajectory",
        positions=np.stack([x, y, z + 0.1], axis=-1).astype(np.float32),
        color=(255, 220, 0),
        line_width=3.0,
    )
    car = server.scene.add_box(
        "/vehicle",
        color=(40, 200, 90),
        dimensions=(car_l2 * 2, car_w2 * 2, car_h),
    )
    axes = server.scene.add_frame("/vehicle_axes", axes_length=2.0, axes_radius=0.06)

    with server.gui.add_folder("Replay"):
        server.gui.add_markdown(
            f"Map type: **{map_type}**  \n"
            f"{n_states} states along the trajectory"
        )
        playing = server.gui.add_checkbox("Play", True)
        step = server.gui.add_slider(
            "State", min=0, max=n_states - 1, step=1, initial_value=0
        )
        speed = server.gui.add_slider(
            "States per second", min=1.0, max=120.0, step=1.0, initial_value=fps
        )
        readout = server.gui.add_markdown("")

    def show(index: int) -> None:
        quaternion = _quaternion(float(roll[index]), float(pitch[index]), float(yaw[index]))
        position = (float(x[index]), float(y[index]), float(z[index]) + car_h / 2)
        car.wxyz = quaternion
        car.position = position
        axes.wxyz = quaternion
        axes.position = position
        readout.content = (
            f"x {x[index]:7.2f} m  y {y[index]:7.2f} m  z {z[index]:7.2f} m  \n"
            f"yaw {math.degrees(yaw[index]):7.2f} deg  \n"
            f"roll {math.degrees(roll[index]):7.2f} deg  \n"
            f"pitch {math.degrees(pitch[index]):7.2f} deg  \n"
            f"velocity {velocity[index]:7.2f} m/s"
        )

    @step.on_update
    def _(_event: Any) -> None:
        show(int(step.value))

    show(0)
    print(f"Viser replay running on http://localhost:{port}")
    if not block:
        return server
    try:
        while True:
            if playing.value:
                step.value = (int(step.value) + 1) % n_states
            time.sleep(1.0 / max(float(speed.value), 1.0))
    except KeyboardInterrupt:
        pass
    return server


def main(
    yaml_path: str,
    path_file: Optional[str] = None,
    map_type: Optional[str] = None,
    stride: int = 2,
    port: int = 8080,
) -> None:
    with open(yaml_path, "r") as handle:
        configs = yaml.safe_load(handle)
    map_info = configs["map"]
    node_info = configs["experiment_info_default"]["node_info"]
    if map_type is not None:
        node_info["map_type"] = map_type

    bitmap = get_map(
        map_info["name"],
        map_dir=resolve_map_dir(map_info["dir"]),
        map_size=map_info["size"],
        node_info=node_info,
    )

    if path_file is None:
        raise SystemExit(
            "No trajectory given. Run example.py first (it saves a *_path.npy) "
            "and pass it with --path."
        )
    path = np.load(path_file)
    print(f"Loaded {len(path)} states from {path_file}")
    replay_trajectory(path, bitmap, node_info, stride=stride, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay an IGHAStar trajectory in Viser")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="Configs/kinodynamic_example.yml",
        help="Path to YAML configuration file",
    )
    parser.add_argument("--path", type=str, default=None, help="Saved trajectory .npy")
    parser.add_argument(
        "--map-type",
        type=str,
        default=None,
        choices=["elevation", "esdf"],
        help="Override the config map type, to replay the same path over both",
    )
    parser.add_argument("--stride", type=int, default=2, help="Terrain mesh subsampling")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(STANDALONE_DIR, config_path)
    main(
        yaml_path=config_path,
        path_file=args.path,
        map_type=args.map_type,
        stride=args.stride,
        port=args.port,
    )
