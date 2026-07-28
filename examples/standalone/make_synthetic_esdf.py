#!/usr/bin/env python3
"""Convert an elevation map + costmap into a synthetic ESDF and inspect it.

The ESDF is built from exactly the maps the planner consumes (``get_map()`` in
elevation mode), so the two representations can be compared directly.  The
Viser view is a debugging aid: it should make it obvious at a glance whether
the generated ESDF matches the original elevation and cost maps.

    python3 make_synthetic_esdf.py -c Configs/kinodynamic_example.yml
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import esdf_utils as eu
from utils import esdf_cache_path, get_map, resolve_map_dir

STANDALONE_DIR = os.path.dirname(os.path.abspath(__file__))


def _diverging_colors(values: np.ndarray, scale: float) -> np.ndarray:
    """Blue (inside terrain) to white (surface) to red (free space)."""
    t = np.clip(values / max(scale, 1e-6), -1.0, 1.0)
    colors = np.ones(values.shape + (3,), dtype=np.float32)
    negative, positive = t < 0.0, t >= 0.0
    colors[negative, 0] = 1.0 + t[negative]
    colors[negative, 1] = 1.0 + t[negative]
    colors[positive, 1] = 1.0 - t[positive]
    colors[positive, 2] = 1.0 - t[positive]
    return (colors * 255).astype(np.uint8)


def visualize_esdf(
    esdf: Dict[str, Any],
    elevation: np.ndarray,
    costmap: np.ndarray,
    stride: int = 2,
    port: int = 8080,
    block: bool = True,
) -> Any:
    """Show the ESDF, and the maps it came from, in Viser."""
    import viser

    distance = np.asarray(esdf["distance"], dtype=np.float32)
    color = esdf["color"]
    voxel_xy, voxel_z, z_min = esdf["voxel_xy"], esdf["voxel_z"], esdf["z_min"]
    ny, nx, nz = distance.shape
    extent_x, extent_y = nx * voxel_xy, ny * voxel_xy

    server = viser.ViserServer(port=port)
    server.scene.add_grid(
        "/grid",
        width=extent_x,
        height=extent_y,
        width_segments=int(extent_x),
        height_segments=int(extent_y),
        position=(extent_x / 2, extent_y / 2, z_min),
    )

    # Voxels straddling the zero crossing, coloured by the ESDF colour layer.
    sub = distance[::stride, ::stride, :]
    iy, ix, ik = np.nonzero(np.abs(sub) < voxel_z)
    surface_points = np.stack(
        [
            ix * stride * voxel_xy,
            iy * stride * voxel_xy,
            z_min + ik * voxel_z,
        ],
        axis=-1,
    ).astype(np.float32)
    surface_colors = color[::stride, ::stride][iy, ix, ik]
    surface_handle = server.scene.add_point_cloud(
        "/esdf/surface_voxels",
        points=surface_points,
        colors=surface_colors,
        point_size=voxel_xy * stride,
    )
    print(f"Surface band: {len(surface_points)} voxels (stride {stride})")

    # Reference views of the maps the ESDF was built from.  Rows run +y, so the
    # arrays are flipped to keep the image aligned with the world frame.
    elev_norm = (elevation - elevation.min()) / max(
        float(elevation.max() - elevation.min()), 1e-6
    )
    elev_image = np.repeat((elev_norm * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    cost_image = np.repeat(
        np.clip(costmap, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2
    )
    server.scene.add_image(
        "/reference/elevation",
        image=elev_image[::-1],
        render_width=extent_x,
        render_height=extent_y,
        position=(extent_x / 2, -extent_y * 0.6, z_min),
    )
    server.scene.add_image(
        "/reference/costmap",
        image=cost_image[::-1],
        render_width=extent_x,
        render_height=extent_y,
        position=(extent_x * 1.6, extent_y / 2, z_min),
    )

    # Recovered surface, i.e. what map_to_elev() returns in ESDF mode.
    recovered = eu.surface_grid(esdf)[::stride, ::stride]
    grid_y, grid_x = np.indices(recovered.shape)
    recovered_points = np.stack(
        [
            grid_x.ravel() * stride * voxel_xy,
            grid_y.ravel() * stride * voxel_xy,
            recovered.ravel(),
        ],
        axis=-1,
    ).astype(np.float32)
    recovered_handle = server.scene.add_point_cloud(
        "/esdf/recovered_surface",
        points=recovered_points,
        colors=eu.surface_color_grid(esdf)[::stride, ::stride].reshape(-1, 3),
        point_size=voxel_xy * stride,
        visible=False,
    )

    slice_scale = float(np.abs(distance).max())
    slice_handle: Dict[str, Any] = {"handle": None}

    def draw_slice(k: int) -> None:
        if slice_handle["handle"] is not None:
            slice_handle["handle"].remove()
        image = _diverging_colors(distance[:, :, k], slice_scale)
        slice_handle["handle"] = server.scene.add_image(
            "/esdf/slice",
            image=image[::-1],
            render_width=extent_x,
            render_height=extent_y,
            position=(extent_x / 2, extent_y / 2, z_min + k * voxel_z),
        )

    with server.gui.add_folder("ESDF"):
        server.gui.add_markdown(
            f"**{nx} x {ny} x {nz}** voxels  \n"
            f"xy {voxel_xy:.3f} m, z {voxel_z:.3f} m  \n"
            f"z range {z_min:.2f} to {z_min + (nz - 1) * voxel_z:.2f} m  \n"
            f"method `{esdf.get('method', 'plane')}`"
        )
        show_band = server.gui.add_checkbox("Surface band", True)
        show_recovered = server.gui.add_checkbox("Recovered surface", False)
        show_slice = server.gui.add_checkbox("Distance slice", False)
        slice_index = server.gui.add_slider(
            "Slice k", min=0, max=nz - 1, step=1, initial_value=nz // 2
        )

    @show_band.on_update
    def _(_event: Any) -> None:
        surface_handle.visible = show_band.value

    @show_recovered.on_update
    def _(_event: Any) -> None:
        recovered_handle.visible = show_recovered.value

    @show_slice.on_update
    def _(_event: Any) -> None:
        if show_slice.value:
            draw_slice(int(slice_index.value))
        elif slice_handle["handle"] is not None:
            slice_handle["handle"].remove()
            slice_handle["handle"] = None

    @slice_index.on_update
    def _(_event: Any) -> None:
        if show_slice.value:
            draw_slice(int(slice_index.value))

    print(f"Viser running on http://localhost:{port}")
    if block:
        try:
            server.sleep_forever()
        except KeyboardInterrupt:
            pass
    return server


def main(
    yaml_path: str,
    output: Optional[str] = None,
    voxel_z: Optional[float] = None,
    method: Optional[str] = None,
    show: bool = True,
    stride: int = 2,
    port: int = 8080,
) -> None:
    with open(yaml_path, "r") as handle:
        configs = yaml.safe_load(handle)

    map_info = configs["map"]
    node_info = configs["experiment_info_default"]["node_info"]
    esdf_config = dict(node_info.get("esdf") or {})
    if voxel_z is not None:
        esdf_config["voxel_z"] = voxel_z
    if method is not None:
        esdf_config["method"] = method

    map_dir = resolve_map_dir(map_info["dir"])
    map_name = map_info["name"]

    # Always load the source maps through the elevation pipeline so the ESDF is
    # built from the same costmap the planner uses.
    elevation_info = dict(node_info)
    elevation_info["map_type"] = "elevation"
    bitmap = get_map(
        map_name, map_dir=map_dir, map_size=map_info["size"], node_info=elevation_info
    )
    costmap = bitmap[..., 0].numpy()
    elevation = bitmap[..., 1].numpy()
    print(f"Loaded {map_name}: {elevation.shape}, elevation range "
          f"{elevation.min():.2f} to {elevation.max():.2f} m")

    start = time.perf_counter()
    esdf = eu.build_synthetic_esdf(
        elevation,
        costmap,
        voxel_xy=float(node_info["map_res"]),
        voxel_z=float(esdf_config.get("voxel_z", 0.25)),
        z_min=esdf_config.get("z_min"),
        z_margin=float(esdf_config.get("z_margin", 1.0)),
        method=str(esdf_config.get("method", "plane")),
        extrude_obstacles=bool(esdf_config.get("extrude_obstacles", False)),
    )
    meta = eu.esdf_metadata(esdf)
    print(
        f"Built ESDF in {time.perf_counter() - start:.2f} s: "
        f"{meta['nx']} x {meta['ny']} x {meta['nz']} voxels, "
        f"{esdf['distance'].nbytes / 1e6:.0f} MB distance + "
        f"{esdf['color'].nbytes / 1e6:.0f} MB colour"
    )

    # Correctness gate: the recovered surface must match the source heightmap,
    # otherwise the vehicle would see different terrain in ESDF mode.
    recovered = eu.surface_grid(esdf)
    surface_error = np.abs(recovered - elevation)
    cost_agreement = (
        (eu.cost_grid(esdf) <= eu.OBSTACLE_THRESHOLD)
        == (costmap <= eu.OBSTACLE_THRESHOLD)
    ).mean()
    print(f"Surface error vs heightmap: max {surface_error.max():.6f} m, "
          f"mean {surface_error.mean():.6f} m")
    print(f"Obstacle agreement vs costmap: {100 * cost_agreement:.2f}%")

    path = output or esdf_cache_path(node_info, map_dir, map_name)
    eu.save_esdf(path, esdf)
    print(f"Saved ESDF to: {path}")

    if show:
        visualize_esdf(esdf, elevation, costmap, stride=stride, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a synthetic ESDF from an elevation map and costmap"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="Configs/kinodynamic_example.yml",
        help="Path to YAML configuration file",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output .npz path")
    parser.add_argument("--voxel-z", type=float, default=None, help="Override z voxel size")
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["plane", "edt"],
        help="Distance field construction method",
    )
    parser.add_argument("--no-viser", action="store_true", help="Skip the Viser view")
    parser.add_argument("--stride", type=int, default=2, help="Visualization subsampling")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(STANDALONE_DIR, config_path)
    main(
        yaml_path=config_path,
        output=args.output,
        voxel_z=args.voxel_z,
        method=args.method,
        show=not args.no_viser,
        stride=args.stride,
        port=args.port,
    )
