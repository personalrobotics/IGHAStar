"""Synthetic ESDF construction, storage and query helpers.

The synthetic ESDF stands in for a real (nvblox) ESDF layer until one is wired
up.  It is a dense 3D grid indexed ``[iy, ix, k]`` in C order, so the z axis is
contiguous within a column: that is the layout the CUDA query walks when it
marches down a column looking for the terrain surface.

Two layers are stored:

* ``distance``: signed distance to the terrain surface in metres, negative
  below the surface.
* ``color``: an RGB voxel colour mirroring nvblox's ColorVoxel layer.  White is
  free space, black is an obstacle.

Terrain height is recovered from the zero crossing of the distance layer rather
than being stored, which is what makes the ESDF a drop-in replacement for the
elevation map.
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:  # torch is only needed to hand the ESDF to the planner
    import torch
except ImportError:  # pragma: no cover - torch is a hard dep of the planner
    torch = None  # type: ignore

FREE_COLOR = (255, 255, 255)
OBSTACLE_COLOR = (0, 0, 0)
# Matches the threshold in check_validity_batch_kernel (kinodynamic.cu).
OBSTACLE_THRESHOLD = 250.0
# ITU-R BT.601 luminance weights, matching map_to_cost() in kinodynamic.cu.
LUMINANCE_WEIGHTS = (0.299, 0.587, 0.114)


def _slope_cosine(elevation: np.ndarray, voxel_xy: float) -> np.ndarray:
    """Cosine of the terrain slope, i.e. the z component of the unit normal."""
    dzdy, dzdx = np.gradient(elevation.astype(np.float64), voxel_xy)
    return (1.0 / np.sqrt(1.0 + dzdx * dzdx + dzdy * dzdy)).astype(np.float32)


def _z_grid(
    elevation: np.ndarray,
    voxel_z: float,
    z_min: Optional[float],
    z_margin: float,
) -> Tuple[float, int]:
    lo = float(elevation.min()) - z_margin if z_min is None else float(z_min)
    hi = float(elevation.max()) + z_margin
    # At least one sample must sit below the lowest terrain point so that every
    # column contains a sign change for the surface query to interpolate.
    lo = min(lo, float(elevation.min()) - voxel_z)
    nz = int(np.ceil((hi - lo) / voxel_z)) + 1
    return lo, nz


def _plane_distance(
    elevation: np.ndarray, z_levels: np.ndarray, voxel_xy: float
) -> np.ndarray:
    """Signed distance to the local tangent plane of the terrain.

    Exact for planar terrain and correct to sub-voxel precision at the zero
    crossing, which is what keeps the recovered height (and therefore the roll
    and pitch the vehicle model sees) identical to the elevation map.  The
    magnitude never exceeds the vertical gap to the surface, so the column
    march in the CUDA query cannot step past the surface.
    """
    cos_slope = _slope_cosine(elevation, voxel_xy)
    gap = z_levels[None, None, :] - elevation[:, :, None].astype(np.float32)
    return (gap * cos_slope[:, :, None]).astype(np.float32)


def _edt_distance(
    occupancy: np.ndarray, voxel_xy: float, voxel_z: float
) -> np.ndarray:
    """Euclidean signed distance from a voxelised occupancy grid.

    A true 3D Euclidean distance transform, but the zero crossing lands halfway
    between voxel centres, so the recovered surface is quantised to ``voxel_z``.
    Kept as a reference implementation and for obstacle extrusion experiments;
    ``method="plane"`` is what preserves the dynamics.
    """
    from scipy.ndimage import distance_transform_edt

    sampling = (voxel_xy, voxel_xy, voxel_z)
    outside = distance_transform_edt(~occupancy, sampling=sampling)
    inside = distance_transform_edt(occupancy, sampling=sampling)
    return (outside - inside).astype(np.float32)


def build_synthetic_esdf(
    elevation: np.ndarray,
    costmap: np.ndarray,
    voxel_xy: float = 0.1,
    voxel_z: float = 0.25,
    z_min: Optional[float] = None,
    z_margin: float = 1.0,
    method: str = "plane",
    extrude_obstacles: bool = False,
    obstacle_height: float = 2.0,
    store_float16: bool = True,
) -> Dict[str, Any]:
    """Build a synthetic ESDF from an elevation map and a costmap.

    Geometry comes from the elevation map alone; obstacles from the costmap are
    encoded in the colour layer only.  Extruding them into the geometry would
    move the surface for obstacle cells and change the vehicle's roll and pitch
    there, so it is off by default and only supported by the EDT method.
    """
    elevation = np.ascontiguousarray(elevation, dtype=np.float32)
    costmap = np.ascontiguousarray(costmap, dtype=np.float32)
    if elevation.shape != costmap.shape:
        raise ValueError(
            "elevation %s and costmap %s must have the same shape"
            % (elevation.shape, costmap.shape)
        )

    z_min_val, nz = _z_grid(elevation, voxel_z, z_min, z_margin)
    z_levels = (z_min_val + voxel_z * np.arange(nz)).astype(np.float32)

    if method == "plane":
        if extrude_obstacles:
            raise ValueError(
                "extrude_obstacles requires method='edt'; the plane method "
                "represents the terrain surface only"
            )
        distance = _plane_distance(elevation, z_levels, voxel_xy)
    elif method == "edt":
        occupancy = z_levels[None, None, :] <= elevation[:, :, None]
        if extrude_obstacles:
            walls = (costmap <= OBSTACLE_THRESHOLD)[:, :, None] & (
                z_levels[None, None, :]
                <= (elevation[:, :, None] + obstacle_height)
            )
            occupancy = occupancy | walls
        distance = _edt_distance(occupancy, voxel_xy, voxel_z)
    else:
        raise ValueError("unknown ESDF method %r (expected 'plane' or 'edt')" % method)

    obstacle = costmap <= OBSTACLE_THRESHOLD
    color = np.empty(elevation.shape + (3,), dtype=np.uint8)
    color[~obstacle] = FREE_COLOR
    color[obstacle] = OBSTACLE_COLOR
    # The colour is constant along a column so the surface colour lookup is
    # insensitive to a one-voxel error in the recovered surface index.
    color = np.repeat(color[:, :, None, :], nz, axis=2)

    return {
        "distance": distance.astype(np.float16 if store_float16 else np.float32),
        "color": color,
        "voxel_xy": float(voxel_xy),
        "voxel_z": float(voxel_z),
        "z_min": float(z_min_val),
        "origin": (0.0, 0.0),
        "method": method,
    }


def esdf_metadata(esdf: Dict[str, Any]) -> Dict[str, Any]:
    """Geometry of the ESDF grid, as the C++ environment needs to see it."""
    ny, nx, nz = esdf["distance"].shape
    return {
        "voxel_xy": float(esdf["voxel_xy"]),
        "voxel_z": float(esdf["voxel_z"]),
        "z_min": float(esdf["z_min"]),
        "nz": int(nz),
        "nx": int(nx),
        "ny": int(ny),
    }


def save_esdf(path: str, esdf: Dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savez_compressed(
        path,
        distance=esdf["distance"],
        color=esdf["color"],
        voxel_xy=np.float32(esdf["voxel_xy"]),
        voxel_z=np.float32(esdf["voxel_z"]),
        z_min=np.float32(esdf["z_min"]),
        origin=np.asarray(esdf["origin"], dtype=np.float32),
        method=np.array(esdf.get("method", "plane")),
    )


def load_esdf(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "distance": data["distance"],
            "color": data["color"],
            "voxel_xy": float(data["voxel_xy"]),
            "voxel_z": float(data["voxel_z"]),
            "z_min": float(data["z_min"]),
            "origin": tuple(np.asarray(data["origin"]).tolist()),
            "method": str(data["method"]),
        }


def esdf_to_world_tensor(esdf: Dict[str, Any]) -> "torch.Tensor":
    """Pack the ESDF into the ``H x W x nz x 4`` tensor ``set_world()`` expects.

    Channel 0 is the signed distance, channels 1-3 are the RGB colour.
    """
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required to build the world tensor")
    distance = np.asarray(esdf["distance"], dtype=np.float32)
    color = np.asarray(esdf["color"], dtype=np.float32)
    world = np.empty(distance.shape + (4,), dtype=np.float32)
    world[..., 0] = distance
    world[..., 1:] = color
    return torch.from_numpy(world)


def esdf_from_world_tensor(
    world: Any, voxel_xy: float, voxel_z: float, z_min: float
) -> Dict[str, Any]:
    """Inverse of :func:`esdf_to_world_tensor`, for visualization and checks."""
    array = world.numpy() if hasattr(world, "numpy") else np.asarray(world)
    return {
        "distance": np.ascontiguousarray(array[..., 0], dtype=np.float32),
        "color": np.ascontiguousarray(array[..., 1:], dtype=np.uint8),
        "voxel_xy": float(voxel_xy),
        "voxel_z": float(voxel_z),
        "z_min": float(z_min),
        "origin": (0.0, 0.0),
        "method": "unpacked",
    }


def _surface_z_from_columns(
    columns: np.ndarray, voxel_z: float, z_min: float
) -> np.ndarray:
    """Zero crossing of one or more distance columns, in metres.

    Mirrors ``esdf_surface_z()`` in kinodynamic.cu: take the topmost sample that
    is inside the terrain and linearly interpolate against the sample above it.
    """
    nz = columns.shape[-1]
    below = columns <= 0.0
    has_crossing = below.any(axis=-1)
    k_lo = nz - 1 - np.argmax(below[..., ::-1], axis=-1)
    k_lo = np.where(has_crossing, k_lo, 0)
    k_hi = np.minimum(k_lo + 1, nz - 1)
    d_lo = np.take_along_axis(columns, k_lo[..., None], axis=-1)[..., 0]
    d_hi = np.take_along_axis(columns, k_hi[..., None], axis=-1)[..., 0]
    denom = d_hi - d_lo
    safe = np.abs(denom) > 1e-12
    frac = np.where(safe, -d_lo / np.where(safe, denom, 1.0), 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    return (z_min + (k_lo + frac) * voxel_z).astype(np.float32)


def _cell_indices(esdf: Dict[str, Any], x: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    ny, nx, _ = esdf["distance"].shape
    res_inv = 1.0 / esdf["voxel_xy"]
    ix = np.clip((np.asarray(x, dtype=np.float32) * res_inv).astype(np.int32), 0, nx - 1)
    iy = np.clip((np.asarray(y, dtype=np.float32) * res_inv).astype(np.int32), 0, ny - 1)
    return ix, iy


def surface_z_from_esdf(esdf: Dict[str, Any], x: Any, y: Any) -> np.ndarray:
    """Terrain height at world coordinates ``(x, y)``; NumPy mirror of the GPU query."""
    ix, iy = _cell_indices(esdf, x, y)
    columns = np.asarray(esdf["distance"][iy, ix], dtype=np.float32)
    return _surface_z_from_columns(columns, esdf["voxel_z"], esdf["z_min"])


def surface_grid(esdf: Dict[str, Any]) -> np.ndarray:
    """Recovered terrain height for every column of the ESDF."""
    columns = np.asarray(esdf["distance"], dtype=np.float32)
    return _surface_z_from_columns(columns, esdf["voxel_z"], esdf["z_min"])


def surface_color_grid(esdf: Dict[str, Any]) -> np.ndarray:
    """Colour of the surface voxel for every column of the ESDF."""
    surface_z = surface_grid(esdf)
    nz = esdf["distance"].shape[2]
    k = np.clip(
        ((surface_z - esdf["z_min"]) / esdf["voxel_z"]).astype(np.int32), 0, nz - 1
    )
    iy, ix = np.indices(surface_z.shape)
    return esdf["color"][iy, ix, k]


def luminance(color: np.ndarray) -> np.ndarray:
    """Traversability value of an RGB colour; mirrors map_to_cost() in ESDF mode."""
    color = np.asarray(color, dtype=np.float32)
    weights = np.asarray(LUMINANCE_WEIGHTS, dtype=np.float32)
    return (color * weights).sum(axis=-1)


def cost_grid(esdf: Dict[str, Any]) -> np.ndarray:
    """Costmap-equivalent view of the ESDF colour layer (0 obstacle, 255 free)."""
    return luminance(surface_color_grid(esdf))
