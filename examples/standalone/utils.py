import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, tan, pi
from typing import Any, Dict, Optional, List, Tuple

STANDALONE_DIR = os.path.dirname(os.path.abspath(__file__))
if STANDALONE_DIR not in sys.path:
    sys.path.insert(0, STANDALONE_DIR)

import esdf_utils


def resolve_map_dir(map_dir: str) -> str:
    """Make a config map directory absolute, relative to this example folder."""
    if os.path.isabs(map_dir):
        return map_dir
    return os.path.join(STANDALONE_DIR, map_dir)


def get_map_type(node_info: Optional[dict]) -> str:
    """Which terrain representation the config asks for; elevation by default."""
    if not node_info:
        return "elevation"
    return str(node_info.get("map_type", "elevation")).lower()


def esdf_cache_path(node_info: dict, map_dir: str, map_name: str) -> str:
    """Where the synthetic ESDF for this map lives on disk."""
    cache = (node_info.get("esdf") or {}).get("cache")
    if cache:
        return cache if os.path.isabs(cache) else os.path.join(STANDALONE_DIR, cache)
    return os.path.join(map_dir, f"{map_name.split('.')[0]}_esdf.npz")


def get_map(
    map_name: str,
    map_dir: str = "",
    map_size: List[int] = [512, 512],
    node_info: Optional[dict] = None,
) -> torch.Tensor:
    assert map_dir != "", "Map directory must be specified."
    assert node_info is not None, "node_info must be provided."
    node = node_info["node_type"]
    if node == "simple":
        map_path = os.path.join(map_dir, map_name)
        bitmap = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        bitmap = cv2.resize(bitmap, (map_size[0], map_size[1]))
        bitmap = cv2.normalize(bitmap, None, 0, 255, cv2.NORM_MINMAX)
        bitmap = torch.from_numpy(bitmap).to(dtype=torch.float32)
        return bitmap
    elif node == "kinematic":
        map_path = os.path.join(map_dir, map_name)
        bitmap = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        bitmap = cv2.resize(bitmap, (map_size[0], map_size[1]))
        bitmap = cv2.normalize(bitmap, None, 0, 255, cv2.NORM_MINMAX)
        map_tensor = torch.from_numpy(bitmap).float().unsqueeze(2)
        bitmap = torch.cat((map_tensor, map_tensor), dim=2)
        bitmap[..., 1] *= 0
        bitmap[..., 0] = (bitmap[..., 0] > 1) * 255.0
        return bitmap
    elif node == "kinodynamic":
        bitmap = get_kinodynamic_map(map_name, map_dir, map_size, node_info)
        if get_map_type(node_info) != "esdf":
            return bitmap
        return get_kinodynamic_esdf(bitmap, map_name, map_dir, node_info)


def get_kinodynamic_map(
    map_name: str,
    map_dir: str,
    map_size: List[int],
    node_info: dict,
) -> torch.Tensor:
    """Elevation map + costmap as an ``H x W x 2`` tensor."""
    name = map_name.split(".")[0]
    map_path = os.path.join(map_dir, name)
    elevation_map_path = map_path + "_height.npy"
    if not os.path.exists(elevation_map_path):
        costmap = cv2.imread(map_path + ".png", cv2.IMREAD_GRAYSCALE)
        _map_size = [0, 0]
        _map_size[0] = int(map_path.split("_")[-1])
        _map_size[1] = _map_size[0]
        costmap = cv2.normalize(costmap, None, 0, 255, cv2.NORM_MINMAX)
        costmap = cv2.resize(costmap, (_map_size[0], _map_size[0]))
        costmap = (costmap > 1) * 255.0
        elevation_map = np.zeros((_map_size[0], _map_size[0]), dtype=np.float32)
        bitmap = torch.ones((_map_size[0], _map_size[1], 2), dtype=torch.float32)
    else:
        elevation_map = np.load(elevation_map_path)
        elevation_map -= np.min(elevation_map)
        elevation_map = cv2.resize(elevation_map, (map_size[0], map_size[1]))
        costmap = compute_surface_normals(elevation_map, node_info["max_theta"] * 57.3)
        bitmap = torch.ones((map_size[0], map_size[1], 2), dtype=torch.float32)
    bitmap[..., 1] = torch.from_numpy(elevation_map)
    bitmap[..., 0] = torch.from_numpy(costmap)
    return bitmap


def get_kinodynamic_esdf(
    bitmap: torch.Tensor,
    map_name: str,
    map_dir: str,
    node_info: dict,
) -> torch.Tensor:
    """Synthetic ESDF as an ``H x W x nz x 4`` tensor of ``[distance, R, G, B]``.

    The cached ESDF is reused when it matches the map and the requested voxel
    sizes, and rebuilt otherwise.  The grid geometry is written back into
    ``node_info["esdf"]`` because the C++ environment reads ``voxel_z`` and
    ``z_min`` from the config: the tensor carries voxel data only.
    """
    esdf_config = node_info.setdefault("esdf", {}) or {}
    node_info["esdf"] = esdf_config
    voxel_xy = float(node_info["map_res"])
    voxel_z = float(esdf_config.get("voxel_z", 0.25))
    path = esdf_cache_path(node_info, map_dir, map_name)

    esdf = None
    if os.path.exists(path):
        esdf = esdf_utils.load_esdf(path)
        expected_shape = (bitmap.shape[0], bitmap.shape[1])
        if esdf["distance"].shape[:2] != expected_shape or not np.isclose(
            esdf["voxel_z"], voxel_z
        ):
            print(f"Cached ESDF at {path} does not match the config, rebuilding")
            esdf = None
    if esdf is None:
        print("Building synthetic ESDF...")
        esdf = esdf_utils.build_synthetic_esdf(
            bitmap[..., 1].numpy(),
            bitmap[..., 0].numpy(),
            voxel_xy=voxel_xy,
            voxel_z=voxel_z,
            z_min=esdf_config.get("z_min"),
            z_margin=float(esdf_config.get("z_margin", 1.0)),
            method=str(esdf_config.get("method", "plane")),
            extrude_obstacles=bool(esdf_config.get("extrude_obstacles", False)),
        )
        esdf_utils.save_esdf(path, esdf)
        print(f"Saved synthetic ESDF to: {path}")

    esdf_config.update(esdf_utils.esdf_metadata(esdf))
    return esdf_utils.esdf_to_world_tensor(esdf)


def esdf_from_config(bitmap: torch.Tensor, node_info: dict) -> Dict[str, Any]:
    """Rebuild the ESDF dict from a world tensor plus the synced config metadata."""
    esdf_config = node_info.get("esdf") or {}
    return esdf_utils.esdf_from_world_tensor(
        bitmap,
        voxel_xy=float(esdf_config.get("voxel_xy", node_info["map_res"])),
        voxel_z=float(esdf_config.get("voxel_z", 0.25)),
        z_min=float(esdf_config.get("z_min", 0.0)),
    )


def terrain_layers(
    bitmap: torch.Tensor, node_info: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """``(costmap, elevation)`` view of a world tensor, whatever its representation."""
    if bitmap.dim() == 3:
        return bitmap[..., 0].cpu().numpy(), bitmap[..., 1].cpu().numpy()
    assert node_info is not None, "node_info is required to interpret an ESDF tensor"
    esdf = esdf_from_config(bitmap, node_info)
    return esdf_utils.cost_grid(esdf), esdf_utils.surface_grid(esdf)


def compute_surface_normals(elevation: np.ndarray, threshold_deg: float) -> np.ndarray:
    BEV_normal = np.copy(elevation)
    BEV_normal = cv2.resize(
        BEV_normal,
        (int(BEV_normal.shape[0] * 4), int(BEV_normal.shape[0] * 4)),
        cv2.INTER_AREA,
    )
    BEV_normal = cv2.GaussianBlur(BEV_normal, (3, 3), 0)
    BEV_normal = cv2.resize(
        BEV_normal,
        (int(BEV_normal.shape[0] / 4), int(BEV_normal.shape[0] / 4)),
        cv2.INTER_AREA,
    )
    # Compute the normal vector as the cross product of the x and y gradients
    normal_x = -cv2.Sobel(BEV_normal, cv2.CV_64F, 1, 0, ksize=3)
    normal_y = -cv2.Sobel(BEV_normal, cv2.CV_64F, 0, 1, ksize=3)
    normal_z = np.ones_like(BEV_normal)
    normals = np.stack([normal_x, normal_y, normal_z], axis=-1)
    # Normalize the normal vectors
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (norms + 1e-6)
    dot_product = normals[
        :, :, 2
    ]  # This is equivalent to cosine of the angle to vertical
    # Convert the threshold angle from degrees to cosine
    threshold_cos = np.cos(np.radians(threshold_deg))
    # Create the costmap based on the threshold
    costmap = np.where(dot_product >= threshold_cos, 255, 0)
    costmap = costmap.astype(np.float32)
    return costmap


def show_map(
    plt: Any,
    bitmap: torch.Tensor,
    node_type: str,
    alpha: float = 0.6,
    node_info: Optional[dict] = None,
) -> None:
    if node_type == "simple":
        plt.imshow(bitmap, cmap="gray", alpha=alpha)
    elif node_type == "kinodynamic":
        costmap, elevation_map = terrain_layers(bitmap, node_info)
        costmap_color = np.clip(costmap, 0, 255).astype(np.uint8)
        pink = np.array([255, 105, 180], dtype=np.uint8)  # BGR format
        white = np.array([255, 255, 255], dtype=np.uint8)
        color_map = np.zeros(
            (costmap_color.shape[0], costmap_color.shape[1], 3), dtype=np.uint8
        )
        mask_white = costmap_color >= 250
        mask_pink = ~mask_white
        color_map[mask_white] = white
        color_map[mask_pink] = pink
        costmap_color = color_map
        vmin = np.min(elevation_map)
        vmax = np.max(elevation_map)
        elev_norm = np.clip((elevation_map - vmin) / (vmax - vmin), 0, 1)
        elev_uint8 = (elev_norm * 255).astype(np.uint8)
        elev_color = np.stack([elev_uint8] * 3, axis=-1)
        costmap = costmap_color
        costmap[mask_white] = elev_color[mask_white]
        plt.imshow(costmap)
    elif node_type == "kinematic":
        costmap = bitmap[..., 0]
        plt.imshow(costmap, cmap="gray", alpha=alpha)


def rot_mat_2d(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s], [s, c]])


def plot_arrow(
    x: float,
    y: float,
    yaw: float,
    length: float = 1.0,
    width: float = 0.5,
    fc: str = "r",
    ec: str = "k",
) -> None:
    """Plot arrow."""
    # Treat any array-like (list / numpy array / numpy scalar with ndim>0) as a
    # batch; numpy scalars (e.g. numpy.float32) are NOT Python floats, so check
    # dimensionality rather than the exact type.
    if np.ndim(x) > 0:
        for i_x, i_y, i_yaw in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw)
    else:
        plt.arrow(
            x,
            y,
            length * cos(yaw),
            length * sin(yaw),
            fc=fc,
            ec=ec,
            head_width=width,
            head_length=width,
            alpha=0.4,
        )


def plot_car(
    plt: Any,
    x: float,
    y: float,
    yaw: float,
    color: str = "-r",
    map_res: float = 0.1,
    W: float = 1.5,
    LF: float = 1.3,
    LB: float = 1.3,
    label: Optional[str] = None,
    width: int = 1,
    zorder: int = 0,
) -> None:
    VRX = [LF, LF, -LB, -LB, LF]
    VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]
    car_color = color
    c, s = cos(yaw), sin(yaw)
    rot = rot_mat_2d(-yaw)
    car_outline_x, car_outline_y = [], []
    for rx, ry in zip(VRX, VRY):
        converted_xy = np.stack([rx, ry]).T @ rot
        converted_xy = converted_xy / map_res
        car_outline_x.append(converted_xy[0] + x)
        car_outline_y.append(converted_xy[1] + y)

    arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
    plot_arrow(arrow_x, arrow_y, arrow_yaw, length=1.5 / map_res)
    if label is not None:
        plt.plot(
            car_outline_x,
            car_outline_y,
            car_color,
            label=label,
            linewidth=width,
            zorder=zorder,
        )
    else:
        plt.plot(
            car_outline_x, car_outline_y, car_color, linewidth=width, zorder=zorder
        )


def pi_2_pi(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


def move(
    x: float, y: float, yaw: float, distance: float, steer: float, L: float = 3.0
) -> Tuple[float, float, float]:
    x += distance * cos(yaw)
    y += distance * sin(yaw)
    yaw += pi_2_pi(distance * tan(steer) / L)  # distance/2
    return x, y, yaw
