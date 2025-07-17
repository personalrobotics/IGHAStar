import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, tan, pi
from typing import Any, Optional, List, Tuple

def get_map(map_name: str, map_dir: str = "", map_size: List[int] = [512, 512], node_info: Optional[dict] = None) -> torch.Tensor:
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
            costmap = compute_surface_normals(elevation_map, node_info["max_theta"]*57.3)
            bitmap = torch.ones((map_size[0], map_size[1], 2), dtype=torch.float32)
        bitmap[..., 1] = torch.from_numpy(elevation_map)
        bitmap[..., 0] = torch.from_numpy(costmap)
        return bitmap

def compute_surface_normals(elevation: np.ndarray, threshold_deg: float) -> np.ndarray:
    BEV_normal = np.copy(elevation)
    BEV_normal = cv2.resize(BEV_normal, (int(BEV_normal.shape[0]*4), int(BEV_normal.shape[0]*4)), cv2.INTER_AREA)
    BEV_normal = cv2.GaussianBlur(BEV_normal, (3,3), 0)
    BEV_normal = cv2.resize(BEV_normal, (int(BEV_normal.shape[0]/4), int(BEV_normal.shape[0]/4)), cv2.INTER_AREA)
    # Compute the normal vector as the cross product of the x and y gradients
    normal_x = -cv2.Sobel(BEV_normal, cv2.CV_64F, 1, 0, ksize=3)
    normal_y = -cv2.Sobel(BEV_normal, cv2.CV_64F, 0, 1, ksize=3)
    normal_z = np.ones_like(BEV_normal)
    normals = np.stack([normal_x, normal_y, normal_z], axis=-1)
    # Normalize the normal vectors
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (norms + 1e-6)
    dot_product = normals[:, :, 2]  # This is equivalent to cosine of the angle to vertical
    # Convert the threshold angle from degrees to cosine
    threshold_cos = np.cos(np.radians(threshold_deg))
    # Create the costmap based on the threshold
    costmap = np.where(dot_product >= threshold_cos, 255, 0)
    costmap = costmap.astype(np.float32)
    return costmap

def show_map(plt: Any, bitmap: torch.Tensor, node_type: str, alpha: float = 0.6) -> None:
    if node_type == "simple":
        plt.imshow(bitmap, cmap='gray', alpha=alpha)
    elif node_type == "kinodynamic":
        costmap = bitmap[..., 0].cpu().numpy()
        elevation_map = bitmap[..., 1].cpu().numpy()
        costmap_color = np.clip(costmap, 0, 255).astype(np.uint8)
        pink = np.array([255, 105, 180], dtype=np.uint8)  # BGR format
        white = np.array([255, 255, 255], dtype=np.uint8)
        color_map = np.zeros((costmap_color.shape[0], costmap_color.shape[1], 3), dtype=np.uint8)
        mask_white = costmap_color == 255
        mask_pink = ~mask_white
        color_map[mask_white] = white
        color_map[mask_pink] = pink
        costmap_color = color_map
        vmin = np.min(elevation_map)
        vmax = np.max(elevation_map)
        elev_norm = np.clip((elevation_map - vmin) / (vmax - vmin), 0, 1)
        elev_uint8 = (elev_norm * 255).astype(np.uint8)
        elev_color = np.stack([elev_uint8]*3, axis=-1)
        costmap = costmap_color
        costmap[mask_white] = elev_color[mask_white]
        plt.imshow(costmap)
    elif node_type == "kinematic":
        costmap = bitmap[..., 0]
        plt.imshow(costmap, cmap='gray', alpha=alpha)

def rot_mat_2d(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s], [s, c]])

def plot_arrow(x: float, y: float, yaw: float, length: float = 1.0, width: float = 0.5, fc: str = "r", ec: str = "k") -> None:
    """Plot arrow."""
    if not isinstance(x, float):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw)
    else:
        plt.arrow(x, y, length * cos(yaw), length * sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width, alpha=0.4)

def plot_car(plt: Any, x: float, y: float, yaw: float, color: str = "-r", map_res: float = 0.1, W: float = 1.5, LF: float = 1.3, LB: float = 1.3, label: Optional[str] = None, width: int = 1, zorder: int = 0) -> None:
    VRX = [LF, LF, -LB, -LB, LF]
    VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]
    car_color = color
    c, s = cos(yaw), sin(yaw)
    rot = rot_mat_2d(-yaw)
    car_outline_x, car_outline_y = [], []
    for rx, ry in zip(VRX, VRY):
        converted_xy = np.stack([rx, ry]).T @ rot
        converted_xy = converted_xy / map_res
        car_outline_x.append(converted_xy[0]+x)
        car_outline_y.append(converted_xy[1]+y)

    arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
    plot_arrow(arrow_x, arrow_y, arrow_yaw, length = 1.5/map_res)
    if label is not None:
        plt.plot(car_outline_x, car_outline_y, car_color, label=label, linewidth=width, zorder=zorder)
    else:
        plt.plot(car_outline_x, car_outline_y, car_color, linewidth=width,zorder=zorder)

def pi_2_pi(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi

def move(x: float, y: float, yaw: float, distance: float, steer: float, L: float = 3.0) -> Tuple[float, float, float]:
    x += distance * cos(yaw)
    y += distance * sin(yaw)
    yaw += pi_2_pi(distance * tan(steer) / L)  # distance/2
    return x, y, yaw
