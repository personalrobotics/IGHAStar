import numpy as np
import cv2
import torch
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from torch.utils.cpp_extension import load
from nav_msgs.msg import Path
import rospy
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from grid_map_msgs.msg import GridMap
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import NavSatFix
from mavros_msgs.msg import WaypointList
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
import os

def generate_normal( elev, k=3):
    dzdx = -cv2.Sobel(elev, cv2.CV_32F, 1, 0, ksize=k)
    dzdy = -cv2.Sobel(elev, cv2.CV_32F, 0, 1, ksize=k)
    dzdz = np.ones_like(elev)
    normal = np.stack((dzdx, dzdy, dzdz), axis=2)
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / norm
    return normal

def map_from_gridmap(matrix):
    return np.float32(
        cv2.flip(
            np.reshape(
                matrix.data,
                (matrix.layout.dim[1].size, matrix.layout.dim[0].size),
                order="F",
            ).T,
            -1,
        )
    )


def generate_costmap_from_BEVmap(
    BEV_lethal, BEV_normal, costmap_cosine_thresh=np.cos(np.radians(60))
):
    """Generate a costmap from BEV lethal and normal maps."""
    dot_product = BEV_normal[:, :, 2]
    angle_cost = np.where(dot_product >= costmap_cosine_thresh, 255, 0).astype(
        np.float32
    )
    costmap = 255.0 * (1 - BEV_lethal[..., 0])
    costmap = np.minimum(costmap, angle_cost)
    costmap[costmap < 255.0] = 0.0
    return costmap


def process_grid_map(heightmap, lethalmap=None, map_res=0.1, blur_kernel=0, costmap_cosine_thresh=np.cos(np.radians(60))):
    if lethalmap is None:
        lethalmap = np.zeros_like(heightmap, dtype=np.float32)
    normal = generate_normal(heightmap)
    costmap = generate_costmap_from_BEVmap(lethalmap, normal, costmap_cosine_thresh)
    if blur_kernel != 0:
        costmap = cv2.GaussianBlur(costmap, (blur_kernel, blur_kernel), 0)
        costmap[costmap < 255.0] = 0.0
    bitmap = torch.ones((heightmap.shape[0], heightmap.shape[1], 2), dtype=torch.float32)
    bitmap[..., 0] = torch.from_numpy(costmap)
    bitmap[..., 1] = torch.from_numpy(heightmap)
    offset = map_res * np.array(bitmap.shape[:2]) * 0.5
    return bitmap, offset

def clip_goal_to_map(bitmap, map_res=0.25, start=None, goal=None, num_samples=200):
    H, W = bitmap.shape[:2]
    H *= map_res
    W *= map_res
    
    x0, y0 = start[:2]
    x1, y1 = goal[:2]

    # Generate t in [0, 1]
    t_vals = torch.linspace(0, 1, num_samples, device=bitmap.device)

    # Line points: start + t * (goal - start)
    dx = x1 - x0
    dy = y1 - y0
    xs = x0 + t_vals * dx
    ys = y0 + t_vals * dy

    # Check which are inside map bounds
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    if torch.any(valid):
        last_valid_idx = torch.where(valid)[0][-1]
        goal[0] = xs[last_valid_idx]
        goal[1] = ys[last_valid_idx]
    else:
        # All points are out of bounds; snap to start
        goal[0] = x0
        goal[1] = y0
    return goal
    
def obtain_state(odom, state):
    ## obtain the state from the odometry and imu messages:
    quaternion = (
        odom.pose.pose.orientation.x,
        odom.pose.pose.orientation.y,
        odom.pose.pose.orientation.z,
        odom.pose.pose.orientation.w,
    )
    rpy = euler_from_quaternion(quaternion)
    state[0] = odom.pose.pose.position.x
    state[1] = odom.pose.pose.position.y
    state[2] = odom.pose.pose.position.z
    state[3] = rpy[0]
    state[4] = rpy[1]
    state[5] = rpy[2]
    state[6] = odom.twist.twist.linear.x
    state[7] = odom.twist.twist.linear.y
    state[8] = odom.twist.twist.linear.z
    return state

def publish_goal(goal, marker_pub):
    marker_array = MarkerArray()
    marker = Marker()
    marker.header.frame_id = "odom"
    marker.id = 0
    marker.type = marker.SPHERE
    marker.action = marker.ADD
    marker.scale.x = 4.0
    marker.scale.y = 4.0
    marker.scale.z = 0.1
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.pose.orientation.w = 1.0
    marker.pose.position.x = goal[0]
    marker.pose.position.y = goal[1]
    marker.pose.position.z = 1
    marker_array.markers.append(marker)
    marker_pub.publish(marker_array)

def publish_path(path, path_publisher):
    ros_path = Path()
    ros_path.header.stamp = rospy.Time.now()
    ros_path.header.frame_id = "map"  # or whatever your fixed frame is

    for waypoint in path:
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"

        pose.pose.position.x = waypoint[0]
        pose.pose.position.y = waypoint[1]
        pose.pose.position.z = 1 # would be nice to get the extended state here...

        # Convert theta to quaternion
        quat = quaternion_from_euler(0, 0, waypoint[2])
        pose.pose.orientation = Quaternion(*quat)

        ros_path.poses.append(pose)

    path_publisher.publish(ros_path)

def diagnostic_publisher(status, expansion_counter, time_taken, hysteresis, diagnostics_pub):
    diagnostics_array = DiagnosticArray()
    diagnostics_status = DiagnosticStatus()
    diagnostics_status.name = "planner"
    diagnostics_status.level = status
    diagnostics_status.values.append(KeyValue(key="Expansion Count", value=str(expansion_counter)))
    diagnostics_status.values.append(KeyValue(key="Time Taken (s)", value=f"{time_taken:.4f}"))
    diagnostics_status.values.append(KeyValue(key="Expansions/second", value=f"{expansion_counter/time_taken:.4f}"))
    diagnostics_status.values.append(KeyValue(key="Hysteresis", value=f"{hysteresis}"))

    diagnostics_array.status.append(diagnostics_status)
    diagnostics_pub.publish(diagnostics_array)

def set_headings(local_waypoints):
    target_Vhat = np.zeros_like(local_waypoints)  # 0 out everything
    for i in range(1, len(local_waypoints) - 1):  # all points except first and last
        V_prev = local_waypoints[i] - local_waypoints[i - 1]
        V_next = local_waypoints[i + 1] - local_waypoints[i]
        target_Vhat[i] = (V_next + V_prev) / np.linalg.norm(V_next + V_prev)
    target_Vhat[0] = local_waypoints[1] - local_waypoints[0]
    target_Vhat[0] /= np.linalg.norm(target_Vhat[0])
    target_Vhat[-1] = local_waypoints[-1] - local_waypoints[-2]
    target_Vhat[-1] /= np.linalg.norm(target_Vhat[-1])

    waypoints = np.zeros((len(local_waypoints), 4))
    waypoints[:, :2] = local_waypoints[:,:2]
    waypoints[:, 2] = np.arctan2(target_Vhat[:, 1], target_Vhat[:, 0])
    waypoints[:, 3] = 2.0
    return waypoints

earthRadius = 6378145.0
DEG_TO_RAD = 0.01745329252
RAD_TO_DEG = 57.2957795131

def calcposLLH( lat, lon, dX, dY):
    lat += dY / (earthRadius * DEG_TO_RAD)
    lon += dX / (earthRadius * np.cos(lat * DEG_TO_RAD) * DEG_TO_RAD)
    return lat, lon

def calcposNED( lat, lon, latReference, lonReference):
    Y = earthRadius * (lat - latReference) * DEG_TO_RAD
    X = (
        earthRadius
        * np.cos(latReference * DEG_TO_RAD)
        * (lon - lonReference)
        * DEG_TO_RAD
    )
    return X, Y

def start_goal_logic(bitmap, map_res, start_state, goal, map_center, offset, stop=False):
    start = np.zeros(4, dtype=np.float32)
    goal_ = np.zeros(4, dtype=np.float32)
    start[:2] = start_state[:2] + offset - map_center[:2]
    goal_[:2] = goal[:2] + offset - map_center[:2]
    goal_ = clip_goal_to_map(bitmap, map_res=map_res, start=start, goal=goal_)
    start[2] = start_state[2]
    start[3] = start_state[3]
    goal_[2] = goal[2]
    goal_[3] = 0.0 if stop else 2.0
    return torch.from_numpy(start).to(dtype=torch.float32), torch.from_numpy(goal_).to(dtype=torch.float32)

def get_local_frame_waypoints(waypoint_list, gps_origin):
    local_waypoints = []
    for waypoint in waypoint_list:
        if waypoint.frame != 3:
            continue
        lat = waypoint.x_lat
        lon = waypoint.y_long
        # generate X,Y locations using calcposNED and append to path
        X, Y = calcposNED(lat, lon, gps_origin[0], gps_origin[1])
        local_waypoints.append(np.array([X, Y, 0]))
    local_waypoints = set_headings(np.array(local_waypoints))
    return local_waypoints

def visualize_map_with_path(costmap, elevation_map, path, goal, state, wp_radius, map_center, map_size, resolution_inv):
    """
    Visualize the costmap and elevation map with the path, goal, and optionally the current state.
    Args:
        costmap: 2D numpy array (uint8)
        elevation_map: 2D numpy array (float)
        path: Nx4 or Nx2 numpy array (path points)
        goal: (2,) or (4,) array-like (goal position)
        state: (N,) array-like, current state (x, y, theta, ...)
        wp_radius: float, waypoint radius (for goal circle)
        map_center: (2,) or (3,) array-like, map center offset
        map_size: int, output image size in pixels
        resolution_inv: float, 1.0 / map resolution
    Returns:
        costmap_img: RGB image (np.uint8) for display with OpenCV
    """
    # Resize maps to output size
    costmap = cv2.resize(costmap, (map_size, map_size))
    elevation_map = cv2.resize(elevation_map, (map_size, map_size))

    # Colorize costmap: white for 255, pink for others
    costmap_color = np.clip(costmap, 0, 255).astype(np.uint8)
    pink = np.array([255, 105, 180], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype=np.uint8)
    color_map = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    mask_white = costmap_color == 255
    mask_pink = ~mask_white
    color_map[mask_white] = white
    color_map[mask_pink] = pink

    # Colorize elevation map and blend with costmap (where costmap is white)
    elev_norm = np.clip((elevation_map + 4) / 8, 0, 1)
    elev_uint8 = (elev_norm * 255).astype(np.uint8)
    elev_color = np.stack([elev_uint8] * 3, axis=-1)
    display_img = color_map.copy()
    display_img[mask_white] = elev_color[mask_white]

    # Draw goal (convert to local map frame)
    goal_disp = np.array(goal[:2]) - np.array(map_center[:2])
    goal_x = int(np.clip((goal_disp[0] * resolution_inv) + map_size // 2, 0, map_size - 1))
    goal_y = int(np.clip((goal_disp[1] * resolution_inv) + map_size // 2, 0, map_size - 1))
    radius = max(2, int(wp_radius * resolution_inv))
    cv2.circle(display_img, (goal_x, goal_y), radius, (255, 255, 255), 2)

    # Draw path
    if path is not None and len(path) > 0:
        path_disp = np.copy(path)
        path_disp[..., :2] -= np.array(map_center[:2])
        path_X = np.clip((path_disp[..., 0] * resolution_inv) + map_size // 2, 0, map_size - 1).astype(int)
        path_Y = np.clip((path_disp[..., 1] * resolution_inv) + map_size // 2, 0, map_size - 1).astype(int)
        car_width_px = max(1, int(0.15 * resolution_inv))
        if path.shape[1] > 3:
            velocity = path[..., 3]
            velocity_norm = (velocity - np.min(velocity)) / (np.max(velocity) - np.min(velocity) + 1e-6)
            velocity_color = np.clip((velocity_norm * 255), 0, 255).astype(int)
        else:
            velocity_color = np.full(len(path_X), 128, dtype=int)
        for i in range(len(path_X) - 1):
            cv2.line(display_img, (path_X[i], path_Y[i]), (path_X[i + 1], path_Y[i + 1]), (0, int(velocity_color[i]), 0), car_width_px)

    # Draw current state as a rectangle (if provided)
    if state is not None and len(state) >= 3:
        x = state[0] - map_center[0]
        y = state[1] - map_center[1]
        theta = np.pi - state[2]
        x_px = int(x * resolution_inv + map_size // 2)
        y_px = int(y * resolution_inv + map_size // 2)
        car_width_px = max(2, int(0.29 * resolution_inv))
        car_height_px = max(2, int(0.15 * resolution_inv))
        half_width = car_width_px // 2
        half_height = car_height_px // 2
        corners = np.array([
            [x_px - half_width, y_px - half_height],
            [x_px + half_width, y_px - half_height],
            [x_px + half_width, y_px + half_height],
            [x_px - half_width, y_px + half_height]
        ], dtype=np.int32)
        rotation_matrix = cv2.getRotationMatrix2D((x_px, y_px), np.degrees(theta), 1.0)
        rotated_corners = cv2.transform(np.array([corners]), rotation_matrix)[0]
        cv2.polylines(display_img, [rotated_corners], isClosed=True, color=(0, 0, 0), thickness=2)

    # Flip for visualization
    display_img = cv2.flip(display_img, 0)
    return display_img