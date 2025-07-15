import numpy as np
import cv2
import time
import threading
import torch


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
    costmap = cv2.GaussianBlur(costmap, (3, 3), 0)
    costmap[costmap < 255.0] = 0.0
    return costmap


def convert_global_path_to_bng(bng_interface=None, path=None, Map_config=None):
    """Convert a global path to BeamNG waypoints with elevation and quaternion."""
    target_wp = []
    map_res = Map_config["map_res"]
    full_map_size = int(9000 * 0.1 / map_res)  # 900m at 0.1 m resolution
    offset = full_map_size // 2
    for i in range(0, len(path), 10):
        wp = np.zeros(7)
        wp[:2] = path[i][:2]
        # Query the heightmap at the xy location, given the map resolution
        x = int((path[i][0] / map_res) + offset)
        y = int((path[i][1] / map_res) + offset)
        z = (
            bng_interface.elevation_map_full[y, x] + 0.5
        )  # 0.5 m offset so that we start above ground
        wp[2] = z
        rpy = np.array([0, 0, path[i][2]])  # yaw angle
        quat = bng_interface.quat_from_rpy(rpy)
        bng_quat = bng_interface.convert_REP103_to_beamng(quat)
        wp[3:] = bng_quat
        target_wp.append(wp)
    return np.array(target_wp)


def update_goal(goal, pos, target_WP, current_wp_index, lookahead, wp_radius=1.0):
    """Update the goal position based on lookahead and proximity to final waypoint."""
    final_wp = target_WP[-1]
    dx = final_wp[0] - pos[0]
    dy = final_wp[1] - pos[1]
    dist_to_goal = np.hypot(dx, dy)
    success = False
    if dist_to_goal <= lookahead:
        # Close enough to final goal
        goal_x, goal_y = final_wp[0], final_wp[1]
        if dist_to_goal < wp_radius:
            success = True
    else:
        # Project onto lookahead circle in direction of global goal
        angle = np.arctan2(dy, dx)
        goal_x = pos[0] + lookahead * np.cos(angle)
        goal_y = pos[1] + lookahead * np.sin(angle)
    goal = [goal_x, goal_y]
    return goal, success, current_wp_index


def steering_limiter(steer, state, RPS_config):
    """Limit steering to prevent rollovers and respect physical constraints."""
    steering_setpoint = steer * RPS_config["steering_max"]
    whspd2 = max(1.0, np.linalg.norm(state[6:8])) ** 2  # speed squared in world frame
    Aylim = (
        0.5
        * RPS_config["track_width"]
        * np.cos(state[3] + state[12] * 0.5)
        / RPS_config["cg_height"]
    ) * max(1.0, abs(state[11]))
    steering_limit_max = (
        np.arctan2(RPS_config["wheelbase"] * (Aylim - 9.8 * np.sin(state[3])), whspd2)
        + RPS_config["steer_slack"] * RPS_config["steering_max"]
    )
    steering_limit_min = (
        -np.arctan2(RPS_config["wheelbase"] * (Aylim + 9.8 * np.sin(state[3])), whspd2)
        - RPS_config["steer_slack"] * RPS_config["steering_max"]
    )
    steering_setpoint = min(
        steering_limit_max, max(steering_limit_min, steering_setpoint)
    )
    delta_steering = 0
    Ay = state[10]
    Ay_error = 0
    if abs(Ay) > Aylim:
        if Ay >= 0:
            Ay_error = min(Aylim - Ay, 0)
            delta_steering = (
                4.0
                * (
                    Ay_error * RPS_config["accel_gain"]
                    - RPS_config["roll_rate_gain"] * abs(state[11]) * state[12]
                )
                * (np.cos(steering_setpoint) ** 2)
                * RPS_config["wheelbase"]
                / whspd2
            )
            delta_steering = min(delta_steering, 0)
        else:
            Ay_error = max(-Aylim - Ay, 0)
            delta_steering = (
                4.0
                * (
                    Ay_error * RPS_config["accel_gain"]
                    - RPS_config["roll_rate_gain"] * abs(state[11]) * state[12]
                )
                * (np.cos(steering_setpoint) ** 2)
                * RPS_config["wheelbase"]
                / whspd2
            )
            delta_steering = max(
                delta_steering, 0
            )  # prevents turning in the opposite direction and causing a rollover
        steering_setpoint += delta_steering
    steering_setpoint = steering_setpoint / RPS_config["steering_max"]
    steering_setpoint = min(max(steering_setpoint, -1.0), 1.0)
    return steering_setpoint


class PlannerVis:
    def __init__(self, map_size, resolution_inv):
        self.map_size = map_size
        self.resolution_inv = resolution_inv
        self.costmap = None
        self.elevation_map = None
        self.lock = threading.Lock()
        self.path = None
        self.states = None
        self.cosine_thresh = np.cos(np.radians(45))
        self.vis_thread = threading.Thread(target=self.costmap_vis)
        self.vis_thread.daemon = True
        self.vis_thread.start()

    def update_vis(
        self,
        states,
        path,
        costmap,
        elevation_map,
        resolution_inv,
        goal,
        expansion_counter,
        hysteresis,
    ):
        with self.lock:
            if isinstance(states, torch.Tensor):
                self.states = states.cpu().numpy()
            else:
                self.states = np.copy(states)
            self.path = path
            self.costmap = costmap
            self.elevation_map = elevation_map
            self.resolution_inv = resolution_inv
            self.goal = goal
            self.hysteresis = hysteresis
            self.expansion_counter = expansion_counter

    def generate_costmap_from_BEVmap(self, normal):
        dot_product = normal[:, :, 2]
        costmap = np.where(dot_product >= self.cosine_thresh, 255, 0).astype(np.float32)
        return costmap

    def costmap_vis(self):
        while True:
            if (
                self.states is not None
                and self.costmap is not None
                and self.elevation_map is not None
            ):
                # Normalize costmap to 8-bit grayscale (0-255)
                if len(self.costmap.shape) == 3:
                    costmap_color = self.generate_costmap_from_BEVmap(self.costmap)
                    costmap_color = np.clip(costmap_color, 0, 255).astype(np.uint8)
                else:
                    costmap_color = np.clip(self.costmap, 0, 255).astype(np.uint8)
                pink = np.array([255, 105, 180], dtype=np.uint8)  # BGR format
                white = np.array([255, 255, 255], dtype=np.uint8)
                color_map = np.zeros(
                    (costmap_color.shape[0], costmap_color.shape[1], 3), dtype=np.uint8
                )
                mask_white = costmap_color == 255
                mask_pink = ~mask_white
                color_map[mask_white] = white
                color_map[mask_pink] = pink
                costmap_color = color_map
                # Normalize elevation map to 8-bit and apply colormap
                elev_norm = np.clip((self.elevation_map + 4) / 8, 0, 1)
                elev_uint8 = (elev_norm * 255).astype(np.uint8)
                elev_color = np.stack([elev_uint8] * 3, axis=-1)
                # Blend the two images
                costmap = costmap_color
                costmap[mask_white] = elev_color[mask_white]
                # Draw goal
                goal_x = int(
                    np.clip(
                        (self.goal[0] * self.resolution_inv) + self.map_size // 2,
                        0,
                        self.map_size - 1,
                    )
                )
                goal_y = int(
                    np.clip(
                        (self.goal[1] * self.resolution_inv) + self.map_size // 2,
                        0,
                        self.map_size - 1,
                    )
                )
                radius = int(5 * self.resolution_inv)
                cv2.circle(costmap, (goal_x, goal_y), radius, (0, 0, 255), 1)
                # Draw path
                if self.path is not None:
                    path_X = np.array(
                        np.clip(
                            (self.path[..., 0] * self.resolution_inv)
                            + self.map_size // 2,
                            0,
                            self.map_size - 1,
                        ),
                        dtype=int,
                    )
                    path_Y = np.array(
                        np.clip(
                            (self.path[..., 1] * self.resolution_inv)
                            + self.map_size // 2,
                            0,
                            self.map_size - 1,
                        ),
                        dtype=int,
                    )
                    car_width_px = int(2.0 * self.resolution_inv)
                    velocity = self.path[..., 3]
                    velocity_norm = (velocity - np.min(velocity)) / (
                        np.max(velocity) - np.min(velocity)
                    )
                    velocity_color = np.array(
                        np.clip((velocity_norm * 255), 0, 255), dtype=int
                    )
                    for i in range(len(path_X) - 1):
                        cv2.line(
                            costmap,
                            (path_X[i], path_Y[i]),
                            (path_X[i + 1], path_Y[i + 1]),
                            (0, int(velocity_color[i]), 0),
                            car_width_px,
                        )
                # Draw states
                if self.states is not None:
                    if len(self.costmap.shape) < 3:
                        print_states = self.states
                        x = print_states[:, :, :, 0].flatten()
                        y = print_states[:, :, :, 1].flatten()
                        X = np.clip(
                            np.array(
                                (x * self.resolution_inv) + self.map_size // 2,
                                dtype=np.int32,
                            ),
                            0,
                            self.map_size - 1,
                        )
                        Y = np.clip(
                            np.array(
                                (y * self.resolution_inv) + self.map_size // 2,
                                dtype=np.int32,
                            ),
                            0,
                            self.map_size - 1,
                        )
                        costmap[Y, X] = 0
                    else:
                        print_states = self.states
                        x = print_states[:, :, :, 0].flatten()
                        y = print_states[:, :, :, 1].flatten()
                        X = np.clip(
                            np.array(
                                (x * self.resolution_inv) + self.map_size // 2,
                                dtype=np.int32,
                            ),
                            0,
                            self.map_size - 1,
                        )
                        Y = np.clip(
                            np.array(
                                (y * self.resolution_inv) + self.map_size // 2,
                                dtype=np.int32,
                            ),
                            0,
                            self.map_size - 1,
                        )
                        costmap[Y, X] = np.array([0, 0, 0])
                # resize and flip the costmap for visualization
                costmap = cv2.resize(costmap, (500, 500), interpolation=cv2.INTER_AREA)
                costmap = cv2.flip(costmap, 0)
                if self.hysteresis == -1:
                    var = "HA*M"
                else:
                    var = f"IGHA*-{self.hysteresis}"
                cv2.putText(
                    costmap,
                    f"Var: {var}",
                    (10, 40),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1,
                    (0, 255, 0),
                    1,
                )
                if self.expansion_counter == 1:
                    self.expansion_counter = "FAILED"
                cv2.putText(
                    costmap,
                    f"Exp: {self.expansion_counter}",
                    (10, 70),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1,
                    (0, 255, 0),
                    1,
                )
                cv2.imshow("map", costmap)
                cv2.waitKey(1)
                # Reset for next update
                self.path = None
                self.states = None
                self.costmap = None
                self.elevation_map = None
            else:
                time.sleep(0.02)
