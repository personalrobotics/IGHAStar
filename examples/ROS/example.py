#!/usr/bin/env python3
"""
ROS wrapper for IGHA* planner (single-class version).
- Can be used live (with ROS subscribers) or offline (by calling callbacks with rosbag messages).
- Maintains internal state and publishes planned paths.
- Clean, modern, and easy to adapt for both online and offline use.
"""
import rospy
import numpy as np
import torch
import yaml
from nav_msgs.msg import Path, Odometry
from mavros_msgs.msg import WaypointList
from grid_map_msgs.msg import GridMap
from visualization_msgs.msg import MarkerArray
from utils import *
import cv2
import time
import os

class PlannerNode:
    """
    ROS node wrapper for IGHAStar planner.
    Handles all ROS communication, map/state/waypoint management, and planning logic.
    Can be used live or with rosbag replay by calling the callbacks directly.
    """
    def __init__(self, config):
        # --- Planner config ---
        planner_cfg = config["Planner_config"]
        self.map_res = planner_cfg["experiment_info_default"]["node_info"]["map_res"]
        self.expansion_limit = planner_cfg["experiment_info_default"]["max_expansions"]
        self.hysteresis = planner_cfg["experiment_info_default"]["hysteresis"]
        self.planner = create_planner(planner_cfg)
        # --- Costmap parameters ---
        self.blur_kernel = config["blur_kernel"]
        self.costmap_cosine_thresh = np.cos(np.radians(config["lethal_slope"]))
        self.wp_radius = config["wp_radius"]
        # --- IO variables ---
        self.state = np.zeros(9)
        self.local_waypoints = None
        self.global_pos = None
        self.map_init = False
        self.height_index = None
        self.bitmap = None
        self.offset = None
        self.map_center = None
        self.map_size_px = None
        # --- ROS publishers ---
        topics = config["topics"]
        self.path_pub = rospy.Publisher(topics["path"], Path, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher(topics["goal_marker"], MarkerArray, queue_size=1)
        self.diagnostics_pub = rospy.Publisher(topics["diagnostics"], DiagnosticArray, queue_size=1)
        # --- ROS subscribers ---
        rospy.Subscriber(topics["elevation_map"], GridMap, self.map_callback, queue_size=1)
        rospy.Subscriber(topics["odom"], Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(topics["waypoints"], WaypointList, self.waypoint_callback, queue_size=1)
        rospy.Subscriber(topics["global_position"], NavSatFix, self.global_pos_callback, queue_size=1)
        self.plan_loop()

    def map_callback(self, grid_map):
        self.grid_map = grid_map
        if self.height_index is None or self.layers is None:
            self.layers = self.grid_map.layers
            self.height_index = self.layers.index("elevation")
        cent = self.grid_map.info.pose.position
        self.map_center = np.array([cent.x, cent.y, cent.z])
        matrix = self.grid_map.data[self.height_index]
        self.heightmap = map_from_gridmap(matrix)
        if np.any(np.isnan(self.heightmap)):
            self.map_init = False
        else:
            self.bitmap, self.offset = process_grid_map(self.heightmap, lethalmap=None, map_res=self.map_res, blur_kernel=self.blur_kernel, costmap_cosine_thresh=self.costmap_cosine_thresh)
            self.map_init = True

    def plan(self, start_state, goal_, stop=False):
        if self.bitmap is None:
            return None, False, 0, 0.0, goal_
        bitmap = torch.clone(self.bitmap)
        start, goal = start_goal_logic(bitmap, self.map_res, start_state, goal_, self.map_center, self.offset, stop=stop)
        now = time.perf_counter()
        success = self.planner.search(start, goal, bitmap, self.expansion_limit, self.hysteresis, True)
        time_taken = time.perf_counter() - now
        avg_successor_time, avg_goal_check_time, avg_overhead_time, avg_g_update_time, switches, max_level_profile, Q_v_size, expansion_counter, expansion_list= self.planner.get_profiler_info()
        output_goal = goal[:2] - (self.offset - self.map_center[:2])
        if success:
            path = self.planner.get_best_path().numpy()
            path = np.flip(path, axis=0)
            path[..., :2] -= self.offset
            path[..., :2] += self.map_center[:2]
            return path, True, expansion_counter, time_taken, output_goal
        else:
            return None, False, expansion_counter, time_taken, output_goal

    def odom_callback(self, msg):
        """Update local position from Odometry message."""
        self.state = obtain_state(msg, self.state)
    
    def global_pos_callback(self, msg):
        self.global_pos = msg

    def waypoint_callback(self, msg):
        """Update local waypoints from WaypointList message."""
        print("Got waypoints")
        self.waypoint_list = msg.waypoints
        while self.global_pos is None or self.state is None:
            time.sleep(1)
        gps_origin = calcposLLH(self.global_pos.latitude, self.global_pos.longitude, -self.state[0], -self.state[1])
        self.local_waypoints = get_local_frame_waypoints(self.waypoint_list, gps_origin)

    def plan_loop(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            rate.sleep()
            if self.local_waypoints is not None and self.state is not None and self.bitmap is not None and len(self.local_waypoints):
                self.local_waypoints = np.array(self.local_waypoints)
                plan_state = np.array([self.state[0], self.state[1], self.state[5], np.linalg.norm(self.state[6:8])])
                goal = self.local_waypoints[0]
                path, success, expansions, time_taken, output_goal = self.plan(plan_state, goal, stop=len(self.local_waypoints) == 0)
                expansions_per_second = max(expansions/time_taken, 1000)
                self.expansion_limit = int(expansions_per_second * 0.5)
                if success:
                    publish_path(path, self.path_pub)
                    publish_goal(output_goal, self.marker_pub)
                    # Visualize map with path

                diagnostic_publisher(success, expansions, time_taken, self.hysteresis, self.diagnostics_pub)
                display_map = visualize_map_with_path(self.bitmap[..., 0].numpy(), self.bitmap[..., 1].numpy(), path, output_goal, plan_state, self.wp_radius, self.map_center, 480, 7.5/self.map_res)
                cv2.imshow("planner_vis", display_map)
                cv2.waitKey(1)
                # reduce length of local waypoints:
                dist = np.linalg.norm(self.local_waypoints[0][:2] - plan_state[:2])
                if dist < self.wp_radius:
                    if len(self.local_waypoints) > 1:
                        self.local_waypoints = self.local_waypoints[1:]
                

if __name__ == "__main__":
    rospy.init_node("ighastar_planner_node")
    config_name = "ros_example.yml"
    config_path = os.path.join(os.path.dirname(__file__), "Configs", config_name)
    with open(config_path) as f:
        Config = yaml.safe_load(f)
    node = PlannerNode(Config)
    rospy.spin()