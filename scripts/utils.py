import numpy as np
import matplotlib.pyplot as plt
import sys
import pathlib
import cv2
sys.path.append(str(pathlib.Path(__file__).parent.parent))
import os
import random
import pickle
import torch
import scipy.stats as st
from scipy.stats import ttest_ind
from copy import deepcopy
import psutil

def convert_tuples_to_lists(data):
    """
    Recursively convert all tuples in a data structure to lists.
    Args:
        data (any): The input data, which could be a dictionary, list, or tuple.
    Returns:
        any: The data with all tuples converted to lists.
    """
    if isinstance(data, tuple):
        return list(data)
    elif isinstance(data, list):
        return [convert_tuples_to_lists(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_tuples_to_lists(value) for key, value in data.items()}
    return data

def write_configs_to_pickle(configs, pickle_file):
    """
    Writes start-goal configurations to a pickle file.
    Args:
        configs (dict): Dictionary of configurations in the format:
                        {map_name: {"configs": [[start, goal], ...]}}
        pickle_file (str): Path to the pickle file to write to.
    """
    # Use pickle to save the configs directly
    with open(pickle_file, 'wb') as file:
        pickle.dump(configs, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Configurations written to {pickle_file} in pickle format.")

def read_configs_from_pickle(pickle_file):
    """
    Reads configurations from a pickle file.
    Args:
        pickle_file (str): Path to the pickle file to read from.
    Returns:
        dict: Dictionary of configurations in the format:
              {map_name: {"configs": [[start, goal], ...]}}
    """
    with open(pickle_file, 'rb') as file:
        configs = pickle.load(file)
    print(f"Configurations read from {pickle_file}.")
    return configs

def save_to_pickle(data, file_path):
    """
    Saves a dictionary to a pickle file.

    Args:
        data (dict): Dictionary to save.
        file_path (str): Path to the pickle file.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def load_from_pickle(file_path):
    """
    Loads a dictionary from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        dict: Loaded dictionary.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def sample_node(map, valid_indices, node_type):
    if node_type == "simple":
        return sample_simple(map, valid_indices)
    elif node_type == "kinematic":
        return sample_kinematic(map, valid_indices)
    elif node_type == "kinodynamic":
        return sample_kinodynamic(map, valid_indices)
    else:
        raise ValueError(f"Invalid node type: {node_type}")

def sample_simple(map, valid_indices):
    """
    Samples a position near a given corner within a box.
    Args:
        corner (tuple): The corner coordinates (x, y).
        box_size (int): The size of the sampling box (e.g., 50x50).
        map_shape (tuple): The shape of the map (height, width).
    Returns:
        tuple: A random position near the corner.
    """
    # Randomly select indices
    sample_indices = np.random.choice(len(valid_indices[0]), 1, replace=False)
    x,y = valid_indices[1][sample_indices[0]], valid_indices[0][sample_indices[0]]
    # yaw should be facing toward the center of the map
    return x,y

def sample_kinematic(map, valid_indices, map_res=0.1):
    """
    Samples a position near a given corner within a box.
    Args:
        corner (tuple): The corner coordinates (x, y).
        box_size (int): The size of the sampling box (e.g., 50x50).
        map_shape (tuple): The shape of the map (height, width).
    Returns:
        tuple: A random position near the corner.
    """
    # Randomly select indices
    sample_indices = np.random.choice(len(valid_indices[0]), 1, replace=False)
    x,y = valid_indices[1][sample_indices[0]]*map_res, valid_indices[0][sample_indices[0]]*map_res
    # yaw should be facing toward the center of the map
    yaw = random.uniform(-np.pi, np.pi)
    return x,y,yaw

def sample_kinodynamic(map, valid_indices, map_res=0.1):
    """
    Samples a position near a given corner within a box.
    Args:
        corner (tuple): The corner coordinates (x, y).
        box_size (int): The size of the sampling box (e.g., 50x50).
        map_shape (tuple): The shape of the map (height, width).
    Returns:
        tuple: A random position near the corner.
    """
    # Randomly select indices
    sample_indices = np.random.choice(len(valid_indices[0]), 1, replace=False)
    x,y = valid_indices[1][sample_indices[0]]*map_res, valid_indices[0][sample_indices[0]]*map_res
    # yaw should be facing toward the center of the map
    yaw = random.uniform(-np.pi, np.pi)
    vel = random.uniform(3.0, 5.0)
    return x,y,yaw, vel, 0

def conf(data, confidence=0.95):
    CI = st.norm.interval(confidence, loc=np.mean(data), scale=st.sem(data))
    return (CI[1] - CI[0])/2

def bernoulli_confidence_test(binary_outcomes, confidence=0.95):
    """
    Perform a Bernoulli confidence interval test to see if A beats B significantly.

    Parameters:
        binary_outcomes (list): Binary outcomes (1 if A beats B, 0 otherwise).
        confidence (float): Confidence level (default is 95%).

    Returns:
        p_hat (float): Estimated probability that A beats B.
        delta (float): Confidence interval margin.
        reject_null (bool): True if we reject the null hypothesis, otherwise False.
    """
    n = len(binary_outcomes)
    if n == 0:
        raise ValueError("No data available for testing.")
    
    # Estimate p_hat
    p_hat = np.mean(binary_outcomes)

    # Compute confidence interval delta
    z = st.norm.ppf(1 - (1 - confidence) / 2)  # Z-score for confidence level
    delta = z * np.sqrt(p_hat * (1 - p_hat) / n)

    # Null hypothesis: \hat{p} - delta <= 0.5
    reject_null = (p_hat - delta > 0.5)
    # clip delta so that p_hat + delta is within 1.0 and 0.0:
    delta = min(delta, 1.0 - p_hat)
    delta = min(delta, p_hat)
    return p_hat, delta, reject_null

def generate_binary_comparisons(level, algorithm_names, expansion_results):
    num_algorithms = len(algorithm_names)
    binary_matrix = np.zeros((num_algorithms, num_algorithms), dtype=object)  # Store binary results

    for i, algo_a in enumerate(algorithm_names):
        for j, algo_b in enumerate(algorithm_names):
            # if i <= j:
            #     continue  # Only fill upper triangle
            # Get data for both algorithms
            data_a = expansion_results[i][level]
            data_b = expansion_results[j][level]
            # Ensure both have data
            if len(data_a) > 0 and len(data_b) > 0:
                # Binary comparisons: 1 if A < B, 0 otherwise
                min_length = min(len(data_a), len(data_b))
                binary_array = [1 if a < b else 0 for a, b in zip(data_a, data_b)]
                binary_matrix[i, j] = np.array(binary_array)
            else:
                binary_matrix[i, j] = None  # No valid data
    return binary_matrix

def generate_pvalue_matrix(binary_matrix, algorithm_names):
    num_algorithms = len(algorithm_names)
    phat_matrix = np.zeros((num_algorithms, num_algorithms))
    delta_matrix=  np.zeros((num_algorithms, num_algorithms))
    reject_null_matrix = np.zeros((num_algorithms, num_algorithms))

    for i in range(num_algorithms):
        for j in range(num_algorithms):
            # if i <= j or binary_matrix[i, j] is None:
            #     pvalue_matrix[i, j] = np.nan  # Only upper triangle
            #     continue

            # Perform a one-sample t-test on the binary data
            binary_array = binary_matrix[i, j]
            phat_matrix[i, j], delta_matrix[i, j], reject_null_matrix[i, j] = bernoulli_confidence_test(binary_array)

    return phat_matrix, delta_matrix, reject_null_matrix

def compute_ratio_and_ci(data_a, data_b, num_bootstrap=1000, confidence=0.95):
    """
    Compute the ratio of expansions (A/B) and its confidence interval using bootstrapping.
    
    Parameters:
        data_a (list): Expansions for algorithm A.
        data_b (list): Expansions for algorithm B (baseline).
        num_bootstrap (int): Number of bootstrap samples.
        confidence (float): Confidence level for the interval (default is 95%).
    
    Returns:
        mean_ratio (float): Mean of the A/B ratio.
        ci (tuple): Confidence interval (lower, upper).
    """
    ratios = np.array(data_a) / np.array(data_b)
    # bootstrap_samples = [
    #     np.mean(np.random.choice(ratios, size=len(ratios), replace=True)) for _ in range(num_bootstrap)
    # ]
    # lower = np.percentile(bootstrap_samples, (1 - confidence) / 2 * 100)
    # upper = np.percentile(bootstrap_samples, (1 + confidence) / 2 * 100)
    CI = conf(ratios)
    # clip CI so that p_hat + delta is within 1.0 and 0.0:
    CI = min(CI, 1.0 - np.mean(ratios))
    CI = min(CI, np.mean(ratios))
    lower = np.mean(ratios) - CI
    upper = np.mean(ratios) + CI
    return np.mean(ratios), (lower, upper)

def compare_levels_ratios(ratios_by_level):
    """
    Check if the difference between the ratios at consecutive levels is significant.
    
    Parameters:
        ratios_by_level (dict): Dictionary of ratios indexed by level.
    
    Returns:
        significance_results (dict): Dictionary indicating if differences between levels are significant.
    """
    significance_results = {}
    levels = sorted(ratios_by_level.keys())
    
    for i in range(len(levels) - 1):
        level_a = levels[i]
        level_b = levels[i + 1]
        
        # Two-sample t-test between consecutive levels
        ratios_a = ratios_by_level[level_a]
        ratios_b = ratios_by_level[level_b]
        t_stat, p_val = ttest_ind(ratios_a, ratios_b, equal_var=False)
        
        significance_results[(level_a, level_b)] = p_val  # Significant if p < 0.05
    
    return significance_results

def get_path(path):
    out = []
    for v in path:
        out.append(v.pose)
    return np.array(out)

def combine_cost_maps():
    folder_path = "PathPlanning/HybridAStar/off_road_maps"
    obst_map_path = "PathPlanning/HybridAStar/street-png"
    output_path = "PathPlanning/HybridAStar/off_road_maps_obstacles"
    resolution = 512
    maps = []
    # ["Berlin_2_512.png", "Boston_2_512.png", "Shanghai_0_512.png", "Sydney_2_512.png"]
    map_list = ["map_0.png", "map_1.png", "map_2.png", "map_3.png", "map_4.png", "map_5.png", "map_6.png", "map_7.png"]
    for name in map_list:
        map_path = os.path.join(obst_map_path, name)
        map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        map = cv2.resize(map, (resolution, resolution))
        map = np.array(map, dtype=np.float32)
        maps.append(map)
    count = 0
    for index, file in enumerate(os.listdir(folder_path)):
        if file.endswith(".npy"):
            if "height" in file:
                continue
            map_path = os.path.join(folder_path, file.strip(".npy"))
            elevation_map = map_path + "_height.npy"
            costmap = map_path + ".npy"
            costmap = np.load(costmap)
            elevation_map = np.load(elevation_map)
            elevation_map -= np.min(elevation_map)
            costmap = np.clip(costmap, 0, 255)
            costmap = compute_surface_normals(elevation_map, 45)
            costmap = (maps[index%8] + costmap)/2
            costmap[costmap >= 128] = 255.0
            costmap[costmap < 128] = 0.0
            print(costmap.max(), costmap.min())
            np.save(os.path.join(output_path, file), costmap)
            print("Saved: ", output_path, file)
            # compute elevation map normals:
            # plt.imshow(costmap, cmap='gray')
            # plt.imshow(elevation_map, cmap='jet', alpha=0.5)
            # plt.show()
            # np.save(os.path.join(output_path, file.strip(".npy") + "_height.npy"), elevation_map)
            count += 1
    print(count)

def compute_surface_normals(elevation, threshold_deg):
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

def get_map(map_name, map_dir="", map_size=[512, 512], node_info = None):
    assert map_dir!="", "Map directory must be specified."
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

def get_costmap(bitmap, node="simple", bloat_size = 30):
    if node == "simple":
        return bitmap
    elif node == "kinematic" or node == "kinodynamic":
        costmap = torch.clone(bitmap[..., 0])
        bloated_costmap = torch.clone(costmap).numpy()
        # Create a circular kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*bloat_size + 1, 2*bloat_size + 1))
        # Apply dilation
        bloated_costmap = cv2.dilate(bloated_costmap, kernel)
        bloated_costmap = (bloated_costmap > 250) * 255.0
        return bloated_costmap

def get_files(folder_path, experiment_info):
    node_type = experiment_info["node_info"]["node_type"]
    map_size  = experiment_info["map_size"]
    if node_type == "kinematic" or node_type == "simple":
        return [file for file in os.listdir(folder_path) if file.endswith(f"{map_size[0]}.png")]
    elif node_type == "kinodynamic":
        files = os.listdir(folder_path)
        files = [file for file in os.listdir(folder_path) if "_height" not in file and not file.endswith("256.png")]
        return files

def generate_node_config(node_info):
    node_type = node_info["node_type"]
    if node_type == "simple":
        return node_info
    elif node_type == "kinematic":
        info = {}
        info["node_type"] = node_info["node_type"]
        info["length"] = node_info["length"]
        info["width"] = node_info["width"]
        info["map_res"] = node_info["map_res"]
        info["dt"] = node_info["dt"]
        info["timesteps"] = node_info["timesteps"]
        info["step_size"] = node_info["step_size"]
        info["del_theta"] = node_info["del_theta"]
        # info["hash_theta"] = node_info["hash_theta"]
        if "hash_theta" in node_info:
            info["hash_theta"] = node_info["hash_theta"]
        else:
            info["hash_theta"] = False
        throttle_list = np.array(node_info["throttle_list"])
        steering_angles = np.array(node_info["steering_list"])/57.3
        kinematic_controls = []
        for throttle in throttle_list:
            for angle in steering_angles:
                kinematic_controls.append((angle, throttle))
        kinematic_controls = np.array(kinematic_controls)
        kinematic_controls = torch.Tensor(kinematic_controls).to(dtype=torch.float32)
        kinematic_K = np.int32(len(kinematic_controls))
        info["controls"] = kinematic_controls
        info["K"] = kinematic_K
        max_curvature = np.tan(steering_angles.max())/node_info["length"]
        info["max_curvature"] = max_curvature
        node_info = deepcopy(info)
        return node_info
    elif node_type == "kinodynamic":
        info = {}
        info["node_type"] = node_info["node_type"]
        info["length"] = node_info["length"]
        info["width"] = node_info["width"]
        info["map_res"] = node_info["map_res"]
        info["dt"] = node_info["dt"]
        info["timesteps"] = node_info["timesteps"]
        info["step_size"] = node_info["step_size"]
        info["del_theta"] = node_info["del_theta"]
        info["del_vel"] = node_info["del_vel"]
        info["RI"] = node_info["RI"]
        info["max_vert"] = node_info["max_vert"]
        info["min_vel"] = node_info["min_vel"]
        info["max_vel"] = node_info["max_vel"]
        info["max_theta"] = node_info["max_theta"]/57.3
        if "hash_theta" in node_info:
            info["hash_theta"] = node_info["hash_theta"]
        else:
            info["hash_theta"] = False

        throttle_list = np.array(node_info["throttle_list"])
        steering_angles = np.array(node_info["steering_list"])/57.3
        controls = []
        for throttle in throttle_list:
            for angle in steering_angles:
                controls.append((angle, throttle))
        controls = np.array(controls)
        controls = torch.Tensor(controls).to(dtype=torch.float32)
        K = np.int32(len(controls))
        info["controls"] = controls
        info["K"] = K
        max_curvature = np.tan(steering_angles.max())/node_info["length"]
        info["max_curvature"] = max_curvature
        node_info = deepcopy(info)
        return node_info
    else:
        raise ValueError(f"Invalid node type: {node_type}")

def pin_to_cpu(core_id):
    """Pins the process to a specific CPU core."""
    pid = os.getpid()  # Get process ID
    os.sched_setaffinity(pid, {core_id})  # Assign process to specific CPU core

def get_least_utilized_core():
    """Finds the CPU core with the lowest usage."""
    cpu_usages = psutil.cpu_percent(interval=0.1, percpu=True)  # Get per-core usage
    return min(range(len(cpu_usages)), key=lambda i: cpu_usages[i])  # Return the least used core index

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def rot_mat_2d(angle):
    """
    Create 2D rotation matrix from an angle

    Parameters
    ----------
    angle :

    Returns
    -------
    A 2D rotation matrix

    Examples
    --------
    >>> angle_mod(-4.0)


    """
    return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]


def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)

    Parameters
    ----------
    x : float or array_like
        A angle or an array of angles. This array is flattened for
        the calculation. When an angle is provided, a float angle is returned.
    zero_2_2pi : bool, optional
        Change angle modulo range to [0, 2pi)
        Default is False.
    degree : bool, optional
        If True, then the given angles are assumed to be in degrees.
        Default is False.

    Returns
    -------
    ret : float or ndarray
        an angle or an array of modulated angle.

    Examples
    --------
    >>> angle_mod(-4.0)
    2.28318531

    >>> angle_mod([-4.0])
    np.array(2.28318531)

    >>> angle_mod([-150.0, 190.0, 350], degree=True)
    array([-150., -170.,  -10.])

    >>> angle_mod(-60.0, zero_2_2pi=True, degree=True)
    array([300.])

    """
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle
