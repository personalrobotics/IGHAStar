#!/usr/bin/env python3
import numpy as np
import time
import yaml
import torch
from torch.utils.cpp_extension import load
import cv2
import pathlib
import sys
import os
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / 'scripts'))
sys.path.append(str(BASE_DIR / 'src'))
from plotting import show_map, plot_car
from utils import get_map
import matplotlib.pyplot as plt

def check_cuda_available():
    """Check if CUDA is available for compilation and execution"""
    try:
        # Check if CUDA is available in PyTorch
        if not torch.cuda.is_available():
            print("PyTorch CUDA not available")
            return False
        
        # Check if we have CUDA devices
        if torch.cuda.device_count() == 0:
            print("No CUDA devices found")
            return False
        
        # Check if CUDA toolkit is available by trying to import torch.utils.cpp_extension
        # and checking if CUDA compilation would work
        try:
            # This is a simple check - if we can import and CUDA is available in PyTorch,
            # we assume CUDA toolkit is available
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            return True
        except Exception as e:
            print(f"CUDA toolkit check failed: {e}")
            return False
            
    except Exception as e:
        print(f"CUDA availability check failed: {e}")
        return False

def create_planner(configs):
    env_name = configs["experiment_info_default"]["node_info"]["node_type"]
    # Check CUDA availability
    cuda_available = check_cuda_available()

    if not cuda_available:
        print("CUDA not available, using CPU versions of environments")
        # Use CPU versions
        env_macro = {
            'simple': '-DUSE_SIMPLE_ENV',
            'kinematic': '-DUSE_KINEMATIC_CPU_ENV',
            'kinodynamic': '-DUSE_KINODYNAMIC_CPU_ENV',
        }[env_name]
    else:
        print("CUDA available, using GPU versions of environments")
        # Use CUDA versions
        env_macro = {
            'simple': '-DUSE_SIMPLE_ENV',
            'kinematic': '-DUSE_KINEMATIC_ENV',
            'kinodynamic': '-DUSE_KINODYNAMIC_ENV',
        }[env_name]
    
    cpp_path = f'{BASE_DIR}/src/ighastar.cpp'
    header_path = f'{BASE_DIR}/src'

    if env_name != "simple":
        if cuda_available:
            # Use CUDA version
            cuda_path = f'{BASE_DIR}/src/{env_name}.cu'
            kernel = load(
                name="ighastar",
                sources=[cpp_path, cuda_path],
                extra_include_paths=[header_path],
                extra_cflags=['-std=c++17', '-O3', env_macro],
                extra_cuda_cflags=['-O3'],
                verbose=True,
            )
        else:
            # Use CPU version - compile with CPU header and .cpp file included
            cpu_cpp_path = f'{BASE_DIR}/src/{env_name}_cpu.cpp'
            kernel = load(
                name="ighastar",
                sources=[cpp_path, cpu_cpp_path],
                extra_include_paths=[header_path],
                extra_cflags=['-std=c++17', '-O3', env_macro],
                verbose=True,
            )
    else:
        # Simple environment (already CPU-based)
        kernel = load(
            name="ighastar",
            sources=[cpp_path],
            extra_include_paths=[header_path],
            extra_cflags=['-std=c++17', '-O3', env_macro],
            verbose=True,
        )
    
    planner = kernel.IGHAStar(configs, False)
    return planner


def main(yaml_path="", test_case=None):
    assert yaml_path, "Please provide a valid YAML configuration file path."
    print("Loading config from:", yaml_path)
    with open(yaml_path, 'r') as file:
        configs = yaml.safe_load(file)
    assert configs, "Failed to load configurations from the YAML file."
    print("Config loaded successfully")

    # ============= problem definition parameters =============
    map_info = configs["map"]
    map_dir = map_info["dir"]
    map_name = map_info["name"]
    map_size = map_info["size"]
    start = torch.tensor(map_info["start"], dtype=torch.float32)
    goal = torch.tensor(map_info["goal"], dtype=torch.float32)
    experiment_info = configs["experiment_info_default"]
    node_info = experiment_info["node_info"]
    node_type = node_info["node_type"]
    map_res = node_info["map_res"]
    epsilon = experiment_info["epsilon"][0]
    print(f"Node type: {node_type}")
    print(f"Map: {map_name}")
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    
    print("Loading bitmap...")
    # Fix the map directory path to be relative to the examples directory
    if not os.path.isabs(map_dir):
        map_dir = os.path.join(BASE_DIR, "examples", map_dir)
    print(f"Map directory: {map_dir}")
    bitmap = get_map(map_name, map_dir=map_dir, map_size=map_size, node_info=node_info)
    print(f"Bitmap loaded, shape: {bitmap.shape}")

    expansion_limit = experiment_info["max_expansions"]
    hysteresis = experiment_info["hysteresis"]
    print(f"Expansion limit: {expansion_limit}")
    print(f"Hysteresis: {hysteresis}")

    print("Creating planner...")
    planner = create_planner(configs)
    print("Planner created successfully")
    
    print("Starting search...")
    now = time.perf_counter()
    success = planner.search(start, goal, bitmap, expansion_limit, hysteresis, False)
    end = time.perf_counter()
    print(f"Search completed, success: {success}")
    
    print("Getting profiler info...")
    avg_successor_time, avg_goal_check_time, avg_overhead_time, avg_g_update_time, switches, max_level_profile, Q_v_size, expansion_counter, expansion_list= planner.get_profiler_info()
    print("Profiler info retrieved")
    
    dt = end - now
    # print all the stats:
    print("Search statistics:")
    print(f"Search took {dt:.4f} seconds")
    print(f"Average successor time: {avg_successor_time:.4f} microseconds")
    print(f"Average goal check time: {avg_goal_check_time:.4f} microseconds")
    print(f"Average overhead time: {avg_overhead_time:.4f} microseconds")
    print(f"Average G update time: {avg_g_update_time:.4f} microseconds")
    print(f"Switches: {switches}")
    print(f"Max level profile: {max_level_profile}")
    print(f"Q_v size: {Q_v_size}")
    print(f"Expansion counter: {expansion_counter}")
    print(f"Expansion list: {expansion_list}")
    
    if success:
        print("Getting best path...")
        path = planner.get_best_path().numpy()
        show_map(plt, bitmap, node_type)
        plt.plot(path[:, 0]/map_res, path[:, 1]/map_res, color='black', linewidth=2)
        if node_type == "kinodynamic" or node_type == "kinematic":
            plot_car(plt, start[0].item()/map_res, start[1].item()/map_res, start[2].item(), color='black')
            for i in range(len(path)):
                if i % node_info["timesteps"] == 0:
                    plot_car(plt, path[i, 0]/map_res, path[i, 1]/map_res, path[i, 2], color='black')
        else:
            # simple environment, just plot the path and place circles at each vertex:
            for i in range(len(path)):
                plt.scatter(path[i, 0]/map_res, path[i, 1]/map_res, color='black', s=10)
        goal_circle = plt.Circle((goal[0]/map_res, goal[1]/map_res),facecolor='gray', edgecolor='black', radius=epsilon/map_res, alpha=0.75, zorder=100)
        plt.gca().add_patch(goal_circle)
        # vertically flip the plot
        plt.gca().invert_yaxis()
        plt.show()
    else:
        print("no path found")
    
if __name__ == "__main__":
    yaml_path = BASE_DIR / 'examples' /'Configs' / 'kinematic_example.yml'
    main(yaml_path=str(yaml_path))