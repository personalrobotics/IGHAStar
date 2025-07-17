#!/usr/bin/env python3
import time
import yaml
import torch
from torch.utils.cpp_extension import load
import os
import argparse
from utils import *
from ighastar.scripts.common_utils import create_planner, BASE_DIR
from typing import Optional


def main(yaml_path: str = "", test_case: Optional[str] = None) -> None:
    assert yaml_path, "Please provide a valid YAML configuration file path."
    print("Loading config from:", yaml_path)
    with open(yaml_path, "r") as file:
        configs = yaml.safe_load(file)
    assert configs, "Failed to load configurations from the YAML file."
    print("Config loaded successfully")

    # ============= problem definition parameters =============
    map_info = configs["map"]
    map_dir = map_info["dir"]
    map_name = map_info["name"]
    map_size = map_info["size"]

    # Handle test case selection
    if test_case and "test_cases" in map_info and test_case in map_info["test_cases"]:
        test_config = map_info["test_cases"][test_case]
        start = torch.tensor(test_config["start"], dtype=torch.float32)
        goal = torch.tensor(test_config["goal"], dtype=torch.float32)
        print(f"Using test case: {test_case}")
    else:
        start = torch.tensor(map_info["start"], dtype=torch.float32)
        goal = torch.tensor(map_info["goal"], dtype=torch.float32)
        if test_case:
            print(f"Test case '{test_case}' not found, using default start/goal")
        else:
            print("Using default start/goal")
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
        map_dir = os.path.join(BASE_DIR.parent, "examples/standalone", map_dir)
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
    (
        avg_successor_time,
        avg_goal_check_time,
        avg_overhead_time,
        avg_g_update_time,
        switches,
        max_level_profile,
        Q_v_size,
        expansion_counter,
        expansion_list,
    ) = planner.get_profiler_info()
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
        print("✓ Optimal path found!")
        print("Getting best path...")
        path = planner.get_best_path().numpy()

        # Create visualization
        show_map(plt, bitmap, node_type)

        # Plot path
        plt.plot(
            path[:, 0] / map_res,
            path[:, 1] / map_res,
            color="blue",
            linewidth=3,
            label="Path",
        )

        plt.scatter(
            goal[0].item() / map_res,
            goal[1].item() / map_res,
            color="red",
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=2,
            label="Goal",
            zorder=10,
        )

        if node_type == "kinodynamic" or node_type == "kinematic":
            # Plot car orientations along the path
            plot_car(
                plt,
                start[0].item() / map_res,
                start[1].item() / map_res,
                start[2].item(),
                color="green",
                label="Start",
            )
            for i in range(len(path) - 1):
                if i % node_info["timesteps"] == 0:
                    plot_car(
                        plt,
                        path[i, 0] / map_res,
                        path[i, 1] / map_res,
                        path[i, 2],
                        color="blue",
                    )
            # plot_car(plt, path[-1, 0]/map_res, path[-1, 1]/map_res, path[-1, 2], color='blue')
        else:
            # Simple environment - plot path vertices
            for i in range(len(path)):
                plt.scatter(
                    path[i, 0] / map_res,
                    path[i, 1] / map_res,
                    color="blue",
                    s=20,
                    alpha=0.6,
                )

        # Add goal region circle
        goal_circle = plt.Circle(
            (goal[0] / map_res, goal[1] / map_res),
            facecolor="red",
            edgecolor="black",
            radius=epsilon / map_res,
            alpha=0.3,
            zorder=5,
            label="Goal Region",
        )
        plt.gca().add_patch(goal_circle)

        # Add legend and labels
        plt.legend(loc="upper right")
        plt.title(f"IGHAStar Path Planning - {node_type.capitalize()} Environment")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")

        # Vertically flip the plot (image coordinates)
        plt.gca().invert_yaxis()

        print("Displaying visualization...")
        plt.show()
        print("✓ Visualization complete!")
    else:
        print("✗ No path found - the goal may be unreachable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IGHAStar Path Planning Example")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=str("Configs/kinematic_example.yml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--test-case", type=str, default="case1", help="Test case identifier (optional)"
    )
    # we assume the config is from examples/standalone folder:
    args = parser.parse_args()
    yaml_path = os.path.join(BASE_DIR.parent, "examples", "standalone", args.config)
    main(yaml_path=yaml_path, test_case=args.test_case)
