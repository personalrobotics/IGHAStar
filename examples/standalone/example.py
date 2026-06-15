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


def main(
    yaml_path: str = "",
    test_case: Optional[str] = None,
    bidirectional: bool = False,
    print_controls: bool = False,
) -> None:
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

    default_expansion_limit = experiment_info["max_expansions"]
    hysteresis = experiment_info["hysteresis"]
    if bidirectional:
        if node_type == "kinodynamic" or node_type == "kinematic":
            expansion_limit = 1000  # Per-thread limit for bidirectional search
        else:
            expansion_limit = 5000  # Per-thread limit for bidirectional search
        print(
            f"\033[92mExpansion limit: {expansion_limit} (default unidirectional: {default_expansion_limit})\033[0m"
        )
    else:
        expansion_limit = default_expansion_limit
        print(f"Expansion limit: {expansion_limit}")
    print(f"Hysteresis: {hysteresis}")
    print(f"Bidirectional: {bidirectional}")

    print("Creating planner...")
    planner = create_planner(configs, bidirectional=bidirectional)
    planner_type = "BiIGHAStar" if bidirectional else "IGHAStar"
    print(f"{planner_type} planner created successfully")

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
        cost_exp_list,
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
    # print the expansions to first solution, and the cost of the best solution:
    # first solution is when cost_exp_list is less than 1e4:
    cost_exp_arr = np.array(cost_exp_list)
    solution_indices = np.where(cost_exp_arr < 1e4)[0]
    if len(solution_indices) > 0:
        first_solution_expansion = solution_indices[0]
        first_solution_cost = cost_exp_arr[first_solution_expansion]
        print(f"First solution expansion: {first_solution_expansion}")
        print(f"First solution cost: {first_solution_cost}")
        # best solution is the minimum cost in the cost_exp_list:
        best_solution_cost = np.min(cost_exp_arr)
        print(f"Best solution cost: {best_solution_cost}")
    else:
        print("No solution found")

    # print(f"Cost list (at each expansion): {cost_exp_list}")

    if success:
        print("✓ Optimal path found!")
        print("Getting best path...")
        if print_controls:
            path, controls = planner.get_best_path_with_controls()
            path = path.numpy()
            controls = controls.numpy()
            print(f"Controls ({controls.shape[0]} rows, {controls.shape[1]} dims):")
            print(controls)
        else:
            path = planner.get_best_path().numpy()

        # Create visualization
        show_map(plt, bitmap, node_type)

        # Check for direction info (last column) to color by search direction
        # Positive = forward search (green), Negative = backward search (purple)
        direction = path[:, -1]

        # Plot path segments colored by direction
        for i in range(len(path) - 1):
            segment_color = "green" if direction[i] >= 0 else "darkviolet"
            plt.plot(
                [path[i, 0] / map_res, path[i + 1, 0] / map_res],
                [path[i, 1] / map_res, path[i + 1, 1] / map_res],
                color=segment_color,
                linewidth=3,
                zorder=3,
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

            timesteps = node_info["timesteps"]
            for i in range(len(path) - 1):
                if i % timesteps == 0:
                    # Green for forward search, dark purple for backward search
                    car_color = "green" if direction[i] >= 0 else "darkviolet"
                    plot_car(
                        plt,
                        path[i, 0] / map_res,
                        path[i, 1] / map_res,
                        path[i, 2],
                        color=car_color,
                    )
        else:
            # Simple environment - plot path vertices colored by direction
            for i in range(len(path)):
                # Green for forward search, dark purple for backward search
                point_color = "green" if direction[i] >= 0 else "darkviolet"
                plt.scatter(
                    path[i, 0] / map_res,
                    path[i, 1] / map_res,
                    color=point_color,
                    s=20,
                    alpha=0.7,
                    zorder=5,
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

        # Add legend entries for forward/backward search colors
        plt.scatter([], [], color="green", label="Forward Search", s=50)
        plt.scatter([], [], color="darkviolet", label="Backward Search", s=50)

        # Add legend and labels
        plt.legend(loc="upper right")
        plt.title(
            f"{planner_type} Path Planning - {node_type.capitalize()} Environment"
        )
        plt.xlabel("X Position")
        plt.ylabel("Y Position")

        # Vertically flip the plot (image coordinates)
        plt.gca().invert_yaxis()

        print("Displaying visualization...")
        # Save figure to output directory (create if needed)
        output_dir = os.path.join(BASE_DIR.parent, "Content/standalone")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"{map_name}_{node_type}_{planner_type}.png"
        )
        plt.savefig(output_path)
        print(f"Saved to: {output_path}")
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
    parser.add_argument(
        "--bidirectional",
        "-b",
        action="store_true",
        help="Use bidirectional BiIGHAStar planner instead of unidirectional IGHAStar",
    )
    parser.add_argument(
        "--print-controls",
        action="store_true",
        help="Print the control sequence alongside the path",
    )
    # we assume the config is from examples/standalone folder:
    args = parser.parse_args()
    yaml_path = os.path.join(BASE_DIR.parent, "examples", "standalone", args.config)
    main(
        yaml_path=yaml_path,
        test_case=args.test_case,
        bidirectional=args.bidirectional,
        print_controls=args.print_controls,
    )
