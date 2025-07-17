import numpy as np
import matplotlib.pyplot as plt
import sys
import pathlib
import cv2
import time
import matplotlib
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as patches
from termcolor import colored
sys.path.append(str(pathlib.Path(__file__).parent.parent))
import yaml
import os
import time
import torch
from car import plot_car
import concurrent.futures
from collections import defaultdict
from scipy.stats import ttest_ind
import seaborn as sns
import pandas as pd
from utils import read_configs_from_pickle, load_from_pickle, conf, generate_binary_comparisons, generate_pvalue_matrix, bernoulli_confidence_test
from utils import get_path, compute_ratio_and_ci, compare_levels_ratios, get_map, conf, save_to_pickle
from concurrent.futures import ProcessPoolExecutor
import traceback
import pickle

def bezier_curve(u, v, T):
    u = np.array(u)
    v = np.array(v)
    p0 = np.array(u[:2])
    p3 = np.array(v[:2])
    L = np.linalg.norm(p3 - p0) * 0.33
    p1 = p0 + L * np.array([np.cos(u[2]), np.sin(u[2])])
    if np.dot(p1 - p0, p3 - p0) < 0:
        p1 = p0 - L * np.array([np.cos(u[2]), np.sin(u[2])])
    p2 = p3 - L * np.array([np.cos(v[2]), np.sin(v[2])])
    if np.dot(p3 - p2, p3 - p0) < 0:
        p2 = p3 + L * np.array([np.cos(v[2]), np.sin(v[2])])
    poses = []
    for t in T:
        point = (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3
        poses.append(point)
    return np.array(poses), p1, p2

def plot_vertex_list(plt, vertex_list, closed_list, color='gray', label="", experiment_info=None, width=1, path_list=[], freeze=False, extend_Q_v_with_Closed=False):
    if len(vertex_list) == 0 or experiment_info is None:
        return
    start_pose = tuple(experiment_info["start"])
    num_steps = 10
    step_size = experiment_info["node_info"]["step_size"]/num_steps
    if "map_res" in experiment_info["node_info"]:
        map_res = experiment_info["node_info"]["map_res"]
    else:
        map_res = 1.0
    node_info = experiment_info["node_info"]
    seen_set = set()

    if len(path_list) > 0:
        plt.scatter(path_list[0].pose[0]/map_res, path_list[0].pose[1]/map_res, color='black', label="Path", s=width*15)
    count = 0
    if freeze:
        inactive_color = 'blue'
        inactive_alpha = 0.1
    else:
        inactive_color = color
        inactive_alpha = 0.25

    offset_length = np.sqrt(width*15/np.pi)

    closed_set = set()
    for v in closed_list:
        closed_set.add(v.hash)
    if extend_Q_v_with_Closed:
        vertex_list.extend(closed_list)

    if "plot_tree" in experiment_info and experiment_info["plot_tree"]:
        for v in vertex_list:
            while v.parent:
                if not v.hash in seen_set:
                    seen_set.add(v.hash)
                else:
                    break
                u = v.parent
                if node_info["node_type"] == "simple":
                    x = np.array([u.pose[0], v.pose[0]]) / map_res
                    y = np.array([u.pose[1], v.pose[1]]) / map_res
                    if v.isvalid:
                        if v.active == 0:
                            if v.hash in closed_set:
                                plt.plot(x, y, color=color, linewidth=width, zorder=50)
                                plt.scatter(x[-1], y[-1], facecolor=color, edgecolor=color, s=width*15, zorder=50)
                            # else:
                            #     plt.scatter(x[-1], y[-1], facecolor="none", edgecolor=color, s=width*15, zorder=50)
                        else:
                            # convert circle size to length:
                            # s implies area:
                            # s = np.pi * r^2, r = sqrt(s/pi), s = width*15, therefore r = sqrt(width*15/np.pi)
                            # I then want to reduce the x-y length by r so that the line does not enter the circle:
                            # the vector of the line is (x[-1] - x[0], y[-1] - y[0])
                            plt.scatter(x[-1], y[-1], facecolor="white", s=width*15, edgecolor=inactive_color, alpha=inactive_alpha, zorder=1)
                            vec = np.array([x[-1] - x[0], y[-1] - y[0]])
                            vec = vec / np.linalg.norm(vec)
                            x[-1] -= offset_length * vec[0]
                            y[-1] -= offset_length * vec[1]
                            plt.plot(x, y, color=inactive_color, linewidth=width, alpha=inactive_alpha,zorder=0)
                    else:
                        plt.plot(x, y, color='r', linewidth=width,zorder=0)
                        plt.scatter(x[-1], y[-1], color='r', s=width*25)
                elif node_info["node_type"] == "kinematic" or node_info["node_type"] == "kinodynamic":
                    poses, p1, p2 = bezier_curve(u.pose, v.pose, np.linspace(0, 1, num_steps))
                    x = poses[:, 0] / map_res
                    y = poses[:, 1] / map_res
                    if v.isvalid:
                        if v.active == 0:
                            if v.hash in closed_set:
                                plt.scatter(x[-1], y[-1], facecolor=color, edgecolor=color, s=width*15, zorder=50)
                                plt.plot(x, y, color=color, linewidth=width, zorder=50)
                        else:
                            plt.plot(x, y, color=inactive_color, linewidth=width, alpha=inactive_alpha,zorder=0)
                            plt.scatter(x[-1], y[-1], facecolor="white", s=width*15, edgecolor=inactive_color, alpha=inactive_alpha, zorder=0)
                    else:
                        plt.plot(x, y, color='r', linewidth=width,zorder=0)
                        plot_car(plt, v.pose[0]/map_res, v.pose[1]/map_res, v.pose[2], color='r')
                v = v.parent

    seen_set.clear()
    for v in path_list:
        while v.parent:
            if not v.hash in seen_set:
                seen_set.add(v.hash)
            else:
                break
            u = v.parent
            if node_info["node_type"] == "simple":
                x = np.array([u.pose[0], v.pose[0]]) / map_res
                y = np.array([u.pose[1], v.pose[1]]) / map_res
                plt.plot(x, y, color='black', linewidth=width*2,zorder=10000)
                plt.scatter(x[-1], y[-1], color='black', s=width*25, zorder=10000)
            elif node_info["node_type"] == "kinematic" or node_info["node_type"] == "kinodynamic":
                poses, p1, p2 = bezier_curve(u.pose, v.pose, np.linspace(0, 1, num_steps))
                x = poses[:, 0] / map_res
                y = poses[:, 1] / map_res
                # plt.scatter(x[-1], y[-1], color='black', s=width*25, alpha=0.75, zorder=10000)
                plt.plot(x, y, color='black', linewidth=2*width, alpha=0.75, zorder=1000)
                plot_car(plt, v.pose[0]/map_res, v.pose[1]/map_res, v.pose[2], color='black', zorder=1000)

def plot_single_result_tree(yml_file="", variant_list=["dense_ResetBlind"], target_map="Berlin_2_1024.png", config_index=0, problem_resolution=2.0, focus_map=None, expansion_normalization=None):
    # load the yaml file:
    experiment_config_file = yaml.safe_load(open(yml_file, 'r'))
    pkl_file = experiment_config_file["pkl_file"]
    map_dir = experiment_config_file["map_dir"]
    cwd = os.getcwd()
    pkl_file = os.path.join(cwd, pkl_file)
    map_dir = os.path.join(cwd, map_dir)
    # create output_path directory if it does not exist:
    output_path = experiment_config_file["output_dir"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # load the result file:
    res_index = int(problem_resolution*100)
    search_res_index = int(experiment_config_file["search_res"]*100)
    map_res = experiment_config_file["experiment_info_default"]["node_info"]["map_res"]
    linewidth = experiment_config_file["linewidth"]
    if experiment_config_file["experiment_info_default"]["node_info"]["node_type"] == "simple":
        grid_line_width = 0.5
    else:
        grid_line_width = 0.01
    configs = read_configs_from_pickle(pkl_file)
    for map_name, resolution in configs:
        if map_name != target_map or resolution != problem_resolution:
            continue
        for i, experiment_info in enumerate(configs[(map_name, resolution)]["configs"]):
            if i != config_index:
                continue
            start = experiment_info["start"]
            goal = experiment_info["goal"]
            bitmap = get_map(map_name, map_dir=map_dir, map_size=experiment_info["map_size"], node_info=experiment_info["node_info"])

            for variant in variant_list:
                file_path = os.path.join(experiment_config_file["results_dir"], f"{map_name}_{i}_{res_index}_{search_res_index}_{variant}.pkl")
                tree_path = os.path.join(experiment_config_file["results_dir"], f"tree_snaps/{map_name}_{i}_{res_index}_{search_res_index}_{variant}.pkl")
                if not os.path.exists(file_path):
                    print(f"File {file_path} does not exist")
                    continue
                result = load_from_pickle(file_path)
                tree_results = load_from_pickle(tree_path)
                path_list = result["path_list"]
                path_cost_list = result["path_cost_list"]
                level_information = result["level_information"]
                                    
                print("start, goal: ", start, goal)
                interval = int(experiment_config_file["search_res"]/map_res)
                map_width = bitmap.shape[1]
                epsilon = experiment_config_file["experiment_info_default"]["epsilon"]/map_res
                max_level = experiment_config_file["max_level"]
                level_range = range(max_level+1)
                if "denseSparse" in variant:
                    first_solution_index = np.argmax(np.array(path_cost_list) < 1e8)
                    level_range = [0, first_solution_index]
                for level in level_range:
                    print(f"Path {variant} {level} cost: {path_cost_list[level]}")
                    Q_v = tree_results["Q_v"][level]
                    closed_list = tree_results["closed_set"][level]
                    fig = plt.figure(figsize=(4, 4))
                    plt.clf()
                    if level == 0:
                        continue
                    interval = (experiment_config_file["search_res"]/map_res)/(experiment_config_file["experiment_info_default"]["division_factor"]**level_information[level])
                    for x in np.arange(0, map_width + interval, interval):
                        plt.plot([x, x], [0, map_width], color='black', linewidth=grid_line_width,zorder = 0)
                    for y in np.arange(0, map_width + interval, interval):
                        plt.plot([0, map_width], [y, y], color='black', linewidth=grid_line_width,zorder = 0)
                    show_map(plt, bitmap, experiment_info["node_info"]["node_type"], alpha=0.8)
                    plot_vertex_list(plt, Q_v, closed_list, experiment_info=experiment_config_file["experiment_info_default"], color='gray', label="Q_v", path_list=path_list[level], width=linewidth, freeze=(level!=0))
                    # plt.scatter(goal[0]/map_res, goal[1]/map_res, color='gray', label="Goal", s=800, alpha=0.75, edgecolors='black',zorder=100)
                    # start_circle = plt.Circle((start[0]/map_res, start[1]/map_res),color='gray', edgecolor='black', radius=epsilon/2, alpha=0.75, zorder=100)
                    # plt.scatter(start[0]/map_res, start[1]/map_res, color='black', label="Start", s=25, alpha=0.75, zorder=10000)
                    if experiment_info["node_info"]["node_type"] == "simple":
                        plt.scatter(start[0]/map_res, start[1]/map_res, color='black', label="Start", s=linewidth*25, zorder=10000)
                    elif experiment_info["node_info"]["node_type"] == "kinematic" or experiment_info["node_info"]["node_type"] == "kinodynamic":
                        plot_car(plt, start[0]/map_res, start[1]/map_res, start[2], color='black',zorder=10000)
                    plt.scatter(start[0]/map_res, start[1]/map_res, color="black", s=linewidth*15,zorder=10000)
                    goal_circle = plt.Circle((goal[0]/map_res, goal[1]/map_res),facecolor='gray', edgecolor='black', radius=epsilon, alpha=0.75, zorder=100)
                    plt.gca().add_patch(goal_circle)
                    # plt.legend()
                    # plt.xlabel("X (m)")
                    # plt.ylabel("Y (m)")
                    if path_cost_list[level] < 1e8:
                        cost = path_cost_list[level]
                    else:
                        cost = "No path"
                    # plt.title(f"Level {level}, Exp: {result['expansion_list'][level]}, Cost: {cost}")
                    # plt.xticks(ticks=np.arange(0, bitmap.shape[1], 200), labels=np.arange(0, bitmap.shape[1]*experiment_info["node_info"]["map_res"], 20))
                    # plt.yticks(ticks=np.arange(0, bitmap.shape[0], 200), labels=np.arange(0, bitmap.shape[0]*experiment_info["node_info"]["map_res"], 20))
                    plt.xticks([])
                    plt.yticks([])
                    if focus_map is not None:
                        plt.xlim(focus_map[0], focus_map[1])
                        plt.ylim(focus_map[2], focus_map[3])
                    else:
                        plt.xlim(0, bitmap.shape[1])
                        plt.ylim(bitmap.shape[0], 0)
                    # Add a new axis for the progress bar

                    # Add the bar in a new axes above the image
                    # Add progress bar above image
                    # [left, bottom, width, height] -- tune these as needed
                    if expansion_normalization is not None:
                        bar_ax = fig.add_axes([0.04, 0.97, 0.92, 0.025])
                        bar_ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black'))
                        bar_ax.add_patch(patches.Rectangle((0, 0), result['expansion_list'][level]/expansion_normalization, 1, facecolor='gray'))
                        bar_ax.set_xlim(0, 1)
                        bar_ax.set_ylim(0, 1)
                        bar_ax.axis('off')
                    plt.tight_layout()
                    plt.savefig(f"{output_path}/{map_name}_{i}_{res_index}_{search_res_index}_{variant}_{level}_E_{result['expansion_list'][level]}_C_{cost}.pdf")
                    # plt.show()
                    # exit()

def show_map(plt, bitmap, node_type, alpha=0.6):
    if node_type == "simple":
        plt.imshow(bitmap, cmap='gray', alpha=alpha)
    elif node_type == "kinodynamic":
        costmap = bitmap[..., 0].cpu().numpy()
        elevation_map = bitmap[..., 1].cpu().numpy()
        costmap_color = np.clip(costmap, 0, 255).astype(np.uint8)
        pink = np.array([255, 105, 180], dtype=np.uint8)  # BGR format
        white = np.array([255, 255, 255], dtype=np.uint8)
        # Create an output color image with shape (H, W, 3)
        color_map = np.zeros((costmap_color.shape[0], costmap_color.shape[1], 3), dtype=np.uint8)
        # Use broadcasting to assign colors
        mask_white = costmap_color == 255
        mask_pink = ~mask_white

        color_map[mask_white] = white
        color_map[mask_pink] = pink
        costmap_color = color_map
        vmin = np.min(elevation_map)
        vmax = np.max(elevation_map)
        elev_norm = np.clip((elevation_map - vmin) / (vmax - vmin), 0, 1)
        elev_uint8 = (elev_norm * 255).astype(np.uint8)
        elev_color = np.stack([elev_uint8]*3, axis=-1)  # shape becomes (H, W, 3)

        # Blend the two images
        costmap = costmap_color
        costmap[mask_white] = elev_color[mask_white]
        plt.imshow(costmap)
    elif node_type == "kinematic":
        costmap = bitmap[..., 0]
        plt.imshow(costmap, cmap='gray', alpha=alpha)

def plot_path(plt, path_list, node_type, node_info, color, variant, solution_index=0, solution_expansions=0, solution_cost=0):
    map_res = node_info["map_res"]
    if node_type == "kinematic" or node_type == "kinodynamic":
        if variant == "Final":
            line_width = 3
        else:
            line_width = 1
        for index, path in enumerate(path_list):
            coords = get_path(path)
            if index == solution_index:
                for i in range(len(coords)):
                    if i == 0:
                        plot_car(plt, coords[i, 0]/map_res, coords[i, 1]/map_res, coords[i, 2], color=color, width=line_width, label=f"{variant}") #: W/E:{np.round(solution_cost,2)}/{solution_expansions}")
                    else:
                        plot_car(plt, coords[i, 0]/map_res, coords[i, 1]/map_res, coords[i, 2], color=color, width=line_width)
                break
        return

def profiler_plot(yml_file="", hysteresis_list=["0", "10", "50", "250", "1250"]):
    # profiling data is stored in yml_file["profiler_output_dir"]
    # plotting output sits in yml_file["output_dir"]
    cwd = os.getcwd()
    yml_file = yaml.safe_load(open(yml_file))
    pkl_file = os.path.join(cwd, yml_file["pkl_file"])
    results_dir = os.path.join(cwd, yml_file["results_dir"])
    map_dir = os.path.join(cwd, yml_file["map_dir"])
    node_type = yml_file["experiment_info_default"]["node_info"]["node_type"]
    # Read configurations from YAML
    configs = read_configs_from_pickle(pkl_file)
    results_folder = os.path.join(cwd, yml_file["profiler_data_dir"])
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    avg_successor_time = []
    avg_goal_check_time = []
    avg_g_update_time = []
    avg_overhead_time = []
    switches = []
    max_level_profile = []
    Q_v_size = []
    expansion_counter = []
    time_taken = []
    success = []
    expansion_list = []

    for hysteresis in hysteresis_list:
        avg_successor_time.append([])
        avg_goal_check_time.append([])
        avg_g_update_time.append([])
        avg_overhead_time.append([])
        switches.append([])
        max_level_profile.append([])
        Q_v_size.append([])
        expansion_counter.append([])
        time_taken.append([])
        success.append([])
        expansion_list.append([])


    for (map_name, resolution) in configs: # this "resolution" refers to the resolution at which the solution was found.
        # if "race" in map_name:
        #     continue
        for i, experiment_info in enumerate(configs[(map_name, resolution)]["configs"]):
            for hysteresis in hysteresis_list:
                # results = {
                #     "avg_successor_time": avg_successor_time,
                #     "avg_goal_check_time": avg_goal_check_time,
                #     "avg_overhead_time": avg_overhead_time,
                #     "switches": switches,
                #     "max_level_profile": max_level_profile,
                #     "Q_v_size": Q_v_size,
                #     "expansion_counter": expansion_counter,
                #     "time_taken": end - now,
                # }
                # save_to_pickle(results, f"{results_folder}/{map_name}_{i}_{hysteresis}.pkl")
                # the above is how we save the results, now we need to load them:
                file_path = os.path.join(results_folder, f"{map_name}_{i}_{hysteresis}.pkl")
                if not os.path.exists(file_path):
                    # print(f"File {file_path} does not exist")
                    continue
                result = load_from_pickle(file_path)
                avg_successor_time[hysteresis_list.index(hysteresis)].append(result["avg_successor_time"])
                avg_goal_check_time[hysteresis_list.index(hysteresis)].append(result["avg_goal_check_time"])
                avg_g_update_time[hysteresis_list.index(hysteresis)].append(result["avg_g_update_time"])
                avg_overhead_time[hysteresis_list.index(hysteresis)].append(result["avg_overhead_time"])
                switches[hysteresis_list.index(hysteresis)].append(result["switches"])
                max_level_profile[hysteresis_list.index(hysteresis)].append(result["max_level_profile"])
                Q_v_size[hysteresis_list.index(hysteresis)].append(result["Q_v_size"])
                expansion_counter[hysteresis_list.index(hysteresis)].append(result["expansion_counter"])
                time_taken[hysteresis_list.index(hysteresis)].append(result["time_taken"])
                success[hysteresis_list.index(hysteresis)].append(result["success"])
                if result["success"] == 1:
                    expansion_list[hysteresis_list.index(hysteresis)].append(result["expansion_list"])
                    # print(f"Success: {result['expansion_list']}")
                else:
                    expansion_list[hysteresis_list.index(hysteresis)].append(result["expansion_list"])
                # expansions_to_first_path = result["expansion_list"][0]
    for hysteresis in hysteresis_list:
        successor_time = np.array(avg_successor_time[hysteresis_list.index(hysteresis)])
        goal_check_time = np.array(avg_goal_check_time[hysteresis_list.index(hysteresis)])
        g_update_time = np.array(avg_g_update_time[hysteresis_list.index(hysteresis)])
        overhead_time = np.array(avg_overhead_time[hysteresis_list.index(hysteresis)])
        print(overhead_time)
        avg_successor_time = successor_time.mean()
        conf_successor_time = conf(successor_time)
        avg_goal_check_time = goal_check_time.mean()
        conf_goal_check_time = conf(goal_check_time)
        avg_g_update_time = g_update_time.mean()
        conf_g_update_time = conf(g_update_time)
        avg_overhead_time = overhead_time.mean()
        conf_overhead_time = conf(overhead_time)
        total_inner_loop_time = avg_successor_time + avg_goal_check_time + avg_g_update_time
        avg_overhead_time = overhead_time.mean()
        conf_overhead_time = conf(overhead_time)
        # print the average successor, goal check, g_update, overhead time and total time with confidence intervals denoted as /pm:
        print(f"Hysteresis: {hysteresis}")
        print(f"Avg. Successor Time: {avg_successor_time:.2f} +/- {conf_successor_time:.2f}")
        print(f"Avg. Goal Check Time: {avg_goal_check_time:.2f} +/- {conf_goal_check_time:.2f}")
        print(f"Avg. G Update Time: {avg_g_update_time:.2f} +/- {conf_g_update_time:.2f}")
        print(f"Avg. Overhead Time: {avg_overhead_time:.2f} +/- {conf_overhead_time:.2f}")
        print(f"Avg. Total Time: {total_inner_loop_time:.2f} +/- {conf_successor_time + conf_goal_check_time + conf_g_update_time:.2f}")
        
    exit()

    # what is the average overhead x expansion_counter divided by the total time taken?
    line_thickness = 2
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    num_vars = len(hysteresis_list)


    from brokenaxes import brokenaxes
    x_labels = []
    for hysteresis in hysteresis_list:
        if hysteresis == 100000:
            hysteresis = "$\infty$"
        x_labels.append(str(hysteresis))
    x_pos = np.arange(len(hysteresis_list))

    plt.figure(figsize=(6,3))
    # adjust figure top and bottom:
    plt.subplots_adjust(top=0.95, bottom=0.15, right = 1.0, left = 0.12)
    bax = brokenaxes(xlims=((-0.5, 0.5), (0.5, num_vars-1.5), (num_vars-1.5, num_vars-0.5)))

    scaler = [1, 10]
    color_list = ["black", "grey"]
    succ_time = np.mean(avg_successor_time)
    label = [f"Succ on GPU:~50us", f"Succ on CPU:~500us"]
    for j in range(len(scaler)):
        mean_list = []
        CI_list = []
        for i, hysteresis in enumerate(hysteresis_list):
            adjusted_time = np.array(time_taken[i]) - 1e-6*np.array(avg_successor_time[i]) * np.array(expansion_counter[i])
            adjusted_time += 1e-6 * np.array(avg_successor_time[i]) * np.array(expansion_counter[i]) * scaler[j]
            ratio = 1e-6*np.array(avg_overhead_time[i]) * np.array(expansion_counter[i]) / adjusted_time
            mean = ratio.mean()
            CI = conf(ratio)
            mean_list.append(mean)
            CI_list.append(CI)
        bax.errorbar(x_pos, mean_list, yerr=CI_list, fmt='o', color=color_list[j],
                linewidth=line_thickness, capsize=5, markersize=6, label=label[j], alpha=0.75)
        bax.plot(x_pos, mean_list, color=color_list[j], linewidth=line_thickness)

    bax.axs[0].set_xticks([0])
    bax.axs[0].set_xticklabels([x_labels[0]])

    bax.axs[1].set_xticks(np.arange(1,num_vars-1))
    bax.axs[1].set_xticklabels(x_labels[1:num_vars-1])

    bax.axs[2].set_xticks([num_vars-1])
    bax.axs[2].set_xticklabels([x_labels[num_vars-1]])
    bax.set_xlabel(r"$H_\mathrm{Thresh}$", fontsize=12)
    bax.set_ylabel("Overhead / Total Time", fontsize=12)
    bax.legend(loc='upper right', fontsize=12, frameon=False)
    plt.savefig(os.path.join(results_folder, "overhead_to_succ_ratio.pdf"), bbox_inches='tight', pad_inches=0)
    plt.show()

    plt.figure(figsize=(6,3))
    # adjust figure top and bottom:
    plt.subplots_adjust(top=0.95, bottom=0.15, right = 1.0, left = 0.12)
    bax = brokenaxes(xlims=((-0.5, 0.5), (0.5, num_vars-1.5), (num_vars-1.5, num_vars-0.5)))

    mean_list = []
    CI_list = []
    j = 0
    for i, hysteresis in enumerate(hysteresis_list):
        ratio = np.array(switches[i]) #/adjusted_time
        mean = ratio.mean()
        CI = conf(ratio)
        mean_list.append(mean)
        CI_list.append(CI)
    bax.errorbar(x_pos, mean_list, yerr=CI_list, fmt='o', color=color_list[j],
            linewidth=line_thickness, capsize=5, markersize=6, label=label[j], alpha=0.75)
    bax.plot(x_pos, mean_list, color=color_list[j], linewidth=line_thickness)

    bax.axs[0].set_xticks([0])
    bax.axs[0].set_xticklabels([x_labels[0]])

    bax.axs[1].set_xticks(np.arange(1,num_vars-1))
    bax.axs[1].set_xticklabels(x_labels[1:num_vars-1])

    bax.axs[2].set_xticks([num_vars-1])
    bax.axs[2].set_xticklabels([x_labels[num_vars-1]])
    bax.set_xlabel(r"$H_\mathrm{Thresh}$", fontsize=12)
    bax.set_ylabel("Avg. Number of Level Changes", fontsize=12)
    # bax.legend(loc='upper right', fontsize=12, frameon=False)
    plt.savefig(os.path.join(results_folder, "avg_level_changes.pdf"), bbox_inches='tight', pad_inches=0)
    # plt.show()

    plt.figure(figsize=(6,3))
    # adjust figure top and bottom:
    plt.subplots_adjust(top=0.95, bottom=0.15, right = 1.0, left = 0.12)
    bax = brokenaxes(xlims=((-0.5, 0.5), (0.5, num_vars-1.5), (num_vars-1.5, num_vars-0.5)))

    mean_list = []
    CI_list = []
    j = 0
    for i, hysteresis in enumerate(hysteresis_list):
        # first path expansions only exist if you were successful:
        first_path = []
        for k in range(len(expansion_list[i])):
            if success[i][k] == 1:
                first_path.append(expansion_list[i][k][0])
        # ratio = np.array(first_path) #/adjusted_time
        ratio = np.array(success[i])
        # ratio = np.array(time_taken[i])
        mean = ratio.mean()
        CI = conf(ratio)
        mean_list.append(mean)
        CI_list.append(CI)
    bax.errorbar(x_pos, mean_list, yerr=CI_list, fmt='o', color=color_list[j],
            linewidth=line_thickness, capsize=5, markersize=6, label=label[j], alpha=0.75)
    bax.plot(x_pos, mean_list, color=color_list[j], linewidth=line_thickness)

    bax.axs[0].set_xticks([0])
    bax.axs[0].set_xticklabels([x_labels[0]])

    bax.axs[1].set_xticks(np.arange(1,num_vars-1))
    bax.axs[1].set_xticklabels(x_labels[1:num_vars-1])

    bax.axs[2].set_xticks([num_vars-1])
    bax.axs[2].set_xticklabels([x_labels[num_vars-1]])
    bax.set_xlabel(r"$H_\mathrm{Thresh}$", fontsize=12)
    bax.set_ylabel("success rate", fontsize=12)
    # bax.legend(loc='upper right', fontsize=12, frameon=False)
    # plt.savefig(os.path.join(results_folder, "terminal_expansions.pdf"), bbox_inches='tight', pad_inches=0)
    plt.show()

    plt.figure(figsize=(6,3))
    # adjust figure top and bottom:
    num_vars = len(hysteresis_list)
    plt.subplots_adjust(top=0.95, bottom=0.15, right = 1.0, left = 0.12)
    bax = brokenaxes(xlims=((-0.5, 0.5), (0.5, num_vars-1.5), (num_vars-1.5, num_vars-0.5)))

    mean_list = []
    CI_list = []
    j = 0
    for i, hysteresis in enumerate(hysteresis_list):
        # first path expansions only exist if you were successful:
        first_path = []
        for k in range(len(expansion_list[i])):
            if success[i][k] == 1:
                first_path.append(expansion_list[i][k][0])
        ratio = np.array(first_path) #/adjusted_time
        # ratio = np.array(success[i])
        # ratio = np.array(time_taken[i])
        mean = ratio.mean()
        CI = conf(ratio)
        mean_list.append(mean)
        CI_list.append(CI)
    bax.errorbar(x_pos, mean_list, yerr=CI_list, fmt='o', color=color_list[j],
            linewidth=line_thickness, capsize=5, markersize=6, label=label[j], alpha=0.75)
    bax.plot(x_pos, mean_list, color=color_list[j], linewidth=line_thickness)

    bax.axs[0].set_xticks([0])
    bax.axs[0].set_xticklabels([x_labels[0]])

    bax.axs[1].set_xticks(np.arange(1,num_vars-1))
    bax.axs[1].set_xticklabels(x_labels[1:num_vars-1])

    bax.axs[2].set_xticks([num_vars-1])
    bax.axs[2].set_xticklabels([x_labels[num_vars-1]])
    bax.set_xlabel(r"$H_\mathrm{Thresh}$", fontsize=12)
    bax.set_ylabel("Exp to first path", fontsize=12)
    # bax.legend(loc='upper right', fontsize=12, frameon=False)
    # plt.savefig(os.path.join(results_folder, "terminal_expansions.pdf"), bbox_inches='tight', pad_inches=0)
    plt.show()

def in_the_loop_plot(yml_file="", data_dir=""):
    experiment_config_file = yaml.safe_load(open(yml_file, 'r'))
    scenarios = experiment_config_file["scenarios"]
    hysteresis_list = experiment_config_file["hysteresis"]
    num_trials = experiment_config_file["num_iters"]
    # -1 corresponds to HA_M, otherwise it is IGHAStar-hysteresis
    from pathlib import Path
    variant = {}
    cwd = os.getcwd()
    for hyst in hysteresis_list:
        variant[hyst] = {}
        if hyst == -1:
            variant_name = "IGHAStar--1"
        elif hyst == "MPC":
            variant_name = "MPC"
        else:
            variant_name = f"IGHAStar-{hyst}"
        for scenario in scenarios:
            WP_file = str(Path(os.getcwd()).parent.absolute()) + "/IGHAStar/Global_Planning_Waypoints/" + scenario + ".pkl"
            WP_data = pickle.load(open(WP_file, "rb"))
            WP_path = WP_data["path"]
            goal_wp = WP_path[-1] # last waypoint is the goal
            start = WP_path[0] # first waypoint is the start
            
            variant[hyst][scenario] = {}
            for trial in range(num_trials):
                variant[hyst][scenario][trial] = {}
                if hyst == -1 or hyst == 100:
                    file_path = os.path.join(os.path.join(cwd, "MPC/benchmark/"), f"{scenario}_{variant_name}_{trial}.pkl")
                else:
                    file_path = os.path.join(data_dir, f"{scenario}_{variant_name}_{trial}.pkl")
                data = load_from_pickle(file_path)
                timestamps = data["timestamps"]
                state_data = data["state_data"]
                reset_data = data["reset_data"]
                path_data = data["path_data"]
                expansions_data = data["expansions_data"]
                goal_data = data["goal_data"]
                # first index at which state[6] > 0.5 m/s to know when the car started moving:
                start_index = 0
                time_taken = timestamps[-1] - timestamps[start_index]
                if time_taken < 118:
                    goal = goal_data[-1]  # last goal in the goal data
                else:
                    goal = goal_wp # you timed out, the best guess is the last goal wp.
                start_goal_distance = np.linalg.norm(np.array(goal[:2]) - np.array(state_data[start_index][:2]))
                progress = np.linalg.norm(np.array(state_data)[:, :2] - np.array(goal[:2]), axis=1)/start_goal_distance
                progress = 1 - progress
                # if last state is within wp radius of goal, progress is set to 1:
                progress[np.where(np.linalg.norm(np.array(state_data)[:, :2] - np.array(goal[:2]), axis=1) < experiment_config_file["wp_radius"][0])] = 1
                # print(np.linalg.norm(np.array(state_data)[-1, :2] - np.array(goal[:2])))
                # shape of progress is (num_steps). Num steps is 120 x 25 = 3000, but we may have terminated early, so pad the end with last progress value
                prog_array = np.ones(3000) * progress[-1]
                prog_array[:len(progress)] = progress[:3000]
                
                variant[hyst][scenario][trial]["time_taken"] = time_taken
                variant[hyst][scenario][trial]["path_data"] = path_data
                variant[hyst][scenario][trial]["state_data"] = state_data
                variant[hyst][scenario][trial]["expansions_data"] = expansions_data
                variant[hyst][scenario][trial]["progress"] = prog_array
    # TODO: for each hysteresis, we want to plot the progress vs time (time goes from 0 120, in steps of 0.04). Take mean and CI of progress for each hysteresis value.
    if 1:
        time_array = np.arange(0, 120, 0.04)  # 3000 steps
        assert len(time_array) == 3000, "Time array should have 3000 steps corresponding to 120 seconds at 0.04s intervals."

        plt.figure(figsize=(4, 4))


        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        # font size 12:
        plt.rcParams["font.size"] = 12
        color_list = ['#ca0020','#f4a582','#404040', '#bababa']
        for hyst in variant:
            all_times = []
            all_success = []
            for scenario in variant[hyst]:
                for trial in variant[hyst][scenario]:
                    time_taken = variant[hyst][scenario][trial]["time_taken"]
                    all_times.append(time_taken)
                    success = variant[hyst][scenario][trial]["progress"][-1] == 1.0 # if the last progress is greater than 0.99, we consider it a success
                    all_success.append(success)
            all_times = np.array(all_times)
            # sorted indices of all_times
            sorted_indices = np.argsort(all_times)
            all_times = all_times[sorted_indices]
            success = np.array(all_success)[sorted_indices]
        
            CI = []
            cdf = []
            for t in range(len(all_times)):
                completion = np.zeros_like(all_times)
                completion[all_times <= all_times[t]] = success[all_times <= all_times[t]]
                mean, delta, _  = bernoulli_confidence_test(completion)
                cdf.append(mean)
                CI.append(delta)
            cdf = np.array(cdf)
            CI = np.array(CI)
            if hyst == -1:
                label = "HA*M"
            elif hyst == "MPC":
                label = "MPC"
            elif hyst == "MPC_bad":
                label = "MPC_short"
            else:
                label = f"IGHA*"
            clr = color_list[hysteresis_list.index(hyst) % len(color_list)]
            # use matplotlib to generate a new color:
            # clr = plt.cm.viridis(hysteresis_list.index(hyst) / len(hysteresis_list))  # Use a colormap to generate a color
            plt.plot(all_times, cdf, label=label, color=clr)
            plt.scatter(all_times, cdf, s=10, color=clr)  # scatter points for better visibility
            plt.fill_between(all_times, cdf - CI, cdf + CI, alpha=0.2, color=clr)

        plt.xlabel("Time Taken(s)", fontsize=12)
        plt.ylabel("Success Rate", fontsize=12)
        # set line thickness of Y axis and X axis to 2
        plt.gca().spines["left"].set_linewidth(2)
        plt.gca().spines["bottom"].set_linewidth(2)
        plt.gca().spines["top"].set_linewidth(0)
        plt.gca().spines["right"].set_linewidth(0)
        plt.xticks(np.arange(0, 140, 40))  # Show ticks at 0, 40, 80, 120
        plt.yticks(np.arange(0.0, 1.25, 0.25))  # Show ticks at 0.0, 0.2, ..., 1.0
        plt.legend()
        plt.ylim(0, 1.0)

        # Save figure
        plt.tight_layout()
        plt.savefig("Success_vs_time.pdf", bbox_inches='tight', pad_inches=0)
        plt.show()

        # plt.figure(figsize=(4, 3))


        # plt.rcParams["font.family"] = "serif"
        # plt.rcParams["font.serif"] = ["Times New Roman"]
        # # font size 12:
        # plt.rcParams["font.size"] = 12
        # color_list = ["green", '#853d03', "grey"]
        # import scipy.stats as st
        # for hyst in variant:
        #     all_progress = []

        #     for scenario in variant[hyst]:
        #         for trial in variant[hyst][scenario]:
        #             progress = variant[hyst][scenario][trial]["progress"]
        #             all_progress.append(progress)

        #     all_progress = np.array(all_progress)  # shape: (num_trials_total, 3000)
        #     mean_progress = np.mean(all_progress, axis=0)
        #     CI = st.norm.interval(0.95, loc=np.mean(all_progress, axis=0), scale=st.sem(all_progress,axis=0))
        #     ci_progress = (CI[1] - CI[0])/2
        #     # print(ci_progress)
        #     clr = color_list[hysteresis_list.index(hyst) % len(color_list)]
        #     label = "HA*M" if hyst == -1 else f"IGHA*-{hyst}" if hyst != "MPC" else "MPC"
        #     plt.plot(time_array, mean_progress, label=label, color=clr)
        #     plt.fill_between(time_array, mean_progress - ci_progress, mean_progress + ci_progress, alpha=0.2, color=clr)
        # plt.xlabel("Time Taken(s)")
        # plt.ylabel("Progress")
        # # set line thickness of Y axis and X axis to 2
        # plt.gca().spines["left"].set_linewidth(2)
        # plt.gca().spines["bottom"].set_linewidth(2)
        # plt.gca().spines["top"].set_linewidth(0)
        # plt.gca().spines["right"].set_linewidth(0)
        # plt.xticks(np.arange(0, 140, 40))  # Show ticks at 0, 40, 80, 120
        # plt.yticks(np.arange(0.0, 1.25, 0.25))  # Show ticks at 0.0, 0.2, ..., 1.0
        # plt.legend()
        # plt.ylim(0, 1.0)

        # # Save figure
        # plt.tight_layout()
        # plt.savefig("Progress_vs_time.pdf", bbox_inches='tight', pad_inches=0)
        # plt.show()
        exit()

    # now, go through the scenarios and trials, and find the ratio of time taken between the two hysteresis values:
    ratio_dict = {}
    speed_dict = {}
    non_DNF_dict = {}
    expansions_dict = {}
    time_per_expansion = 60e-6
    DNF = {}
    DNF_stuck = {}

    for hyst in hysteresis_list:
        # if hyst == -1:
        #     continue
        ratio_list = []
        non_DNF_ratio = []
        speed_list = []
        expansions_list = []
        DNF[hyst] = []
        DNF_stuck[hyst] = []
        for scenario in scenarios:
            for trial in range(num_trials):
                time_taken = variant[hyst][scenario][trial]["time_taken"]
                ratio =  time_taken / variant[-1][scenario][trial]["time_taken"]
                speed = np.array(variant[hyst][scenario][trial]["state_data"])[:, 6:9]
                expansions = np.array(variant[hyst][scenario][trial]["expansions_data"])
                expansions = expansions[expansions > 2] # remove the invalid cases where the start/goal is invalid
                expansions = np.mean(expansions)*time_per_expansion
                raw_speed = np.linalg.norm(speed, axis=1)
                speed = raw_speed.mean()
                if time_taken > 118:
                    DNF[hyst].append(1)
                    # were you moving or were you stuck
                    if raw_speed[-1] < 1.0:
                        DNF_stuck[hyst].append(1)
                if time_taken < 118 and variant[-1][scenario][trial]["time_taken"] < 118:
                    non_DNF_ratio.append(ratio)
                if hyst == 100 and ratio < 0.8:
                    MPC_time = variant["MPC"][scenario][trial]["time_taken"]
                    IGHA_time = variant[100][scenario][trial]["time_taken"]
                    print(f"Scenario: {scenario}, Trial: {trial}, ratio: {ratio}, MPC time: {MPC_time}, IGHA time: {IGHA_time}")
                ratio_list.append(ratio)
                speed_list.append(speed)
                expansions_list.append(expansions)
        mean_ratio = np.mean(ratio_list)
        conf_ratio = conf(ratio_list)
        ratio_dict[hyst] = {"mean": mean_ratio, "conf": conf_ratio}
        mean_speed = np.mean(speed_list)
        conf_speed = conf(speed_list)
        speed_dict[hyst] = {"mean": mean_speed, "conf": conf_speed}
        non_DNF_ratio = np.array(non_DNF_ratio)
        mean_when_faster = np.mean(non_DNF_ratio)
        conf_when_faster = conf(non_DNF_ratio)
        non_DNF_dict[hyst] = {"mean": mean_when_faster, "conf": conf_when_faster}
        mean_expansions = np.mean(expansions_list)
        conf_expansions = conf(expansions_list)
        expansions_dict[hyst] = {"mean": mean_expansions, "conf": conf_expansions}
    exit()
    print("average time per plan")
    # for hyst in hysteresis_list:
    #     print(f"Hysteresis {hyst}: {expansions_dict[hyst]['mean']} +/- {expansions_dict[hyst]['conf']}")
    #     # print(f"Hysteresis {hyst}: {expansions_dict[hyst]['mean'] + expansions_dict[hyst]['conf']} - {expansions_dict[hyst]['mean'] - expansions_dict[hyst]['conf']}")
    # exit()
    # now, print the results:
    print("Hysteresis Ratio:")
    for hyst in hysteresis_list:
        if hyst == -1:
            continue
        print(f"Hysteresis {hyst}: {ratio_dict[hyst]['mean']} +/- {ratio_dict[hyst]['conf']}")
    print("Speed Ratio:")
    for hyst in hysteresis_list:
        if hyst == -1:
            continue
        print(f"Hysteresis {hyst}: {speed_dict[hyst]['mean']} +/- {speed_dict[hyst]['conf']}")
    print("Non_DNF_ratio:")
    for hyst in hysteresis_list:
        if hyst == -1:
            continue
        print(f"Hysteresis {hyst}: {non_DNF_dict[hyst]['mean']} +/- {non_DNF_dict[hyst]['conf']}")
    for hyst in hysteresis_list:
        print(f"DNFs {hyst}: DNF: {len(DNF[hyst])} / {num_trials*len(scenarios)}")
        print(f"DNFs {hyst}: DNF_stuck: {len(DNF_stuck[hyst])} / {num_trials*len(scenarios)}")
    exit()
    # Filter out hyst = -1
    hyst_values = [h for h in hysteresis_list if h != -1]
    hyst_labels = [str(h) for h in hyst_values]
    # change hyst_label 5000 to "$\infty$"
    hyst_labels = [hyst_labels[i].replace("5000", "IGHA*-$\infty$") for i in range(len(hyst_labels))]
    hyst_labels = [hyst_labels[i].replace("100", "IGHA*-100") for i in range(len(hyst_labels))]

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 12

    # line thickness = 2
    line_thickness = 2
    # Get bar values
    ratios = [non_DNF_dict[h]['mean'] for h in hyst_values]
    ratios_conf = [non_DNF_dict[h]['conf'] for h in hyst_values]

    # Set positions
    x = np.arange(len(hyst_values))
    width = 1.0  # width of bars

    # Create plot
    fig = plt.figure(figsize=(3, 4))

    full_ratio = 1.0
    # Light gray full bar (background)
    plt.bar(x, [full_ratio]*len(hyst_labels), width,
        color='lightgray', edgecolor='black', linewidth=line_thickness)

    # Dark gray actual bar (overlay)
    plt.bar(x, ratios, width,
        color='dimgray', edgecolor='black', linewidth=line_thickness)

    # Error bars on top of dark gray bars
    plt.errorbar(x, ratios, yerr=ratios_conf, fmt='none',
                ecolor='black', capsize=4, linewidth=line_thickness)

    # Y-axis and grid
    plt.ylim(0, 1.5)
    plt.xlim(-0.5, len(hyst_values)-0.5)

    # Labels and ticks
    # plt.set_ylabel('Ratio')
    # remove y ticks:
    # plt.set_yticks([])
    plt.tick_params(axis='x', which='both', bottom=False, top=False)  # Disables tick marks
    plt.gca().spines["top"].set_linewidth(line_thickness)
    plt.gca().spines["right"].set_linewidth(line_thickness)
    plt.gca().spines["left"].set_linewidth(line_thickness)
    plt.gca().spines["bottom"].set_linewidth(line_thickness)
    plt.xticks(x, hyst_labels, rotation=0)
    plt.tick_params(axis='y', length=0)
    plt.tick_params(axis='x', length=0)
    plt.title('In-The-Loop Time w.r.t HA*M', fontsize=13.5)
    # Optional grid

    plt.tight_layout()
    # plt.savefig("in_the_loop_ratio.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()

def kinematic_kinodynamic(yml_file="", data_dir=""):
    experiment_config_file = yaml.safe_load(open(yml_file, 'r'))
    scenarios = experiment_config_file["scenarios"]
    planner_list = experiment_config_file["planner"]
    num_trials = experiment_config_file["num_iters"]
    # -1 corresponds to HA_M, otherwise it is IGHAStar-hysteresis
    variant = {}
    for variant_name in planner_list:
        variant[variant_name] = {}
        for scenario in scenarios:
            variant[variant_name][scenario] = {}
            for trial in range(num_trials):
                variant[variant_name][scenario][trial] = {}
                file_path = os.path.join(data_dir, f"{scenario}_{variant_name}_{trial}.pkl")
                data = load_from_pickle(file_path)
                timestamps = data["timestamps"]
                state_data = data["state_data"]
                reset_data = data["reset_data"]
                path_data = data["path_data"]
                expansions_data = data["expansions_data"]
                # first index at which state[6] > 0.5 m/s to know when the car started moving:
                start_index = np.argmax(np.array(state_data)[:, 6] > 0.5)
                time_taken = timestamps[-1] - timestamps[start_index]
                
                variant[variant_name][scenario][trial]["time_taken"] = time_taken
                variant[variant_name][scenario][trial]["path_data"] = path_data
                variant[variant_name][scenario][trial]["state_data"] = state_data
                variant[variant_name][scenario][trial]["expansions_data"] = expansions_data
    from scipy.signal import butter, filtfilt
    # Sampling parameters
    fs = 25.0       # Sampling frequency (Hz)
    cutoff = 0.25    # Desired cutoff frequency (Hz)
    order = 2       # 2nd-order filter

    # Design the Butterworth filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # now, go through the scenarios and trials, and find the ratio of time taken between the two variant_nameeresis values:
    violations_dict = {}
    for variant_name in planner_list:
        print(f"Variant: {variant_name} ======================")
        for scenario in scenarios:
            constraint_violations = []
            time_taken_list = []
            for trial in range(num_trials):
                time_taken = variant[variant_name][scenario][trial]["time_taken"]
                rpy = np.array(variant[variant_name][scenario][trial]["state_data"])[:, 3:6]
                vel = np.array(variant[variant_name][scenario][trial]["state_data"])[:, 6:9]
                acc = np.array(variant[variant_name][scenario][trial]["state_data"])[:, 9:12]
                cr = np.cos(rpy[:,0])
                cp = np.cos(rpy[:,1])
                # for i in range(3):
                #     acc[:, i] = filtfilt(b, a, acc[:, i])
                RI = np.fabs(acc[:, 1]/acc[:, 2])
                vert_acc = np.fabs(acc[:, 2] - 9.81*cr*cp)
                RI_indices = np.where(RI > 0.8)[0]
                vert_indices = np.where(vert_acc > 4.0)[0]
                indices = np.unique(np.concatenate((RI_indices, vert_indices)))
                # if len(indices):
                #     print(f"Scenario: {scenario}, Trial: {trial}, Time Taken: {time_taken:.2f}s, Constraint Violations: {len(indices)}") #, {ay}, {az}, {az_mean}, {np.fabs(vert_acc[indices]).mean()}")
                constraint_violations.append(len(indices)*0.04)
                time_taken_list.append(time_taken)
            mean_constraint_violations = np.mean(constraint_violations)
            conf_constraint_violations = conf(constraint_violations)
            violations_dict[scenario] = {"mean": mean_constraint_violations, "conf": conf_constraint_violations}
            print(f"Constraint Violations: {mean_constraint_violations} +/- {conf_constraint_violations}")
            mean_time_taken = np.mean(time_taken_list)
            conf_time_taken = conf(time_taken_list)
            print(f"Time Taken: {mean_time_taken:.2f} +/- {conf_time_taken:.2f}")

def costmap_planner_vis(path, costmap, elevation_map, resolution_inv, goal, map_size, states=None, state_data=None, variant_name=-1, expansion_counter=0, 
                        global_map=None, global_costmap=None, global_heightmap= None, global_state = None, color_dict=None, global_map_size=None,
                        global_start_goal=None):
    # Normalize costmap to 8-bit grayscale (0-255)
    costmap_color = np.clip(costmap, 0, 255).astype(np.uint8)
    pink = np.array([255, 105, 180], dtype=np.uint8)   # BGR format
    white = np.array([255, 255, 255], dtype=np.uint8)

    # Create an output color image with shape (H, W, 3)
    color_map = np.zeros((costmap_color.shape[0], costmap_color.shape[1], 3), dtype=np.uint8)

    # Use broadcasting to assign colors
    mask_white = costmap_color == 255
    mask_pink = ~mask_white

    color_map[mask_white] = white
    color_map[mask_pink] = pink
    costmap_color = color_map
    # Normalize elevation map to 8-bit and apply colormap
    # vmin = np.min(elevation_map)
    # vmax = np.max(elevation_map)
    elev_norm = np.clip((elevation_map + 4)/8, 0, 1) #np.clip((elevation_map - vmin) / (vmax - vmin), 0, 1)
    elev_uint8 = (elev_norm * 255).astype(np.uint8)
    elev_color = np.stack([elev_uint8]*3, axis=-1)  # shape becomes (H, W, 3)

    # Blend the two images
    alpha = 0.5
    costmap = costmap_color
    costmap[mask_white] = elev_color[mask_white]
    if goal is not None:
        goal_x = np.clip( np.int32((goal[0] * resolution_inv) + map_size//2), 0, map_size - 1)
        goal_y = np.clip( np.int32((goal[1] * resolution_inv) + map_size//2), 0, map_size - 1)
        radius = int(5*resolution_inv)
        cv2.circle(costmap, (goal_x, goal_y), radius, (255, 255, 255), 3)  # hollow circle

    if path is not None:
        car_width_px = int(1.6*resolution_inv)
        path_X = np.clip( np.int32((path[..., 0] * resolution_inv) + map_size//2), 0, map_size - 1)
        path_Y = np.clip( np.int32((path[..., 1] * resolution_inv) + map_size//2), 0, map_size - 1)
        velocity = path[..., 3]
        velocity_norm = velocity / 15
        velocity_color = np.clip((velocity_norm * 255), 0, 255).astype(np.uint8)
        for i in range(len(path_X) - 1):
            cv2.line(costmap, (path_X[i], path_Y[i]), (path_X[i + 1], path_Y[i + 1]), (0, int(velocity_color[i]), 0), car_width_px)

    if states is not None:
        if(len(costmap.shape)<3):
            print_states = states
            x = print_states[:, :, :, 0].flatten()
            y = print_states[:, :, :, 1].flatten()
            X = np.clip( np.array((x * resolution_inv) + map_size//2, dtype=np.int32), 0, map_size - 1)
            Y = np.clip( np.array((y * resolution_inv) + map_size//2, dtype=np.int32), 0, map_size - 1)
            costmap[Y, X] = 0
        else:
            print_states = states
            x = print_states[:, :, :, 0].flatten()
            y = print_states[:, :, :, 1].flatten()
            X = np.clip( np.array((x * resolution_inv) + map_size//2, dtype=np.int32), 0, map_size - 1)
            Y = np.clip( np.array((y * resolution_inv) + map_size//2, dtype=np.int32), 0, map_size - 1)
            costmap[Y, X] = np.array([0, 0, 0])
    if state_data is not None:
        # draw a rectangle using the x,y, theta from state which is at index 0, 1, 5
        x = state_data[0]
        y = state_data[1]
        theta = np.pi - state_data[5]
        # convert to pixel coordinates
        x_px = map_size // 2
        y_px = map_size // 2
        # draw a rectangle with the center at (x_px, y_px) and width and height of 1.6*resolution_inv
        car_width_px = int(2.6 * resolution_inv)
        car_height_px = int(1.6 * resolution_inv)
        half_width = car_width_px // 2
        half_height = car_height_px // 2
        # Calculate the corners of the rectangle
        corners = np.array([
            [x_px - half_width, y_px - half_height],
            [x_px + half_width, y_px - half_height],
            [x_px + half_width, y_px + half_height],
            [x_px - half_width, y_px + half_height]
        ], dtype=np.int32)
        # Rotate the rectangle around the center point
        rotation_matrix = cv2.getRotationMatrix2D((x_px, y_px), np.degrees(theta), 1.0)
        rotated_corners = cv2.transform(np.array([corners]), rotation_matrix)[0]
        # Draw the rotated rectangle on the costmap
        cv2.polylines(costmap, [rotated_corners], isClosed=True, color=(0, 0, 0), thickness=2)
    if global_costmap is not None and global_heightmap is not None and global_state is not None and color_dict is not None:
        if global_map is None:
            # Normalize costmap to 8-bit grayscale (0-255)
            global_costmap_color = np.clip(global_costmap, 0, 255).astype(np.uint8)
            pink = np.array([255, 105, 180], dtype=np.uint8)   # BGR format
            white = np.array([255, 255, 255], dtype=np.uint8)

            # Create an output color image with shape (H, W, 3)
            global_color_map = np.zeros((global_costmap_color.shape[0], global_costmap_color.shape[1], 3), dtype=np.uint8)

            # Use broadcasting to assign colors
            mask_white = global_costmap_color == 255
            mask_pink = ~mask_white

            global_color_map[mask_white] = white
            global_color_map[mask_pink] = pink
            global_costmap_color = global_color_map
            # Normalize elevation map to 8-bit and apply colormap
            # vmin = np.min(elevation_map)
            # vmax = np.max(elevation_map)
            global_elev_norm = np.clip((global_heightmap + 16)/32, 0, 1) #np.clip((elevation_map - vmin) / (vmax - vmin), 0, 1)
            global_elev_uint8 = (global_elev_norm * 255).astype(np.uint8)
            global_elev_color = np.stack([global_elev_uint8]*3, axis=-1)  # shape becomes (H, W, 3)

            # Blend the two images
            global_map = global_costmap_color
            global_map[mask_white] = global_elev_color[mask_white]
            # draw a circle at global goal
            global_goal_x = int(global_map_size//2 + int(global_start_goal[1][0] * resolution_inv))
            global_goal_y = int(global_map_size//2 + int(global_start_goal[1][1] * resolution_inv))
            print(global_goal_x, global_goal_y)
            radius = int(10*resolution_inv)
            cv2.circle(global_map, (global_goal_x, global_goal_y), radius, (255, 255, 255), 5)
        
        x = global_state[0]
        y = global_state[1]
        # convert to pixel coordinates
        x_px = int(global_map_size//2 + int(x * resolution_inv))
        y_px = int(global_map_size//2 + int(y * resolution_inv))
        # draw a circle at x_px y_px using variant color
        radius = int(0.8 * resolution_inv)
        cv2.circle(global_map, (x_px, y_px), radius, color_dict[variant_name], 1)  # hollow circle
        # write the legend for the variant name using color dict:
        global_map = cv2.flip(global_map, 0)  # flip the map vertically for better visualization
        # for i, (name, color) in enumerate(color_dict.items()):
        #     cv2.putText(global_map, name, (10, 100 + i * 100), cv2.FONT_HERSHEY_TRIPLEX, 3.5, color, 5)
        global_map = cv2.flip(global_map, 0)  # flip it back to original orientation
        # text: Red color is 
    # Draw the path on the costmap using cv2.line
    # place text on the costmap, hysteresis and expansion counter:
    # draw an empty circle with black border at the goal location. First convert goal location to pixel.

    # blend costmap and elevation map with 0.6 alpha
    costmap = cv2.flip(costmap, 0)  # this is just for visualization

    if global_costmap is None:
        cv2.putText(costmap, f"Var: {variant_name}", (10, 40), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 1)
        if expansion_counter == 1:
            expansion_counter = "Invalid"
        cv2.putText(costmap, f"Exp: {expansion_counter}", (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 1)
    else:
        cv2.putText(costmap, f"{variant_name}", (10, 40), cv2.FONT_HERSHEY_TRIPLEX, 1.5, color_dict[variant_name], 2)
    if global_costmap is not None:
        return costmap, global_map
    return costmap

def combine_map_trajs(config_path, costmap_vis):
    Config = yaml.safe_load(open(config_path))
    Map_config = Config["Map_config"]
    map_name = Config["map_name"]

    map_res = Map_config["map_res"]
    map_size = Map_config["map_size"]
    map_center = map_size * 0.5
    heightmap = np.load(f"{map_name}_height.npy")
    costmap = np.load(f"{map_name}_cost.npy")
    # bloat the costmap using gaussian blur
    costmap = cv2.GaussianBlur(costmap, (3, 3), 0)
    # # convert the costmap to a binary map
    heightmap = np.clip(heightmap, -50, 100)
    if not os.path.exists("Global_Planning_Waypoints"):
        os.makedirs("Global_Planning_Waypoints")
    costmap = costmap_vis([], costmap, heightmap, 1/map_res, [], int(map_size/map_res), is_list=True)
    costmap = cv2.cvtColor(costmap, cv2.COLOR_BGR2RGB)
    path_list = []
    start_list = set()
    goal_list = []
    deletion_set = []

    fig = plt.figure(figsize=(20, 20))
    for filename in os.listdir("Global_Planning_Waypoints"):
        if filename.endswith(".pkl"):
            with open(os.path.join("Global_Planning_Waypoints", filename), "rb") as f:
                data = pickle.load(f)
            path = data["path"]
            path = np.array(path)
            path = path[::5, :]
            start = tuple(path[0, :2])
            goal = tuple(path[-1, :2])
            plt.plot((map_size*0.5 + path[:, 0])/map_res, (map_size*0.5 + path[:, 1])/map_res, color='black', linewidth=5)
            goal_circle = plt.Circle(((map_size*0.5 + goal[0])/map_res, (map_size*0.5 + goal[1])/map_res),facecolor='gray', edgecolor='black', radius=50, alpha=0.75, zorder=100)
            plt.gca().add_patch(goal_circle)

    plt.imshow(costmap)
    plt.axis('off')
    plt.savefig("demonstration_paths.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()

def benchmark_plot(yaml_file="", output_folder = "", search_resolution = None):
    if yaml_file != "":
        with open(yaml_file, 'r') as file:
            experiment_configs = yaml.safe_load(file)
    pkl_file = experiment_configs["pkl_file"]
    configs = read_configs_from_pickle(pkl_file)
    resolution_list = experiment_configs["resolution_list"]
    results_folder = experiment_configs["results_dir"]
    use_resolution_list = experiment_configs["use_resolution_list"]
    expansion_limit = experiment_configs["max_expansions"]
    level_limit = experiment_configs["max_level"]
    search_res = experiment_configs["search_res"]

    aggregated_results_first_path = defaultdict(lambda: {"expansions": [], "path_cost": []})
    aggregated_results_best_path = defaultdict(lambda: {"expansions": [], "path_cost": [], "max_level": [], "max_expansions": []})
    aggregated_results_terminal_path = defaultdict(lambda: {"expansions": [], "path_cost": [], "max_level": [], "max_expansions": []})
    agg_results = defaultdict(lambda: {"expansions": [], "path_cost": [], "success": []})
    total_counter = 0
    counter = 0
    algorithm_names = [
        "HA*_MR",
        "HA*_LR",
        "IGHA*-10",
        "IGHA*-50",
        "IGHA*-250",
        "IGHA*-1250",
        r"IGHA*-$\infty$",
        "IGHA*-0",        
        "HA*M"
    ]
    expansion_results_first_path = []
    expansion_results_best_path = []
    expansion_results_best_path_cost_to_expansion_ratio = []
    expansion_results_first_path_cost_to_expansion_ratio = []
    coverage_results_first_path = []
    coverage_results_best_path = []
    efficiency_results_first_path = []
    efficiency_results_best_path = []
    max_level = 5
    for algo in algorithm_names:
        expansion_results_first_path.append([[], [], [], [], [], []])
        expansion_results_best_path.append([[], [], [], [], [], []])
        expansion_results_first_path_cost_to_expansion_ratio.append([[], [], [], [], [], []])
        expansion_results_best_path_cost_to_expansion_ratio.append([[], [], [], [], [], []])
        coverage_results_first_path.append([[], [], [], [], [], []])
        coverage_results_best_path.append([[], [], [], [], [], []])
        efficiency_results_first_path.append([[], [], [], [], [], []])
        efficiency_results_best_path.append([[], [], [], [], [], []])

    bottle_neck_ratio = []
    count = 0
    max_level = 3
    total_configs = np.zeros(max_level + 1)
    data = np.zeros((len(algorithm_names), len(algorithm_names), len(total_configs)))
    cost_data = np.zeros((len(algorithm_names), len(algorithm_names), len(total_configs)))
    node_type = None
    data_exists = os.path.exists(os.path.join(output_folder, "aggregated_results_first_path.pkl"))
    if not data_exists:
        iteration = 0
        for map_name, resolution in configs:
            for i, experiment_info in enumerate(configs[(map_name, resolution)]["configs"]):
                if node_type is None:
                    node_type = experiment_info["node_info"]["node_type"]
                if resolution not in use_resolution_list:
                    continue
                    
                try:
                    res_index = int(resolution * 100)
                    search_res_index = int(search_res * 100)

                    file_paths = {
                        r"IGHA*-$\infty$": os.path.join(results_folder, f"{map_name}_{i}_{res_index}_{search_res_index}_dense_rank.pkl"),
                        "HA*_LR": os.path.join(results_folder, f"{map_name}_{i}_{res_index}_{search_res_index}_dense_ResetBlind.pkl"),
                        "IGHA*-0": os.path.join(results_folder, f"{map_name}_{i}_{res_index}_{search_res_index}_denseSparse_rank.pkl"),
                        "IGHA*-10": os.path.join(results_folder, f"{map_name}_{i}_{res_index}_{search_res_index}_denseSparseHysteresisStatic.10_rank.pkl"),
                        "IGHA*-50": os.path.join(results_folder, f"{map_name}_{i}_{res_index}_{search_res_index}_denseSparseHysteresisStatic.50_rank.pkl"),
                        "IGHA*-250": os.path.join(results_folder, f"{map_name}_{i}_{res_index}_{search_res_index}_denseSparseHysteresisStatic.250_rank.pkl"),
                        "IGHA*-1250": os.path.join(results_folder, f"{map_name}_{i}_{res_index}_{search_res_index}_denseSparseHysteresisStatic.1250_rank.pkl"),
                        "HA*_MR": os.path.join(results_folder, f"{map_name}_{i}_{res_index}_{search_res_index}_dense_ResetBlind.pkl"),
                        "HA*M": os.path.join(results_folder, f"{map_name}_{i}_{res_index}_{search_res_index}_dense_Reset.pkl")
                    }

                    try:
                        for file in file_paths:
                            data = load_from_pickle(file_paths[file])
                    except Exception as e:
                        counter += 1
                        # print(traceback.format_exc())
                        continue
                    # print(res_index, search_res_index)
                    results = {}
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_to_key = {executor.submit(load_from_pickle, path): key for key, path in file_paths.items()}
                        for future in concurrent.futures.as_completed(future_to_key):
                            key = future_to_key[future]
                            try:
                                results[key] = future.result()
                            except Exception as e:
                                print(f"Failed to load file for {key}: {e}")
                                pass

                    for algo in algorithm_names:
                        if algo in results and results[algo] is not None:

                            if algo == "HA*-Direct":
                                continue
                            if algo == "HA*_MR":
                                path_cost = results[algo].get("path_cost_list", None)
                                # get unique path cost values:
                                expansions = results[algo].get("expansion_list", None)
                                valid_expansions = [expansions[j] for j in range(len(expansions)) if path_cost[j] != 1e9]

                                path_cost_, indices = np.unique(path_cost, return_index=True)
                                # indices where path_cost_ is not 1e9
                                indices = [i for i in indices if path_cost[i] != 1e9]
                                best_path_cost = path_cost_[0]
                                max_expansions = np.sum(results[algo]["expansion_list"])

                                success = 1 if best_path_cost != 1e9 else 0
                                
                                Direct_expansions = results[algo]["expansion_list"][indices[-1]]
                                first_path_expansions = np.sum(results[algo]["expansion_list"][:indices[-1]+1])
                                first_path_cost = results[algo]["path_cost_list"][indices[-1]]
                                aggregated_results_first_path["HA*-Direct"]["expansions"].append(Direct_expansions/max_expansions)

                                aggregated_results_first_path["HA*-Direct"]["path_cost"].append(first_path_cost/best_path_cost)
                                aggregated_results_best_path["HA*-Direct"]["expansions"].append(Direct_expansions/max_expansions)
                                aggregated_results_best_path["HA*-Direct"]["path_cost"].append(first_path_cost/best_path_cost)
                                aggregated_results_first_path[algo]["expansions"].append(first_path_expansions/max_expansions)
                                aggregated_results_first_path[algo]["path_cost"].append(first_path_cost/best_path_cost)
                                aggregated_results_best_path[algo]["expansions"].append(1)
                                aggregated_results_best_path[algo]["path_cost"].append(1)
                                agg_results[algo]["success"].append(success)
                                normalizing_expansions = max_expansions
                                normalizing_cost = best_path_cost
                                max_level = max(results[algo]["level_information"])

                                aggregated_results_best_path[algo]["max_level"].append(max_level)
                                aggregated_results_best_path[algo]["max_expansions"].append(max_expansions)
                                
                                # print(algo, expansions, path_cost)
                            elif algo == "HA*_LR":

                                expansions = results[algo].get("expansion_list", None)
                                expansions = expansions[0]
                                path_cost = results[algo].get("path_cost_list", None)
                                path_cost = path_cost[0]
                                success = 1 if path_cost != 1e9 else 0
                                # if not success:
                                #     expansions = normalizing_expansions
                                aggregated_results_first_path[algo]["expansions"].append(expansions/normalizing_expansions)
                                aggregated_results_first_path[algo]["path_cost"].append(path_cost/normalizing_cost)
                                aggregated_results_best_path[algo]["expansions"].append(expansions/normalizing_expansions)
                                aggregated_results_best_path[algo]["path_cost"].append(path_cost/normalizing_cost)
                                agg_results[algo]["success"].append(success)
                                # print(algo, expansions, path_cost)
                            else:
                                # first path:
                                expansions = results[algo].get("expansion_list", None)
                                path_cost = results[algo].get("path_cost_list", None)
                                expansions = [expansions[j] for j in range(len(expansions)) if path_cost[j] != 1e9]
                                # print(algo, expansions)
                                unique_path_cost, indices = np.unique(path_cost, return_index=True)

                                if len(expansions) == 0:
                                    success = 0
                                    first_path_expansions = normalizing_expansions
                                    first_path_cost = 1e9
                                    best_path_cost = 1e9
                                    best_path_expansions = normalizing_expansions
                                    aggregated_results_first_path[algo]["expansions"].append(1)
                                    aggregated_results_first_path[algo]["path_cost"].append(1)
                                    aggregated_results_best_path[algo]["expansions"].append(1)
                                    aggregated_results_best_path[algo]["path_cost"].append(1)
                                    agg_results[algo]["success"].append(success)
                                    max_level = max(results[algo]["level_information"])
                                    max_expansions = results[algo]["expansion_list"][-1]
                                    aggregated_results_best_path[algo]["max_level"].append(max_level)
                                    aggregated_results_best_path[algo]["max_expansions"].append(max_expansions)
                                    aggregated_results_terminal_path[algo]["expansions"].append(max_expansions/normalizing_expansions)
                                    aggregated_results_terminal_path[algo]["path_cost"].append(best_path_cost/normalizing_cost)
                                else:
                                    first_path_expansions = expansions[0]
                                    valid_cost_list = [path_cost[i] for i in range(len(path_cost)) if path_cost[i] != 1e9]
                                    first_path_cost = valid_cost_list[0]
                                    success = 1 # since len(expansions) > 0
                                    best_path_cost = unique_path_cost[0]
                                    index = np.argmin(path_cost)
                                    expansions = results[algo].get("expansion_list", None) # reset expansions
                                    best_path_expansions = expansions[-1]

                                    aggregated_results_first_path[algo]["expansions"].append(first_path_expansions/normalizing_expansions)
                                    aggregated_results_first_path[algo]["path_cost"].append(first_path_cost/normalizing_cost)
                                    aggregated_results_best_path[algo]["expansions"].append(best_path_expansions/normalizing_expansions)
                                    aggregated_results_best_path[algo]["path_cost"].append(best_path_cost/normalizing_cost)
                                    agg_results[algo]["success"].append(success)
                                    max_level = max(results[algo]["level_information"])
                                    max_expansions = results[algo]["expansion_list"][-1]
                                    aggregated_results_best_path[algo]["max_level"].append(max_level)
                                    aggregated_results_best_path[algo]["max_expansions"].append(max_expansions)
                                    aggregated_results_terminal_path[algo]["expansions"].append(max_expansions/normalizing_expansions)
                                    aggregated_results_terminal_path[algo]["path_cost"].append(best_path_cost/normalizing_cost)

                        else:
                            pass
                            # print(f"Algorithm {algo} not found in results")
                    a = aggregated_results_first_path["IGHA*-0"]["expansions"][-1]
                    b = aggregated_results_first_path["IGHA*-10"]["expansions"][-1]
                    c = aggregated_results_first_path["IGHA*-50"]["expansions"][-1]
                    d = aggregated_results_first_path[r"IGHA*-$\infty$"]["expansions"][-1]
                    # ratio = b/a
                    first_solution_level = np.argmax(np.array(results[r"IGHA*-$\infty$"]["path_cost_list"]) < 1e8)
                    # if c/d < 0.8 and a/d > 1.1 and d*normalizing_expansions < 1000: # and first_solution_level == 1: # and 1/ratio < 2:
                    #     # if aggregated_results_first_path["DRK"]["expansions"][-1]*normalizing_expansions < 500:
                    #     print(map_name, i, res_index, search_res_index, c/d, a*normalizing_expansions, b*normalizing_expansions, c*normalizing_expansions, d*normalizing_expansions)
                    if a/b < 1/1.2 and b/c < 1/1.2 and c/d < 1/1.2 and d*normalizing_expansions < 2000 and first_solution_level == 1: # and 1/ratio < 2:
                        # if aggregated_results_first_path["DRK"]["expansions"][-1]*normalizing_expansions < 500:
                        print(map_name, i, res_index, search_res_index, c/d, a*normalizing_expansions, b*normalizing_expansions, c*normalizing_expansions, d*normalizing_expansions)
                    # if c/d < 0.8 and a/d > 1.1 and d*normalizing_expansions < 1000: # and first_solution_level == 1: # and 1/ratio < 2:
                    #     # if aggregated_results_first_path["DRK"]["expansions"][-1]*normalizing_expansions < 500:
                    #     print(map_name, i, res_index, search_res_index, c/d, a*normalizing_expansions, b*normalizing_expansions, c*normalizing_expansions, d*normalizing_expansions)
                    # if aggregated_results_first_path["DRKS_50"]["expansions"][-1] < aggregated_results_first_path["DRK"]["expansions"][-1]:
                    #     drk_path_cost_list = results["DRKS_50"].get("path_cost_list", None)
                    #     first_solution_level = np.argmax(np.array(drk_path_cost_list) < 1e8)
                    #     if aggregated_results_first_path["DRKS_50"]["expansions"][-1]*normalizing_expansions < 1000 and first_solution_level == 1:
                    #         print(map_name, i, res_index, search_res_index,aggregated_results_first_path["DRKS_50"]["expansions"][-1]*normalizing_expansions, aggregated_results_first_path["DRK"]["expansions"][-1]/aggregated_results_first_path["DRKS_50"]["expansions"][-1])
                    iteration += 1

                except Exception as e:
                    print(f"Error processing map {map_name}, config {i}: {e}, algo: {algo}")
                    print(indices)
                    print(traceback.format_exc())
                    continue
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        save_to_pickle(dict(aggregated_results_first_path), os.path.join(output_folder, "aggregated_results_first_path.pkl"))
        save_to_pickle(dict(aggregated_results_best_path), os.path.join(output_folder, "aggregated_results_best_path.pkl"))
        save_to_pickle(dict(agg_results), os.path.join(output_folder, "agg_results.pkl"))
    else:
        aggregated_results_first_path = load_from_pickle(os.path.join(output_folder, "aggregated_results_first_path.pkl"))
        aggregated_results_best_path = load_from_pickle(os.path.join(output_folder, "aggregated_results_best_path.pkl"))
        agg_results = load_from_pickle(os.path.join(output_folder, "agg_results.pkl"))
        # guess node type from output folder name:
        if "kinematic" in output_folder:
            node_type = "Kinematic"
        elif "kinodynamic" in output_folder:
            node_type = "Kinodynamic"

    node_type = experiment_configs["experiment_info_default"]["node_info"]["node_type"]
    if node_type == "kinematic":
        node_type = "Kinematic"
    elif node_type == "kinodynamic":
        node_type = "Kinodynamic"
    elif node_type == "simple":
        if "multi_bottleneck" in pkl_file:
            node_type = "Multi-Bottleneck"
        elif "single_bottleneck" in pkl_file:
            node_type = "Single-Bottleneck"
    color_mapping_dictionary = {
        "D+O": "tab:blue",
        "DS+RK": "tab:orange",
        "DSRK": "tab:orange",
        "D+RK": "tab:green",
        "DRK": "tab:green",
        "D+RN": "tab:red",
        "DRN": "tab:red",
        "IHA*": "tab:purple",
        "HA*_MR": "tab:brown",
        "HA*_LR": "tab:pink",
    }
    # success_rate_plot(algorithm_names, agg_results, output_folder, node_type, color_mapping_dictionary)
    # expansions_to_best_solution_plot(["IGHA*-0", "IGHA*-10", "IGHA*-50", "IGHA*-250", "IGHA*-1250", r"IGHA*-$\infty$"], aggregated_results_best_path, agg_results, output_folder, node_type, color_mapping_dictionary)
    # # expansions_to_first_solution_plot(algorithm_names, aggregated_results_first_path, agg_results, output_folder, node_type, color_mapping_dictionary)
    # beat_Gold_solution_plot(algorithm_names, aggregated_results_first_path, agg_results, output_folder, node_type, color_mapping_dictionary)
    # expansion_or_level_limit_hit(algorithm_names, aggregated_results_best_path, agg_results, output_folder, node_type, expansion_limit, level_limit)
    # non_DNF_ratio(algorithm_names, aggregated_results_first_path, output_folder, node_type, color_mapping_dictionary)
    # expansion_or_none(algorithm_names, aggregated_results_best_path, agg_results, output_folder, node_type, expansion_limit, level_limit)
    # custom_rank_plot(["DSRK", "DRK"], aggregated_results_first_path, agg_results, output_folder, node_type, color_mapping_dictionary)
    # energy_expense([], aggregated_results_first_path, aggregated_results_best_path, agg_results, output_folder)
    # custom_rank_plot(["DSRK", "DRK"], aggregated_results_best_path, agg_results, output_folder, node_type, color_mapping_dictionary, best=True)
    # custom_rank_plot(["DSRK", "DSHRK_25", "DSHRK_50", "DSHRK_75", "DRK"], aggregated_results_first_path, agg_results, output_folder, node_type, color_mapping_dictionary)
    # custom_rank_plot(["DRKS_10", "DRKS_25", "DRKS_50", "DRK"], aggregated_results_first_path, agg_results, output_folder, node_type, color_mapping_dictionary)
    # custom_rank_plot(["DSRK", "DSHRK_25", "DSHRK_50", "DSHRK_75", "DRK"], aggregated_results_best_path, agg_results, output_folder, node_type, color_mapping_dictionary, best=True)
    # custom_rank_plot(["DRKS_10", "DRKS_25", "DRKS_50", "DRK"], aggregated_results_best_path, agg_results, output_folder, node_type, color_mapping_dictionary, best=True)
    # custom_rank_plot(["DSRK", "DSH_25_RKS_10", "DSH_50_RKS_10", "DSH_75_RKS_10", "DSH_25_RKS_25", "DSH_50_RKS_25", "DSH_75_RKS_25", "DSH_25_RKS_50", "DSH_50_RKS_50", "DSH_75_RKS_50","DSHRK_25", "DSHRK_50", "DSHRK_75", "DRKS_10", "DRKS_25", "DRKS_50", "DRK","IHA*"], aggregated_results_first_path, agg_results, output_folder, node_type, color_mapping_dictionary)
    # custom_rank_plot(["DSRK", "DSH_25_RKS_10", "DSH_50_RKS_10", "DSH_75_RKS_10", "DSH_25_RKS_25", "DSH_50_RKS_25", "DSH_75_RKS_25", "DSH_25_RKS_50", "DSH_50_RKS_50", "DSH_75_RKS_50","DSHRK_25", "DSHRK_50", "DSHRK_75", "DRKS_10", "DRKS_25", "DRKS_50", "DRK","IHA*"], aggregated_results_best_path, agg_results, output_folder, node_type, color_mapping_dictionary, best=True)
    # custom_rank_plot(["DSRK","DSHRK_25", "DSHRK_50", "DSHRK_75", "DRK", "IHA*"], aggregated_results_first_path, agg_results, output_folder, node_type, color_mapping_dictionary)
    # custom_rank_plot(["DSRK","DSH_25_RKS_25", "IHA*"], aggregated_results_best_path, agg_results, output_folder, node_type, color_mapping_dictionary, best=True)
    # custom_rank_plot(["IGHA*-0", "IGHA*-10", "IGHA*-50", r"IGHA*-$\infty$"], aggregated_results_first_path, agg_results, output_folder, node_type, color_mapping_dictionary)
    # custom_rank_plot(["IGHA*-0", "IGHA*-10", "IGHA*-50", "IGHA*-250", "IGHA*-1250", r"IGHA*-$\infty$", "HA*M"], aggregated_results_best_path, agg_results, output_folder, node_type, color_mapping_dictionary, best=True)
    # avg_expansions(["HA*_MR"], aggregated_results_first_path, aggregated_results_best_path)
    # return expansion_ratio_plot(["IGHA*-0", "IGHA*-10", "IGHA*-50", "IGHA*-250", "IGHA*-1250", r"IGHA*-$\infty$"], expansion_limit, aggregated_results_best_path, agg_results, output_folder, node_type, color_mapping_dictionary, best=True)

def avg_expansions(algorithm_names, aggregated_results_first_path, aggregated_results_best_path):
    # avg expansions per iteration for given algorithm
    for algo in algorithm_names:
        if algo not in aggregated_results_first_path:
            continue
        expansion_list = []
        for i in range(len(aggregated_results_first_path[algo]["expansions"])):
            expansions = aggregated_results_first_path[algo]["expansions"][i]
            max_expansions = aggregated_results_best_path[algo]["max_expansions"][i]
            expansions *= max_expansions
            expansion_list.append(expansions)
        avg_expansions = np.mean(expansion_list)
        print(f"Avg expansions for {algo}: {avg_expansions}")

def expansion_ratio_plot(algorithm_names, expansion_limit, aggregated_results_first_path, agg_results, output_folder, node_type, color_mapping_dictionary, best = False):
    algo_b = "HA*M"

    x_labels = []
    means_clause = []
    conf_intervals_clause = []
    means_no_clause = []
    conf_intervals_no_clause = []
    indices = [i for i in range(len(agg_results["HA*_LR"]["success"])) if agg_results["HA*_LR"]["success"][i] == 0]
    for algo_a in algorithm_names:
        if algo_a == algo_b:
            continue
        A_beats_B_clause = []
        Average_ratio_clause = []
        A_beats_B_no_clause = []
        Average_ratio_no_clause = []

        for i in range(len(aggregated_results_first_path["HA*_LR"]["expansions"])):
            if i not in indices and best == False:
                continue
            if aggregated_results_first_path[algo_b]["expansions"][i]*aggregated_results_first_path["HA*_MR"]["max_expansions"][i] >= expansion_limit:
                A_beats_B_no_clause.append(aggregated_results_first_path[algo_a]["expansions"][i] < aggregated_results_first_path[algo_b]["expansions"][i])
                Average_ratio_no_clause.append(aggregated_results_first_path[algo_b]["expansions"][i] / aggregated_results_first_path[algo_a]["expansions"][i])
            else:
                A_beats_B_clause.append(aggregated_results_first_path[algo_a]["expansions"][i] < aggregated_results_first_path[algo_b]["expansions"][i])
                Average_ratio_clause.append(aggregated_results_first_path[algo_b]["expansions"][i] / aggregated_results_first_path[algo_a]["expansions"][i])
        algo_name = algo_a.split("-")[1]
        x_labels.append(algo_name)

        Average_ratio_clause = np.array(Average_ratio_clause)
        mean_speedup = np.mean(Average_ratio_clause)
        conf_speedup = conf(Average_ratio_clause)
        means_clause.append(mean_speedup)
        conf_intervals_clause.append(conf_speedup)

        Average_ratio_no_clause = np.array(Average_ratio_no_clause)
        mean_speedup = np.mean(Average_ratio_no_clause)
        conf_speedup = conf(Average_ratio_no_clause)
        means_no_clause.append(mean_speedup)
        conf_intervals_no_clause.append(conf_speedup)

    # return a dictionary with the means and confidence intervals and node type
    output = {
        "means_clause": means_clause,
        "conf_intervals_clause": conf_intervals_clause,
        "means_no_clause": means_no_clause,
        "conf_intervals_no_clause": conf_intervals_no_clause,
        "node_type": node_type,
        "algorithm_names": algorithm_names,
        "x_labels": x_labels,
        "best": best,
        "output_folder": output_folder,
        "algo_b": algo_b
    }
    return output

def combine_results(data_list):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    x_labels = data_list[0]["x_labels"]
    best = data_list[0]["best"]
    output_folder = data_list[0]["output_folder"]
    algo_b = data_list[0]["algo_b"]
    line_thickness = 2
    from brokenaxes import brokenaxes
    x_pos = np.arange(len(x_labels))
    plt.figure(figsize=(6,3))
    # adjust figure top and bottom:
    plt.subplots_adjust(top=0.95, bottom=0.15, right = 1.0, left = 0.12)
    bax = brokenaxes(xlims=((-0.5, 0.5), (0.5, 4.5), (4.5, 5.5)))

    num_data = len(data_list)
    format_list = ['o', 's']
    color_list = ["black", "gray"]
    for i, data in enumerate(data_list):
        means_clause = data["means_clause"]
        conf_intervals_clause = data["conf_intervals_clause"]
        means_no_clause = data["means_no_clause"]
        conf_intervals_no_clause = data["conf_intervals_no_clause"]

        node_type = data["node_type"]
        bax.errorbar(x_pos, means_clause, yerr=conf_intervals_clause, fmt='o', color=color_list[i], label=node_type,
                linewidth=line_thickness, capsize=5, markersize=6)
        bax.plot(x_pos, means_clause, color=color_list[i], linewidth=line_thickness)

        bax.errorbar(x_pos, means_no_clause, yerr=conf_intervals_no_clause, fmt='o', color=color_list[i],
                linewidth=line_thickness, capsize=5, markersize=6, linestyle=':')
        bax.plot(x_pos, means_no_clause, color=color_list[i], linewidth=line_thickness, linestyle=':')


    bax.axs[0].set_xticks([0])
    bax.axs[0].set_xticklabels([x_labels[0]])

    bax.axs[1].set_xticks([1, 2, 3, 4])
    bax.axs[1].set_xticklabels(x_labels[1:5])

    bax.axs[2].set_xticks([5])
    bax.axs[2].set_xticklabels([x_labels[5]])

    bax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1)
    bax.set_ylabel(f"Faster (relative to {algo_b})", fontsize=12)
    bax.set_xlabel(r"$H_\mathrm{Thresh}$", fontsize=12)
    bax.legend(loc='lower left', fontsize=12, frameon=False)
    # bax.tight_layout()
    # bax.box(False)
    if best == True:
        # plt.title(f"{node_type} Relative Expansions to Best Path", fontsize=10)
        plt.savefig(os.path.join(output_folder, f"{node_type}_{algo_b}_best_avg_faster.pdf"))
    else:
        # plt.title(f"{node_type} Relative Expansions to First Path", fontsize=10)
        plt.savefig(os.path.join(output_folder, f"{node_type}_{algo_b}_first_avg_faster.pdf"))
    plt.show()

def custom_rank_plot(algorithm_names, aggregated_results_first_path, agg_results, output_folder, node_type, color_mapping_dictionary, best=False):
    success_rates = []
    error_bars = []
    new_names = []
    for algo in algorithm_names:
        if algo == "HA*_MR" or algo == "HA*_LR":
            continue
        new_names.append(algo)
    algorithm_names = new_names

    num_algos = len(algorithm_names)
    data = np.zeros((num_algos, num_algos))
    indices = [i for i in range(len(agg_results["HA*_LR"]["success"])) if agg_results["HA*_LR"]["success"][i] == 0]
    # find rank of each algorithm on the basis of expansions
    total_configs = 0
    for i in range(len(aggregated_results_first_path["HA*_LR"]["expansions"])):
        if i not in indices and best == False:
            continue
        # if all expansions are equal, skip:
        if len(set([aggregated_results_first_path[algo]["expansions"][i] for algo in algorithm_names])) == 1:
            continue
        expansions = {algo: aggregated_results_first_path[algo]["expansions"][i] for algo in algorithm_names}
        sorted_algos = sorted(expansions, key=lambda x: expansions[x])  # Sort by expansions
        # print(sorted_algos, expansions)
        # exit()
        ranks = {algo: rank for rank, algo in enumerate(sorted_algos, start=1)}

        # Increment rank counts
        for algo in sorted_algos:
            data[algorithm_names.index(algo), ranks[algo]-1] += 1

        total_configs += 1

    # aggregated_data = np.sum(data)
    aggregated_data = data / total_configs * 100
    COM_plot = np.zeros_like(aggregated_data)
    # MOI_plot = np.zeros_like(aggregated_data)
    for i in range(len(aggregated_data)):
        distance = np.arange(len(aggregated_data[i]))
        COM = np.sum(aggregated_data[i] * distance) / np.sum(aggregated_data[i])
        MOI = np.sum(aggregated_data[i] * (distance - COM) ** 2) / np.sum(aggregated_data[i])/len(aggregated_data[i])
        COM_plot[i][int(COM)] = MOI

    # exit()
    algorithms = algorithm_names
    ranks = np.arange(len(algorithm_names)) + 1
    print(aggregated_data)
    df = pd.DataFrame(aggregated_data, index=algorithms, columns=ranks)
    # df = df.rename(index={
    #     "DSRK": "IGHA*-0",
    #     "DRK": r"IGHA*-$\infty$"
    # })
    # Create the heatmap
    fig_size = len(algorithm_names) * 1.5
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        df,
        annot=True,  # Show percentage values
        fmt=".1f",  # Format the numbers (1 decimal point)
        cmap="Greys",  # Use a grayscale colormap
        cbar=False,  # Turn off the color bar if not needed
        linewidths=0.5,  # Add grid lines
        linecolor="white",  # Grid line color
        annot_kws={"size": 10},  # Font size of the annotation
        mask=df == 0,  # Mask the zeros
        # vmax=100,
        # vmin=0,
        cbar_kws={"label": "Percentage"}  # Color bar label
    )
    # Move rank (X-axis) labels to the top and set X-axis label on top
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')
    plt.gca().set_aspect('equal')
    # Add labels and title
    extra = "Best" if best else "First"
    plt.xlabel(f"Rank for Expansions to {extra} Path")
    plt.ylabel("Algorithm")
    # plt.title(f"{node_type} Rank for Expansions to First Path")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{node_type}_rank_{extra}_path_aggregate.pdf"))
    plt.show()

    algorithms = algorithm_names
    ranks = np.arange(len(algorithm_names)) + 1
    # print(aggregated_data)
    df = pd.DataFrame(COM_plot, index=algorithms, columns=ranks)
    # Create the heatmap
    df_COM = pd.DataFrame(COM_plot, index=algorithms, columns=ranks)
    # df_COM = df_COM.rename(index={
    #     "DSRK": "IGHA*-0",
    #     "DRK": r"IGHA*-$\infty$"
    # })
    # df_MOI = pd.DataFrame(MOI_plot, index=algorithms, columns=ranks)

    fig_size = len(algorithm_names) * 1.5
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        df_COM,
        annot=True,  # Show percentage values
        fmt=".1f",  # Format the numbers (1 decimal point)
        cmap="Greys",  # Use a grayscale colormap
        cbar=False,  # Turn off the color bar if not needed
        linewidths=0.5,  # Add grid lines
        linecolor="white",  # Grid line color
        annot_kws={"size": 10},  # Font size of the annotation
        mask=df == 0,  # Mask the zeros
        # vmax=100,
        # vmin=0,
        cbar_kws={"label": "Percentage"}  # Color bar label
    )
    # Move rank (X-axis) labels to the top and set X-axis label on top
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')
    plt.gca().set_aspect('equal')
    # Add labels and title
    extra = "Best" if best else "First"
    plt.xlabel(f"Rank for Expansions to {extra} Path")
    plt.ylabel("Algorithm")
    # plt.title(f"{node_type} Rank for Expansions to First Path")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{node_type}_rank_{extra}_path_mean_variance.pdf"))
    plt.show()

    # tell me how much does A beat B with confidence interval:
    for algo_a in algorithm_names:
        for algo_b in algorithm_names:
            if algo_a == algo_b:
                continue
            A_beats_B = []
            A_B_ratio = []
            Average_ratio = []
            inv_Average_ratio = []
            for i in range(len(aggregated_results_first_path["HA*_LR"]["expansions"])):
                if i not in indices and best == False:
                    continue
                A_beats_B.append(aggregated_results_first_path[algo_a]["expansions"][i] < aggregated_results_first_path[algo_b]["expansions"][i])
                if A_beats_B[-1] == 1:
                    A_B_ratio.append(aggregated_results_first_path[algo_b]["expansions"][i]/aggregated_results_first_path[algo_a]["expansions"][i])
                Average_ratio.append(aggregated_results_first_path[algo_b]["expansions"][i]/aggregated_results_first_path[algo_a]["expansions"][i])
                inv_Average_ratio.append(aggregated_results_first_path[algo_a]["expansions"][i]/aggregated_results_first_path[algo_b]["expansions"][i])
            print(len(A_beats_B))
            Average_ratio = np.array(Average_ratio)
            phat, error, _ = bernoulli_confidence_test(A_beats_B)
            print(f"{algo_a} beats {algo_b} by {phat} \pm {error}")
            print(f"{algo_a} is faster than {algo_b} by {np.mean(A_B_ratio)} \pm {conf(A_B_ratio)}")
            print(f"{algo_a} is on average faster than {algo_b} by {np.mean(Average_ratio):.2f} \pm {conf(Average_ratio)}")

    fig_size = 4
    line_thickness = 2.0  # <-- Adjust this to your preferred thickness
    algo_b = "HA*M"

    x_labels = []
    means = []
    conf_intervals = []

    for algo_a in algorithm_names:
        if algo_a == algo_b:
            continue
        A_beats_B = []
        A_B_ratio = []
        Average_ratio = []

        for i in range(len(aggregated_results_first_path["HA*_LR"]["expansions"])):
            if i not in indices and best == False:
                continue
            A_beats_B.append(aggregated_results_first_path[algo_a]["expansions"][i] < aggregated_results_first_path[algo_b]["expansions"][i])
            if A_beats_B[-1]:
                A_B_ratio.append(aggregated_results_first_path[algo_b]["expansions"][i] / aggregated_results_first_path[algo_a]["expansions"][i])
            Average_ratio.append(aggregated_results_first_path[algo_b]["expansions"][i] / aggregated_results_first_path[algo_a]["expansions"][i])

        Average_ratio = np.array(Average_ratio)
        mean_speedup = np.mean(Average_ratio)
        conf_speedup = conf(Average_ratio)
        algo_name = algo_a.split("-")[1]
        x_labels.append(algo_name)
        means.append(mean_speedup)
        conf_intervals.append(conf_speedup)

    from brokenaxes import brokenaxes
    x_pos = np.arange(len(x_labels))
    plt.figure(figsize=(6,4))
    # adjust figure top and bottom:
    plt.subplots_adjust(top=0.9, bottom=0.1)
    bax = brokenaxes(xlims=((-0.5, 0.5), (0.5, 4.5), (4.5, 5.5)))
    bax.errorbar(x_pos, means, yerr=conf_intervals, fmt='o', color='black',
                linewidth=line_thickness, capsize=5, markersize=5)
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "Helvetica"
    # })
    # Set ticks + labels per subplot
    bax.axs[0].set_xticks([0])
    bax.axs[0].set_xticklabels([x_labels[0]])

    bax.axs[1].set_xticks([1, 2, 3, 4])
    bax.axs[1].set_xticklabels(x_labels[1:5])

    bax.axs[2].set_xticks([5])
    bax.axs[2].set_xticklabels([x_labels[5]])

    bax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1)
    bax.set_ylabel(f"Faster (relative to {algo_b})", fontsize=10)
    bax.set_xlabel(r"$H_\mathrm{Thresh}$", fontsize=10)

    # bax.tight_layout()
    # bax.box(False)
    if best == True:
        # plt.title(f"{node_type} Relative Expansions to Best Path", fontsize=10)
        plt.savefig(os.path.join(output_folder, f"{node_type}_{algo_b}_best_avg_faster.pdf"))
    else:
        # plt.title(f"{node_type} Relative Expansions to First Path", fontsize=10)
        plt.savefig(os.path.join(output_folder, f"{node_type}_{algo_b}_first_avg_faster.pdf"))
    plt.show()

    algo_ranks = {}
    for algo in algorithm_names:
        algo_ranks[algo] = []

    for i in range(len(aggregated_results_first_path["HA*_LR"]["expansions"])):
        if i not in indices and best == False:
            continue
        # if all expansions are equal, skip:
        if len(set([aggregated_results_first_path[algo]["expansions"][i] for algo in algorithm_names])) == 1:
            continue
        expansions = {algo: aggregated_results_first_path[algo]["expansions"][i] for algo in algorithm_names}
        sorted_algos = sorted(expansions, key=lambda x: expansions[x])  # Sort by expansions
        # print(sorted_algos, expansions)
        # exit()
        ranks = {algo: rank for rank, algo in enumerate(sorted_algos, start=1)}

        # Increment rank counts
        for algo in sorted_algos:
            algo_ranks[algo].append(ranks[algo])

        total_configs += 1
    avg_ranks = []
    rank_errors = []

    for algo in algorithm_names:
        ranks = np.array(algo_ranks[algo])
        avg_ranks.append(np.mean(ranks))
        rank_errors.append(conf(ranks))
        
    plt.figure(figsize=(fig_size * 1.5, fig_size))
    x_pos = np.arange(len(algorithm_names))

    plt.errorbar(x_pos, avg_ranks, yerr=rank_errors,
                fmt='o', color='black', linewidth=2.0,
                capsize=5, markersize=5)

    algorithm_names = [algo.replace("DSRK", "IGHA*-0").replace("DRK", r"IGHA*-$\infty$") for algo in algorithm_names]
    plt.xticks(ticks=x_pos, labels=algorithm_names, rotation=45, ha='right')
    plt.ylabel("Average Rank (lower is better)")
    plt.ylim(bottom=1)  # Ranks start at 1

    # Minimalism
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{node_type}_rank_mean_conf.pdf"))
    plt.show()