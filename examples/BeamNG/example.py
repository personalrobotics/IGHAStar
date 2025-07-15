#!/usr/bin/env python3
import numpy as np
import torch
from BeamNGRL.BeamNG.beamng_interface import *
from BeamNGRL.control.UW_mppi.MPPI import MPPI
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarDynamicsTCUDA import SimpleCarDynamics
from TrackingCost import SimpleCarCost
from BeamNGRL.control.UW_mppi.Sampling.Delta_Sampling import Delta_Sampling
import yaml
import cv2
import argparse
import sys
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR / "scripts"))
from utils import (
    generate_costmap_from_BEVmap,
    convert_global_path_to_bng,
    update_goal,
    steering_limiter,
    PlannerVis,
)
from IGHAStarMP import IGHAStarMP
import pickle

torch.manual_seed(0)


def main(
    config_path=None,
    hal_config_path=None,
    waypoint_folder=None,
    output_folder=None,
    args=None,
):
    if config_path is None:
        print("no config file provided!")
        exit()
    if hal_config_path is None:
        print("no hal config file provided!")
        exit()

    if waypoint_folder is None:
        print("no waypoint folder provided!")
        exit()

    with open(config_path) as f:
        Config = yaml.safe_load(f)
    with open(hal_config_path) as f:
        hal_Config = yaml.safe_load(f)

    Dynamics_config = Config["Dynamics_config"]
    Cost_config = Config["Cost_config"]
    Sampling_config = Config["Sampling_config"]
    MPPI_config = Config["MPPI_config"]
    Map_config = Config["Map_config"]
    vehicle = Config["vehicle"]
    start_pos = np.array(
        Config["start_pos"]
    )  # Default start position, may be overwritten by scenario file
    start_quat = np.array(Config["start_quat"])
    map_res = Map_config["map_res"]
    map_size = Map_config["map_size"]

    # Check that all scenario waypoint files exist and load them
    scenario_waypoints = {}
    for scenario in Config["scenarios"]:
        WP_file = waypoint_folder + scenario + ".pkl"
        if not os.path.isfile(WP_file):
            raise ValueError(f"Waypoint file for scenario {scenario} does not exist")
        scenario_waypoints[scenario] = pickle.load(open(WP_file, "rb"))

    dtype = torch.float
    device = torch.device("cuda")
    # initialize the beamng interface
    bng_interface = get_beamng_default(
        car_model=vehicle["model"],
        start_pos=start_pos,
        start_quat=start_quat,
        car_make=vehicle["make"],
        map_config=Map_config,
        host_IP=args.host_IP,
        remote=args.remote,
        camera_config=hal_Config["camera"],
        lidar_config=hal_Config["lidar"],
        accel_config=hal_Config["mavros"],
        burn_time=Config["burn_time"],
        run_lockstep=Config["run_lockstep"],
    )
    bng_interface.smooth_map = Config["smooth_map"]

    # initialize the planner visualization -- this updates the visualization in a separate thread
    plan_vis = PlannerVis(int(map_size / map_res), 1 / map_res)

    try:
        for hysteresis in Config["hysteresis"]:
            costs = SimpleCarCost(Cost_config, Map_config, device=device)
            sampling = Delta_Sampling(Sampling_config, MPPI_config, device=device)
            dynamics = SimpleCarDynamics(
                Dynamics_config, Map_config, MPPI_config, device=device
            )
            controller = MPPI(dynamics, costs, sampling, MPPI_config, device)
            for scenario in Config["scenarios"]:
                data = scenario_waypoints[scenario]
                target_WP = convert_global_path_to_bng(
                    bng_interface=bng_interface,
                    path=data["path"],
                    Map_config=Map_config,
                )
                # Convert loaded waypoints to BeamNG coordinates

                time_limit = Config["time_limit"][0]
                lookahead = Config["lookahead"][0]
                wp_radius = Config["wp_radius"][0]

                bng_interface.reset(
                    start_pos=target_WP[0, :3], start_quat=target_WP[0, 3:]
                )
                current_wp_index = 0
                goal = None
                action = np.zeros(2)
                controller.reset()
                success = False

                last_reset_time = bng_interface.timestamp
                ts = bng_interface.timestamp - last_reset_time

                bng_interface.state_poll()
                cooldown_timer = 0

                first_path = False
                expansion_limit = Config["Planner_config"]["experiment_info_default"][
                    "max_expansions"
                ]
                planner = IGHAStarMP(Config["Planner_config"])
                time.sleep(2)
                planner.reset()

                while ts < time_limit:
                    bng_interface.state_poll()

                    state = np.copy(bng_interface.state)
                    ts = bng_interface.timestamp - last_reset_time
                    # Get car position in world frame
                    pos = np.copy(state[:2])

                    goal, success, current_wp_index = update_goal(
                        goal,
                        pos,
                        target_WP,
                        current_wp_index,
                        lookahead,
                        wp_radius=wp_radius,
                    )

                    BEV_heght = torch.from_numpy(bng_interface.BEV_heght).to(
                        dtype=dtype, device=device
                    )
                    BEV_normal = torch.from_numpy(bng_interface.BEV_normal).to(
                        dtype=dtype, device=device
                    )
                    costmap = generate_costmap_from_BEVmap(
                        bng_interface.BEV_lethal, bng_interface.BEV_normal
                    )

                    ## Planner code
                    _, path, expansions = planner.update()
                    planner.set_query(
                        pos,
                        state,
                        goal,
                        costmap,
                        bng_interface.BEV_heght,
                        hysteresis,
                        expansion_limit,
                    )
                    if path is None:
                        if not first_path:
                            expansion_limit = min(
                                expansion_limit * 2,
                                Config["Planner_config"]["unstuck_expansions"],
                            )
                        continue
                    else:
                        first_path = True
                        expansion_limit = Config["Planner_config"][
                            "experiment_info_default"
                        ]["max_expansions"]
                    # Transform path to robot-centric coordinates
                    controller_path = np.copy(path)
                    controller_path[:, :2] -= np.copy(pos)
                    reference_index = np.argmin(
                        np.linalg.norm(controller_path[:, :2], axis=1)
                    )
                    # handle the case where the path is too short
                    if (
                        reference_index
                        < len(controller_path) - MPPI_config["TIMESTEPS"]
                    ):
                        reference_path = controller_path[
                            reference_index : reference_index
                            + MPPI_config["TIMESTEPS"],
                            :4,
                        ]
                    else:
                        reference_path = np.zeros((MPPI_config["TIMESTEPS"], 4))
                        available = controller_path[reference_index:, :4]
                        reference_path[: len(available)] = available
                        reference_path[len(available) :, :3] = reference_path[
                            len(available) - 1, :3
                        ]
                        reference_path[len(available) :, 3] = 0.0
                        if reference_index >= len(controller_path) - 10:
                            print("stopping")
                            action *= 0
                            bng_interface.send_ctrl(
                                action,
                                speed_ctrl=True,
                                speed_max=Dynamics_config["throttle_to_wheelspeed"],
                                Kp=2,
                                Ki=1,
                                Kd=0.0,
                                FF_gain=0.0,
                            )
                    # convert the reference path to a tensor
                    reference_path = torch.from_numpy(reference_path).to(
                        device=device, dtype=dtype
                    )

                    ## Controller initialization
                    controller.Dynamics.set_BEV(BEV_heght, BEV_normal)
                    controller.Costs.set_BEV(
                        bng_interface.BEV_heght,
                        bng_interface.BEV_normal,
                        torch.from_numpy(costmap).to(dtype=dtype, device=device),
                    )
                    controller.Costs.set_path(reference_path)
                    state_to_ctrl = np.copy(state)
                    state_to_ctrl[:3] = np.zeros(3)
                    # Previous control output is used as input for next cycle
                    state_to_ctrl[15:17] = action  # Wheelspeed hack
                    ## Controller forward pass
                    action = np.array(
                        controller.forward(
                            torch.from_numpy(state_to_ctrl).to(
                                device=device, dtype=dtype
                            )
                        )
                        .cpu()
                        .numpy(),
                        dtype=np.float64,
                    )[0]
                    # Clamp actions for safety
                    action[1] = np.clip(
                        action[1],
                        Sampling_config["min_thr"],
                        Sampling_config["max_thr"],
                    )
                    action[0] = steering_limiter(action[0], state, Config["RPS_config"])
                    # the following code block is to back the car out of a bad state.
                    if (
                        controller.Costs.constraint_violation
                    ):  # check constraint violation.
                        action = np.zeros(2)
                        if state[6] > 0.5:
                            controller.reset()
                        cooldown_timer = int(
                            0.5 / Config["burn_time"]
                        )  # set the car up to drive backwards for 0.5 seconds.
                    if cooldown_timer > 0:
                        action = np.zeros(2)
                        action[1] = -0.2  # Reverse to recover from bad state
                        cooldown_timer -= 1
                        bng_interface.send_ctrl(
                            action,
                            speed_ctrl=True,
                            speed_max=Dynamics_config["throttle_to_wheelspeed"],
                            Kp=10,
                            Ki=1,
                            Kd=0.0,
                            FF_gain=np.fabs(np.sin(state[4])),
                        )
                        continue

                    if success:  # halt the system
                        print("Reached goal!")
                        action *= 0
                        bng_interface.send_ctrl(
                            action,
                            speed_ctrl=True,
                            speed_max=Dynamics_config["throttle_to_wheelspeed"],
                            Kp=2,
                            Ki=1,
                            Kd=0.0,
                            FF_gain=0.0,
                        )
                        break
                    bng_interface.send_ctrl(
                        action,
                        speed_ctrl=True,
                        speed_max=Dynamics_config["throttle_to_wheelspeed"],
                        Kp=4,
                        Ki=0.5,
                        Kd=0.0,
                        FF_gain=np.fabs(np.sin(state[4])),
                    )

                    # Visualization update
                    _, indices = torch.topk(
                        controller.Sampling.cost_total, k=10, largest=False
                    )
                    controller_states = (
                        controller.Dynamics.states[:, indices, ...].cpu().numpy()
                    )
                    plan_vis.update_vis(
                        controller_states,
                        controller_path,
                        costmap,
                        bng_interface.BEV_heght,
                        1 / map_res,
                        goal - state[:2],
                        expansions,
                        hysteresis,
                    )

                planner.shutdown()
                del planner

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(traceback.format_exc())
        if Config["run_lockstep"]:
            bng_interface.bng.close()
        cv2.destroyAllWindows()
        os._exit(1)
    if Config["run_lockstep"]:
        bng_interface.bng.close()
    cv2.destroyAllWindows()
    os._exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str,
        default="example.yaml",
        help="name of the config file to use",
    )
    parser.add_argument(
        "--hal_config_name",
        type=str,
        default="offroad.yaml",
        help="name of the config file to use",
    )
    parser.add_argument(
        "--remote",
        type=bool,
        default=True,
        help="whether to connect to a remote beamng server",
    )
    parser.add_argument(
        "--host_IP",
        type=str,
        default="169.254.216.9",
        help="host ip address if using remote beamng",
    )
    parser.add_argument(
        "--waypoint_folder",
        type=str,
        default="/examples/BeamNG/Waypoints/",
        help="folder containing the waypoint files",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="/examples/BeamNG/Results/example/",
        help="folder containing the output",
    )

    args = parser.parse_args()
    config_name = args.config_name
    config_path = str(Path(os.getcwd())) + "/examples/BeamNG/Configs/" + config_name

    hal_config_name = args.hal_config_name
    hal_config_path = (
        str(Path(os.getcwd())) + "/examples/BeamNG/Configs/" + hal_config_name
    )

    waypoint_folder = str(Path(os.getcwd())) + args.waypoint_folder
    output_folder = str(Path(os.getcwd())) + args.output_folder

    with torch.no_grad():
        main(
            config_path=config_path,
            hal_config_path=hal_config_path,
            waypoint_folder=waypoint_folder,
            output_folder=output_folder,
            args=args,
        )
