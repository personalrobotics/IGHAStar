# BeamNG Example Integration

This folder contains an example integration of the IGHA* planner with the BeamNG simulator, using the [BeamNGRL](https://github.com/prl-mushr/BeamNGRL/tree/devel) framework.

** disclaimer **: The actual performance of the MPC may vary on your system; you may need to tweak the parameters for the sampling scheme a little bit. The values used here are the values that worked on our system when running in lock-step with the simulator.

## What does this example do?

This example demonstrates how to use the IGHA* global planner and MPPI controller to drive a simulated vehicle in BeamNG. The system plans a path using a costmap generated from BEV (bird's-eye view) map data, then tracks that path in closed-loop with a learned or analytical vehicle model.

## Prerequisites

- **BeamNGRL** must be installed separately. You must use the `devel` branch, as the TCUDA dynamics model is only available there.
- **Follow the installation instructions on the [BeamNGRL repository](https://github.com/prl-mushr/BeamNGRL/tree/devel)** to set up BeamNGRL and its dependencies.
- Other dependencies are listed in the root `requirements.txt`.

## Running the Example

1. **Configure your test case:**
   - Edit `examples/BeamNG/Configs/example.yaml` to set planner, vehicle, and scenario parameters.
   - The `scenarios` field (a list of scenario names) determines which test case(s) will be run. Each scenario should have a corresponding waypoint file in `examples/BeamNG/Waypoints/` (e.g., `0.pkl`).

2. **Run the example:**
   ```bash
   cd examples/BeamNG
   python3 example.py --remote True --host_IP <your_beamng_host_ip>
   ```
   - You can override the config or waypoint folder using command-line arguments.
   - You can find the rest of the flags in the example.py's main function (for passing in a different config or waypoint folder).

## Configuration File Structure

The main configuration file is `examples/BeamNG/Configs/example.yaml`. Key sections:

- **MPPI_config**: Parameters for the MPPI controller (number of rollouts, time steps, etc.).
- **Map_config**: Map size, resolution, and BEV map settings.
- **Dynamics_config**: Vehicle model parameters (wheelbase, steering limits, etc.).
- **Sampling_config**: Noise and sampling settings for the controller.
- **Planner_config**: Parameters for the IGHA* planner, including the main `experiment_info_default` block (resolution, epsilon, max_expansions, etc.).
- **Cost_config**: Weights and thresholds for the cost function used in planning and control.
- **RPS_config**: Parameters for the steering limiter and rollover prevention.
- **scenarios**: List of scenario names to run (each must have a corresponding waypoint file).
- **start_pos/start_quat**: Default vehicle start pose (can be overridden by scenario).
- **vehicle**: Vehicle make and model.

## Adding or Changing a Scenario

- To run a different scenario, add its name to the `scenarios` list in `example.yaml`.
- Each scenario name (e.g., `0`) must have a corresponding waypoint file (e.g., `0.pkl`) in the `Waypoints/` folder.
- **To generate a new waypoint file:**
  - You can use a separate script or tool to create a list of waypoints (as a path) and save it as a `.pkl` file (Python pickle format). The file should contain a dictionary with a `"path"` key mapping to a list/array of waypoints.
  - The format is typically: `{ "path": np.ndarray of shape (N, 3 or 4) }` where each row is `[x, y, heading, (optional speed)]`.
- The scenario name in the config should match the filename (without `.pkl`).

## Code Overview (Planner Usage)

- The main entry point is `example.py`.
- The script loads configuration files and initializes the BeamNG interface, planner, and controller.
- For each scenario and hysteresis value:
  1. Loads the scenario waypoints and resets the simulation environment.
  2. In a main loop:
     - Polls the simulator for the current vehicle state.
     - Updates the goal using the current position and the loaded waypoints.
     - Generates a costmap from BEV map data.
     - Calls the IGHA* planner (`IGHAStarMP`) to update the plan based on the current state, goal, and costmap. It is called IGHAStarMP because we run the planner in parallel using torch Multi-processing
     - Transforms the planned path to robot-centric coordinates and feeds it to the MPPI controller.
     - The controller computes control actions, which are sent to the simulator.
     - Visualization is updated in real time.
     - The loop continues until the goal is reached or the time limit is exceeded.
- Planner and controller parameters can be tuned in the YAML config.

## Troubleshooting & Tips

- **Missing waypoint file:** If you get an error about a missing waypoint file, make sure the scenario name matches a `.pkl` file in the `Waypoints/` folder.
- **BeamNG connection issues:** Ensure BeamNG is running and accessible at the IP address you provide. The `--remote` and `--host_IP` arguments must match your setup.
- **Vehicle drops/falls through map:** Make sure the Z value in `start_pos` or the waypoint file is correct for your map. The car should start just above the ground.
- **No visualization:** The script uses OpenCV for visualization. Make sure you have a display available, or run with X forwarding if using SSH.
- **Parameter tuning:** If the vehicle is not following the path well, try adjusting the cost weights, planner resolution, or controller noise parameters in the config.

---

For more details on BeamNGRL, see the [BeamNGRL documentation](https://github.com/prl-mushr/BeamNGRL/tree/devel). 