# IGHAStar ROS Example

This folder contains a ROS-based example for the IGHA* path planner. The code here is really an example script to demonstrate how one would use the planner in the loop on their robot. 
It demonstrates how to integrate the IGHA* planner with ROS topics for maps, odometry, waypoints, and visualization.
For simplicity (not needing to fiddle with rviz), there is a built in visualization in the example.

## TODO: does one need to install all grid-map stuff to get grid-map visuals? Might be useful to get rviz visualizations here.


## Prerequisites
- ROS Noetic
- Python 3
- [catkin_tools](https://catkin-tools.readthedocs.io/en/latest/)
- [PyTorch](https://pytorch.org/)
- OpenCV (`cv2`)
- numpy, yaml
- IGHAStar C++/CUDA requirements (requirements.txt)
- ROS messages: `nav_msgs`, `geometry_msgs`, `mavros_msgs`, `grid_map_msgs`, `visualization_msgs`, `diagnostic_msgs`

## Setup (confirm these instructions)
1. **Clone this repository** into your catkin workspace:
   ```bash
   cd ~/catkin_ws/src
   git clone <this-repo>
   cd ~/catkin_ws
   catkin_make
   source devel/setup.bash
   ```
2. **Install Python dependencies:**
   ```bash
   pip install torch opencv-python numpy pyyaml
   ```
3. **Check ROS dependencies:**
   ```bash
   rosdep install --from-paths src --ignore-src -r -y
   ```

## Configuration
- Edit `Configs/ros_example.yml` to set planner parameters, topic names, and costmap settings.

## Running the Example


<figure align="center">
  <img src="../../Content/ROS/ros_example.png" alt="ROS Example Visualization" width="600"/>
  <figcaption><b>Fig. 1:</b> Example ROS planner visualization. The window shows the costmap (pink/white), elevation map (grayscale), the planned path (green), the goal (white circle), and the robot state (black rectangle).</figcaption>
</figure>

We provide rosbags that you can use to run the example script. Place them in the `rosbags` folder. Download here: [rosbag 1](https://drive.google.com/file/d/1zV0f3NbPuyewwbHUlrcuboj-PMrPixwG/view?usp=sharing), [rosbag 2](https://drive.google.com/file/d/1BPsypv83_W5EtcodyV2W75BSK3rVE3mX/view?usp=sharing)

1. **Start ROS core:**
   ```bash
   roscore
   ```
2. **Run the planner node:**
   ```bash
   cd examples/ROS
   python3 example.py
   ```
   Or launch with rosrun/roslaunch if you have a package setup.
3. **Play a rosbag or run your robot simulation** to publish the required topics.
   ```
   cd examples/ROS/rosbags/
   rosbag play hound_95.bag
   ```

## Visualization
- The planner will open an OpenCV window showing the costmap, elevation map, path, goal, and robot state.
- Path and goal are also published as ROS topics for use in RViz.

