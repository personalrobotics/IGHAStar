# IGHAStar

## TODO: add a nice description of what the planner is about

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ighastar.git
cd ighastar
```

### 2. Set Up Conda Environment
```bash
conda env create -f ighastar.yml
conda activate ighastar
```

The project uses dynamic compilation, so C++ extensions will be compiled automatically when you run the examples.

## Usage

### Running the Standalone Example

The main example script supports different planning modes through configuration files:

```bash
# Use default kinematic configuration
python examples/standalone_example.py

# Specify a different configuration file
python examples/standalone_example.py --config examples/Configs/kinodynamic_example.yml

# Use simple planning configuration
python examples/standalone_example.py --config examples/Configs/simple_example.yml

# Use a specific test case
python examples/standalone_example.py --config examples/Configs/kinematic_example.yml --test-case case2
```

### ROS Integration Example

For ROS integration, a minimal ROS example is provided:

```bash
python examples/ros_kinodynamic_example.py
```

This example:
1. Subscribes to `/odom` for vehicle state
2. Subscribes to `/elevation_map` for terrain data
3. Publishes planned paths to `/ighastar/path`
4. Publishes goal markers to `/ighastar/goal`
5. Uses kinodynamic planning with real-time map updates

**Required ROS Topics:**
- **Input**: `/odom` (nav_msgs/Odometry) - Vehicle pose and velocity
- **Input**: `/elevation_map` (grid_map_msgs/GridMap) - Terrain elevation map
- **Output**: `/ighastar/path` (nav_msgs/Path) - Planned trajectory
- **Output**: `/ighastar/goal` (visualization_msgs/MarkerArray) - Goal visualization

**Configuration**: The ROS example uses `examples/Configs/ros_kinodynamic_example.yml` for planner parameters.

### Configuration

Configuration files are located in `examples/Configs/` and include:

- **Vehicle Parameters**: Length, width, maximum velocity, steering limits
- **Planning Parameters**: Resolution, tolerance, epsilon values
- **Environment Settings**: Map resolution, timesteps, control discretization

#### Available Configuration Files
- `kinematic_example.yml` - Kinematic planning configuration
- `kinodynamic_example.yml` - Kinodynamic planning configuration  
- `simple_example.yml` - Simple planning configuration
- `ros_kinodynamic_example.yml` - ROS integration configuration

#### Modifying Start and Goal Points

To change the start and goal positions for path planning, you can either:

1. **Edit the configuration file** to modify the default start/goal or add test cases
2. **Use the `--test-case` argument** to select from predefined test cases

**Configuration file structure:**

**For Kinematic Planning:**
```yaml
map:
  name: "Berlin_0_1024.png"
  dir: "Maps/street-png"
  start: [94.5, 19.5, 2.8284062641509644]  # [x, y, heading]
  goal: [38.7, 81.6, 0.6324707282184407]   # [x, y, heading]
  size: [1024, 1024]
  res: 0.1
  test_cases:
    case1:
      start: [94.5, 19.5, 2.8284062641509644]
      goal: [38.7, 81.6, 0.6324707282184407]
    case2:
      start: [17.6, 72.0, 0.9402905929256757]
      goal: [40.3, 16.9, 1.0911003058968491]
```

**For Kinodynamic Planning:**
```yaml
map:
  name: "race-2"
  dir: "Maps/Offroad"
  start: [5.6, 8.9, 1.5844149127199794, 4.935323678240241, 0]  # [x, y, heading, velocity, unused]
  goal: [21.1, 46.1, 0.608009209539623, 4.516091247319389, 0] # [x, y, heading, velocity, unused]
  size: [512, 512]
  res: 0.1
  test_cases:
    case1:
      start: [5.6, 8.9, 1.5844149127199794, 4.935323678240241, 0]
      goal: [21.1, 46.1, 0.608009209539623, 4.516091247319389, 0]
    case2:
      start: [11.9, 16.8, 1.3574384016819936, 3.317266656251059, 0]
      goal: [46.6, 25.0, 0.19076360687480998, 3.2507584385704855, 0]
```

**Coordinate System:**
- **x, y**: Position coordinates in meters
- **heading**: Orientation in radians (0 = east, π/2 = north, π = west, -π/2 = south)
- **velocity**: Speed in m/s (kinodynamic only)

### Map Files

Map files are stored in `examples/Maps/` with the following structure:
- `generated_maps/` - Procedurally generated test maps
- `street-png/` - Street network maps
- `Offroad/` - Off-road terrain maps

### Compute Environment Selection

The system automatically detects CUDA availability:
- If CUDA is available: Uses GPU-accelerated planning
- If CUDA is not available: Falls back to CPU implementation

## Project Structure

```
ighastar/
├── src/                    # Core C++/CUDA source files
│   ├── *.cu              # CUDA implementations
│   ├── *_cpu.cpp         # CPU implementations  
│   ├── *.h               # Header files
│   └── *.cpp             # C++ implementations
├── examples/              # Example scripts and configurations
│   ├── Configs/          # Configuration files
│   ├── Maps/             # Map files
│   └── *.py              # Example scripts
├── scripts/              # Utility scripts
├── ighastar.yml         # Conda environment file
└── README.md            # This file
```

## Features

- **Hierarchical A* Planning**: Multi-resolution path planning for efficient exploration
- **GPU Acceleration**: CUDA-accelerated planning for high performance
- **CPU Fallback**: Automatic fallback to CPU implementation when CUDA is unavailable
- **Multiple Planning Modes**: Support for kinematic and kinodynamic planning
- **ROS Integration**: Ready-to-use ROS node for robotic applications
- **Configurable**: YAML-based configuration system
- **Extensible**: Modular design for easy customization

## Troubleshooting

### CUDA Issues
If you encounter CUDA-related errors:
1. Verify CUDA installation: `nvcc --version`
2. Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
3. The system will automatically fall back to CPU if CUDA is not available

### Compilation Issues
If you encounter compilation errors:
1. Ensure all Python dependencies are installed
2. Check that your compiler supports C++14 or higher
3. Verify that the conda environment is activated

### Map Loading Issues
If maps fail to load:
1. Verify map files exist in `examples/Maps/`
2. Check the `dir` path in your configuration file
3. Ensure map file names match the `name` field in the config

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both CPU and GPU environments
5. Submit a pull request

## License

[Add your license information here]
