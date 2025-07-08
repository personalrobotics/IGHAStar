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

Configuration files are located in `examples/Configs/` and include vehicle parameters, planning parameters, and environment settings.

For detailed configuration and usage instructions, see [examples/EXAMPLE_USAGE.md](examples/EXAMPLE_USAGE.md).

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
