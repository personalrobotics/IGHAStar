# Example Usage Guide

This document explains how to configure and use IGHAStar for different planning scenarios.

## Configuration Files

Configuration files are located in `examples/standalone/Configs/` and include:

- **Vehicle Parameters**: Length, width, maximum velocity, steering limits
- **Planning Parameters**: Resolution, tolerance, epsilon values
- **Environment Settings**: Map resolution, timesteps, control discretization

### Available Configuration Files
- `kinematic_example.yml` - Kinematic planning configuration
- `kinodynamic_example.yml` - Kinodynamic planning configuration  
- `simple_example.yml` - Simple planning configuration
- `ros_kinodynamic_example.yml` - ROS integration configuration

## Modifying Start and Goal Points

To change the start and goal positions for path planning, you can either:

1. **Edit the configuration file** to modify the default start/goal or add test cases
2. **Use the `--test-case` argument** to select from predefined test cases

### Configuration File Structure

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

### Coordinate System
- **x, y**: Position coordinates in meters
- **heading**: Orientation in radians (0 = east, π/2 = north, π = west, -π/2 = south)
- **velocity**: Speed in m/s (kinodynamic only)

## Map Files

Map files are stored in `examples/standalone/Maps/` with the following structure:
- `generated_maps/` - Procedurally generated test maps
- `street-png/` - Street network maps
- `Offroad/` - Off-road terrain maps

## Test Cases

Each configuration file can include multiple test cases with different start/goal positions. To use a specific test case:

```bash
python examples/standalone/example.py --config examples/standalone/Configs/kinematic_example.yml --test-case case2
```

If no test case is specified, the default start/goal from the configuration file will be used.

## Parameter Descriptions

### Map Parameters
- `name`: Name of the map file
- `dir`: Directory containing the map file (relative to examples/)
- `start`: Starting position [x, y, heading] or [x, y, heading, velocity, unused]
- `goal`: Goal position [x, y, heading] or [x, y, heading, velocity, unused]
- `size`: Map dimensions [width, height] in pixels
- `res`: Map resolution in meters per pixel

### Planning Parameters
- `resolution`: Starting resolution used for discretization
- `epsilon`: Goal region tolerance [ate, cte, heading, vel] - along-track error, cross-track error, heading tolerance, velocity tolerance
- `tolerance`: Minimum separation between vertices (your perception/map resolution should be at least this)
- `max_level`: Maximum level to which the system can go (used to cache hash values)
- `division_factor`: Factor by which the resolution increases every level
- `max_expansions`: Maximum number of node expansions allowed
- `hysteresis`: Hysteresis threshold for IGHA*-H algorithm

### Vehicle Parameters
- `length`: Vehicle length in meters
- `width`: Vehicle width in meters
- `steering_list`: Available steering angles in degrees
- `throttle_list`: Available throttle values (negative = reverse, positive = forward)
- `max_vel`: Maximum velocity in m/s
- `min_vel`: Minimum velocity in m/s
- `del_theta`: Maximum steering angle change in degrees
- `max_theta`: Maximum steering angle in degrees
- `del_vel`: Maximum velocity change in m/s
- `max_vert_acc`: Maximum vertical acceleration in m/s²
- `RI`: Rolling resistance coefficient
- `gear_switch_time`: Time penalty for gear switching (multiplies reverse distance) 