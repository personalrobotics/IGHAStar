# Creating a Custom Environment for IGHAStar

Four paths exist:

| Path | Use when | Doc |
|---|---|---|
| **Generic (Python callbacks)** | You own dynamics, cost, validity, heuristic, and goal test in Python | [GENERIC.md](GENERIC.md) |
| **generic_torch (compiled torch callbacks)** | The same callbacks as pure torch functions, compiled with AOTInductor so the search runs without Python | [GENERIC.md](GENERIC.md#generic_torch-compiled-torch-callbacks) |
| **C++ Environment** | You implement `Succ` and related methods in C++/CUDA | This file |
| **External C++ Environment** | The same, but the environment lives in your own project rather than this repo | [Below](#external-environments-use_external_env) |

## Folder Structure

```
Environments/
├── include/           # Header files (.h)
│   ├── kinematic.h
│   ├── kinematic_cpu.h
│   ├── kinodynamic.h
│   ├── kinodynamic_cpu.h
│   ├── simple.h
│   ├── generic_common.h   # Node + shared logic of the generic envs
│   ├── generic.h          # Python callbacks (py::function)
│   └── generic_torch.h    # AOTInductor-compiled torch callbacks
├── src/               # Source files (.cpp, .cu)
│   ├── kinematic.cu
│   ├── kinematic_cpu.cpp
│   ├── kinodynamic.cu
│   └── kinodynamic_cpu.cpp
└── README.md
```

## What is an environment?
The envrionment is effectively what provides the successor function, edge evaluation, heuristic, goal checking, and what the node data structure looks like.
For instance, in the example environments used here, the world is represented using 2D costmaps and elevation maps. 
However, the IGHA* algorithm in of itself does not care about the representation format used; the representation used is only relevant for getting successors and or evaluating edges.

This document explains how to create your own environment for use with the IGHAStar planner. To integrate with IGHAStar, your environment must implement a specific interface and define a compatible Node structure.

## Quick note about example environment conventions:
The map convention used here is as follows:
Map origin (0, 0) is at the bottom left (if visualized using cv2.imshow, then top left).
X is East, Y is North. We use the [REP103](https://www.ros.org/reps/rep-0103.html) standard conventions.

For custom environments you can pick a convention of your choice.

## Environment Interface Requirements

Your custom `Environment` class **must** provide the following methods and members:

### **Required Methods**

**Constructor**
  ```cpp
  Environment(const py::dict& config, int time_direction_ = 1);
```
Initializes the environment using a Python dictionary of configuration parameters.
The `time_direction` parameter is used for bidirectional search: `1` for forward search, `-1` for backward search.

**set_world**
  ```cpp
  void set_world(torch::Tensor world);
  ```
Loads the map or world representation from a PyTorch tensor.

The CUDA kinodynamic environment accepts two representations and picks between
them on the rank of the tensor: a 3D `H x W x 2` tensor is an elevation map plus
costmap, and a 4D `H x W x nz x 4` tensor is an ESDF whose channels are
`[signed distance, R, G, B]`. Which one the terrain queries expect is set by
`map_type` (`"elevation"` or `"esdf"`) in `node_info`; it defaults to
`"elevation"`. In ESDF mode `map_to_elev()` recovers the terrain height from the
zero crossing of a column and `map_to_cost()` reads the colour of the surface
voxel, so the physics and the collision checker are unchanged. The ESDF grid
geometry (`voxel_z`, `z_min`) travels in the `esdf` block of `node_info`, since
the tensor carries voxel data only. The CPU kinodynamic environment supports the
elevation representation only and rejects 4D tensors.

**create_Node**
  ```cpp
  std::shared_ptr<Node> create_Node(float *pose);
  ```
  Returns a new Node at the given pose (with no parent).

**distance**
  ```cpp
  float distance(float *pose, float *goal);
  ```
  Computes the distance between two poses (used for heuristics and goal checks).

**reached_goal_region**
  ```cpp
  bool reached_goal_region(std::shared_ptr<Node> v, std::shared_ptr<Node> goal);
  ```
  Returns true if node `v` is within the goal region of `goal`.

**heuristic**
  ```cpp
  float heuristic(float *pose, float *goal);
  ```
  Returns a heuristic estimate of the cost from `pose` to `goal`.

**check_validity**
  ```cpp
  void check_validity(float *start, float *goal, bool *result);
  ```
  Sets `result[0]` and `result[1]` to true if `start` and `goal` are valid, respectively.

**check_validity_batched**
  ```cpp
  void check_validity_batched(const std::vector<float *> &states, std::vector<bool> &results);
  ```
  Checks validity for a batch of states. Used for goal region sampling in bidirectional search.

**compute_near_meet_distance**
  ```cpp
  float compute_near_meet_distance(float *pose1, float *pose2);
  ```
  Computes the distance between two poses for near-meet detection in bidirectional search.

**Succ**
  ```cpp
  std::vector<std::shared_ptr<Node>> Succ(std::shared_ptr<Node> node,
                                          std::shared_ptr<Node> goal);
  ```
  Returns a vector of successor nodes for a given node.
  This function internally performs the forward rollout from the input `node`,
  then, for the successors that are valid, generates new nodes and assigns
  their corresponding `f` values.

**convert_node_list_to_path_tensor**
  ```cpp
  torch::Tensor convert_node_list_to_path_tensor(std::vector<std::shared_ptr<Node>> node_list);
  ```
  Converts a list of nodes (the path) to a PyTorch tensor for output/visualization.
  The tensor should have shape `[path_length, n_dims + 1]`, where the last column contains `g * time_direction` to indicate search direction.

### **Required Members**
- `int max_level;` - The maximum number of resolution levels (used for multi-resolution search).
- `int time_direction;` - Direction of search: `1` for forward, `-1` for backward.
- `int timesteps;` - Number of intermediate states between nodes.
- `float local_controllability_radius[n_dims];` - Local controllability radius for each dimension.
- `float resolution[n_dims];` - Resolution for each dimension.
- `float tolerance[n_dims];` - Tolerance for each dimension.

## Node Structure Requirements

Your environment must define a `Node` struct or class with the following members and methods:

### **Required Members**
- `float pose[n_dims];` // State vector (e.g., [x, y, ...])
- `float *intermediate_poses;` // (Optional) Intermediate states between parent and this node
- `float g, f;` // Cost-to-come and total cost
- `bool active;` // Whether the node is active in the search
- `int rank, level;` // Rank and resolution level
- `size_t hash;` // Hash value for fast lookup
- `size_t LCR_index;` // Index for local controllability radius table (used in bidirectional search)
- `int time_direction;` // Direction of search: `1` for forward, `-1` for backward
- `std::vector<size_t> index;` // Multi-resolution indices
- `std::shared_ptr<Node> parent;` // Pointer to parent node

### **Required Methods**
- **Constructor**
  ```cpp
  Node(const float *pose_in, const float *intermediate_poses_,
       std::shared_ptr<Node> parent_in, float g_in,
       const float *resolution_, float *tolerance, int max_level,
       float division_factor, int timesteps,
       const float *LCR_, int time_direction_ = 1);
  ```
  Initializes a node with the given state, parent, and other parameters.

- **Destructor**
  ```cpp
  ~Node();
  ```
  Frees any dynamically allocated memory (e.g., `intermediate_poses`).

## Node Comparison

You must also define a comparison functor for use in priority queues:

```cpp
struct NodePtrCompare {
    bool operator()(const std::shared_ptr<Node> &a,
                    const std::shared_ptr<Node> &b) const {
        return a->f > b->f; // min-heap: smaller f has higher priority
    }
};
```

## Example

See the files in `include/` (e.g., `kinematic.h`, `kinodynamic.h`, `simple.h`) for concrete examples of how to implement these interfaces.

## External environments (`USE_EXTERNAL_ENV`)

`ighastar.cpp` picks its environment at compile time, with one preprocessor macro per built-in header:

| Macro | Header |
|---|---|
| `USE_SIMPLE_ENV` | `simple.h` |
| `USE_KINEMATIC_ENV` / `USE_KINEMATIC_CPU_ENV` | `kinematic.h` / `kinematic_cpu.h` |
| `USE_KINODYNAMIC_ENV` / `USE_KINODYNAMIC_CPU_ENV` | `kinodynamic.h` / `kinodynamic_cpu.h` |
| `USE_GENERIC_ENV` | `generic.h` |
| `USE_GENERIC_TORCH_ENV` | `generic_torch.h` |
| `USE_EXTERNAL_ENV` | the header named by `EXTERNAL_ENV_HEADER` |

`USE_EXTERNAL_ENV` lets a project outside this repo supply its own environment without editing or copying
`ighastar.cpp`. These are compiler flags (`-D...`), not environment variables:

| Macro | Required | Meaning |
|---|---|---|
| `USE_EXTERNAL_ENV` | yes | Use an external environment header instead of a built-in one. |
| `EXTERNAL_ENV_HEADER` | yes, with `USE_EXTERNAL_ENV` | The header to include, as a quoted string, e.g. `"my_env.h"`. It is found on the include path, so add its directory with `-I`. Leaving it out is a compile error. |
| `EXTERNAL_ENV_RELEASES_GIL` | no | Release Python's GIL for the whole `search` call, as `generic_torch` does, so other Python threads keep running. Only define it if the environment never calls into Python during a search (no `py::function` callbacks, no Python objects touched after construction). |

The header must provide the same things a built-in one does: an `Environment` class with the interface above, a
`Node` struct and `NodePtrCompare`. The simplest way is to build on `generic_common.h`, as `generic.h` and
`generic_torch.h` do: derive `Environment` from `GenericEnvironmentBase` (config parsing, `create_Node`, path
conversion) and implement the callbacks (`Succ_batched`, `Succ`, `heuristic`, `reached_goal_region`,
`check_validity`, `check_validity_batched`, `set_world`) yourself. `generic_common.h` takes its dimensions from
`N_DIMS`, `N_CONT` and `HASH_DIMS`, so define those too. The constructor receives the whole config dict passed to
`IGHAStar(config, debug)`, so the environment can read its own settings from any key you add.

`create_planner` only builds the built-in environments, so build an external one yourself, e.g. with torch's
extension loader:

```python
from pathlib import Path

from ighastar.scripts.common_utils import BASE_DIR
from torch.utils.cpp_extension import load

src = Path(BASE_DIR) / "src"
kernel = load(
    name="my_planner",
    sources=[str(src / "ighastar.cpp")],
    extra_include_paths=[
        "/path/to/my_project/include",  # where my_env.h lives
        str(src / "Environments" / "include"),  # generic_common.h, if you build on it
        str(src / "utils"),
        str(src),
    ],
    extra_cflags=[
        "-std=c++20",
        "-O3",
        "-DUSE_EXTERNAL_ENV",
        # The extra single quotes keep the double quotes through the shell that
        # ninja runs the compiler in: the macro's value must be `"my_env.h"`.
        "-DEXTERNAL_ENV_HEADER='\"my_env.h\"'",
        "-DEXTERNAL_ENV_RELEASES_GIL",  # only if the search never calls Python
        "-DN_DIMS=5",
        "-DN_CONT=2",
        "-DHASH_DIMS=5",
    ],
)
planner = kernel.IGHAStar({"experiment_info_default": info, "my_env": my_settings}, False)
```

Give each distinct build its own `name`: torch caches extensions by name, and IGHA*'s classes are
`py::module_local`, so several builds (external and built-in) can be loaded in one process.

---

**Note:**
- All methods and members must match the signatures above for IGHAStar to work correctly.
- You may add additional methods or members as needed for your environment's logic.
- For bidirectional search support, ensure proper handling of `time_direction` in the Node constructor and `Succ` method.
