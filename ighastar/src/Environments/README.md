# Creating a Custom Environment for IGHAStar

Four paths exist:

| Path | Use when | Doc |
|---|---|---|
| **Generic (Python callbacks)** | You own dynamics, cost, validity, heuristic, and goal test in Python | [GENERIC.md](GENERIC.md) |
| **generic_torch (compiled torch callbacks)** | The same callbacks as pure torch functions, compiled with AOTInductor so the search runs without Python | [GENERIC.md](GENERIC.md#generic_torch-compiled-torch-callbacks) |
| **C++ Environment** | You implement `Succ` and related methods in C++/CUDA | This file |
| **Out-of-tree C++ Environment** | The same, but the header lives in your own project (`-I` + `IGHASTAR_ENV_HEADER`) | [Below](#choosing-the-environment-header-ighastar_env_header) |

## Folder Structure

```
Environments/
├── include/           # Header files (.h)
│   ├── node.h             # Node and NodePtrCompare, included by every env
│   ├── trajectory_export.h # path and control tensors for get_best_path
│   ├── lazy_index.h
│   ├── kinematic.h
│   ├── kinematic_cpu.h
│   ├── kinodynamic.h
│   ├── kinodynamic_cpu.h
│   ├── simple.h
│   ├── pusht.h            # PushT (Chipmunk CPU)
│   ├── pusht_multi.h
│   ├── pusht_qs_cuda.h
│   ├── pusht_physics.h
│   ├── generic_common.h   # shared logic of the generic envs (includes node.h)
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

**solve_2pBVP**
  ```cpp
  void solve_2pBVP(
      const std::vector<std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>>> &pairs,
      std::vector<bool> &valid,
      std::vector<std::vector<float>> &trajectories);
  ```
  2-point boundary-value solver for bidirectional near-meet checks. `pairs[i]` is a source node and a target node (the nodes, not just their coordinates, so the solver can use parent state). `valid[i]` is true when that connection is usable. `trajectories[i]` is the connecting states, row-major `[n_states * n_dims]` (empty if the solver does not return one). A single connection is a batch of size 1. `near_meet_config.solver` is `geometric` or `mpc`. An omitted solver is geometric. A near-meet call without `LCR` throws. `mpc` is the CEM rollout and requires its five parameters.

**sample_goals**
  ```cpp
  virtual std::vector<std::shared_ptr<Node>>
  sample_goals(std::shared_ptr<Node> start, std::shared_ptr<Node> goal);
  ```
  `sample_goals` returns goal nodes for the backward search. The default method calls `sampling::sample_goal_region`. Override `sample_goals` to supply a different sampler. `sampling::load_goal_sampling` reads `goal_sampling_config`. `space_type` is `SE2` or `R_N`. Any other value throws. The environment stores that config. Interpolation uses the angular dimensions from that config. The generic environment calls `sample_goals_fn` when the config provides it. That function takes the start pose and the goal pose. It returns a tensor of shape `[N, n_dims]`.

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
  The tensor has shape `[path_length, n_dims + 2]`. Column `n_dims` is the clock of that sample. The last column is `g * time_direction`. Solution cost stays `path[0, -1]`.

**convert_node_list_to_controls_tensor**
  ```cpp
  torch::Tensor convert_node_list_to_controls_tensor(
      std::vector<std::shared_ptr<Node>> node_list);
  ```
  Controls for the same rows as the path tensor. `get_best_path_with_controls` calls this. Both writers live in `trajectory_export.h`. On a backward edge the stored poses are reversed and the controls are not; the control read undoes that so each control row matches its path row.

### **Required Members**
- `int time_direction;` - Direction of search: `1` for forward, `-1` for backward.
- `int timesteps;` - Number of intermediate states between nodes.
- `float local_controllability_radius[n_dims];` - Local controllability radius for each dimension.
- `float resolution[n_dims];` - Resolution for each dimension.
- `float tolerance[n_dims];` - Tolerance for each dimension.
- `float division_factor;` - How much finer each search level is. `Node` reads it when building `index`.

## Node Structure Requirements

Do not define `Node` in the environment header. Define `n_dims`, `n_cont`, `hash_dims`, and `calc_hash(float *, const float *)`, then `#include "node.h"`. That header provides `Node` and `NodePtrCompare`. `simple` and `kinodynamic.h` set `hash_dims` equal to `n_dims`. `pusht_multi` defines `NODE_NFLAGS` as its tee count before the include, which adds `bool cleared[NFlags]`. Every other environment leaves `NODE_NFLAGS` at 0, so that array is absent.

`node.h` stores:

- `float pose[n_dims];` // State vector (e.g., [x, y, ...])
- `float *intermediate_poses;` // States between the parent and this node, `timesteps * n_dims`. Reversed when `time_direction` is `-1`.
- `float *intermediate_times;` // Clock of each of those states, `timesteps` values, reversed with the poses. Written when the edge is integrated (`fill_substep_times`). Null on a geometric edge.
- `float *intermediate_controls;` // Controls for that edge, `timesteps * n_cont`. `control[n_cont]` holds the first sample.
- `float g, f;` // Cost-to-come and total cost
- `float time;` // Arrival clock of this node. Set from the last generated substep. If no substep times were passed, it stays equal to `g`.
- `bool active;` // Whether the node is active in the search
- `int rank, level;` // Rank and resolution level
- `size_t hash;` // Hash value for fast lookup
- `size_t LCR_index;` // Index for local controllability radius table (used in bidirectional search)
- `int time_direction;` // Direction of search: `1` for forward, `-1` for backward
- `LazyIndex<hash_dims, calc_hash> index;` // `index[l]` is the hash at level `l`, computed the first time that level is read.
- `std::shared_ptr<Node> parent;`
- `bool preexpanded;` and `std::vector<std::shared_ptr<Node>> cached_succ;` // Successor cache
- `int discover_index, expand_index;` // `-1` until the search sets them

The constructor is:

```cpp
Node(const float *pose_in, const float *intermediate_poses_,
     std::shared_ptr<Node> parent_in, float g_in,
     const float *resolution_, const float *tolerance,
     float division_factor, int timesteps, const float *LCR_,
     int time_direction_ = 1,
     const float *intermediate_controls_ = nullptr);
```

The destructor frees `intermediate_poses` and `intermediate_controls`.

`Node::g` is a cost plus the pose hash. `a->g < b->g` orders by cost, then hash. Comparisons with a bare float (`get_G`, `Omega`) use the cost only. `NodePtrCompare` orders the open set as a min-heap on `f`, then pose hash (`Node::f_before`).

## Example

See the files in `include/` (e.g., `kinematic.h`, `kinodynamic.h`, `simple.h`) for concrete examples of how to implement these interfaces.

## Choosing the environment header (`IGHASTAR_ENV_HEADER`)

`ighastar.cpp` always includes one header named by the compile flag
`IGHASTAR_ENV_HEADER`. Built-in and out-of-tree envs use the same path; there is
no per-env `#elif` ladder. `create_planner` sets the flag from `node_type`
(e.g. `kinodynamic` → `"kinodynamic.h"`). These are compiler flags (`-D...`),
not shell environment variables.

| Macro | Required | Meaning |
|---|---|---|
| `IGHASTAR_ENV_HEADER` | yes | Header to include, as a quoted string, e.g. `"kinodynamic.h"` or `"my_env.h"`. Found on the include path (`-I`). Missing → compile error. Use `env_header_flag("my_env.h")` from `common_utils` to quote it for ninja. |
| `IGHASTAR_RELEASES_GIL` | no | Release Python's GIL for the whole `search` call. Set for `generic_torch` and for out-of-tree envs that never call into Python during search. |

Built-in headers under `Environments/include/`: `simple.h`, `kinematic.h` /
`kinematic_cpu.h`, `kinodynamic.h` / `kinodynamic_cpu.h`, `generic.h`,
`generic_torch.h`, `pusht.h`, `pusht_multi.h`, `pusht_qs_cuda.h`.

An out-of-tree header must provide an `Environment` class with the interface
above. It gets `Node` and `NodePtrCompare` by including `node.h` after
`n_dims`, `n_cont`, `hash_dims`, and `calc_hash`. The generic environments do
that inside `generic_common.h`, which `generic.h` and `generic_torch.h` include.
`generic_common.h` takes dimensions from `N_DIMS`, `N_CONT` and `HASH_DIMS`.
The constructor receives the whole config dict passed to `IGHAStar(config, debug)`.

`create_planner` only builds the built-in `node_type`s, so build an out-of-tree
env yourself:

```python
from pathlib import Path

from ighastar.scripts.common_utils import BASE_DIR, CXX_STD_FLAG, env_header_flag
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
        CXX_STD_FLAG,  # -std=c++20, or -std=c++2a on GCC 9
        "-O3",
        env_header_flag("my_env.h"),
        "-DIGHASTAR_RELEASES_GIL",  # only if the search never calls Python
        "-DN_DIMS=5",
        "-DN_CONT=2",
        "-DHASH_DIMS=5",
    ],
)
planner = kernel.IGHAStar({"experiment_info_default": info, "my_env": my_settings}, False)
```

A compile smoke for an out-of-tree header lives in
`tests/unit/environments/test_external_env.py` (stub header next to it).

Give each distinct build its own `name`: torch caches extensions by name, and IGHA*'s classes are
`py::module_local`, so several builds can be loaded in one process.

---

**Note:**
- All methods and members must match the signatures above for IGHAStar to work correctly.
- You may add additional methods or members as needed for your environment's logic.
- For bidirectional search support, ensure proper handling of `time_direction` in the Node constructor and `Succ` method.
