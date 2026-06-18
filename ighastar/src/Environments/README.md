# Creating a Custom Environment for IGHAStar

**Most users should start with the [generic environment](../../docs/generic_environment.md)** (Python callbacks, no new C++/CUDA). This document covers **hand-written C++ environments** for maximum performance with a fixed representation.

See also [docs/extending.md](../../docs/extending.md) for choosing an integration path.

## Folder Structure

```
Environments/
├── include/           # Header files (.h)
│   ├── kinematic.h
│   ├── kinematic_cpu.h
│   ├── kinodynamic.h
│   ├── kinodynamic_cpu.h
│   └── simple.h
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

---

**Note:**
- All methods and members must match the signatures above for IGHAStar to work correctly.
- You may add additional methods or members as needed for your environment's logic.
- For bidirectional search support, ensure proper handling of `time_direction` in the Node constructor and `Succ` method.
