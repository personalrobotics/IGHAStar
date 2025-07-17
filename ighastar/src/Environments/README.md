#Creating a Custom Environment for IGHAStar

This document explains how to create your own environment for use with the IGHAStar planner. To integrate with IGHAStar, your environment must implement a specific interface and define a compatible Node structure.

## Environment Interface Requirements

Your custom `Environment` class **must** provide the following methods and members:

### **Required Methods**

- **Constructor**
  ```cpp
  Environment(const py::dict& config);
``` Initializes the environment using a
        Python dictionary of configuration parameters.

    - **set_world **
  ```cpp void set_world(torch::Tensor world);
``` Loads the map or world representation from a PyTorch tensor.

                         - **cleanup **
  ```cpp void cleanup();
``` Frees any allocated resources(e.g., memory, CUDA buffers).

    - **create_Node **
  ```cpp std::shared_ptr<Node> create_Node(float *pose);
``` Returns a new Node at the given pose(with no parent).

    - **distance **
  ```cpp float distance(float *pose, float *goal);
  ```
  Computes the distance between two poses (used for heuristics and goal checks).

- **reached_goal_region**
  ```cpp
  bool reached_goal_region(std::shared_ptr<Node> v, std::shared_ptr<Node> goal);
  ``` Returns true if node `v` is within the goal region of `goal`.

      - **heuristic **
  ```cpp float heuristic(float *pose, float *goal);
  ``` Returns a heuristic estimate of the cost from `pose` to `goal`.

      - **check_validity **
  ```cpp void check_validity(float *start, float *goal, bool *result);
  ``` Sets `result[0]` and `result[1]` to true if `start` and `goal` are valid,
      respectively.

          - **Succ **
  ```cpp std::vector<std::shared_ptr<Node>> Succ(std::shared_ptr<Node> node,
                                                  std::shared_ptr<Node> goal);
  ```
  Returns a vector of successor nodes for a given node.

- **convert_node_list_to_path_tensor**
  ```cpp
  torch::Tensor convert_node_list_to_path_tensor(std::vector<std::shared_ptr<Node>> node_list);
  ```
  Converts a list of nodes (the path) to a PyTorch tensor for output/visualization.

### **Required Members**
- `int max_level;`  
  The maximum number of resolution levels (used for multi-resolution search).


## Node Structure Requirements

Your environment must define a `Node` struct or class with the following members and methods:

### **Required Members**
- `float pose[n_dims];
  ` // State vector (e.g., [x, y, ...])
      - `float *intermediate_poses;
  ` // (Optional) Intermediate states between parent and this node
      - `float g,
      f;
  ` // Cost-to-come and total cost
      - `bool active;
  ` // Whether the node is active in the search
      - `int rank,
      level;
  ` // Rank and resolution level
      - `size_t hash;
  ` // Hash value for fast lookup
      - `std::vector<size_t> index;
  ` // Multi-resolution indices
      - `std::shared_ptr<Node> parent;
  ` // Pointer to parent node

      ## #**Required Methods * *
      -**Constructor **
  ```cpp Node(const float *pose_in, const float *intermediate_poses_,
               std::shared_ptr<Node> parent_in, float g_in,
               const float *resolution_, float *tolerance, int max_level,
               float division_factor, int timesteps);
  ``` Initializes a node with the given state, parent,
      and other parameters.

          - **Destructor **
  ```cpp ~Node();
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

See the files in this directory (e.g., `kinematic.h`, `kinodynamic.h`, `simple.h`) for concrete examples of how to implement these interfaces.

---

**Note:**
- All methods and members must match the signatures above for IGHAStar to work correctly.
- You may add additional methods or members as needed for your environment's logic. 