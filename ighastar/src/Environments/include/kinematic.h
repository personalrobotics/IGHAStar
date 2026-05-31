#include <boost/functional/hash.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <torch/extension.h>

using namespace std;

constexpr int n_dims = 4;
constexpr int hash_dims =
    3; // because we don't differentiate in the velocity dimension
constexpr int n_cont = 2;
constexpr int warp_size = 32;

// Launches kinematic simulation for multiple rollouts on GPU
// Uses per-instance device memory and stream for parallel execution
void kinematic_launcher(
    float *state, float *intermediate_states, const float *heightmap,
    const float *costmap, bool *valid, float *cost, float dt, int timesteps,
    int rollouts, int n_dims, int n_cont, const int map_size_px, float map_res,
    float car_l2, float car_w2, float max_vel, float min_vel, float RI,
    float max_vert_acc, float max_theta, float gear_switch_time,
    int patch_length_px, int patch_width_px, const int blocks,
    const int threads, float *d_state, float *d_intermediate_states,
    float *d_controls, bool *d_valid, float *d_cost, cudaStream_t stream);

// Checks validity of multiple states against a bitmap on GPU
void check_validity_launcher(const float *bitmap, int map_size_px,
                             float map_res, float *states, int patch_length_px,
                             int patch_width_px, float car_l2, float car_w2,
                             int n_states, int n_dims, bool *result);

size_t calc_hash(float *pose, const float *resolution) {
  std::size_t hash_val =
      0; // this is the "seed", replace with #define value later
  for (int i = 0; i < hash_dims; ++i) {
    boost::hash_combine(hash_val, std::hash<size_t>{}(static_cast<int>(
                                      std::round(pose[i] / resolution[i]))));
  }
  return hash_val;
}

struct Node {
public:
  float pose[n_dims];
  float *intermediate_poses;
  float g, f;
  bool active;
  int rank, level;
  size_t hash;
  size_t
      LCR_index; // Hash for local controllability radius (near-meet detection)
  int time_direction; // 1 for forward, -1 for backward
  std::vector<size_t> index;
  std::shared_ptr<Node> parent;
  // Preemptive expansion bookkeeping: whether this node's successors have
  // already been computed (and cached) before it was popped from Q_v.
  bool preexpanded = false;
  std::vector<std::shared_ptr<Node>> cached_succ;
  // Constructor
  Node(const float *pose_in, const float *intermediate_poses_,
       std::shared_ptr<Node> parent_in, float g_in, const float *resolution_,
       float *tolerance, int max_level, float division_factor, int timesteps,
       const float *LCR_, int time_direction_ = 1)
      : intermediate_poses(nullptr), g(g_in), f(0), parent(parent_in),
        active(true), rank(0), level(0), time_direction(time_direction_) {
    for (int i = 0; i < n_dims; i++) {
      pose[i] = pose_in[i];
    }
    if (parent_in != nullptr && intermediate_poses_ != nullptr) {
      // intermediate poses are the poses between the parent and the current
      // node, they come in the shape timestep x n_dims
      intermediate_poses = new float[timesteps * n_dims];
      if (time_direction_ == -1) {
        // Reverse the sequence for backward search
        for (int t = 0; t < timesteps; t++) {
          int src_index = (timesteps - 1 - t) * n_dims;
          int dst_index = t * n_dims;
          memcpy(&intermediate_poses[dst_index],
                 &intermediate_poses_[src_index], n_dims * sizeof(float));
        }
      } else {
        memcpy(intermediate_poses, intermediate_poses_,
               timesteps * n_dims * sizeof(float));
      }
    }
    // Compute LCR_index using local_controllability_radius from config
    LCR_index = calc_hash(pose, LCR_);
    hash = calc_hash(pose, tolerance);
    float resolution[n_dims];
    for (int i = 0; i < n_dims; i++) {
      resolution[i] = resolution_[i];
    }
    for (int i = 0; i < max_level; i++) {
      index.push_back(calc_hash(pose, resolution));
      for (int j = 0; j < n_dims; j++) {
        resolution[j] = resolution[j] / division_factor;
      }
    }
  }

  // Destructor to clean up intermediate_poses
  ~Node() {
    if (intermediate_poses) {
      delete[] intermediate_poses;
    }
  }
};

struct NodePtrCompare {
  bool operator()(const std::shared_ptr<Node> &a,
                  const std::shared_ptr<Node> &b) const {
    return a->f > b->f; // min-heap: smaller f has higher priority
  }
};

class Environment {
  int map_size_px, threads, blocks, n_succ, patch_length_px, patch_width_px;
  float epsilon[n_dims];
  float *d_heightmap, *d_costmap;
  float dt, map_res;
  float car_l2, car_w2;
  float max_vel, min_vel, RI, max_theta, max_vert_acc, gear_switch_time;

  // Per-instance CUDA device memory (allows parallel execution without
  // conflicts)
  float *d_controls_instance;
  float *d_state_instance;
  float *d_cost_instance;
  float *d_intermediate_states_instance;
  bool *d_valid_instance;

  // Batched device memory for preemptive expansion. These buffers hold
  // up to batch_capacity * n_succ rollouts so that several parent nodes can be
  // expanded in a single GPU kernel launch. They are allocated lazily via
  // ensure_batch_capacity() and remain nullptr when preemptive expansion is
  // disabled.
  float *d_controls_batched = nullptr;
  float *d_state_batched = nullptr;
  float *d_cost_batched = nullptr;
  float *d_intermediate_batched = nullptr;
  bool *d_valid_batched = nullptr;
  int batch_capacity = 0;

  // Per-instance CUDA stream (allows parallel kernel execution)
  cudaStream_t cuda_stream;

public:
  int max_level;
  int timesteps;
  int time_direction;
  float resolution[n_dims];
  float tolerance[n_dims];
  float local_controllability_radius[n_dims];
  float division_factor;

  Environment(const py::dict &config, int time_direction_ = 1)
      : d_heightmap(nullptr), d_costmap(nullptr),
        time_direction(time_direction_), d_controls_instance(nullptr),
        d_state_instance(nullptr), d_cost_instance(nullptr),
        d_intermediate_states_instance(nullptr), d_valid_instance(nullptr) {
    auto info = config["experiment_info_default"].cast<py::dict>();
    auto node_info = info["node_info"].cast<py::dict>();
    // Top-level fields
    map_res = node_info["map_res"].cast<float>();
    max_level = info["max_level"].cast<int>();
    division_factor = info["division_factor"].cast<float>();

    set_resolutions(info, node_info);
    set_car_params(node_info);

    threads = int(std::ceil(float(n_succ) / float(warp_size)) *
                  warp_size); // thread and block should be a macro.
    blocks = (n_succ + threads - 1) / threads;
  }
  // destructor
  ~Environment() {
    // Free per-instance CUDA memory
    if (d_controls_instance != nullptr) {
      cudaFree(d_controls_instance);
      d_controls_instance = nullptr;
    }
    if (d_state_instance != nullptr) {
      cudaFree(d_state_instance);
      d_state_instance = nullptr;
    }
    if (d_intermediate_states_instance != nullptr) {
      cudaFree(d_intermediate_states_instance);
      d_intermediate_states_instance = nullptr;
    }
    if (d_valid_instance != nullptr) {
      cudaFree(d_valid_instance);
      d_valid_instance = nullptr;
    }
    if (d_cost_instance != nullptr) {
      cudaFree(d_cost_instance);
      d_cost_instance = nullptr;
    }
    // Free batched device memory (preemptive expansion)
    if (d_controls_batched != nullptr) {
      cudaFree(d_controls_batched);
      d_controls_batched = nullptr;
    }
    if (d_state_batched != nullptr) {
      cudaFree(d_state_batched);
      d_state_batched = nullptr;
    }
    if (d_cost_batched != nullptr) {
      cudaFree(d_cost_batched);
      d_cost_batched = nullptr;
    }
    if (d_intermediate_batched != nullptr) {
      cudaFree(d_intermediate_batched);
      d_intermediate_batched = nullptr;
    }
    if (d_valid_batched != nullptr) {
      cudaFree(d_valid_batched);
      d_valid_batched = nullptr;
    }
    // Destroy CUDA stream
    cudaStreamDestroy(cuda_stream);
    // Free map memory
    if (d_costmap != nullptr) {
      cudaFree(d_costmap);
      d_costmap = nullptr;
    }
    if (d_heightmap != nullptr) {
      cudaFree(d_heightmap);
      d_heightmap = nullptr;
    }
  }

  // Sets resolution, tolerance, epsilon, and LCR values for different
  // dimensions
  void set_resolutions(py::dict &info, py::dict &node_info) {
    float res = info["resolution"].cast<float>();
    float tol = info["tolerance"].cast<float>();
    auto eps = info["epsilon"].cast<std::vector<float>>();
    // Use backward_epsilon for backward search if available
    if (time_direction == -1 && info.contains("backward_epsilon")) {
      eps = info["backward_epsilon"].cast<std::vector<float>>();
    }
    float del_theta = node_info["del_theta"].cast<float>() / 57.3;
    float del_vel = node_info["del_vel"].cast<float>();
    resolution[0] = res;
    resolution[1] = res;
    resolution[2] = res * del_theta;
    resolution[3] = res * del_vel;
    tolerance[0] = tol;
    tolerance[1] = tol;
    tolerance[2] = tol * del_theta;
    tolerance[3] = tol * del_vel;
    for (int i = 0; i < n_dims && i < static_cast<int>(eps.size()); i++) {
      epsilon[i] = eps[i];
    }
    // Set LCR from config if available, otherwise use epsilon
    if (info.contains("LCR")) {
      auto lcr = info["LCR"].cast<std::vector<float>>();
      for (int i = 0; i < n_dims; i++) {
        local_controllability_radius[i] =
            (i < static_cast<int>(lcr.size())) ? lcr[i] : epsilon[i];
      }
    } else {
      for (int i = 0; i < n_dims; i++) {
        local_controllability_radius[i] = epsilon[i];
      }
    }
  }

  // Sets car parameters and initializes per-instance CUDA memory
  void set_car_params(py::dict &info) {
    car_l2 = info["length"].cast<float>() / 2;
    car_w2 = info["width"].cast<float>() / 2;
    max_vel = info["max_vel"].cast<float>();
    min_vel = info["min_vel"].cast<float>();
    RI = info["RI"].cast<float>();
    max_theta = info["max_theta"].cast<float>() / 57.3;
    max_vert_acc = info["max_vert_acc"].cast<float>();
    patch_length_px = int(car_l2 * 2 / map_res);
    patch_width_px = int(car_w2 * 2 / map_res);
    timesteps = info["timesteps"].cast<int>();
    dt = info["dt"].cast<float>() * float(time_direction);
    gear_switch_time = info["gear_switch_time"].cast<float>();

    auto steering_list = info["steering_list"].cast<std::vector<float>>();
    auto throttle_list = info["throttle_list"].cast<std::vector<float>>();
    n_succ = int(steering_list.size()) * int(throttle_list.size());
    float step_size = info["step_size"].cast<float>();

    float controls[n_succ * n_cont];
    int idx = 0;
    for (int i = 0; i < throttle_list.size(); ++i) {
      for (int j = 0; j < steering_list.size(); ++j) {
        controls[idx * n_cont] = tanf(steering_list[j] / 57.3) / (car_l2 * 2);
        controls[idx * n_cont + 1] = throttle_list[i] * step_size;
        ++idx;
      }
    }

    // Allocate per-instance CUDA device memory (replaces global cuda_setup)
    cudaError_t err;
    err = cudaMalloc(&d_controls_instance, sizeof(float) * n_succ * n_cont);
    if (err != cudaSuccess) {
      std::cerr << "CUDA error allocating d_controls_instance: "
                << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMalloc(&d_state_instance, n_succ * n_dims * sizeof(float));
    if (err != cudaSuccess) {
      std::cerr << "CUDA error allocating d_state_instance: "
                << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMalloc(&d_intermediate_states_instance,
                     n_succ * timesteps * n_dims * sizeof(float));
    if (err != cudaSuccess) {
      std::cerr << "CUDA error allocating d_intermediate_states_instance: "
                << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMalloc(&d_valid_instance, n_succ * sizeof(bool));
    if (err != cudaSuccess) {
      std::cerr << "CUDA error allocating d_valid_instance: "
                << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMalloc(&d_cost_instance, n_succ * sizeof(float));
    if (err != cudaSuccess) {
      std::cerr << "CUDA error allocating d_cost_instance: "
                << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMemcpy(d_controls_instance, controls,
                     sizeof(float) * n_succ * n_cont, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      std::cerr << "CUDA error copying controls: " << cudaGetErrorString(err)
                << std::endl;
    }
    // Create CUDA stream for parallel execution
    err = cudaStreamCreate(&cuda_stream);
    if (err != cudaSuccess) {
      std::cerr << "CUDA error creating stream: " << cudaGetErrorString(err)
                << std::endl;
    }
  }

  // Sets the world map (costmap and heightmap) from a PyTorch tensor
  void set_world(torch::Tensor world) {
    // Clean up existing CUDA memory
    if (d_costmap != nullptr) {
      cudaFree(d_costmap);
      d_costmap = nullptr;
    }
    if (d_heightmap != nullptr) {
      cudaFree(d_heightmap);
      d_heightmap = nullptr;
    }

    TORCH_CHECK(world.dim() == 3, "World tensor must be 3D (H x W x 2)");
    TORCH_CHECK(world.size(2) == 2,
                "Last dimension must have size 2 (costmap + heightmap)");
    TORCH_CHECK(world.dtype() == torch::kFloat32,
                "World tensor must be float32");
    TORCH_CHECK(world.device().is_cpu(),
                "World tensor must be on CPU"); // change if needed

    int H = world.size(0);
    int W = world.size(1);
    int C = world.size(2);
    int map_size = H * W;

    auto costmap =
        world.index({torch::indexing::Slice(), torch::indexing::Slice(), 0})
            .contiguous();
    auto heightmap =
        world.index({torch::indexing::Slice(), torch::indexing::Slice(), 1})
            .contiguous();

    // Allocate and copy to device
    cudaError_t costmap_alloc =
        cudaMalloc(&d_costmap, map_size * sizeof(float));
    cudaError_t heightmap_alloc =
        cudaMalloc(&d_heightmap, map_size * sizeof(float));

    if (costmap_alloc != cudaSuccess || heightmap_alloc != cudaSuccess) {
      // Clean up on allocation failure
      if (d_costmap != nullptr) {
        cudaFree(d_costmap);
        d_costmap = nullptr;
      }
      if (d_heightmap != nullptr) {
        cudaFree(d_heightmap);
        d_heightmap = nullptr;
      }
      throw std::runtime_error("CUDA memory allocation failed");
    }

    cudaMemcpy(d_costmap, costmap.data_ptr<float>(), map_size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_heightmap, heightmap.data_ptr<float>(),
               map_size * sizeof(float), cudaMemcpyHostToDevice);

    map_size_px = H; // assume H = W
                     // print the map size:
  }

  // Creates a new Node with the given pose
  std::shared_ptr<Node> create_Node(float *pose) {
    return std::make_shared<Node>(pose, nullptr, nullptr, 0, resolution,
                                  tolerance, max_level, division_factor,
                                  timesteps, local_controllability_radius,
                                  time_direction);
  }

  // Calculates Euclidean distance between two poses
  float distance(float *pose, float *goal) {
    float dx = pose[0] - goal[0];
    float dy = pose[1] - goal[1];
    return std::sqrt(dx * dx + dy * dy);
  }

  // Checks if a node has reached the goal region within epsilon tolerance
  bool reached_goal_region(std::shared_ptr<Node> v,
                           std::shared_ptr<Node> goal) {
    // float dist = distance(v->pose, goal->pose);
    // cross-track error
    float dx = v->pose[0] - goal->pose[0];
    float dy = v->pose[1] - goal->pose[1];
    float cte = dx * cosf(goal->pose[2]) + dy * sinf(goal->pose[2]);
    float ate = dx * sinf(goal->pose[2]) - dy * cosf(goal->pose[2]);
    if (fabs(cte) < epsilon[0] && fabs(ate) < epsilon[1]) {
      float dot_product = cosf(v->pose[2]) * cosf(goal->pose[2]) +
                          sinf(v->pose[2]) * sinf(goal->pose[2]);
      if (dot_product > cosf(epsilon[2])) {
        return true;
      }
    }
    return false;
  }

  // Calculates heuristic value for A* search
  float heuristic(float *pose, float *goal) { return distance(pose, goal); }

  // Compute distance between two poses for near-meet checking in bidirectional
  // search
  float compute_near_meet_distance(float *pose1, float *pose2) {
    return heuristic(pose1, pose2);
  }

  // Batched validity check for multiple states
  void check_validity_batched(const std::vector<float *> &states,
                              std::vector<bool> &results) {
    int n_states = states.size();
    if (n_states == 0) {
      results.clear();
      return;
    }

    results.resize(n_states);

    // Pack states into contiguous array
    float *states_array = new float[n_states * n_dims];
    for (int i = 0; i < n_states; i++) {
      memcpy(&states_array[i * n_dims], states[i], n_dims * sizeof(float));
    }

    // Allocate result array
    bool *result_array = new bool[n_states];
    std::fill(result_array, result_array + n_states, true);

    // Call CUDA launcher
    check_validity_launcher(d_costmap, map_size_px, map_res, states_array,
                            patch_length_px, patch_width_px, car_l2, car_w2,
                            n_states, n_dims, result_array);

    // Copy results back
    for (int i = 0; i < n_states; i++) {
      results[i] = result_array[i];
    }

    delete[] states_array;
    delete[] result_array;
  }

  // Checks validity of start and goal positions
  void check_validity(float *start, float *goal, bool *result) {
    // Check if CUDA memory is allocated
    if (d_costmap == nullptr || d_heightmap == nullptr) {
      std::cerr << "Error: CUDA memory not allocated. Call set_world() first."
                << std::endl;
      result[0] = false;
      result[1] = false;
      return;
    }

    // put start and goal into an array called states
    float states[2 * n_dims];
    memcpy(states, start, n_dims * sizeof(float));
    memcpy(states + n_dims, goal, n_dims * sizeof(float));
    check_validity_launcher(d_costmap, map_size_px, map_res, states,
                            patch_length_px, patch_width_px, car_l2, car_w2, 2,
                            n_dims, result);
  }

  // function that returns a vector of nodes:
  std::vector<std::shared_ptr<Node>> Succ(std::shared_ptr<Node> node,
                                          std::shared_ptr<Node> goal) {
    std::vector<std::shared_ptr<Node>> neighbors;
    std::shared_ptr<Node> neighbor;
    float new_pose[n_dims], f;

    // Check if CUDA memory is allocated
    if (d_costmap == nullptr || d_heightmap == nullptr) {
      std::cerr << "Error: CUDA memory not allocated. Call set_world() first."
                << std::endl;
      return neighbors;
    }

    // state is a copy of the node pose repeated for each thread
    // this variable creation can be optimized.
    float state[n_succ * n_dims];
    float intermediate_states[timesteps * n_succ * n_dims];
    bool valid[n_succ];
    float cost[n_succ];
    std::fill(valid, valid + n_succ, true);
    std::fill(cost, cost + n_succ, 0.0);

    for (int i = 0; i < n_succ; i++) {
      memcpy(&state[i * n_dims], node->pose, n_dims * sizeof(float));
    }
    kinematic_launcher(
        state, intermediate_states, d_heightmap, d_costmap, valid, cost, dt,
        timesteps, n_succ, n_dims, n_cont, map_size_px, map_res, car_l2, car_w2,
        max_vel, min_vel, RI, max_vert_acc, max_theta, gear_switch_time,
        patch_length_px, patch_width_px, blocks, threads, d_state_instance,
        d_intermediate_states_instance, d_controls_instance, d_valid_instance,
        d_cost_instance, cuda_stream);

    for (int i = 0; i < n_succ; i++) {
      if (valid[i]) {
        // printf("State %d: x=%.3f, y=%.3f, yaw=%.3f, vx=%.3f\n",
        //        i, state[i * n_dims + 0], state[i * n_dims + 1],
        //        state[i * n_dims + 2], state[i * n_dims + 3]);
        memcpy(new_pose, &state[i * n_dims], n_dims * sizeof(float));
        // Allocate intermediate poses dynamically
        float *new_intermediate_pose = new float[timesteps * n_dims];
        memcpy(new_intermediate_pose,
               &intermediate_states[i * timesteps * n_dims],
               timesteps * n_dims * sizeof(float));
        neighbor = std::make_shared<Node>(
            new_pose, new_intermediate_pose, node, node->g + cost[i],
            resolution, tolerance, max_level, division_factor, timesteps,
            local_controllability_radius, time_direction);
        f = neighbor->g + heuristic(neighbor->pose, goal->pose);
        neighbor->f = f;
        neighbors.push_back(neighbor);
      }
    }

    return neighbors;
  }

  // Ensure batched device buffers can hold up to max_batch parents
  // (max_batch * n_succ rollouts) in a single kernel launch. Buffers are grown
  // (never shrunk) and the static control set is pre-tiled max_batch times so
  // the existing launcher/kernel can be reused unchanged.
  void ensure_batch_capacity(int max_batch) {
    if (max_batch <= batch_capacity) {
      return;
    }
    if (d_controls_batched != nullptr) {
      cudaFree(d_controls_batched);
      d_controls_batched = nullptr;
    }
    if (d_state_batched != nullptr) {
      cudaFree(d_state_batched);
      d_state_batched = nullptr;
    }
    if (d_cost_batched != nullptr) {
      cudaFree(d_cost_batched);
      d_cost_batched = nullptr;
    }
    if (d_intermediate_batched != nullptr) {
      cudaFree(d_intermediate_batched);
      d_intermediate_batched = nullptr;
    }
    if (d_valid_batched != nullptr) {
      cudaFree(d_valid_batched);
      d_valid_batched = nullptr;
    }

    int total = max_batch * n_succ;
    cudaMalloc(&d_controls_batched, sizeof(float) * total * n_cont);
    cudaMalloc(&d_state_batched, sizeof(float) * total * n_dims);
    cudaMalloc(&d_intermediate_batched,
               sizeof(float) * total * timesteps * n_dims);
    cudaMalloc(&d_valid_batched, sizeof(bool) * total);
    cudaMalloc(&d_cost_batched, sizeof(float) * total);

    // Tile the static control set (already on the device in
    // d_controls_instance) max_batch times so rollout k reads controls[k*NC].
    for (int b = 0; b < max_batch; b++) {
      cudaMemcpy(d_controls_batched + b * n_succ * n_cont, d_controls_instance,
                 sizeof(float) * n_succ * n_cont, cudaMemcpyDeviceToDevice);
    }
    batch_capacity = max_batch;
  }

  // Batched successor: expands every node in `nodes` (M parents) in a single
  // GPU kernel launch (M * n_succ rollouts) and returns one neighbor list per
  // parent, in the same order as the input.
  std::vector<std::vector<std::shared_ptr<Node>>>
  Succ_batched(const std::vector<std::shared_ptr<Node>> &nodes,
               std::shared_ptr<Node> goal) {
    int M = static_cast<int>(nodes.size());
    std::vector<std::vector<std::shared_ptr<Node>>> results(M);
    if (M == 0) {
      return results;
    }

    if (d_costmap == nullptr || d_heightmap == nullptr) {
      std::cerr << "Error: CUDA memory not allocated. Call set_world() first."
                << std::endl;
      return results;
    }

    ensure_batch_capacity(M);

    int total = M * n_succ;
    std::vector<float> state(static_cast<size_t>(total) * n_dims);
    std::vector<float> intermediate_states(static_cast<size_t>(total) *
                                           timesteps * n_dims);
    std::vector<float> cost(total, 0.0f);
    bool *valid = new bool[total];
    std::fill(valid, valid + total, true);

    for (int i = 0; i < M; i++) {
      for (int s = 0; s < n_succ; s++) {
        int row = i * n_succ + s;
        memcpy(&state[static_cast<size_t>(row) * n_dims], nodes[i]->pose,
               n_dims * sizeof(float));
      }
    }

    int batched_blocks = (total + threads - 1) / threads;
    // std::cout << "Launching kinematic_launcher with " << total << " rollouts" << std::endl;
    kinematic_launcher(
        state.data(), intermediate_states.data(), d_heightmap, d_costmap, valid,
        cost.data(), dt, timesteps, total, n_dims, n_cont, map_size_px, map_res,
        car_l2, car_w2, max_vel, min_vel, RI, max_vert_acc, max_theta,
        gear_switch_time, patch_length_px, patch_width_px, batched_blocks,
        threads, d_state_batched, d_intermediate_batched, d_controls_batched,
        d_valid_batched, d_cost_batched, cuda_stream);

    float new_pose[n_dims];
    for (int i = 0; i < M; i++) {
      for (int s = 0; s < n_succ; s++) {
        int row = i * n_succ + s;
        if (valid[row]) {
          memcpy(new_pose, &state[static_cast<size_t>(row) * n_dims],
                 n_dims * sizeof(float));
          float *new_intermediate_pose = new float[timesteps * n_dims];
          memcpy(new_intermediate_pose,
                 &intermediate_states[static_cast<size_t>(row) * timesteps *
                                      n_dims],
                 timesteps * n_dims * sizeof(float));
          auto neighbor = std::make_shared<Node>(
              new_pose, new_intermediate_pose, nodes[i],
              nodes[i]->g + cost[row], resolution, tolerance, max_level,
              division_factor, timesteps, local_controllability_radius,
              time_direction);
          delete[] new_intermediate_pose;
          neighbor->f = neighbor->g + heuristic(neighbor->pose, goal->pose);
          results[i].push_back(neighbor);
        }
      }
    }

    delete[] valid;
    return results;
  }

  torch::Tensor convert_node_list_to_path_tensor(
      std::vector<std::shared_ptr<Node>> node_list) {
    if (node_list.empty()) {
      return torch::zeros({0, n_dims + 1},
                          torch::TensorOptions().dtype(torch::kFloat32));
    }

    // First pass: build extended list with all states we want to include
    struct PathState {
      float pose[n_dims];
      float g;
    };
    std::vector<PathState> extended_path;
    for (size_t i = 0; i < node_list.size(); i++) {
      if (node_list[i]->parent == nullptr) {
        // Start node (no intermediate poses): skip it
        continue;
      } else {
        // Node with parent: add intermediate poses (in reverse order as stored)
        for (int j = 0; j < timesteps; j++) {
          int intermediate_index = (timesteps - j - 1) * n_dims;
          PathState state;
          for (int d = 0; d < n_dims; d++) {
            state.pose[d] =
                node_list[i]->intermediate_poses[intermediate_index + d];
          }
          state.g = node_list[i]->g * node_list[i]->time_direction;
          extended_path.push_back(state);
        }
      }
    }

    // Second pass: create tensor of exact size and populate it
    int tensor_size = extended_path.size();
    auto path_tensor =
        torch::zeros({tensor_size, n_dims + 1},
                     torch::TensorOptions().dtype(torch::kFloat32));

    for (size_t i = 0; i < extended_path.size(); i++) {
      for (int d = 0; d < n_dims; d++) {
        path_tensor[i][d] = extended_path[i].pose[d];
      }
      path_tensor[i][n_dims] = extended_path[i].g;
    }

    return path_tensor;
  }
};
