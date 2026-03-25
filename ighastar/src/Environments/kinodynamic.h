#include <boost/functional/hash.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <random>
#include <torch/extension.h>

using namespace std;

constexpr int n_dims = 4;
constexpr int n_cont = 2;
constexpr int warp_size = 32;

// Launches kinodynamic simulation for multiple rollouts on GPU
// Uses per-instance device memory and stream for parallel execution
void kinodynamic_launcher(float *state, float *intermediate_states,
                          const float *heightmap, const float *costmap,
                          bool *valid, float *cost, float dt, int timesteps,
                          int rollouts, int n_dims, int n_cont,
                          const int map_size_px, float map_res, float car_l2,
                          float car_w2, float max_vel, float min_vel, float RI,
                          float max_vert_acc, float max_theta,
                          float gear_switch_time, int patch_length_px,
                          int patch_width_px, const int blocks, const int threads,
                          float *d_state, float *d_intermediate_states,
                          float *d_controls, bool *d_valid, float *d_cost,
                          cudaStream_t stream);

// Checks validity of multiple states against a bitmap on GPU
void check_validity_launcher(const float *bitmap, int map_size_px,
                             float map_res, float *states, int patch_length_px,
                             int patch_width_px, float car_l2, float car_w2,
                             int n_states, int n_dims, bool *result);

size_t calc_hash(float *pose, const float *resolution) {
  std::size_t hash_val =
      0; // this is the "seed", replace with #define value later
  for (int i = 0; i < n_dims; ++i) {
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
  std::vector<size_t> index;
  size_t LCR_index;
  std::shared_ptr<Node> parent;
  int time_direction;
  // Constructor
  Node(const float *pose_in, const float *intermediate_poses_,
       std::shared_ptr<Node> parent_in, float g_in, const float *resolution_,
       float *tolerance, int max_level, float division_factor, int timesteps,
       const float *LCR_, int time_direction_ = 1)
      : g(g_in), f(0), parent(parent_in), active(true), rank(0), level(0),
        intermediate_poses(nullptr), time_direction(time_direction_) {
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
          memcpy(&intermediate_poses[dst_index], &intermediate_poses_[src_index],
                 n_dims * sizeof(float));
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
  float *d_heightmap, *d_costmap;
  float dt, map_res;
  float car_l2, car_w2;
  float max_vel, min_vel, RI, max_theta, max_vert_acc, gear_switch_time;

  // Per-instance CUDA device memory (allows parallel execution without conflicts)
  float *d_controls_instance;
  float *d_state_instance;
  float *d_cost_instance;
  float *d_intermediate_states_instance;
  bool *d_valid_instance;

  // Per-instance CUDA stream (allows parallel kernel execution)
  cudaStream_t cuda_stream;

public:
  float resolution[n_dims], epsilon[n_dims], division_factor;
  float local_controllability_radius[n_dims];
  float tolerance[n_dims];
  int max_level;
  int timesteps;
  int time_direction;
  Environment(const py::dict &config, int time_direction_ = 1)
      : d_heightmap(nullptr), d_costmap(nullptr), time_direction(time_direction_),
        d_controls_instance(nullptr), d_state_instance(nullptr),
        d_cost_instance(nullptr), d_intermediate_states_instance(nullptr),
        d_valid_instance(nullptr) {
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
    // Free per-instance CUDA device memory
    if (d_controls_instance) cudaFree(d_controls_instance);
    if (d_state_instance) cudaFree(d_state_instance);
    if (d_cost_instance) cudaFree(d_cost_instance);
    if (d_intermediate_states_instance) cudaFree(d_intermediate_states_instance);
    if (d_valid_instance) cudaFree(d_valid_instance);
    
    // Destroy CUDA stream
    if (cuda_stream) cudaStreamDestroy(cuda_stream);
    
    if (d_costmap != nullptr) {
      cudaFree(d_costmap);
      d_costmap = nullptr;
    }
    if (d_heightmap != nullptr) {
      cudaFree(d_heightmap);
      d_heightmap = nullptr;
    }
  }

  // Sets resolution, tolerance, and epsilon values for different dimensions
  void set_resolutions(py::dict &info, py::dict &node_info) {
    float res = info["resolution"].cast<float>();
    float tol = info["tolerance"].cast<float>();
    auto eps = info["epsilon"].cast<std::vector<float>>();
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
    epsilon[0] = eps[0];
    epsilon[1] = eps[1];
    epsilon[2] = eps[2];
    epsilon[3] = eps[3];
    // Read LCR (Local Controllability Radius) for anchor resolution
    if (info.contains("LCR")) {
      auto lcr = info["LCR"].cast<std::vector<float>>();
      local_controllability_radius[0] = lcr[0];
      local_controllability_radius[1] = lcr[1];
      local_controllability_radius[2] = lcr[2];
      local_controllability_radius[3] = lcr[3];
    } else {
      // Default: use epsilon as local_controllability_radius
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
      std::cerr << "CUDA error allocating d_controls_instance: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMalloc(&d_state_instance, n_succ * n_dims * sizeof(float));
    if (err != cudaSuccess) {
      std::cerr << "CUDA error allocating d_state_instance: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMalloc(&d_intermediate_states_instance, n_succ * timesteps * n_dims * sizeof(float));
    if (err != cudaSuccess) {
      std::cerr << "CUDA error allocating d_intermediate_states_instance: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMalloc(&d_valid_instance, n_succ * sizeof(bool));
    if (err != cudaSuccess) {
      std::cerr << "CUDA error allocating d_valid_instance: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMalloc(&d_cost_instance, n_succ * sizeof(float));
    if (err != cudaSuccess) {
      std::cerr << "CUDA error allocating d_cost_instance: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMemcpy(d_controls_instance, controls, sizeof(float) * n_succ * n_cont, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      std::cerr << "CUDA error copying controls: " << cudaGetErrorString(err) << std::endl;
    }
    // Create CUDA stream for parallel execution
    err = cudaStreamCreate(&cuda_stream);
    if (err != cudaSuccess) {
      std::cerr << "CUDA error creating stream: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();
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
                                  timesteps, local_controllability_radius, time_direction);
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
        if (fabs(v->pose[3] - goal->pose[3]) < epsilon[3]) {
          return true;
        }
      }
    }
    return false;
  }

  // Calculates heuristic value for A* search
  float heuristic(float *pose, float *goal) {
    return distance(pose, goal) / max_vel;
  }

  // Compute distance between two poses for near-meet checking in bidirectional search
  // Note that this implementation is left up to the user and can be different for different environments.
  // The search method does not care "how" this is computed, as long as it is a valid distance metric.
  float compute_near_meet_distance(float *pose1, float *pose2) {
    return heuristic(pose1, pose2);
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

  // Sample states around a start pose within epsilon bounds
  // Returns vector of valid Node pointers
  std::vector<std::shared_ptr<Node>> sample_valid_states_around_start(
      std::shared_ptr<Node> start_node, std::shared_ptr<Node> goal_node,
      float epsilon_multiplier = 1.0f, int target_count = 100,
      const float *sampling_epsilon = nullptr) {
    std::vector<std::shared_ptr<Node>> valid_nodes;

    // Generate random sampled states
    std::vector<float *> sampled_states;
    std::vector<float> state_storage;

    state_storage.reserve(target_count * 4);
    sampled_states.reserve(target_count);

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Use Gaussian distributions centered at 0 with standard deviation =
    // epsilon * epsilon_multiplier
    std::normal_distribution<float> dist_cte(
        0.0f, sampling_epsilon[0] * epsilon_multiplier);
    std::normal_distribution<float> dist_ate(
        0.0f, sampling_epsilon[1] * epsilon_multiplier);
    std::normal_distribution<float> dist_yaw(
        0.0f, sampling_epsilon[2] * epsilon_multiplier);
    std::normal_distribution<float> dist_vel(
        0.0f, sampling_epsilon[3] * epsilon_multiplier);

    // Pre-compute bounds
    float clearance = sqrt(car_l2 * car_l2 + car_w2 * car_w2);
    float min_x = clearance;
    float max_x = (map_size_px * map_res) - clearance;
    float min_y = clearance;
    float max_y = (map_size_px * map_res) - clearance;
    float epsilon_ball =
        sqrt(sampling_epsilon[0] * sampling_epsilon[0] / 2 +
             sampling_epsilon[1] * sampling_epsilon[1] / 2);

    // Sample target_count random states
    for (int i = 0; i < target_count; i++) {
      float cte = dist_cte(gen);
      float ate = dist_ate(gen);
      float dyaw = dist_yaw(gen);
      float dvel = dist_vel(gen);

      // Transform from goal's local frame to world coordinates
      float pose_yaw = start_node->pose[2];
      float cos_yaw = cosf(pose_yaw);
      float sin_yaw = sinf(pose_yaw);

      float world_dx = cte * cos_yaw - ate * sin_yaw;
      float world_dy = cte * sin_yaw + ate * cos_yaw;

      float sampled_pose[4] = {start_node->pose[0] + world_dx,
                               start_node->pose[1] + world_dy,
                               start_node->pose[2] + dyaw,
                               start_node->pose[3] + dvel};

      // Clamp to valid bounds
      sampled_pose[0] = std::clamp(sampled_pose[0], min_x, max_x);
      sampled_pose[1] = std::clamp(sampled_pose[1], min_y, max_y);
      sampled_pose[0] =
          std::clamp(sampled_pose[0], start_node->pose[0] - epsilon_ball,
                     start_node->pose[0] + epsilon_ball);
      sampled_pose[1] =
          std::clamp(sampled_pose[1], start_node->pose[1] - epsilon_ball,
                     start_node->pose[1] + epsilon_ball);

      // Handle yaw wrap-around
      const float PI = 3.14159265358979323846f;
      while (sampled_pose[2] > PI)
        sampled_pose[2] -= 2.0f * PI;
      while (sampled_pose[2] < -PI)
        sampled_pose[2] += 2.0f * PI;
      // Clamp velocity
      sampled_pose[3] = std::clamp(sampled_pose[3], epsilon[3], epsilon[3]);

      // Store state data
      int idx = state_storage.size();
      state_storage.resize(idx + 4);
      memcpy(&state_storage[idx], sampled_pose, 4 * sizeof(float));
      sampled_states.push_back(&state_storage[idx]);
    }

    // Check validity in batch
    std::vector<bool> validities;
    check_validity_batched(sampled_states, validities);

    // Create nodes for valid states
    for (size_t i = 0; i < sampled_states.size(); i++) {
      if (validities[i]) {
        float *sampled_pose = sampled_states[i];

        // Interpolate intermediate states between start_node and sampled_pose
        float *intermediate_poses = new float[timesteps * n_dims];
        for (int t = 0; t < timesteps; t++) {
          float alpha = static_cast<float>(t + 1) / static_cast<float>(timesteps);
          for (int d = 0; d < n_dims; d++) {
            if (d == 2) {
              // For yaw, handle wrap-around
              float start_yaw = start_node->pose[2];
              float end_yaw = sampled_pose[2];
              float diff = end_yaw - start_yaw;
              const float PI = 3.14159265358979323846f;
              while (diff > PI)
                diff -= 2.0f * PI;
              while (diff < -PI)
                diff += 2.0f * PI;
              intermediate_poses[t * n_dims + d] = start_yaw + alpha * diff;
            } else {
              intermediate_poses[t * n_dims + d] =
                  start_node->pose[d] +
                  alpha * (sampled_pose[d] - start_node->pose[d]);
            }
          }
        }

        // Calculate g value
        float g_value = heuristic(start_node->pose, sampled_pose);

        // Create node
        auto node = std::make_shared<Node>(
            sampled_pose, intermediate_poses, start_node, start_node->g + g_value,
            resolution, tolerance, max_level, division_factor, timesteps,
            local_controllability_radius, time_direction);

        delete[] intermediate_poses;

        // Set f value
        float f_value = node->g + heuristic(node->pose, goal_node->pose);
        node->f = f_value;

        valid_nodes.push_back(node);
      }
    }

    return valid_nodes;
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
    // No mutex needed - each Environment instance has its own CUDA device memory and stream
    kinodynamic_launcher(state, intermediate_states, d_heightmap, d_costmap,
                         valid, cost, dt, timesteps, n_succ, n_dims, n_cont,
                         map_size_px, map_res, car_l2, car_w2, max_vel, min_vel,
                         RI, max_vert_acc, max_theta, gear_switch_time,
                         patch_length_px, patch_width_px, blocks, threads,
                         d_state_instance, d_intermediate_states_instance,
                         d_controls_instance, d_valid_instance, d_cost_instance,
                         cuda_stream);

    for (int i = 0; i < n_succ; i++) {
      if (valid[i]) {
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
        delete[] new_intermediate_pose;
        f = neighbor->g + heuristic(neighbor->pose, goal->pose);
        neighbor->f = f;
        neighbors.push_back(neighbor);
      }
    }

    return neighbors;
  }

  torch::Tensor convert_node_list_to_path_tensor(
      std::vector<std::shared_ptr<Node>> node_list) {
    if (node_list.empty()) {
      return torch::zeros({0, n_dims + 1},
                          torch::TensorOptions().dtype(torch::kFloat32));
    }

    // First pass: build extended list with all states we want to include
    struct PathState {
      float pose[4];
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
    auto path_tensor = torch::zeros({tensor_size, n_dims + 1},
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
