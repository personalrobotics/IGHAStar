#include <algorithm>
#include <boost/functional/hash.hpp>
#include <cfloat>
#include <cmath>
#include <pybind11/stl.h>
#include <torch/extension.h>

using namespace std;

constexpr int n_dims = 4;
constexpr int hash_dims = 4;
constexpr int n_cont = 2;

// CPU implementations of CUDA functions
// Launches kinodynamic simulation for multiple rollouts on CPU
void kinodynamic_launcher_cpu(
    float *state, float *intermediate_states, const float *heightmap,
    const float *costmap, bool *valid, float *cost, float dt, int timesteps,
    int rollouts, int n_dims, int n_cont, const int map_size_px, float map_res,
    float car_l2, float car_w2, float max_vel, float min_vel, float RI,
    float max_vert_acc, float max_theta, float gear_switch_time,
    int patch_length_px, int patch_width_px, const int blocks,
    const int threads, const float *controls);

// Checks validity of multiple states against a bitmap on CPU
void check_validity_launcher_cpu(const float *bitmap, int map_size_px,
                                 float map_res, float *states,
                                 int patch_length_px, int patch_width_px,
                                 float car_l2, float car_w2, int n_states,
                                 int n_dims, bool *result);

size_t calc_hash(float *pose, const float *resolution) {
  std::size_t hash_val = 0;
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
  std::vector<size_t> index;
  std::shared_ptr<Node> parent;

  // Constructor
  Node(const float *pose_in, const float *intermediate_poses_,
       std::shared_ptr<Node> parent_in, float g_in, const float *resolution_,
       float *tolerance, int max_level, float division_factor, int timesteps)
      : intermediate_poses(nullptr), g(g_in), f(0), parent(parent_in),
        active(true), rank(0), level(0) {
    for (int i = 0; i < n_dims; i++) {
      pose[i] = pose_in[i];
    }
    if (parent_in != nullptr && intermediate_poses_ != nullptr) {
      intermediate_poses = new float[timesteps * n_dims];
      memcpy(intermediate_poses, intermediate_poses_,
             timesteps * n_dims * sizeof(float));
    }
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
  int map_size_px, threads, blocks, n_succ, timesteps, patch_length_px,
      patch_width_px;
  float resolution[n_dims], tolerance[n_dims], epsilon[n_dims], division_factor;
  float *h_heightmap, *h_costmap; // CPU versions
  float dt, map_res;
  float car_l2, car_w2;
  float max_vel, min_vel, RI, max_theta, max_vert_acc, gear_switch_time;
  float *controls; // Store controls for CPU version

public:
  int max_level;

  Environment(const py::dict &config) {
    auto info = config["experiment_info_default"].cast<py::dict>();
    auto node_info = info["node_info"].cast<py::dict>();

    // Top-level fields
    map_res = node_info["map_res"].cast<float>();
    max_level = info["max_level"].cast<int>();
    division_factor = info["division_factor"].cast<float>();

    set_resolutions(info, node_info);
    set_car_params(node_info);

    threads = 1; // CPU version doesn't need thread/block calculations
    blocks = 1;

    h_heightmap = nullptr;
    h_costmap = nullptr;
    // controls is set in set_car_params, don't overwrite it here
  }

  ~Environment() {
    if (h_costmap)
      delete[] h_costmap;
    if (h_heightmap)
      delete[] h_heightmap;
    if (controls)
      delete[] controls;
  }

  // Sets resolution, tolerance, and epsilon values for different dimensions
  void set_resolutions(py::dict &info, py::dict &node_info) {
    float res = info["resolution"].cast<float>();
    float tol = info["tolerance"].cast<float>();
    auto eps = info["epsilon"].cast<std::vector<float>>();
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
  }

  // Sets car parameters and allocates controls array
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
    dt = info["dt"].cast<float>();
    gear_switch_time = info["gear_switch_time"].cast<float>();

    auto steering_list = info["steering_list"].cast<std::vector<float>>();
    auto throttle_list = info["throttle_list"].cast<std::vector<float>>();
    n_succ = int(steering_list.size()) * int(throttle_list.size());
    float step_size = info["step_size"].cast<float>();

    controls = new float[n_succ * n_cont];
    int idx = 0;
    for (int i = 0; i < throttle_list.size(); ++i) {
      for (int j = 0; j < steering_list.size(); ++j) {
        controls[idx * n_cont] = tanf(steering_list[j] / 57.3) / (car_l2 * 2);
        controls[idx * n_cont + 1] = throttle_list[i] * step_size;
        ++idx;
      }
    }
  }

  // Sets the world map (costmap and heightmap) from a PyTorch tensor
  void set_world(torch::Tensor world) {
    TORCH_CHECK(world.dim() == 3, "World tensor must be 3D (H x W x 2)");
    TORCH_CHECK(world.size(2) == 2,
                "Last dimension must have size 2 (costmap + heightmap)");
    TORCH_CHECK(world.dtype() == torch::kFloat32,
                "World tensor must be float32");
    TORCH_CHECK(world.device().is_cpu(), "World tensor must be on CPU");

    int H = world.size(0);
    int W = world.size(1);
    int map_size = H * W;

    auto costmap =
        world.index({torch::indexing::Slice(), torch::indexing::Slice(), 0})
            .contiguous();
    auto heightmap =
        world.index({torch::indexing::Slice(), torch::indexing::Slice(), 1})
            .contiguous();

    // Allocate CPU memory
    if (h_costmap)
      delete[] h_costmap;
    if (h_heightmap)
      delete[] h_heightmap;
    h_costmap = new float[map_size];
    h_heightmap = new float[map_size];

    // Copy data
    memcpy(h_costmap, costmap.data_ptr<float>(), map_size * sizeof(float));
    memcpy(h_heightmap, heightmap.data_ptr<float>(), map_size * sizeof(float));

    map_size_px = H;

    // Verify allocation
    if (!h_costmap || !h_heightmap) {
      throw std::runtime_error("Failed to allocate map memory");
    }
  }

  // Cleanup function for CPU environment (no-op)
  void cleanup() { return; }

  // Creates a new Node with the given pose
  std::shared_ptr<Node> create_Node(float *pose) {
    return std::make_shared<Node>(pose, nullptr, nullptr, 0, resolution,
                                  tolerance, max_level, division_factor,
                                  timesteps);
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

  // Checks validity of start and goal positions
  void check_validity(float *start, float *goal, bool *result) {
    float states[2 * n_dims];
    memcpy(states, start, n_dims * sizeof(float));
    memcpy(states + n_dims, goal, n_dims * sizeof(float));
    check_validity_launcher_cpu(h_costmap, map_size_px, map_res, states,
                                patch_length_px, patch_width_px, car_l2, car_w2,
                                2, n_dims, result);
  }

  // function that returns a vector of nodes:
  std::vector<std::shared_ptr<Node>> Succ(std::shared_ptr<Node> node,
                                          std::shared_ptr<Node> goal) {
    std::vector<std::shared_ptr<Node>> neighbors;
    std::shared_ptr<Node> neighbor;
    float new_pose[n_dims], f;

    // Individual safety checks for initialized maps
    if (!h_costmap) {
      printf("ERROR: h_costmap pointer is null\n");
      return neighbors; // Return empty vector if maps not initialized
    }
    if (!h_heightmap) {
      printf("ERROR: h_heightmap pointer is null\n");
      return neighbors; // Return empty vector if maps not initialized
    }

    float state[n_succ * n_dims];
    float intermediate_states[timesteps * n_succ * n_dims];
    bool valid[n_succ];
    float cost[n_succ];
    std::fill(valid, valid + n_succ, true);
    std::fill(cost, cost + n_succ, 0.0);

    for (int i = 0; i < n_succ; i++) {
      memcpy(&state[i * n_dims], node->pose, n_dims * sizeof(float));
    }

    kinodynamic_launcher_cpu(
        state, intermediate_states, h_heightmap, h_costmap, valid, cost, dt,
        timesteps, n_succ, n_dims, n_cont, map_size_px, map_res, car_l2, car_w2,
        max_vel, min_vel, RI, max_vert_acc, max_theta, gear_switch_time,
        patch_length_px, patch_width_px, blocks, threads, controls);

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
            resolution, tolerance, max_level, division_factor, timesteps);
        f = neighbor->g + heuristic(neighbor->pose, goal->pose);
        neighbor->f = f;
        neighbors.push_back(neighbor);
      }
    }

    return neighbors;
  }

  torch::Tensor convert_node_list_to_path_tensor(
      std::vector<std::shared_ptr<Node>> node_list) {
    int path_length = node_list.size();
    auto path_tensor =
        torch::zeros({1 + (path_length - 1) * timesteps, n_dims + 1},
                     torch::TensorOptions().dtype(torch::kFloat32));
    for (int i = 0; i < path_length; i++) {
      int base_index = i * timesteps;
      if (node_list[i]->parent == nullptr) {
        // start node doesn't have any intermediates:
        path_tensor[base_index][0] = node_list[i]->pose[0];
        path_tensor[base_index][1] = node_list[i]->pose[1];
        path_tensor[base_index][2] = node_list[i]->pose[2];
        path_tensor[base_index][3] = node_list[i]->pose[3];
        path_tensor[base_index][4] =
            node_list[i]->g; // this is the timestamp of that node
        break;
      }
      for (int j = 0; j < timesteps; j++) {
        int intermediate_index = (timesteps - j - 1) * n_dims;
        path_tensor[base_index + j][0] =
            node_list[i]->intermediate_poses[intermediate_index + 0];
        path_tensor[base_index + j][1] =
            node_list[i]->intermediate_poses[intermediate_index + 1];
        path_tensor[base_index + j][2] =
            node_list[i]->intermediate_poses[intermediate_index + 2];
        path_tensor[base_index + j][3] =
            node_list[i]->intermediate_poses[intermediate_index + 3];
        path_tensor[base_index + j][4] =
            node_list[i]->g; // this is the timestamp of that node
      }
    }
    return path_tensor; // this might cause segfault
  }
};