// Generic, OMPL-style environment for IGHAStar.
//
// Unlike the hand-crafted car environments, this environment is configured
// entirely at runtime from user-provided Python callbacks:
//   - sample_controls() -> [K, n_cont]
//   - dynamics(states[B, n_dims], controls[B, n_cont]) -> next_states[B, n_dims]
//   - cost(states[B, n_dims], controls[B, n_cont], next_states[B, n_dims]) -> [B]
//   - validity(states[B, n_dims]) -> [B]   (nonzero == valid)
//   - heuristic(states[B, n_dims]) -> [B]  (cost-to-go; encodes goal knowledge)
//   - goal_test(state[n_dims]) -> bool
//
// State / control dimensionality is fixed at (JIT) compile time via the
// N_DIMS / N_CONT / HASH_DIMS macros, mirroring how the other environments
// hard-code these as compile-time constants. This keeps Node and the planning
// core unchanged.
#include <boost/functional/hash.hpp>
#include <cmath>
#include <cstring>
#include <memory>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>

using namespace std;

// Compile-time dimensions (overridden via -DN_DIMS / -DN_CONT / -DHASH_DIMS).
#ifndef N_DIMS
#define N_DIMS 4
#endif
#ifndef N_CONT
#define N_CONT 2
#endif
#ifndef HASH_DIMS
#define HASH_DIMS N_DIMS
#endif

constexpr int n_dims = N_DIMS;
constexpr int hash_dims = HASH_DIMS;
constexpr int n_cont = N_CONT;

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
  // User-provided Python callbacks.
  py::function sample_controls_fn;
  py::function dynamics_fn;
  py::function cost_fn;
  py::function validity_fn;
  py::function heuristic_fn;
  py::function goal_test_fn;

  int n_succ; // branching factor K (controls applied per expansion)

public:
  // Fields referenced by the planning core / bidirectional machinery.
  int max_level;
  int timesteps;
  int time_direction;
  float resolution[n_dims];
  float tolerance[n_dims];
  float local_controllability_radius[n_dims];
  float lower[n_dims];
  float upper[n_dims];
  float division_factor;

  Environment(const py::dict &config, int time_direction_ = 1)
      : time_direction(time_direction_) {
    auto info = config["experiment_info_default"].cast<py::dict>();

    max_level = info["max_level"].cast<int>();
    division_factor = info["division_factor"].cast<float>();
    n_succ = info.contains("num_controls") ? info["num_controls"].cast<int>()
                                           : 1;

    auto res = info["resolution"].cast<std::vector<float>>();
    auto tol = info["tolerance"].cast<std::vector<float>>();
    for (int i = 0; i < n_dims; i++) {
      resolution[i] = res[i];
      tolerance[i] = tol[i];
    }

    // Optional state-space bounds (used to reject out-of-bounds children if the
    // user does not handle bounds in their validity callback).
    if (info.contains("bounds_lower") && info.contains("bounds_upper")) {
      auto lo = info["bounds_lower"].cast<std::vector<float>>();
      auto hi = info["bounds_upper"].cast<std::vector<float>>();
      for (int i = 0; i < n_dims; i++) {
        lower[i] = lo[i];
        upper[i] = hi[i];
      }
    } else {
      for (int i = 0; i < n_dims; i++) {
        lower[i] = -1e9f;
        upper[i] = 1e9f;
      }
    }

    // Local controllability radius (for bidirectional near-meet); defaults to
    // resolution.
    if (info.contains("LCR")) {
      auto lcr = info["LCR"].cast<std::vector<float>>();
      for (int i = 0; i < n_dims; i++) {
        local_controllability_radius[i] = lcr[i];
      }
    } else {
      for (int i = 0; i < n_dims; i++) {
        local_controllability_radius[i] = resolution[i];
      }
    }

    // timesteps only affects intermediate-pose storage (unused here; generic
    // nodes store a single state per node).
    if (info.contains("node_info")) {
      auto node_info = info["node_info"].cast<py::dict>();
      timesteps = node_info.contains("timesteps")
                      ? node_info["timesteps"].cast<int>()
                      : 1;
    } else {
      timesteps = 1;
    }

    // User callbacks (passed at the top level of the config dict).
    sample_controls_fn = config["sample_controls_fn"].cast<py::function>();
    dynamics_fn = config["dynamics_fn"].cast<py::function>();
    cost_fn = config["cost_fn"].cast<py::function>();
    validity_fn = config["validity_fn"].cast<py::function>();
    heuristic_fn = config["heuristic_fn"].cast<py::function>();
    goal_test_fn = config["goal_test_fn"].cast<py::function>();
  }

  ~Environment() {}

  // Creates a new Node with the given pose (start / goal nodes).
  std::shared_ptr<Node> create_Node(float *pose) {
    return std::make_shared<Node>(pose, nullptr, nullptr, 0, resolution,
                                  tolerance, max_level, division_factor,
                                  timesteps, local_controllability_radius,
                                  time_direction);
  }

  // Convert a Python-returned object to a contiguous CPU float32 tensor.
  static torch::Tensor to_cpu_float(py::handle obj) {
    return obj.cast<torch::Tensor>()
        .to(torch::kCPU)
        .to(torch::kFloat32)
        .contiguous();
  }

  // The generic env does not own a world tensor; the world (e.g. an MJX model)
  // lives in Python and is captured by the user's callbacks.
  void set_world(torch::Tensor world) { (void)world; }

  // Heuristic for a single state (used to initialize the start node). The
  // goal argument is ignored: the user's heuristic encodes goal knowledge.
  float heuristic(float *pose, float *goal) {
    (void)goal;
    auto state_t = torch::empty({1, n_dims}, torch::kFloat32);
    memcpy(state_t.data_ptr<float>(), pose, n_dims * sizeof(float));
    torch::Tensor h = to_cpu_float(heuristic_fn(state_t));
    return h.data_ptr<float>()[0];
  }

  // Distance metric used by bidirectional near-meet checks (Euclidean stub).
  float compute_near_meet_distance(float *pose1, float *pose2) {
    float s = 0.0f;
    for (int i = 0; i < n_dims; i++) {
      float d = pose1[i] - pose2[i];
      s += d * d;
    }
    return std::sqrt(s);
  }

  // Goal test for a single popped node.
  bool reached_goal_region(std::shared_ptr<Node> v, std::shared_ptr<Node> goal) {
    (void)goal;
    auto state_t = torch::empty({n_dims}, torch::kFloat32);
    memcpy(state_t.data_ptr<float>(), v->pose, n_dims * sizeof(float));
    return goal_test_fn(state_t).cast<bool>();
  }

  // Validity of start and goal positions.
  void check_validity(float *start, float *goal, bool *result) {
    auto states_t = torch::empty({2, n_dims}, torch::kFloat32);
    float *sp = states_t.data_ptr<float>();
    memcpy(sp, start, n_dims * sizeof(float));
    memcpy(sp + n_dims, goal, n_dims * sizeof(float));
    torch::Tensor v = to_cpu_float(validity_fn(states_t));
    const float *vp = v.data_ptr<float>();
    result[0] = vp[0] != 0.0f;
    result[1] = vp[1] != 0.0f;
  }

  // Batched validity check for multiple states.
  void check_validity_batched(const std::vector<float *> &states,
                              std::vector<bool> &results) {
    int n = static_cast<int>(states.size());
    results.resize(n);
    if (n == 0) {
      return;
    }
    auto t = torch::empty({n, n_dims}, torch::kFloat32);
    float *p = t.data_ptr<float>();
    for (int i = 0; i < n; i++) {
      memcpy(p + i * n_dims, states[i], n_dims * sizeof(float));
    }
    torch::Tensor v = to_cpu_float(validity_fn(t));
    const float *vp = v.data_ptr<float>();
    for (int i = 0; i < n; i++) {
      results[i] = vp[i] != 0.0f;
    }
  }

  // Generic env does not need device-side batch buffers; batching memory lives
  // in Python (e.g. JAX/MJX).
  void ensure_batch_capacity(int max_batch) { (void)max_batch; }

  // Batched successor: expands every node in `nodes` (M parents) by applying
  // the K user-sampled controls to each, in a single batched callback round.
  // Returns one neighbor list per parent, in input order.
  std::vector<std::vector<std::shared_ptr<Node>>>
  Succ_batched(const std::vector<std::shared_ptr<Node>> &nodes,
               std::shared_ptr<Node> goal) {
    (void)goal;
    int M = static_cast<int>(nodes.size());
    std::vector<std::vector<std::shared_ptr<Node>>> results(M);
    if (M == 0) {
      return results;
    }

    // Sample the control set for this batch (user decides fixed vs random).
    torch::Tensor controls_t = to_cpu_float(sample_controls_fn());
    int K = static_cast<int>(controls_t.size(0));
    if (K == 0) {
      return results;
    }
    const float *cp = controls_t.data_ptr<float>();

    int B = M * K;
    auto states_t = torch::empty({B, n_dims}, torch::kFloat32);
    auto controls_batch_t = torch::empty({B, n_cont}, torch::kFloat32);
    float *sp = states_t.data_ptr<float>();
    float *cbp = controls_batch_t.data_ptr<float>();
    for (int i = 0; i < M; i++) {
      for (int k = 0; k < K; k++) {
        int row = i * K + k;
        memcpy(sp + static_cast<size_t>(row) * n_dims, nodes[i]->pose,
               n_dims * sizeof(float));
        memcpy(cbp + static_cast<size_t>(row) * n_cont, cp + k * n_cont,
               n_cont * sizeof(float));
      }
    }

    // Dynamics. The callback may return either:
    //   [B, n_dims]            -> a single resulting state (rollout_T = 1), or
    //   [B, rollout_T, n_dims] -> the full sub-trajectory per control.
    torch::Tensor next_t = to_cpu_float(dynamics_fn(states_t, controls_batch_t));
    int rollout_T = (next_t.dim() == 3) ? static_cast<int>(next_t.size(1)) : 1;
    // The node's final state is the last sub-step of the rollout.
    torch::Tensor final_t =
        (next_t.dim() == 3) ? next_t.select(1, rollout_T - 1).contiguous()
                            : next_t;
    // Keep the env's timesteps in sync so the path expansion below matches.
    timesteps = rollout_T;

    // cost / validity / heuristic operate on the final resulting state.
    torch::Tensor cost_t =
        to_cpu_float(cost_fn(states_t, controls_batch_t, final_t));
    torch::Tensor valid_t = to_cpu_float(validity_fn(final_t));
    torch::Tensor h_t = to_cpu_float(heuristic_fn(final_t));

    const float *rollp = next_t.data_ptr<float>(); // rollout (or final if 2D)
    const float *fp = final_t.data_ptr<float>();   // [B, n_dims]
    const float *costp = cost_t.data_ptr<float>();
    const float *validp = valid_t.data_ptr<float>();
    const float *hp = h_t.data_ptr<float>();

    float new_pose[n_dims];
    for (int i = 0; i < M; i++) {
      for (int k = 0; k < K; k++) {
        int row = i * K + k;
        if (validp[row] == 0.0f) {
          continue;
        }
        memcpy(new_pose, fp + static_cast<size_t>(row) * n_dims,
               n_dims * sizeof(float));
        // Pointer to this row's rollout (rollout_T * n_dims floats); for the
        // 2D case this is just the single final state.
        const float *roll =
            rollp + static_cast<size_t>(row) * rollout_T * n_dims;
        auto neighbor = std::make_shared<Node>(
            new_pose, roll, nodes[i], nodes[i]->g + costp[row], resolution,
            tolerance, max_level, division_factor, rollout_T,
            local_controllability_radius, time_direction);
        neighbor->f = neighbor->g + hp[row];
        results[i].push_back(neighbor);
      }
    }
    return results;
  }

  // Single-node successor (delegates to the batched path).
  std::vector<std::shared_ptr<Node>> Succ(std::shared_ptr<Node> node,
                                          std::shared_ptr<Node> goal) {
    std::vector<std::shared_ptr<Node>> batch = {node};
    auto results = Succ_batched(batch, goal);
    if (results.empty()) {
      return {};
    }
    return results[0];
  }

  // Build a [path_len, n_dims + 1] tensor of states + g for the path. Each
  // non-start node contributes its stored rollout (timesteps sub-steps), so the
  // returned trajectory is dense when dynamics returns a full rollout, or one
  // state per node when it returns a single next state (timesteps == 1).
  torch::Tensor
  convert_node_list_to_path_tensor(std::vector<std::shared_ptr<Node>> node_list) {
    struct PathState {
      float pose[n_dims];
      float g;
    };
    std::vector<PathState> extended_path;
    for (size_t i = 0; i < node_list.size(); i++) {
      std::shared_ptr<Node> node = node_list[i];
      // Start node (and any node without a stored rollout) contributes nothing.
      if (node->parent == nullptr || node->intermediate_poses == nullptr) {
        continue;
      }
      for (int j = 0; j < timesteps; j++) {
        int intermediate_index = (timesteps - j - 1) * n_dims;
        PathState s;
        for (int d = 0; d < n_dims; d++) {
          s.pose[d] = node->intermediate_poses[intermediate_index + d];
        }
        s.g = node->g * node->time_direction;
        extended_path.push_back(s);
      }
    }

    int n = static_cast<int>(extended_path.size());
    auto path_tensor =
        torch::zeros({n, n_dims + 1}, torch::TensorOptions().dtype(torch::kFloat32));
    if (n == 0) {
      return path_tensor;
    }
    auto a = path_tensor.accessor<float, 2>();
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < n_dims; d++) {
        a[i][d] = extended_path[i].pose[d];
      }
      a[i][n_dims] = extended_path[i].g;
    }
    return path_tensor;
  }
};
