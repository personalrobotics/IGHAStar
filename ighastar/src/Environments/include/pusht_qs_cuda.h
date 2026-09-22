// PushT Mason/Lynch QS on CUDA — separate env for A/B vs CPU pusht (Chipmunk/QS).
// State [10] / control [2] match pusht.h. Succ batches rollouts on GPU.
#pragma once

#include <algorithm>
#include <boost/functional/hash.hpp>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <memory>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <torch/extension.h>
#include <vector>

#include <pusht_qs.h>

using namespace std;

constexpr int n_dims = pusht_qs::kStateDim;
constexpr int hash_dims = 5;
constexpr int n_cont = 2;

// Defined in pusht_qs.cu
extern "C" void pusht_qs_launcher(float *h_states, float *h_intermediates,
                                  const float *h_actions, bool *h_valid,
                                  int n_substeps, int rollouts,
                                  float max_agent_vel, float support_radius,
                                  float workspace_lo, float workspace_hi,
                                  float max_penetration_px, float *d_states,
                                  float *d_intermediates, float *d_actions,
                                  bool *d_valid, int blocks, int threads,
                                  cudaStream_t stream);

inline size_t calc_hash(float *pose, const float *resolution) {
  std::size_t hash_val = 0;
  for (int i = 0; i < hash_dims; ++i) {
    boost::hash_combine(hash_val, std::hash<size_t>{}(static_cast<int64_t>(
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
  size_t LCR_index;
  int time_direction;
  std::vector<size_t> index;
  std::shared_ptr<Node> parent;
  bool preexpanded = false;
  std::vector<std::shared_ptr<Node>> cached_succ;
  // Expansion bookkeeping for tree export / visualization.
  int discover_index = -1;
  int expand_index = -1;
  float control[n_cont] = {};
  float *intermediate_controls = nullptr;

  Node(const float *pose_in, const float *intermediate_poses_,
       std::shared_ptr<Node> parent_in, float g_in, const float *resolution_,
       const float *tolerance, int max_level, float division_factor,
       int timesteps, const float *LCR_, int time_direction_ = 1,
       const float *intermediate_controls_ = nullptr)
      : intermediate_poses(nullptr), intermediate_controls(nullptr), g(g_in),
        f(0), parent(parent_in), active(true), rank(0), level(0),
        time_direction(time_direction_) {
    for (int i = 0; i < n_dims; i++)
      pose[i] = pose_in[i];
    if (parent_in != nullptr && intermediate_poses_ != nullptr) {
      intermediate_poses = new float[timesteps * n_dims];
      memcpy(intermediate_poses, intermediate_poses_,
             timesteps * n_dims * sizeof(float));
    }
    if (parent_in != nullptr && intermediate_controls_ != nullptr) {
      intermediate_controls = new float[timesteps * n_cont];
      memcpy(intermediate_controls, intermediate_controls_,
             timesteps * n_cont * sizeof(float));
      memcpy(control, intermediate_controls_, n_cont * sizeof(float));
    }
    hash = calc_hash(pose, tolerance);
    LCR_index = calc_hash(pose, LCR_);
    float resolution[n_dims];
    for (int i = 0; i < n_dims; i++)
      resolution[i] = resolution_[i];
    for (int i = 0; i < max_level; i++) {
      index.push_back(calc_hash(pose, resolution));
      for (int j = 0; j < n_dims; j++)
        resolution[j] = resolution[j] / division_factor;
    }
  }

  ~Node() {
    delete[] intermediate_poses;
    delete[] intermediate_controls;
  }
};

struct NodePtrCompare {
  bool operator()(const std::shared_ptr<Node> &a,
                  const std::shared_ptr<Node> &b) const {
    return a->f > b->f;
  }
};

class Environment {
  int n_succ;
  int n_substeps;
  int n_directions;
  float epsilon[n_dims];
  float workspace_lo, workspace_hi;
  std::vector<float> action_radii;
  float edge_cost;
  float max_agent_vel;
  float max_penetration_px;
  float qs_support_radius;

  // Per-Succ device buffers (n_succ rollouts).
  float *d_states = nullptr;
  float *d_intermediates = nullptr;
  float *d_actions = nullptr;
  bool *d_valid = nullptr;
  // Batched buffers (capacity * n_succ).
  float *d_states_b = nullptr;
  float *d_intermediates_b = nullptr;
  float *d_actions_b = nullptr;
  bool *d_valid_b = nullptr;
  int batch_capacity = 0;
  cudaStream_t stream = nullptr;
  int threads = 256;

public:
  int max_level;
  int timesteps;
  int time_direction;
  float resolution[n_dims];
  float tolerance[n_dims];
  float local_controllability_radius[n_dims];
  float division_factor;

  Environment(const py::dict &config, int time_direction_ = 1)
      : time_direction(time_direction_) {
    auto info = config["experiment_info_default"].cast<py::dict>();
    auto node_info = info["node_info"].cast<py::dict>();

    max_level = info["max_level"].cast<int>();
    division_factor = info["division_factor"].cast<float>();
    n_substeps = node_info.contains("n_substeps")
                     ? node_info["n_substeps"].cast<int>()
                     : pusht_qs::kDefaultSubsteps;
    timesteps = n_substeps;
    workspace_lo = node_info.contains("workspace_lo")
                       ? node_info["workspace_lo"].cast<float>()
                       : 0.0f;
    workspace_hi = node_info.contains("workspace_hi")
                       ? node_info["workspace_hi"].cast<float>()
                       : 512.0f;
    edge_cost = static_cast<float>(n_substeps) * pusht_qs::kDt;

    set_resolutions(info);
    set_controls(node_info);

    if (cudaStreamCreate(&stream) != cudaSuccess)
      throw std::runtime_error("pusht_qs_cuda: cudaStreamCreate failed");
    alloc_instance_buffers();
  }

  ~Environment() {
    free_instance_buffers();
    free_batch_buffers();
    if (stream)
      cudaStreamDestroy(stream);
  }

  static std::vector<float> as_fvec(const py::handle &h) {
    std::vector<float> out;
    for (auto item : h.cast<py::sequence>())
      out.push_back(py::float_(item).cast<float>());
    return out;
  }

  void set_resolutions(py::dict &info) {
    auto res = as_fvec(info["resolution"]);
    auto tol = as_fvec(info["tolerance"]);
    std::vector<float> eps;
    if (time_direction == -1 && info.contains("backward_epsilon"))
      eps = as_fvec(info["backward_epsilon"]);
    else
      eps = as_fvec(info["epsilon"]);
    for (int i = 0; i < n_dims; i++) {
      resolution[i] = (i < (int)res.size()) ? res[i] : res.back();
      tolerance[i] = (i < (int)tol.size()) ? tol[i] : tol.back();
      epsilon[i] = (i < (int)eps.size()) ? eps[i] : eps.back();
    }
    if (info.contains("LCR")) {
      auto lcr = as_fvec(info["LCR"]);
      for (int i = 0; i < n_dims; i++)
        local_controllability_radius[i] =
            (i < (int)lcr.size()) ? lcr[i] : epsilon[i];
    } else {
      for (int i = 0; i < n_dims; i++)
        local_controllability_radius[i] = epsilon[i];
    }
  }

  void set_controls(py::dict &node_info) {
    n_directions = node_info.contains("n_directions")
                       ? node_info["n_directions"].cast<int>()
                       : 16;
    action_radii = node_info.contains("action_radii")
                       ? as_fvec(node_info["action_radii"])
                       : std::vector<float>{200.f};
    max_agent_vel = node_info.contains("max_agent_vel")
                        ? node_info["max_agent_vel"].cast<float>()
                        : 500.0f;
    max_penetration_px = node_info.contains("max_penetration_px")
                             ? node_info["max_penetration_px"].cast<float>()
                             : 1.0f;
    qs_support_radius = node_info.contains("qs_support_radius")
                            ? node_info["qs_support_radius"].cast<float>()
                            : pusht_qs::kDefaultSupportRadius;
    n_succ = n_directions * int(action_radii.size()) + 1;
  }

  void alloc_instance_buffers() {
    free_instance_buffers();
    const size_t R = size_t(n_succ);
    cudaMalloc(&d_states, R * n_dims * sizeof(float));
    cudaMalloc(&d_intermediates, R * size_t(timesteps) * n_dims * sizeof(float));
    cudaMalloc(&d_actions, R * n_cont * sizeof(float));
    cudaMalloc(&d_valid, R * sizeof(bool));
  }

  void free_instance_buffers() {
    if (d_states)
      cudaFree(d_states);
    if (d_intermediates)
      cudaFree(d_intermediates);
    if (d_actions)
      cudaFree(d_actions);
    if (d_valid)
      cudaFree(d_valid);
    d_states = d_intermediates = d_actions = nullptr;
    d_valid = nullptr;
  }

  void free_batch_buffers() {
    if (d_states_b)
      cudaFree(d_states_b);
    if (d_intermediates_b)
      cudaFree(d_intermediates_b);
    if (d_actions_b)
      cudaFree(d_actions_b);
    if (d_valid_b)
      cudaFree(d_valid_b);
    d_states_b = d_intermediates_b = d_actions_b = nullptr;
    d_valid_b = nullptr;
    batch_capacity = 0;
  }

  void ensure_batch_capacity(int max_batch) {
    if (max_batch <= batch_capacity)
      return;
    free_batch_buffers();
    batch_capacity = max_batch;
    const size_t R = size_t(batch_capacity) * size_t(n_succ);
    cudaMalloc(&d_states_b, R * n_dims * sizeof(float));
    cudaMalloc(&d_intermediates_b,
               R * size_t(timesteps) * n_dims * sizeof(float));
    cudaMalloc(&d_actions_b, R * n_cont * sizeof(float));
    cudaMalloc(&d_valid_b, R * sizeof(bool));
  }

  void set_world(torch::Tensor /*world*/) {}

  std::shared_ptr<Node> create_Node(float *pose) {
    return std::make_shared<Node>(pose, nullptr, nullptr, 0, resolution,
                                  tolerance, max_level, division_factor,
                                  timesteps, local_controllability_radius,
                                  time_direction);
  }

  float block_distance(float *pose, float *goal) {
    float dx = pose[2] - goal[2];
    float dy = pose[3] - goal[3];
    return std::sqrt(dx * dx + dy * dy);
  }

  float distance(float *pose, float *goal) { return block_distance(pose, goal); }

  float heuristic(float *pose, float *goal) {
    return max_agent_vel > 0.f ? block_distance(pose, goal) / max_agent_vel
                               : 0.f;
  }

  float compute_near_meet_distance(float *pose1, float *pose2) {
    return heuristic(pose1, pose2);
  }

  bool reached_goal_region(std::shared_ptr<Node> v,
                           std::shared_ptr<Node> goal) {
    float dx = v->pose[2] - goal->pose[2];
    float dy = v->pose[3] - goal->pose[3];
    float dth = v->pose[4] - goal->pose[4];
    while (dth > float(M_PI))
      dth -= 2.f * float(M_PI);
    while (dth < -float(M_PI))
      dth += 2.f * float(M_PI);
    return std::fabs(dx) < epsilon[2] && std::fabs(dy) < epsilon[3] &&
           std::fabs(dth) < epsilon[4];
  }

  bool in_workspace(float *pose) {
    for (int i = 0; i < 4; i++) {
      if (pose[i] < workspace_lo || pose[i] > workspace_hi)
        return false;
    }
    return true;
  }

  bool check_validity_single(float *pose) { return in_workspace(pose); }

  void check_validity(float *start, float *goal, bool *result) {
    result[0] = check_validity_single(start);
    result[1] = check_validity_single(goal);
  }

  void check_validity_batched(const std::vector<float *> &states,
                              std::vector<bool> &results) {
    results.resize(states.size());
    for (size_t i = 0; i < states.size(); i++)
      results[i] = check_validity_single(states[i]);
  }

  void fill_actions(const float *pose, std::vector<float> &actions) const {
    actions.clear();
    actions.reserve(n_succ * n_cont);
    for (float r : action_radii) {
      pusht_qs::append_qs_ring_targets(pose, r, n_directions, workspace_lo,
                                       workspace_hi, pusht_qs::kContactFrictionMu,
                                       actions);
    }
    actions.push_back(pose[2]);
    actions.push_back(pose[3]);
  }

  std::vector<std::shared_ptr<Node>>
  neighbors_from_rollouts(std::shared_ptr<Node> node,
                          std::shared_ptr<Node> goal,
                          const std::vector<float> &actions,
                          std::vector<float> &next_states,
                          std::vector<float> &intermediates,
                          std::vector<char> &valid) {
    std::vector<std::shared_ptr<Node>> neighbors;
    neighbors.reserve(n_succ);
    for (int i = 0; i < n_succ; i++) {
      if (!valid[i])
        continue;
      float *ns = &next_states[i * n_dims];
      float *inter = &intermediates[i * timesteps * n_dims];
      float ctrl[n_cont] = {actions[i * n_cont], actions[i * n_cont + 1]};
      float *controls_seq = new float[timesteps * n_cont];
      for (int t = 0; t < timesteps; t++)
        memcpy(&controls_seq[t * n_cont], ctrl, n_cont * sizeof(float));
      auto neighbor = std::make_shared<Node>(
          ns, inter, node, node->g + edge_cost, resolution, tolerance,
          max_level, division_factor, timesteps, local_controllability_radius,
          time_direction, controls_seq);
      delete[] controls_seq;
      neighbor->f = neighbor->g + heuristic(neighbor->pose, goal->pose);
      neighbors.push_back(neighbor);
    }
    return neighbors;
  }

  std::vector<std::shared_ptr<Node>> Succ(std::shared_ptr<Node> node,
                                          std::shared_ptr<Node> goal) {
    std::vector<float> actions;
    fill_actions(node->pose, actions);
    std::vector<float> next_states(n_succ * n_dims);
    std::vector<float> intermediates(n_succ * timesteps * n_dims);
    std::vector<char> valid(n_succ, 0);

    for (int i = 0; i < n_succ; i++)
      memcpy(&next_states[i * n_dims], node->pose, n_dims * sizeof(float));

    const int blocks = (n_succ + threads - 1) / threads;
    // Host bool buffer (vector<bool> is not contiguous).
    std::unique_ptr<bool[]> valid_b(new bool[n_succ]);
    pusht_qs_launcher(next_states.data(), intermediates.data(), actions.data(),
                      valid_b.get(), n_substeps, n_succ, max_agent_vel,
                      qs_support_radius, workspace_lo, workspace_hi,
                      max_penetration_px, d_states, d_intermediates, d_actions,
                      d_valid, blocks, threads, stream);
    for (int i = 0; i < n_succ; i++)
      valid[i] = valid_b[i] ? 1 : 0;
    return neighbors_from_rollouts(node, goal, actions, next_states,
                                   intermediates, valid);
  }

  std::vector<std::vector<std::shared_ptr<Node>>>
  Succ_batched(const std::vector<std::shared_ptr<Node>> &nodes,
               std::shared_ptr<Node> goal) {
    const int M = int(nodes.size());
    std::vector<std::vector<std::shared_ptr<Node>>> results(M);
    if (M == 0)
      return results;
    if (M == 1) {
      results[0] = Succ(nodes[0], goal);
      return results;
    }

    ensure_batch_capacity(M);
    const int total = M * n_succ;
    std::vector<float> states(total * n_dims);
    std::vector<float> actions(total * n_cont);
    std::vector<float> intermediates(total * timesteps * n_dims);
    std::unique_ptr<bool[]> valid(new bool[total]);

    for (int i = 0; i < M; i++) {
      std::vector<float> acts;
      fill_actions(nodes[i]->pose, acts);
      for (int s = 0; s < n_succ; s++) {
        const int row = i * n_succ + s;
        memcpy(&states[row * n_dims], nodes[i]->pose, n_dims * sizeof(float));
        actions[row * n_cont + 0] = acts[s * n_cont + 0];
        actions[row * n_cont + 1] = acts[s * n_cont + 1];
      }
    }

    const int blocks = (total + threads - 1) / threads;
    pusht_qs_launcher(states.data(), intermediates.data(), actions.data(),
                      valid.get(), n_substeps, total, max_agent_vel,
                      qs_support_radius, workspace_lo, workspace_hi,
                      max_penetration_px, d_states_b, d_intermediates_b,
                      d_actions_b, d_valid_b, blocks, threads, stream);

    for (int i = 0; i < M; i++) {
      std::vector<float> acts;
      fill_actions(nodes[i]->pose, acts);
      std::vector<float> next_states(n_succ * n_dims);
      std::vector<float> inter(n_succ * timesteps * n_dims);
      std::vector<char> valid_c(n_succ, 0);
      for (int s = 0; s < n_succ; s++) {
        const int row = i * n_succ + s;
        memcpy(&next_states[s * n_dims], &states[row * n_dims],
               n_dims * sizeof(float));
        memcpy(&inter[s * timesteps * n_dims],
               &intermediates[row * timesteps * n_dims],
               timesteps * n_dims * sizeof(float));
        valid_c[s] = valid[row] ? 1 : 0;
      }
      results[i] = neighbors_from_rollouts(nodes[i], goal, acts, next_states,
                                           inter, valid_c);
    }
    return results;
  }

  torch::Tensor convert_node_list_to_path_tensor(
      std::vector<std::shared_ptr<Node>> node_list) {
    if (node_list.empty()) {
      return torch::zeros({0, n_dims + 1},
                          torch::TensorOptions().dtype(torch::kFloat32));
    }
    struct PathState {
      float pose[n_dims];
      float g;
    };
    std::vector<PathState> extended;
    for (size_t i = 0; i < node_list.size(); i++) {
      if (node_list[i]->parent == nullptr ||
          node_list[i]->intermediate_poses == nullptr)
        continue;
      const float g0 = node_list[i]->parent->g;
      const float g1 = node_list[i]->g;
      const float td = static_cast<float>(node_list[i]->time_direction);
      for (int j = 0; j < timesteps; j++) {
        const int k = timesteps - j - 1;
        const int src = k * n_dims;
        PathState state;
        for (int d = 0; d < n_dims; d++)
          state.pose[d] = node_list[i]->intermediate_poses[src + d];
        const float alpha =
            static_cast<float>(k + 1) / static_cast<float>(timesteps);
        state.g = (g0 + alpha * (g1 - g0)) * td;
        extended.push_back(state);
      }
    }
    auto path_tensor = torch::zeros(
        {(int)extended.size(), n_dims + 1},
        torch::TensorOptions().dtype(torch::kFloat32));
    for (size_t i = 0; i < extended.size(); i++) {
      for (int d = 0; d < n_dims; d++)
        path_tensor[i][d] = extended[i].pose[d];
      path_tensor[i][n_dims] = extended[i].g;
    }
    return path_tensor;
  }

  torch::Tensor convert_node_list_to_controls_tensor(
      std::vector<std::shared_ptr<Node>> node_list) {
    // Align with dense path: skip root, one control per substep, same emit order.
    if (node_list.empty()) {
      return torch::zeros({0, n_cont},
                          torch::TensorOptions().dtype(torch::kFloat32));
    }
    int n = 0;
    for (size_t i = 0; i < node_list.size(); i++) {
      if (node_list[i]->parent != nullptr)
        n += timesteps;
    }
    auto controls_tensor = torch::zeros(
        {n, n_cont}, torch::TensorOptions().dtype(torch::kFloat32));
    int r = 0;
    for (size_t i = 0; i < node_list.size(); i++) {
      if (node_list[i]->parent == nullptr)
        continue;
      for (int j = 0; j < timesteps; j++) {
        const int k = timesteps - j - 1;
        for (int d = 0; d < n_cont; d++) {
          if (node_list[i]->intermediate_controls != nullptr) {
            controls_tensor[r][d] =
                node_list[i]->intermediate_controls[k * n_cont + d];
          } else {
            controls_tensor[r][d] = node_list[i]->control[d];
          }
        }
        r++;
      }
    }
    return controls_tensor;
  }
};
