// PushT environment for IGHAStar (Chipmunk CPU, gym-pusht PD + optional v_max).
// State [10]: ax, ay, bx, by, angle, avx, avy, bvx, bvy, bw
// Control [2]: agent PD target xy in workspace
#pragma once

#include <algorithm>
#include <boost/functional/hash.hpp>
#include <cmath>
#include <cstring>
#include <string>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>

#include <pusht_physics.h>
#include <pusht_qs.h>

using namespace std;

constexpr int n_dims = pusht_physics::kStateDim;
constexpr int hash_dims = 5; // hash pose only (not velocities)
constexpr int n_cont = 2;

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
    for (int i = 0; i < n_dims; i++) {
      pose[i] = pose_in[i];
    }
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
  std::vector<float> action_radii; // agent-relative PD offsets
  float edge_cost;
  float max_agent_vel; // px/s; clamps PD; heuristic = block_dist / this
  float max_penetration_px; // invalidate Succ if pen deepens by more than this
  bool use_quasistatic;     // Mason/Lynch QS (pusht_qs) vs Chipmunk
  float qs_support_radius;  // ellipsoidal LS ℓ = (2/3) R

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
                     : pusht_physics::kDefaultSubsteps;
    // Path stores one pose per physics substep (matches gym control hold).
    timesteps = n_substeps;
    workspace_lo =
        node_info.contains("workspace_lo") ? node_info["workspace_lo"].cast<float>()
                                           : 0.0f;
    workspace_hi =
        node_info.contains("workspace_hi") ? node_info["workspace_hi"].cast<float>()
                                           : 512.0f;
    // g increments by simulated edge duration (n_substeps * dt seconds).
    edge_cost = static_cast<float>(n_substeps) *
                static_cast<float>(pusht_physics::kDt);

    set_resolutions(info);
    set_controls(node_info);
  }

  ~Environment() = default;

  static std::vector<float> as_fvec(const py::handle &h) {
    std::vector<float> out;
    for (auto item : h.cast<py::sequence>()) {
      out.push_back(py::float_(item).cast<float>());
    }
    return out;
  }

  void set_resolutions(py::dict &info) {
    auto res = as_fvec(info["resolution"]);
    auto tol = as_fvec(info["tolerance"]);
    std::vector<float> eps;
    if (time_direction == -1 && info.contains("backward_epsilon")) {
      eps = as_fvec(info["backward_epsilon"]);
    } else {
      eps = as_fvec(info["epsilon"]);
    }
    for (int i = 0; i < n_dims; i++) {
      resolution[i] = (i < (int)res.size()) ? res[i] : res.back();
      tolerance[i] = (i < (int)tol.size()) ? tol[i] : tol.back();
      epsilon[i] = (i < (int)eps.size()) ? eps[i] : eps.back();
    }
    if (info.contains("LCR")) {
      auto lcr = as_fvec(info["LCR"]);
      for (int i = 0; i < n_dims; i++) {
        local_controllability_radius[i] =
            (i < (int)lcr.size()) ? lcr[i] : epsilon[i];
      }
    } else {
      for (int i = 0; i < n_dims; i++) {
        local_controllability_radius[i] = epsilon[i];
      }
    }
  }

  // Agent-relative rings (+ aim-at-block added in Succ).
  void set_controls(py::dict &node_info) {
    n_directions = node_info.contains("n_directions")
                       ? node_info["n_directions"].cast<int>()
                       : 8;
    action_radii = node_info.contains("action_radii")
                       ? as_fvec(node_info["action_radii"])
                       : std::vector<float>{40.f, 80.f, 120.f};
    max_agent_vel = node_info.contains("max_agent_vel")
                        ? node_info["max_agent_vel"].cast<float>()
                        : 100.0f;
    max_penetration_px = node_info.contains("max_penetration_px")
                             ? node_info["max_penetration_px"].cast<float>()
                             : 1.0f;
    use_quasistatic = false;
    if (node_info.contains("dynamics")) {
      const std::string dyn = node_info["dynamics"].cast<std::string>();
      use_quasistatic = (dyn == "quasistatic" || dyn == "qs" || dyn == "mason");
    } else if (node_info.contains("use_quasistatic")) {
      use_quasistatic = node_info["use_quasistatic"].cast<bool>();
    }
    qs_support_radius = node_info.contains("qs_support_radius")
                            ? node_info["qs_support_radius"].cast<float>()
                            : pusht_qs::kDefaultSupportRadius;
    n_succ = n_directions * int(action_radii.size()) + 1;
  }

  // No map; accept any tensor (unused).
  void set_world(torch::Tensor /*world*/) {}

  std::shared_ptr<Node> create_Node(float *pose) {
    return std::make_shared<Node>(pose, nullptr, nullptr, 0, resolution,
                                  tolerance, max_level, division_factor,
                                  timesteps, local_controllability_radius,
                                  time_direction);
  }

  float compute_near_meet_distance(float *pose1, float *pose2) {
    return heuristic(pose1, pose2);
  }

  // Block xy (+ mild angle) distance — planning objective is seating the tee.
  float block_distance(float *pose, float *goal) {
    float dx = pose[2] - goal[2];
    float dy = pose[3] - goal[3];
    // float dth = pose[4] - goal[4];
    // while (dth > float(M_PI))
    //   dth -= 2.f * float(M_PI);
    // while (dth < -float(M_PI))
    //   dth += 2.f * float(M_PI);
    return std::sqrt(dx * dx + dy * dy);// + 40.0f * std::fabs(dth);
  }

  float distance(float *pose, float *goal) { return block_distance(pose, goal); }

  float heuristic(float *pose, float *goal) {
    // Time lower bound assuming agent never exceeds max_agent_vel (clamped in PD).
    return block_distance(pose, goal) / max_agent_vel;
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
    for (size_t i = 0; i < states.size(); i++) {
      results[i] = check_validity_single(states[i]);
    }
  }

  std::vector<std::shared_ptr<Node>> Succ(std::shared_ptr<Node> node,
                                          std::shared_ptr<Node> goal) {
    std::vector<std::shared_ptr<Node>> neighbors;
    neighbors.reserve(n_succ);

    // Agent-relative rings at action_radii + aim at block center.
    std::vector<float> actions;
    actions.reserve(n_succ * n_cont);
    for (float r : action_radii) {
      for (int i = 0; i < n_directions; i++) {
        float th = i * (2.0f * float(M_PI) / n_directions);
        float tx = node->pose[0] + r * std::cos(th);
        float ty = node->pose[1] + r * std::sin(th);
        tx = std::min(std::max(tx, workspace_lo), workspace_hi);
        ty = std::min(std::max(ty, workspace_lo), workspace_hi);
        actions.push_back(tx);
        actions.push_back(ty);
      }
    }
    actions.push_back(node->pose[2]);
    actions.push_back(node->pose[3]);

    const int n_actions = int(actions.size()) / n_cont;
    std::vector<float> next_states(n_actions * n_dims);
    std::vector<float> intermediates(n_actions * timesteps * n_dims);
    std::vector<char> valid(n_actions, 0);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_actions; i++) {
      float out[n_dims];
      float pen_inc = 0.f;
      float *inter = &intermediates[i * timesteps * n_dims];
      if (use_quasistatic) {
        pusht_qs::step_from_snapshot(
            node->pose, actions[i * n_cont], actions[i * n_cont + 1], out,
            n_substeps, inter, max_agent_vel, &pen_inc, qs_support_radius,
            workspace_lo, workspace_hi);
      } else {
        pusht_physics::step_from_snapshot(
            node->pose, actions[i * n_cont], actions[i * n_cont + 1], out,
            n_substeps, inter, max_agent_vel, &pen_inc);
      }
      memcpy(&next_states[i * n_dims], out, n_dims * sizeof(float));
      // Reject edges that deepen contact penetration by > max_penetration_px.
      valid[i] = (in_workspace(out) && pen_inc <= max_penetration_px) ? 1 : 0;
    }

    for (int i = 0; i < n_actions; i++) {
      if (!valid[i])
        continue;
      float *ns = &next_states[i * n_dims];
      float *inter = &intermediates[i * timesteps * n_dims];
      float ctrl[n_cont] = {actions[i * n_cont], actions[i * n_cont + 1]};
      float *controls_seq = new float[timesteps * n_cont];
      for (int t = 0; t < timesteps; t++) {
        memcpy(&controls_seq[t * n_cont], ctrl, n_cont * sizeof(float));
      }
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

  void ensure_batch_capacity(int /*max_batch*/) {}

  std::vector<std::vector<std::shared_ptr<Node>>>
  Succ_batched(const std::vector<std::shared_ptr<Node>> &nodes,
               std::shared_ptr<Node> goal) {
    std::vector<std::vector<std::shared_ptr<Node>>> results;
    results.reserve(nodes.size());
    for (const auto &node : nodes) {
      results.push_back(Succ(node, goal));
    }
    return results;
  }

  torch::Tensor convert_node_list_to_path_tensor(
      std::vector<std::shared_ptr<Node>> node_list) {
    // Dense path: every physics substep (same layout as kinodynamic_cpu).
    // best_path is goal-first; skip the root (no intermediates).
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
          node_list[i]->intermediate_poses == nullptr) {
        continue;
      }
      for (int j = 0; j < timesteps; j++) {
        // Stored chronological; emit reverse so goal-first path stays consistent
        int src = (timesteps - j - 1) * n_dims;
        PathState state;
        for (int d = 0; d < n_dims; d++) {
          state.pose[d] = node_list[i]->intermediate_poses[src + d];
        }
        state.g = node_list[i]->g * node_list[i]->time_direction;
        extended.push_back(state);
      }
    }
    auto path_tensor = torch::zeros(
        {(int)extended.size(), n_dims + 1},
        torch::TensorOptions().dtype(torch::kFloat32));
    for (size_t i = 0; i < extended.size(); i++) {
      for (int d = 0; d < n_dims; d++) {
        path_tensor[i][d] = extended[i].pose[d];
      }
      path_tensor[i][n_dims] = extended[i].g;
    }
    return path_tensor;
  }

  torch::Tensor convert_node_list_to_controls_tensor(
      std::vector<std::shared_ptr<Node>> node_list) {
    if (node_list.empty()) {
      return torch::zeros({0, n_cont},
                          torch::TensorOptions().dtype(torch::kFloat32));
    }
    int path_length = int(node_list.size());
    auto controls_tensor = torch::zeros(
        {path_length, n_cont}, torch::TensorOptions().dtype(torch::kFloat32));
    for (int i = 0; i < path_length; i++) {
      for (int d = 0; d < n_cont; d++) {
        controls_tensor[i][d] = node_list[i]->control[d];
      }
    }
    return controls_tensor;
  }
};
