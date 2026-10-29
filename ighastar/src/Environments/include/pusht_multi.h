// Two-tee Push-T for IGHAStar. Chipmunk only. Open right wall, one static square.
// State [16]: ax ay t0x t0y t0a t1x t1y t1a avx avy t0vx t0vy t0w t1vx t1vy t1w
// Clear bits live on Node, not in the state and not in the hash.
#pragma once

#include <algorithm>
#include <boost/functional/hash.hpp>
#include <cmath>
#include <cstring>
#include <string>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>

#include <pusht_multi_physics.h>
#include <pusht_qs.h>

using namespace std;

constexpr int n_dims = pusht_multi_physics::kStateDim;
constexpr int hash_dims = 8; // agent xy + both tee poses
constexpr int n_cont = 2;
constexpr int kNumTees = pusht_multi_physics::kNumTees;

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
  int discover_index = -1;
  int expand_index = -1;
  float control[n_cont] = {};
  float *intermediate_controls = nullptr;
  // Derived from the path. Not hashed. Sticky once a tee CoG enters the exit box.
  bool cleared[kNumTees] = {};

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
  float workspace_lo, workspace_hi, agent_x_hi;
  float exit_x0, exit_x1, exit_y0, exit_y1;
  float obstacle_x, obstacle_y, obstacle_side;
  // Row-major [B, 2] when the last set_world passed one center per state.
  std::vector<float> world_rows;
  std::vector<float> action_radii;
  float edge_cost;
  float max_agent_vel;
  float max_penetration_px;

public:
  int max_level;
  int timesteps;
  int time_direction;
  float resolution[n_dims];
  float tolerance[n_dims];
  float local_controllability_radius[n_dims];
  float division_factor;

  Environment(const py::dict &config, int time_direction_ = 1)
      : obstacle_x(256.f), obstacle_y(256.f), time_direction(time_direction_) {
    auto info = config["experiment_info_default"].cast<py::dict>();
    auto node_info = info["node_info"].cast<py::dict>();

    max_level = info["max_level"].cast<int>();
    division_factor = info["division_factor"].cast<float>();
    n_substeps = node_info.contains("n_substeps")
                     ? node_info["n_substeps"].cast<int>()
                     : pusht_multi_physics::kDefaultSubsteps;
    timesteps = n_substeps;
    workspace_lo = node_info.contains("workspace_lo")
                       ? node_info["workspace_lo"].cast<float>()
                       : 0.f;
    workspace_hi = node_info.contains("workspace_hi")
                       ? node_info["workspace_hi"].cast<float>()
                       : 512.f;
    agent_x_hi =
        node_info.contains("agent_x_hi") ? node_info["agent_x_hi"].cast<float>()
                                        : 700.f;
    exit_x0 = node_info.contains("exit_x0") ? node_info["exit_x0"].cast<float>()
                                           : 512.f;
    exit_x1 = node_info.contains("exit_x1") ? node_info["exit_x1"].cast<float>()
                                           : 700.f;
    exit_y0 = node_info.contains("exit_y0") ? node_info["exit_y0"].cast<float>()
                                           : 0.f;
    exit_y1 = node_info.contains("exit_y1") ? node_info["exit_y1"].cast<float>()
                                           : 512.f;
    obstacle_side = node_info.contains("obstacle_side")
                        ? node_info["obstacle_side"].cast<float>()
                        : static_cast<float>(pusht_multi_physics::kTeeScale);
    edge_cost = static_cast<float>(n_substeps) *
                static_cast<float>(pusht_multi_physics::kDt);
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
    std::vector<float> eps = as_fvec(info["epsilon"]);
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

  void set_controls(py::dict &node_info) {
    n_directions = node_info.contains("n_directions")
                       ? node_info["n_directions"].cast<int>()
                       : 8;
    action_radii = node_info.contains("action_radii")
                       ? as_fvec(node_info["action_radii"])
                       : std::vector<float>{200.f};
    max_agent_vel = node_info.contains("max_agent_vel")
                        ? node_info["max_agent_vel"].cast<float>()
                        : 500.f;
    max_penetration_px = node_info.contains("max_penetration_px")
                             ? node_info["max_penetration_px"].cast<float>()
                             : 1.f;
    n_succ = n_directions * int(action_radii.size()) + 1;
  }

  // World tensor: one center [2], or one center per state [B, 2].
  void set_world(torch::Tensor world) {
    world_rows.clear();
    if (!world.defined() || world.numel() < 2)
      return;
    auto w = world.to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (w.dim() == 2 && w.size(1) == 2) {
      world_rows.assign(w.data_ptr<float>(),
                        w.data_ptr<float>() + w.numel());
      obstacle_x = world_rows[0];
      obstacle_y = world_rows[1];
      return;
    }
    obstacle_x = w.data_ptr<float>()[0];
    obstacle_y = w.data_ptr<float>()[1];
  }

  void tee_cog(const float *pose, int tee, float &cx, float &cy) const {
    const int b = 2 + 3 * tee;
    pusht_qs::Vec2 cog_l;
    pusht_qs::tee_cog_local(cog_l);
    const float c = std::cos(pose[b + 2]);
    const float s = std::sin(pose[b + 2]);
    pusht_qs::Vec2 w = pusht_qs::rot(cog_l, c, s);
    cx = pose[b] + w.x;
    cy = pose[b + 1] + w.y;
  }

  bool cog_in_exit(const float *pose, int tee) const {
    float cx, cy;
    tee_cog(pose, tee, cx, cy);
    return cx >= exit_x0 && cx <= exit_x1 && cy >= exit_y0 && cy <= exit_y1;
  }

  void stamp_cleared(Node &node, const Node *parent,
                     const float *intermediates) const {
    for (int i = 0; i < kNumTees; i++) {
      node.cleared[i] = parent ? parent->cleared[i] : false;
      if (cog_in_exit(node.pose, i))
        node.cleared[i] = true;
    }
    if (!intermediates)
      return;
    for (int t = 0; t < timesteps; t++) {
      const float *s = intermediates + t * n_dims;
      for (int i = 0; i < kNumTees; i++) {
        if (cog_in_exit(s, i))
          node.cleared[i] = true;
      }
    }
  }

  float h_cleared(const float *pose, const bool *cleared) const {
    const float gx = 0.5f * (exit_x0 + exit_x1);
    const float gy = 0.5f * (exit_y0 + exit_y1);
    float dist = 0.f;
    for (int i = 0; i < kNumTees; i++) {
      if (cleared[i])
        continue;
      float cx, cy;
      tee_cog(pose, i, cx, cy);
      const float dx = cx - gx;
      const float dy = cy - gy;
      dist += std::sqrt(dx * dx + dy * dy);
    }
    return max_agent_vel > 0.f ? dist / max_agent_vel : 0.f;
  }

  std::shared_ptr<Node> create_Node(float *pose) {
    auto node = std::make_shared<Node>(pose, nullptr, nullptr, 0, resolution,
                                       tolerance, max_level, division_factor,
                                       timesteps, local_controllability_radius,
                                       time_direction);
    stamp_cleared(*node, nullptr, nullptr);
    return node;
  }

  float compute_near_meet_distance(float *pose1, float *pose2) {
    return heuristic(pose1, pose2);
  }

  float distance(float *pose, float *goal) { return heuristic(pose, goal); }

  // Pose-only. A tee counts as cleared only if its CoG is in the exit box now.
  // Successors set f from Node::cleared, which also remembers a past entry.
  float heuristic(float *pose, float * /*goal*/) {
    bool now[kNumTees];
    for (int i = 0; i < kNumTees; i++)
      now[i] = cog_in_exit(pose, i);
    return h_cleared(pose, now);
  }

  bool reached_goal_region(std::shared_ptr<Node> v, std::shared_ptr<Node>) {
    return v->cleared[0] && v->cleared[1];
  }

  bool finite_pose(const float *pose) const {
    for (int i = 0; i < hash_dims; i++) {
      if (!std::isfinite(pose[i]))
        return false;
    }
    return true;
  }

  bool in_agent_workspace(const float *pose) const {
    return pose[0] >= workspace_lo && pose[0] <= agent_x_hi &&
           pose[1] >= workspace_lo && pose[1] <= workspace_hi;
  }

  bool valid_at(float *pose, float ox, float oy) {
    if (!finite_pose(pose) || !in_agent_workspace(pose))
      return false;
    for (int i = 0; i < kNumTees; i++) {
      if (cog_in_exit(pose, i))
        return false;
    }
    return !pusht_multi_physics::overlaps_forbidden(pose, ox, oy, obstacle_side);
  }

  bool check_validity_single(float *pose) {
    return valid_at(pose, obstacle_x, obstacle_y);
  }

  void check_validity(float *start, float *goal, bool *result) {
    result[0] = check_validity_single(start);
    result[1] = check_validity_single(goal);
  }

  void check_validity_batched(const std::vector<float *> &states,
                              std::vector<bool> &results) {
    const int n = static_cast<int>(states.size());
    results.resize(n);
    const bool per_row = static_cast<int>(world_rows.size()) == 2 * n;
    std::vector<char> ok(n, 0);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
      const float ox = per_row ? world_rows[2 * i] : obstacle_x;
      const float oy = per_row ? world_rows[2 * i + 1] : obstacle_y;
      ok[i] = valid_at(states[i], ox, oy) ? 1 : 0;
    }
    for (int i = 0; i < n; i++)
      results[i] = ok[i] != 0;
  }

  // Uncleared tee the agent is touching, else the nearest uncleared tee.
  int select_tee(const Node &node) const {
    const float *p = node.pose;
    int best = -1;
    float best_score = -1e30f;
    for (int i = 0; i < kNumTees; i++) {
      if (node.cleared[i])
        continue;
      float pen = 0.f;
      pusht_qs::Vec2 nrm, contact, cog;
      const bool hit = pusht_qs::deepest_contact(
          {p[0], p[1]}, {p[2 + 3 * i], p[3 + 3 * i]}, p[4 + 3 * i], pen, nrm,
          contact, cog);
      const float dx = p[0] - p[2 + 3 * i];
      const float dy = p[1] - p[3 + 3 * i];
      const float score = hit ? pen : -std::sqrt(dx * dx + dy * dy);
      if (score > best_score) {
        best_score = score;
        best = i;
      }
    }
    return best < 0 ? 0 : best;
  }

  std::vector<std::shared_ptr<Node>> Succ(std::shared_ptr<Node> node,
                                          std::shared_ptr<Node> goal) {
    std::vector<std::shared_ptr<Node>> neighbors;
    neighbors.reserve(n_succ);

    const int tee = select_tee(*node);
    const int b = 2 + 3 * tee;
    float ring_pose[5] = {node->pose[0], node->pose[1], node->pose[b],
                          node->pose[b + 1], node->pose[b + 2]};
    std::vector<float> actions;
    actions.reserve(n_succ * n_cont);
    for (float r : action_radii) {
      pusht_qs::append_qs_ring_targets(
          ring_pose, r, n_directions, workspace_lo, workspace_hi,
          pusht_multi_physics::contact_friction_mu(), actions);
    }
    actions.push_back(node->pose[b]);
    actions.push_back(node->pose[b + 1]);

    const int n_actions = int(actions.size()) / n_cont;
    std::vector<float> next_states(n_actions * n_dims);
    std::vector<float> intermediates(n_actions * timesteps * n_dims);
    std::vector<char> valid(n_actions, 0);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_actions; i++) {
      float out[n_dims];
      float pen_inc = 0.f;
      int hit_obs = 0;
      float *inter = &intermediates[i * timesteps * n_dims];
      pusht_multi_physics::step_from_snapshot(
          node->pose, actions[i * n_cont], actions[i * n_cont + 1], out,
          n_substeps, inter, max_agent_vel, &pen_inc, &hit_obs, obstacle_x,
          obstacle_y, obstacle_side);
      memcpy(&next_states[i * n_dims], out, n_dims * sizeof(float));
      valid[i] = (finite_pose(out) && in_agent_workspace(out) &&
                  pen_inc <= max_penetration_px && hit_obs == 0)
                     ? 1
                     : 0;
    }

    for (int i = 0; i < n_actions; i++) {
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
      stamp_cleared(*neighbor, node.get(), inter);
      neighbor->f = neighbor->g + h_cleared(neighbor->pose, neighbor->cleared);
      (void)goal;
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
    for (const auto &node : nodes)
      results.push_back(Succ(node, goal));
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
          if (node_list[i]->intermediate_controls != nullptr)
            controls_tensor[r][d] =
                node_list[i]->intermediate_controls[k * n_cont + d];
          else
            controls_tensor[r][d] = node_list[i]->control[d];
        }
        r++;
      }
    }
    return controls_tensor;
  }
};
