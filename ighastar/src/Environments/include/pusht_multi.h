// Two-tee Push-T for IGHAStar. Chipmunk only. Open right wall, one static square.
// State [16]: ax ay t0x t0y t0a t1x t1y t1a avx avy t0vx t0vy t0w t1vx t1vy t1w
// Clear bits live on Node, not in the state and not in the hash.
#pragma once

#include <algorithm>
#include "solve_2pbvp.h"
#include <boost/functional/hash.hpp>
#include <cmath>
#include <cstring>
#include <string>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>

#include <pusht_multi_physics.h>
#include <pusht_qs.h>
#include "lazy_index.h"

using namespace std;

constexpr int n_dims = pusht_multi_physics::kStateDim;
constexpr int kNumAgents = pusht_multi_physics::kNumAgents;
constexpr int kNumTees = pusht_multi_physics::kNumTees;
// Poses only: every agent xy, then both tee poses. Velocities are not hashed.
constexpr int hash_dims = 2 * kNumAgents + 3 * kNumTees;
constexpr int n_cont = 2 * kNumAgents;

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
  LazyIndex<hash_dims, calc_hash> index;
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
       const float *tolerance, float division_factor,
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
    index.init(pose, resolution_, division_factor);
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
  float max_agent_force;
  // Active interior walls for the search. A batch validity call also keeps
  // the raw rows so each state can use its own maze.
  pusht_multi_physics::WallSpec walls_[pusht_multi_physics::kMaxWalls];
  int n_walls_ = 0;
  std::vector<float> world_rows;
  int world_stride = 0;
  bool world_is_batch = false;
  std::vector<float> action_radii;
  float edge_cost;
  float max_agent_vel;
  float max_penetration_px;
  bool rearrange;     // pose goal per tee. Clear uses the exit box.
  bool close_right;   // fourth wall. Rearrange closes it. Clear leaves it open.
  // Two-agent constraints. Agent 0 keeps y >= agent_y_split (top half, y is up)
  // and agent 1 keeps y <= agent_y_split. Centers stay min_agent_sep apart.
  // Both are off unless the config sets them.
  bool split_agents = false;
  float agent_y_split = 256.f;
  float min_agent_sep = 0.f;


public:
  int timesteps;
  int time_direction;
  float resolution[n_dims];
  float tolerance[n_dims];
  float local_controllability_radius[n_dims];
  NearMeet2PBVP near_meet_;
  sampling::GoalSamplingConfig goal_sampling;
  float division_factor;

  Environment(const py::dict &config, int time_direction_ = 1)
      : max_agent_force(1.0e9f), rearrange(false), close_right(false),
        time_direction(time_direction_) {
    auto info = config["experiment_info_default"].cast<py::dict>();
    auto node_info = info["node_info"].cast<py::dict>();

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
    if (node_info.contains("agent_y_split")) {
      split_agents = true;
      agent_y_split = node_info["agent_y_split"].cast<float>();
    }
    if (node_info.contains("min_agent_separation"))
      min_agent_sep = node_info["min_agent_separation"].cast<float>();
    if (node_info.contains("max_agent_force"))
      max_agent_force = node_info["max_agent_force"].cast<float>();
    if (node_info.contains("objective")) {
      rearrange = node_info["objective"].cast<std::string>() == "rearrange";
    }
    // Clear leaves the right wall open. Rearrange closes the box.
    close_right = rearrange;
    if (node_info.contains("open_right")) {
      close_right = !node_info["open_right"].cast<bool>();
    }
    edge_cost = static_cast<float>(n_substeps) *
                static_cast<float>(pusht_multi_physics::kDt);
    set_resolutions(info);
    set_controls(node_info);
    goal_sampling = sampling::load_goal_sampling(
        config, n_dims, local_controllability_radius);
    near_meet_.load(config, n_dims, local_controllability_radius,
                    goal_sampling.angular_dims);
  }

  virtual ~Environment() {
  }

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
    // One agent moves per successor. The others are commanded to hold.
    n_succ = kNumAgents * (n_directions * int(action_radii.size()) + 1);
  }

  // Row is [n, then n segments of x1, y1, x2, y2]. n above kMaxWalls throws.
  // A row of length 17 is the old gap encoding: orient, coord, gap_lo, gap_hi.
  // Each of those becomes the two solid pieces of a wall across the box.
  static int read_walls(const float *row, int stride,
                        pusht_multi_physics::WallSpec *out) {
    if (!row || stride < 5)
      return 0;
    int n = static_cast<int>(row[0]);
    if (n < 0)
      n = 0;
    if (stride == 1 + 4 * 4) {
      if (n > 4)
        n = 4;
      int w = 0;
      for (int i = 0; i < n; i++) {
        const float *p = row + 1 + 4 * i;
        float g0 = p[2];
        float g1 = p[3];
        if (g0 > g1)
          std::swap(g0, g1);
        const float a = pusht_multi_physics::kBoxLo;
        const float b = pusht_multi_physics::kBoxHi;
        if (p[0] < 0.5f) {
          if (w < pusht_multi_physics::kMaxWalls)
            out[w++] = {a, p[1], g0, p[1]};
          if (w < pusht_multi_physics::kMaxWalls)
            out[w++] = {g1, p[1], b, p[1]};
        } else {
          if (w < pusht_multi_physics::kMaxWalls)
            out[w++] = {p[1], a, p[1], g0};
          if (w < pusht_multi_physics::kMaxWalls)
            out[w++] = {p[1], g1, p[1], b};
        }
      }
      return w;
    }
    const int avail = (stride - 1) / 4;
    if (n > avail)
      n = avail;
    if (n > pusht_multi_physics::kMaxWalls)
      throw std::runtime_error("pusht_multi: " + std::to_string(n) + " wall segments, at most " +
                               std::to_string(pusht_multi_physics::kMaxWalls) + " are supported");
    for (int i = 0; i < n; i++) {
      const float *p = row + 1 + 4 * i;
      out[i] = {p[0], p[1], p[2], p[3]};
    }
    return n;
  }

  // One maze [n, wall...], or one maze per state [B, stride].
  void set_world(torch::Tensor world) {
    world_rows.clear();
    n_walls_ = 0;
    world_stride = 0;
    world_is_batch = false;
    if (!world.defined() || world.numel() < 1)
      return;
    auto w = world.to(torch::kCPU).to(torch::kFloat32).contiguous();
    world_rows.assign(w.data_ptr<float>(), w.data_ptr<float>() + w.numel());
    if (w.dim() == 2) {
      world_is_batch = true;
      world_stride = static_cast<int>(w.size(1));
    } else {
      world_stride = static_cast<int>(w.numel());
    }
    n_walls_ = read_walls(world_rows.data(), world_stride, walls_);
  }

  void tee_cog(const float *pose, int tee, float &cx, float &cy) const {
    const int b = pusht_multi_physics::tee_pose(tee);
    pusht_qs::Vec2 cog_l;
    pusht_qs::tee_cog_local(cog_l);
    const float c = std::cos(pose[b + 2]);
    const float s = std::sin(pose[b + 2]);
    pusht_qs::Vec2 w = pusht_qs::rot(cog_l, c, s);
    cx = pose[b] + w.x;
    cy = pose[b + 1] + w.y;
  }

  bool point_in_exit(float x, float y) const {
    return x >= exit_x0 && x <= exit_x1 && y >= exit_y0 && y <= exit_y1;
  }

  bool cog_in_exit(const float *pose, int tee) const {
    float cx, cy;
    tee_cog(pose, tee, cx, cy);
    return point_in_exit(cx, cy);
  }

  // Both rects of the tee. The exit box is convex, so a CoG outside it means
  // some corner is outside and the corner loop can be skipped.
  bool tee_in_exit(const float *pose, int tee) const {
    if (!cog_in_exit(pose, tee))
      return false;
    const int b = pusht_multi_physics::tee_pose(tee);
    const float c = std::cos(pose[b + 2]);
    const float s = std::sin(pose[b + 2]);
    pusht_qs::LocalRect rects[2];
    pusht_qs::tee_rects(rects[0], rects[1]);
    for (const auto &r : rects) {
      const float xs[2] = {r.xmin, r.xmax};
      const float ys[2] = {r.ymin, r.ymax};
      for (float lx : xs) {
        for (float ly : ys) {
          const pusht_qs::Vec2 w = pusht_qs::rot({lx, ly}, c, s);
          if (!point_in_exit(pose[b] + w.x, pose[b + 1] + w.y))
            return false;
        }
      }
    }
    return true;
  }

  void stamp_cleared(Node &node, const Node *parent,
                     const float *intermediates) const {
    for (int i = 0; i < kNumTees; i++) {
      node.cleared[i] = parent ? parent->cleared[i] : false;
      if (tee_in_exit(node.pose, i))
        node.cleared[i] = true;
    }
    if (!intermediates)
      return;
    for (int t = 0; t < timesteps; t++) {
      const float *s = intermediates + t * n_dims;
      for (int i = 0; i < kNumTees; i++) {
        if (tee_in_exit(s, i))
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
                                       tolerance, division_factor,
                                       timesteps, local_controllability_radius,
                                       time_direction);
    stamp_cleared(*node, nullptr, nullptr);
    return node;
  }

  float compute_near_meet_distance(float *pose1, float *pose2) {
    return heuristic(pose1, pose2);
  }

  void solve_2pBVP(
      const std::vector<std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>>>
          &pairs,
      std::vector<bool> &valid,
      std::vector<std::vector<float>> &trajectories) {
    mpc_solve(pairs, valid, trajectories);
  }

  // Default: geometric interpolation. Override for a dynamics-aware MPC.
  virtual void mpc_solve(
      const std::vector<std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>>>
          &pairs,
      std::vector<bool> &valid,
      std::vector<std::vector<float>> &trajectories) {
    near_meet_.mpc_solve(
        pairs, valid, trajectories,
        [this](const std::vector<float *> &states, std::vector<bool> &results) {
          check_validity_batched(states, results);
        });
  }

  // Default calls sampling::sample_goal_region. Override to replace the sampler.
  virtual std::vector<std::shared_ptr<Node>>
  sample_goals(std::shared_ptr<Node> start, std::shared_ptr<Node> goal) {
    return sampling::default_sample_goals(this, std::move(start), std::move(goal));
  }

  float distance(float *pose, float *goal) { return heuristic(pose, goal); }

  static float wrap_angle(float d) {
    while (d > float(M_PI))
      d -= 2.f * float(M_PI);
    while (d < -float(M_PI))
      d += 2.f * float(M_PI);
    return d;
  }

  // Body xy plus a small angle term (40 px per radian), same scale single-T used.
  // Without the angle term the search sits near the goal pose and misses epsilon.
  float h_rearrange(const float *pose, const float *goal) const {
    float dist = 0.f;
    for (int i = 0; i < kNumTees; i++) {
      const int b = pusht_multi_physics::tee_pose(i);
      const float dx = pose[b] - goal[b];
      const float dy = pose[b + 1] - goal[b + 1];
      const float dth = wrap_angle(pose[b + 2] - goal[b + 2]);
      dist += std::sqrt(dx * dx + dy * dy) + 40.f * std::fabs(dth);
    }
    return max_agent_vel > 0.f ? dist / max_agent_vel : 0.f;
  }

  bool tee_at_goal(const float *pose, const float *goal, int tee) const {
    const int b = pusht_multi_physics::tee_pose(tee);
    return std::fabs(pose[b] - goal[b]) < epsilon[b] &&
           std::fabs(pose[b + 1] - goal[b + 1]) < epsilon[b + 1] &&
           std::fabs(wrap_angle(pose[b + 2] - goal[b + 2])) < epsilon[b + 2];
  }

  // Clear: a tee counts as cleared only if the whole tee is in the exit box
  // now. Successors set f from Node::cleared, which also remembers a past
  // entry. Rearrange: distance of each tee body to its goal pose.
  float heuristic(float *pose, float *goal) {
    if (rearrange)
      return h_rearrange(pose, goal);
    bool now[kNumTees];
    for (int i = 0; i < kNumTees; i++)
      now[i] = tee_in_exit(pose, i);
    return h_cleared(pose, now);
  }

  bool reached_goal_region(std::shared_ptr<Node> v, std::shared_ptr<Node> goal) {
    if (!rearrange)
      return v->cleared[0] && v->cleared[1];
    return tee_at_goal(v->pose, goal->pose, 0) &&
           tee_at_goal(v->pose, goal->pose, 1);
  }

  bool finite_pose(const float *pose) const {
    for (int i = 0; i < hash_dims; i++) {
      if (!std::isfinite(pose[i]))
        return false;
    }
    return true;
  }

  bool in_agent_workspace(const float *pose) const {
    for (int i = 0; i < kNumAgents; i++) {
      const int p = pusht_multi_physics::agent_pose(i);
      if (pose[p] < workspace_lo || pose[p] > agent_x_hi ||
          pose[p + 1] < workspace_lo || pose[p + 1] > workspace_hi)
        return false;
      if (split_agents && kNumAgents == 2 &&
          (i == 0 ? pose[p + 1] < agent_y_split : pose[p + 1] > agent_y_split))
        return false;
    }
    if (kNumAgents == 2 && min_agent_sep > 0.f) {
      const int p0 = pusht_multi_physics::agent_pose(0);
      const int p1 = pusht_multi_physics::agent_pose(1);
      const float dx = pose[p0] - pose[p1];
      const float dy = pose[p0 + 1] - pose[p1 + 1];
      if (dx * dx + dy * dy < min_agent_sep * min_agent_sep)
        return false;
    }
    return true;
  }

  bool valid_at(float *pose, const pusht_multi_physics::WallSpec *walls,
                int n_walls) {
    if (!finite_pose(pose) || !in_agent_workspace(pose))
      return false;
    if (!rearrange) {
      for (int i = 0; i < kNumTees; i++) {
        if (tee_in_exit(pose, i))
          return false;
      }
    }
    return !pusht_multi_physics::overlaps_forbidden(pose, walls, n_walls,
                                                   close_right);
  }

  bool check_validity_single(float *pose) {
    return valid_at(pose, walls_, n_walls_);
  }

  void check_validity(float *start, float *goal, bool *result) {
    result[0] = check_validity_single(start);
    result[1] = check_validity_single(goal);
  }

  void check_validity_batched(const std::vector<float *> &states,
                              std::vector<bool> &results) {
    const int n = static_cast<int>(states.size());
    results.resize(n);
    const bool per_row =
        world_is_batch && world_stride > 0 &&
        static_cast<int>(world_rows.size()) == world_stride * n;
    std::vector<char> ok(n, 0);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
      pusht_multi_physics::WallSpec row[pusht_multi_physics::kMaxWalls];
      int n_row = n_walls_;
      const pusht_multi_physics::WallSpec *use = walls_;
      if (per_row) {
        n_row = read_walls(world_rows.data() + i * world_stride, world_stride,
                           row);
        use = row;
      }
      ok[i] = valid_at(states[i], use, n_row) ? 1 : 0;
    }
    for (int i = 0; i < n; i++)
      results[i] = ok[i] != 0;
  }

  // Uncleared tee this agent is touching, else the nearest uncleared tee.
  int select_tee(const Node &node, const float *goal_pose, int agent) const {
    const float *p = node.pose;
    const int ap = pusht_multi_physics::agent_pose(agent);
    int best = -1;
    float best_score = -1e30f;
    for (int i = 0; i < kNumTees; i++) {
      if (rearrange) {
        if (goal_pose && tee_at_goal(p, goal_pose, i))
          continue;
      } else if (node.cleared[i]) {
        continue;
      }
      const int tp = pusht_multi_physics::tee_pose(i);
      float pen = 0.f;
      pusht_qs::Vec2 nrm, contact, cog;
      const bool hit = pusht_qs::deepest_contact(
          {p[ap], p[ap + 1]}, {p[tp], p[tp + 1]}, p[tp + 2], pen, nrm, contact,
          cog);
      const float dx = p[ap] - p[tp];
      const float dy = p[ap + 1] - p[tp + 1];
      const float score = hit ? pen : -std::sqrt(dx * dx + dy * dy);
      if (score > best_score) {
        best_score = score;
        best = i;
      }
    }
    return best < 0 ? 0 : best;
  }

  // The actions of one successor expansion: for each agent, its single-pusher ring (Coulomb cone on
  // contact, uniform off contact) plus a hold, with every other agent commanded to hold.
  // Appends n_succ actions of n_cont floats each.
  void gen_actions(const std::shared_ptr<Node> &node,
                   const std::shared_ptr<Node> &goal,
                   std::vector<float> &actions) {
    const float mu = pusht_multi_physics::contact_friction_mu();
    for (int agent = 0; agent < kNumAgents; agent++) {
      const int tee = select_tee(*node, goal->pose, agent);
      const int tp = pusht_multi_physics::tee_pose(tee);
      const int ap = pusht_multi_physics::agent_pose(agent);
      float ring_pose[5] = {node->pose[ap], node->pose[ap + 1], node->pose[tp],
                            node->pose[tp + 1], node->pose[tp + 2]};
      std::vector<float> ring;
      for (float r : action_radii)
        pusht_qs::append_qs_ring_targets(ring_pose, r, n_directions,
                                         workspace_lo, workspace_hi, mu, ring);
      ring.push_back(node->pose[tp]);
      ring.push_back(node->pose[tp + 1]);
      const int n_ring = int(ring.size()) / 2;
      for (int k = 0; k < n_ring; k++) {
        for (int j = 0; j < kNumAgents; j++) {
          if (j == agent) {
            actions.push_back(ring[2 * k]);
            actions.push_back(ring[2 * k + 1]);
          } else {
            const int jp = pusht_multi_physics::agent_pose(j);
            actions.push_back(node->pose[jp]);
            actions.push_back(node->pose[jp + 1]);
          }
        }
      }
    }
  }

  // Whether a rolled-out edge is usable: finite, inside the workspace at the end and along the whole
  // edge (the half-plane and separation limits hold along it), and within the penetration and force
  // limits.
  bool rollout_valid(const float *out, const float *inter, float pen_inc,
                     float peak_force) const {
    bool path_ok = finite_pose(out) && in_agent_workspace(out);
    for (int t = 0; path_ok && t < timesteps; t++)
      path_ok = in_agent_workspace(inter + t * n_dims);
    return path_ok && pen_inc <= max_penetration_px &&
           peak_force <= max_agent_force;
  }

  // The child node for one valid edge: state ns, the substep states inter, and the action that made it.
  std::shared_ptr<Node> make_neighbor(const std::shared_ptr<Node> &node,
                                      const std::shared_ptr<Node> &goal,
                                      const float *action, const float *ns,
                                      const float *inter) {
    float ctrl[n_cont];
    memcpy(ctrl, action, n_cont * sizeof(float));
    float *controls_seq = new float[timesteps * n_cont];
    for (int t = 0; t < timesteps; t++)
      memcpy(&controls_seq[t * n_cont], ctrl, n_cont * sizeof(float));
    auto neighbor = std::make_shared<Node>(
        ns, inter, node, node->g + edge_cost, resolution, tolerance, division_factor, timesteps, local_controllability_radius,
        time_direction, controls_seq);
    delete[] controls_seq;
    stamp_cleared(*neighbor, node.get(), inter);
    neighbor->f =
        neighbor->g + (rearrange
                           ? h_rearrange(neighbor->pose, goal->pose)
                           : h_cleared(neighbor->pose, neighbor->cleared));
    return neighbor;
  }

  std::vector<std::shared_ptr<Node>> Succ(std::shared_ptr<Node> node,
                                          std::shared_ptr<Node> goal) {
    std::vector<std::shared_ptr<Node>> neighbors;
    neighbors.reserve(n_succ);

    std::vector<float> actions;
    actions.reserve(n_succ * n_cont);
    gen_actions(node, goal, actions);

    const int n_actions = int(actions.size()) / n_cont;
    std::vector<float> next_states(n_actions * n_dims);
    std::vector<float> intermediates(n_actions * timesteps * n_dims);
    std::vector<char> valid(n_actions, 0);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_actions; i++) {
      float out[n_dims];
      float pen_inc = 0.f;
      float peak_force = 0.f;
      float *inter = &intermediates[i * timesteps * n_dims];
      pusht_multi_physics::step_from_snapshot(
          node->pose, &actions[i * n_cont], out, n_substeps, inter,
          max_agent_vel, &pen_inc, &peak_force, walls_, n_walls_, close_right);
      memcpy(&next_states[i * n_dims], out, n_dims * sizeof(float));
      valid[i] = rollout_valid(out, inter, pen_inc, peak_force) ? 1 : 0;
    }

    for (int i = 0; i < n_actions; i++) {
      if (!valid[i])
        continue;
      neighbors.push_back(make_neighbor(
          node, goal, &actions[i * n_cont], &next_states[i * n_dims],
          &intermediates[i * timesteps * n_dims]));
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

// Open-loop Chipmunk step. target is one xy per agent.
// Returns [state, penetration increase, peak agent force].
// world is one row: [n, x1, y1, x2, y2, ...]. close_right adds the box wall at x=506.
#define IGHASTAR_HAS_PUSHT_MULTI_STEP
inline torch::Tensor pusht_multi_step(torch::Tensor state, torch::Tensor world,
                                      torch::Tensor target, int n_substeps,
                                      bool close_right, double max_agent_vel) {
  auto s = state.to(torch::kCPU).to(torch::kFloat32).contiguous().view({-1});
  if (s.size(0) < pusht_multi_physics::kStateDim)
    throw std::runtime_error("pusht_multi_step: state is short");
  auto tgt = target.to(torch::kCPU).to(torch::kFloat32).contiguous().view({-1});
  if (tgt.size(0) < n_cont)
    throw std::runtime_error("pusht_multi_step: target needs one xy per agent");
  float in[pusht_multi_physics::kStateDim];
  float tgt_xy[n_cont];
  std::memcpy(in, s.data_ptr<float>(), sizeof(in));
  std::memcpy(tgt_xy, tgt.data_ptr<float>(), sizeof(tgt_xy));

  pusht_multi_physics::WallSpec walls[pusht_multi_physics::kMaxWalls];
  int n = 0;
  if (world.defined() && world.numel() > 0) {
    auto w = world.to(torch::kCPU).to(torch::kFloat32).contiguous().view({-1});
    n = Environment::read_walls(w.data_ptr<float>(), static_cast<int>(w.size(0)),
                                walls);
  }
  if (n_substeps < 1)
    n_substeps = 1;
  if (n_substeps > 500)
    n_substeps = 500;
  float out[pusht_multi_physics::kStateDim];
  float pen = 0.f;
  float force = 0.f;
  pusht_multi_physics::step_from_snapshot(
      in, tgt_xy, out, n_substeps, nullptr, static_cast<cpFloat>(max_agent_vel),
      &pen, &force, walls, n, close_right);
  auto result = torch::empty({pusht_multi_physics::kStateDim + 2},
                             torch::TensorOptions().dtype(torch::kFloat32));
  std::memcpy(result.data_ptr<float>(), out,
              sizeof(float) * pusht_multi_physics::kStateDim);
  result[pusht_multi_physics::kStateDim] = pen;
  result[pusht_multi_physics::kStateDim + 1] = force;
  return result;
}
