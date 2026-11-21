#pragma once

// Default 2-point BVP for built-in environments. Environment::solve_2pBVP calls
// Environment::mpc_solve. The validity-only mpc_solve is geometric_solve.
// An environment that wants a dynamics-aware connection instead calls the
// rollout overload below; kinodynamic and the others stay on geometric_solve.
// shortcut: checks one pair at a time. Pack every pair's samples into one
// check_validity_batched if this shows up hot.

#include "../../utils/config_utils.h"
#include "../../utils/sampling_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <utility>
#include <vector>

inline bool mpc_shooting_self_check();

inline const char *cem_config_error() {
  return "near_meet_config must set mpc_num_rollouts, action_noise, "
         "num_iters, topk, and cost_function_weights";
}

// Default problem filter: every stored trajectory stays eligible. An
// environment that can tell problems apart overrides filter_trajectories.
#define IGHASTAR_FILTER_TRAJECTORIES_ALL                                          \
  void filter_trajectories(                                                       \
      const std::vector<std::vector<float>> &stored_goals,                        \
      const std::vector<std::vector<float>> &stored_worlds, const float *start,   \
      const float *goal, std::vector<uint8_t> &keep) const {                      \
    (void)stored_goals;                                                           \
    (void)stored_worlds;                                                          \
    (void)start;                                                                  \
    (void)goal;                                                                   \
    for (uint8_t &bit : keep)                                                     \
      bit = 1;                                                                    \
  }

// One near-meet edge, in generator order. poses has timesteps * n_dims
// samples, or is empty when the meet is the source state itself. ok is false
// when the connection cannot be stored.
struct MeetEdge {
  std::vector<float> poses;
  std::vector<float> controls;
  std::vector<float> times;
  float cost = 0.f;
  float arrival = 0.f;
  bool ok = false;
};

struct NearMeet2PBVP {
  int n_dims = 0;
  int num_interpolation_points = 5;
  int num_perturbations = 3;
  float perturbation_scale = 1.0f;
  std::vector<int> angular_dims;
  std::vector<std::vector<float>> cached_perturbations;
  // Shooting only. Geometric interpolation does not read these.
  bool cem_ready = false;
  int mpc_num_rollouts = 0;
  int num_iters = 0;
  int topk = 0;
  std::vector<float> action_noise;
  std::vector<float> cost_weights;

  void load(const py::dict &config, int dims, const float *lcr,
            const std::vector<int> &angular) {
    n_dims = dims;
    angular_dims = angular;
    auto nm = config_utils::get_config_dict(config, "near_meet_config").first;
    if (nm.contains("num_interpolation_points")) {
      num_interpolation_points = nm["num_interpolation_points"].cast<int>();
    }
    if (nm.contains("num_perturbations")) {
      num_perturbations = nm["num_perturbations"].cast<int>();
    }
    if (nm.contains("perturbation_scale")) {
      perturbation_scale = nm["perturbation_scale"].cast<float>();
    }
    const bool has_rollouts = nm.contains("mpc_num_rollouts");
    const bool has_noise = nm.contains("action_noise");
    const bool has_iters = nm.contains("num_iters");
    const bool has_topk = nm.contains("topk");
    const bool has_weights = nm.contains("cost_function_weights");
    const int n_set = static_cast<int>(has_rollouts) + static_cast<int>(has_noise) +
                      static_cast<int>(has_iters) + static_cast<int>(has_topk) +
                      static_cast<int>(has_weights);
    if (n_set != 0 && n_set != 5) {
      throw std::runtime_error(cem_config_error());
    }
    if (n_set == 5) {
      mpc_num_rollouts = nm["mpc_num_rollouts"].cast<int>();
      num_iters = nm["num_iters"].cast<int>();
      topk = nm["topk"].cast<int>();
      action_noise = nm["action_noise"].cast<std::vector<float>>();
      cost_weights = nm["cost_function_weights"].cast<std::vector<float>>();
      if (mpc_num_rollouts < 1 || num_iters < 1 || topk < 1 ||
          topk > mpc_num_rollouts) {
        throw std::runtime_error("near_meet_config CEM counts are invalid");
      }
      if (static_cast<int>(cost_weights.size()) != n_dims) {
        throw std::runtime_error(
            "near_meet_config cost_function_weights must have one entry per state dimension");
      }
      for (float scale : action_noise) {
        if (scale < 0.f) {
          throw std::runtime_error(
              "near_meet_config action_noise entries must be >= 0");
        }
      }
      cem_ready = true;
    }
    cached_perturbations = sampling::generate_perturbation_cache(
        lcr, n_dims, num_perturbations, perturbation_scale);
    if (!mpc_shooting_self_check()) {
      throw std::runtime_error("mpc shooting self-check failed");
    }
  }

  template <typename NodeT, typename ValidityChecker>
  void geometric_solve(const std::vector<std::pair<std::shared_ptr<NodeT>,
                                         std::shared_ptr<NodeT>>> &pairs,
             std::vector<bool> &valid,
             std::vector<std::vector<float>> &trajectories,
             ValidityChecker &&validity_checker) const {
    const size_t n = pairs.size();
    valid.assign(n, true);
    trajectories.assign(n, {});
    if (num_interpolation_points == 0 && num_perturbations == 0) {
      return;
    }
    for (size_t i = 0; i < n; ++i) {
      const float *source = pairs[i].first->pose;
      const float *target = pairs[i].second->pose;
      valid[i] = sampling::check_interpolation_validity(
          source, target, n_dims, angular_dims,
          num_interpolation_points, cached_perturbations, validity_checker);
      if (!valid[i] || num_interpolation_points <= 1) {
        continue;
      }
      const int n_interior = num_interpolation_points - 1;
      std::vector<float> traj(static_cast<size_t>(n_interior) * n_dims);
      for (int k = 1; k < num_interpolation_points; ++k) {
        float t = static_cast<float>(k) /
                  static_cast<float>(num_interpolation_points);
        sampling::interpolate_pose(source, target, t, n_dims, angular_dims,
                                   traj.data() + (k - 1) * n_dims);
      }
      trajectories[i] = std::move(traj);
    }
  }

  // Default MPC is geometric interpolation. Environment::mpc_solve calls this.
  template <typename NodeT, typename ValidityChecker>
  void mpc_solve(const std::vector<std::pair<std::shared_ptr<NodeT>,
                                             std::shared_ptr<NodeT>>> &pairs,
                 std::vector<bool> &valid,
                 std::vector<std::vector<float>> &trajectories,
                 ValidityChecker &&validity_checker) const {
    geometric_solve(pairs, valid, trajectories,
                    std::forward<ValidityChecker>(validity_checker));
  }

  // True when each component of a is within tol of b. Angular components use
  // the shortest wrap. A null tolerance accepts nothing.
  bool within_tolerance(const float *a, const float *b,
                        const float *tol) const {
    if (tol == nullptr || a == nullptr || b == nullptr) {
      return false;
    }
    for (int d = 0; d < n_dims; ++d) {
      const float diff = sampling::is_angular_dim(d, angular_dims)
                             ? sampling::angular_diff(a[d], b[d])
                             : a[d] - b[d];
      if (std::fabs(diff) > tol[d]) {
        return false;
      }
    }
    return true;
  }

  // Weighted terminal cost. Angular dimensions use the shortest wrap.
  float weighted_cost(const float *a, const float *b) const {
    float cost = 0.f;
    for (int d = 0; d < n_dims; ++d) {
      const float diff = sampling::is_angular_dim(d, angular_dims)
                             ? sampling::angular_diff(a[d], b[d])
                             : a[d] - b[d];
      cost += cost_weights[static_cast<size_t>(d)] * diff * diff;
    }
    return cost;
  }

  // One-shot Gaussian used when the config does not opt into CEM.
  template <typename NodeT, typename Rollout, typename Distance>
  void mpc_solve_shooting(
      const std::vector<std::pair<std::shared_ptr<NodeT>,
                                  std::shared_ptr<NodeT>>> &pairs,
      std::vector<bool> &valid, std::vector<std::vector<float>> &trajectories,
      int n_cont, int timesteps, Rollout &&rollout, Distance &&distance,
      const float *tolerance, bool hold_action) const {
    const size_t n = pairs.size();
    valid.assign(n, false);
    trajectories.assign(n, {});
    if (n_cont <= 0 || timesteps <= 0) {
      return;
    }
    const int n_ctrl = timesteps * n_cont;
    const int n_shots = num_perturbations > 0 ? num_perturbations : 8;
    std::mt19937 gen(1);
    std::normal_distribution<float> noise(0.f, perturbation_scale);
    std::vector<float> controls(static_cast<size_t>(n_ctrl));
    std::vector<float> final_pose(static_cast<size_t>(n_dims > 0 ? n_dims : 1));
    for (size_t i = 0; i < n; ++i) {
      const auto &u = pairs[i].first;
      const auto &v = pairs[i].second;
      if (!u || !v) {
        continue;
      }
      if (!u->parent || u->intermediate_controls == nullptr) {
        valid[i] = within_tolerance(u->pose, v->pose, tolerance);
        continue;
      }
      const int shot_n = hold_action ? n_cont : n_ctrl;
      float best = 1e30f;
      std::vector<float> best_controls;
      for (int s = 0; s <= n_shots; ++s) {
        for (int c = 0; c < shot_n; ++c) {
          controls[static_cast<size_t>(c)] =
              u->intermediate_controls[c] + (s == 0 ? 0.f : noise(gen));
        }
        if (hold_action) {
          for (int t = 1; t < timesteps; ++t) {
            std::memcpy(controls.data() + static_cast<size_t>(t) * n_cont,
                        controls.data(),
                        static_cast<size_t>(n_cont) * sizeof(float));
          }
        }
        if (!rollout(u->pose, controls.data(), n_cont, timesteps,
                     final_pose.data())) {
          continue;
        }
        if (!within_tolerance(final_pose.data(), v->pose, tolerance)) {
          continue;
        }
        const float d = distance(final_pose.data(), v->pose);
        if (d < best) {
          best = d;
          best_controls = controls;
        }
      }
      if (best_controls.empty()) {
        continue;
      }
      valid[i] = true;
      trajectories[i] = std::move(best_controls);
    }
  }

  // CEM on the held action when near_meet_config sets the five CEM keys.
  // Otherwise this is the previous one-shot Gaussian, so an environment that
  // does not opt in keeps its meets. pairs are (u, v): u is the vertex the
  // search attaches to, v is the library state to reach. The warmstart is the
  // control that produced u. hold_action repeats the first n_cont block
  // across timesteps. CEM draws mpc_num_rollouts samples from a diagonal
  // Gaussian, keeps the topk lowest weighted costs, and refits the mean and
  // std. The stored control is the lowest-cost sample that is also inside
  // tolerance. A parentless u is a meet only when its pose is already inside
  // tolerance.
  // ponytail: diagonal covariance, fixed seed, std may collapse to 0.
  // Upgrade path is a full covariance or an MPPI weight if topk shows up hot.
  template <typename NodeT, typename Rollout, typename Distance>
  void mpc_solve(const std::vector<std::pair<std::shared_ptr<NodeT>,
                                             std::shared_ptr<NodeT>>> &pairs,
                 std::vector<bool> &valid,
                 std::vector<std::vector<float>> &trajectories, int n_cont,
                 int timesteps, Rollout &&rollout, Distance &&distance,
                 const float *tolerance, bool hold_action = false) const {
    if (!cem_ready) {
      mpc_solve_shooting(pairs, valid, trajectories, n_cont, timesteps,
                         std::forward<Rollout>(rollout),
                         std::forward<Distance>(distance), tolerance,
                         hold_action);
      return;
    }
    const size_t n = pairs.size();
    valid.assign(n, false);
    trajectories.assign(n, {});
    if (n_cont <= 0 || timesteps <= 0) {
      return;
    }
    const int n_ctrl = timesteps * n_cont;
    const int shot_n = hold_action ? n_cont : n_ctrl;
    if (static_cast<int>(action_noise.size()) != shot_n) {
      throw std::runtime_error(
          "near_meet_config action_noise must have one entry per held action");
    }
    if (static_cast<int>(cost_weights.size()) != n_dims) {
      throw std::runtime_error(
          "near_meet_config cost_function_weights must have one entry per state dimension");
    }
    std::mt19937 gen(1);
    std::normal_distribution<float> unit(0.f, 1.f);
    std::vector<float> final_pose(static_cast<size_t>(n_dims > 0 ? n_dims : 1));
    for (size_t i = 0; i < n; ++i) {
      const auto &u = pairs[i].first;
      const auto &v = pairs[i].second;
      if (!u || !v) {
        continue;
      }
      if (!u->parent || u->intermediate_controls == nullptr) {
        valid[i] = within_tolerance(u->pose, v->pose, tolerance);
        continue;
      }
      std::vector<float> mean(static_cast<size_t>(shot_n));
      std::vector<float> stddev(static_cast<size_t>(shot_n));
      for (int c = 0; c < shot_n; ++c) {
        mean[static_cast<size_t>(c)] = u->intermediate_controls[c];
        stddev[static_cast<size_t>(c)] = action_noise[static_cast<size_t>(c)];
      }
      struct Shot {
        std::vector<float> held;
        std::vector<float> full;
        float cost;
        bool in_tol;
      };
      float best_cost = std::numeric_limits<float>::infinity();
      std::vector<float> best_full;
      for (int iter = 0; iter < num_iters; ++iter) {
        std::vector<Shot> shots;
        shots.reserve(static_cast<size_t>(mpc_num_rollouts));
        for (int s = 0; s < mpc_num_rollouts; ++s) {
          Shot shot;
          shot.held.resize(static_cast<size_t>(shot_n));
          const bool warm = iter == 0 && s == 0;
          for (int c = 0; c < shot_n; ++c) {
            const float z = warm ? 0.f : unit(gen);
            shot.held[static_cast<size_t>(c)] =
                mean[static_cast<size_t>(c)] +
                stddev[static_cast<size_t>(c)] * z;
          }
          shot.full.assign(static_cast<size_t>(n_ctrl), 0.f);
          std::memcpy(shot.full.data(), shot.held.data(),
                      static_cast<size_t>(shot_n) * sizeof(float));
          if (hold_action) {
            for (int t = 1; t < timesteps; ++t) {
              std::memcpy(shot.full.data() + static_cast<size_t>(t) * n_cont,
                          shot.held.data(),
                          static_cast<size_t>(n_cont) * sizeof(float));
            }
          }
          if (!rollout(u->pose, shot.full.data(), n_cont, timesteps,
                       final_pose.data())) {
            continue;
          }
          shot.cost = weighted_cost(final_pose.data(), v->pose);
          shot.in_tol = within_tolerance(final_pose.data(), v->pose, tolerance);
          if (shot.in_tol && shot.cost < best_cost) {
            best_cost = shot.cost;
            best_full = shot.full;
          }
          shots.push_back(std::move(shot));
        }
        if (shots.empty()) {
          break;
        }
        std::sort(shots.begin(), shots.end(),
                  [](const Shot &a, const Shot &b) { return a.cost < b.cost; });
        const int k = std::min(topk, static_cast<int>(shots.size()));
        std::vector<float> next_mean(static_cast<size_t>(shot_n), 0.f);
        for (int e = 0; e < k; ++e) {
          for (int c = 0; c < shot_n; ++c) {
            next_mean[static_cast<size_t>(c)] +=
                shots[static_cast<size_t>(e)].held[static_cast<size_t>(c)];
          }
        }
        const float inv = 1.f / static_cast<float>(k);
        for (int c = 0; c < shot_n; ++c) {
          next_mean[static_cast<size_t>(c)] *= inv;
        }
        for (int c = 0; c < shot_n; ++c) {
          float var = 0.f;
          for (int e = 0; e < k; ++e) {
            const float d =
                shots[static_cast<size_t>(e)].held[static_cast<size_t>(c)] -
                next_mean[static_cast<size_t>(c)];
            var += d * d;
          }
          mean[static_cast<size_t>(c)] = next_mean[static_cast<size_t>(c)];
          stddev[static_cast<size_t>(c)] = std::sqrt(var * inv);
        }
      }
      if (best_full.empty()) {
        continue;
      }
      valid[i] = true;
      trajectories[i] = std::move(best_full);
    }
  }
};

// The warmstart is applied at u, not at u->parent. A rollout that misses the
// tolerance is not a meet. hold_action copies the first control across steps.
inline bool mpc_shooting_self_check() {
  struct Node {
    float pose[2] = {};
    std::shared_ptr<Node> parent;
    float *intermediate_controls = nullptr;
  };
  auto parent = std::make_shared<Node>();
  parent->pose[0] = 0.f;
  auto u = std::make_shared<Node>();
  u->pose[0] = 1.f;
  u->parent = parent;
  float warm[2] = {1.f, 7.f};
  u->intermediate_controls = warm;
  auto v = std::make_shared<Node>();
  v->pose[0] = 2.f;
  NearMeet2PBVP solver;
  solver.n_dims = 1;
  solver.num_perturbations = 0;
  solver.perturbation_scale = 0.f;
  solver.cem_ready = true;
  solver.mpc_num_rollouts = 1;
  solver.num_iters = 1;
  solver.topk = 1;
  solver.action_noise = {0.f};
  solver.cost_weights = {1.f};
  std::vector<std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>>> pairs = {
      {u, v}};
  std::vector<bool> valid;
  std::vector<std::vector<float>> trajectories;
  const float *seen = nullptr;
  auto step = [&](const float *start, const float *controls, int, int steps,
                  float *final_pose) {
    seen = start;
    if (steps > 1 && controls[0] != controls[steps - 1]) {
      return false;
    }
    final_pose[0] = start[0] + controls[0];
    return true;
  };
  auto dist = [](const float *a, const float *b) {
    return std::fabs(a[0] - b[0]);
  };
  float tol[1] = {1e-3f};
  solver.mpc_solve(pairs, valid, trajectories, 1, 2, step, dist, tol, true);
  if (!valid[0] || seen != u->pose || trajectories[0].size() != 2 ||
      trajectories[0][0] != 1.f || trajectories[0][1] != 1.f) {
    return false;
  }
  v->pose[0] = 2.5f;
  tol[0] = 0.1f;
  solver.mpc_solve(pairs, valid, trajectories, 1, 2, step, dist, tol, true);
  if (valid[0]) {
    return false;
  }
  tol[0] = 1e-3f;
  v->pose[0] = 2.f;
  auto reject = [](const float *, const float *, int, int, float *) {
    return false;
  };
  solver.mpc_solve(pairs, valid, trajectories, 1, 1, reject, dist, tol);
  if (valid[0]) {
    return false;
  }
  auto start = std::make_shared<Node>();
  start->pose[0] = 2.f;
  pairs = {{start, v}};
  solver.mpc_solve(pairs, valid, trajectories, 1, 1, reject, dist, tol);
  if (!valid[0]) {
    return false;
  }
  start->pose[0] = 2.f + 10.f * tol[0];
  solver.mpc_solve(pairs, valid, trajectories, 1, 1, reject, dist, tol);
  if (valid[0]) {
    return false;
  }
  solver.angular_dims = {0};
  start->pose[0] = 0.f;
  v->pose[0] = 6.f;
  tol[0] = 0.3f;
  solver.mpc_solve(pairs, valid, trajectories, 1, 1, reject, dist, tol);
  if (!valid[0]) {
    return false;
  }
  v->pose[0] = 3.f;
  solver.mpc_solve(pairs, valid, trajectories, 1, 1, reject, dist, tol);
  if (valid[0]) {
    return false;
  }
  // Lower cost outside the tolerance box, higher cost inside it. The meet
  // is the inside sample. One rollout sees only the outside sample.
  solver.n_dims = 2;
  solver.angular_dims.clear();
  solver.mpc_num_rollouts = 2;
  solver.cost_weights = {1.f, 100.f};
  u->parent = parent;
  u->intermediate_controls = warm;
  v->pose[0] = 0.f;
  v->pose[1] = 0.f;
  pairs = {{u, v}};
  float tol2[2] = {0.1f, 100.f};
  int which = 0;
  auto two = [&](const float *, const float *, int, int, float *final_pose) {
    if (which++ == 0) {
      final_pose[0] = 1.f;
      final_pose[1] = 0.f;
    } else {
      final_pose[0] = 0.f;
      final_pose[1] = 0.5f;
    }
    return true;
  };
  solver.mpc_solve(pairs, valid, trajectories, 1, 1, two, dist, tol2);
  if (!valid[0]) {
    return false;
  }
  which = 0;
  solver.mpc_num_rollouts = 1;
  solver.mpc_solve(pairs, valid, trajectories, 1, 1, two, dist, tol2);
  return !valid[0];
}
