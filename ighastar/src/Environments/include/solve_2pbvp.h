#pragma once

// Default 2-point BVP for built-in environments. Environment::solve_2pBVP calls
// Environment::mpc_solve. The validity-only mpc_solve is geometric_solve.
// An environment that wants a dynamics-aware connection instead calls the
// rollout overload below; kinodynamic and the others stay on geometric_solve.
// shortcut: checks one pair at a time. Pack every pair's samples into one
// check_validity_batched if this shows up hot.

#include "../../utils/config_utils.h"
#include "../../utils/sampling_utils.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

inline bool mpc_shooting_self_check();

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

  // Random-shooting MPC. pairs are (u, v): u is the vertex the search attaches
  // to, v is the library state to reach. The warmstart is the control that
  // produced u, integrated from u->pose (the attached edge leaves u, not
  // u->parent). hold_action repeats the first n_cont block across timesteps:
  // the dynamics hold one target, so a per-substep sequence would not be the
  // motion that was rolled out. A shot whose terminal state is outside
  // tolerance is dropped, then the closest remaining shot is kept.
  // trajectories[i] is the winning control sequence (timesteps * n_cont).
  // ponytail: one Gaussian per control element, fixed seed per call. Upgrade
  // path is a cross-entropy or MPPI update if the sample count shows up hot.
  template <typename NodeT, typename Rollout, typename Distance>
  void mpc_solve(const std::vector<std::pair<std::shared_ptr<NodeT>,
                                             std::shared_ptr<NodeT>>> &pairs,
                 std::vector<bool> &valid,
                 std::vector<std::vector<float>> &trajectories, int n_cont,
                 int timesteps, Rollout &&rollout, Distance &&distance,
                 const float *tolerance, bool hold_action = false) const {
    const size_t n = pairs.size();
    valid.assign(n, false);
    trajectories.assign(n, {});
    if (n_cont <= 0 || timesteps <= 0) {
      return;
    }
    const int n_ctrl = timesteps * n_cont;
    // num_perturbations == 0 means "geometric accepts every pair". A caller
    // that opted into shooting still needs samples around the warmstart.
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
      // The search start has no incoming edge, so there is no warmstart.
      // The same state, within tolerance, is already the meet.
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
};

// The warmstart is applied at u, not at u->parent. A rollout that misses the
// tolerance is not a meet. hold_action copies the first control across steps.
inline bool mpc_shooting_self_check() {
  struct Node {
    float pose[1];
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
  return !valid[0];
}
