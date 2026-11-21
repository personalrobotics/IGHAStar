#pragma once

// One 2-point BVP for every environment. near_meet_config.solver selects
// geometric or mpc. An omitted solver is geometric. A near-meet call without
// LCR throws: the caller has to name the solver and give that solver's
// parameters. mpc is CEM on a rollout the environment supplies. It does not
// fall back to another shooter.
// shortcut: checks one pair at a time. Pack every pair's samples into one
// check_validity_batched if this shows up hot.

#include "../../utils/config_utils.h"
#include "../../utils/sampling_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

inline bool mpc_shooting_self_check();

inline const char *cem_config_error() {
  return "near_meet_config solver mpc must set mpc_num_rollouts, action_noise, "
         "num_iters, topk, and cost_function_weights";
}

inline const char *meet_config_error() {
  return "near-meet was called without LCR. Set near_meet_config.solver to "
         "geometric or mpc, and set that solver's parameters.";
}

// Search entry only. The self-check calls mpc_solve directly and does not warn.
// The function-local static is initialized once, including across threads.
inline void warn_mpc_solve_untested() {
  static const bool shown = [] {
    std::cerr << "\033[1;31m"
              << "MPC_SOLVE IS UNTESTED AND SHOULD NOT BE USED. "
              << "https://github.com/personalrobotics/IGHAStar/issues/57\n"
              << "\033[0m";
    return true;
  }();
  (void)shown;
}

enum class MeetSolver { geometric, mpc };

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
  MeetSolver solver = MeetSolver::geometric;
  bool lcr_set = false;
  int mpc_num_rollouts = 0;
  int num_iters = 0;
  int topk = 0;
  std::vector<float> action_noise;
  std::vector<float> cost_weights;

  void load(const py::dict &config, int dims, const float *lcr,
            const std::vector<int> &angular, int n_cont) {
    n_dims = dims;
    angular_dims = angular;
    lcr_set = config.contains("LCR");
    if (!lcr_set && config.contains("experiment_info_default")) {
      lcr_set = config["experiment_info_default"].cast<py::dict>().contains("LCR");
    }
    auto nm = config_utils::get_config_dict(config, "near_meet_config").first;
    solver = MeetSolver::geometric;
    if (nm.contains("solver")) {
      const std::string name = nm["solver"].cast<std::string>();
      if (name == "mpc") {
        solver = MeetSolver::mpc;
      } else if (name != "geometric") {
        throw std::runtime_error(
            "near_meet_config.solver must be geometric or mpc");
      }
    }
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
    if (solver == MeetSolver::geometric && n_set != 0) {
      throw std::runtime_error(
          "near_meet_config.solver is geometric. Remove the CEM parameters, "
          "or set solver: mpc.");
    }
    if (solver == MeetSolver::mpc && n_set != 5) {
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
      if (static_cast<int>(action_noise.size()) != n_cont) {
        throw std::runtime_error(
            "near_meet_config action_noise must have one entry per held action");
      }
      for (float w : cost_weights) {
        if (w < 0.f && w != -1.f) {
          throw std::runtime_error(
              "near_meet_config cost_function_weights entries must be >= 0, or -1 for 1/tolerance");
        }
      }
      for (float scale : action_noise) {
        if (scale < 0.f) {
          throw std::runtime_error(
              "near_meet_config action_noise entries must be >= 0");
        }
      }
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

  // A stored weight of -1 is 1/tolerance for that dimension. An infinite
  // tolerance becomes weight 0.
  void resolve_weights(const float *tol, std::vector<float> &out) const {
    out.resize(static_cast<size_t>(n_dims));
    for (int d = 0; d < n_dims; ++d) {
      const float w = cost_weights[static_cast<size_t>(d)];
      if (w != -1.f) {
        out[static_cast<size_t>(d)] = w;
        continue;
      }
      if (tol == nullptr || !(tol[d] > 0.f)) {
        throw std::runtime_error(
            "cost weight -1 needs a positive tolerance on that dimension");
      }
      out[static_cast<size_t>(d)] = 1.f / tol[d];
    }
  }

  // Weighted terminal cost. Angular dimensions use the shortest wrap.
  float weighted_cost(const float *a, const float *b,
                      const float *weights) const {
    float cost = 0.f;
    for (int d = 0; d < n_dims; ++d) {
      const float diff = sampling::is_angular_dim(d, angular_dims)
                             ? sampling::angular_diff(a[d], b[d])
                             : a[d] - b[d];
      cost += weights[d] * diff * diff;
    }
    return cost;
  }

  // solver picks geometric_solve or mpc_solve. Missing LCR throws here,
  // which is the first time a near-meet actually runs.
  template <typename NodeT, typename ValidityChecker, typename Rollout>
  void solve(const std::vector<std::pair<std::shared_ptr<NodeT>,
                                         std::shared_ptr<NodeT>>> &pairs,
             std::vector<bool> &valid,
             std::vector<std::vector<float>> &trajectories,
             ValidityChecker &&validity_checker, Rollout &&rollout, int n_cont,
             int timesteps, const float *tolerance) const {
    if (!lcr_set) {
      throw std::runtime_error(meet_config_error());
    }
    if (solver == MeetSolver::geometric) {
      geometric_solve(pairs, valid, trajectories,
                      std::forward<ValidityChecker>(validity_checker));
      return;
    }
    warn_mpc_solve_untested();
    mpc_solve(pairs, valid, trajectories, n_cont, timesteps,
              std::forward<Rollout>(rollout), tolerance);
  }

  // CEM over every meet that has a parent control. One rollout call per
  // iteration covers n_shoot x mpc_num_rollouts samples. The held action is
  // repeated across timesteps while the controls are packed. topk refits the
  // next distribution. The stored control is the lowest-cost sample of every
  // last-iteration rollout that is also inside tolerance. A parentless u is a
  // meet only when its pose is already inside tolerance.
  // shortcut: diagonal covariance, fixed seed, std may collapse to 0.
  // Upgrade path is a full covariance or an MPPI weight if topk shows up hot.
  template <typename NodeT, typename Rollout>
  void mpc_solve(const std::vector<std::pair<std::shared_ptr<NodeT>,
                                             std::shared_ptr<NodeT>>> &pairs,
                 std::vector<bool> &valid,
                 std::vector<std::vector<float>> &trajectories, int n_cont,
                 int timesteps, Rollout &&rollout,
                 const float *tolerance) const {
    if (mpc_num_rollouts < 1 || num_iters < 1 || topk < 1) {
      throw std::runtime_error(cem_config_error());
    }
    if (n_cont <= 0 || timesteps <= 0) {
      throw std::runtime_error(
          "mpc near-meet needs a positive control dimension and timestep count");
    }
    const int n_meets = static_cast<int>(pairs.size());
    valid.assign(n_meets, false);
    trajectories.assign(n_meets, {});
    std::vector<int> shoot_ids;
    shoot_ids.reserve(static_cast<size_t>(n_meets));
    for (int meet = 0; meet < n_meets; ++meet) {
      const auto &u = pairs[static_cast<size_t>(meet)].first;
      const auto &v = pairs[static_cast<size_t>(meet)].second;
      if (!u || !v) {
        continue;
      }
      if (!u->parent || u->intermediate_controls == nullptr) {
        valid[meet] = within_tolerance(u->pose, v->pose, tolerance);
        continue;
      }
      shoot_ids.push_back(meet);
    }
    const int n_shoot = static_cast<int>(shoot_ids.size());
    if (n_shoot == 0) {
      return;
    }
    std::vector<float> weights;
    resolve_weights(tolerance, weights);
    const int n_rollouts = mpc_num_rollouts;
    const int n_ctrl = timesteps * n_cont;
    const size_t n_samples = static_cast<size_t>(n_shoot) * n_rollouts;
    std::vector<float> mean(static_cast<size_t>(n_shoot) * n_cont);
    std::vector<float> stddev(static_cast<size_t>(n_shoot) * n_cont);
    std::vector<float> states(static_cast<size_t>(n_shoot) * n_dims);
    for (int m = 0; m < n_shoot; ++m) {
      const auto &u = pairs[static_cast<size_t>(shoot_ids[m])].first;
      std::copy(u->pose, u->pose + n_dims,
                states.begin() + static_cast<size_t>(m) * n_dims);
      for (int c = 0; c < n_cont; ++c) {
        mean[static_cast<size_t>(m) * n_cont + c] = u->intermediate_controls[c];
        stddev[static_cast<size_t>(m) * n_cont + c] =
            action_noise[static_cast<size_t>(c)];
      }
    }
    std::vector<float> controls(n_samples * n_ctrl);
    std::vector<float> finals(n_samples * n_dims);
    std::vector<uint8_t> ok(n_samples, 0);
    std::vector<float> costs(n_samples);
    std::mt19937 gen(1);
    std::normal_distribution<float> unit(0.f, 1.f);
    const float no_cost = std::numeric_limits<float>::infinity();
    for (int iter = 0; iter < num_iters; ++iter) {
      for (int m = 0; m < n_shoot; ++m) {
        for (int r = 0; r < n_rollouts; ++r) {
          const bool warm = iter == 0 && r == 0;
          float *sample = controls.data() +
                          (static_cast<size_t>(m) * n_rollouts + r) * n_ctrl;
          for (int c = 0; c < n_cont; ++c) {
            const float z = warm ? 0.f : unit(gen);
            const float held =
                mean[static_cast<size_t>(m) * n_cont + c] +
                stddev[static_cast<size_t>(m) * n_cont + c] * z;
            for (int t = 0; t < timesteps; ++t) {
              sample[static_cast<size_t>(t) * n_cont + c] = held;
            }
          }
        }
      }
      rollout(states.data(), controls.data(), n_shoot, n_rollouts, timesteps,
              n_cont, finals.data(), ok.data());
      for (size_t s = 0; s < n_samples; ++s) {
        const int m = static_cast<int>(s) / n_rollouts;
        const auto &target = pairs[static_cast<size_t>(shoot_ids[m])].second;
        costs[s] = ok[s] ? weighted_cost(finals.data() + s * n_dims, target->pose,
                                         weights.data())
                         : no_cost;
      }
      const bool last = iter + 1 == num_iters;
      for (int m = 0; m < n_shoot; ++m) {
        std::vector<int> rank(static_cast<size_t>(n_rollouts));
        for (int r = 0; r < n_rollouts; ++r) {
          rank[static_cast<size_t>(r)] = r;
        }
        const float *meet_cost =
            costs.data() + static_cast<size_t>(m) * n_rollouts;
        std::sort(rank.begin(), rank.end(), [&](int a, int b) {
          return meet_cost[a] < meet_cost[b];
        });
        int k = 0;
        while (k < topk && k < n_rollouts && meet_cost[rank[k]] < no_cost) {
          ++k;
        }
        if (k > 0) {
          std::vector<float> next_mean(static_cast<size_t>(n_cont), 0.f);
          for (int e = 0; e < k; ++e) {
            const float *sample =
                controls.data() +
                (static_cast<size_t>(m) * n_rollouts + rank[e]) * n_ctrl;
            for (int c = 0; c < n_cont; ++c) {
              next_mean[static_cast<size_t>(c)] += sample[c];
            }
          }
          const float inv = 1.f / static_cast<float>(k);
          for (int c = 0; c < n_cont; ++c) {
            float var = 0.f;
            for (int e = 0; e < k; ++e) {
              const float *sample =
                  controls.data() +
                  (static_cast<size_t>(m) * n_rollouts + rank[e]) * n_ctrl;
              const float d = sample[c] - next_mean[static_cast<size_t>(c)] * inv;
              var += d * d;
            }
            mean[static_cast<size_t>(m) * n_cont + c] =
                next_mean[static_cast<size_t>(c)] * inv;
            stddev[static_cast<size_t>(m) * n_cont + c] = std::sqrt(var * inv);
          }
        }
        if (!last) {
          continue;
        }
        const auto &target = pairs[static_cast<size_t>(shoot_ids[m])].second;
        int winner = -1;
        float best = no_cost;
        for (int r = 0; r < n_rollouts; ++r) {
          const size_t s = static_cast<size_t>(m) * n_rollouts + r;
          if (!ok[s] ||
              !within_tolerance(finals.data() + s * n_dims, target->pose,
                                tolerance) ||
              meet_cost[r] >= best) {
            continue;
          }
          best = meet_cost[r];
          winner = r;
        }
        if (winner < 0) {
          continue;
        }
        const int id = shoot_ids[m];
        const float *sample =
            controls.data() +
            (static_cast<size_t>(m) * n_rollouts + winner) * n_ctrl;
        trajectories[id].assign(sample, sample + n_ctrl);
        valid[id] = true;
      }
    }
  }
};

// Environments with no dynamics rollout. solver: mpc throws from the rollout.
template <typename Env, typename NodeT>
void solve_near_meet(
    Env *env,
    const std::vector<std::pair<std::shared_ptr<NodeT>, std::shared_ptr<NodeT>>>
        &pairs,
    std::vector<bool> &valid, std::vector<std::vector<float>> &trajectories) {
  env->near_meet_.solve(
      pairs, valid, trajectories,
      [env](const std::vector<float *> &states, std::vector<bool> &results) {
        env->check_validity_batched(states, results);
      },
      [](const float *, const float *, int, int, int, int, float *,
         uint8_t *) {
        throw std::runtime_error(
            "near_meet_config.solver is mpc, and this environment has no rollout");
      },
      0, 0, nullptr);
}

// The warmstart is applied at u, not at u->parent. A rollout that misses the
// tolerance is not a meet. The held action is copied across timesteps.
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
  solver.mpc_num_rollouts = 1;
  solver.num_iters = 1;
  solver.topk = 1;
  solver.action_noise = {0.f};
  solver.cost_weights = {1.f};
  std::vector<std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>>> pairs = {
      {u, v}};
  std::vector<bool> valid;
  std::vector<std::vector<float>> trajectories;
  float seen_state = -1.f;
  bool called = false;
  auto shoot = [&](auto &&sample, int n_c, int steps, const float *tol) {
    auto batched = [&](const float *states, const float *controls, int n_shoot,
                       int n_rollouts, int n_steps, int n_ctrl, float *finals,
                       uint8_t *ok) {
      called = true;
      seen_state = states[0];
      for (int s = 0; s < n_shoot * n_rollouts; ++s) {
        const int meet = s / n_rollouts;
        ok[s] = sample(states + static_cast<size_t>(meet) * solver.n_dims,
                       controls + static_cast<size_t>(s) * n_steps * n_ctrl,
                       n_steps,
                       finals + static_cast<size_t>(s) * solver.n_dims)
                    ? 1
                    : 0;
      }
    };
    solver.mpc_solve(pairs, valid, trajectories, n_c, steps, batched, tol);
  };
  auto step = [&](const float *start, const float *controls, int steps,
                  float *final_pose) {
    if (steps > 1 && controls[0] != controls[(steps - 1) * 1]) {
      return false;
    }
    final_pose[0] = start[0] + controls[0];
    return true;
  };
  float tol[1] = {1e-3f};
  shoot(step, 1, 2, tol);
  if (!valid[0] || !called || seen_state != u->pose[0] ||
      trajectories[0].size() != 2 || trajectories[0][0] != 1.f ||
      trajectories[0][1] != 1.f) {
    return false;
  }
  v->pose[0] = 2.5f;
  tol[0] = 0.1f;
  shoot(step, 1, 2, tol);
  if (valid[0]) {
    return false;
  }
  tol[0] = 1e-3f;
  v->pose[0] = 2.f;
  auto reject = [](const float *, const float *, int, float *) { return false; };
  shoot(reject, 1, 1, tol);
  if (valid[0]) {
    return false;
  }
  auto start = std::make_shared<Node>();
  start->pose[0] = 2.f;
  pairs = {{start, v}};
  shoot(reject, 1, 1, tol);
  if (!valid[0]) {
    return false;
  }
  start->pose[0] = 2.f + 10.f * tol[0];
  shoot(reject, 1, 1, tol);
  if (valid[0]) {
    return false;
  }
  solver.angular_dims = {0};
  start->pose[0] = 0.f;
  v->pose[0] = 6.f;
  tol[0] = 0.3f;
  shoot(reject, 1, 1, tol);
  if (!valid[0]) {
    return false;
  }
  v->pose[0] = 3.f;
  shoot(reject, 1, 1, tol);
  if (valid[0]) {
    return false;
  }
  // Lower cost outside the tolerance box, higher cost inside it. topk is 1,
  // so the inside sample is not elite. It still wins because every
  // last-iteration rollout is checked. One rollout sees only the outside sample.
  solver.n_dims = 2;
  solver.angular_dims.clear();
  solver.mpc_num_rollouts = 2;
  solver.topk = 1;
  solver.cost_weights = {1.f, 100.f};
  u->parent = parent;
  u->intermediate_controls = warm;
  v->pose[0] = 0.f;
  v->pose[1] = 0.f;
  pairs = {{u, v}};
  float tol2[2] = {0.1f, 100.f};
  int which = 0;
  auto two = [&](const float *, const float *, int, float *final_pose) {
    if (which++ == 0) {
      final_pose[0] = 1.f;
      final_pose[1] = 0.f;
    } else {
      final_pose[0] = 0.f;
      final_pose[1] = 0.5f;
    }
    return true;
  };
  shoot(two, 1, 1, tol2);
  if (!valid[0]) {
    return false;
  }
  which = 0;
  solver.mpc_num_rollouts = 1;
  shoot(two, 1, 1, tol2);
  if (valid[0]) {
    return false;
  }
  solver.n_dims = 1;
  solver.cost_weights = {-1.f};
  std::vector<float> inv;
  float quarter = 0.25f;
  solver.resolve_weights(&quarter, inv);
  if (inv.size() != 1 || inv[0] != 4.f) {
    return false;
  }
  float wide = std::numeric_limits<float>::infinity();
  solver.resolve_weights(&wide, inv);
  if (inv[0] != 0.f) {
    return false;
  }
  NearMeet2PBVP bare;
  bare.n_dims = 1;
  bool threw = false;
  try {
    bare.solve(
        pairs, valid, trajectories,
        [](const std::vector<float *> &, std::vector<bool> &) {},
        [](const float *, const float *, int, int, int, int, float *,
           uint8_t *) {},
        1, 1, tol2);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  return threw;
}
