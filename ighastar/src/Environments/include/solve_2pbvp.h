#pragma once

// Default 2-point BVP for built-in environments. Environment::solve_2pBVP calls
// Environment::mpc_solve. The default mpc_solve is geometric_solve; override
// mpc_solve on the Environment for a dynamics-aware solver.
// shortcut: checks one pair at a time. Pack every pair's samples into one
// check_validity_batched if this shows up hot.

#include "../../utils/config_utils.h"
#include "../../utils/sampling_utils.h"

#include <memory>
#include <utility>
#include <vector>

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
};
