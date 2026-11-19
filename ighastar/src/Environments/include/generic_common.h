// Shared pieces of the generic (OMPL-style) environments.
//
// Both `generic.h` (Python callbacks via py::function) and `generic_torch.h`
// (AOTInductor-compiled torch callbacks, no Python during search) use the same
// Node layout, hashing, config parsing, successor construction, and path
// conversion. Only the way the user's callbacks are evaluated differs; that
// lives in each header's `Environment` class, which derives from
// `GenericEnvironmentBase` below.
//
// State / control dimensionality is fixed at (JIT) compile time via the
// N_DIMS / N_CONT / HASH_DIMS macros, mirroring how the other environments
// hard-code these as compile-time constants. This keeps Node and the planning
// core unchanged.
#pragma once

// A generic rollout length is learned at expansion time. Import adopts the
// length stored with the library instead of the config hint.
#define IGHASTAR_LIBRARY_TIMESTEPS_FROM_EXPORT

#include "solve_2pbvp.h"

#include <boost/functional/hash.hpp>
#include <cmath>
#include <cstring>
#include <memory>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>
#include "lazy_index.h"

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

inline size_t calc_hash(float *pose, const float *resolution) {
  std::size_t hash_val = 0;
  for (int i = 0; i < hash_dims; ++i) {
    boost::hash_combine(hash_val, std::hash<size_t>{}(static_cast<int>(
                                      std::round(pose[i] / resolution[i]))));
  }
  return hash_val;
}

#include "node.h"
#include "trajectory_export.h"

// Callback-independent part of the generic environments: config parsing,
// node creation, batch packing, successor construction and path conversion.
class GenericEnvironmentBase {
protected:
  int n_succ; // branching factor K (controls applied per expansion)

public:
  // Fields referenced by the planning core / bidirectional machinery.
  int timesteps;
  int time_direction;
  float resolution[n_dims];
  float tolerance[n_dims];
  float local_controllability_radius[n_dims];
  NearMeet2PBVP near_meet_;
#define IGHASTAR_MEET_ANGULAR_DIMS(env) ((env)->near_meet_.angular_dims)
  sampling::GoalSamplingConfig goal_sampling;
  float lower[n_dims];
  float upper[n_dims];
  float division_factor;

  GenericEnvironmentBase(const py::dict &config, int time_direction_ = 1)
      : time_direction(time_direction_) {
    auto info = config["experiment_info_default"].cast<py::dict>();

    division_factor = info["division_factor"].cast<float>();
    n_succ = info.contains("num_controls") ? info["num_controls"].cast<int>()
                                           : 1;

    auto require_len = [](const std::vector<float> &v, const char *name) {
      if (v.size() != static_cast<size_t>(n_dims)) {
        throw std::runtime_error(
            std::string("generic: '") + name + "' must have length n_dims=" +
            std::to_string(n_dims) + ", got " + std::to_string(v.size()));
      }
    };

    auto res = info["resolution"].cast<std::vector<float>>();
    auto tol = info["tolerance"].cast<std::vector<float>>();
    require_len(res, "resolution");
    require_len(tol, "tolerance");
    for (int i = 0; i < n_dims; i++) {
      resolution[i] = res[i];
      tolerance[i] = tol[i];
    }

    // Optional state-space bounds: reject children outside [lower, upper] in
    // make_successors (in addition to validity_fn).
    if (info.contains("bounds_lower") && info.contains("bounds_upper")) {
      auto lo = info["bounds_lower"].cast<std::vector<float>>();
      auto hi = info["bounds_upper"].cast<std::vector<float>>();
      require_len(lo, "bounds_lower");
      require_len(hi, "bounds_upper");
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
      require_len(lcr, "LCR");
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
    goal_sampling = sampling::load_goal_sampling(
        config, n_dims, local_controllability_radius);
    near_meet_.load(config, n_dims, local_controllability_radius,
                    goal_sampling.angular_dims);
  }

  virtual void check_validity_batched(const std::vector<float *> &states,
                                      std::vector<bool> &results) = 0;

  virtual float heuristic(float *pose, float *goal) = 0;

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

  virtual ~GenericEnvironmentBase() {}

  // Creates a new Node with the given pose (start / goal nodes).
  std::shared_ptr<Node> create_Node(float *pose) {
    return std::make_shared<Node>(pose, nullptr, nullptr, 0, resolution,
                                  tolerance, division_factor,
                                  timesteps, local_controllability_radius,
                                  time_direction);
  }

  // The generic envs do not own a world tensor; the world (e.g. an MJX model
  // or an obstacle map) is captured by the user's callbacks.
  void set_world(torch::Tensor world) { (void)world; }

  // Distance metric used by bidirectional near-meet checks (Euclidean stub).
  float compute_near_meet_distance(float *pose1, float *pose2) {
    float s = 0.0f;
    for (int i = 0; i < n_dims; i++) {
      float d = pose1[i] - pose2[i];
      s += d * d;
    }
    return std::sqrt(s);
  }

  IGHASTAR_FILTER_TRAJECTORIES_ALL

  // Generic envs do not need device-side batch buffers.
  void ensure_batch_capacity(int max_batch) { (void)max_batch; }

  // Pack M parent states x K controls into row-major [B = M*K] CPU tensors:
  // row i*K + k is (nodes[i]->pose, controls[k]).
  static void pack_batch(const std::vector<std::shared_ptr<Node>> &nodes,
                         const float *cp, int K, torch::Tensor &states_t,
                         torch::Tensor &controls_batch_t) {
    int M = static_cast<int>(nodes.size());
    int B = M * K;
    states_t = torch::empty({B, n_dims}, torch::kFloat32);
    controls_batch_t = torch::empty({B, n_cont}, torch::kFloat32);
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
  }

  // Build successor Nodes from the evaluated batch. All tensors are contiguous
  // CPU float32: next_t is [B, n_dims] or [B, rollout_T, n_dims], final_t is
  // [B, n_dims], cost/valid/h are [B]. Returns one neighbor list per parent.
  std::vector<std::vector<std::shared_ptr<Node>>>
  make_successors(const std::vector<std::shared_ptr<Node>> &nodes,
                  const float *cp, int K, const torch::Tensor &next_t,
                  const torch::Tensor &final_t, const torch::Tensor &cost_t,
                  const torch::Tensor &valid_t, const torch::Tensor &h_t) {
    int M = static_cast<int>(nodes.size());
    const int B = M * K;
    std::vector<std::vector<std::shared_ptr<Node>>> results(M);
    int rollout_T = 1;
    if (next_t.dim() == 3) {
      if (next_t.size(0) != B || next_t.size(2) != n_dims) {
        throw std::runtime_error(
            "generic: dynamics/expand next_states must be [B, T, n_dims] "
            "(B=M*K, last dim = n_dims)");
      }
      rollout_T = static_cast<int>(next_t.size(1));
    } else if (next_t.dim() == 2) {
      if (next_t.size(0) != B || next_t.size(1) != n_dims) {
        throw std::runtime_error(
            "generic: dynamics/expand next_states must be [B, n_dims] "
            "(B=M*K)");
      }
    } else {
      throw std::runtime_error(
          "generic: dynamics/expand next_states must be rank 2 or 3");
    }
    if (final_t.dim() != 2 || final_t.size(0) != B || final_t.size(1) != n_dims) {
      throw std::runtime_error(
          "generic: final states must be [B, n_dims] with B=M*K");
    }
    if (cost_t.numel() < B || valid_t.numel() < B || h_t.numel() < B) {
      throw std::runtime_error(
          "generic: cost/valid/heuristic must each have at least B=" +
          std::to_string(B) + " elements");
    }
    if (!next_t.is_contiguous() || !final_t.is_contiguous() ||
        !cost_t.is_contiguous() || !valid_t.is_contiguous() ||
        !h_t.is_contiguous()) {
      throw std::runtime_error(
          "generic: cost/valid/heuristic/next/final tensors must be contiguous");
    }
    // Keep the env's timesteps in sync so the path expansion matches.
    timesteps = rollout_T;

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
        bool in_bounds = true;
        for (int d = 0; d < n_dims; d++) {
          if (new_pose[d] < lower[d] || new_pose[d] > upper[d]) {
            in_bounds = false;
            break;
          }
        }
        if (!in_bounds) {
          continue;
        }
        // Pointer to this row's rollout (rollout_T * n_dims floats); for the
        // 2D case this is just the single final state.
        const float *roll =
            rollp + static_cast<size_t>(row) * rollout_T * n_dims;
        float *controls_seq = new float[rollout_T * n_cont];
        for (int t = 0; t < rollout_T; t++) {
          memcpy(controls_seq + t * n_cont, cp + k * n_cont,
                 n_cont * sizeof(float));
        }
        auto neighbor = std::make_shared<Node>(
            new_pose, roll, nodes[i], nodes[i]->g + costp[row], resolution,
            tolerance, division_factor, rollout_T,
            local_controllability_radius, time_direction, controls_seq);
        delete[] controls_seq;
        neighbor->f = neighbor->g + hp[row];
        results[i].push_back(neighbor);
      }
    }
    return results;
  }

  torch::Tensor convert_node_list_to_path_tensor(
      std::vector<std::shared_ptr<Node>> node_list) {
    return path_tensor_from_edges(node_list, timesteps);
  }

  torch::Tensor convert_node_list_to_controls_tensor(
      std::vector<std::shared_ptr<Node>> node_list) {
    return controls_tensor_from_edges(node_list, timesteps);
  }
};
