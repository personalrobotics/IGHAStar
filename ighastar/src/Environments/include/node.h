#pragma once

// One Node for every environment. Include this after the environment has
// defined n_dims, n_cont, hash_dims, and calc_hash(float *, const float *).
// NODE_NFLAGS is 0 except pusht_multi, which sets it to the tee count so
// Node::cleared exists only there.
//
// Backward search (time_direction == -1) reverses intermediate poses and the
// substep times stored with them. Push-T searches are forward, so that branch
// does not run for them.
//
// intermediate_times_ is generator order: index 0 is the first step from the
// parent, index timesteps-1 is this node's arrival. Pass nullptr when the
// edge has no clock. time then stays equal to g.

#include "lazy_index.h"

#include <cstring>
#include <memory>
#include <vector>

// Clock of each generated substep. dt is the integration step (already signed
// for a backward car search). out[t] is the time after step t + 1.
inline void fill_substep_times(float *out, float start_time, float dt, int steps) {
  for (int t = 0; t < steps; ++t) {
    out[t] = start_time + static_cast<float>(t + 1) * dt;
  }
}

#ifndef NODE_NFLAGS
#define NODE_NFLAGS 0
#endif

template <int NFlags>
struct NodeCleared {
  bool cleared[NFlags] = {};
};

template <>
struct NodeCleared<0> {};

template <int NFlags>
struct NodeT : NodeCleared<NFlags> {
  float pose[n_dims];
  float *intermediate_poses;
  float *intermediate_times; // one clock value per intermediate pose
  float g, f;
  float time; // arrival clock. g is the cost and may differ.
  bool active;
  int rank, level;
  size_t hash;
  size_t LCR_index;
  int time_direction; // 1 for forward, -1 for backward
  LazyIndex<hash_dims, calc_hash> index;
  std::shared_ptr<NodeT> parent;
  bool preexpanded = false;
  std::vector<std::shared_ptr<NodeT>> cached_succ;
  int discover_index = -1;
  int expand_index = -1;
  float control[n_cont] = {};
  float *intermediate_controls = nullptr;

  // time_is_set: use time_in instead of the last generated sample. The
  // trajectory library passes an already-stored buffer with time_direction +1,
  // so that last index is not the arrival on a backward edge.
  NodeT(const float *pose_in, const float *intermediate_poses_,
        std::shared_ptr<NodeT> parent_in, float g_in, const float *resolution_,
        const float *tolerance, float division_factor, int timesteps,
        const float *LCR_, int time_direction_ = 1,
        const float *intermediate_controls_ = nullptr,
        const float *intermediate_times_ = nullptr, float time_in = 0.f,
        bool time_is_set = false)
      : intermediate_poses(nullptr), intermediate_times(nullptr),
        intermediate_controls(nullptr), g(g_in), time(g_in), f(0),
        parent(parent_in), active(true), rank(0), level(0),
        time_direction(time_direction_) {
    for (int i = 0; i < n_dims; i++) {
      pose[i] = pose_in[i];
    }
    if (parent_in != nullptr && intermediate_poses_ != nullptr) {
      intermediate_poses = new float[timesteps * n_dims];
      if (time_direction_ == -1) {
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
    if (parent_in != nullptr && intermediate_times_ != nullptr) {
      intermediate_times = new float[timesteps];
      if (time_direction_ == -1) {
        for (int t = 0; t < timesteps; t++) {
          intermediate_times[t] = intermediate_times_[timesteps - 1 - t];
        }
      } else {
        memcpy(intermediate_times, intermediate_times_,
               timesteps * sizeof(float));
      }
      time = time_is_set ? time_in : intermediate_times_[timesteps - 1];
    } else if (time_is_set) {
      time = time_in;
    }
    if (parent_in != nullptr && intermediate_controls_ != nullptr) {
      intermediate_controls = new float[timesteps * n_cont];
      memcpy(intermediate_controls, intermediate_controls_,
             timesteps * n_cont * sizeof(float));
      memcpy(control, intermediate_controls_, n_cont * sizeof(float));
    }
    LCR_index = calc_hash(pose, LCR_);
    hash = calc_hash(pose, tolerance);
    index.init(pose, resolution_, division_factor);
  }

  ~NodeT() {
    delete[] intermediate_poses;
    delete[] intermediate_times;
    delete[] intermediate_controls;
  }
};

using Node = NodeT<NODE_NFLAGS>;

struct NodePtrCompare {
  bool operator()(const std::shared_ptr<Node> &a,
                  const std::shared_ptr<Node> &b) const {
    return a->f > b->f; // min-heap: smaller f has higher priority
  }
};

#undef NODE_NFLAGS
