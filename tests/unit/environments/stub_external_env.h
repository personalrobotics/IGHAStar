// Minimal Environment for out-of-tree IGHASTAR_ENV_HEADER smoke tests. Lives
// under tests/: proves ighastar.cpp can include a header outside Environments/
// and still construct IGHAStar.
#pragma once

#include <generic_common.h>

#include <cmath>
#include <memory>
#include <vector>

class Environment : public GenericEnvironmentBase {
public:
  Environment(const py::dict &config, int time_direction_ = 1)
      : GenericEnvironmentBase(config, time_direction_) {}

  float heuristic(float *pose, float *goal) {
    float s = 0.0f;
    for (int i = 0; i < n_dims; i++) {
      float d = pose[i] - goal[i];
      s += d * d;
    }
    return std::sqrt(s);
  }

  bool reached_goal_region(std::shared_ptr<Node> v,
                           std::shared_ptr<Node> goal) {
    return heuristic(v->pose, goal->pose) < 0.5f;
  }

  void check_validity(float *start, float *goal, bool *result) {
    (void)start;
    (void)goal;
    result[0] = true;
    result[1] = true;
  }

  void check_validity_batched(const std::vector<float *> &states,
                              std::vector<bool> &results) {
    results.assign(states.size(), true);
  }

  // Empty successors: construction / validity are enough for the smoke test.
  std::vector<std::vector<std::shared_ptr<Node>>>
  Succ_batched(const std::vector<std::shared_ptr<Node>> &nodes,
               std::shared_ptr<Node> goal) {
    (void)goal;
    return std::vector<std::vector<std::shared_ptr<Node>>>(nodes.size());
  }

  std::vector<std::shared_ptr<Node>> Succ(std::shared_ptr<Node> node,
                                          std::shared_ptr<Node> goal) {
    auto results = Succ_batched({node}, goal);
    return results.empty() ? std::vector<std::shared_ptr<Node>>{} : results[0];
  }
};
