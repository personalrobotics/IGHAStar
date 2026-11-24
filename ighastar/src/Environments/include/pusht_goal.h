#pragma once

// Push-T goal test and heuristic, shared by the planner and the tree labeler.
// Yaw is not part of the distance: the planner's block_distance leaves that term commented out.
// shortcut: a cell is kept by pusht_optimal (Ethan's test). Wire pusht_subopt in when a factor other than 1 is wanted.

#include <cmath>

#ifdef __CUDACC__
#define PUSHT_HD __host__ __device__
#else
#define PUSHT_HD
#endif

namespace pusht_goal {

constexpr float kCostTolerance = 1e-5f;

PUSHT_HD inline float wrap_pi(float dth) {
  const float pi = 3.14159265358979323846f;
  while (dth > pi) dth -= 2.f * pi;
  while (dth < -pi) dth += 2.f * pi;
  return dth;
}

PUSHT_HD inline float block_distance(const float *pose, const float *goal) {
  float dx = pose[2] - goal[2];
  float dy = pose[3] - goal[3];
  return sqrtf(dx * dx + dy * dy);
}

PUSHT_HD inline float heuristic(const float *pose, const float *goal, float max_agent_vel) {
  return max_agent_vel > 0.f ? block_distance(pose, goal) / max_agent_vel : 0.f;
}

// Block x, y, yaw inside epsilon[2..4], and the block at rest under epsilon[7..9].
PUSHT_HD inline bool reached_goal(const float *pose, const float *goal, const float *epsilon) {
  float dx = pose[2] - goal[2];
  float dy = pose[3] - goal[3];
  float dth = wrap_pi(pose[4] - goal[4]);
  return fabsf(dx) < epsilon[2] && fabsf(dy) < epsilon[3] && fabsf(dth) < epsilon[4] &&
         fabsf(pose[7]) < epsilon[7] && fabsf(pose[8]) < epsilon[8] && fabsf(pose[9]) < epsilon[9];
}

// True when the cell is covered and no unexpanded node has g + h below best_cost.
PUSHT_HD inline bool optimal(float best_cost, float min_f) {
  if (!(best_cost < 1e30f)) return false;
  float bound = best_cost * (1.f - kCostTolerance);
  return !(static_cast<double>(min_f) < static_cast<double>(bound));
}

PUSHT_HD inline float subopt(float best_cost, float min_f, float min_edge_cost) {
  float denom = min_f + min_edge_cost;
  return denom > 0.f ? best_cost / denom : INFINITY;
}

}  // namespace pusht_goal
