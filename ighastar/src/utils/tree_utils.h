#pragma once

// Requires Node, NodePtrCompare, and n_dims to already be defined (include the
// active environment header before this file).

#include <cstdint>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

class Environment;

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils/hash_map.h"

namespace py = pybind11;

namespace tree_utils {

// Collect nodes from Seen, active Q_v, and inactive_Q_v into a Python dict.
// Q_v is taken by value so the caller's queue is not modified.
// Uses igha::unordered_set so this matches the planner containers whether the
// backend is std or robin_hood (-DIGHA_USE_ROBIN_HOOD).
py::dict export_search_tree(
    const igha::unordered_set<std::shared_ptr<Node>> &Seen,
    std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>,
                        NodePtrCompare>
        Q_v,
    const igha::unordered_set<std::shared_ptr<Node>> &inactive_Q_v, int timesteps,
    int expansion_count);

// Solution path plus off-path children of those vertices. Parent pointers are
// the links. The next path vertex is not copied into children.
struct Trajectory {
  std::vector<std::shared_ptr<Node>> path;
  std::vector<std::shared_ptr<Node>> children;
  // Problem that produced this trajectory. goal is the query goal. world is
  // the environment map, empty when the environment does not stamp one.
  std::vector<float> goal;
  std::vector<float> world;
};

// Trajectories plus a used-flag and an LCR index of path vertices. The index is
// built at import and when a solution is recorded. reset clears only the flags.
// applicable is the current query's filter (1 = eligible). Empty means unfiltered.
struct Library {
  std::vector<Trajectory> trajs;
  std::vector<uint8_t> used;
  std::vector<uint8_t> applicable;
  igha::unordered_map<size_t, std::vector<std::pair<int, int>>> lcr;
};

void reset_library(Library &library);

Trajectory capture_trajectory(
    const std::vector<std::shared_ptr<Node>> &path,
    const igha::unordered_set<std::shared_ptr<Node>> &Seen,
    std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>,
                        NodePtrCompare>
        Q_v,
    const igha::unordered_set<std::shared_ptr<Node>> &inactive_Q_v);

// Packed arrays, one buffer per field. Per-vertex Python lists are the cost
// that dominates load and export.
py::dict export_trajectory(const Trajectory &traj, int timesteps);

Library import_trajectories(const py::list &items, Environment *env);

void record_solution(
    Library &library,
    const std::vector<std::shared_ptr<Node>> &path,
    const igha::unordered_set<std::shared_ptr<Node>> &Seen,
    std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>,
                        NodePtrCompare>
        Q_v,
    const igha::unordered_set<std::shared_ptr<Node>> &inactive_Q_v,
    const float *goal_pose, Environment *env);

// Ask the environment which stored trajectories belong to this start, goal,
// and the map currently loaded on env. Writes library.applicable.
void select_trajectories(Library &library, Environment *env, const float *start,
                         const float *goal);

py::list export_library(const Library &library, int timesteps);

bool library_links_ok(const Library &library);

// Copies of the goal-side suffix of each unused trajectory that near-meets u.
// One LCR lookup, then one solve_2pBVP. Library nodes are not modified.
std::vector<std::shared_ptr<Node>>
attach_nearmeets(Library &library, const std::shared_ptr<Node> &u,
                 const std::shared_ptr<Node> &goal, Environment *env);

} // namespace tree_utils
