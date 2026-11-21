#pragma once

// Requires Node, NodePtrCompare, and n_dims to already be defined (include the
// active environment header before this file).

#include <algorithm>
#include <cstdint>
#include <memory>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

class Environment;

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils/hash_map.h"

namespace py = pybind11;

namespace tree_utils {

using OpenQueue =
    std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>,
                        NodePtrCompare>;

// Read the priority_queue container without popping it.
template <class PriorityQueue>
const typename PriorityQueue::container_type &
heap_container(const PriorityQueue &queue) {
  struct Access : PriorityQueue {
    static const typename PriorityQueue::container_type &
    get(const PriorityQueue &q) {
      return q.*&Access::c;
    }
  };
  return Access::get(queue);
}

// Flat per-node arrays. Vertex i has state poses[i]. parents[i] is the parent
// index, or -1 for the root. edges[i] is the edge from the parent.
struct SearchTree {
  enum Status : int8_t {
    Expanded = 0,
    Solution = 1,
    Open = 2,
    Inactive = 3,
    Pruned = 4,
  };
  int64_t num_nodes = 0;
  int state_dim = n_dims, control_dim = n_cont, timesteps = 0;
  std::vector<float> poses, g, f, time;
  std::vector<int64_t> parents, hash;
  std::vector<int32_t> levels, time_direction, discover_index, expand_index;
  std::vector<int8_t> status;
  std::vector<uint8_t> has_edge, has_times;
  std::vector<float> edges, edge_controls, intermediate_times;
  float best_cost = 1e5f;
  int64_t best_index = -1;
  int expansions = 0;
  bool budget_exhausted = false;

  std::vector<int64_t> path_to(int64_t index) const {
    if (index < 0 || index >= num_nodes) {
      throw std::out_of_range("node index out of range");
    }
    std::vector<int64_t> path;
    for (int64_t i = index; i >= 0; i = parents[i]) {
      path.push_back(i);
    }
    std::reverse(path.begin(), path.end());
    return path;
  }
};

// Seen, Q_v, inactive_Q_v, and pruned. Q_v is not modified.
std::shared_ptr<SearchTree> export_search_tree(
    const igha::unordered_set<std::shared_ptr<Node>> &Seen,
    const OpenQueue &Q_v,
    const igha::unordered_set<std::shared_ptr<Node>> &inactive_Q_v,
    const std::vector<std::shared_ptr<Node>> &pruned,
    const std::vector<std::vector<std::shared_ptr<Node>>> &best_path_list,
    const std::vector<std::shared_ptr<Node>> &best_path, float best_cost,
    int timesteps, int expansion_counter, int expansion_limit, bool success);

void bind_search_tree(py::module_ &m);

// Solution path plus off-path children of those vertices. Parent pointers are
// the links. The next path vertex is not copied into children.
struct Trajectory {
  std::vector<std::shared_ptr<Node>> path;
  std::vector<std::shared_ptr<Node>> children;
  // Problem that produced this trajectory. goal is the query goal. world is
  // the environment map, empty when the environment does not stamp one.
  std::vector<float> goal;
  std::vector<float> world;
  // Length of the stored edge buffers. 0 means the env's current timesteps.
  int steps = 0;
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
