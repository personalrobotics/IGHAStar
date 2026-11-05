#pragma once

// Requires Node, NodePtrCompare, and n_dims to already be defined (include the
// active environment header before this file).

#include <memory>
#include <queue>
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
};

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

std::vector<Trajectory> import_trajectories(const py::list &items,
                                            Environment *env);

void record_solution(
    std::vector<Trajectory> &library,
    const std::vector<std::shared_ptr<Node>> &path,
    const igha::unordered_set<std::shared_ptr<Node>> &Seen,
    std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>,
                        NodePtrCompare>
        Q_v,
    const igha::unordered_set<std::shared_ptr<Node>> &inactive_Q_v);

py::list export_library(const std::vector<Trajectory> &library, int timesteps);

bool library_links_ok(const std::vector<Trajectory> &library);

} // namespace tree_utils
