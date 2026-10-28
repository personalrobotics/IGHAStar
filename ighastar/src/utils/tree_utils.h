#pragma once

// Requires Node, NodePtrCompare, and n_dims to already be defined (include the
// active environment header before this file).

#include <memory>
#include <queue>
#include <vector>

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

} // namespace tree_utils
