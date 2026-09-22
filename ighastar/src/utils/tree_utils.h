#pragma once

// Requires Node, NodePtrCompare, and n_dims to already be defined (include the
// active environment header before this file).

#include <memory>
#include <queue>
#include <unordered_set>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace tree_utils {

// Collect nodes from Seen, active Q_v, and inactive_Q_v into a Python dict.
// Q_v is taken by value so the caller's queue is not modified.
py::dict export_search_tree(
    const std::unordered_set<std::shared_ptr<Node>> &Seen,
    std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>,
                        NodePtrCompare>
        Q_v,
    const std::unordered_set<std::shared_ptr<Node>> &inactive_Q_v, int timesteps,
    int expansion_count);

} // namespace tree_utils
