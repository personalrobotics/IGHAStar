// Implementation of tree_utils::export_search_tree.
// Included from ighastar.cpp (same TU as the environment Node type) so we do
// not re-include environment headers or duplicate symbols like calc_hash.
// Do not compile this file as a separate translation unit.

namespace tree_utils {

namespace {

py::list pose_to_list(const float *pose) {
  py::list out;
  for (int d = 0; d < n_dims; ++d) {
    out.append(pose[d]);
  }
  return out;
}

py::object intermediate_poses_to_list(const float *intermediate_poses,
                                      int timesteps) {
  if (intermediate_poses == nullptr || timesteps <= 0) {
    return py::none();
  }
  py::list frames;
  for (int t = 0; t < timesteps; ++t) {
    frames.append(pose_to_list(intermediate_poses + t * n_dims));
  }
  return frames;
}

py::dict node_to_dict(const std::shared_ptr<Node> &node, int timesteps) {
  py::dict d;
  d["hash"] = py::int_(node->hash);
  if (node->parent) {
    d["parent_hash"] = py::int_(node->parent->hash);
  } else {
    d["parent_hash"] = py::none();
  }
  d["pose"] = pose_to_list(node->pose);
  d["g"] = node->g;
  d["f"] = node->f;
  d["discover_index"] = node->discover_index;
  d["expand_index"] = node->expand_index;
  d["time_direction"] = node->time_direction;
  d["intermediate_poses"] =
      intermediate_poses_to_list(node->intermediate_poses, timesteps);
  return d;
}

} // namespace

py::dict export_search_tree(
    const igha::unordered_set<std::shared_ptr<Node>> &Seen,
    std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>,
                        NodePtrCompare>
        Q_v,
    const igha::unordered_set<std::shared_ptr<Node>> &inactive_Q_v, int timesteps,
    int expansion_count) {
  igha::unordered_map<size_t, std::shared_ptr<Node>> unique;
  auto consider = [&](const std::shared_ptr<Node> &node) {
    if (node && unique.find(node->hash) == unique.end()) {
      unique.emplace(node->hash, node);
    }
  };

  for (const auto &node : Seen) {
    consider(node);
  }
  while (!Q_v.empty()) {
    consider(Q_v.top());
    Q_v.pop();
  }
  for (const auto &node : inactive_Q_v) {
    consider(node);
  }

  py::list nodes;
  for (const auto &kv : unique) {
    nodes.append(node_to_dict(kv.second, timesteps));
  }

  py::dict out;
  out["n_dims"] = n_dims;
  out["timesteps"] = timesteps;
  out["expansion_count"] = expansion_count;
  out["nodes"] = nodes;
  return out;
}

} // namespace tree_utils
