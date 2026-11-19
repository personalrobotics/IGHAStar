// Implementation of tree_utils::export_search_tree and the trajectory library.
// Included from ighastar.cpp (same TU as the environment Node type) so we do
// not re-include environment headers or duplicate symbols like calc_hash.
// Do not compile this file as a separate translation unit.

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include <pybind11/numpy.h>

#include "utils/sampling_utils.h"

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
  d["time"] = node->time;
  d["f"] = node->f;
  d["discover_index"] = node->discover_index;
  d["expand_index"] = node->expand_index;
  d["time_direction"] = node->time_direction;
  d["intermediate_poses"] =
      intermediate_poses_to_list(node->intermediate_poses, timesteps);
  // Stored controls stay in generator order. Pose rows are reversed on a
  // backward edge, so the control row is flipped to match intermediate_poses[t].
  if (node->intermediate_controls == nullptr || timesteps <= 0 || n_cont <= 0) {
    d["intermediate_controls"] = py::none();
  } else {
    const bool backward = node->time_direction < 0;
    py::list controls;
    for (int t = 0; t < timesteps; ++t) {
      const int src = backward ? timesteps - 1 - t : t;
      py::list sample;
      for (int c = 0; c < n_cont; ++c) {
        sample.append(node->intermediate_controls[src * n_cont + c]);
      }
      controls.append(sample);
    }
    d["intermediate_controls"] = controls;
  }
  if (node->intermediate_times == nullptr || timesteps <= 0) {
    d["intermediate_times"] = py::none();
  } else {
    py::list times;
    for (int t = 0; t < timesteps; ++t) {
      times.append(node->intermediate_times[t]);
    }
    d["intermediate_times"] = times;
  }
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

namespace {

using FArr = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IArr = py::array_t<int, py::array::c_style | py::array::forcecast>;
using U8Arr = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

template <typename T>
py::array_t<T> copy_array(const std::vector<T> &src,
                          std::vector<py::ssize_t> shape) {
  py::ssize_t n = 1;
  for (py::ssize_t s : shape) {
    n *= s;
  }
  if (n != static_cast<py::ssize_t>(src.size())) {
    throw std::invalid_argument("trajectory array shape does not match its data");
  }
  py::array_t<T> out(shape);
  if (n > 0) {
    std::memcpy(out.mutable_data(), src.data(), sizeof(T) * src.size());
  }
  return out;
}

void append_vertex(const std::shared_ptr<Node> &node, int parent_index,
                   int timesteps, std::vector<float> &pose, std::vector<float> &g,
                   std::vector<float> &f, std::vector<int> &time_direction,
                   std::vector<int> &parent, std::vector<float> &inter,
                   std::vector<uint8_t> &has_inter, std::vector<float> &controls,
                   std::vector<uint8_t> &has_controls, std::vector<float> &clock,
                   std::vector<float> &inter_times,
                   std::vector<uint8_t> &has_times) {
  pose.insert(pose.end(), node->pose, node->pose + n_dims);
  g.push_back(node->g);
  clock.push_back(node->time);
  f.push_back(node->f);
  time_direction.push_back(node->time_direction);
  parent.push_back(parent_index);
  const int inter_n = timesteps > 0 ? timesteps * n_dims : 0;
  if (node->intermediate_poses != nullptr && inter_n > 0) {
    has_inter.push_back(1);
    inter.insert(inter.end(), node->intermediate_poses,
                 node->intermediate_poses + inter_n);
  } else {
    has_inter.push_back(0);
    inter.insert(inter.end(), inter_n, 0.f);
  }
  const int ctrl_n = timesteps > 0 ? timesteps * n_cont : 0;
  if (node->intermediate_controls != nullptr && ctrl_n > 0) {
    has_controls.push_back(1);
    controls.insert(controls.end(), node->intermediate_controls,
                    node->intermediate_controls + ctrl_n);
  } else {
    has_controls.push_back(0);
    controls.insert(controls.end(), ctrl_n, 0.f);
  }
  if (node->intermediate_times != nullptr && timesteps > 0) {
    has_times.push_back(1);
    inter_times.insert(inter_times.end(), node->intermediate_times,
                       node->intermediate_times + timesteps);
  } else {
    has_times.push_back(0);
    inter_times.insert(inter_times.end(), timesteps > 0 ? timesteps : 0, 0.f);
  }
}

void pack_side(py::dict &out, const char *prefix,
               const std::vector<std::shared_ptr<Node>> &nodes,
               const igha::unordered_map<const Node *, int> &path_index,
               int timesteps, bool require_parent) {
  const std::string p(prefix);
  const int n = static_cast<int>(nodes.size());
  std::vector<float> pose, g, f, inter, controls, clock, inter_times;
  std::vector<int> time_direction, parent;
  std::vector<uint8_t> has_inter, has_controls, has_times;
  pose.reserve(static_cast<size_t>(n) * n_dims);
  const int inter_n = timesteps > 0 ? timesteps * n_dims : 0;
  const int ctrl_n = timesteps > 0 ? timesteps * n_cont : 0;
  inter.reserve(static_cast<size_t>(n) * inter_n);
  controls.reserve(static_cast<size_t>(n) * ctrl_n);
  for (const auto &node : nodes) {
    int parent_index = -1;
    if (node->parent) {
      auto it = path_index.find(node->parent.get());
      if (it != path_index.end()) {
        parent_index = it->second;
      }
    }
    if (require_parent && parent_index < 0) {
      throw std::invalid_argument("child parent is not on the trajectory");
    }
    append_vertex(node, parent_index, timesteps, pose, g, f, time_direction,
                  parent, inter, has_inter, controls, has_controls, clock,
                  inter_times, has_times);
  }
  out[py::str(p + "_pose")] = copy_array<float>(pose, {n, n_dims});
  out[py::str(p + "_g")] = copy_array<float>(g, {n});
  out[py::str(p + "_time")] = copy_array<float>(clock, {n});
  out[py::str(p + "_f")] = copy_array<float>(f, {n});
  out[py::str(p + "_time_direction")] = copy_array<int>(time_direction, {n});
  out[py::str(p + "_parent_index")] = copy_array<int>(parent, {n});
  out[py::str(p + "_has_intermediates")] = copy_array<uint8_t>(has_inter, {n});
  out[py::str(p + "_has_controls")] = copy_array<uint8_t>(has_controls, {n});
  out[py::str(p + "_has_times")] = copy_array<uint8_t>(has_times, {n});
  if (timesteps > 0) {
    out[py::str(p + "_intermediate_times")] =
        copy_array<float>(inter_times, {n, timesteps});
  }
  if (inter_n > 0) {
    out[py::str(p + "_intermediate_poses")] =
        copy_array<float>(inter, {n, timesteps, n_dims});
  }
  if (ctrl_n > 0) {
    out[py::str(p + "_controls")] = copy_array<float>(controls, {n, ctrl_n});
  }
}

FArr field_f(const py::dict &d, const std::string &key) { return d[key.c_str()].cast<FArr>(); }

IArr field_i(const py::dict &d, const std::string &key) { return d[key.c_str()].cast<IArr>(); }

U8Arr field_u8(const py::dict &d, const std::string &key) {
  return d[key.c_str()].cast<U8Arr>();
}

std::shared_ptr<Node> rebuild_vertex(const float *pose, float g, float f,
                                     int time_direction, const float *inter,
                                     const float *controls,
                                     const std::shared_ptr<Node> &parent,
                                     Environment *env, int steps,
                                     const float *times, float time_in,
                                     bool time_is_set) {
  // Pass time_direction +1 so the constructor copies the stored buffers.
  // The saved time_direction is written after, or a backward node would be
  // reversed a second time. time_is_set keeps the stored arrival: the buffer
  // is already in stored order, so its last sample is not the arrival on a
  // backward edge.
  auto node = std::make_shared<Node>(
      pose, inter, parent, g, env->resolution, env->tolerance,
      env->division_factor, steps, env->local_controllability_radius,
      /*time_direction_=*/1, controls, times, time_in, time_is_set);
  node->time_direction = time_direction;
  node->f = f;
#ifdef IGHASTAR_RESTORE_NODE
  IGHASTAR_RESTORE_NODE(env, *node);
#endif
  return node;
}

MeetEdge geometric_meet_edge(Environment *env, const std::shared_ptr<Node> &source,
                            const std::shared_ptr<Node> &target) {
  MeetEdge edge;
  if (!env || !source || !target) {
    return edge;
  }
  // A geometric shortcut has no duration. g still pays the near-meet distance.
  // The clock stays on the source so the suffix can be rebased onto this query.
  edge.ok = true;
  edge.cost = env->compute_near_meet_distance(source->pose, target->pose);
  edge.arrival = source->time;
  const int steps = env->timesteps;
  if (steps <= 0) {
    return edge;
  }
  edge.poses.resize(static_cast<size_t>(steps) * n_dims);
  edge.times.assign(steps, source->time);
#ifdef IGHASTAR_MEET_ANGULAR_DIMS
  const std::vector<int> &angular = IGHASTAR_MEET_ANGULAR_DIMS(env);
#else
  static const std::vector<int> angular;
  (void)env;
#endif
  for (int t = 0; t < steps; ++t) {
    const float alpha =
        static_cast<float>(t + 1) / static_cast<float>(steps);
    sampling::interpolate_pose(source->pose, target->pose, alpha, n_dims,
                               angular,
                               edge.poses.data() + static_cast<size_t>(t) * n_dims);
  }
  return edge;
}

} // namespace

Trajectory capture_trajectory(
    const std::vector<std::shared_ptr<Node>> &path,
    const igha::unordered_set<std::shared_ptr<Node>> &Seen,
    std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>,
                        NodePtrCompare>
        Q_v,
    const igha::unordered_set<std::shared_ptr<Node>> &inactive_Q_v) {
  Trajectory traj;
  traj.path = path;
  igha::unordered_set<std::shared_ptr<Node>> on_path(path.begin(), path.end());
  igha::unordered_set<std::shared_ptr<Node>> seen_child;
  auto consider = [&](const std::shared_ptr<Node> &node) {
    if (!node || !node->parent) {
      return;
    }
    if (on_path.find(node) != on_path.end()) {
      return;
    }
    if (on_path.find(node->parent) == on_path.end()) {
      return;
    }
    if (seen_child.find(node) != seen_child.end()) {
      return;
    }
    seen_child.insert(node);
    traj.children.push_back(node);
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
  return traj;
}

py::dict export_trajectory(const Trajectory &traj, int timesteps) {
  igha::unordered_map<const Node *, int> path_index;
  path_index.reserve(traj.path.size());
  for (size_t i = 0; i < traj.path.size(); ++i) {
    path_index.emplace(traj.path[i].get(), static_cast<int>(i));
  }
  py::dict out;
  out["n_dims"] = n_dims;
  out["n_cont"] = n_cont;
  out["timesteps"] = timesteps;
  pack_side(out, "path", traj.path, path_index, timesteps, false);
  pack_side(out, "child", traj.children, path_index, timesteps, true);
  if (static_cast<int>(traj.goal.size()) == n_dims) {
    out["goal_pose"] = copy_array<float>(traj.goal, {n_dims});
  }
  if (!traj.world.empty()) {
    out["world"] = copy_array<float>(
        traj.world, {static_cast<py::ssize_t>(traj.world.size())});
  }
  return out;
}

void index_path(Library &library, int traj_id) {
  const auto &path = library.trajs[traj_id].path;
  for (int i = 0; i < static_cast<int>(path.size()); ++i) {
    library.lcr[path[i]->LCR_index].push_back({traj_id, i});
  }
}

void index_library(Library &library) {
  library.lcr.clear();
  library.used.assign(library.trajs.size(), 0);
  for (int t = 0; t < static_cast<int>(library.trajs.size()); ++t) {
    index_path(library, t);
  }
}

void reset_library(Library &library) {
  std::fill(library.used.begin(), library.used.end(), 0);
}

Library import_trajectories(const py::list &items, Environment *env) {
  Library library;
  for (py::handle item : items) {
    py::dict d = item.cast<py::dict>();
    int steps = env->timesteps;
    if (d.contains("timesteps")) {
      const int saved = d["timesteps"].cast<int>();
#ifdef IGHASTAR_LIBRARY_TIMESTEPS_FROM_EXPORT
      if (saved < 0) {
        throw std::invalid_argument("trajectory timesteps must be non-negative");
      }
      // Keep this item's length. Do not write it onto env->timesteps: a later
      // item, or Succ, would then make the previous buffers look longer.
      steps = saved;
#else
      if (saved != env->timesteps) {
        throw std::invalid_argument(
            "trajectory timesteps do not match this environment");
      }
#endif
    }
    const int inter_n = steps > 0 ? steps * n_dims : 0;
    const int ctrl_n = steps > 0 ? steps * n_cont : 0;
    auto pose = field_f(d, "path_pose");
    auto g = field_f(d, "path_g");
    auto f = field_f(d, "path_f");
    auto time_direction = field_i(d, "path_time_direction");
    auto parent_arr = field_i(d, "path_parent_index");
    auto has_inter = field_u8(d, "path_has_intermediates");
    auto has_controls = field_u8(d, "path_has_controls");
    if (pose.ndim() != 2 || pose.shape(1) != n_dims) {
      throw std::invalid_argument("path_pose must have shape [N, n_dims]");
    }
    const int n = static_cast<int>(pose.shape(0));
    if (g.size() != n || f.size() != n || time_direction.size() != n ||
        parent_arr.size() != n || has_inter.size() != n ||
        has_controls.size() != n) {
      throw std::invalid_argument("path field lengths must match path_pose");
    }
    FArr inter, controls;
    const float *inter_base = nullptr;
    const float *ctrl_base = nullptr;
    if (inter_n > 0) {
      inter = field_f(d, "path_intermediate_poses");
      if (inter.ndim() != 3 || inter.shape(0) != n || inter.shape(1) != steps ||
          inter.shape(2) != n_dims) {
        throw std::invalid_argument(
            "path_intermediate_poses must have shape [N, timesteps, n_dims]");
      }
      inter_base = inter.data();
    }
    if (ctrl_n > 0) {
      controls = field_f(d, "path_controls");
      if (controls.ndim() != 2 || controls.shape(0) != n ||
          controls.shape(1) != ctrl_n) {
        throw std::invalid_argument(
            "path_controls must have shape [N, timesteps * n_cont]");
      }
      ctrl_base = controls.data();
    }
    FArr clock, step_times;
    U8Arr has_times;
    const float *clock_base = nullptr;
    const float *step_base = nullptr;
    if (d.contains("path_time")) {
      clock = field_f(d, "path_time");
      if (clock.size() != n) {
        throw std::invalid_argument("path_time length must match path_pose");
      }
      clock_base = clock.data();
    }
    if (d.contains("path_intermediate_times") && steps > 0) {
      step_times = field_f(d, "path_intermediate_times");
      has_times = field_u8(d, "path_has_times");
      if (step_times.ndim() != 2 || step_times.shape(0) != n ||
          step_times.shape(1) != steps || has_times.size() != n) {
        throw std::invalid_argument(
            "path_intermediate_times must have shape [N, timesteps]");
      }
      step_base = step_times.data();
    }
    std::vector<int> parent_index(n);
    int root = -1;
    for (int i = 0; i < n; ++i) {
      parent_index[i] = parent_arr.data()[i];
      if (parent_index[i] < 0) {
        if (root >= 0) {
          throw std::invalid_argument("trajectory has more than one root");
        }
        root = i;
      } else if (parent_index[i] >= n) {
        throw std::invalid_argument("path parent_index is outside the path");
      }
    }
    if (n > 0 && root < 0) {
      throw std::invalid_argument("trajectory has no root");
    }
    Trajectory traj;
    traj.path.resize(n);
    std::vector<char> built(n, 0);
    for (int placed = 0; placed < n; ++placed) {
      int next = -1;
      for (int i = 0; i < n; ++i) {
        if (built[i]) {
          continue;
        }
        if (parent_index[i] < 0 || built[parent_index[i]]) {
          next = i;
          break;
        }
      }
      if (next < 0) {
        throw std::invalid_argument("trajectory parent links are cyclic");
      }
      std::shared_ptr<Node> parent;
      if (parent_index[next] >= 0) {
        parent = traj.path[parent_index[next]];
      }
      const float *inter =
          (inter_base != nullptr && has_inter.data()[next])
              ? inter_base + static_cast<size_t>(next) * inter_n
              : nullptr;
      const float *controls =
          (ctrl_base != nullptr && has_controls.data()[next])
              ? ctrl_base + static_cast<size_t>(next) * ctrl_n
              : nullptr;
      const float *times =
          (step_base != nullptr && has_times.data()[next])
              ? step_base + static_cast<size_t>(next) * steps
              : nullptr;
      traj.path[next] = rebuild_vertex(
          pose.data() + static_cast<size_t>(next) * n_dims, g.data()[next],
          f.data()[next], time_direction.data()[next], inter, controls, parent,
          env, steps, times, clock_base ? clock_base[next] : 0.f,
          clock_base != nullptr);
      built[next] = 1;
    }
    auto child_pose = field_f(d, "child_pose");
    if (child_pose.ndim() != 2 || child_pose.shape(1) != n_dims) {
      throw std::invalid_argument("child_pose must have shape [N, n_dims]");
    }
    const int nc = static_cast<int>(child_pose.shape(0));
    auto child_g = field_f(d, "child_g");
    auto child_f = field_f(d, "child_f");
    auto child_time = field_i(d, "child_time_direction");
    auto child_parent = field_i(d, "child_parent_index");
    auto child_has_inter = field_u8(d, "child_has_intermediates");
    auto child_has_controls = field_u8(d, "child_has_controls");
    if (child_g.size() != nc || child_f.size() != nc || child_time.size() != nc ||
        child_parent.size() != nc || child_has_inter.size() != nc ||
        child_has_controls.size() != nc) {
      throw std::invalid_argument("child field lengths must match child_pose");
    }
    FArr child_inter, child_controls;
    const float *child_inter_base = nullptr;
    const float *child_ctrl_base = nullptr;
    if (inter_n > 0) {
      child_inter = field_f(d, "child_intermediate_poses");
      if (child_inter.ndim() != 3 || child_inter.shape(0) != nc ||
          child_inter.shape(1) != steps || child_inter.shape(2) != n_dims) {
        throw std::invalid_argument(
            "child_intermediate_poses must have shape [N, timesteps, n_dims]");
      }
      child_inter_base = child_inter.data();
    }
    if (ctrl_n > 0) {
      child_controls = field_f(d, "child_controls");
      if (child_controls.ndim() != 2 || child_controls.shape(0) != nc ||
          child_controls.shape(1) != ctrl_n) {
        throw std::invalid_argument(
            "child_controls must have shape [N, timesteps * n_cont]");
      }
      child_ctrl_base = child_controls.data();
    }
    FArr child_clock, child_step_times;
    U8Arr child_has_times;
    const float *child_clock_base = nullptr;
    const float *child_step_base = nullptr;
    if (d.contains("child_time")) {
      child_clock = field_f(d, "child_time");
      if (child_clock.size() != nc) {
        throw std::invalid_argument("child_time length must match child_pose");
      }
      child_clock_base = child_clock.data();
    }
    if (d.contains("child_intermediate_times") && steps > 0) {
      child_step_times = field_f(d, "child_intermediate_times");
      child_has_times = field_u8(d, "child_has_times");
      if (child_step_times.ndim() != 2 || child_step_times.shape(0) != nc ||
          child_step_times.shape(1) != steps ||
          child_has_times.size() != nc) {
        throw std::invalid_argument(
            "child_intermediate_times must have shape [N, timesteps]");
      }
      child_step_base = child_step_times.data();
    }
    traj.children.reserve(nc);
    for (int i = 0; i < nc; ++i) {
      const int p = child_parent.data()[i];
      if (p < 0 || p >= n) {
        throw std::invalid_argument("child parent_index is outside the path");
      }
      const float *inter =
          (child_inter_base != nullptr && child_has_inter.data()[i])
              ? child_inter_base + static_cast<size_t>(i) * inter_n
              : nullptr;
      const float *controls =
          (child_ctrl_base != nullptr && child_has_controls.data()[i])
              ? child_ctrl_base + static_cast<size_t>(i) * ctrl_n
              : nullptr;
      const float *times =
          (child_step_base != nullptr && child_has_times.data()[i])
              ? child_step_base + static_cast<size_t>(i) * steps
              : nullptr;
      traj.children.push_back(rebuild_vertex(
          child_pose.data() + static_cast<size_t>(i) * n_dims, child_g.data()[i],
          child_f.data()[i], child_time.data()[i], inter, controls, traj.path[p],
          env, steps, times, child_clock_base ? child_clock_base[i] : 0.f,
          child_clock_base != nullptr));
    }
    if (d.contains("goal_pose")) {
      auto goal_pose = field_f(d, "goal_pose");
      if (goal_pose.size() != n_dims) {
        throw std::invalid_argument("goal_pose must have length n_dims");
      }
      traj.goal.assign(goal_pose.data(), goal_pose.data() + n_dims);
    }
    if (d.contains("world")) {
      auto world = field_f(d, "world");
      traj.world.assign(world.data(), world.data() + world.size());
    }
    traj.steps = steps;
    library.trajs.push_back(std::move(traj));
  }
  index_library(library);
  return library;
}

void record_solution(
    Library &library,
    const std::vector<std::shared_ptr<Node>> &path,
    const igha::unordered_set<std::shared_ptr<Node>> &Seen,
    std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>,
                        NodePtrCompare>
        Q_v,
    const igha::unordered_set<std::shared_ptr<Node>> &inactive_Q_v,
    const float *goal_pose, Environment *env) {
  if (path.empty()) {
    return;
  }
  library.trajs.push_back(
      capture_trajectory(path, Seen, std::move(Q_v), inactive_Q_v));
  auto &traj = library.trajs.back();
  if (goal_pose != nullptr) {
    traj.goal.assign(goal_pose, goal_pose + n_dims);
  }
#ifdef IGHASTAR_STAMP_WORLD
  if (env != nullptr) {
    IGHASTAR_STAMP_WORLD(env, traj.world);
  }
#else
  (void)env;
#endif
  if (env != nullptr) {
    traj.steps = env->timesteps;
  }
  index_path(library, static_cast<int>(library.trajs.size()) - 1);
  library.used.push_back(0);
}

void select_trajectories(Library &library, Environment *env, const float *start,
                         const float *goal) {
  const int n = static_cast<int>(library.trajs.size());
  std::vector<std::vector<float>> goals(n);
  std::vector<std::vector<float>> worlds(n);
  for (int i = 0; i < n; ++i) {
    goals[i] = library.trajs[i].goal;
    worlds[i] = library.trajs[i].world;
  }
  std::vector<uint8_t> keep(n, 1);
  if (env != nullptr) {
    env->filter_trajectories(goals, worlds, start, goal, keep);
  }
  library.applicable = std::move(keep);
}

int reused_expand_index(bool saved_terminal, bool in_goal) {
  return (saved_terminal && !in_goal) ? -1 : 0;
}

int buffer_steps(int stored, int live) {
  return stored > 0 ? stored : live;
}

bool library_reuse_self_check() {
  return reused_expand_index(true, false) == -1 &&
         reused_expand_index(true, true) == 0 &&
         reused_expand_index(false, false) == 0 && buffer_steps(1, 10) == 1 &&
         buffer_steps(0, 10) == 10;
}

py::list export_library(const Library &library, int timesteps) {
  py::list out;
  for (const auto &traj : library.trajs) {
    const int steps = buffer_steps(traj.steps, timesteps);
    out.append(export_trajectory(traj, steps));
  }
  return out;
}

bool library_links_ok(const Library &library) {
  if (library.trajs.empty()) {
    return false;
  }
  for (const auto &traj : library.trajs) {
    if (traj.path.empty()) {
      return false;
    }
    int roots = 0;
    for (const auto &node : traj.path) {
      if (!node->parent) {
        roots++;
        continue;
      }
      bool found = false;
      for (const auto &other : traj.path) {
        if (other == node->parent) {
          found = true;
        }
      }
      if (!found) {
        return false;
      }
    }
    if (roots != 1) {
      return false;
    }
    for (const auto &child : traj.children) {
      if (!child->parent) {
        return false;
      }
      bool parent_on_path = false;
      bool self_on_path = false;
      for (const auto &node : traj.path) {
        if (node == child->parent) {
          parent_on_path = true;
        }
        if (node == child) {
          self_on_path = true;
        }
      }
      if (!parent_on_path || self_on_path) {
        return false;
      }
    }
  }
  return true;
}

std::vector<std::shared_ptr<Node>>
attach_nearmeets(Library &library, const std::shared_ptr<Node> &u,
                 const std::shared_ptr<Node> &goal, Environment *env) {
  std::vector<std::shared_ptr<Node>> out;
  if (!library_reuse_self_check()) {
    throw std::logic_error("library reuse self-check failed");
  }
  if (!u || !goal || !env || library.trajs.empty()) {
    return out;
  }
  auto found = library.lcr.find(u->LCR_index);
  if (found == library.lcr.end()) {
    return out;
  }
  // Every LCR hit goes into one solve. A failed connection must not hide
  // another vertex of the same trajectory; the valid hit with the smallest
  // spliced cost is the meet. A failed connection leaves the trajectory unused.
  struct Cand {
    int traj;
    int idx;
    float spliced;
  };
  std::vector<Cand> cands;
  std::vector<std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>>> pairs;
  for (const auto &hit : found->second) {
    const int traj = hit.first;
    const int idx = hit.second;
    if (traj < 0 || traj >= static_cast<int>(library.used.size()) ||
        library.used[traj]) {
      continue;
    }
    if (!library.applicable.empty() &&
        (traj >= static_cast<int>(library.applicable.size()) ||
         !library.applicable[traj])) {
      continue;
    }
    const auto &path = library.trajs[traj].path;
    if (path.empty() || idx < 0 || idx >= static_cast<int>(path.size())) {
      continue;
    }
    float dist = env->compute_near_meet_distance(u->pose, path[idx]->pose);
    cands.push_back(
        {traj, idx, u->g + dist + (path[0]->g - path[idx]->g)});
    pairs.emplace_back(u, path[idx]);
  }
  if (cands.empty()) {
    return out;
  }
  std::vector<bool> valid;
  std::vector<std::vector<float>> connections;
  env->solve_2pBVP(pairs, valid, connections);
  struct Best {
    int idx;
    float spliced;
    int pair;
  };
  igha::unordered_map<int, Best> best;
  for (size_t n = 0; n < cands.size(); ++n) {
    if (n >= valid.size() || !valid[n]) {
      continue;
    }
    auto it = best.find(cands[n].traj);
    if (it == best.end() || cands[n].spliced < it->second.spliced) {
      best[cands[n].traj] = Best{cands[n].idx, cands[n].spliced,
                                 static_cast<int>(n)};
    }
  }
  for (const auto &kv : best) {
    const int traj = kv.first;
    const int k = kv.second.idx;
    const auto &path = library.trajs[traj].path;
    const std::vector<float> &conn =
        connections[static_cast<size_t>(kv.second.pair)];
#ifdef IGHASTAR_REALIZE_MEET
    MeetEdge edge = IGHASTAR_REALIZE_MEET(env, u, path[k], conn);
#else
    (void)conn;
    MeetEdge edge = geometric_meet_edge(env, u, path[k]);
#endif
    if (!edge.ok) {
      continue;
    }
    library.used[traj] = 1;
    const int stored = buffer_steps(library.trajs[traj].steps, env->timesteps);
    const float g_meet = u->g + edge.cost;
    const float g_old_meet = path[k]->g;
    const float offset = edge.arrival - path[k]->time;
    std::vector<std::shared_ptr<Node>> copies(static_cast<size_t>(k) + 1);
    for (int i = k; i >= 0; --i) {
      const float g_new = g_meet + (path[i]->g - g_old_meet);
      const float h = env->heuristic(path[i]->pose, goal->pose);
      const std::shared_ptr<Node> parent =
          (i == k) ? u : copies[static_cast<size_t>(i) + 1];
      std::vector<float> shifted;
      const float *times = nullptr;
      const float *poses = path[i]->intermediate_poses;
      const float *ctrls = path[i]->intermediate_controls;
      float arrival = path[i]->time + offset;
      if (i == k) {
        poses = edge.poses.empty() ? nullptr : edge.poses.data();
        ctrls = edge.controls.empty() ? nullptr : edge.controls.data();
        times = edge.times.empty() ? nullptr : edge.times.data();
        arrival = edge.arrival;
      } else if (path[i]->intermediate_times != nullptr && stored > 0) {
        shifted.resize(stored);
        for (int t = 0; t < stored; ++t) {
          shifted[static_cast<size_t>(t)] =
              path[i]->intermediate_times[t] + offset;
        }
        times = shifted.data();
      }
      // The meeting edge was just built at the live rollout length. Suffix
      // nodes keep the length they were stored with.
      const int node_steps = (i == k) ? env->timesteps : stored;
      auto copy = rebuild_vertex(path[i]->pose, g_new, g_new + h,
                                 path[i]->time_direction, poses, ctrls, parent,
                                 env, node_steps, times, arrival, true);
      // path[0] is the saved terminal. Leave it expandable when this query's
      // goal is somewhere else, so the search can continue through it.
      copy->expand_index = reused_expand_index(
          i == 0, env->reached_goal_region(copy, goal));
      copies[static_cast<size_t>(i)] = copy;
      out.push_back(copy);
    }
    for (const auto &child : library.trajs[traj].children) {
      int parent_on_suffix = -1;
      for (int i = 0; i <= k; ++i) {
        if (path[i] == child->parent) {
          parent_on_suffix = i;
          break;
        }
      }
      if (parent_on_suffix < 0) {
        continue;
      }
      const float g_new = copies[static_cast<size_t>(parent_on_suffix)]->g +
                          (child->g - path[parent_on_suffix]->g);
      const float h = env->heuristic(child->pose, goal->pose);
      std::vector<float> shifted;
      const float *times = nullptr;
      if (child->intermediate_times != nullptr && stored > 0) {
        shifted.resize(stored);
        for (int t = 0; t < stored; ++t) {
          shifted[static_cast<size_t>(t)] =
              child->intermediate_times[t] + offset;
        }
        times = shifted.data();
      }
      auto copy = rebuild_vertex(
          child->pose, g_new, g_new + h, child->time_direction,
          child->intermediate_poses, child->intermediate_controls,
          copies[static_cast<size_t>(parent_on_suffix)], env, stored, times,
          child->time + offset, true);
      copy->expand_index = -1;
      out.push_back(copy);
    }
  }
  return out;
}

} // namespace tree_utils
