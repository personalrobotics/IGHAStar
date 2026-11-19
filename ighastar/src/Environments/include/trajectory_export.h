#pragma once

// Plan export. Include after node.h.
// Path shape is [rows, n_dims + 2]: state, clock, then g * time_direction.
// Controls have one row per path row, [rows, n_cont].
//
// node.h reverses stored poses on a backward edge and leaves controls in
// generator order. The control read below undoes that so row i is the action
// that produced path row i.

#include <torch/extension.h>

#include <stdexcept>
#include <vector>

inline void trajectory_export_self_check() {
  float t[3];
  fill_substep_times(t, 1.f, -0.5f, 3);
  if (t[0] != 0.5f || t[1] != 0.f || t[2] != -0.5f) {
    throw std::runtime_error("fill_substep_times failed");
  }
  // Backward storage reverses poses. The arrival sample moves to index 0.
  float stored[3];
  for (int i = 0; i < 3; ++i) {
    stored[i] = t[2 - i];
  }
  if (stored[0] != -0.5f || stored[2] != 0.5f) {
    throw std::runtime_error("reversed substep times failed");
  }
  // Controls stay in generator order, so the control for stored pose k
  // on a backward edge is at timesteps - 1 - k.
  const int timesteps = 3;
  const int k = timesteps - 1;
  const int ck = timesteps - 1 - k;
  if (ck != 0) {
    throw std::runtime_error("backward control index failed");
  }
}

inline void trajectory_export_check_once() {
  static int checked = 0;
  if (!checked) {
    trajectory_export_self_check();
    checked = 1;
  }
}

// Dense edge playback. Skips the root and any node with no stored rollout.
inline torch::Tensor path_tensor_from_edges(
    const std::vector<std::shared_ptr<Node>> &node_list, int timesteps) {
  trajectory_export_check_once();
  const int cols = n_dims + 2;
  auto empty = torch::zeros({0, cols}, torch::TensorOptions().dtype(torch::kFloat32));
  if (node_list.empty()) {
    return empty;
  }
  struct Row {
    float pose[n_dims];
    float g;
    float time;
  };
  std::vector<Row> rows;
  for (const auto &node : node_list) {
    if (node->parent == nullptr || node->intermediate_poses == nullptr) {
      continue;
    }
    const float g0 = node->parent->g;
    const float g1 = node->g;
    const float t0 = node->parent->time;
    const float t1 = node->time;
    const float td = static_cast<float>(node->time_direction);
    for (int j = 0; j < timesteps; ++j) {
      const int k = timesteps - j - 1;
      Row s;
      for (int d = 0; d < n_dims; ++d) {
        s.pose[d] = node->intermediate_poses[k * n_dims + d];
      }
      // Backward Node stores intermediates reversed; map k → chrono fraction.
      const float alpha = static_cast<float>(td < 0.f ? timesteps - k : k + 1) /
                          static_cast<float>(timesteps);
      s.g = (g0 + alpha * (g1 - g0)) * td;
      s.time = node->intermediate_times != nullptr
                   ? node->intermediate_times[k]
                   : t0 + alpha * (t1 - t0);
      rows.push_back(s);
    }
  }
  if (rows.empty()) {
    return empty;
  }
  auto path = torch::zeros({static_cast<int>(rows.size()), cols},
                           torch::TensorOptions().dtype(torch::kFloat32));
  for (size_t i = 0; i < rows.size(); ++i) {
    for (int d = 0; d < n_dims; ++d) {
      path[i][d] = rows[i].pose[d];
    }
    path[i][n_dims] = rows[i].time;
    path[i][n_dims + 1] = rows[i].g;
  }
  return path;
}

inline torch::Tensor controls_tensor_from_edges(
    const std::vector<std::shared_ptr<Node>> &node_list, int timesteps) {
  trajectory_export_check_once();
  auto empty = torch::zeros({0, n_cont}, torch::TensorOptions().dtype(torch::kFloat32));
  if (node_list.empty()) {
    return empty;
  }
  int n = 0;
  for (const auto &node : node_list) {
    if (node->parent == nullptr || node->intermediate_poses == nullptr) {
      continue;
    }
    n += timesteps;
  }
  if (n == 0) {
    return empty;
  }
  auto controls = torch::zeros({n, n_cont}, torch::TensorOptions().dtype(torch::kFloat32));
  int r = 0;
  for (const auto &node : node_list) {
    if (node->parent == nullptr || node->intermediate_poses == nullptr) {
      continue;
    }
    for (int j = 0; j < timesteps; ++j) {
      const int k = timesteps - j - 1;
      const int ck = node->time_direction < 0 ? timesteps - 1 - k : k;
      for (int d = 0; d < n_cont; ++d) {
        if (node->intermediate_controls != nullptr) {
          controls[r][d] = node->intermediate_controls[ck * n_cont + d];
        } else {
          controls[r][d] = node->control[d];
        }
      }
      ++r;
    }
  }
  return controls;
}

// One row per node, including the root. simple has no stored substeps.
inline torch::Tensor path_tensor_from_nodes(
    const std::vector<std::shared_ptr<Node>> &node_list) {
  trajectory_export_check_once();
  const int cols = n_dims + 2;
  if (node_list.empty()) {
    return torch::zeros({0, cols}, torch::TensorOptions().dtype(torch::kFloat32));
  }
  const int n = static_cast<int>(node_list.size());
  auto path = torch::zeros({n, cols}, torch::TensorOptions().dtype(torch::kFloat32));
  for (int i = 0; i < n; ++i) {
    for (int d = 0; d < n_dims; ++d) {
      path[i][d] = node_list[i]->pose[d];
    }
    path[i][n_dims] = node_list[i]->time;
    path[i][n_dims + 1] = node_list[i]->g * node_list[i]->time_direction;
  }
  return path;
}

inline torch::Tensor controls_tensor_from_nodes(
    const std::vector<std::shared_ptr<Node>> &node_list) {
  trajectory_export_check_once();
  if (node_list.empty()) {
    return torch::zeros({0, n_cont}, torch::TensorOptions().dtype(torch::kFloat32));
  }
  const int n = static_cast<int>(node_list.size());
  auto controls = torch::zeros({n, n_cont}, torch::TensorOptions().dtype(torch::kFloat32));
  for (int i = 0; i < n; ++i) {
    for (int d = 0; d < n_cont; ++d) {
      if (node_list[i]->intermediate_controls != nullptr) {
        controls[i][d] = node_list[i]->intermediate_controls[d];
      } else {
        controls[i][d] = node_list[i]->control[d];
      }
    }
  }
  return controls;
}
