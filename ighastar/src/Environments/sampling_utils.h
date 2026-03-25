#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <vector>

namespace sampling {

constexpr float PI = 3.14159265358979323846f;

enum class SpaceType {
  R_N,  // Euclidean space (all dimensions are linear)
  SE2   // Special Euclidean group (has a rotational component)
};

struct SamplingConfig {
  int n_dims;
  SpaceType space_type;
  std::vector<int> angular_dims;  // Which dimensions are angular (for SE2), empty if none
  int num_samples;
  float epsilon_multiplier;
  
  // Bounds for each dimension (optional, nullptr means no bounds)
  const float* min_bounds;  // size: n_dims
  const float* max_bounds;  // size: n_dims
  
  // Sampling standard deviations for each dimension
  const float* sampling_sigma;  // size: n_dims
};

// Check if a dimension is angular
inline bool is_angular_dim(int d, const std::vector<int>& angular_dims) {
  for (int ad : angular_dims) {
    if (d == ad) return true;
  }
  return false;
}

// Wrap angle to [-PI, PI]
inline float wrap_angle(float angle) {
  while (angle > PI) angle -= 2.0f * PI;
  while (angle < -PI) angle += 2.0f * PI;
  return angle;
}

// Compute shortest angular difference
inline float angular_diff(float target, float source) {
  float diff = target - source;
  while (diff > PI) diff -= 2.0f * PI;
  while (diff < -PI) diff += 2.0f * PI;
  return diff;
}

// Interpolate between two poses, handling angular dimensions
inline void interpolate_pose(
    const float* start, const float* end, float alpha,
    int n_dims, const std::vector<int>& angular_dims, float* result) {
  for (int d = 0; d < n_dims; d++) {
    if (is_angular_dim(d, angular_dims)) {
      float diff = angular_diff(end[d], start[d]);
      result[d] = wrap_angle(start[d] + alpha * diff);
    } else {
      result[d] = start[d] + alpha * (end[d] - start[d]);
    }
  }
}

// Generate sampled states around a center pose
// Returns vector of sampled poses (each is n_dims floats)
// 
// For SE2: samples in the local frame of center_pose using cross-track/along-track errors
// For R_N: samples using simple Gaussian perturbations
template <typename ValidityChecker>
std::vector<std::vector<float>> sample_states_around_pose(
    const float* center_pose,
    const SamplingConfig& config,
    ValidityChecker&& validity_checker) {
  
  std::vector<std::vector<float>> valid_samples;
  std::vector<std::vector<float>> all_samples;
  
  std::random_device rd;
  std::mt19937 gen(rd());
  
  // Create distributions for each dimension
  std::vector<std::normal_distribution<float>> dists;
  for (int d = 0; d < config.n_dims; d++) {
    float sigma = config.sampling_sigma[d] * config.epsilon_multiplier;
    dists.emplace_back(0.0f, sigma);
  }
  
  // Generate samples
  for (int i = 0; i < config.num_samples; i++) {
    std::vector<float> sampled_pose(config.n_dims);
    
    if (config.space_type == SpaceType::SE2 && !config.angular_dims.empty()) {
      // SE(2) sampling: use cross-track/along-track errors
      // Uses first angular_dim as the primary heading dimension
      int primary_angular = config.angular_dims[0];
      float yaw = center_pose[primary_angular];
      float cos_yaw = cosf(yaw);
      float sin_yaw = sinf(yaw);
      
      // Sample offsets
      std::vector<float> offsets(config.n_dims);
      for (int d = 0; d < config.n_dims; d++) {
        offsets[d] = dists[d](gen);
      }
      
      // For SE(2), first two dims are typically x,y
      // Transform from local frame (cte, ate) to world frame
      if (config.n_dims >= 2) {
        float cte = offsets[0];  // cross-track error (perpendicular to heading)
        float ate = offsets[1];  // along-track error (parallel to heading)
        sampled_pose[0] = center_pose[0] + cte * cos_yaw - ate * sin_yaw;
        sampled_pose[1] = center_pose[1] + cte * sin_yaw + ate * cos_yaw;
      }
      
      // Handle remaining dimensions (including angular)
      for (int d = 2; d < config.n_dims; d++) {
        if (is_angular_dim(d, config.angular_dims)) {
          sampled_pose[d] = wrap_angle(center_pose[d] + offsets[d]);
        } else {
          sampled_pose[d] = center_pose[d] + offsets[d];
        }
      }
    } else {
      // R^N sampling: simple Gaussian perturbations
      for (int d = 0; d < config.n_dims; d++) {
        float offset = dists[d](gen);
        if (is_angular_dim(d, config.angular_dims)) {
          sampled_pose[d] = wrap_angle(center_pose[d] + offset);
        } else {
          sampled_pose[d] = center_pose[d] + offset;
        }
      }
    }
    
    // Apply bounds if specified
    if (config.min_bounds && config.max_bounds) {
      for (int d = 0; d < config.n_dims; d++) {
        if (!is_angular_dim(d, config.angular_dims)) {  // Don't clamp angular dimensions
          sampled_pose[d] = std::clamp(sampled_pose[d], 
                                        config.min_bounds[d], 
                                        config.max_bounds[d]);
        }
      }
    }
    
    all_samples.push_back(std::move(sampled_pose));
  }
  
  // Batch validity check
  std::vector<float*> state_ptrs;
  for (auto& s : all_samples) {
    state_ptrs.push_back(s.data());
  }
  
  std::vector<bool> validities;
  validity_checker(state_ptrs, validities);
  
  // Collect valid samples
  for (size_t i = 0; i < all_samples.size(); i++) {
    if (validities[i]) {
      valid_samples.push_back(std::move(all_samples[i]));
    }
  }
  
  return valid_samples;
}

// Generate interpolated intermediate states between two poses
// Returns vector of timesteps, each containing n_dims floats
inline std::vector<std::vector<float>> generate_intermediate_states(
    const float* start_pose, const float* end_pose,
    int n_dims, const std::vector<int>& angular_dims, int timesteps) {
  
  std::vector<std::vector<float>> intermediates(timesteps);
  
  for (int t = 0; t < timesteps; t++) {
    float alpha = static_cast<float>(t + 1) / static_cast<float>(timesteps);
    intermediates[t].resize(n_dims);
    interpolate_pose(start_pose, end_pose, alpha, n_dims, angular_dims, 
                     intermediates[t].data());
  }
  
  return intermediates;
}

// Generate perturbation cache (pre-scaled random offsets)
inline std::vector<std::vector<float>> generate_perturbation_cache(
    const float* scale_per_dim, int n_dims, 
    int num_perturbations, float perturbation_scale) {
  
  std::vector<std::vector<float>> perturbations(num_perturbations);
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0f, 1.0f);
  
  for (int i = 0; i < num_perturbations; i++) {
    perturbations[i].resize(n_dims);
    for (int d = 0; d < n_dims; d++) {
      perturbations[i][d] = dist(gen) * perturbation_scale * scale_per_dim[d];
    }
  }
  
  return perturbations;
}

// Check interpolation validity with perturbations
// Returns true if all interpolated (and perturbed) states are valid
// Generates interior points at t = i/num_interpolation_points for i = 1..num_interpolation_points-1
// (endpoints at t=0 and t=1 are skipped since they are the input poses themselves)
template <typename ValidityChecker>
bool check_interpolation_validity(
    const float* pose1, const float* pose2,
    int n_dims, const std::vector<int>& angular_dims,
    int num_interpolation_points,
    const std::vector<std::vector<float>>& perturbations,
    ValidityChecker&& validity_checker) {
  
  if (num_interpolation_points <= 1) {
    return true;  // No interior points to check
  }
  
  std::vector<std::vector<float>> states_storage;
  std::vector<float*> states_to_check;
  
  // Generate interior points (skip endpoints i=0 and i=num_interpolation_points)
  for (int i = 1; i < num_interpolation_points; i++) {
    float t = static_cast<float>(i) / static_cast<float>(num_interpolation_points);
    
    // Base interpolated state
    std::vector<float> base_state(n_dims);
    interpolate_pose(pose1, pose2, t, n_dims, angular_dims, base_state.data());
    states_storage.push_back(base_state);
    states_to_check.push_back(states_storage.back().data());
    
    // Perturbed states
    for (const auto& perturbation : perturbations) {
      std::vector<float> perturbed_state(n_dims);
      for (int d = 0; d < n_dims; d++) {
        if (is_angular_dim(d, angular_dims)) {
          perturbed_state[d] = wrap_angle(base_state[d] + perturbation[d]);
        } else {
          perturbed_state[d] = base_state[d] + perturbation[d];
        }
      }
      states_storage.push_back(perturbed_state);
      states_to_check.push_back(states_storage.back().data());
    }
  }
  
  // Batch validity check
  std::vector<bool> results;
  validity_checker(states_to_check, results);
  
  // All must be valid
  for (bool valid : results) {
    if (!valid) return false;
  }
  return true;
}

// Compute distance between two poses (Euclidean, ignoring angular wrapping for now)
inline float compute_distance(const float* pose1, const float* pose2, int n_dims) {
  float sum_sq = 0.0f;
  for (int d = 0; d < n_dims; d++) {
    float diff = pose1[d] - pose2[d];
    sum_sq += diff * diff;
  }
  return std::sqrt(sum_sq);
}

// Compute SE(2) aware distance (Euclidean for position, angular for heading)
inline float compute_se2_distance(
    const float* pose1, const float* pose2, 
    int n_dims, const std::vector<int>& angular_dims,
    const float* weights = nullptr) {
  
  float sum_sq = 0.0f;
  for (int d = 0; d < n_dims; d++) {
    float diff;
    if (is_angular_dim(d, angular_dims)) {
      diff = angular_diff(pose1[d], pose2[d]);
    } else {
      diff = pose1[d] - pose2[d];
    }
    float w = weights ? weights[d] : 1.0f;
    sum_sq += w * diff * diff;
  }
  return std::sqrt(sum_sq);
}

// Full pipeline: sample states, validate, and create nodes using a factory
// NodeFactory signature: (float* pose, float* intermediate_poses, float g_value, float f_value) -> NodeType
// HeuristicFn signature: (float* pose1, float* pose2) -> float
template <typename NodeType, typename ValidityChecker, typename HeuristicFn, typename NodeFactory>
std::vector<NodeType> sample_and_create_nodes(
    const float* center_pose,
    const float* goal_pose,
    const SamplingConfig& config,
    int timesteps,
    ValidityChecker&& validity_checker,
    HeuristicFn&& heuristic,
    NodeFactory&& node_factory) {
  
  // Step 1: Sample valid poses
  auto valid_poses = sample_states_around_pose(center_pose, config, 
                                                std::forward<ValidityChecker>(validity_checker));
  
  // Step 2: Convert to nodes
  std::vector<NodeType> nodes;
  nodes.reserve(valid_poses.size());
  
  for (auto& sampled_pose : valid_poses) {
    // Generate intermediate states
    auto intermediates = generate_intermediate_states(
        center_pose, sampled_pose.data(),
        config.n_dims, config.angular_dims, timesteps);
    
    // Flatten to contiguous array
    std::vector<float> intermediate_poses(timesteps * config.n_dims);
    for (int t = 0; t < timesteps; t++) {
      std::copy(intermediates[t].begin(), intermediates[t].end(),
                intermediate_poses.begin() + t * config.n_dims);
    }
    
    // Compute g and f values
    float g_value = heuristic(center_pose, sampled_pose.data());
    float h_to_goal = heuristic(sampled_pose.data(), goal_pose);
    float f_value = g_value + h_to_goal;
    
    // Create node using factory (factory handles parent's g addition internally)
    auto node = node_factory(sampled_pose.data(), intermediate_poses.data(), g_value, f_value);
    
    nodes.push_back(std::move(node));
  }
  
  return nodes;
}

// Goal sampling configuration structure
struct GoalSamplingConfig {
  bool enabled = true;
  SpaceType space_type = SpaceType::SE2;
  std::vector<int> angular_dims = {2};
  int num_samples = 32;
  std::vector<float> sigma;
  int n_dims = 4;
  
  // Default sigma values
  GoalSamplingConfig() : sigma({2.0f, 2.0f, 0.2f, 5.0f}) {}
};

// Sample states in the goal region and create nodes
// This is the main entry point for goal region sampling in bidirectional search
// 
// Template parameters:
//   NodeType: The node type (e.g., std::shared_ptr<Node>)
//   ValidityChecker: (const std::vector<float*>&, std::vector<bool>&) -> void
//   HeuristicFn: (const float*, const float*) -> float
//   NodeFactory: (float* pose, float* intermediates, float g, float f) -> NodeType
//   NearMeetChecker: (const NodeType&) -> bool (returns true if valid)
//
// Returns vector of valid sampled nodes
template <typename NodeType, typename ValidityChecker, typename HeuristicFn, 
          typename NodeFactory, typename NearMeetChecker>
std::vector<NodeType> sample_goal_region(
    const float* start_pose,
    const float* goal_pose,
    const GoalSamplingConfig& gs_config,
    int timesteps,
    ValidityChecker&& validity_checker,
    HeuristicFn&& heuristic_fn,
    NodeFactory&& node_factory,
    NearMeetChecker&& near_meet_checker) {
  
  if (!gs_config.enabled) {
    return {};
  }
  
  const int n_dims = gs_config.n_dims;
  
  // Compute sampling ball radius for clamping based on sigma
  float sigma_ball = std::sqrt(gs_config.sigma[0] * gs_config.sigma[0] / 2.0f + 
                               gs_config.sigma[1] * gs_config.sigma[1] / 2.0f);
  
  // Set up bounds (angular dimensions use [-PI, PI])
  std::vector<float> min_bounds(n_dims);
  std::vector<float> max_bounds(n_dims);
  
  for (int d = 0; d < n_dims; d++) {
    if (is_angular_dim(d, gs_config.angular_dims)) {
      min_bounds[d] = -PI;
      max_bounds[d] = PI;
    } else if (d < 2) {
      // Position dimensions use sigma_ball
      min_bounds[d] = start_pose[d] - sigma_ball;
      max_bounds[d] = start_pose[d] + sigma_ball;
    } else {
      // Other dimensions use their specific sigma
      min_bounds[d] = start_pose[d] - gs_config.sigma[d];
      max_bounds[d] = start_pose[d] + gs_config.sigma[d];
    }
  }
  
  // Build SamplingConfig
  SamplingConfig sample_config;
  sample_config.n_dims = n_dims;
  sample_config.space_type = gs_config.space_type;
  sample_config.angular_dims = gs_config.angular_dims;
  sample_config.num_samples = gs_config.num_samples;
  sample_config.epsilon_multiplier = 1.0f;  // sigma already specifies the scale
  sample_config.sampling_sigma = gs_config.sigma.data();
  sample_config.min_bounds = min_bounds.data();
  sample_config.max_bounds = max_bounds.data();
  
  // Sample and create nodes
  auto sampled_nodes = sample_and_create_nodes<NodeType>(
      start_pose, goal_pose,
      sample_config, timesteps,
      std::forward<ValidityChecker>(validity_checker),
      std::forward<HeuristicFn>(heuristic_fn),
      std::forward<NodeFactory>(node_factory));
  
  // Filter using near-meet validity check
  sampled_nodes.erase(
      std::remove_if(sampled_nodes.begin(), sampled_nodes.end(),
          [&near_meet_checker](const NodeType& node) {
            return !near_meet_checker(node);
          }),
      sampled_nodes.end());
  
  return sampled_nodes;
}

}  // namespace sampling
