// Generic, OMPL-style environment for IGHAStar.
//
// Unlike the hand-crafted car environments, this environment is configured
// entirely at runtime from user-provided Python callbacks:
//   - sample_controls() -> [K, n_cont]
//   - dynamics(states[B, n_dims], controls[B, n_cont]) -> next_states[B, n_dims]
//   - cost(states[B, n_dims], controls[B, n_cont], next_states[B, n_dims]) -> [B]
//   - validity(states[B, n_dims]) -> [B]   (nonzero == valid)
//   - heuristic(states[B, n_dims]) -> [B]  (cost-to-go; encodes goal knowledge)
//   - goal_test(state[n_dims]) -> bool
//
// Node, hashing, config parsing and path conversion are shared with the
// compiled `generic_torch` environment (see generic_common.h).
#include <generic_common.h>

class Environment : public GenericEnvironmentBase {
  // User-provided Python callbacks.
  py::function sample_controls_fn;
  py::function dynamics_fn;
  py::function cost_fn;
  py::function validity_fn;
  py::function heuristic_fn;
  py::function goal_test_fn;

public:
  Environment(const py::dict &config, int time_direction_ = 1)
      : GenericEnvironmentBase(config, time_direction_) {
    // User callbacks (passed at the top level of the config dict).
    sample_controls_fn = config["sample_controls_fn"].cast<py::function>();
    dynamics_fn = config["dynamics_fn"].cast<py::function>();
    cost_fn = config["cost_fn"].cast<py::function>();
    validity_fn = config["validity_fn"].cast<py::function>();
    heuristic_fn = config["heuristic_fn"].cast<py::function>();
    goal_test_fn = config["goal_test_fn"].cast<py::function>();
  }

  ~Environment() {}

  // Convert a Python-returned object to a contiguous CPU float32 tensor.
  static torch::Tensor to_cpu_float(py::handle obj) {
    return obj.cast<torch::Tensor>()
        .to(torch::kCPU)
        .to(torch::kFloat32)
        .contiguous();
  }

  // Heuristic for a single state (used to initialize the start node). The
  // goal argument is ignored: the user's heuristic encodes goal knowledge.
  float heuristic(float *pose, float *goal) {
    (void)goal;
    auto state_t = torch::empty({1, n_dims}, torch::kFloat32);
    memcpy(state_t.data_ptr<float>(), pose, n_dims * sizeof(float));
    torch::Tensor h = to_cpu_float(heuristic_fn(state_t));
    return h.data_ptr<float>()[0];
  }

  // Goal test for a single popped node.
  bool reached_goal_region(std::shared_ptr<Node> v, std::shared_ptr<Node> goal) {
    (void)goal;
    auto state_t = torch::empty({n_dims}, torch::kFloat32);
    memcpy(state_t.data_ptr<float>(), v->pose, n_dims * sizeof(float));
    return goal_test_fn(state_t).cast<bool>();
  }

  // Validity of start and goal positions.
  void check_validity(float *start, float *goal, bool *result) {
    auto states_t = torch::empty({2, n_dims}, torch::kFloat32);
    float *sp = states_t.data_ptr<float>();
    memcpy(sp, start, n_dims * sizeof(float));
    memcpy(sp + n_dims, goal, n_dims * sizeof(float));
    torch::Tensor v = to_cpu_float(validity_fn(states_t));
    const float *vp = v.data_ptr<float>();
    result[0] = vp[0] != 0.0f;
    result[1] = vp[1] != 0.0f;
  }

  // Batched validity check for multiple states.
  void check_validity_batched(const std::vector<float *> &states,
                              std::vector<bool> &results) {
    int n = static_cast<int>(states.size());
    results.resize(n);
    if (n == 0) {
      return;
    }
    auto t = torch::empty({n, n_dims}, torch::kFloat32);
    float *p = t.data_ptr<float>();
    for (int i = 0; i < n; i++) {
      memcpy(p + i * n_dims, states[i], n_dims * sizeof(float));
    }
    torch::Tensor v = to_cpu_float(validity_fn(t));
    const float *vp = v.data_ptr<float>();
    for (int i = 0; i < n; i++) {
      results[i] = vp[i] != 0.0f;
    }
  }

  // Batched successor: expands every node in `nodes` (M parents) by applying
  // the K user-sampled controls to each, in a single batched callback round.
  // Returns one neighbor list per parent, in input order.
  std::vector<std::vector<std::shared_ptr<Node>>>
  Succ_batched(const std::vector<std::shared_ptr<Node>> &nodes,
               std::shared_ptr<Node> goal) {
    (void)goal;
    int M = static_cast<int>(nodes.size());
    if (M == 0) {
      return {};
    }

    // Sample the control set for this batch (user decides fixed vs random).
    torch::Tensor controls_t = to_cpu_float(sample_controls_fn());
    int K = static_cast<int>(controls_t.size(0));
    if (K == 0) {
      return std::vector<std::vector<std::shared_ptr<Node>>>(M);
    }
    const float *cp = controls_t.data_ptr<float>();

    torch::Tensor states_t, controls_batch_t;
    pack_batch(nodes, cp, K, states_t, controls_batch_t);

    // Dynamics. The callback may return either:
    //   [B, n_dims]            -> a single resulting state (rollout_T = 1), or
    //   [B, rollout_T, n_dims] -> the full sub-trajectory per control.
    torch::Tensor next_t = to_cpu_float(dynamics_fn(states_t, controls_batch_t));
    // The node's final state is the last sub-step of the rollout.
    torch::Tensor final_t =
        (next_t.dim() == 3) ? next_t.select(1, next_t.size(1) - 1).contiguous()
                            : next_t;

    // cost / validity / heuristic operate on the final resulting state.
    torch::Tensor cost_t =
        to_cpu_float(cost_fn(states_t, controls_batch_t, final_t));
    torch::Tensor valid_t = to_cpu_float(validity_fn(final_t));
    torch::Tensor h_t = to_cpu_float(heuristic_fn(final_t));

    return make_successors(nodes, cp, K, next_t, final_t, cost_t, valid_t, h_t);
  }

  // Single-node successor (delegates to the batched path).
  std::vector<std::shared_ptr<Node>> Succ(std::shared_ptr<Node> node,
                                          std::shared_ptr<Node> goal) {
    std::vector<std::shared_ptr<Node>> batch = {node};
    auto results = Succ_batched(batch, goal);
    if (results.empty()) {
      return {};
    }
    return results[0];
  }
  // Path / controls export lives on GenericEnvironmentBase (generic_common.h).
};
