// Generic environment with *compiled* torch callbacks ("generic_torch").
//
// Same Python-facing contract as generic.h, but the user's torch callbacks are
// exported with torch.export and compiled ahead of time with AOTInductor on
// the Python side (ighastar/scripts/generic_torch.py). This header only loads
// the resulting .pt2 packages with torch::inductor::AOTIModelPackageLoader and
// runs them directly from C++: the search loop never takes the GIL and never
// calls back into Python.
//
// Compiled models (all inputs/outputs float32 unless noted; B is dynamic):
//   expand(states[B, n], controls[B, c]) ->
//       (rollout [B, n] or [B, T, n], cost [B], valid [B], heuristic [B])
//     = dynamics -> (final state) -> cost / validity / heuristic, fused into a
//       single call per batched expansion.
//   validity(states[B, n]) -> [B]        (nonzero == valid)
//   heuristic(states[B, n]) -> [B]
//   goal_test(state[n]) -> bool[]        (0-d bool tensor)
//   sample_controls() -> [K, c]          (only when the sampler is random;
//                                         a deterministic sampler is evaluated
//                                         once in Python and passed as the
//                                         constant `fixed_controls` tensor)
//
// The packages and the device are passed in config["_generic_torch"] by
// create_planner; users never set that key themselves.
#include <generic_common.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <string>

class Environment : public GenericEnvironmentBase {
  using Loader = torch::inductor::AOTIModelPackageLoader;

  torch::Device device;
  std::unique_ptr<Loader> expand_model;
  std::unique_ptr<Loader> validity_model;
  std::unique_ptr<Loader> heuristic_model;
  std::unique_ptr<Loader> goal_test_model;
  std::unique_ptr<Loader> sample_controls_model; // null when controls fixed
  torch::Tensor fixed_controls;                   // CPU float32 [K, c]

  std::unique_ptr<Loader> load_model(const py::dict &packages,
                                     const char *name) {
    if (!packages.contains(name)) {
      throw std::runtime_error(
          std::string("generic_torch: missing compiled package for '") + name +
          "'");
    }
    auto path = packages[name].cast<std::string>();
    c10::DeviceIndex idx = device.is_cuda() ? device.index() : -1;
    return std::make_unique<Loader>(path, "model", false, 1, idx);
  }

  // Run a compiled model on `device` and return its outputs as contiguous CPU
  // tensors (float32 unless `keep_dtype`).
  std::vector<torch::Tensor> run(Loader &model,
                                 std::vector<torch::Tensor> inputs,
                                 bool keep_dtype = false) {
    for (auto &t : inputs) {
      t = t.to(device);
    }
    std::vector<torch::Tensor> outs = model.run(inputs);
    for (auto &o : outs) {
      o = keep_dtype ? o.to(torch::kCPU).contiguous()
                     : o.to(torch::kCPU, torch::kFloat32).contiguous();
    }
    return outs;
  }

public:
  Environment(const py::dict &config, int time_direction_ = 1)
      : GenericEnvironmentBase(config, time_direction_), device(torch::kCPU) {
    if (!config.contains("_generic_torch")) {
      throw std::runtime_error(
          "generic_torch: config has no '_generic_torch' entry; build the "
          "planner with ighastar.scripts.common_utils.create_planner");
    }
    auto gt = config["_generic_torch"].cast<py::dict>();
    device = torch::Device(gt["device"].cast<std::string>());
    auto packages = gt["packages"].cast<py::dict>();

    expand_model = load_model(packages, "expand");
    validity_model = load_model(packages, "validity");
    heuristic_model = load_model(packages, "heuristic");
    goal_test_model = load_model(packages, "goal_test");
    if (gt.contains("fixed_controls") && !gt["fixed_controls"].is_none()) {
      fixed_controls = gt["fixed_controls"]
                           .cast<torch::Tensor>()
                           .to(torch::kCPU, torch::kFloat32)
                           .contiguous()
                           .clone();
    } else {
      sample_controls_model = load_model(packages, "sample_controls");
    }
  }

  ~Environment() {}

  float heuristic(float *pose, float *goal) {
    (void)goal;
    c10::InferenceMode guard;
    auto state_t = torch::empty({1, n_dims}, torch::kFloat32);
    memcpy(state_t.data_ptr<float>(), pose, n_dims * sizeof(float));
    auto h = run(*heuristic_model, {state_t})[0];
    return h.data_ptr<float>()[0];
  }

  bool reached_goal_region(std::shared_ptr<Node> v, std::shared_ptr<Node> goal) {
    (void)goal;
    c10::InferenceMode guard;
    auto state_t = torch::empty({n_dims}, torch::kFloat32);
    memcpy(state_t.data_ptr<float>(), v->pose, n_dims * sizeof(float));
    auto out = run(*goal_test_model, {state_t}, /*keep_dtype=*/true)[0];
    return out.to(torch::kBool).item<bool>();
  }

  void check_validity(float *start, float *goal, bool *result) {
    c10::InferenceMode guard;
    auto states_t = torch::empty({2, n_dims}, torch::kFloat32);
    float *sp = states_t.data_ptr<float>();
    memcpy(sp, start, n_dims * sizeof(float));
    memcpy(sp + n_dims, goal, n_dims * sizeof(float));
    auto v = run(*validity_model, {states_t})[0];
    const float *vp = v.data_ptr<float>();
    result[0] = vp[0] != 0.0f;
    result[1] = vp[1] != 0.0f;
  }

  void check_validity_batched(const std::vector<float *> &states,
                              std::vector<bool> &results) {
    int n = static_cast<int>(states.size());
    results.resize(n);
    if (n == 0) {
      return;
    }
    c10::InferenceMode guard;
    auto t = torch::empty({n, n_dims}, torch::kFloat32);
    float *p = t.data_ptr<float>();
    for (int i = 0; i < n; i++) {
      memcpy(p + i * n_dims, states[i], n_dims * sizeof(float));
    }
    auto v = run(*validity_model, {t})[0];
    const float *vp = v.data_ptr<float>();
    for (int i = 0; i < n; i++) {
      results[i] = vp[i] != 0.0f;
    }
  }

  std::vector<std::vector<std::shared_ptr<Node>>>
  Succ_batched(const std::vector<std::shared_ptr<Node>> &nodes,
               std::shared_ptr<Node> goal) {
    (void)goal;
    int M = static_cast<int>(nodes.size());
    if (M == 0) {
      return {};
    }
    c10::InferenceMode guard;

    torch::Tensor controls_t =
        sample_controls_model ? run(*sample_controls_model, {})[0]
                              : fixed_controls;
    int K = static_cast<int>(controls_t.size(0));
    if (K == 0) {
      return std::vector<std::vector<std::shared_ptr<Node>>>(M);
    }
    const float *cp = controls_t.data_ptr<float>();

    torch::Tensor states_t, controls_batch_t;
    pack_batch(nodes, cp, K, states_t, controls_batch_t);

    // One fused call: dynamics + cost / validity / heuristic on final state.
    auto outs = run(*expand_model, {states_t, controls_batch_t});
    torch::Tensor next_t = outs[0];
    torch::Tensor final_t =
        (next_t.dim() == 3) ? next_t.select(1, next_t.size(1) - 1).contiguous()
                            : next_t;
    return make_successors(nodes, cp, K, next_t, final_t, outs[1], outs[2],
                           outs[3]);
  }

  std::vector<std::shared_ptr<Node>> Succ(std::shared_ptr<Node> node,
                                          std::shared_ptr<Node> goal) {
    std::vector<std::shared_ptr<Node>> batch = {node};
    auto results = Succ_batched(batch, goal);
    if (results.empty()) {
      return {};
    }
    return results[0];
  }
};
