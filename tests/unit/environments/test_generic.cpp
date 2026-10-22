// Environment-level tests for the generic environments ("generic" with Python
// callbacks, "generic_torch" with AOTInductor-compiled callbacks). Mirrors
// test.cpp, but exposes the Environment methods individually so the Python
// side can compare both modes on identical inputs. Built by
// test_generic_environment.py with -DIGHASTAR_ENV_HEADER plus -DN_DIMS / -DN_CONT / -DHASH_DIMS.
#include <memory>
#include <torch/extension.h>
#include <tuple>
#include <vector>

#ifndef IGHASTAR_ENV_HEADER
#error "test_generic.cpp needs -DIGHASTAR_ENV_HEADER='\"generic.h\"' or generic_torch.h"
#endif
#include IGHASTAR_ENV_HEADER

class GenericEnvironmentTester {
  std::unique_ptr<Environment> env;

  static std::vector<float> to_vec(const torch::Tensor &t) {
    auto c = t.to(torch::kCPU, torch::kFloat32).contiguous();
    return std::vector<float>(c.data_ptr<float>(),
                              c.data_ptr<float>() + c.numel());
  }

public:
  GenericEnvironmentTester(const py::dict &config) {
    env = std::make_unique<Environment>(config);
  }

  int state_dim() const { return n_dims; }
  int control_dim() const { return n_cont; }

  // Number of valid states among (a, b) via check_validity (0, 1 or 2).
  int check_validity(torch::Tensor a, torch::Tensor b) {
    auto va = to_vec(a), vb = to_vec(b);
    bool result[2] = {false, false};
    env->check_validity(va.data(), vb.data(), result);
    return int(result[0]) + int(result[1]);
  }

  // Per-row validity via check_validity_batched.
  std::vector<bool> check_validity_batched(torch::Tensor states) {
    auto s = states.to(torch::kCPU, torch::kFloat32).contiguous();
    std::vector<float *> ptrs;
    for (int64_t i = 0; i < s.size(0); i++) {
      ptrs.push_back(s.data_ptr<float>() + i * n_dims);
    }
    std::vector<bool> results;
    env->check_validity_batched(ptrs, results);
    return results;
  }

  float heuristic(torch::Tensor state) {
    auto v = to_vec(state);
    return env->heuristic(v.data(), v.data());
  }

  bool reached_goal(torch::Tensor state) {
    auto v = to_vec(state);
    auto node = env->create_Node(v.data());
    return env->reached_goal_region(node, node);
  }

  // Successors of a single state: (poses [S, n], g [S], f [S]).
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  successors(torch::Tensor state) {
    return successors_batched(state.reshape({1, n_dims}));
  }

  // Successors of M states expanded in one Succ_batched call, concatenated in
  // parent order: (poses [S, n], g [S], f [S]).
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  successors_batched(torch::Tensor states) {
    auto s = states.to(torch::kCPU, torch::kFloat32).contiguous();
    std::vector<std::shared_ptr<Node>> parents;
    for (int64_t i = 0; i < s.size(0); i++) {
      parents.push_back(env->create_Node(s.data_ptr<float>() + i * n_dims));
    }
    auto lists = env->Succ_batched(parents, parents[0]);
    std::vector<std::shared_ptr<Node>> flat;
    for (auto &l : lists) {
      flat.insert(flat.end(), l.begin(), l.end());
    }
    int S = static_cast<int>(flat.size());
    auto poses = torch::zeros({S, n_dims});
    auto g = torch::zeros({S});
    auto f = torch::zeros({S});
    for (int i = 0; i < S; i++) {
      for (int d = 0; d < n_dims; d++) {
        poses[i][d] = flat[i]->pose[d];
      }
      g[i] = flat[i]->g;
      f[i] = flat[i]->f;
    }
    return {poses, g, f};
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<GenericEnvironmentTester>(m, "GenericEnvironmentTester",
                                       py::module_local())
      .def(py::init<const py::dict &>())
      .def("state_dim", &GenericEnvironmentTester::state_dim)
      .def("control_dim", &GenericEnvironmentTester::control_dim)
      .def("check_validity", &GenericEnvironmentTester::check_validity)
      .def("check_validity_batched",
           &GenericEnvironmentTester::check_validity_batched)
      .def("heuristic", &GenericEnvironmentTester::heuristic)
      .def("reached_goal", &GenericEnvironmentTester::reached_goal)
      .def("successors", &GenericEnvironmentTester::successors)
      .def("successors_batched", &GenericEnvironmentTester::successors_batched);
}
