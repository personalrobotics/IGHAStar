// Test bindings for peg_mujoco.h (not used by the planner): rollouts, state encoding, heuristic / goal / validity.
// UNREVIEWED: written by Claude, not yet read or reviewed by a human.
#include <torch/extension.h>

#include "utils/config_utils.h"
#include "utils/sampling_utils.h"
#include <peg_mujoco.h>

namespace {
struct PegEnv {
  std::unique_ptr<Environment> env;
  explicit PegEnv(const py::dict &cfg) : env(new Environment(cfg, 1)) {}

  // states [B, n], controls [B, 9] -> (traj [B, T, n], valid [B] bool)
  std::tuple<torch::Tensor, torch::Tensor> rollout(torch::Tensor x, torch::Tensor u) {
    x = x.to(torch::kFloat32).contiguous(), u = u.to(torch::kFloat32).contiguous();
    const int B = (int)x.size(0), T = env->task.policy_steps();
    auto traj = torch::empty({B, T, n_dims}, torch::kFloat32);
    std::vector<char> ok(B);
    {
      py::gil_scoped_release nogil;
      env->rollout(x.data_ptr<float>(), u.data_ptr<float>(), B, traj.data_ptr<float>(), ok.data());
    }
    auto v = torch::empty({B}, torch::kBool);
    for (int i = 0; i < B; ++i) v[i] = (bool)ok[i];
    return {traj, v};
  }
  // restore each state into MuJoCo and encode it back (checks the state conversion and the feature computation)
  torch::Tensor encode(torch::Tensor x) {
    x = x.to(torch::kFloat32).contiguous();
    const int B = (int)x.size(0);
    auto out = torch::empty({B, n_dims}, torch::kFloat32);
    auto &t = env->task;
    mjData *d = env->datas[0];
    float u[9] = {0, 0, 0, 0, 0, 0, 1, 0, 0};
    for (int i = 0; i < B; ++i) {
      const float *xi = x.data_ptr<float>() + (size_t)i * n_dims;
      u[6] = xi[t.o_grip];
      t.begin(env->model, d, xi, u, env->scratch[0]);
      mj_kinematics(env->model, d);
      t.encode(env->model, d, xi[t.o_grip], out.data_ptr<float>() + (size_t)i * n_dims);
    }
    return out;
  }
  torch::Tensor heuristic(torch::Tensor x) {
    x = x.to(torch::kFloat32).contiguous();
    auto o = torch::empty({x.size(0)}, torch::kFloat32);
    for (int i = 0; i < x.size(0); ++i) o[i] = env->task.heuristic(x.data_ptr<float>() + (size_t)i * n_dims);
    return o;
  }
  torch::Tensor goal(torch::Tensor x) {
    x = x.to(torch::kFloat32).contiguous();
    auto o = torch::empty({x.size(0)}, torch::kBool);
    for (int i = 0; i < x.size(0); ++i) o[i] = env->task.goal(x.data_ptr<float>() + (size_t)i * n_dims);
    return o;
  }
  torch::Tensor alignment(torch::Tensor x) {
    x = x.to(torch::kFloat32).contiguous();
    auto o = torch::empty({x.size(0), 2}, torch::kFloat64);
    for (int i = 0; i < x.size(0); ++i) {
      double dd, aa;
      env->task.alignment(x.data_ptr<float>() + (size_t)i * n_dims, dd, aa);
      o[i][0] = dd, o[i][1] = aa;
    }
    return o;
  }
  torch::Tensor valid(torch::Tensor x) {
    x = x.to(torch::kFloat32).contiguous();
    auto o = torch::empty({x.size(0)}, torch::kBool);
    for (int i = 0; i < x.size(0); ++i) o[i] = env->task.state_valid(x.data_ptr<float>() + (size_t)i * n_dims);
    return o;
  }
  torch::Tensor sample_controls(int K) {
    auto o = torch::empty({K, 9}, torch::kFloat32);
    env->task.sample_controls(env->rng, K, o.data_ptr<float>());
    return o;
  }
  int nthread() const { return env->pool->size(); }
};
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<PegEnv>(m, "PegEnv")
      .def(py::init<const py::dict &>())
      .def("rollout", &PegEnv::rollout)
      .def("encode", &PegEnv::encode)
      .def("heuristic", &PegEnv::heuristic)
      .def("goal", &PegEnv::goal)
      .def("alignment", &PegEnv::alignment)
      .def("valid", &PegEnv::valid)
      .def("sample_controls", &PegEnv::sample_controls)
      .def("nthread", &PegEnv::nthread);
}
