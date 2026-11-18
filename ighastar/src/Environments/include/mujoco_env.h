// MuJoCo (CPU) environment framework for IGHA*: the whole expansion -- control sampling, state restore, physics,
// controller, validity, cost, heuristic, goal test -- runs in C++; the search never calls back into Python.
// UNREVIEWED: written by Claude, not yet read or reviewed by a human.
//
// A concrete environment is a *Task* class plus one line:
//
//     #include <mujoco_env.h>
//     struct MyTask { ... };                                   // see the interface below
//     class Environment : public MujocoEnvironment<MyTask> {
//      public: using MujocoEnvironment<MyTask>::MujocoEnvironment; };
//
// and is compiled like any other env header (-DIGHASTAR_ENV_HEADER, -DN_DIMS/-DN_CONT/-DHASH_DIMS, linked against
// libmujoco; see node_type "mujoco" in ighastar/scripts/common_utils.py). Node layout, hashing, config parsing
// (resolution, tolerance, LCR, bounds, goal sampling) and path export are the generic environment's
// (generic_common.h), so planner configs and outputs are interchangeable with the Python "generic" env.
//
// Config: everything the generic env reads, plus config["mujoco"] (a dict) with
//     model     path of a compiled model (.mjb) or MJCF (.xml)
//     nthread   rollout threads (default: hardware concurrency)
//     seed      control-sampler seed (default 0)
//   and whatever the Task reads from the same dict in Task::init.
//
// One edge = Task::policy_steps() policy steps of Task::substeps() physics steps each. Per edge row (one parent node x
// one control) on one worker thread with its own mjData:
//     ok = task.begin(m, d, x0, u, scratch)              restore x0 into d (mj_resetData first), resolve the control
//     for t < T:  for s < substeps:  mj_step1;  task.control(m, d, t, s, scratch);  mj_step2
//                 ok &= task.record(m, d, t, scratch, x_t)   pack the state after policy step t (+ per-step validity)
//     ok &= no new mjWARN_BADQACC;  ok &= task.state_valid(x_T)
// The child keeps all T states (the dense path); cost / heuristic / goal are evaluated on x_T.
//
// Task interface (all methods may be called concurrently from several threads, except init/sample_controls):
//     struct Scratch;                                                   per-thread work space
//     void init(const py::dict &mj, const mjModel *m);                  read parameters (GIL held)
//     void make_scratch(const mjModel *m, Scratch &s) const;
//     int policy_steps() const;  int substeps() const;  int num_controls() const;
//     void sample_controls(std::mt19937_64 &rng, int K, float *u);      K x N_CONT, one set per batch
//     bool begin(const mjModel*, mjData*, const float *x, const float *u, Scratch&) const;
//     void control(const mjModel*, mjData*, int t, int sub, Scratch&) const;
//     bool record(const mjModel*, mjData*, int t, Scratch&, float *x_out) const;
//     bool state_valid(const float *x) const;
//     float cost(const float *x0, const float *u, const float *x1) const;
//     float heuristic(const float *x) const;
//     bool goal(const float *x) const;
#pragma once

#include <generic_common.h>
#include <mujoco/mujoco.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace mjenv {

// Persistent worker pool: run(n, f) calls f(worker, i) for i < n, items handed out dynamically; blocks until done.
// The calling thread works too (it is worker 0).
class ThreadPool {
 public:
  explicit ThreadPool(int n) : n_(std::max(1, n)) {
    for (int w = 1; w < n_; ++w) th_.emplace_back([this, w] { loop(w); });
  }
  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
      ++gen_;
    }
    cv_.notify_all();
    for (auto &t : th_) t.join();
  }
  int size() const { return n_; }
  void run(int n, const std::function<void(int, int)> &f) {
    if (n <= 0) return;
    if (n_ == 1 || n == 1) {
      for (int i = 0; i < n; ++i) f(0, i);
      return;
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
      job_ = &f;
      njob_ = n;
      next_.store(0);
      active_ = n_ - 1;
      ++gen_;
    }
    cv_.notify_all();
    work(0);
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [this] { return active_ == 0; });
    job_ = nullptr;
  }

 private:
  void work(int w) {
    for (int i; (i = next_.fetch_add(1)) < njob_;) (*job_)(w, i);
  }
  void loop(int w) {
    long seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return gen_ != seen; });
        seen = gen_;
        if (stop_) return;
      }
      work(w);
      {
        std::lock_guard<std::mutex> lk(mu_);
        if (--active_ == 0) done_cv_.notify_one();
      }
    }
  }
  int n_;
  std::vector<std::thread> th_;
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(int, int)> *job_ = nullptr;
  int njob_ = 0, active_ = 0;
  std::atomic<int> next_{0};
  long gen_ = 0;
  bool stop_ = false;
};

inline mjModel *load_model(const std::string &path) {
  char err[1000] = "";
  mjModel *m = nullptr;
  if (path.size() > 4 && path.compare(path.size() - 4, 4, ".mjb") == 0)
    m = mj_loadModel(path.c_str(), nullptr);
  else
    m = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
  if (!m) throw std::runtime_error("mujoco_env: cannot load model '" + path + "' " + err);
  return m;
}

// small helpers for Task code reading py::dict parameters
inline std::vector<double> dvec(const py::dict &d, const char *k) {
  if (!d.contains(k)) throw std::runtime_error(std::string("mujoco_env: missing parameter '") + k + "'");
  std::vector<double> out;
  for (auto v : d[k].cast<py::sequence>()) out.push_back(py::float_(v).cast<double>());
  return out;
}
inline double dnum(const py::dict &d, const char *k, double dflt) {
  return d.contains(k) ? d[k].cast<double>() : dflt;
}
inline double dnum(const py::dict &d, const char *k) {
  if (!d.contains(k)) throw std::runtime_error(std::string("mujoco_env: missing parameter '") + k + "'");
  return d[k].cast<double>();
}

}  // namespace mjenv

template <class Task>
class MujocoEnvironment : public GenericEnvironmentBase {
 public:
  Task task;
  mjModel *model = nullptr;
  std::vector<mjData *> datas;
  std::vector<typename Task::Scratch> scratch;
  std::unique_ptr<mjenv::ThreadPool> pool;
  std::mt19937_64 rng;
  // profiling (read by the Python side through the test bindings, or ignored)
  long rollouts = 0;
  double rollout_seconds = 0.0;

  MujocoEnvironment(const py::dict &config, int time_direction_ = 1)
      : GenericEnvironmentBase(config, time_direction_) {
    if (time_direction_ != 1) throw std::runtime_error("mujoco_env: only forward search (no backward dynamics)");
    py::dict mj = config["mujoco"].cast<py::dict>();
    model = mjenv::load_model(mj["model"].cast<std::string>());
    task.init(mj, model);
    int nth = mj.contains("nthread") ? mj["nthread"].cast<int>() : 0;
    if (nth <= 0) nth = std::max(1u, std::thread::hardware_concurrency());
    datas.resize(nth);
    scratch.resize(nth);
    for (int i = 0; i < nth; ++i) {
      datas[i] = mj_makeData(model);
      task.make_scratch(model, scratch[i]);
    }
    pool = std::make_unique<mjenv::ThreadPool>(nth);
    rng.seed(mj.contains("seed") ? mj["seed"].cast<uint64_t>() : 0);
    timesteps = task.policy_steps();
    n_succ = task.num_controls();
  }

  virtual ~MujocoEnvironment() {
    pool.reset();
    for (auto *d : datas) mj_deleteData(d);
    if (model) mj_deleteModel(model);
  }

  // B edge rows: states x [B, n_dims], controls u [B, n_cont] -> traj [B, T, n_dims], valid [B].
  void rollout(const float *x, const float *u, int B, float *traj, char *valid) {
    const int T = task.policy_steps(), S = task.substeps();
    auto t0 = std::chrono::steady_clock::now();
    pool->run(B, [&](int w, int r) {
      mjData *d = datas[w];
      typename Task::Scratch &s = scratch[w];
      const float *x0 = x + (size_t)r * n_dims;
      float *out = traj + (size_t)r * T * n_dims;
      bool ok = task.begin(model, d, x0, u + (size_t)r * n_cont, s);
      const int bad0 = d->warning[mjWARN_BADQACC].number;
      for (int t = 0; t < T; ++t) {
        if (ok) {
          for (int k = 0; k < S; ++k) {
            mj_step1(model, d);
            task.control(model, d, t, k, s);
            mj_step2(model, d);
          }
          ok = task.record(model, d, t, s, out + (size_t)t * n_dims) && ok;
        } else {  // already failed: no point simulating on; keep the row well-formed
          std::memcpy(out + (size_t)t * n_dims, t ? out + (size_t)(t - 1) * n_dims : x0, sizeof(float) * n_dims);
        }
      }
      ok = ok && d->warning[mjWARN_BADQACC].number == bad0;
      ok = ok && task.state_valid(out + (size_t)(T - 1) * n_dims);
      valid[r] = ok ? 1 : 0;
    });
    rollouts += B;
    rollout_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }

  std::vector<std::vector<std::shared_ptr<Node>>> Succ_batched(const std::vector<std::shared_ptr<Node>> &nodes,
                                                               std::shared_ptr<Node> goal) {
    (void)goal;
    const int M = (int)nodes.size();
    if (M == 0) return {};
    const int K = task.num_controls(), T = task.policy_steps();
    std::vector<float> cu((size_t)K * n_cont);
    task.sample_controls(rng, K, cu.data());
    const int B = M * K;
    auto states_t = torch::empty({B, n_dims}, torch::kFloat32);
    auto controls_t = torch::empty({B, n_cont}, torch::kFloat32);
    pack_batch(nodes, cu.data(), K, states_t, controls_t);
    auto traj_t = torch::empty({B, T, n_dims}, torch::kFloat32);
    std::vector<char> ok(B);
    rollout(states_t.data_ptr<float>(), controls_t.data_ptr<float>(), B, traj_t.data_ptr<float>(), ok.data());
    auto final_t = traj_t.select(1, T - 1).contiguous();
    auto cost_t = torch::empty({B}, torch::kFloat32), valid_t = torch::empty({B}, torch::kFloat32),
         h_t = torch::empty({B}, torch::kFloat32);
    const float *xs = states_t.data_ptr<float>(), *us = controls_t.data_ptr<float>(), *xf = final_t.data_ptr<float>();
    float *cp = cost_t.data_ptr<float>(), *vp = valid_t.data_ptr<float>(), *hp = h_t.data_ptr<float>();
    for (int r = 0; r < B; ++r) {
      const float *x1 = xf + (size_t)r * n_dims;
      vp[r] = ok[r] ? 1.f : 0.f;
      cp[r] = ok[r] ? task.cost(xs + (size_t)r * n_dims, us + (size_t)r * n_cont, x1) : 0.f;
      hp[r] = ok[r] ? task.heuristic(x1) : 0.f;
    }
    return make_successors(nodes, cu.data(), K, traj_t, final_t, cost_t, valid_t, h_t);
  }

  std::vector<std::shared_ptr<Node>> Succ(std::shared_ptr<Node> node, std::shared_ptr<Node> goal) {
    auto r = Succ_batched({node}, goal);
    return r.empty() ? std::vector<std::shared_ptr<Node>>{} : r[0];
  }

  float heuristic(float *pose, float *goal) override {
    (void)goal;
    return task.heuristic(pose);
  }

  bool reached_goal_region(std::shared_ptr<Node> v, std::shared_ptr<Node> goal) {
    (void)goal;
    return task.goal(v->pose);
  }

  std::vector<char> reached_goal_region_batched(const std::vector<std::shared_ptr<Node>> &nodes,
                                                std::shared_ptr<Node> goal) {
    (void)goal;
    std::vector<char> out(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) out[i] = task.goal(nodes[i]->pose) ? 1 : 0;
    return out;
  }

  void check_validity(float *start, float *goal, bool *result) {
    result[0] = task.state_valid(start);
    result[1] = task.state_valid(goal);
  }

  void check_validity_batched(const std::vector<float *> &states, std::vector<bool> &results) override {
    results.resize(states.size());
    for (size_t i = 0; i < states.size(); ++i) results[i] = task.state_valid(states[i]);
  }
};
