#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <torch/extension.h>
#include <tuple>
#include <vector>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif
// BEGIN ENVIRONMENT
#if defined(USE_SIMPLE_ENV)
#include <simple.h>
#elif defined(USE_KINEMATIC_ENV)
#include <kinematic.h>
#elif defined(USE_KINEMATIC_CPU_ENV)
#include <kinematic_cpu.h>
#elif defined(USE_KINODYNAMIC_ENV)
#include <kinodynamic.h>
#elif defined(USE_KINODYNAMIC_CPU_ENV)
#include <kinodynamic_cpu.h>
#elif defined(USE_GENERIC_ENV)
#include <generic.h>
#else
#error "No environment macro defined"
#endif
// END ENVIRONMENT
#include "utils/config_utils.h"
#include "utils/sampling_utils.h"
#include <chrono>
#include <iomanip>
#include <pybind11/stl.h>
#include <unordered_map>
#include <unordered_set>

using namespace std;

// Simple barrier implementation for C++11/14/17 compatibility
class SimpleBarrier {
public:
  explicit SimpleBarrier(int count)
      : threshold(count), count(count), generation(0) {}

  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mutex);
    int gen = generation;
    if (--count == 0) {
      generation++;
      count = threshold;
      cv.notify_all();
    } else {
      cv.wait(lock, [this, gen] { return gen != generation; });
    }
  }

  // Arrive without waiting - used when one thread finishes and needs to release
  // the other
  void arrive_and_notify() {
    std::unique_lock<std::mutex> lock(mutex);
    if (--count == 0) {
      generation++;
      count = threshold;
      cv.notify_all();
    }
  }

private:
  std::mutex mutex;
  std::condition_variable cv;
  int threshold;
  int count;
  int generation;
};

class IGHAStar {
public:
  std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>,
                      NodePtrCompare>
      Q_v;
  std::unordered_set<size_t> Q_v_hash; // set of hashes for quick lookup
  std::unordered_set<std::shared_ptr<Node>> inactive_Q_v; // inactive_Qv vector
  std::unordered_set<size_t>
      inactive_Q_v_hash; // set of hashes for quick lookup
  std::unordered_set<std::shared_ptr<Node>> Seen;
  std::unordered_set<size_t> Seen_hash;
  std::vector<std::unordered_map<int, std::shared_ptr<Node>>> V;
  std::vector<std::unordered_map<int, float>> G;

  bool SUCCESS;
  int next_level, level, expansion_limit, expansion_counter, hysteresis;
  int hysteresis_threshold;
  std::vector<std::shared_ptr<Node>> best_path;
  std::vector<std::vector<std::shared_ptr<Node>>> best_path_list;
  float Omega;
  Environment *env;
  bool debug;
  bool run_naive;
  // profiling related variables:
  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point end_time;
  float successor_time;
  float goal_check_time;
  float overhead_time;
  float g_update_time;
  int switches;
  int Q_v_size;
  int max_level_profile;
  std::vector<int> expansion_list;
  std::vector<float> cost_exp_list;

  // Preemptive expansion: optionally batch GPU successor calls for several Q_v
  // vertices into a single kernel launch, caching the results until each
  // vertex is popped. Disabled by default.
  // - min_preemptive: gating threshold. A preemptive batch is only launched
  //   once the stash holds at least this many vertices, so batching only
  //   happens when there is enough work to make it worthwhile.
  // - max_preemptive: hardware cap. Once gated on, at most this many vertices
  //   are expanded preemptively in a single launch.
  bool preemptive_enabled = false;
  int min_preemptive = 0;
  int max_preemptive = 0;
  std::deque<std::shared_ptr<Node>> unexpanded_stash;
  long preemptive_expansions = 0;

  IGHAStar(const py::dict &config, bool debug_ = false,
           int time_direction_ = 1) { // maybe have a debug mode?
    // create environment using config:
    env = new Environment(config, time_direction_);
    debug = debug_;

    // Optional preemptive expansion configuration.
    auto [pe_config, pe_found] =
        config_utils::get_config_dict(config, "preemptive_expansion");
    if (pe_found) {
      preemptive_enabled = pe_config.contains("enabled")
                               ? pe_config["enabled"].cast<bool>()
                               : false;
      max_preemptive = pe_config.contains("max_preemptive")
                           ? pe_config["max_preemptive"].cast<int>()
                           : 0;
      // Gating threshold; defaults to max_preemptive (only batch at full size).
      min_preemptive = pe_config.contains("min_preemptive")
                           ? pe_config["min_preemptive"].cast<int>()
                           : max_preemptive;
    }
    if (preemptive_enabled && max_preemptive > 0) {
      // Clamp the gating threshold into [1, max_preemptive].
      if (min_preemptive < 1) {
        min_preemptive = 1;
      }
      if (min_preemptive > max_preemptive) {
        min_preemptive = max_preemptive;
      }
      // Reserve device capacity for the popped vertex plus max_preemptive
      // preemptively-expanded vertices.
      env->ensure_batch_capacity(max_preemptive + 1);
    } else {
      preemptive_enabled = false;
    }
  }

  ~IGHAStar() {
    if (env) {
      delete env;
      env = nullptr;
    }
  }

  void reset() {
    Q_v = std::priority_queue<std::shared_ptr<Node>,
                              std::vector<std::shared_ptr<Node>>,
                              NodePtrCompare>();
    Q_v_hash.clear();
    inactive_Q_v.clear();
    inactive_Q_v_hash.clear();
    Seen.clear();
    Seen_hash.clear();
    V.clear();
    G.clear();
    best_path.clear();
    best_path_list.clear();
    SUCCESS = false;
    expansion_counter = 0;
    next_level = 0;
    level = 0;
    hysteresis = 0;
    Omega = 1e5;
    // profiler reset:
    successor_time = 0;
    goal_check_time = 0;
    overhead_time = 0;
    g_update_time = 0;
    Q_v_size = 0;
    switches = 0;
    max_level_profile = 0;
    expansion_list.clear();
    cost_exp_list.clear();
    // preemptive expansion reset:
    unexpanded_stash.clear();
    preemptive_expansions = 0;
  }

  // activate method for HA*M
  void naive(std::shared_ptr<Node> start_node) {
    Q_v = std::priority_queue<std::shared_ptr<Node>,
                              std::vector<std::shared_ptr<Node>>,
                              NodePtrCompare>();
    Q_v_hash.clear();
    inactive_Q_v.clear();
    inactive_Q_v_hash.clear();
    Seen.clear();
    Seen_hash.clear();
    V[level].clear();
    G[level].clear();
    G[level][start_node->index[level]] = start_node->g;
    V[level][start_node->index[level]] = start_node;
    Q_v.push(start_node);
    Q_v_hash.insert(start_node->hash);
    // Drop any preemptive caches/stash referring to the previous round; the
    // naive baseline restarts from the start node each round.
    unexpanded_stash.clear();
    start_node->preexpanded = false;
    start_node->cached_succ.clear();
  }

  void initialize(std::shared_ptr<Node> start_node) {
    Q_v.push(start_node);
    Q_v_hash.insert(start_node->hash);
    std::unordered_map<int, std::shared_ptr<Node>> v;
    v[start_node->index[level]] = start_node;
    V.push_back(v);
    std::unordered_map<int, float> g;
    g[start_node->index[level]] = start_node->g;
    G.push_back(g);
  }

  void GUpdate(std::shared_ptr<Node> v) {
    // find the lowest resolution level at which the vertex is dominant
    for (int l = 0; l < level; l++) {
      if (v->g < get_G(l, v->index[l])) {
        v->level = l;
        G[l][v->index[l]] = v->g;
        V[l][v->index[l]] = v;
        break;
      }
    }
    // perform the approximate dominance check and freeze existing vertex if it
    // is dominated
    if (G[level].count(v->index[level]) && V[level].count(v->index[level])) {
      std::shared_ptr<Node> v_p = V[level][v->index[level]];

      // Check presence in Q_v via hash
      if (Q_v_hash.count(v_p->hash) && v_p->active == true) {
        v_p->active = false;
      }
    }
    // Update the \hat{g} \hat{v} structures
    G[level][v->index[level]] = v->g;
    V[level][v->index[level]] = v;
  }

  float get_G(int level, int index) {
    if (G[level].count(index)) {
      return G[level][index];
    } else {
      return 1e5;
    }
  }

  // bubble the active nodes to the top of the Q_v while throwing inactive nodes
  // into inactive_Q_v
  void bubbleActive() {
    while (!Q_v.empty() && Q_v.top()->active != true) {
      std::shared_ptr<Node> v = Q_v.top();
      Q_v.pop();
      Q_v_hash.erase(v->hash);
      inactive_Q_v.insert(v);
      inactive_Q_v_hash.insert(v->hash);
    }
  }

  std::vector<std::shared_ptr<Node>>
  reconstruct_path(std::shared_ptr<Node> start, std::shared_ptr<Node> goal) {
    std::vector<std::shared_ptr<Node>> path;
    std::shared_ptr<Node> v = goal;
    path.push_back(v);
    while (v->parent != nullptr) {
      v = v->parent;
      path.push_back(v);
    }
    return path;
  }

  bool shift(bool run) {
    next_level = level + 1;
    if (run) {
      if (Q_v.top()->level == level) {
        run = true;
      } else {
        hysteresis++;
        if (hysteresis > hysteresis_threshold) {
          next_level = Q_v.top()->level;
          hysteresis = 0;
          run = false;
        }
      }
    }
    return run;
  }

  inline void local_g_update(std::shared_ptr<Node> node) {
    if (node->g < get_G(level, node->index[level])) {
      G[level][node->index[level]] = node->g;
      V[level][node->index[level]] = node;
    }
  }

  void combine_Q_v_and_prune() {
    // combine Q_v and inactive_Q_v
    std::unordered_set<std::shared_ptr<Node>> new_inactive_Q_v;
    std::unordered_set<size_t> new_inactive_Q_v_hash;
    // this step "empties" the Q_v, so everything is now only in the inactive
    // set.
    while (!Q_v.empty()) {
      std::shared_ptr<Node> node = Q_v.top();
      Q_v.pop();
      Q_v_hash.erase(node->hash); // always remember to erase the hash!
      node->active = false;
      if (node->f < Omega) {
        new_inactive_Q_v.insert(node);
        new_inactive_Q_v_hash.insert(node->hash);
      }
    }
    for (std::shared_ptr<Node> node : inactive_Q_v) {
      if (node->f < Omega) {
        // put it into new_inactive_Q_v and new inactive_Q_v_hash
        new_inactive_Q_v.insert(node);
        new_inactive_Q_v_hash.insert(node->hash);
      }
    }
    // now delete the old inactive_Q_v and inactive_Q_v_hash
    inactive_Q_v.clear();
    inactive_Q_v_hash.clear();
    inactive_Q_v = new_inactive_Q_v;
    inactive_Q_v_hash = new_inactive_Q_v_hash;
  }

  void change_resolution_g_update() {
    level = next_level;

    // // Extend G and V if new level hasn't been seen yet
    if (G.size() <= level) {
      G.emplace_back(); // std::unordered_map<int, float>
      V.emplace_back(); // std::unordered_map<int, std::shared_ptr<Node>>
      // reserve 100000 for both:
      G[level].reserve(100000); // consider pre-reserving this memeory at the
                                // beginning of the search.
      V[level].reserve(100000);
    }

    for (std::shared_ptr<Node> node : Seen) {
      // project Seen nodes to the new resolution
      local_g_update(node);
    }
    // find rank of all nodes in Q_v
    std::unordered_map<int, std::vector<std::shared_ptr<Node>>> grid_cells;
    for (std::shared_ptr<Node> node : inactive_Q_v) {
      grid_cells[node->index[level]].push_back(node);
      // project Q_v nodes to the new resolution
      local_g_update(node);
    }

    for (auto &[idx, nodes] : grid_cells) {
      std::sort(nodes.begin(), nodes.end(),
                [](std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
                  return a->g < b->g;
                });

      for (int i = 0; i < nodes.size(); i++) {
        nodes[i]->rank = i;
      }
    }
  }

  void Activate() {
    // for performance reasons, we are branching-bounding the nodes in the
    // activate method.
    std::vector<std::shared_ptr<Node>> unsorted_Q_v;
    for (auto it = inactive_Q_v.begin(); it != inactive_Q_v.end();) {
      std::shared_ptr<Node> node = *it;
      node->level = level;
      if (node->rank == 0 && node->g <= get_G(level, node->index[level])) {
        node->active = true;
        unsorted_Q_v.push_back(node);
        Q_v_hash.insert(node->hash);
        it = inactive_Q_v.erase(it); // erase returns the next valid iterator
        inactive_Q_v_hash.erase(node->hash);
        G[level][node->index[level]] = node->g;
        V[level][node->index[level]] = node;
      } else {
        ++it; // move to the next element
      }
    }
    // // sort the Q_v based on f value
    std::make_heap(unsorted_Q_v.begin(), unsorted_Q_v.end(),
                   NodePtrCompare{}); // make a sorted heap from unsorted_Q_v
    // // set the Q_v to the sorted heap
    Q_v =
        std::priority_queue<std::shared_ptr<Node>,
                            std::vector<std::shared_ptr<Node>>, NodePtrCompare>(
            NodePtrCompare{}, std::move(unsorted_Q_v));
  }

  // Run the tolerance + approximate-dominance check on a freshly retrieved
  // successor list and route each survivor into Q_v (active) or inactive_Q_v.
  // When preemptive expansion is enabled, every newly discovered vertex is also
  // appended to the unexpanded_stash so it can be preemptively expanded later.
  void process_successors(std::shared_ptr<Node> v,
                          std::vector<std::shared_ptr<Node>> &neighbors) {
    (void)v;
    for (auto &neighbor : neighbors) {
      // tolerance check is useful in practice to eliminate vertices that
      // are too close to be practically distinguishable (closer than map
      // resolution) you can disable this by setting the tolerance to a very
      // small value (1e-5), but it is not recommended.
      bool tolerance_check = Q_v_hash.count(neighbor->hash) == 0 &&
                             Seen_hash.count(neighbor->hash) == 0 &&
                             inactive_Q_v_hash.count(neighbor->hash) == 0;
      if (tolerance_check) {
        // approximate dominance check:
        if (neighbor->g < get_G(level, neighbor->index[level])) {
          GUpdate(neighbor);
          neighbor->active = true;
          Q_v.push(neighbor);
          Q_v_hash.insert(neighbor->hash);
        } else {
          neighbor->active = false;
          inactive_Q_v.insert(neighbor);
          inactive_Q_v_hash.insert(neighbor->hash);
        }
        if (preemptive_enabled) {
          // Both active and inactive newly-discovered vertices are eligible for
          // preemptive expansion in a later iteration.
          unexpanded_stash.push_back(neighbor);
        }
      }
    }
  }

  // Returns the successors of v while opportunistically expanding up to
  // max_preemptive stash vertices in the same GPU kernel launch. If v was
  // already preemptively expanded its successors are served from cache (no GPU
  // work for v); otherwise v is expanded live as part of the batch. The
  // preemptive batch is only launched once the stash holds at least
  // min_preemptive vertices (so batching only happens when it is worthwhile),
  // and is then capped at max_preemptive vertices (the hardware limit).
  std::vector<std::shared_ptr<Node>>
  expand_with_preemption(std::shared_ptr<Node> v,
                         std::shared_ptr<Node> goal_node) {
    bool v_needs_expand = !v->preexpanded;
    // Only spend a batched GPU launch on preemptive work once the stash has
    // accumulated at least min_preemptive vertices. Doing a preemptive batch
    // with only a handful of nodes would underutilize the GPU, so below that
    // threshold we defer: a cache hit returns immediately with no GPU work, and
    // a cache miss expands only v.
    bool do_preempt =
        static_cast<int>(unexpanded_stash.size()) >= min_preemptive;

    if (!v_needs_expand && !do_preempt) {
      // Cache hit while the stash is still filling: serve the cached successors
      // immediately without launching the GPU.
      std::vector<std::shared_ptr<Node>> v_successors =
          std::move(v->cached_succ);
      v->cached_succ.clear();
      return v_successors;
    }

    std::vector<std::shared_ptr<Node>> batch;
    std::vector<std::shared_ptr<Node>> preempt_nodes;
    if (v_needs_expand) {
      batch.push_back(v);
    }
    // Gather up to max_preemptive not-yet-expanded vertices from the stash, but
    // only when the stash is full enough to make a batched launch worthwhile.
    while (do_preempt &&
           static_cast<int>(preempt_nodes.size()) < max_preemptive &&
           !unexpanded_stash.empty()) {
      std::shared_ptr<Node> p = unexpanded_stash.front();
      unexpanded_stash.pop_front();
      // Skip: v (already handled above), already-expanded vertices, and
      // vertices no longer promising (would be pruned anyway).
      if (p == v || p->preexpanded || p->f >= Omega) {
        continue;
      }
      preempt_nodes.push_back(p);
      batch.push_back(p);
    }

    std::vector<std::shared_ptr<Node>> v_successors;
    if (!batch.empty()) {
      std::vector<std::vector<std::shared_ptr<Node>>> results =
          env->Succ_batched(batch, goal_node);
      int idx = 0;
      if (v_needs_expand) {
        v_successors = std::move(results[idx]);
        // v has been popped already, so we don't cache its successors; just
        // flag it so any stale stash copy is skipped.
        v->preexpanded = true;
        idx++;
      }
      for (auto &p : preempt_nodes) {
        p->cached_succ = std::move(results[idx]);
        p->preexpanded = true;
        idx++;
        preemptive_expansions++;
      }
    }
    if (!v_needs_expand) {
      // Cache hit: consume and release the cached successors.
      v_successors = std::move(v->cached_succ);
      v->cached_succ.clear();
    }
    return v_successors;
  }

  // function to do hybrid A* search and return a path as a tensor I guess:
  void search(float *start, float *goal, int expansion_limit_,
              int hysteresis_threshold_, bool ignore_goal_validity) {
    reset();
    expansion_limit = expansion_limit_;
    if (hysteresis_threshold_ >= 0) {
      hysteresis_threshold = hysteresis_threshold_;
      run_naive = false;
    } else {
      run_naive = true;
      hysteresis_threshold = expansion_limit + 1;
    }

    bool valid[2] = {true, true};
    env->check_validity(start, goal, valid);
    // check start goal validity:
    if (debug) {
      std::cout << "Start node validity: " << valid[0] << std::endl;
      std::cout << "Goal node validity: " << valid[1] << std::endl;
    }
    if (!valid[0]) {
      std::cout << "Start node is not valid" << std::endl;
      return;
    }
    if (!ignore_goal_validity) {
      if (!valid[1]) {
        std::cout << "Goal node is not valid" << std::endl;
        return;
      }
    }
    std::shared_ptr<Node> start_node = env->create_Node(start);
    std::shared_ptr<Node> goal_node = env->create_Node(goal);
    start_node->f = env->heuristic(start_node->pose, goal_node->pose);

    initialize(start_node);

    // main search loop:
    bool goal_reached = false;

    while (expansion_counter < expansion_limit &&
           (!Q_v.empty() or !inactive_Q_v.empty())) {
      int inactive_insertions = 0;
      while (expansion_counter < expansion_limit) {
        // inner time measurement:
        start_time = std::chrono::high_resolution_clock::now();
        bubbleActive();
        bool run =
            !Q_v.empty() && Q_v.top()->active == true && Q_v.top()->f < Omega;
        bool shift_run = shift(run); // update next candidate resolution and
                                     // whether we should continue running
        if (!run) {
          break;
        }

        std::shared_ptr<Node> v = Q_v.top();
        Q_v.pop();
        Q_v_hash.erase(v->hash);
        Seen.insert(v);
        Seen_hash.insert(v->hash);

        goal_reached = env->reached_goal_region(v, goal_node);

        if (goal_reached) {
          SUCCESS = true;
          if (debug) {
            std::cout << "expansions: " << expansion_counter
                      << " cost: " << v->g << std::endl;
          }
          best_path = reconstruct_path(start_node, v);
          Omega = v->g;
          best_path_list.push_back(best_path);
          goal_reached =
              false; // set flag to false to continue searching for better paths
          break;
        }
        if (!shift_run) {
          break;
        }
        // profiling:
        end_time = std::chrono::high_resolution_clock::now();
        goal_check_time +=
            std::chrono::duration<float, std::micro>(end_time - start_time)
                .count();

        start_time = std::chrono::high_resolution_clock::now();
        std::vector<std::shared_ptr<Node>> neighbors;
        if (preemptive_enabled) {
          neighbors = expand_with_preemption(v, goal_node);
        } else {
          neighbors = env->Succ(v, goal_node);
        }
        end_time = std::chrono::high_resolution_clock::now();
        successor_time +=
            std::chrono::duration<float, std::micro>(end_time - start_time)
                .count();

        expansion_counter++;

        start_time = std::chrono::high_resolution_clock::now();
        // get successors:
        process_successors(v, neighbors);
        end_time = std::chrono::high_resolution_clock::now();
        g_update_time +=
            std::chrono::duration<float, std::micro>(end_time - start_time)
                .count();
        cost_exp_list.push_back(Omega);
      }
      start_time = std::chrono::high_resolution_clock::now();
      combine_Q_v_and_prune();      // puts inactive_Q_v into Q_v
      change_resolution_g_update(); // updates G and V
      Activate();                   // activates the nodes in Q_v
      end_time = std::chrono::high_resolution_clock::now();

      // overhead_time += std::chrono::duration<float, std::micro>(end_time -
      // start_time).count();
      overhead_time +=
          std::chrono::duration<float, std::micro>(end_time - start_time)
              .count() /
          (Q_v.size() + inactive_Q_v.size() + Seen.size());
      Q_v_size += Q_v.size();
      switches++;
      expansion_list.push_back(expansion_counter);
      if (Q_v.empty() && inactive_Q_v.empty()) {
        cost_exp_list.push_back(Omega);
        break;
      }
      if (run_naive) {
        naive(start_node);
      }
      max_level_profile = std::max(max_level_profile, level);
      if (debug) {
        std::printf("Expansions: %d, level %d, Q_v: %d, Seen: %d \n",
                    expansion_counter, level, int(Q_v.size()),
                    int(Seen.size()));
        std::cout << "Size of inactive_Q_v: " << inactive_Q_v.size()
                  << " inactive insertions: " << inactive_insertions
                  << std::endl;
      }
    }
  }

  bool search_adapter(torch::Tensor start_tensor, torch::Tensor goal_tensor,
                      torch::Tensor world_tensor, int expansion_limit,
                      int hysteresis_threshold,
                      bool ignore_goal_validity = false) {
    env->set_world(world_tensor);
    auto start = start_tensor.contiguous().data_ptr<float>();
    auto goal = goal_tensor.contiguous().data_ptr<float>();
    search(start, goal, expansion_limit, hysteresis_threshold,
           ignore_goal_validity);
    if (debug) {
      std::cout << "Search finished" << expansion_counter << std::endl;
    }
    return SUCCESS;
  }

  std::vector<torch::Tensor> get_path_list() {
    std::vector<torch::Tensor> path_list;
    for (int i = 0; i < best_path_list.size(); i++) {
      auto path_tensor =
          env->convert_node_list_to_path_tensor(best_path_list[i]);
      path_list.push_back(path_tensor);
    }
    return path_list;
  }
  // get only the best path:
  torch::Tensor get_best_path() {
    auto path_tensor = env->convert_node_list_to_path_tensor(best_path);
    return path_tensor;
  }

  // get the profiler info, I want to get the averages for timing, Q_v size, and
  // number of switches:
  std::tuple<float, float, float, float, int, int, int, int, std::vector<int>,
             std::vector<float>>
  get_profiler_info() {
    float avg_successor_time = successor_time / expansion_counter;
    float avg_goal_check_time = goal_check_time / expansion_counter;
    float avg_overhead_time = overhead_time / switches;
    float avg_g_update_time = g_update_time / expansion_counter;
    return std::make_tuple(avg_successor_time, avg_goal_check_time,
                           avg_overhead_time, avg_g_update_time, switches,
                           max_level_profile, Q_v_size, expansion_counter,
                           expansion_list, cost_exp_list);
  }

  // Number of vertices expanded preemptively (cached) during the last search.
  long get_preemptive_expansions() { return preemptive_expansions; }
};

class BiIGHAStar {
public:
  IGHAStar *forward_search;
  IGHAStar *backward_search;
  bool debug;

  // Shared state protected by mutex
  std::mutex state_mutex;
  std::atomic<float> shared_Omega{1e5f};
  std::atomic<int> total_expansions{0};
  std::atomic<bool> search_complete{false};
  std::atomic<bool> forward_finished{false};
  std::atomic<bool> backward_finished{false};

  // LCR_table data structures for near-meet detection (protected by LCR_mutex)
  // Each LCR_table cell stores a list of (g-value, node) pairs for multiple
  // vertices
  std::mutex LCR_mutex;
  std::unordered_map<int, std::vector<std::pair<float, std::shared_ptr<Node>>>>
      forward_LCR_table;
  std::unordered_map<int, std::vector<std::pair<float, std::shared_ptr<Node>>>>
      backward_LCR_table;

  // Point perturbation configuration for near-meet validity checking
  int num_interpolation_points;
  int num_perturbations;
  std::vector<std::vector<float>> cached_perturbations;
  float perturbation_scale;

  // Goal sampling configuration for backward search
  // Goal sampling is needed in practice because your exact goal state may be
  // unreachable. Sampling around the goal allows the backward search to start
  // from reachable states within the goal region.
  sampling::GoalSamplingConfig goal_sampling_config;

  // Best path storage
  std::mutex path_mutex;
  std::vector<std::shared_ptr<Node>> best_path;
  std::vector<std::vector<std::shared_ptr<Node>>> best_path_list;

  // Profiling lists mutex
  std::mutex profiling_lists_mutex;

  // Synchronization barrier for parallel expansion
  std::unique_ptr<SimpleBarrier> expansion_barrier;

  // Profiling related variables (aggregated from both searches)
  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point end_time;
  float successor_time;
  float goal_check_time;
  float overhead_time;
  float g_update_time;
  int switches;
  int max_level_profile;
  std::vector<int> expansion_list;
  std::vector<float> cost_exp_list;

  BiIGHAStar(const py::dict &config, bool debug_ = false) : debug(debug_) {
    forward_search = new IGHAStar(config, debug_, 1);
    backward_search = new IGHAStar(config, debug_, -1);

    setup_near_meet(config);
    setup_goal_sampling(config);
    initialize_perturbation_cache();
  }

  void setup_near_meet(const py::dict &config) {
    auto [nm_config, found] =
        config_utils::get_config_dict(config, "near_meet_config");
    num_interpolation_points =
        nm_config.contains("num_interpolation_points")
            ? nm_config["num_interpolation_points"].cast<int>()
            : 5;
    num_perturbations = nm_config.contains("num_perturbations")
                            ? nm_config["num_perturbations"].cast<int>()
                            : 3;
    perturbation_scale = nm_config.contains("perturbation_scale")
                             ? nm_config["perturbation_scale"].cast<float>()
                             : 1.0f;
    initialize_perturbation_cache();
  }

  void setup_goal_sampling(const py::dict &config) {
    auto [gs_config, found] =
        config_utils::get_config_dict(config, "goal_sampling_config");

    goal_sampling_config.n_dims = n_dims;
    goal_sampling_config.enabled = gs_config.contains("enabled")
                                       ? gs_config["enabled"].cast<bool>()
                                       : true;
    goal_sampling_config.num_samples =
        gs_config.contains("num_samples") ? gs_config["num_samples"].cast<int>()
                                          : 32;

    // Parse angular_dims as a list (supports multiple angular dimensions, e.g.,
    // for multi-car systems)
    if (gs_config.contains("angular_dims")) {
      goal_sampling_config.angular_dims =
          gs_config["angular_dims"].cast<std::vector<int>>();
    } else {
      goal_sampling_config.angular_dims = {
          2}; // default: dimension 2 is angular
    }

    std::string space_type_str =
        gs_config.contains("space_type")
            ? gs_config["space_type"].cast<std::string>()
            : "SE2";
    if (space_type_str == "SE2" || space_type_str == "se2") {
      goal_sampling_config.space_type = sampling::SpaceType::SE2;
    } else if (space_type_str == "R_N" || space_type_str == "r_n" ||
               space_type_str == "RN" || space_type_str == "rn") {
      goal_sampling_config.space_type = sampling::SpaceType::R_N;
    } else {
      goal_sampling_config.space_type = sampling::SpaceType::SE2;
    }

    // Parse sigma vector (defaults to LCR if not specified)
    goal_sampling_config.sigma.resize(n_dims);
    const float *LCR = forward_search->env->local_controllability_radius;
    if (gs_config.contains("sigma")) {
      auto sigma_vec = gs_config["sigma"].cast<std::vector<float>>();
      for (int i = 0; i < n_dims; i++) {
        goal_sampling_config.sigma[i] =
            (i < static_cast<int>(sigma_vec.size())) ? sigma_vec[i] : LCR[i];
      }
    } else {
      for (int i = 0; i < n_dims; i++) {
        goal_sampling_config.sigma[i] = LCR[i];
      }
    }
  }

  void initialize_perturbation_cache() {
    // This is not part of the core Bi-IGHAStar method; we are simply putting it
    // inside the class because it is convenient to pass the function pointers
    // to the sampling utility function, which is defined elsewhere.)
    const float *LCR = forward_search->env->local_controllability_radius;
    cached_perturbations = sampling::generate_perturbation_cache(
        LCR, n_dims, num_perturbations, perturbation_scale);

    if (debug) {
      std::cout << "Cached perturbations: " << cached_perturbations.size()
                << std::endl;
      for (size_t i = 0; i < cached_perturbations.size(); i++) {
        std::cout << "Perturbation " << i << ": ";
        for (int d = 0; d < n_dims; d++) {
          std::cout << cached_perturbations[i][d] << " ";
        }
      }
      std::cout << std::endl;
    }
  }

  void sample_goals(std::shared_ptr<Node> backward_start,
                    std::shared_ptr<Node> backward_goal,
                    std::ofstream &backward_log) {
    // Define callbacks for the sampling utility
    auto validity_checker = [this](const std::vector<float *> &states,
                                   std::vector<bool> &results) {
      std::vector<float *> non_const_states(states.begin(), states.end());
      backward_search->env->check_validity_batched(non_const_states, results);
    };

    auto heuristic_fn = [this](const float *p1, const float *p2) {
      return backward_search->env->heuristic(const_cast<float *>(p1),
                                             const_cast<float *>(p2));
    };

    auto node_factory = [this, &backward_start](float *pose,
                                                float *intermediate_poses,
                                                float g_value, float f_value) {
      auto node = std::make_shared<Node>(
          pose, intermediate_poses, backward_start, backward_start->g + g_value,
          backward_search->env->resolution, backward_search->env->tolerance,
          backward_search->env->max_level,
          backward_search->env->division_factor,
          backward_search->env->timesteps,
          backward_search->env->local_controllability_radius,
          backward_search->env->time_direction);
      node->f = backward_start->g + f_value;
      return node;
    };

    auto near_meet_checker = [this, &backward_start, &backward_log](
                                 const std::shared_ptr<Node> &node) {
      return check_near_meet_validity(backward_search, node, backward_start,
                                      backward_log);
    };

    // Sample goal region using utility function, then, filter the sampled nodes
    // using the near-meet checker we only consider those sampled states that
    // satisfy the near-meet checker, as they are the ones that are likely to be
    // valid. This function is not part of the core Bi-IGHAStar method; we are
    // simply putting it inside the class because it is convenient to pass the
    // function pointers to the sampling utility function, which is defined
    // elsewhere.
    auto sampled_nodes = sampling::sample_goal_region<std::shared_ptr<Node>>(
        backward_start->pose, backward_goal->pose, goal_sampling_config,
        backward_search->env->timesteps, validity_checker, heuristic_fn,
        node_factory, near_meet_checker);

    if (debug) {
      int num_sampled_nodes = sampled_nodes.size();
      if (num_sampled_nodes > 2) {
        float mean[n_dims];
        float variance[n_dims];
        for (int i = 0; i < n_dims; i++) {
          mean[i] = 0.0f;
          variance[i] = 0.0f;
        }
        for (int i = 0; i < num_sampled_nodes; i++) {
          for (int j = 0; j < n_dims; j++) {
            mean[j] += sampled_nodes[i]->pose[j] / num_sampled_nodes;
          }
        }
        for (int i = 0; i < num_sampled_nodes; i++) {
          for (int j = 0; j < n_dims; j++) {
            float delta = sampled_nodes[i]->pose[j] - backward_start->pose[j];
            variance[j] += delta * delta / num_sampled_nodes;
          }
        }
        std::cout << "Mean of sampled nodes: ";
        for (int i = 0; i < n_dims; i++) {
          std::cout << mean[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Variance of sampled nodes: ";
        for (int i = 0; i < n_dims; i++) {
          std::cout << variance[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Individual nodes: around start: "
                  << backward_start->pose[0] << ", " << backward_start->pose[1]
                  << ", " << backward_start->pose[2] << ", "
                  << backward_start->pose[3] << ":  " << std::endl
                  << "[";
        for (int i = 0; i < num_sampled_nodes; i++) {
          std::cout << "["
                    << sampled_nodes[i]->pose[0] - backward_start->pose[0]
                    << ", "
                    << sampled_nodes[i]->pose[1] - backward_start->pose[1]
                    << ", "
                    << sampled_nodes[i]->pose[2] - backward_start->pose[2]
                    << ", "
                    << sampled_nodes[i]->pose[3] - backward_start->pose[3]
                    << "]," << std::endl;
        }
        std::cout << "]" << std::endl;
      } else {
        std::cout << "Only " << num_sampled_nodes
                  << " sampled nodes found, skipping variance computation"
                  << std::endl;
      }
    }

    // Add valid sampled states to backward search
    for (std::shared_ptr<Node> sampled_node : sampled_nodes) {
      if (backward_search->Q_v_hash.count(sampled_node->hash) > 0) {
        continue; // Skip duplicate
      }
      if (sampled_node->g <
          backward_search->get_G(backward_search->level,
                                 sampled_node->index[backward_search->level])) {
        backward_search->GUpdate(sampled_node);
      }
      sampled_node->active = true;
      backward_search->Q_v.push(sampled_node);
      backward_search->Q_v_hash.insert(sampled_node->hash);

      int LCR_idx = sampled_node->LCR_index;
      update_LCR_table(false, sampled_node, LCR_idx);
    }
  }

  ~BiIGHAStar() {
    delete forward_search;
    delete backward_search;
  }

  void reset_LCR_table(IGHAStar *search, std::shared_ptr<Node> start_node) {
    // Initialize LCR_table for this search direction
    if (search == forward_search) {
      forward_LCR_table.clear();
      int LCR_idx = start_node->LCR_index;
      forward_LCR_table[LCR_idx].push_back({0.0f, start_node});
    } else {
      backward_LCR_table.clear();
      int LCR_idx = start_node->LCR_index;
      backward_LCR_table[LCR_idx].push_back({0.0f, start_node});
    }
  }

  // Update LCR_table: add node to the list for this LCR cell
  void update_LCR_table(bool is_forward, std::shared_ptr<Node> v, int LCR_idx) {
    std::lock_guard<std::mutex> lock(LCR_mutex);
    auto &LCR_table = is_forward ? forward_LCR_table : backward_LCR_table;
    // Add this node to the list (multiple nodes per cell allowed)
    LCR_table[LCR_idx].push_back({v->g, v});
  }

  // Check if connection between two meeting nodes is valid by interpolating
  // points between the two vertices, and then perturbing those points around
  // the interpolated points for a more robust check, the interpolation and
  // perturbation are done using the sampling_utils library Returns true if all
  // interpolated + perturbed states are collision-free
  bool check_near_meet_validity(IGHAStar *search,
                                std::shared_ptr<Node> v_forward,
                                std::shared_ptr<Node> v_backward,
                                std::ostream &log_stream) {
    if (num_interpolation_points == 0 && num_perturbations == 0) {
      return true;
    }

    auto validity_checker = [search](const std::vector<float *> &states,
                                     std::vector<bool> &results) {
      std::vector<float *> non_const_states(states.begin(), states.end());
      search->env->check_validity_batched(non_const_states, results);
    };

    bool valid = sampling::check_interpolation_validity(
        v_forward->pose, v_backward->pose, n_dims,
        goal_sampling_config.angular_dims, num_interpolation_points,
        cached_perturbations, validity_checker);

    if (debug) {
      int num_states = (num_interpolation_points - 1) * (1 + num_perturbations);
      if (valid) {
        log_stream << "Point perturbation check PASSED - " << num_states
                   << " states checked" << std::endl;
      } else {
        log_stream << "Point perturbation check FAILED" << std::endl;
      }
    }
    return valid;
  }

  // Check for near-meet with the other search direction
  // this function takes all the vertices from the opposite direction that are
  // within LCR of the current vertex, then arrange them by increasing total
  // cost, then check the validity of the connection, the first valid connection
  // is returned. Returns: (found, path_cost, path) if a valid connection was
  // found
  std::tuple<bool, float, std::vector<std::shared_ptr<Node>>>
  NearMeet(IGHAStar *this_search, IGHAStar *other_search,
           std::shared_ptr<Node> v, std::shared_ptr<Node> this_start,
           std::shared_ptr<Node> other_start, bool is_forward,
           std::ostream &log_stream) {
    int LCR_idx = v->LCR_index;

    std::lock_guard<std::mutex> lock(LCR_mutex);
    auto &other_LCR_table = is_forward ? backward_LCR_table : forward_LCR_table;

    if (!other_LCR_table.count(LCR_idx) || other_LCR_table[LCR_idx].empty()) {
      return {false, 0.0f, {}};
    }

    // Collect promising candidate pairs (cost < Omega) and sort by cost
    float current_omega = shared_Omega.load();
    std::vector<std::tuple<float, std::shared_ptr<Node>, float>>
        candidates; // (path_cost, v_other, dist)

    for (const auto &pair : other_LCR_table[LCR_idx]) {
      float g_other = pair.first;
      std::shared_ptr<Node> v_other = pair.second;
      float dist = compute_near_meet_distance(this_search, v, v_other);
      float path_cost = v->g + g_other + dist;

      if (path_cost < current_omega) {
        candidates.push_back({path_cost, v_other, dist});
      }
    }

    if (candidates.empty()) {
      if (debug) {
        log_stream << "No candidates below Omega threshold for LCR_idx "
                   << LCR_idx << std::endl;
      }
      return {false, 0.0f, {}};
    }

    // Sort candidates by path cost (best first)
    std::sort(candidates.begin(), candidates.end(),
              [](const auto &a, const auto &b) {
                return std::get<0>(a) < std::get<0>(b);
              });

    if (debug) {
      log_stream << "Checking " << candidates.size()
                 << " candidates for LCR_idx " << LCR_idx << std::endl;
    }

    // Try candidates in order of cost until we find a valid one
    for (const auto &candidate : candidates) {
      float path_cost = std::get<0>(candidate);
      std::shared_ptr<Node> v_other = std::get<1>(candidate);

      // Re-check against current Omega (may have been updated by other thread)
      if (path_cost >= shared_Omega.load()) {
        continue; // Skip if no longer promising
      }

      // Determine forward and backward meeting nodes
      std::shared_ptr<Node> forward_meeting = is_forward ? v : v_other;
      std::shared_ptr<Node> backward_meeting = is_forward ? v_other : v;

      // Perform point perturbation validity check
      if (!check_near_meet_validity(this_search, forward_meeting,
                                    backward_meeting, log_stream)) {
        if (debug) {
          log_stream << "Candidate failed validity check, trying next..."
                     << std::endl;
        }
        continue; // Try next candidate
      }

      // Valid connection found! Build the path
      std::vector<std::shared_ptr<Node>> forward_path, backward_path;

      if (is_forward) {
        forward_path = this_search->reconstruct_path(this_start, v);
        backward_path = other_search->reconstruct_path(other_start, v_other);
      } else {
        backward_path = this_search->reconstruct_path(this_start, v);
        forward_path = other_search->reconstruct_path(other_start, v_other);
      }

      // Build path in order [goal, ..., backward_meeting, forward_meeting, ...,
      // start]
      std::vector<std::shared_ptr<Node>> backward_path_reversed;
      for (auto it = backward_path.rbegin(); it != backward_path.rend(); ++it) {
        backward_path_reversed.push_back(*it);
      }

      // Safety check: warn if paths are very long
      if (backward_path.size() > 10000 || forward_path.size() > 10000) {
        log_stream << "WARNING: Very long paths detected! backward_path="
                   << backward_path.size()
                   << ", forward_path=" << forward_path.size() << std::endl;
      }

      // Combine paths
      std::vector<std::shared_ptr<Node>> connected_path;
      connected_path.reserve(backward_path_reversed.size() +
                             forward_path.size());
      for (auto &node : backward_path_reversed) {
        connected_path.push_back(node);
      }
      for (auto &node : forward_path) {
        connected_path.push_back(node);
      }

      if (debug) {
        log_stream << "near meet found at " << total_expansions.load()
                   << " expansions, cost: " << path_cost << std::endl;
        log_stream << std::endl;
        log_stream.flush();
      }
      return {true, path_cost, connected_path};
    }

    // All candidates failed validity check
    if (debug) {
      log_stream << "All " << candidates.size()
                 << " candidates failed validity check" << std::endl;
    }
    return {false, 0.0f, {}};
  }

  // Compute distance between two nodes for near-meet check
  float compute_near_meet_distance(IGHAStar *search, std::shared_ptr<Node> v1,
                                   std::shared_ptr<Node> v2) {
    return search->env->compute_near_meet_distance(v1->pose, v2->pose);
  }

  // Worker function for one search direction
  // Structure mirrors IGHAStar::search() with added synchronization and
  // near-meet detection
  void search_worker(IGHAStar *search, IGHAStar *other_search,
                     std::shared_ptr<Node> start_node,
                     std::shared_ptr<Node> goal_node,
                     std::shared_ptr<Node> other_start_node,
                     int expansion_limit, int hysteresis_threshold,
                     bool is_forward, std::ostream &log_stream) {
    auto &my_finished = is_forward ? forward_finished : backward_finished;
    auto &other_finished = is_forward ? backward_finished : forward_finished;
    const char *prefix = is_forward ? "[F]" : "[B]";
    int level_limit = 1000; // some very large number

    if (debug) {
      log_stream << "level limit: " << level_limit
                 << " search level: " << search->level << std::endl;
    }
    // print direction of search and the top of the Q_v:
    if (!search->Q_v.empty() and debug) {
      log_stream << prefix << " Direction of search: "
                 << (is_forward ? "forward" : "backward")
                 << " Top of the Q_v: " << search->Q_v.top()->pose[0] << ", "
                 << search->Q_v.top()->pose[1] << ", "
                 << search->Q_v.top()->pose[2] << ", "
                 << search->Q_v.top()->pose[3] << "\n";
      log_stream.flush();
    }
    std::vector<std::shared_ptr<Node>> best_path_local;
    // Outer loop: mirrors IGHAStar's outer while loop
    // while (expansion_counter < expansion_limit && (!Q_v.empty() or
    // !inactive_Q_v.empty()))
    if (debug) {
      log_stream << "entering outer loop" << std::endl;
    }
    while (total_expansions.load() < expansion_limit &&
           (!search->Q_v.empty() || !search->inactive_Q_v.empty()) &&
           !search_complete.load() && search->level <= level_limit) {
      // Inner loop: mirrors IGHAStar's inner while loop
      // while (expansion_counter < expansion_limit) { if (!shift()) break; ...
      // }
      if (debug) {
        log_stream << "entering inner loop" << std::endl;
      }
      while (total_expansions.load() < expansion_limit &&
             !search_complete.load() && search->level <= level_limit) {
        if (debug) {
          log_stream << "arrive and wait" << std::endl;
        }
        // === SYNCHRONIZATION POINT: Wait for both threads before each
        // expansion === If the other search has finished, skip barrier wait and
        // continue
        if (!other_finished.load()) {
          expansion_barrier->arrive_and_wait();
        }
        if (debug) {
          log_stream << "check termination conditions" << std::endl;
        }
        // Check termination conditions after synchronization
        if (search_complete.load())
          break;
        if (my_finished.load() && other_finished.load()) {
          search_complete.store(true);
          break;
        }
        if (my_finished.load())
          continue;

        // Use shift() exactly as in IGHAStar - this handles bubbleActive,
        // run condition check, and hysteresis
        if (debug) {
          log_stream << "sync Omega before shift" << std::endl;
        }
        search->Omega = shared_Omega.load(); // sync Omega before shift
        if (debug) {
          log_stream << "check shift condition" << std::endl;
        }
        search->bubbleActive();
        bool run = !search->Q_v.empty() && search->Q_v.top()->active == true &&
                   search->Q_v.top()->f < search->Omega;
        if (!search->shift(run)) {
          break; // exit inner loop for level change
        }
        if (!run) {
          break;
        }

        // Timing measurement start
        start_time = std::chrono::high_resolution_clock::now();
        if (debug) {
          log_stream << "start time measurement" << std::endl;
        }
        // Pop node - mirrors IGHAStar
        std::shared_ptr<Node> v = search->Q_v.top();
        // print pose of v:
        if (debug) {
          log_stream << "pop node" << std::endl;
        }
        search->Q_v.pop();
        search->Q_v_hash.erase(v->hash);
        search->Seen.insert(v);
        search->Seen_hash.insert(v->hash);
        if (debug) {
          log_stream << "insert node into Seen" << std::endl;
        }
        // Goal check - mirrors IGHAStar
        bool goal_reached = search->env->reached_goal_region(v, goal_node);
        if (goal_reached and not is_forward) {
          if (not check_near_meet_validity(search, v, start_node, log_stream)) {
            if (debug) {
              log_stream << "near meet is not valid, skipping goal reached"
                         << std::endl;
            }
            goal_reached = false;
          }
        }
        if (goal_reached) {
          if (debug) {
            log_stream << "goal reached" << std::endl;
          }

          search->SUCCESS = true;
          float new_cost = v->f;

          if (new_cost < shared_Omega.load()) {
            if (debug) {
              log_stream << "update Omega" << std::endl;
            }
            shared_Omega.store(new_cost);
            search->Omega = new_cost;

            if (debug) {
              log_stream << "lock path mutex" << std::endl;
            }
            best_path_local = search->reconstruct_path(start_node, v);
            if (not is_forward) {
              // reverse the best_path_local:
              std::reverse(best_path_local.begin(), best_path_local.end());
            }
            {
              std::lock_guard<std::mutex> lock(path_mutex);
              best_path_list.push_back(best_path_local);
            }
            if (debug) {
              log_stream << "push best path" << std::endl;

              log_stream << prefix << " [exp:" << total_expansions.load()
                         << " lvl:" << search->level
                         << "] Found goal! cost: " << new_cost
                         << " g value: " << v->g
                         << " heuristic value: " << v->f - v->g << "\n";
              log_stream << prefix << " pose value: " << v->pose[0] << ", "
                         << v->pose[1] << ", " << v->pose[2] << ", "
                         << v->pose[3] << "\n";
              log_stream << prefix << " goal value: " << goal_node->pose[0]
                         << ", " << goal_node->pose[1] << ", "
                         << goal_node->pose[2] << ", " << goal_node->pose[3]
                         << "\n";
              log_stream << prefix << " start value: " << start_node->pose[0]
                         << ", " << start_node->pose[1] << ", "
                         << start_node->pose[2] << ", " << start_node->pose[3]
                         << "\n";
              log_stream.flush();
            }
          }
          if (debug) {
            log_stream << "exit inner loop after finding goal" << std::endl;
          }
          break; // exit inner loop after finding goal (mirrors IGHAStar)
        }
        end_time = std::chrono::high_resolution_clock::now();
        goal_check_time +=
            std::chrono::duration<float, std::micro>(end_time - start_time)
                .count();

        start_time = std::chrono::high_resolution_clock::now();
        // Generate successors - mirrors IGHAStar
        if (debug) {
          log_stream << "generate successors" << std::endl;
        }
        // When preemptive expansion is enabled, this also opportunistically
        // expands stash vertices for this search direction in the same GPU
        // launch (each direction has its own env/stream and stash).
        std::vector<std::shared_ptr<Node>> neighbors =
            search->preemptive_enabled
                ? search->expand_with_preemption(v, goal_node)
                : search->env->Succ(v, goal_node);
        end_time = std::chrono::high_resolution_clock::now();
        successor_time +=
            std::chrono::duration<float, std::micro>(end_time - start_time)
                .count();

        search->expansion_counter++;
        total_expansions.fetch_add(1);

        start_time = std::chrono::high_resolution_clock::now();
        // Process successors - mirrors IGHAStar
        for (auto &neighbor : neighbors) {
          bool tolerance_check =
              search->Q_v_hash.count(neighbor->hash) == 0 &&
              search->Seen_hash.count(neighbor->hash) == 0 &&
              search->inactive_Q_v_hash.count(neighbor->hash) == 0;
          if (debug) {
            log_stream << "check tolerance" << std::endl;
          }
          if (tolerance_check) {
            if (debug) {
              log_stream << "update G" << std::endl;
            }
            if (neighbor->g <
                search->get_G(search->level, neighbor->index[search->level])) {
              search->GUpdate(neighbor);
              neighbor->active = true;
              search->Q_v.push(neighbor);
              search->Q_v_hash.insert(neighbor->hash);
            } else {
              neighbor->active = false;
              search->inactive_Q_v.insert(neighbor);
              search->inactive_Q_v_hash.insert(neighbor->hash);
            }
            if (search->preemptive_enabled) {
              // Newly discovered vertex becomes eligible for preemptive
              // expansion in a later iteration of this search direction.
              search->unexpanded_stash.push_back(neighbor);
            }
            if (debug) {
              log_stream << "inserted node into Q_v and updated G" << std::endl;
            }
            // === BIDIRECTIONAL ADDITION: Update LCR_table and check near-meet
            // ===
            int LCR_idx = neighbor->LCR_index;
            update_LCR_table(is_forward, neighbor, LCR_idx);
            // measure the time for checking near meets:
            auto nm_start_time = std::chrono::high_resolution_clock::now();
            // Check for near-meet with other search
            auto [found, path_cost, connected_path] =
                NearMeet(search, other_search, neighbor, start_node,
                         other_start_node, is_forward, log_stream);
            // technically, NearMeet is overloading the "NearMeet", as well as
            // getNearMeetCost
            auto nm_end_time = std::chrono::high_resolution_clock::now();
            // log it to log_stream:
            if (debug) {
              log_stream << "time for checking near meets: "
                         << std::chrono::duration<float, std::micro>(
                                nm_end_time - nm_start_time)
                                .count()
                         << " microseconds" << std::endl;
            }
            if (found) {
              {
                std::lock_guard<std::mutex> lock(path_mutex);
                best_path_list.push_back(connected_path);
                shared_Omega.store(path_cost);
                search->Omega = path_cost;
              }
            }
          }
        }
        {
          std::lock_guard<std::mutex> lock(profiling_lists_mutex);
          cost_exp_list.push_back(shared_Omega.load());
        }
      }

      if (debug) {
        log_stream << "level change" << std::endl;
      }
      // Level change - mirrors IGHAStar's outer loop end
      // combine_Q_v_and_prune(); change_resolution_g_update(); Activate();
      start_time = std::chrono::high_resolution_clock::now();
      // set the search->Omega to the shared_Omega:
      search->Omega = shared_Omega.load();
      if (debug) {
        log_stream << "set Omega to " << search->Omega
                   << " with Q_v size: " << search->Q_v.size()
                   << " and inactive_Q_v size: " << search->inactive_Q_v.size()
                   << std::endl;
      }
      search->combine_Q_v_and_prune();
      search->change_resolution_g_update();
      search->Activate();
      end_time = std::chrono::high_resolution_clock::now();
      // not pruning seen right now because that might also require figuring out
      // the G_LCR and V_LCR
      if (debug) {
        log_stream << "done with level change" << std::endl;
      }
      int total_nodes = search->Q_v.size() + search->inactive_Q_v.size() +
                        search->Seen.size();
      if (total_nodes > 0) {
        overhead_time +=
            std::chrono::duration<float, std::micro>(end_time - start_time)
                .count() /
            total_nodes;
      }
      // Check if search is exhausted
      if (debug) {
        log_stream << "check if search is exhausted" << std::endl;
      }
      if (search->Q_v.empty() && search->inactive_Q_v.empty()) {
        my_finished.store(true);
        if (debug) {
          log_stream << (is_forward ? "Forward" : "Backward")
                     << " search exhausted at " << search->expansion_counter
                     << " expansions with Q_v size: " << search->Q_v.size()
                     << " and inactive_Q_v size: "
                     << search->inactive_Q_v.size() << std::endl;
        }
        // Release barrier so other thread can continue if it's waiting
        if (!other_finished.load()) {
          expansion_barrier->arrive_and_notify();
        }
        break;
      }
      switches++;
      max_level_profile = std::max(max_level_profile, search->level);
      expansion_list.push_back(total_expansions.load());
      // if running naive:
      if (debug) {
        log_stream << "check if running naive" << std::endl;
      }
      if (search->run_naive) {
        if (debug) {
          log_stream << "running naive" << std::endl;
        }
        search->naive(start_node);
      }
      if (debug) {
        log_stream << prefix << " [exp:" << search->expansion_counter
                   << "] Level change to " << search->level << "\n";
        log_stream.flush();
      }
    }
    if (debug) {
      log_stream << "exit outer loop" << std::endl;
    }
    // Signal that this search is done
    my_finished.store(true);
    if (debug) {
      // print the average successor time at this point:
      log_stream << "average successor time: "
                 << successor_time / total_expansions.load() << std::endl;
    }
    // Release barrier one final time so other thread can continue if it's
    // waiting
    if (!other_finished.load()) {
      expansion_barrier->arrive_and_notify();
    }
  }

  void search(float *start, float *goal, int expansion_limit,
              int hysteresis_threshold, bool ignore_goal_validity = false) {
    // measure time spent before starting search:
    start_time = std::chrono::high_resolution_clock::now();
    // Reset state
    shared_Omega.store(1e5f);
    total_expansions.store(0);
    search_complete.store(false);
    forward_finished.store(false);
    backward_finished.store(false);
    best_path.clear();
    best_path_list.clear();
    forward_LCR_table.clear();
    backward_LCR_table.clear();

    // Reset profiling variables
    successor_time = 0;
    goal_check_time = 0;
    overhead_time = 0;
    g_update_time = 0;
    switches = 0;
    max_level_profile = 0;
    expansion_list.clear();
    cost_exp_list.clear();

    forward_search->reset();
    backward_search->reset();

    // Create start and goal nodes for both directions
    std::shared_ptr<Node> forward_start =
        forward_search->env->create_Node(start);
    std::shared_ptr<Node> forward_goal = forward_search->env->create_Node(goal);
    std::shared_ptr<Node> backward_start =
        backward_search->env->create_Node(goal); // Backward starts at goal
    std::shared_ptr<Node> backward_goal =
        backward_search->env->create_Node(start); // Backward aims for start

    forward_start->f =
        forward_search->env->heuristic(forward_start->pose, forward_goal->pose);
    backward_start->f = backward_search->env->heuristic(backward_start->pose,
                                                        backward_goal->pose);

    // Initialize both searches
    forward_search->initialize(forward_start);
    backward_search->initialize(backward_start);
    // Create log files for each search thread
    std::ofstream forward_log("forward.log", std::ios::out | std::ios::trunc);
    std::ofstream backward_log("backward.log", std::ios::out | std::ios::trunc);
    // Sample valid states around goal and add them to backward search
    // This handles cases where the exact goal state is invalid
    // check start goal validity:
    bool start_goal_valid = true;
    bool valid[2] = {true, true};
    forward_search->env->check_validity(forward_start->pose, forward_goal->pose,
                                        valid);
    // print start goal validity:
    if (debug) {
      std::cout << "Start node validity: " << valid[0] << std::endl;
      std::cout << "Goal node validity: " << valid[1] << std::endl;
    }

    // Goal sampling: sample reachable states around the goal for backward
    // search. This is needed in practice because your exact goal state may be
    // unreachable (e.g., due to discretization or kinematic constraints).
    sample_goals(backward_start, backward_goal, backward_log);

    // Initialize LCR_tables
    reset_LCR_table(forward_search, forward_start);
    reset_LCR_table(backward_search, backward_start);

    // Create barrier for 2 threads
    expansion_barrier = std::make_unique<SimpleBarrier>(2);

    // run naive if hysteresis_threshold is negative:
    if (hysteresis_threshold < 0) {
      forward_search->run_naive = true;
      backward_search->run_naive = true;
      // set hysteresis_threshold to expansion_limit + 1:
      hysteresis_threshold = expansion_limit + 1;
    } else {
      forward_search->run_naive = false;
      backward_search->run_naive = false;
      // Set hysteresis threshold on both searches
      forward_search->hysteresis_threshold = hysteresis_threshold;
      backward_search->hysteresis_threshold = hysteresis_threshold;
    }

    if (debug) {
      // Print initialization info to logs
      if (!forward_search->Q_v.empty()) {
        forward_log << "[F] Initialization - Top of Q_v: "
                    << forward_search->Q_v.top()->pose[0] << ", "
                    << forward_search->Q_v.top()->pose[1] << ", "
                    << forward_search->Q_v.top()->pose[2] << ", "
                    << forward_search->Q_v.top()->pose[3] << "\n";
        forward_log.flush();
      }
      if (!backward_search->Q_v.empty()) {
        backward_log << "[B] Initialization - Top of Q_v: "
                     << backward_search->Q_v.top()->pose[0] << ", "
                     << backward_search->Q_v.top()->pose[1] << ", "
                     << backward_search->Q_v.top()->pose[2] << ", "
                     << backward_search->Q_v.top()->pose[3] << "\n";
        backward_log.flush();
      }
    }
    // Launch parallel search threads with log streams
    std::thread forward_thread(
        &BiIGHAStar::search_worker, this, forward_search, backward_search,
        forward_start, forward_goal, backward_start, expansion_limit,
        hysteresis_threshold, true, std::ref(forward_log));

    std::thread backward_thread(
        &BiIGHAStar::search_worker, this, backward_search, forward_search,
        backward_start, backward_goal, forward_start, expansion_limit,
        hysteresis_threshold, false, std::ref(backward_log));
    end_time = std::chrono::high_resolution_clock::now();
    float initialization_time =
        std::chrono::duration<float, std::micro>(end_time - start_time).count();
    if (debug) {
      std::cout << "Initialization time: " << initialization_time
                << " microseconds" << std::endl;
    }
    // Set thread affinity to ensure threads run on different CPU cores for true
    // parallelism
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset); // Pin forward thread to core 0
    pthread_setaffinity_np(forward_thread.native_handle(), sizeof(cpu_set_t),
                           &cpuset);

    CPU_ZERO(&cpuset);
    CPU_SET(1, &cpuset); // Pin backward thread to core 1
    pthread_setaffinity_np(backward_thread.native_handle(), sizeof(cpu_set_t),
                           &cpuset);

    if (debug) {
      std::cout << "Thread affinity set: forward->core 0, backward->core 1"
                << std::endl;
    }
#endif

    // Wait for both threads to complete
    forward_thread.join();
    backward_thread.join();

    // Close log files
    forward_log.close();
    backward_log.close();

    if (debug) {
      std::cout << "BiIGHAStar search complete. Total expansions: "
                << total_expansions.load()
                << ", Best cost: " << shared_Omega.load() << std::endl;
      std::cout << "Logs written to forward.log and backward.log" << std::endl;
    }
  }

  bool search_adapter(torch::Tensor start_tensor, torch::Tensor goal_tensor,
                      torch::Tensor world_tensor, int expansion_limit,
                      int hysteresis_threshold,
                      bool ignore_goal_validity = false) {
    forward_search->env->set_world(world_tensor);
    backward_search->env->set_world(world_tensor);

    auto start = start_tensor.contiguous().data_ptr<float>();
    auto goal = goal_tensor.contiguous().data_ptr<float>();

    search(start, goal, expansion_limit, hysteresis_threshold,
           ignore_goal_validity);

    return shared_Omega.load() < 1e5f;
  }

  torch::Tensor get_best_path() {
    if (shared_Omega.load() > 1000.0f) {
      // if(debug){
      std::cout << "Best path is not found, returning empty path" << std::endl;
      // }
      return torch::zeros({0, n_dims + 1},
                          torch::TensorOptions().dtype(torch::kFloat32));
    }
    if (debug) {
      if (best_path_list.empty()) {
        std::cerr << "ERROR: best_path_list is empty!" << std::endl;
        return torch::zeros({0, n_dims + 1},
                            torch::TensorOptions().dtype(torch::kFloat32));
      }
      std::cout << "Getting best path from best_path_list, size: "
                << best_path_list.size() << std::endl;
      std::cout << "Best path node count: "
                << best_path_list[best_path_list.size() - 1].size()
                << std::endl;
    }
    auto path = forward_search->env->convert_node_list_to_path_tensor(
        best_path_list[best_path_list.size() - 1]);
    // print the path:
    if (debug) {
      std::cout << "path: " << std::endl;
      for (int i = 0; i < path.size(0); i++) {
        std::cout << "node " << i << ": ";
        for (int d = 0; d < path.size(1); d++) {
          std::cout << path[i][d];
          if (d < path.size(1) - 1)
            std::cout << ", ";
        }
        std::cout << std::endl;
      }
      std::cout << "end of path" << std::endl;
    }
    return path;
  }

  std::vector<torch::Tensor> get_path_list() {
    std::vector<torch::Tensor> path_list;
    for (auto &path : best_path_list) {
      path_list.push_back(
          forward_search->env->convert_node_list_to_path_tensor(path));
    }
    return path_list;
  }

  int get_total_expansions() { return total_expansions.load(); }

  float get_best_cost() { return shared_Omega.load(); }

  // Total vertices expanded preemptively across both search directions.
  long get_preemptive_expansions() {
    return forward_search->preemptive_expansions +
           backward_search->preemptive_expansions;
  }

  // get the profiler info, similar to IGHAStar but aggregated from both
  // searches
  std::tuple<float, float, float, float, int, int, int, int, std::vector<int>,
             std::vector<float>>
  get_profiler_info() {
    int total_exp = total_expansions.load();
    float avg_successor_time = total_exp > 0 ? successor_time / total_exp : 0;
    float avg_goal_check_time = total_exp > 0 ? goal_check_time / total_exp : 0;
    float avg_overhead_time = switches > 0 ? overhead_time / switches : 0;
    float avg_g_update_time = total_exp > 0 ? g_update_time / total_exp : 0;

    int total_Q_v_size =
        forward_search->Q_v.size() + forward_search->inactive_Q_v.size() +
        backward_search->Q_v.size() + backward_search->inactive_Q_v.size();

    return std::make_tuple(avg_successor_time, avg_goal_check_time,
                           avg_overhead_time, avg_g_update_time, switches,
                           max_level_profile, total_Q_v_size, total_exp,
                           expansion_list, cost_exp_list);
  }
};

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("ighastar_search", &ighastar_search, "IGHA*");
// }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<IGHAStar>(m, "IGHAStar")
      .def(py::init<const py::dict &, bool, int>(), py::arg("config"),
           py::arg("debug") = false, py::arg("time_direction") = 1)
      .def("search", &IGHAStar::search_adapter)
      .def("get_path_list", &IGHAStar::get_path_list)
      .def("get_best_path", &IGHAStar::get_best_path)
      .def("get_profiler_info", &IGHAStar::get_profiler_info)
      .def("get_preemptive_expansions",
           &IGHAStar::get_preemptive_expansions);

  py::class_<BiIGHAStar>(m, "BiIGHAStar")
      .def(py::init<const py::dict &, bool>(), py::arg("config"),
           py::arg("debug") = false)
      .def("search", &BiIGHAStar::search_adapter)
      .def("get_path_list", &BiIGHAStar::get_path_list)
      .def("get_best_path", &BiIGHAStar::get_best_path)
      .def("get_total_expansions", &BiIGHAStar::get_total_expansions)
      .def("get_best_cost", &BiIGHAStar::get_best_cost)
      .def("get_profiler_info", &BiIGHAStar::get_profiler_info)
      .def("get_preemptive_expansions",
           &BiIGHAStar::get_preemptive_expansions);
}