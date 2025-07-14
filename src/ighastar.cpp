#include <torch/extension.h>
#include <queue>
#include <vector>
#include <unordered_set>
#include <tuple>
#include <boost/functional/hash.hpp>
#include <memory>
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
#else
#error "No environment macro defined"
#endif
// END ENVIRONMENT
#include <unordered_map>
#include <unordered_set>
#include <iomanip>
#include <pybind11/stl.h>
#include <chrono>

using namespace std;

class IGHAStar {
public:
    std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, NodePtrCompare> Q_v;
    std::unordered_set<size_t> Q_v_hash; // set of hashes for quick lookup
    std::unordered_set<std::shared_ptr<Node>> inactive_Q_v; // inactive_Qv vector
    std::unordered_set<size_t> inactive_Q_v_hash; // set of hashes for quick lookup
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
    Environment* env;
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

    IGHAStar(const py::dict& config, bool debug_=false){ //maybe have a debug mode?
        // create environment using config:
        env = new Environment(config);
        debug = debug_;
    }
    
    ~IGHAStar() {
        if (env) {
            env->cleanup();  // Ensure CUDA cleanup is called
            delete env;
            env = nullptr;
        }
    }

    void reset() {
        Q_v = std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, NodePtrCompare>();
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
    }

    // activate method for HA*M
    void naive(std::shared_ptr<Node> start_node) {
        Q_v = std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, NodePtrCompare>();
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
        // perform the approximate dominance check and freeze existing vertex if it is dominated
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

    // bubble the active nodes to the top of the Q_v while throwing inactive nodes into inactive_Q_v
    void bubbleActive() {
        while (!Q_v.empty() && Q_v.top()->active != true) {
            std::shared_ptr<Node> v = Q_v.top();
            Q_v.pop();
            Q_v_hash.erase(v->hash);
            inactive_Q_v.insert(v);
            inactive_Q_v_hash.insert(v->hash);
        }
    }

    std::vector<std::shared_ptr<Node>> reconstruct_path(std::shared_ptr<Node> start, std::shared_ptr<Node> goal) {
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
        if (run){
            if (Q_v.top()->level == level) {
                run = true;
            } 
            else {
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

    inline void local_g_update(std::shared_ptr<Node> node)
    {
        if (node->g < get_G(level, node->index[level])) {
            G[level][node->index[level]] = node->g;
            V[level][node->index[level]] = node;
        }
    }

    void combine_Q_v_and_prune() {
        // combine Q_v and inactive_Q_v
        std::unordered_set<std::shared_ptr<Node>> new_inactive_Q_v;
        std::unordered_set<size_t> new_inactive_Q_v_hash;
        // this step "empties" the Q_v, so everything is now only in the inactive set.
        while(!Q_v.empty()){
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
            G.emplace_back();  // std::unordered_map<int, float>
            V.emplace_back();  // std::unordered_map<int, std::shared_ptr<Node>>
            // reserve 100000 for both:
            G[level].reserve(100000); // consider pre-reserving this memeory at the beginning of the search.
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

        for (auto& [idx, nodes] : grid_cells) {
            std::sort(nodes.begin(), nodes.end(), [](std::shared_ptr<Node> a, std::shared_ptr<Node> b) {return a->g < b->g;});

            for (int i = 0; i < nodes.size(); i++) {
                nodes[i]->rank = i;
            }
        }

    }

    void Activate() {
        // for performance reasons, we are branching-bounding the nodes in the activate method.
        std::vector<std::shared_ptr<Node>> unsorted_Q_v;
        for (auto it = inactive_Q_v.begin(); it != inactive_Q_v.end(); ) {
            std::shared_ptr<Node> node = *it;
            node->level = level;
            if (node->rank == 0 && node->g <= get_G(level, node->index[level])) {
                node->active = true;
                unsorted_Q_v.push_back(node);
                Q_v_hash.insert(node->hash);
                it = inactive_Q_v.erase(it);  // erase returns the next valid iterator
                inactive_Q_v_hash.erase(node->hash);
                G[level][node->index[level]] = node->g;
                V[level][node->index[level]] = node;
            }
            else{
                ++it;  // move to the next element
            }
        }
        // // sort the Q_v based on f value
        std::make_heap(unsorted_Q_v.begin(), unsorted_Q_v.end(), NodePtrCompare{}); // make a sorted heap from unsorted_Q_v
        // // set the Q_v to the sorted heap
        Q_v = std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, NodePtrCompare>(NodePtrCompare{}, std::move(unsorted_Q_v));
    }

    // function to do hybrid A* search and return a path as a tensor I guess:
    void search(float* start, float* goal, int expansion_limit_, int hysteresis_threshold_, bool ignore_goal_validity)  {
        reset();
        expansion_limit = expansion_limit_;
        if(hysteresis_threshold_ >= 0)
        {
            hysteresis_threshold = hysteresis_threshold_;
            run_naive = false;
        }
        else
        {
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
        if(!ignore_goal_validity)
        {
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
        
        while (expansion_counter < expansion_limit && (!Q_v.empty() or !inactive_Q_v.empty())){
            int inactive_insertions = 0;
            while (expansion_counter < expansion_limit) {
                // inner time measurement:
                start_time = std::chrono::high_resolution_clock::now();
                bubbleActive();
                bool run = !Q_v.empty() && Q_v.top()->active == true && Q_v.top()->f < Omega;
                if(!run){
                    break;
                }     

                run = shift(run); // update next candidate resolution and whether we should continue running

                std::shared_ptr<Node> v = Q_v.top();
                Q_v.pop();
                Q_v_hash.erase(v->hash);
                Seen.insert(v);
                Seen_hash.insert(v->hash);

                goal_reached = env->reached_goal_region(v, goal_node);

                if (goal_reached) {
                    SUCCESS = true;
                    if(debug)
                    {
                        std::cout << "expansions: " << expansion_counter <<" cost: " << v->g << std::endl;
                    }
                    best_path = reconstruct_path(start_node, v);
                    Omega = v->g;
                    best_path_list.push_back(best_path);
                    goal_reached = false; // set flag to false to continue searching for better paths
                    expansion_list.push_back(expansion_counter);
                    break;
                }
                if(!run) {
                    break;
                }
                // profiling:
                end_time = std::chrono::high_resolution_clock::now();
                goal_check_time += std::chrono::duration<float, std::micro>(end_time - start_time).count();

                start_time = std::chrono::high_resolution_clock::now();
                std::vector<std::shared_ptr<Node>> neighbors = env->Succ(v, goal_node);
                end_time = std::chrono::high_resolution_clock::now();
                successor_time += std::chrono::duration<float, std::micro>(end_time - start_time).count();

                expansion_counter++;

                start_time = std::chrono::high_resolution_clock::now();
                // get successors:
                for (auto& neighbor : neighbors) {
                    // tolerance check is useful in practice to eliminate vertices that are too close to be practically distinguishable (closer than map resolution)
                    // you can disable this by setting the tolerance to a very small value (1e-5), but it is not recommended.
                    bool tolerance_check = Q_v_hash.count(neighbor->hash) == 0 && Seen_hash.count(neighbor->hash) == 0 && inactive_Q_v_hash.count(neighbor->hash) == 0;
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
                    }
                }
                end_time = std::chrono::high_resolution_clock::now();
                g_update_time += std::chrono::duration<float, std::micro>(end_time - start_time).count();
            }
            start_time = std::chrono::high_resolution_clock::now();
            combine_Q_v_and_prune(); // puts inactive_Q_v into Q_v
            change_resolution_g_update(); // updates G and V
            Activate(); // activates the nodes in Q_v
            end_time = std::chrono::high_resolution_clock::now();

            // overhead_time += std::chrono::duration<float, std::micro>(end_time - start_time).count();
            overhead_time += std::chrono::duration<float, std::micro>(end_time - start_time).count()/(Q_v.size() + inactive_Q_v.size() + Seen.size());
            Q_v_size += Q_v.size();
            switches++;
            if(Q_v.empty() && inactive_Q_v.empty()){
                break;
            }
            if (run_naive) {
                naive(start_node);
            }
            max_level_profile = std::max(max_level_profile, level);
            if (debug) {
                std::printf("Expansions: %d, level %d, Q_v: %d, Seen: %d \n", expansion_counter, level, int(Q_v.size()), int(Seen.size()));
                std::cout<<"Size of inactive_Q_v: " << inactive_Q_v.size() <<" inactive insertions: " << inactive_insertions << std::endl;
            }
        }
    }

    bool search_adapter(torch::Tensor start_tensor, torch::Tensor goal_tensor, torch::Tensor world_tensor, int expansion_limit, 
                        int hysteresis_threshold, bool ignore_goal_validity=false, bool debug_=false) {
        debug = debug_;
        env->set_world(world_tensor);
        auto start = start_tensor.contiguous().data_ptr<float>();
        auto goal = goal_tensor.contiguous().data_ptr<float>();
        search(start, goal, expansion_limit, hysteresis_threshold, ignore_goal_validity);
        if (debug)
        {
            std::cout << "Search finished" << expansion_counter << std::endl;
        }
        return SUCCESS;
    }

    std::vector<torch::Tensor> get_path_list()
    {
        std::vector<torch::Tensor> path_list;
        for (int i = 0; i < best_path_list.size(); i++) {
            auto path_tensor = env->convert_node_list_to_path_tensor(best_path_list[i]);
            path_list.push_back(path_tensor);
        }
        return path_list;
    }
    // get only the best path:
    torch::Tensor get_best_path() {
        auto path_tensor = env->convert_node_list_to_path_tensor(best_path);
        return path_tensor;
    }

    // get the profiler info, I want to get the averages for timing, Q_v size, and number of switches:
    std::tuple<float, float, float, float, int, int, int, int, std::vector<int> > get_profiler_info() {
        float avg_successor_time = successor_time / expansion_counter;
        float avg_goal_check_time = goal_check_time / expansion_counter;
        float avg_overhead_time = overhead_time / switches;
        float avg_g_update_time = g_update_time / expansion_counter;
        return std::make_tuple(avg_successor_time, avg_goal_check_time, avg_overhead_time, avg_g_update_time, 
                                switches, max_level_profile, Q_v_size, expansion_counter, expansion_list);
    }
};

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("ighastar_search", &ighastar_search, "IGHA*");
// }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<IGHAStar>(m, "IGHAStar")
        .def(py::init<const py::dict&, bool>())
        .def("search", &IGHAStar::search_adapter)
        .def("get_path_list", &IGHAStar::get_path_list)
        .def("get_best_path", &IGHAStar::get_best_path)
        .def("get_profiler_info", &IGHAStar::get_profiler_info);
}