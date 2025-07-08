#include <torch/extension.h>
#include <boost/functional/hash.hpp>
#include <pybind11/stl.h>

using namespace std;

constexpr int n_dims = 2;

size_t calc_hash(float *pose, const float *resolution) 
{
    std::size_t hash_val = 0;
    for (int i = 0; i < n_dims; ++i) {
        boost::hash_combine(hash_val, std::hash<size_t>{}( static_cast<int64_t>( std::round(pose[i]/resolution[i] ) ) ) );
    }
    return hash_val;
}

struct Node {
public:
    float pose[n_dims];
    float g, f;
    bool active;
    int rank, level;
    size_t hash;
    std::vector<size_t> index;
    Node* parent;
    
    // Constructor matching kinodynamic pattern
    Node(const float* pose_in, Node* parent_in, float g_in, const float* resolution_, const float* tolerance, int max_level, float division_factor)
        : g(g_in), f(0), parent(parent_in), active(true), rank(0), level(0)
    {
        for (int i = 0; i < n_dims; i++) {
            pose[i] = pose_in[i];
        }
        hash = calc_hash(pose, tolerance);
        float resolution[n_dims];
        for (int i = 0; i < n_dims; i++) {
            resolution[i] = resolution_[i];
        }
        for (int i = 0; i < max_level; i++) {
            index.push_back(calc_hash(pose, resolution));
            for (int j = 0; j < n_dims; j++) {
                resolution[j] = resolution[j] / division_factor;
            }
        }
    }
};

struct NodePtrCompare {
    bool operator()(const Node* a, const Node* b) const {
        return a->f > b->f;  // min-heap: smaller f has higher priority
    }
};

class Environment {
    int map_size_px, n_succ, timesteps;
    float resolution[n_dims], tolerance[n_dims], epsilon[n_dims], division_factor;
    float step_size, map_res;
    float *costmap;
    int max_level;

public:
    Environment(const py::dict& config) {
        auto info = config["experiment_info_default"].cast<py::dict>();
        auto node_info = info["node_info"].cast<py::dict>();
        
        // Top-level fields
        map_res = node_info["map_res"].cast<float>();
        max_level = info["max_level"].cast<int>();
        division_factor = info["division_factor"].cast<float>();
        
        set_resolutions(info, node_info);
        set_simple_params(node_info);
        
        costmap = nullptr;
    }
    
    // Destructor
    ~Environment() {
        if (costmap) delete[] costmap;
    }

    void set_resolutions(py::dict& info, py::dict& node_info) {
        float res = info["resolution"].cast<float>();
        float tol = info["tolerance"].cast<float>();
        auto eps = info["epsilon"].cast<std::vector<float>>();
        
        resolution[0] = res;
        resolution[1] = res;
        tolerance[0] = tol;
        tolerance[1] = tol;
        epsilon[0] = eps[0];
        epsilon[1] = eps[1];
    }

    void set_simple_params(py::dict& node_info) {
        step_size = node_info["step_size"].cast<float>();
        timesteps = node_info["timesteps"].cast<int>();
        n_succ = node_info["n_succ"].cast<int>();
    }

    void set_world(torch::Tensor world) {
        if (costmap) delete[] costmap;

        TORCH_CHECK(world.dim() == 2, "World tensor must be 2D (H x W) for simple environment");
        TORCH_CHECK(world.dtype() == torch::kFloat32, "World tensor must be float32");
        TORCH_CHECK(world.device().is_cpu(), "World tensor must be on CPU");

        int H = world.size(0);
        int W = world.size(1);
        int map_size = H * W;

        // Allocate and copy costmap
        costmap = new float[map_size];
        auto world_contiguous = world.contiguous();
        memcpy(costmap, world_contiguous.data_ptr<float>(), map_size * sizeof(float));

        map_size_px = H; // assume H = W
    }

    void cleanup() {
        if (costmap) {
            delete[] costmap;
            costmap = nullptr;
        }
    }

    Node* create_Node(float *pose) {
        return new Node(pose, nullptr, 0, resolution, tolerance, max_level, division_factor);
    }

    float distance(float *pose, float *goal) {
        float dx = pose[0] - goal[0];
        float dy = pose[1] - goal[1];
        return std::sqrt(dx*dx + dy*dy);
    }

    bool reached_goal_region(Node* v, Node* goal) {
        return distance(v->pose, goal->pose) < epsilon[0];
    }

    float heuristic(float *pose, float *goal){
        return distance(pose, goal);
    }

    void check_validity(float *start, float *goal, bool *result) {
        // For simple environment, just check if both points are valid
        result[0] = check_validity_single(start);
        result[1] = check_validity_single(goal);
    }

    bool check_validity_single(float *pose) {
        float x = pose[0];
        float y = pose[1];
        if (!in_bounds(x, y)) {
            return false;
        }
        if (costmap[m_to_px(x, y)] < 254) {
            return false;
        }
        return true;
    }

    bool in_bounds(float x, float y) {
        if (x < 0 || x >= map_size_px) {
            return false;
        }
        if (y < 0 || y >= map_size_px) {
            return false;
        }
        return true;
    }

    inline int m_to_px(float x, float y) {
        return int(x/map_res) + int(y/map_res) * map_size_px;
    }

    // function that returns a vector of nodes:
    std::vector<Node*> Succ(Node* node, Node* goal) {
        std::vector<Node*> neighbors;
        float x, y, theta, L, f;
        bool valid;
        Node* neighbor;

        for (int i = 0; i < n_succ; i++) 
        {
            theta = i * (2 * M_PI / n_succ);
            float new_pose[n_dims];
            valid = true;
            for (float t = 0; t <= timesteps; t++) {
                L = step_size * t/timesteps;
                x = node->pose[0] + cosf(theta)*L;
                y = node->pose[1] + sinf(theta)*L;
                if ( (costmap[m_to_px(x, y)] != 255.0f) || !in_bounds(x, y) )
                {
                    valid=false;
                    break;
                }
            }
            if(valid){
                new_pose[0] = x;
                new_pose[1] = y;
                
                neighbor = new Node(new_pose, node, node->g + step_size, resolution, tolerance, max_level, division_factor);
                f = neighbor->g + heuristic(neighbor->pose, goal->pose);
                neighbor->f = f;
                neighbors.push_back(neighbor);
            }
        }
        return neighbors;
    }

    torch::Tensor convert_node_list_to_path_tensor(std::vector<Node*> node_list) {
        int path_length = node_list.size();
        auto path_tensor = torch::zeros({path_length, n_dims}, torch::TensorOptions().dtype(torch::kFloat32));
        for (int i = 0; i < path_length; i++) {
            path_tensor[i][0] = node_list[i]->pose[0];
            path_tensor[i][1] = node_list[i]->pose[1];
        }
        return path_tensor;
    }
};