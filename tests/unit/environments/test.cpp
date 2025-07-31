#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <memory>

// Include the environment based on compilation flags
#if defined(USE_KINEMATIC_ENV)
#include <kinematic.h>
#elif defined(USE_KINEMATIC_CPU_ENV)
#include <kinematic_cpu.h>
#elif defined(USE_KINODYNAMIC_ENV)
#include <kinodynamic.h>
#elif defined(USE_KINODYNAMIC_CPU_ENV)
#include <kinodynamic_cpu.h>
#elif defined(USE_SIMPLE_ENV)
#include <simple.h>
#endif

using namespace std;

class EnvironmentTester {
private:
    std::unique_ptr<Environment> env;
    
public:
    EnvironmentTester(const py::dict& config) {
        env = std::make_unique<Environment>(config);
    }
    
    ~EnvironmentTester() {
        if (env) {
            env->cleanup();
        }
    }
    
    // Test 1: Validity checking
    int test_validity(torch::Tensor world, 
                      torch::Tensor start_tensor,
                      torch::Tensor goal_tensor) {
        env->set_world(world);
        
        // Get pose data from tensor
        auto start = start_tensor.contiguous().data_ptr<float>();
        auto goal = goal_tensor.contiguous().data_ptr<float>();
        bool result[2] ={true, true};
        env->check_validity(start, goal, result);
        
        return int(result[0]) + int(result[1]);
    }
    
    // Test 3: Successor generation - ensure at least one successor differs from start
    bool test_successor_generation(torch::Tensor world,
                                  torch::Tensor start_tensor,
                                  torch::Tensor goal_tensor) {
        std::cout << "Test 3: Successor generation..." << std::endl;
        
        env->set_world(world);
        
        // Get start and goal from tensors
        auto start = start_tensor.contiguous().data_ptr<float>();
        auto goal = goal_tensor.contiguous().data_ptr<float>();
        int pose_dim = start_tensor.size(0);
        
        auto start_node = env->create_Node(start);
        auto goal_node = env->create_Node(goal);
        
        // Generate successors
        auto successors = env->Succ(start_node, goal_node);
        
        if (successors.empty()) {
            std::cout << "✗ No successors generated" << std::endl;
            return false;
        }
        
        // Check that at least one successor is different from start
        bool found_different = false;
        for (const auto& succ : successors) {
            bool is_same = true;
            for (int i = 0; i < pose_dim; i++) {
                if (std::abs(succ->pose[i] - start[i]) > 1e-6) {
                    is_same = false;
                    break;
                }
            }
            
            if (!is_same) {
                found_different = true;
                break;
            }
        }
        
        if (!found_different) {
            std::cout << "✗ All successors identical to start state" << std::endl;
            return false;
        }
        
        std::cout << "✓ Generated " << successors.size() 
                 << " successors, at least one different from start" << std::endl;
        return true;
    }
    
    // Test 4: Goal reaching - close distance should return true
    bool test_goal_reaching_close(torch::Tensor world,
                                 torch::Tensor start_tensor,
                                 torch::Tensor close_goal_tensor) {
        std::cout << "Test 4: Goal reaching (close distance)..." << std::endl;
        
        env->set_world(world);
        
        auto start = start_tensor.contiguous().data_ptr<float>();
        auto goal = close_goal_tensor.contiguous().data_ptr<float>();
        
        auto start_node = env->create_Node(start);
        auto goal_node = env->create_Node(goal);
        
        bool reached = env->reached_goal_region(start_node, goal_node);
        
        if (!reached) {
            std::cout << "✗ Close goal not reached when it should be" << std::endl;
            return false;
        }
        
        std::cout << "✓ Close goal correctly reached" << std::endl;
        return true;
    }
    
    // Test 5: Goal reaching - far distance should return false
    bool test_goal_reaching_far(torch::Tensor world,
                               torch::Tensor start_tensor,
                               torch::Tensor far_goal_tensor) {
        std::cout << "Test 5: Goal reaching (far distance)..." << std::endl;
        
        env->set_world(world);
        
        auto start = start_tensor.contiguous().data_ptr<float>();
        auto goal = far_goal_tensor.contiguous().data_ptr<float>();
        
        auto start_node = env->create_Node(start);
        auto goal_node = env->create_Node(goal);
        
        bool reached = env->reached_goal_region(start_node, goal_node);
        
        if (reached) {
            std::cout << "✗ Far goal reached when it shouldn't be" << std::endl;
            return false;
        }
        
        std::cout << "✓ Far goal correctly not reached" << std::endl;
        return true;
    }
    
    // Test 6: Heuristic and distance calculations
    bool test_heuristic_distance(torch::Tensor start_tensor,
                                 torch::Tensor goal_tensor) {
        std::cout << "Test 6: Heuristic and distance calculations..." << std::endl;
        
        auto start = start_tensor.contiguous().data_ptr<float>();
        auto goal = goal_tensor.contiguous().data_ptr<float>();
        
        // Test heuristic
        float heuristic_cost = env->heuristic(start, goal);
        if (heuristic_cost < 0.0f) {
            std::cout << "✗ Heuristic returned negative value: " << heuristic_cost << std::endl;
            return false;
        }
        
        // Test distance
        float distance = env->distance(start, goal);
        if (distance < 0.0f) {
            std::cout << "✗ Distance returned negative value: " << distance << std::endl;
            return false;
        }
        
        // Test heuristic to same position should be small
        float same_heuristic = env->heuristic(start, start);
        if (same_heuristic > 1e-3) {
            std::cout << "✗ Heuristic to same position too large: " << same_heuristic << std::endl;
            return false;
        }
        
        // Test distance to same position should be zero
        float same_distance = env->distance(start, start);
        if (same_distance > 1e-6) {
            std::cout << "✗ Distance to same position not zero: " << same_distance << std::endl;
            return false;
        }
        
        std::cout << "✓ Heuristic and distance calculations work correctly" << std::endl;
        return true;
    }
    
    // Run all tests
    bool run_all_tests(torch::Tensor valid_map,
                      torch::Tensor invalid_map,
                      torch::Tensor test_pose_tensor,
                      torch::Tensor start_tensor,
                      torch::Tensor goal_tensor,
                      torch::Tensor close_goal_tensor,
                      torch::Tensor far_goal_tensor) {
        
        std::cout << "=" << std::string(60, '=') << std::endl;
        std::cout << "Running Environment Unit Tests" << std::endl;
        std::cout << "=" << std::string(60, '=') << std::endl;
        
        // Run diagnostic first
        std::cout << "=== DIAGNOSTIC TESTS ===" << std::endl;
        
        bool all_passed = true;
        
        // Test validity on invalid map (should return 0)
        std::cout << "Test 1: Validity checking with invalid map..." << std::endl;
        int invalid_result = test_validity(invalid_map, start_tensor, goal_tensor);
        if (invalid_result != 0) {
            std::cout << "✗ Invalid map test failed: expected 0, got " << invalid_result << std::endl;
            all_passed = false;
        } else {
            std::cout << "✓ All positions correctly identified as invalid" << std::endl;
        }
        
        // Test validity on valid map (should return 2)
        std::cout << "Test 2: Validity checking with valid map..." << std::endl;
        int valid_result = test_validity(valid_map, start_tensor, goal_tensor);
        if (valid_result != 2) {
            std::cout << "✗ Valid map test failed: expected 2, got " << valid_result << std::endl;
            all_passed = false;
        } else {
            std::cout << "✓ All positions correctly identified as valid" << std::endl;
        }
        
        all_passed &= test_successor_generation(valid_map, start_tensor, goal_tensor);
        all_passed &= test_goal_reaching_close(valid_map, start_tensor, close_goal_tensor);
        all_passed &= test_goal_reaching_far(valid_map, start_tensor, far_goal_tensor);
        all_passed &= test_heuristic_distance(start_tensor, goal_tensor);
        
        std::cout << "=" << std::string(60, '=') << std::endl;
        if (all_passed) {
            std::cout << "✅ All tests passed!" << std::endl;
        } else {
            std::cout << "❌ Some tests failed!" << std::endl;
        }
        std::cout << "=" << std::string(60, '=') << std::endl;
        
        return all_passed;
    }
};

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<EnvironmentTester>(m, "EnvironmentTester")
        .def(py::init<const py::dict&>())
        .def("test_validity", &EnvironmentTester::test_validity)
        .def("test_successor_generation", &EnvironmentTester::test_successor_generation)
        .def("test_goal_reaching_close", &EnvironmentTester::test_goal_reaching_close)
        .def("test_goal_reaching_far", &EnvironmentTester::test_goal_reaching_far)
        .def("test_heuristic_distance", &EnvironmentTester::test_heuristic_distance)
        .def("run_all_tests", &EnvironmentTester::run_all_tests);
} 