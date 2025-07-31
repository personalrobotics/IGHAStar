#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic environment unit tests.
Python orchestrator that compiles C++ test code and runs environment function tests.
Can test any environment type (kinematic, kinodynamic, simple).
"""

import torch
from torch.utils.cpp_extension import load
import pathlib
import sys
import os

# Add parent directory to path to import common utilities
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from ighastar.scripts.common_utils import BASE_DIR

class EnvironmentTester:
    def __init__(self, env_type="kinematic", force_cpu=False):
        """Initialize the test environment."""
        self.env_type = env_type
        self.force_cpu = force_cpu
        self.setup_test_config()
        self.load_test_maps()
        self.compile_test_environment()
        self.setup_test_conditions()
        
    def setup_test_config(self):
        """Setup configuration for the specified environment."""
        if self.env_type == "kinematic":
            self.test_config = {
                "experiment_info_default": {
                    "resolution": 4.0,
                    "epsilon": [2.0, 2.0, 0.2, 5.0],  # ate, cte, heading, vel for goal region
                    "tolerance": 0.5,
                    "max_level": 5,
                    "division_factor": 2.0,
                    "max_expansions": 100000,
                    "hysteresis": 20000,
                    "node_info": {
                        "node_type": "kinematic",
                        "length": 2.6,
                        "width": 1.6,
                        "map_res": 0.1,
                        "dt": 0.05,
                        "timesteps": 10,
                        "step_size": 1.0,
                        "steering_list": [-25.0, -10.0, 0.0, 10.0, 25.0],
                        "throttle_list": [-5.0, 5.0],
                        "del_theta": 90.0,
                        "max_theta": 30.0,
                        "del_vel": 2.5,
                        "RI": 0.8,
                        "max_vert_acc": 3.0,
                        "min_vel": -10.0,
                        "max_vel": 10.0,
                        "gear_switch_time": 0.0
                    }
                }
            }
        elif self.env_type == "kinodynamic":
            self.test_config = {
                "experiment_info_default": {
                    "resolution": 4.0,
                    "epsilon": [2.0, 2.0, 0.2, 5.0],  # ate, cte, heading, vel for goal region
                    "tolerance": 0.5,
                    "max_level": 5,
                    "division_factor": 2.0,
                    "max_expansions": 100000,
                    "hysteresis": 1000,
                    "node_info": {
                        "node_type": "kinodynamic",
                        "length": 2.6,
                        "width": 1.6,
                        "map_res": 0.1,
                        "dt": 0.05,
                        "timesteps": 10,
                        "step_size": 1.0,
                        "steering_list": [-25.0, -10.0, 0.0, 10.0, 25.0],
                        "throttle_list": [-5.0, 0.0, 5.0],
                        "del_theta": 90.0,
                        "del_vel": 2.5,
                        "RI": 0.8,
                        "max_vert_acc": 3.0,
                        "min_vel": -10.0,
                        "max_vel": 10.0,
                        "max_theta": 30.0,
                        "gear_switch_time": 1.0
                    }
                }
            }
        elif self.env_type == "simple":
            self.test_config = {
                "experiment_info_default": {
                    "resolution": 20.0,
                    "epsilon": [20.0, 20.0],  # x, y for goal region
                    "tolerance": 0.01,
                    "max_level": 5,
                    "division_factor": 2.0,
                    "max_expansions": 100000,
                    "hysteresis": 1000,
                    "node_info": {
                        "node_type": "simple",
                        "step_size": 25.0,
                        "n_succ": 8,
                        "timesteps": 20,
                        "map_res": 1.0,
                    }
                }
            }
        else:
            raise ValueError("Unsupported environment type: {}".format(self.env_type))
        
    def load_test_maps(self):
        """Generate test maps programmatically."""
        # Create 100x100 maps
        map_size = 100
        
        # Convert to torch tensors with proper format based on environment
        if self.env_type == "simple":
            # Simple environment expects single channel
            # All valid map (white pixels = 255)
            self.valid_map = torch.full((map_size, map_size), 255.0, dtype=torch.float32)
            # All invalid map (black pixels = 0)
            self.invalid_map = torch.full((map_size, map_size), 0.0, dtype=torch.float32)
        else:
            # Kinematic/kinodynamic expect 2-channel tensor: [costmap, elevation]
            # All valid map
            self.valid_map = torch.zeros((map_size, map_size, 2), dtype=torch.float32)
            self.valid_map[:, :, 0] = 255.0  # Costmap - all valid (white)
            self.valid_map[:, :, 1] = 0.0    # Elevation - flat
            
            # All invalid map  
            self.invalid_map = torch.zeros((map_size, map_size, 2), dtype=torch.float32)
            self.invalid_map[:, :, 0] = 0.0  # Costmap - all invalid (black)
            self.invalid_map[:, :, 1] = 0.0  # Elevation - flat
        
    def compile_test_environment(self):
        """Compile the test environment using PyTorch JIT."""
        print("Compiling {} environment test...".format(self.env_type))
        
        # Determine CUDA availability and whether to use CPU
        cuda_available = torch.cuda.is_available() and not self.force_cpu
        
        if self.force_cpu:
            print("Forcing CPU compilation (--cpu flag)")
        elif not torch.cuda.is_available():
            print("CUDA not available, using CPU versions")
        else:
            print("CUDA available, using GPU versions")
        
        # Set up environment macro and sources following common_utils.py pattern
        if self.env_type == "simple":
            print("Using header-only version (simple environment)")
            env_macro = "-DUSE_SIMPLE_ENV"
            sources = [str(pathlib.Path(__file__).parent / "test.cpp")]
            extra_cuda_cflags = []
        elif cuda_available:
            # Use CUDA version
            env_macro = "-DUSE_{}_ENV".format(self.env_type.upper())
            cuda_path = BASE_DIR / "src" / "Environments" / "{}.cu".format(self.env_type)
            sources = [
                str(pathlib.Path(__file__).parent / "test.cpp"),
                str(cuda_path)
            ]
            extra_cuda_cflags = ["-O3"]
        else:
            # Use CPU version - compile with CPU header and .cpp file included
            env_macro = "-DUSE_{}_CPU_ENV".format(self.env_type.upper())
            cpu_cpp_path = BASE_DIR / "src" / "Environments" / "{}_cpu.cpp".format(self.env_type)
            sources = [
                str(pathlib.Path(__file__).parent / "test.cpp"),
                str(cpu_cpp_path)
            ]
            extra_cuda_cflags = []
            
        # Set up include paths
        header_path = BASE_DIR / "src" / "Environments"
        extra_includes = [str(header_path)]
        
        # Detect macOS for Boost include path (following common_utils.py pattern)
        is_macos = sys.platform == "darwin"
        if is_macos:
            boost_include = "/opt/homebrew/opt/boost/include"  # Adjust if needed
            extra_includes = [str(header_path), boost_include]
            
        # Compile the test kernel
        self.test_kernel = load(
            name="{}_{}_test".format(self.env_type, "cpu" if not cuda_available else "gpu"),
            sources=sources,
            extra_include_paths=extra_includes,
            extra_cflags=["-std=c++17", "-O3", env_macro],
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=True,
        )
        
        # Create the test environment instance
        self.tester = self.test_kernel.EnvironmentTester(self.test_config)
        
    def setup_test_conditions(self):
        """Define test conditions and poses based on environment type."""
        if self.env_type == "simple":
            # Simple environment uses 2D states [x, y]
            # Map is 100x100 pixels with 1.0 resolution = 100x100 meters
            self.test_pose_tensor = torch.tensor([50.0, 50.0], dtype=torch.float32)
            
            self.start_tensor = torch.tensor([50.0, 50.0], dtype=torch.float32)
            self.goal_tensor = torch.tensor([70.0, 70.0], dtype=torch.float32)
            self.close_goal_tensor = torch.tensor([50.1, 50.1], dtype=torch.float32)
            self.far_goal_tensor = torch.tensor([10.0, 10.0], dtype=torch.float32)
            
        elif self.env_type == "kinematic":
            # Kinematic environment uses 4D states [x, y, theta, unused]
            # Map is 100x100 pixels with 0.1 resolution = 10x10 meters, so coordinates must be [0, 10)
            self.test_pose_tensor = torch.tensor([5.0, 5.0, 0.0, 0.0], dtype=torch.float32)
            
            self.start_tensor = torch.tensor([5.0, 5.0, 0.0, 0.0], dtype=torch.float32)
            self.goal_tensor = torch.tensor([7.0, 7.0, 1.57, 0.0], dtype=torch.float32)
            self.close_goal_tensor = torch.tensor([5.01, 5.01, 0.01, 0.0], dtype=torch.float32)
            self.far_goal_tensor = torch.tensor([1.0, 1.0, 3.14, 0.0], dtype=torch.float32)
            
        elif self.env_type == "kinodynamic":
            # Kinodynamic environment uses 4D states [x, y, theta, velocity]
            # Map is 100x100 pixels with 0.1 resolution = 10x10 meters, so coordinates must be [0, 10)
            self.test_pose_tensor = torch.tensor([5.0, 5.0, 0.0, 3.0], dtype=torch.float32)
            
            self.start_tensor = torch.tensor([5.0, 5.0, 0.0, 3.0], dtype=torch.float32)
            self.goal_tensor = torch.tensor([7.0, 7.0, 1.57, 4.0], dtype=torch.float32)
            self.close_goal_tensor = torch.tensor([5.01, 5.01, 0.01, 3.01], dtype=torch.float32)
            self.far_goal_tensor = torch.tensor([1.0, 1.0, 3.14, 2.0], dtype=torch.float32)
            
    def run_all_tests(self):
        """Run all environment function tests."""
        print("Running {} Environment Function Tests".format(self.env_type.title()))
        print("=" * 60)
        try:
            # Run all tests through the compiled C++ code with tensors
            all_passed = self.tester.run_all_tests(
                self.valid_map,
                self.invalid_map,
                self.test_pose_tensor,
                self.start_tensor,
                self.goal_tensor,
                self.close_goal_tensor,
                self.far_goal_tensor
            )
            
            return all_passed
            
        except Exception as e:
            print("=" * 60)
            print("Test execution failed: {}".format(e))
            print("=" * 60)
            return False

def main():
    """Main test function."""
    import argparse
    parser = argparse.ArgumentParser(description='Test environment functions')
    parser.add_argument('--env', choices=['kinematic', 'kinodynamic', 'simple'], default='kinematic',
                       help='Environment type to test')
    parser.add_argument('--cpu', action='store_true', help='Force CPU compilation')
    args = parser.parse_args()
    
    try:
        test_suite = EnvironmentTester(args.env, args.cpu)
        success = test_suite.run_all_tests()
        return 0 if success else 1
    except Exception as e:
        print("Test suite initialization failed: {}".format(e))
        return 1

if __name__ == "__main__":
    exit(main()) 