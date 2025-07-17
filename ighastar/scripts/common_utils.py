import torch
from torch.utils.cpp_extension import load
import pathlib
from typing import Any, Dict
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

def create_planner(configs: Dict[str, Any]) -> Any:
    env_name = configs["experiment_info_default"]["node_info"]["node_type"]
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()

    if not cuda_available:
        print("CUDA not available, using CPU versions")
        env_macro = {
            'simple': '-DUSE_SIMPLE_ENV',
            'kinematic': '-DUSE_KINEMATIC_CPU_ENV',
            'kinodynamic': '-DUSE_KINODYNAMIC_CPU_ENV',
        }[env_name]
    else:
        print("CUDA available, using GPU versions")
        env_macro = {
            'simple': '-DUSE_SIMPLE_ENV',
            'kinematic': '-DUSE_KINEMATIC_ENV',
            'kinodynamic': '-DUSE_KINODYNAMIC_ENV',
        }[env_name]
    
    cpp_path = BASE_DIR / 'src' / 'ighastar.cpp'
    header_path = BASE_DIR / 'src' / 'Environments'

    if env_name != "simple":
        if cuda_available:
            # Use CUDA version
            cuda_path = BASE_DIR / 'src' / 'Environments' / f'{env_name}.cu'
            kernel = load(
                name="ighastar",
                sources=[str(cpp_path), str(cuda_path)],
                extra_include_paths=[str(header_path)],
                extra_cflags=['-std=c++17', '-O3', env_macro],
                extra_cuda_cflags=['-O3'],
                verbose=True,
            )
        else:
            # Use CPU version - compile with CPU header and .cpp file included
            cpu_cpp_path = BASE_DIR / 'src' / 'Environments' / f'{env_name}_cpu.cpp'
            kernel = load(
                name="ighastar",
                sources=[str(cpp_path), str(cpu_cpp_path)],
                extra_include_paths=[str(header_path)],
                extra_cflags=['-std=c++17', '-O3', env_macro],
                verbose=True,
            )
    else:
        # Simple environment (already CPU-based)
        kernel = load(
            name="ighastar",
            sources=[str(cpp_path)],
            extra_include_paths=[str(header_path)],
            extra_cflags=['-std=c++17', '-O3', env_macro],
            verbose=True,
        )
    
    planner = kernel.IGHAStar(configs, False)
    return planner