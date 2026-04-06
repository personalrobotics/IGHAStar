import torch
from torch.utils.cpp_extension import load
import pathlib
import os
from typing import Any, Dict
import sys


def _find_source_dir() -> pathlib.Path:
    """
    Find the ighastar source directory containing C++ files.

    Priority:
    1. IGHASTAR_SRC_DIR environment variable
    2. Relative to this file (works for editable installs / running from source)
    3. Common development locations
    """
    # Check environment variable first
    env_src = os.environ.get("IGHASTAR_SRC_DIR")
    if env_src:
        src_path = pathlib.Path(env_src)
        if (src_path / "src" / "ighastar.cpp").exists():
            return src_path

    # Try relative to this file (editable install or running from source)
    file_based = pathlib.Path(__file__).resolve().parent.parent
    if (file_based / "src" / "ighastar.cpp").exists():
        return file_based

    # Try common development locations
    common_paths = [
        pathlib.Path.home() / "catkin_ws" / "src" / "ighastar" / "ighastar",
        pathlib.Path("/root/catkin_ws/src/ighastar/ighastar"),
        pathlib.Path.cwd() / "ighastar",
    ]
    for path in common_paths:
        if (path / "src" / "ighastar.cpp").exists():
            return path

    # Fallback to file-based path (will fail later with a clearer error)
    return file_based


BASE_DIR = _find_source_dir()


def create_planner(configs: Dict[str, Any], bidirectional: bool = False) -> Any:
    """
    Create an IGHA* or BiIGHA* planner based on configuration.

    Args:
        configs: Configuration dictionary containing experiment_info_default
        bidirectional: If True, create a BiIGHAStar planner instead of IGHAStar

    Returns:
        The created planner instance
    """
    env_name = configs["experiment_info_default"]["node_info"]["node_type"]
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()

    if not cuda_available:
        print("CUDA not available, using CPU versions")
        env_macro = {
            "simple": "-DUSE_SIMPLE_ENV",
            "kinematic": "-DUSE_KINEMATIC_CPU_ENV",
            "kinodynamic": "-DUSE_KINODYNAMIC_CPU_ENV",
        }[env_name]
    else:
        print("CUDA available, using GPU versions")
        env_macro = {
            "simple": "-DUSE_SIMPLE_ENV",
            "kinematic": "-DUSE_KINEMATIC_ENV",
            "kinodynamic": "-DUSE_KINODYNAMIC_ENV",
        }[env_name]

    cpp_path = BASE_DIR / "src" / "ighastar.cpp"
    env_include_path = BASE_DIR / "src" / "Environments" / "include"
    utils_path = BASE_DIR / "src" / "utils"
    src_path = BASE_DIR / "src"
    extra_includes = [str(env_include_path), str(utils_path), str(src_path)]

    # Detect if running on macOS for Boost include path
    is_macos = sys.platform == "darwin"
    if is_macos:
        boost_include = "/opt/homebrew/opt/boost/include"  # Adjust if needed
        extra_includes = [
            str(env_include_path),
            str(utils_path),
            str(src_path),
            boost_include,
        ]

    if env_name != "simple":
        if cuda_available:
            # Use CUDA version
            cuda_path = BASE_DIR / "src" / "Environments" / "src" / f"{env_name}.cu"
            kernel = load(
                name="ighastar",
                sources=[str(cpp_path), str(cuda_path)],
                extra_include_paths=extra_includes,
                extra_cflags=["-std=c++17", "-O3", env_macro],
                extra_cuda_cflags=["-O3"],
                verbose=True,
            )
        else:
            # Use CPU version - compile with CPU header and .cpp file included
            cpu_cpp_path = (
                BASE_DIR / "src" / "Environments" / "src" / f"{env_name}_cpu.cpp"
            )
            kernel = load(
                name="ighastar",
                sources=[str(cpp_path), str(cpu_cpp_path)],
                extra_include_paths=extra_includes,
                extra_cflags=["-std=c++17", "-O3", env_macro],
                verbose=True,
            )
    else:
        kernel = load(
            name="ighastar",
            sources=[str(cpp_path)],
            extra_include_paths=extra_includes,
            extra_cflags=["-std=c++17", "-O3", env_macro],
            verbose=True,
        )

    if bidirectional:
        planner = kernel.BiIGHAStar(configs, False)
    else:
        planner = kernel.IGHAStar(configs, False)
    return planner
