import pathlib
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional, Set

import torch
from torch.utils.cpp_extension import load

# True after CXX has been checked / pointed at a C++20-capable compiler for Torch.
_CXX_TORCH_READY = False

# Torch headers require ``__cplusplus >= 202002L`` (see torch/all.h). GCC 10 accepts
# ``-std=c++20`` but still sets ``__cplusplus`` to 201709L on Ubuntu 20.04 — that fails
# Torch's check. GCC 11+ or Clang 10+ set 202002L correctly.
_CXX20_PROBE = b"""\
#if !defined(_MSC_VER) && __cplusplus < 202002L
#error __cplusplus < 202002L (Torch needs a compiler that defines C++20 properly)
#endif
#include <compare>
int main(){return 0;}
"""


def _compiler_supports_torch_cpp20(cxx: str) -> bool:
    """Return True if ``cxx -std=c++20`` defines ``__cplusplus >= 202002L`` for Torch."""
    if not cxx or shutil.which(cxx.split()[0]) is None:
        return False
    probe = subprocess.run(
        [cxx, "-std=c++20", "-x", "c++", "-", "-fsyntax-only"],
        input=_CXX20_PROBE,
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


def _find_torch_cpp20_compiler() -> Optional[str]:
    """Prefer an explicit CXX, then compilers that actually set ``__cplusplus`` to C++20."""
    candidates = []
    if os.environ.get("CXX"):
        candidates.append(os.environ["CXX"])
    # GCC 11+ first (GCC 10 fails Torch's __cplusplus check on Ubuntu 20.04).
    for major in range(14, 10, -1):
        candidates.append("g++-%d" % major)
    # Clang 10+ reports 202002L with -std=c++20 (usable when only g++-10 is installed).
    for major in range(14, 9, -1):
        candidates.append("clang++-%d" % major)
    candidates.extend(["clang++", "g++-10", "g++"])
    seen = set()  # type: Set[str]
    for cxx in candidates:
        if cxx in seen:
            continue
        seen.add(cxx)
        if _compiler_supports_torch_cpp20(cxx):
            return cxx
    return None


def _ensure_cxx_accepts_torch_std_flag() -> None:
    """Point ``CXX`` at a compiler Torch's headers accept (``__cplusplus >= 202002L``).

    ``g++-10 -std=c++20`` is not enough on Ubuntu 20.04: ``__cplusplus`` stays 201709L.
    Use GCC 11+ or Clang 10+.
    """
    global _CXX_TORCH_READY
    if _CXX_TORCH_READY or sys.platform == "win32":
        return
    cxx = _find_torch_cpp20_compiler()
    if cxx is None:
        raise RuntimeError(
            "IGHA* Torch extensions need a compiler with __cplusplus >= 202002L "
            "(GCC 11+ or Clang 10+). Note: g++-10 accepts -std=c++20 but still reports "
            "201709L on Ubuntu 20.04, which PyTorch rejects. Install one, then re-run:\n"
            "  sudo apt-get install -y g++-11   # or: sudo apt-get install -y clang\n"
            "  export CXX=g++-11               # or: export CXX=clang++\n"
            "  ./smoke.sh"
        )
    os.environ["CXX"] = cxx
    # Keep CC in sync so ninja/ccache don't mix toolchains.
    if cxx.startswith("g++-"):
        gcc = "gcc-" + cxx[len("g++-") :]
        if shutil.which(gcc):
            os.environ.setdefault("CC", gcc)
    elif cxx.startswith("clang++-"):
        clang = "clang-" + cxx[len("clang++-") :]
        if shutil.which(clang):
            os.environ.setdefault("CC", clang)
    elif cxx == "g++":
        os.environ.setdefault("CC", "gcc")
    elif cxx == "clang++":
        os.environ.setdefault("CC", "clang")
    _CXX_TORCH_READY = True


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
    _ensure_cxx_accepts_torch_std_flag()
    node_info = configs["experiment_info_default"]["node_info"]
    env_name = node_info["node_type"]
    # esdf_bev is flattened to a 2D BEV in Python; the Environment only knows
    # elevation vs live esdf.
    if str(node_info.get("map_type", "elevation")).lower() == "esdf_bev":
        node_info["map_type"] = "elevation"
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()

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

    # Generic, OMPL-style environment: header-only, dims fixed at compile time
    # via -DN_DIMS / -DN_CONT / -DHASH_DIMS read from the config. The user's
    # dynamics/cost/validity/heuristic/goal_test/sample_controls callables are
    # passed through the `configs` dict to the Environment constructor.
    if env_name == "generic":
        info = configs["experiment_info_default"]
        state_dim = int(info["state_dim"])
        control_dim = int(info["control_dim"])
        hash_dims = int(info.get("hash_dims", state_dim))
        # Do not pass -std=c++17: Torch's cpp_extension already adds -std=c++20, and a later
        # -std= flag wins on GCC, which demotes the build back to C++17 and breaks Torch headers.
        extra_cflags = [
            "-O3",
            # Silence benign pybind visibility warnings from py::function members.
            "-Wno-attributes",
            "-DUSE_GENERIC_ENV",
            f"-DN_DIMS={state_dim}",
            f"-DN_CONT={control_dim}",
            f"-DHASH_DIMS={hash_dims}",
        ]
        # Distinct module name per (state_dim, control_dim, hash_dims) so builds
        # with different dimensions do not collide in the torch extension cache.
        kernel = load(
            name=f"ighastar_generic_{state_dim}_{control_dim}_{hash_dims}",
            sources=[str(cpp_path)],
            extra_include_paths=extra_includes,
            extra_cflags=extra_cflags,
            verbose=True,
        )
        if bidirectional:
            return kernel.BiIGHAStar(configs, False)
        return kernel.IGHAStar(configs, False)

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

    if env_name != "simple":
        if cuda_available:
            # Use CUDA version
            cuda_path = BASE_DIR / "src" / "Environments" / "src" / f"{env_name}.cu"
            kernel = load(
                name="ighastar",
                sources=[str(cpp_path), str(cuda_path)],
                extra_include_paths=extra_includes,
                extra_cflags=["-O3", env_macro],
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
                extra_cflags=["-O3", env_macro],
                verbose=True,
            )
    else:
        kernel = load(
            name="ighastar",
            sources=[str(cpp_path)],
            extra_include_paths=extra_includes,
            extra_cflags=["-O3", env_macro],
            verbose=True,
        )

    if bidirectional:
        planner = kernel.BiIGHAStar(configs, False)
    else:
        planner = kernel.IGHAStar(configs, False)
    return planner
