import torch
from torch.utils.cpp_extension import load
import pathlib
import os
import subprocess
from typing import Any, Dict
import sys


def _cxx_std_flag() -> str:
    """C++20 dialect flag. GCC 9 needs -std=c++2a; GCC 10+ / Clang accept -std=c++20."""
    override = os.environ.get("IGHASTAR_CXX_STD")
    if override:
        return override if override.startswith("-") else f"-std={override}"
    try:
        ver = subprocess.check_output(
            ["c++", "-dumpversion"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        major = int(ver.split(".")[0])
        if major < 10:
            return "-std=c++2a"
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return "-std=c++20"


def _cuda_cxx_std_flag(host_flag: str) -> str:
    """nvcc dialect: rejects GCC's c++2a alias; CUDA 12 accepts c++17/c++20."""
    override = os.environ.get("IGHASTAR_CUDA_CXX_STD")
    if override:
        return override if override.startswith("-") else f"-std={override}"
    # Host GCC9 uses c++2a; map to the nearest nvcc-known dialect.
    if "c++2a" in host_flag or "c++20" in host_flag:
        return "-std=c++17"
    return host_flag


CXX_STD_FLAG = _cxx_std_flag()
CUDA_CXX_STD_FLAG = _cuda_cxx_std_flag(CXX_STD_FLAG)


def env_header_flag(header: str) -> str:
    """-DIGHASTAR_ENV_HEADER=... with quotes that survive ninja's shell.

    header is the include name only, e.g. kinodynamic.h or my_env.h
    (found on the include path).
    """
    # Macro value must be the tokens `"foo.h"` after the shell strips quotes.
    return "-DIGHASTAR_ENV_HEADER='" + chr(34) + header + chr(34) + "'"


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
    # Optional MIT robin_hood backend (utils/hash_map.h). Default: on when present.
    # Bundled under ighastar/third_party/robin_hood (pip install clones if missing).
    # Override ROBIN_HOOD_DIR; disable with IGHA_NO_ROBIN_HOOD=1.
    from ighastar.scripts.fetch_robin_hood import ensure_robin_hood

    hash_cflags: list = []
    hash_tag = "std"
    robin_hood_include = ensure_robin_hood(quiet=True)
    if robin_hood_include is not None:
        extra_includes.append(str(robin_hood_include))
        hash_cflags = ["-DIGHA_USE_ROBIN_HOOD"]
        hash_tag = "rh"
        print(f"hash maps: robin_hood ({robin_hood_include})")
    else:
        print("hash maps: std::unordered_*")
    # Include dialect + hash backend so torch extension cache doesn't reuse old ABIs.
    cxx_std = CXX_STD_FLAG
    cuda_std = CUDA_CXX_STD_FLAG
    std_tag = cxx_std.replace("-std=", "").replace("+", "x")
    build_tag = f"{std_tag}_{hash_tag}"

    # Detect if running on macOS for Boost include path
    is_macos = sys.platform == "darwin"
    if is_macos:
        boost_include = "/opt/homebrew/opt/boost/include"  # Adjust if needed
        extra_includes.append(boost_include)

    # Generic, OMPL-style environment: header-only, dims fixed at compile time
    # via -DN_DIMS / -DN_CONT / -DHASH_DIMS read from the config. The user's
    # dynamics/cost/validity/heuristic/goal_test/sample_controls callables are
    # passed through the `configs` dict to the Environment constructor.
    if env_name == "generic":
        info = configs["experiment_info_default"]
        state_dim = int(info["state_dim"])
        control_dim = int(info["control_dim"])
        hash_dims = int(info.get("hash_dims", state_dim))
        extra_cflags = [
            cxx_std,
            "-O3",
            # Silence benign pybind visibility warnings from py::function members.
            "-Wno-attributes",
            env_header_flag("generic.h"),
            f"-DN_DIMS={state_dim}",
            f"-DN_CONT={control_dim}",
            f"-DHASH_DIMS={hash_dims}",
            *hash_cflags,
        ]
        # Distinct module name per (state_dim, control_dim, hash_dims) so builds
        # with different dimensions do not collide in the torch extension cache.
        kernel = load(
            name=f"ighastar_generic_{state_dim}_{control_dim}_{hash_dims}_{build_tag}_eh",
            sources=[str(cpp_path)],
            extra_include_paths=extra_includes,
            extra_cflags=extra_cflags,
            verbose=True,
        )
        if bidirectional:
            return kernel.BiIGHAStar(configs, False)
        return kernel.IGHAStar(configs, False)

    # Generic environment with *compiled* torch callbacks: same API as
    # "generic", but each callback is exported with torch.export and compiled
    # ahead of time with AOTInductor (cached on disk), then run directly from
    # C++ (generic_torch.h) -- the search never calls back into Python.
    if env_name == "generic_torch":
        from .generic_torch import compile_generic_torch_callbacks

        info = configs["experiment_info_default"]
        state_dim = int(info["state_dim"])
        control_dim = int(info["control_dim"])
        hash_dims = int(info.get("hash_dims", state_dim))
        compiled = compile_generic_torch_callbacks(configs)
        extra_cflags = [
            cxx_std,
            "-O3",
            "-Wno-attributes",
            env_header_flag("generic_torch.h"),
            "-DIGHASTAR_RELEASES_GIL",
            f"-DN_DIMS={state_dim}",
            f"-DN_CONT={control_dim}",
            f"-DHASH_DIMS={hash_dims}",
            *hash_cflags,
        ]
        kernel = load(
            name=f"ighastar_generic_torch_{state_dim}_{control_dim}_{hash_dims}_{build_tag}_eh",
            sources=[str(cpp_path)],
            extra_include_paths=extra_includes,
            extra_cflags=extra_cflags,
            verbose=True,
        )
        # Shallow copy: the compiled package paths go to the C++ Environment
        # without mutating the caller's config dict.
        planner_configs = dict(configs)
        planner_configs["_generic_torch"] = compiled
        if bidirectional:
            return kernel.BiIGHAStar(planner_configs, False)
        return kernel.IGHAStar(planner_configs, False)

    # PushT: no backward/predecessor dynamics yet (Succ always steps +dt).
    if bidirectional and env_name in ("pusht", "pusht_qs_cuda", "pusht_qs"):
        raise NotImplementedError(
            "PushT backward dynamics are not implemented; use bidirectional=False."
        )

    # PushT QS on CUDA (Mason/Lynch) — A/B vs CPU pusht + dynamics: quasistatic.
    if env_name in ("pusht_qs_cuda", "pusht_qs"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "node_type pusht_qs_cuda requires CUDA; "
                "use node_type: pusht with dynamics: quasistatic for CPU QS."
            )
        cuda_path = BASE_DIR / "src" / "Environments" / "src" / "pusht_qs.cu"
        kernel = load(
            name=f"ighastar_pusht_qs_cuda_{build_tag}_eh",
            sources=[str(cpp_path), str(cuda_path)],
            extra_include_paths=extra_includes,
            extra_cflags=[
                cxx_std,
                "-O3",
                env_header_flag("pusht_qs_cuda.h"),
                *hash_cflags,
            ],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                cuda_std,
                *hash_cflags,
            ],
            verbose=True,
        )
        if bidirectional:
            return kernel.BiIGHAStar(configs, False)
        return kernel.IGHAStar(configs, False)

    # PushT: Chipmunk CPU env (header + vendored Chipmunk sources). No CUDA.
    if env_name == "pusht":
        chipmunk_root = (
            pathlib.Path(os.environ["PUSHT_CHIPMUNK_DIR"])
            if os.environ.get("PUSHT_CHIPMUNK_DIR")
            else BASE_DIR.parent.parent / "pusht_ext" / "Chipmunk2D"
        )
        if not (chipmunk_root / "include" / "chipmunk" / "chipmunk.h").exists():
            raise FileNotFoundError(
                f"Chipmunk not found at {chipmunk_root}. "
                "Clone Chipmunk 7 into pusht_ext/Chipmunk2D or set PUSHT_CHIPMUNK_DIR."
            )
        chipmunk_sources = sorted(
            p
            for p in (chipmunk_root / "src").glob("*.c")
            if p.name != "cpHastySpace.c"
        )
        kernel = load(
            name=f"ighastar_pusht_v2_{build_tag}_eh",
            sources=[str(cpp_path), *[str(p) for p in chipmunk_sources]],
            extra_include_paths=extra_includes + [str(chipmunk_root / "include")],
            # Chipmunk .c files share cflags with the .cpp; last -std= wins over
            # torch's default -std=c++17.
            extra_cflags=[
                cxx_std,
                "-O3",
                "-fopenmp",
                "-DNDEBUG",
                "-DCP_USE_DOUBLES=1",
                env_header_flag("pusht.h"),
                *hash_cflags,
            ],
            extra_ldflags=["-fopenmp", "-lm"],
            verbose=True,
        )
        if bidirectional:
            return kernel.BiIGHAStar(configs, False)
        return kernel.IGHAStar(configs, False)

    if not cuda_available:
        print("CUDA not available, using CPU versions")
        env_header = {
            "simple": "simple.h",
            "kinematic": "kinematic_cpu.h",
            "kinodynamic": "kinodynamic_cpu.h",
        }[env_name]
    else:
        print("CUDA available, using GPU versions")
        env_header = {
            "simple": "simple.h",
            "kinematic": "kinematic.h",
            "kinodynamic": "kinodynamic.h",
        }[env_name]
    env_macro = env_header_flag(env_header)

    # Per-env module name: simple/kinematic/kinodynamic must not share one .so
    # (torch caches by name; colliding macros silently load the wrong Env).
    # "_eh" = IGHASTAR_ENV_HEADER era; avoids reusing old USE_*_ENV builds.
    mod_name = f"ighastar_{env_name}_{build_tag}_eh"
    if env_name != "simple":
        if cuda_available:
            # Use CUDA version
            cuda_path = BASE_DIR / "src" / "Environments" / "src" / f"{env_name}.cu"
            kernel = load(
                name=mod_name,
                sources=[str(cpp_path), str(cuda_path)],
                extra_include_paths=extra_includes,
                extra_cflags=[cxx_std, "-O3", env_macro, *hash_cflags],
                extra_cuda_cflags=["-O3", cuda_std, *hash_cflags],
                verbose=True,
            )
        else:
            # Use CPU version - compile with CPU header and .cpp file included
            cpu_cpp_path = (
                BASE_DIR / "src" / "Environments" / "src" / f"{env_name}_cpu.cpp"
            )
            kernel = load(
                name=mod_name,
                sources=[str(cpp_path), str(cpu_cpp_path)],
                extra_include_paths=extra_includes,
                extra_cflags=[cxx_std, "-O3", env_macro, *hash_cflags],
                verbose=True,
            )
    else:
        kernel = load(
            name=mod_name,
            sources=[str(cpp_path)],
            extra_include_paths=extra_includes,
            extra_cflags=[cxx_std, "-O3", env_macro, *hash_cflags],
            verbose=True,
        )

    if bidirectional:
        planner = kernel.BiIGHAStar(configs, False)
    else:
        planner = kernel.IGHAStar(configs, False)
    return planner
