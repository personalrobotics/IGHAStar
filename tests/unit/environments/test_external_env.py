#!/usr/bin/env python3
"""Smoke: USE_EXTERNAL_ENV builds ighastar.cpp against an out-of-tree header."""
import pathlib
import sys

import torch
from torch.utils.cpp_extension import load

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))  # repo root

from ighastar.scripts.common_utils import (  # noqa: E402
    BASE_DIR,
    CXX_STD_FLAG,
)


def test_external_env_compiles_and_constructs():
    src = pathlib.Path(BASE_DIR) / "src"
    # Distinct name so this build never collides with a built-in extension.
    tag = CXX_STD_FLAG.replace("-std=", "").replace("+", "x")
    kernel = load(
        name=f"ighastar_external_stub_2_1_2_{tag}",
        sources=[str(src / "ighastar.cpp")],
        extra_include_paths=[
            str(HERE),  # stub_external_env.h
            str(src / "Environments" / "include"),
            str(src / "utils"),
            str(src),
        ],
        extra_cflags=[
            CXX_STD_FLAG,
            "-O3",
            "-Wno-attributes",
            "-DUSE_EXTERNAL_ENV",
            # Keep the quotes through ninja's shell: value must be `"stub_external_env.h"`.
            "-DEXTERNAL_ENV_HEADER='\"stub_external_env.h\"'",
            "-DN_DIMS=2",
            "-DN_CONT=1",
            "-DHASH_DIMS=2",
        ],
        verbose=False,
    )
    configs = {
        "experiment_info_default": {
            "state_dim": 2,
            "control_dim": 1,
            "hash_dims": 2,
            "num_controls": 1,
            "resolution": [0.5, 0.5],
            "tolerance": [0.1, 0.1],
            "max_level": 2,
            "division_factor": 2.0,
            "max_expansions": 10,
            "hysteresis": 100,
            "preemptive_expansion": {
                "enabled": False,
                "min_preemptive": 1,
                "max_preemptive": 1,
            },
            "node_info": {"node_type": "external_stub", "timesteps": 1},
        }
    }
    planner = kernel.IGHAStar(configs, False)
    # Construction is the check; empty Succ means search cannot reach the goal.
    ok = planner.search(
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.zeros(1),
        10,
        100,
        True,
    )
    assert ok is False


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
