#!/usr/bin/env python3
"""Smoke-test CUDA QS env (pusht_qs_cuda). CPU QS: pusht_qs_selfcheck.py.

A/B: same YAML knobs; switch node_type pusht+dynamics:quasistatic vs pusht_qs_cuda.
Cannot load both torch extensions in one process (pybind IGHAStar name clash).
"""
from __future__ import annotations

import copy
import os
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from ighastar.scripts.common_utils import BASE_DIR, create_planner


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required for pusht_qs_cuda selfcheck"

    # CPU QS in a child process so this process can load the CUDA extension.
    cpu_script = os.path.join(
        BASE_DIR.parent, "examples", "standalone", "pusht_qs_selfcheck.py"
    )
    print("=== cpu_qs (subprocess) ===")
    subprocess.check_call([sys.executable, cpu_script])

    yaml_path = os.path.join(
        BASE_DIR.parent, "examples", "standalone", "Configs", "pusht_qs_cuda.yml"
    )
    with open(yaml_path) as f:
        cfg = copy.deepcopy(yaml.safe_load(f))
    cfg["experiment_info_default"]["max_expansions"] = 500

    print("=== cuda_qs ===")
    t0 = time.perf_counter()
    planner = create_planner(cfg, bidirectional=False)
    build_s = time.perf_counter() - t0
    start = torch.tensor(
        [200.0, 220.0, 230.0, 240.0, 0.5, 0, 0, 0, 0, 0], dtype=torch.float32
    )
    goal = torch.tensor(
        [256.0, 256.0, 256.0, 256.0, float(np.pi / 4), 0, 0, 0, 0, 0],
        dtype=torch.float32,
    )
    info = cfg["experiment_info_default"]
    t1 = time.perf_counter()
    ok = planner.search(
        start, goal, torch.zeros(1), info["max_expansions"], info["hysteresis"], True
    )
    search_s = time.perf_counter() - t1
    print(f"cuda_qs: success={bool(ok)}  build={build_s:.2f}s  search={search_s:.3f}s")
    assert ok, "CUDA QS planner failed on easy example"


if __name__ == "__main__":
    main()
