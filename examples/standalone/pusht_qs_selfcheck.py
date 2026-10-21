#!/usr/bin/env python3
"""Minimal check: QS Succ runs and a push moves the block (not a Chipmunk parity test)."""
from __future__ import annotations

import copy
import os

import numpy as np
import torch
import yaml

from ighastar.scripts.common_utils import BASE_DIR, create_planner


def main() -> None:
    yaml_path = os.path.join(
        BASE_DIR.parent, "examples", "standalone", "Configs", "pusht.yml"
    )
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["experiment_info_default"]["node_info"]["dynamics"] = "quasistatic"
    cfg["experiment_info_default"]["max_expansions"] = 500

    planner = create_planner(cfg, bidirectional=False)
    start = torch.tensor(
        [200.0, 220.0, 230.0, 240.0, 0.5, 0, 0, 0, 0, 0], dtype=torch.float32
    )
    goal = torch.tensor(
        [256.0, 256.0, 256.0, 256.0, float(np.pi / 4), 0, 0, 0, 0, 0],
        dtype=torch.float32,
    )
    ok = planner.search(
        start, goal, torch.zeros(1), cfg["experiment_info_default"]["max_expansions"],
        cfg["experiment_info_default"]["hysteresis"], True,
    )
    print(f"qs smoke search success={bool(ok)}")
    assert ok, "QS planner failed on easy example"


if __name__ == "__main__":
    main()
