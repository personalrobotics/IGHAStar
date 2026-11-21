#!/usr/bin/env python3
"""The same CPU search, in two fresh processes, must return the same path.

Equal edge costs leave f and g tied. Those ties used to follow pointer
addresses, so a second process took a different path. Each case is
expansion-bounded and uses a small hysteresis, so the search also changes
resolution and ranks equal-g nodes.

Kinematic and kinodynamic are CPU-capable too. They are not in this list:
each one compiles its own extension against the street maps. Add a runner
below to cover another environment.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import pytest
import torch
import yaml

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO))

import generic_problems as gp  # noqa: E402
from ighastar.scripts.common_utils import create_planner  # noqa: E402

STANDALONE = REPO / "examples" / "standalone"
CPU_CASES = ("generic", "simple", "pusht")


def _digest(tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _result(planner, ok) -> dict:
    path, controls = planner.get_best_path_with_controls()
    info = planner.get_profiler_info()
    levels = planner.get_level_history()
    return {
        "ok": bool(ok),
        "expansions": int(info[7]),
        "max_level": max(levels) if levels else 0,
        "path": _digest(path),
        "controls": _digest(controls),
        "rows": int(path.shape[0]),
    }


def _search_generic():
    configs = gp.make_configs(
        "generic", gp.callbacks(), max_expansions=400, hysteresis=4
    )
    planner = create_planner(configs, bidirectional=False)
    ok = gp.run_search(planner, configs, goal=gp.GOAL)
    return _result(planner, ok)


def _load_yaml(name):
    with open(STANDALONE / "Configs" / name) as f:
        return yaml.safe_load(f)


def _search_simple():
    sys.path.insert(0, str(STANDALONE))
    from utils import get_map, resolve_map_dir

    cfg = _load_yaml("simple_example.yml")
    info = cfg["experiment_info_default"]
    info["max_expansions"] = 800
    info["hysteresis"] = 1
    node = info["node_info"]
    bitmap = get_map(
        cfg["map"]["name"],
        resolve_map_dir(cfg["map"]["dir"]),
        cfg["map"]["size"],
        node,
    )
    # Far enough that the search changes resolution, close enough to solve
    # inside the cap. The yaml goal is across the whole map.
    start = [41.0, 141.0]
    goal = [200.0, 200.0]
    planner = create_planner(cfg, bidirectional=False)
    ok = planner.search(
        torch.tensor(start, dtype=torch.float32),
        torch.tensor(goal, dtype=torch.float32),
        bitmap,
        info["max_expansions"],
        info["hysteresis"],
        False,
    )
    return _result(planner, ok)


def _search_pusht():
    cfg = _load_yaml("pusht.yml")
    info = cfg["experiment_info_default"]
    info["max_expansions"] = 8000
    info["hysteresis"] = 40
    start = torch.tensor(
        [131.0, 258.0, 345.82956, 281.74796, 6.1006074, 0, 0, 0, 0, 0]
    )
    goal = torch.zeros(10)
    goal[2:5] = torch.tensor([256.0, 256.0, torch.pi / 4])
    planner = create_planner(cfg, bidirectional=False)
    ok = planner.search(
        start, goal, torch.zeros(1), info["max_expansions"], info["hysteresis"], False
    )
    return _result(planner, ok)


RUNNERS = {
    "generic": _search_generic,
    "simple": _search_simple,
    "pusht": _search_pusht,
}


def _worker(name: str) -> None:
    print(json.dumps(RUNNERS[name]()), flush=True)


def _run_process(name: str) -> dict:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    proc = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "--worker", name],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr[-4000:] or proc.stdout[-4000:])
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1]
    return json.loads(line)


def _available(name: str) -> bool:
    if name == "simple":
        return (
            STANDALONE / "Maps/generated_maps/multi_bottleneck/17_512.png"
        ).is_file()
    if name == "pusht":
        root = os.environ.get("PUSHT_CHIPMUNK_DIR")
        if not root:
            root = str(REPO.parent / "pusht_ext" / "Chipmunk2D")
        return (pathlib.Path(root) / "include" / "chipmunk" / "chipmunk.h").is_file()
    return True


@pytest.mark.parametrize("name", CPU_CASES)
def test_cpu_search_matches_across_processes(name):
    if not _available(name):
        pytest.skip(name + " inputs are not on this machine")
    first = _run_process(name)
    second = _run_process(name)
    assert first["ok"] and second["ok"]
    assert first["max_level"] >= 1 and second["max_level"] >= 1
    assert first["rows"] > 0
    assert first["path"] == second["path"]
    assert first["controls"] == second["controls"]
    assert first["expansions"] == second["expansions"]


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        _worker(sys.argv[2])
