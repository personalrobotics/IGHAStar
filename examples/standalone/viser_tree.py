#!/usr/bin/env python3
"""Export an IGHA* search tree and scrub its growth in Viser.

Shows how to call ``export_search_tree()`` and draw the flat arrays.
``edges[i]`` is the poses from the parent to vertex ``i``. Green is
``time_direction`` 1. Purple is -1. The slider keeps edges whose
``discover_index`` is at most the slider value. The black solution path
appears once ``cost_exp_list`` first drops below 1e4.

    # Plan, export, and scrub (bidirectional by default)
    python3 viser_tree.py -c Configs/kinodynamic_example.yml --test-case case1

    # Save for later
    python3 viser_tree.py -c Configs/kinodynamic_example.yml --test-case case1 \
        --save ../../Content/standalone/race-2_tree.npz

    # Load a saved tree (skip planning)
    python3 viser_tree.py -c Configs/kinodynamic_example.yml \
        --tree ../../Content/standalone/race-2_tree.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import get_map, get_map_type, resolve_map_dir, terrain_layers
from viser_replay import add_terrain
from ighastar.scripts.common_utils import create_planner

FORWARD_COLOR = (40, 200, 90)
BACKWARD_COLOR = (160, 80, 200)
SOLUTION_COLOR = (0, 0, 0)


def sample_height(elevation: np.ndarray, x: Any, y: Any, map_res: float) -> np.ndarray:
    """Terrain height lookup (same truncation as the CUDA query)."""
    ny, nx = elevation.shape
    res_inv = np.float32(1.0 / map_res)
    ix = np.clip((np.asarray(x, dtype=np.float32) * res_inv).astype(np.int32), 0, nx - 1)
    iy = np.clip((np.asarray(y, dtype=np.float32) * res_inv).astype(np.int32), 0, ny - 1)
    return elevation[iy, ix]


def first_solution_expansion(cost_exp_list: List[float], threshold: float = 1e4) -> int:
    """First expansion index where Omega drops below ``threshold`` (same as example.py)."""
    cost = np.asarray(cost_exp_list, dtype=np.float64)
    hits = np.where(cost < threshold)[0]
    return int(hits[0]) if len(hits) else -1


_ARRAYS = (
    "poses",
    "parents",
    "g",
    "f",
    "levels",
    "status",
    "has_edge",
    "edges",
    "edge_controls",
    "time_direction",
    "discover_index",
    "expand_index",
)


def _tree_arrays(tree: Any) -> Dict[str, Any]:
    """Copy one SearchTree into numpy arrays. The C++ object is not saved."""
    out: Dict[str, Any] = {name: np.asarray(getattr(tree, name)).copy() for name in _ARRAYS}
    out["num_nodes"] = int(tree.num_nodes)
    out["state_dim"] = int(tree.state_dim)
    out["expansions"] = int(tree.expansions)
    out["best_index"] = int(tree.best_index)
    out["best_cost"] = float(tree.best_cost)
    out["expansion_count"] = int(tree.expansions)
    return out


def load_tree(path: str) -> Dict[str, Any]:
    """Load a tree previously written by ``save_tree``."""
    with np.load(path) as data:
        kind = str(data["kind"])
        sol = np.array(data["solution_path"])
        found = int(data["solution_found_at"])
        if kind == "bi":
            tree: Dict[str, Any] = {
                "expansion_count": int(data["expansion_count"]),
                "forward": {},
                "backward": {},
            }
            for key in data.files:
                if "." not in key:
                    continue
                side, name = key.split(".", 1)
                tree[side][name] = np.array(data[key])
        else:
            tree = {
                key: np.array(data[key])
                for key in data.files
                if key not in ("kind", "solution_path", "solution_found_at")
            }
            tree["expansion_count"] = int(tree["expansion_count"])
        tree["solution_path"] = None if sol.shape[0] == 0 else sol
        tree["solution_found_at"] = found
    return tree


def save_tree(tree: Dict[str, Any], path: str) -> None:
    """Write the tree arrays. A SearchTree object is not pickled."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload: Dict[str, Any] = {}
    if "forward" in tree:
        payload["kind"] = np.array("bi")
        payload["expansion_count"] = np.int64(tree["expansion_count"])
        for side in ("forward", "backward"):
            for key, value in tree[side].items():
                payload[f"{side}.{key}"] = np.asarray(value)
    else:
        payload["kind"] = np.array("uni")
        for key, value in tree.items():
            if key in ("solution_path", "solution_found_at"):
                continue
            payload[key] = np.asarray(value)
    sol = tree.get("solution_path")
    payload["solution_path"] = (
        np.zeros((0, 2), dtype=np.float32) if sol is None else np.asarray(sol, dtype=np.float32)
    )
    payload["solution_found_at"] = np.int64(tree.get("solution_found_at", -1))
    np.savez(path, **payload)


def plan_and_export(
    configs: Dict[str, Any],
    start: torch.Tensor,
    goal: torch.Tensor,
    bitmap: torch.Tensor,
    bidirectional: bool,
) -> Dict[str, Any]:
    """Run search and return the exported tree dict, plus solution path metadata."""
    experiment_info = configs["experiment_info_default"]
    node_type = experiment_info["node_info"]["node_type"]
    default_limit = experiment_info["max_expansions"]
    hysteresis = experiment_info["hysteresis"]
    if bidirectional:
        expansion_limit = 1000 if node_type in ("kinodynamic", "kinematic") else 5000
    else:
        expansion_limit = default_limit

    planner = create_planner(configs, bidirectional=bidirectional)
    success = planner.search(start, goal, bitmap, expansion_limit, hysteresis, False)
    print(f"Search done (success={success}), exporting tree...")

    exported = planner.export_search_tree()
    if bidirectional:
        tree = {
            "forward": _tree_arrays(exported["forward"]),
            "backward": _tree_arrays(exported["backward"]),
            "expansion_count": int(exported["expansion_count"]),
        }
    else:
        tree = _tree_arrays(exported)
    *_, cost_exp_list = planner.get_profiler_info()
    found_at = first_solution_expansion(cost_exp_list)
    if success and found_at >= 0:
        # get_best_path() is goal-first (x, y, ..., time, g).
        path = np.asarray(planner.get_best_path(), dtype=np.float32)
        tree["solution_path"] = path
        tree["solution_found_at"] = found_at
        print(f"Solution found at expansion {found_at}, path states={len(path)}")
    else:
        tree["solution_path"] = None
        tree["solution_found_at"] = -1
        print("No solution path to overlay")
    return tree


def _sides(tree: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    """One SearchTree for a unidirectional export. Two for bidirectional."""
    if "forward" in tree and "backward" in tree:
        return [tree["forward"], tree["backward"]], int(tree["expansion_count"])
    return [tree], int(tree["expansion_count"])


def _segments_from_polyline(
    poses: np.ndarray,
    elevation: np.ndarray,
    map_res: float,
    z_offset: float = 0.12,
) -> np.ndarray:
    """Build (N, 2, 3) segments from an (M, >=2) pose array (x, y, ...)."""
    if poses is None or len(poses) < 2:
        return np.zeros((0, 2, 3), dtype=np.float32)
    x = poses[:, 0].astype(np.float32)
    y = poses[:, 1].astype(np.float32)
    z = sample_height(elevation, x, y, map_res).astype(np.float32) + z_offset
    pts = np.stack([x, y, z], axis=-1)
    return np.stack([pts[:-1], pts[1:]], axis=1).astype(np.float32)


def _segments_for_sides(
    sides: List[Dict[str, Any]],
    expansion: int,
    elevation: np.ndarray,
    map_res: float,
    forward: bool,
) -> np.ndarray:
    """Build (N, 2, 3) segments from edges. Green is time_direction 1."""
    pieces: List[np.ndarray] = []
    for side in sides:
        direction = np.asarray(side["time_direction"])
        keep = direction >= 0 if forward else direction < 0
        keep &= np.asarray(side["has_edge"]).astype(bool)
        keep &= np.asarray(side["discover_index"]) <= expansion
        for edge in np.asarray(side["edges"])[keep]:
            if len(edge) < 2:
                continue
            z = sample_height(elevation, edge[:, 0], edge[:, 1], map_res).astype(np.float32)
            pts = np.stack(
                [edge[:, 0], edge[:, 1], z + np.float32(0.08)], axis=-1
            ).astype(np.float32)
            pieces.append(np.stack([pts[:-1], pts[1:]], axis=1))
    if not pieces:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.concatenate(pieces, axis=0)


def _set_segments(handle: Any, pts: np.ndarray, color: Tuple[int, int, int]) -> None:
    if len(pts):
        colors = np.zeros_like(pts, dtype=np.uint8)
        colors[...] = np.asarray(color, dtype=np.uint8)
        handle.colors = colors
        handle.points = pts
        handle.visible = True
    else:
        handle.visible = False


def playback_tree(
    tree: Dict[str, Any],
    costmap: np.ndarray,
    elevation: np.ndarray,
    map_res: float,
    port: int = 8080,
    stride: int = 2,
) -> None:
    """Scrub tree growth over elevation+cost terrain; green fwd, purple bwd, black solution."""
    import viser

    sides, expansion_count = _sides(tree)
    n_nodes = sum(len(np.asarray(side["time_direction"])) for side in sides)
    n_fwd = sum(int((np.asarray(side["time_direction"]) >= 0).sum()) for side in sides)
    n_bwd = n_nodes - n_fwd
    solution_path = tree.get("solution_path")
    if solution_path is not None:
        solution_path = np.asarray(solution_path, dtype=np.float32)
    solution_found_at = int(tree.get("solution_found_at", -1))
    print(
        f"Tree: {n_nodes} nodes (fwd={n_fwd}, bwd={n_bwd}), "
        f"expansion_count={expansion_count}, "
        f"solution_found_at={solution_found_at}"
    )

    server = viser.ViserServer(port=port)
    add_terrain(server, costmap, elevation, map_res, stride=stride)

    handles = {
        "forward": server.scene.add_line_segments(
            "/tree/forward",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=FORWARD_COLOR,
            line_width=2.5,
        ),
        "backward": server.scene.add_line_segments(
            "/tree/backward",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=BACKWARD_COLOR,
            line_width=2.5,
        ),
        "solution": server.scene.add_line_segments(
            "/tree/solution",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=SOLUTION_COLOR,
            line_width=4.0,
        ),
    }
    color_by_name = {
        "forward": FORWARD_COLOR,
        "backward": BACKWARD_COLOR,
        "solution": SOLUTION_COLOR,
    }
    solution_segs = _segments_from_polyline(solution_path, elevation, map_res)

    with server.gui.add_folder("Tree growth"):
        slider = server.gui.add_slider(
            "Expansion",
            min=0,
            max=max(expansion_count, 0),
            step=1,
            initial_value=max(expansion_count, 0),
        )
        readout = server.gui.add_markdown("")

    def redraw(expansion: int) -> None:
        visible = 0
        for name, is_forward in (("forward", True), ("backward", False)):
            pts = _segments_for_sides(
                sides, expansion, elevation, map_res, forward=is_forward
            )
            _set_segments(handles[name], pts, color_by_name[name])
            visible += len(pts)

        show_sol = (
            solution_found_at >= 0
            and expansion >= solution_found_at
            and len(solution_segs) > 0
        )
        _set_segments(
            handles["solution"],
            solution_segs if show_sol else np.zeros((0, 2, 3), dtype=np.float32),
            SOLUTION_COLOR,
        )

        sol_note = (
            f"solution **on** (found @ {solution_found_at})"
            if show_sol
            else (
                f"solution **off** (appears @ {solution_found_at})"
                if solution_found_at >= 0
                else "solution **none**"
            )
        )
        readout.content = (
            f"expansion **{expansion}** / {expansion_count}  \n"
            f"tree segments **{visible}**  \n"
            f"{sol_note}  \n"
            f"green = forward, purple = backward, black = solution"
        )

    @slider.on_update
    def _(_event: Any) -> None:
        redraw(int(slider.value))

    redraw(int(slider.value))
    print(f"Viser tree playback on http://localhost:{port}")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Export / scrub an IGHA* search tree")
    parser.add_argument("-c", "--config", required=True, help="YAML config path")
    parser.add_argument("--test-case", default=None, help="Named start/goal test case")
    parser.add_argument(
        "--bidirectional",
        dest="bidirectional",
        action="store_true",
        default=True,
        help="Use BiIGHAStar (default: true)",
    )
    parser.add_argument(
        "--no-bidirectional",
        dest="bidirectional",
        action="store_false",
        help="Use unidirectional IGHAStar",
    )
    parser.add_argument("--tree", default=None, help="Load an .npz tree and skip planning")
    parser.add_argument("--save", default=None, help="Write the exported tree arrays here")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)

    map_info = configs["map"]
    node_info = configs["experiment_info_default"]["node_info"]
    map_res = float(node_info["map_res"])

    map_dir = resolve_map_dir(map_info["dir"])
    bitmap = get_map(
        map_info["name"],
        map_dir=map_dir,
        map_size=map_info["size"],
        node_info=node_info,
    )
    costmap, elevation = terrain_layers(bitmap, node_info)

    if args.tree:
        tree = load_tree(args.tree)
        print(f"Loaded tree from {args.tree}")
    else:
        if args.test_case and "test_cases" in map_info and args.test_case in map_info["test_cases"]:
            tc = map_info["test_cases"][args.test_case]
            start = torch.tensor(tc["start"], dtype=torch.float32)
            goal = torch.tensor(tc["goal"], dtype=torch.float32)
        else:
            start = torch.tensor(map_info["start"], dtype=torch.float32)
            goal = torch.tensor(map_info["goal"], dtype=torch.float32)
        print(
            f"Planning ({'BiIGHAStar' if args.bidirectional else 'IGHAStar'}), "
            f"map_type={get_map_type(node_info)}"
        )
        tree = plan_and_export(configs, start, goal, bitmap, args.bidirectional)
        if args.save:
            save_tree(tree, args.save)
            print(f"Saved tree to {args.save}")

    playback_tree(tree, costmap, elevation, map_res, port=args.port)


if __name__ == "__main__":
    main()
