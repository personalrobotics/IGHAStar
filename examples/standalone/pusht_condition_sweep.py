#!/usr/bin/env python3
"""Shardable with/without-heuristic sweep for pusht and pusht_multi.

The orchestrator in sweep.sh launches one process per shard. This file does
not start its own workers.

  --dry-run   print the resolved run and a line "TOTAL <n>", then exit
  --warmup    construct the planner once so later shards do not compile together
  --save-trees  pack the search tree into the result npz
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from ighastar.scripts.common_utils import create_planner

COST_INF = 1e4


def difficulty_order(n: int, rank_csv: Path) -> np.ndarray:
    """Easiest first, using the ndir=16 chipmunk cell that matches pusht.yml."""
    if not rank_csv.is_file():
        return np.arange(n)
    rows = [
        r
        for r in csv.DictReader(rank_csv.open())
        if r["dynamics"] == "chipmunk"
        and r["res_xy"] == "40.0"
        and r["action_radii"] == "200"
        and r["max_agent_vel"] == "500.0"
        and r["n_directions"] == "16"
        and r["n_substeps"] == "10"
    ]
    rank = {}
    for r in rows:
        i = int(r["problem_idx"])
        first = int(float(r["first_sol_exp"]))
        rank[i] = first if int(r["success"]) and first >= 0 else 10**9
    return np.array(sorted(range(n), key=lambda i: (rank.get(i, 10**9), i)))


def order_indices(data, rank_csv: Path | None) -> np.ndarray:
    n = len(data["starts"])
    if "first_sol_expansions" in data.files:
        first = np.asarray(data["first_sol_expansions"])
        keyed = []
        for i, value in enumerate(first):
            fv = int(value)
            keyed.append((fv if fv >= 0 else 10**9, i))
        return np.array([i for _, i in sorted(keyed)])
    if rank_csv is not None:
        return difficulty_order(n, rank_csv)
    return np.arange(n)


def even_sample(order: np.ndarray, n_take: int) -> np.ndarray:
    if len(order) <= n_take:
        return order
    idx = np.unique(np.round(np.linspace(0, len(order) - 1, n_take)).astype(int))
    return order[idx]


def load_cfg(path: Path, use_heuristic: bool, max_exp: int):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    info = cfg["experiment_info_default"]
    info["max_expansions"] = int(max_exp)
    info["node_info"]["use_heuristic"] = bool(use_heuristic)
    return cfg


def search(planner, start, goal, world, limit, hyst):
    ok = bool(
        planner.search(
            torch.as_tensor(start, dtype=torch.float32),
            torch.as_tensor(goal, dtype=torch.float32),
            torch.as_tensor(world, dtype=torch.float32),
            int(limit),
            int(hyst),
            True,
        )
    )
    path = planner.get_best_path()
    if ok and path.numel() == 0:
        ok = False
    curve = np.asarray(planner.get_profiler_info()[9], dtype=np.float64)
    return ok, curve


def save_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.npz")
    np.savez(tmp, **payload)
    os.replace(tmp, path)


def parse_conditions(text: str) -> list[str]:
    names = [part.strip() for part in text.split(",") if part.strip()]
    if not names or any(name not in ("with", "without") for name in names):
        raise SystemExit(f"conditions must be with and/or without, got {text!r}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--max-expansions", type=int, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--save-trees", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--conditions", default="with,without")
    parser.add_argument("--rank-csv", type=Path, default=None)
    args = parser.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard_id < args.num_shards):
        raise SystemExit(
            f"bad shard: shard-id={args.shard_id} num-shards={args.num_shards}"
        )
    if args.n < 1 or args.max_expansions < 1:
        raise SystemExit("--n and --max-expansions must be >= 1")
    conditions = parse_conditions(args.conditions)

    data = np.load(args.problems, allow_pickle=True)
    chosen = even_sample(order_indices(data, args.rank_csv), args.n)
    mine = [
        int(idx)
        for pos, idx in enumerate(chosen)
        if pos % args.num_shards == args.shard_id
    ]
    shown = load_cfg(args.config, True, args.max_expansions)["experiment_info_default"]
    shown["node_info"].pop("use_heuristic", None)

    print(f"config {args.config}", flush=True)
    print(yaml.safe_dump({"experiment_info_default": shown}, sort_keys=False), end="", flush=True)
    print(
        f"problems {args.problems}  chosen {len(chosen)} of {len(data['starts'])}",
        flush=True,
    )
    print(f"conditions {','.join(conditions)}  save_trees {int(args.save_trees)}", flush=True)
    print(f"shard {args.shard_id}/{args.num_shards} indices {' '.join(map(str, mine))}", flush=True)
    for name in conditions:
        print(f"out {name} {args.out / name}", flush=True)
    print(f"TOTAL {len(chosen) * len(conditions)}", flush=True)

    if args.dry_run:
        return
    if args.warmup:
        create_planner(load_cfg(args.config, conditions[0] == "with", args.max_expansions), False)
        print("warmup ok", flush=True)
        return

    hyst = int(shown["hysteresis"])
    worlds = data["obstacles"] if "obstacles" in data.files else None
    for name in conditions:
        cfg = load_cfg(args.config, name == "with", args.max_expansions)
        planner = create_planner(cfg, bidirectional=False)
        dest = args.out / name
        for k, i in enumerate(mine):
            path = dest / f"p{i:03d}.npz"
            if path.is_file():
                print(f"DONE {name} {i} cached", flush=True)
                continue
            world = worlds[i] if worlds is not None else np.zeros(1, np.float32)
            ok, curve = search(
                planner, data["starts"][i], data["goals"][i], world,
                args.max_expansions, hyst,
            )
            payload = {
                "best_cost": np.asarray(curve, dtype=np.float64),
                "problem_idx": np.int32(i),
                "success": np.uint8(ok),
            }
            n_nodes = -1
            if args.save_trees:
                tree = planner.export_search_tree_arrays()
                n_nodes = int(tree["n_nodes"])
                payload.update({key: np.asarray(tree[key]) for key in tree})
                del tree
            save_result(path, payload)
            del payload
            hit = np.where(curve < COST_INF)[0]
            first = int(hit[0]) if hit.size else -1
            print(
                f"DONE {name} {i} ok={ok} first={first} nodes={n_nodes} "
                f"{k + 1}/{len(mine)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
