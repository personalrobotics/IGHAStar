#!/usr/bin/env python3
"""Farm solved split-agent 2T1W2A clear problems with the best sweep cell.

3 worker processes x 8 OpenMP threads. Each worker samples its own seeded starts, keeps
the ones the planner solves within the expansion limit, and writes a video per solve.
The parent merges the shards (trimmed to --n) into one npz.

  python3 examples/standalone/pusht_multi_farm_split_clear.py --n 100 --expansions 300000 \
      --res-xy 40 --radii 200 --vmax 500 --ndir 24 --nsub 10
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pusht_multi_generate_2T1O_clear import CONTENT, WORLD_STRIDE, cost_expansion  # noqa: E402
from pusht_multi_generate_2T1W2A_split_clear import (  # noqa: E402
    CONFIG, N_DIMS, render_video, sample_candidate,
)
import pusht_multi_split_sweep as sweep  # noqa: E402
from ighastar.scripts.common_utils import create_planner  # noqa: E402

OUT = os.path.join(CONTENT, "pusht_multi_2T1W2A_split_clear_farm.npz")
VIDEOS = os.path.join(CONTENT, "pusht_multi_2T1W2A_split_clear_farm_videos")
KEYS = ("starts", "worlds", "seeds", "paths", "cost_exp", "first")


def shard_path(k):
    return OUT.replace(".npz", f"_shard{k}.npz")


def cell_from(args):
    return dict(res_xy=args.res_xy, radii=tuple(args.radii), vmax=args.vmax,
                n_directions=args.ndir, n_substeps=args.nsub)


def total_kept(args, own):
    """Problems kept across all shards (own shard from memory, others from disk)."""
    n = len(own)
    for k in range(args.workers):
        if k != args.shard and os.path.isfile(shard_path(k)):
            try:
                n += len(np.load(shard_path(k), allow_pickle=True)["starts"])
            except Exception:  # another worker is mid-write; count it next time
                pass
    return n


def worker(args):
    with open(CONFIG) as f:
        cfg = sweep.apply_cell(yaml.safe_load(f), cell_from(args))
    cfg["experiment_info_default"]["max_expansions"] = args.expansions
    planner = create_planner(cfg, bidirectional=False)
    # A respawn after a crash reseeds, so it does not redraw the start that crashed.
    rng = np.random.RandomState(1000 + args.shard + 7919 * args.attempt)
    keep = {k: [] for k in KEYS}
    if os.path.isfile(shard_path(args.shard)):  # resume after a crash
        old = np.load(shard_path(args.shard), allow_pickle=True)
        keep = {k: list(old[k]) for k in KEYS}
    goal = torch.zeros(N_DIMS, dtype=torch.float32)
    planned = draws = 0
    while total_kept(args, keep["starts"]) < args.n:  # shared quota
        batch = [sample_candidate(rng) for _ in range(32)]
        mask = planner.check_validity_batched(
            torch.from_numpy(np.stack([b[0] for b in batch])),
            torch.from_numpy(np.stack([b[1] for b in batch])),
        ).numpy().astype(bool)
        for j, (s, w) in enumerate(batch):
            draws += 1
            if not mask[j] or total_kept(args, keep["starts"]) >= args.n:
                continue
            planned += 1
            ok = bool(planner.search(torch.from_numpy(s), goal, torch.from_numpy(w),
                                     args.expansions, 500, True))
            path = planner.get_best_path()
            first, cost_exp = cost_expansion(planner)
            ok = ok and path.numel() > 0
            if ok:
                p = path.detach().cpu().numpy()
                n = len(keep["starts"])
                for k, v in zip(KEYS, (s, w, 1000000 * args.attempt + 1000 * (args.shard + 1) + draws, p, cost_exp, first)):
                    keep[k].append(v)
                render_video(p, w, os.path.join(VIDEOS, f"s{args.shard}_p{n:03d}_draw{draws}.mp4"))
                np.savez(shard_path(args.shard), **{k: np.array(v, dtype=object) for k, v in keep.items()})
            print(f"[shard {args.shard}] planned {planned} solved={ok} first_sol={first} "
                  f"kept={len(keep['starts'])} total={total_kept(args, keep['starts'])}/{args.n}", flush=True)


def merge(args):
    keep = {k: [] for k in KEYS}
    for k in range(args.workers):
        d = np.load(shard_path(k), allow_pickle=True)
        for key in KEYS:
            keep[key] += list(d[key])
    n = min(args.n, len(keep["starts"]))
    np.savez(
        OUT,
        starts=np.stack(keep["starts"][:n]).astype(np.float32),
        obstacles=np.stack(keep["worlds"][:n]).astype(np.float32).reshape(n, WORLD_STRIDE),
        seeds=np.asarray(keep["seeds"][:n], dtype=np.int64),
        paths=np.array(keep["paths"][:n], dtype=object),
        cost_exp=np.array(keep["cost_exp"][:n], dtype=object),
        first_sol_expansions=np.asarray(keep["first"][:n], dtype=np.int64),
        problem_class=np.array("2T1W2A_clear_split"),
        max_expansions=np.array(args.expansions),
        cell=np.array(sweep.tag(cell_from(args))),
    )
    print(f"wrote {n} problems to {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--expansions", type=int, default=300000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--attempt", type=int, default=0)
    ap.add_argument("--restart", type=int, default=0, help="offset added to the attempt number so a relaunch reseeds")
    ap.add_argument("--res-xy", type=float, default=40.0)
    ap.add_argument("--radii", type=float, nargs="+", default=[200.0])
    ap.add_argument("--vmax", type=float, default=500.0)
    ap.add_argument("--ndir", type=int, default=24)
    ap.add_argument("--nsub", type=int, default=10)
    args = ap.parse_args()
    if args.shard >= 0:
        return worker(args)
    fwd = sys.argv[1:]
    sweep.supervise(lambda k, a: [sys.executable, __file__, *fwd, "--shard", str(k),
                                  "--attempt", str(a + args.restart)], args.workers)
    merge(args)


if __name__ == "__main__":
    main()
