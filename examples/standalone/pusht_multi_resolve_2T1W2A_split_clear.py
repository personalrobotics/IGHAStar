#!/usr/bin/env python3
"""Re-solve the saved 2T1W2A split-clear starts with a different hysteresis.

  python3 examples/standalone/pusht_multi_resolve_2T1W2A_split_clear.py --hysteresis 10000
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pusht_multi_generate_2T1O_clear import CONTENT, cost_expansion  # noqa: E402
from pusht_multi_generate_2T1W2A_split_clear import (  # noqa: E402
    CONFIG, N_DIMS, render_video,
)
from ighastar.scripts.common_utils import create_planner  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hysteresis", type=int, required=True)
    ap.add_argument("--src", default=os.path.join(CONTENT, "pusht_multi_2T1W2A_split_clear.npz"))
    args = ap.parse_args()
    tag = f"hyst{args.hysteresis}"
    out = os.path.join(CONTENT, f"pusht_multi_2T1W2A_split_clear_{tag}.npz")
    vdir = os.path.join(CONTENT, f"pusht_multi_2T1W2A_split_clear_{tag}_videos")

    src = np.load(args.src, allow_pickle=True)
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    limit = int(cfg["experiment_info_default"]["max_expansions"])
    planner = create_planner(cfg, bidirectional=False)
    goal = torch.zeros(N_DIMS, dtype=torch.float32)

    keep = {k: [] for k in ("idx", "paths", "cost_exp", "first")}
    for i, (s, w, sd) in enumerate(zip(src["starts"], src["obstacles"], src["seeds"])):
        ok = bool(planner.search(torch.from_numpy(s), goal, torch.from_numpy(w),
                                 limit, args.hysteresis, True))
        path = planner.get_best_path()
        first, cost_exp = cost_expansion(planner)
        ok = ok and path.numel() > 0
        print(f"draw {sd} solved={ok} first_sol={first} (hyst 500: "
              f"{int(src['first_sol_expansions'][i])}) "
              f"final_cost={float(cost_exp[-1]):.2f} (hyst 500: "
              f"{float(np.asarray(src['cost_exp'][i])[-1]):.2f}) "
              f"expansions={len(cost_exp)}", flush=True)
        if ok:
            p = path.detach().cpu().numpy()
            for k, v in zip(keep, (i, p, cost_exp, first)):
                keep[k].append(v)
            render_video(p, w, os.path.join(vdir, f"p{i:03d}_draw{sd}.mp4"))
    idx = np.asarray(keep["idx"], dtype=np.int64)
    np.savez(out, src_index=idx, starts=src["starts"][idx], obstacles=src["obstacles"][idx],
             seeds=src["seeds"][idx], paths=np.array(keep["paths"], dtype=object),
             cost_exp=np.array(keep["cost_exp"], dtype=object),
             first_sol_expansions=np.asarray(keep["first"], dtype=np.int64),
             hysteresis=np.array(args.hysteresis), max_expansions=np.array(limit))
    print(f"wrote {len(idx)} problems to {out}")


if __name__ == "__main__":
    main()
