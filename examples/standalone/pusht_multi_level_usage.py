#!/usr/bin/env python3
"""Which IGHA* levels does the baseline search actually use? Re-solves the 20 sweep problems (baseline settings, CPU).

  /home/stark/ighastar-venv/bin/python3 examples/standalone/pusht_multi_level_usage.py

Per problem: the level the search moved to at each switch and the expansion counter at each switch (both saved to
Content/standalone/pusht_multi_level_usage.npz), plus the share of expansions spent at each level and the level in
force at the first solution. Level L covers the expansions from its switch to the next switch; level 0 covers the
expansions before the first switch.
"""
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pusht_multi_generate_2T1O_clear import CONTENT, cost_expansion  # noqa: E402
from pusht_multi_generate_2T1W2A_split_clear import CONFIG, N_DIMS  # noqa: E402
import pusht_multi_split_sweep as sweep  # noqa: E402
from ighastar.scripts.common_utils import create_planner  # noqa: E402

OUT = os.path.join(CONTENT, "pusht_multi_level_usage.npz")


def main():
    cfg = sweep.apply_cell(yaml.safe_load(open(CONFIG)), sweep.BASELINE)
    planner = create_planner(cfg, bidirectional=False)
    d = np.load(sweep.FARM, allow_pickle=True)
    goal = torch.zeros(N_DIMS, dtype=torch.float32)
    ids = sweep.problem_ids()
    save = {"ids": np.array(ids)}
    n_levels = 3  # levels shown one by one; deeper levels are pooled out of the printed columns
    print(f"budget {sweep.MAX_EXPANSIONS}; hysteresis {sweep.HYST}", flush=True)
    print("problem solved first_sol  maxlvl switches  share of expansions per level (L0 L1 L2 ...)  level@first_sol wall")
    tot = np.zeros(n_levels + 2)
    for pi in ids:
        t0 = time.perf_counter()
        ok = bool(planner.search(torch.from_numpy(d["starts"][pi]), goal, torch.from_numpy(d["obstacles"][pi]),
                                 sweep.MAX_EXPANSIONS, sweep.HYST, True))
        wall = time.perf_counter() - t0
        first, curve = cost_expansion(planner)
        prof = planner.get_profiler_info()
        maxlvl, switches, terminal, exp_list = int(prof[5]), int(prof[4]), int(prof[7]), list(prof[8])
        lv = list(planner.get_level_history())
        assert len(lv) == len(exp_list), (len(lv), len(exp_list))
        # expansions per level: segment k runs at level (0 if k == 0 else lv[k-1])
        bounds = [0] + exp_list + [terminal]
        share = np.zeros(max(n_levels, maxlvl) + 2)
        for k in range(len(bounds) - 1):
            share[0 if k == 0 else lv[k - 1]] += max(0, bounds[k + 1] - bounds[k])
        tot[:len(share)] += share[:len(tot)]
        at_first = -1
        if first >= 0:
            k = int(np.searchsorted(exp_list, first, side="right"))
            at_first = 0 if k == 0 else lv[k - 1]
        pct = share / max(1, share.sum()) * 100
        print(f"p{pi:03d}  {'yes' if ok else 'NO '}  {first:8d}  {maxlvl:4d} {switches:8d}   "
              + " ".join(f"{x:5.1f}%" for x in pct[:n_levels + 1]) + f"   {at_first:3d}   {wall:5.0f}s", flush=True)
        save[f"levels_{pi}"] = np.array(lv, dtype=np.int32)
        save[f"exp_{pi}"] = np.array(exp_list, dtype=np.int32)
        save[f"meta_{pi}"] = np.array([first, maxlvl, switches, terminal])
    np.savez(OUT, **save)
    pct = tot / tot.sum() * 100
    print("\nall problems pooled, share of expansions per level: "
          + ", ".join(f"L{i} {x:.1f}%" for i, x in enumerate(pct[:n_levels + 1])))


if __name__ == "__main__":
    main()
