"""Plan the peg insertion with IGHA* entirely in C++ on CPU MuJoCo (cpp/peg_mujoco.h) -- no Isaac Sim process.

UNREVIEWED: written by Claude, not yet read or reviewed by a human.

    python plan_peg_cpp.py --consts <problem.npz> --model <cpu_model.mjb> [--max-expansions 3000] [--nthread 32]

--consts: an .npz with mujoco_cpp.constants_from_problem's keys plus "start" (the node state to plan from, PegProblem
layout), e.g. written by export_cpp_problem.py from a live Isaac/Newton session. --model: the CPU MuJoCo scene with the
hole, table, robot support and robot base as mocap bodies.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import mujoco_cpp  # noqa: E402
from ighastar.scripts.common_utils import create_planner  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--consts", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--config", default=str(HERE / "config.yml"))
ap.add_argument("--max-expansions", type=int, default=None)
ap.add_argument("--nthread", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-preemptive", action="store_true", help="turn the config's preemptive expansion off")
ap.add_argument("--out", default=None, help="directory for the path / search tree")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
c = dict(np.load(args.consts, allow_pickle=False))
c["num_controls"] = int(cfg.get("problem", {}).get("num_controls", c["num_controls"]))
pc = dict(cfg["planner"])
if args.no_preemptive:
    pc["preemptive_expansion"] = {**pc.get("preemptive_expansion", {}), "enabled": False}
if args.max_expansions is not None:
    pc["max_expansions"] = args.max_expansions
configs = mujoco_cpp.planner_config(c, pc, cfg.get("task"), args.model, args.nthread, args.seed)
info = configs["experiment_info_default"]
planner = create_planner(configs, bidirectional=False)
start = torch.tensor(c["start"], dtype=torch.float32)
n = int(c["n"])
t0 = time.perf_counter()
ok = planner.search(start, torch.zeros(n), torch.zeros(1), int(info["max_expansions"]), int(info["hysteresis"]), True)
dt = time.perf_counter() - t0
prof = planner.get_profiler_info()
exp = int(prof[7])
cost_curve = [float(x) for x in prof[9]]
first = next((i for i, x in enumerate(cost_curve) if x < 1e5), None)
rec = {"success": bool(ok), "search_s": dt, "expansions": exp, "exp_per_s": exp / dt,
       "preemptive": int(planner.get_preemptive_expansions()), "preemptive_hits": int(planner.get_preemptive_hits()),
       "avg_succ_us": float(prof[0]), "first_solution_expansion": None if first is None else first + 1}
print(json.dumps(rec, indent=1))
if args.out:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if ok:
        path = planner.get_best_path()
        states = torch.cat([start[None], torch.flip(path[:, :-1], dims=[0])], 0)
        np.savez_compressed(out / "plan.npz", states=states.numpy(), record=json.dumps(rec))
    (out / "record.json").write_text(json.dumps(rec, indent=1))
