#!/usr/bin/env python3
"""Full-factorial hyperparameter sweep on the farmed split-agent 2T1W2A clear problems (CPU backend).

  python3 examples/standalone/pusht_multi_split_sweep.py --ndir 12 24 --nsub 10 20 --res 40 80   # sweep + summary
  python3 examples/standalone/pusht_multi_split_sweep.py --div 1.0718 --res 40 80 --tol-theta 0.01 0.05
  python3 examples/standalone/pusht_multi_split_sweep.py --all-problems --budget 300000 --tag farm100 --res 80
  python3 examples/standalone/pusht_multi_split_sweep.py <same flags> --summary                   # re-print only

Every flag takes a list; the cells are the product of the lists. An axis that is not given stays at the baseline
(24 directions, 10 substeps, res 40, division factor 2, tee-theta tolerance 0.01, radii 200, vmax 500, hysteresis
500). The baseline cell is always run too, under the current planner, as the reference the summary compares with.

Problems: 20 of the 100 farmed problems, evenly spaced by rank of the saved CPU solve's expansions to first
solution (easiest to hardest), or all of them with --all-problems. Searches run in 3 worker processes with 8
OpenMP threads each. Jobs are (cell, problem) pairs dealt round-robin; a worker that crashes is respawned and
resumes from its CSV. A search that hits the level error is recorded as LEVEL_ERROR.
"""
from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import gc
import itertools
import os
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pusht_multi_generate_2T1O_clear import CONTENT, cost_expansion  # noqa: E402
from pusht_multi_generate_2T1W2A_split_clear import CONFIG, N_DIMS  # noqa: E402
from ighastar.scripts.common_utils import create_planner  # noqa: E402

FARM = os.path.join(CONTENT, "pusht_multi_2T1W2A_split_clear_farm.npz")
OUT = os.path.join(CONTENT, "pusht_multi_split_sweep_results.csv")
POS_DIMS = [0, 1, 2, 3, 4, 5, 7, 8]  # agent xy and tee xy in the 20D state
THETA_DIMS = (6, 9)  # tee orientations in the 20D state
HYST = 500
N_PROBLEMS = 20
MAX_EXPANSIONS = 150000  # read by apply_cell; the farm script overrides it per run
BASELINE = dict(res_xy=40.0, radii=(200.0,), vmax=500.0, n_directions=24, n_substeps=10, div=2.0, tol_theta=0.01)


def tag(c):
    s = f"ndir={c['n_directions']} nsub={c['n_substeps']} res={c['res_xy']:g}"
    if c.get("div", 2.0) != 2.0 or c.get("tol_theta", 0.01) != 0.01:
        s += f" div={c.get('div', 2.0):.4f} tolth={c.get('tol_theta', 0.01):g}"
    return s


def apply_cell(base, c):
    cfg = copy.deepcopy(base)
    info = cfg["experiment_info_default"]
    res = list(info["resolution"])
    for i in POS_DIMS:
        res[i] = float(c["res_xy"])
    info["resolution"] = res
    info["max_expansions"] = MAX_EXPANSIONS
    ni = info["node_info"]
    ni["action_radii"] = [float(r) for r in c["radii"]]
    ni["max_agent_vel"] = float(c["vmax"])
    ni["n_directions"] = int(c["n_directions"])
    ni["n_substeps"] = int(c["n_substeps"])
    if "div" in c:
        info["division_factor"] = float(c["div"])
    if "tol_theta" in c:
        tol = list(info["tolerance"])
        for i in THETA_DIMS:
            tol[i] = float(c["tol_theta"])
        info["tolerance"] = tol
    return cfg


def make_cells(a):
    """The baseline first, then the product of the axes, without repeating the baseline."""
    cells = [dict(BASELINE)]
    for n, s, r, d, t in itertools.product(a.ndir, a.nsub, a.res, a.div, a.tol_theta):
        c = dict(BASELINE, n_directions=n, n_substeps=s, res_xy=r, div=d, tol_theta=t)
        if c not in cells:
            cells.append(c)
    return cells


def problem_ids(all_problems=False):
    d = np.load(FARM, allow_pickle=True)
    if all_problems:
        return list(range(len(d["starts"])))
    order = np.argsort(d["first_sol_expansions"], kind="stable")
    return [int(order[i]) for i in np.linspace(0, len(order) - 1, N_PROBLEMS).round().astype(int)]


def release():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # hand the freed tree back to the OS
    except OSError:
        pass


def worker(shard, num_shards, base, a):
    global MAX_EXPANSIONS
    MAX_EXPANSIONS = a.budget
    d = np.load(FARM, allow_pickle=True)
    cs = make_cells(a)
    jobs = [(ci, pi) for pi in problem_ids(a.all_problems) for ci in range(len(cs))]
    goal = torch.zeros(N_DIMS, dtype=torch.float32)
    path = out_path(a).replace(".csv", f"_shard{shard}.csv")
    done = set()
    if os.path.isfile(path):
        done = {(r["cell"], int(r["problem"])) for r in csv.DictReader(open(path))}
    new = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        out = csv.writer(f)
        if new:
            out.writerow(["cell", "problem", "status", "solved", "first_sol_exp", "cost", "expansions", "max_level",
                          "switches", "wall_s"])
        for n, (ci, pi) in enumerate(jobs):
            c = cs[ci]
            if n % num_shards != shard or (tag(c), pi) in done:
                continue
            # One planner at a time: a finished search keeps its whole tree alive until the planner is freed.
            p = create_planner(apply_cell(base, c), bidirectional=False)
            t0 = time.perf_counter()
            status = "ok"
            try:
                ok = bool(p.search(torch.from_numpy(d["starts"][pi]), goal, torch.from_numpy(d["obstacles"][pi]),
                                   a.budget, HYST, True))
            except RuntimeError as e:
                if not str(e).startswith("IGHA* level"):
                    raise
                ok, status = False, "LEVEL_ERROR"
            wall = time.perf_counter() - t0
            first, curve = cost_expansion(p)
            solved = first >= 0 and (status != "ok" or (ok and p.get_best_path().numel() > 0))
            if status == "ok" and not solved:
                status = "nosol"
            prof = p.get_profiler_info()
            cost = float(curve[-1]) if first >= 0 else float("inf")
            out.writerow([tag(c), pi, status, int(solved), first, f"{cost:.2f}", int(prof[7]), int(prof[5]),
                          int(prof[4]), f"{wall:.1f}"])
            f.flush()
            line = (f"[shard {shard}] {tag(c)} p{pi:03d} {status} first={first} cost={cost:.2f} maxlvl={prof[5]} "
                    f"wall={wall:.0f}s")
            del p
            release()
            rss_gb = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
            print(f"{line} rss_after={rss_gb:.1f}GB", flush=True)


def out_path(a):
    return OUT.replace(".csv", f"_{a.tag}.csv") if a.tag else OUT


def summary(a):
    rows = []
    for k in range(10):
        p = out_path(a).replace(".csv", f"_shard{k}.csv")
        if os.path.isfile(p):
            rows += list(csv.DictReader(open(p)))
    if not rows:
        return print("no results yet")
    with open(out_path(a), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(), w.writerows(rows)
    by = {}
    for r in rows:
        by.setdefault(r["cell"], {})[int(r["problem"])] = r
    btag = tag(BASELINE)
    base = by.get(btag, {})
    ids = problem_ids(a.all_problems)
    farm = np.load(FARM, allow_pickle=True)
    print(f"{len(ids)} problems, budget {a.budget}, hysteresis {HYST}; baseline = {btag} (re-run)\n")
    print(f"{'cell':44s} done solved lvlerr med_first  mean_cost* vs_base(b/s/w) mean_maxlvl mean_wall  exp/s")
    for cell in [btag] + [t for t in by if t != btag]:
        rs = by[cell]
        s = [r for r in rs.values() if r["solved"] == "1"]
        firsts = sorted(int(r["first_sol_exp"]) for r in s)
        both = [pi for pi, r in rs.items() if r["solved"] == "1" and base.get(pi, {}).get("solved") == "1"]
        mc = np.mean([float(rs[pi]["cost"]) for pi in both]) if both else float("nan")
        mb = np.mean([float(base[pi]["cost"]) for pi in both]) if both else float("nan")
        bet = sum(float(rs[pi]["cost"]) < float(base[pi]["cost"]) - 1e-3 for pi in both)
        sam = sum(abs(float(rs[pi]["cost"]) - float(base[pi]["cost"])) < 1e-3 for pi in both)
        lerr = sum(r["status"] == "LEVEL_ERROR" for r in rs.values())
        wall = sum(float(r["wall_s"]) for r in rs.values())
        exps = sum(int(r["expansions"]) for r in rs.values())
        med = firsts[len(firsts) // 2] if firsts else -1
        ml = np.mean([int(r["max_level"]) for r in rs.values()])
        print(f"{cell:44s} {len(rs):4d} {len(s):2d}/{len(rs):<2d} {lerr:5d} {med:9d} {mc:6.2f} (b {mb:5.2f}) "
              f"{bet:2d}/{sam:2d}/{len(both) - bet - sam:2d} {ml:11.1f} {wall / len(rs):9.1f} {exps / wall:6.0f}")
    print("* mean cost over problems both that cell and the baseline re-run solved")


def supervise(make_cmd, workers, retries=8):
    """Run workers in parallel. A worker that dies (segfault) is respawned, and it
    resumes from its own output file. Gives up after `retries` respawns per shard."""
    env = {**os.environ, "OMP_NUM_THREADS": "8"}
    attempts = [0] * workers
    procs = {k: subprocess.Popen(make_cmd(k, 0), env=env) for k in range(workers)}
    while procs:
        time.sleep(5)
        for k, p in list(procs.items()):
            code = p.poll()
            if code is None:
                continue
            del procs[k]
            if code == 0:
                continue
            attempts[k] += 1
            print(f"[supervisor] shard {k} died with code {code}; respawn {attempts[k]}/{retries}", flush=True)
            if attempts[k] > retries:
                raise SystemExit(f"shard {k} keeps crashing")
            procs[k] = subprocess.Popen(make_cmd(k, attempts[k]), env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--ndir", type=int, nargs="+", default=[BASELINE["n_directions"]])
    ap.add_argument("--nsub", type=int, nargs="+", default=[BASELINE["n_substeps"]])
    ap.add_argument("--res", type=float, nargs="+", default=[BASELINE["res_xy"]])
    ap.add_argument("--div", type=float, nargs="+", default=[BASELINE["div"]], help="division factor")
    ap.add_argument("--tol-theta", type=float, nargs="+", default=[BASELINE["tol_theta"]])
    ap.add_argument("--budget", type=int, default=MAX_EXPANSIONS)
    ap.add_argument("--all-problems", action="store_true", help="all farmed problems, not the 20 sweep problems")
    ap.add_argument("--tag", default="", help="suffix for the results files, so runs do not overwrite each other")
    a = ap.parse_args()
    if a.summary:
        return summary(a)
    with open(CONFIG) as f:
        base = yaml.safe_load(f)
    if a.shard >= 0:
        return worker(a.shard, a.workers, base, a)
    print(f"{len(problem_ids(a.all_problems))} problems; {len(make_cells(a))} cells; budget {a.budget}", flush=True)
    flags = sum(([f"--{k}", *map(str, v)] for k, v in (("ndir", a.ndir), ("nsub", a.nsub), ("res", a.res),
                                                       ("div", a.div), ("tol-theta", a.tol_theta))), [])
    flags += ["--budget", str(a.budget), "--tag", a.tag] + (["--all-problems"] if a.all_problems else [])
    supervise(lambda k, att: [sys.executable, __file__, "--shard", str(k), "--workers", str(a.workers), *flags],
              a.workers)
    summary(a)


if __name__ == "__main__":
    main()
