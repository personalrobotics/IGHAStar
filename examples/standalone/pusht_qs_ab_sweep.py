#!/usr/bin/env python3
"""A/B dino-50: best CPU QS config on CPU vs CUDA (separate processes).

Best cell from qs_oat_50k: res=40 radii=200 vmax=500 ndir=24 nsub=10 hyst=500
R=60 max_expansions=50000 (49/50, fail problem_idx=23).

  python3 examples/standalone/pusht_qs_ab_sweep.py --backend cpu
  python3 examples/standalone/pusht_qs_ab_sweep.py --backend cuda
  python3 examples/standalone/pusht_qs_ab_sweep.py --compare
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from ighastar.scripts.common_utils import BASE_DIR, create_planner

CONTENT = os.path.join(BASE_DIR.parent, "Content", "standalone")

# Best CPU QS OAT cell @ 50k.
BEST = {
    "res_xy": 40.0,
    "radii": (200.0,),
    "vmax": 500.0,
    "n_directions": 24,
    "n_substeps": 10,
    "hysteresis": 500,
    "qs_support_radius": 60.0,
    "max_expansions": 50000,
}


def apply_best(
    base: Dict[str, Any],
    backend: str,
    *,
    vmax: float | None = None,
    radii: tuple | None = None,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    info = cfg["experiment_info_default"]
    res = list(info["resolution"])
    for i in range(4):
        res[i] = float(BEST["res_xy"])
    info["resolution"] = res
    info["hysteresis"] = int(BEST["hysteresis"])
    info["max_expansions"] = int(BEST["max_expansions"])
    ni = info["node_info"]
    r = BEST["radii"] if radii is None else tuple(float(x) for x in radii)
    ni["action_radii"] = [float(x) for x in r]
    ni["max_agent_vel"] = float(BEST["vmax"] if vmax is None else vmax)
    ni["n_directions"] = int(BEST["n_directions"])
    ni["n_substeps"] = int(BEST["n_substeps"])
    ni["qs_support_radius"] = float(BEST["qs_support_radius"])
    if backend == "cpu":
        ni["node_type"] = "pusht"
        ni["dynamics"] = "quasistatic"
    else:
        ni["node_type"] = "pusht_qs_cuda"
        ni.pop("dynamics", None)
    return cfg


def run_one(planner, start, goal, max_expansions, hysteresis):
    t0 = time.perf_counter()
    success = bool(
        planner.search(
            torch.as_tensor(start),
            torch.as_tensor(goal),
            torch.zeros(1, dtype=torch.float32),
            max_expansions,
            hysteresis,
            True,
        )
    )
    wall = time.perf_counter() - t0
    (
        avg_succ,
        _gct,
        _ovh,
        _gup,
        _sw,
        _ml,
        _qv,
        expansions,
        _elist,
        _cost_exp_list,
    ) = planner.get_profiler_info()
    exp = max(int(expansions), 1)
    return {
        "success": int(success),
        "wall_s": wall,
        "expansions": int(expansions),
        "avg_succ_us": float(avg_succ),
        "exp_per_s": exp / wall if wall > 0 else float("nan"),
    }


def run_backend(
    backend: str,
    out_csv: str,
    n_problems: int,
    *,
    vmax: float | None = None,
    radii: tuple | None = None,
) -> None:
    yaml_name = "pusht.yml" if backend == "cpu" else "pusht_qs_cuda.yml"
    yaml_path = os.path.join(
        BASE_DIR.parent, "examples", "standalone", "Configs", yaml_name
    )
    with open(yaml_path) as f:
        base = yaml.safe_load(f)
    cfg = apply_best(base, backend, vmax=vmax, radii=radii)
    ni = cfg["experiment_info_default"]["node_info"]
    print(
        f"[{backend}] cell ndir={BEST['n_directions']} "
        f"vmax={float(ni['max_agent_vel']):g} "
        f"radii={ni['action_radii']} max_exp={BEST['max_expansions']}",
        flush=True,
    )
    problems = dict(np.load(os.path.join(CONTENT, "pusht_dino_problems.npz")))
    n = min(n_problems, len(problems["seeds"]))

    planner = create_planner(cfg, bidirectional=False)
    fieldnames = [
        "backend",
        "problem_idx",
        "dino_seed",
        "success",
        "wall_s",
        "expansions",
        "avg_succ_us",
        "exp_per_s",
    ]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    rows: List[Dict[str, Any]] = []
    t_all = time.perf_counter()
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for pi in range(n):
            m = run_one(
                planner,
                problems["starts"][pi],
                problems["goals"][pi],
                int(BEST["max_expansions"]),
                int(BEST["hysteresis"]),
            )
            row = {
                "backend": backend,
                "problem_idx": pi,
                "dino_seed": int(problems["seeds"][pi]),
                **m,
            }
            rows.append(row)
            w.writerow(row)
            f.flush()
            ok = sum(r["success"] for r in rows)
            elapsed = time.perf_counter() - t_all
            print(
                f"[{backend}] {pi + 1}/{n}  succ={ok}/{pi + 1}  "
                f"last wall={m['wall_s']:.2f}s  "
                f"avg_succ_us={m['avg_succ_us']:.1f}  "
                f"exp/s={m['exp_per_s']:.0f}  elapsed={elapsed:.0f}s",
                flush=True,
            )

    ok_rows = [r for r in rows if r["success"]]
    mean_eps_succ = (
        float(np.mean([r["exp_per_s"] for r in ok_rows])) if ok_rows else float("nan")
    )
    print(
        f"\n[{backend}] done  success={len(ok_rows)}/{n}  "
        f"mean avg_succ_us={np.mean([r['avg_succ_us'] for r in rows]):.2f}  "
        f"mean exp/s (all)={np.mean([r['exp_per_s'] for r in rows]):.0f}  "
        f"mean exp/s (succ)={mean_eps_succ:.0f}"
    )
    fails = [r["problem_idx"] for r in rows if not r["success"]]
    print(f"[{backend}] failures: {fails}")
    print(f"wrote {out_csv}")


def compare(cpu_csv: str, cuda_csv: str) -> None:
    def load(path: str):
        with open(path) as f:
            return list(csv.DictReader(f))

    cpu = {int(r["problem_idx"]): r for r in load(cpu_csv)}
    cuda = {int(r["problem_idx"]): r for r in load(cuda_csv)}
    assert set(cpu) == set(cuda), "problem set mismatch"
    idxs = sorted(cpu)
    cpu_ok = {i for i in idxs if int(cpu[i]["success"])}
    cuda_ok = {i for i in idxs if int(cuda[i]["success"])}
    only_cpu = sorted(cpu_ok - cuda_ok)
    only_cuda = sorted(cuda_ok - cpu_ok)
    both_fail = sorted(set(idxs) - cpu_ok - cuda_ok)

    def means(rows, key):
        return float(np.mean([float(r[key]) for r in rows]))

    cpu_rows = [cpu[i] for i in idxs]
    cuda_rows = [cuda[i] for i in idxs]
    print("=== success set ===")
    print(f"cpu  {len(cpu_ok)}/{len(idxs)}  fails={sorted(set(idxs) - cpu_ok)}")
    print(f"cuda {len(cuda_ok)}/{len(idxs)}  fails={sorted(set(idxs) - cuda_ok)}")
    print(f"identical success set: {cpu_ok == cuda_ok}")
    if only_cpu or only_cuda:
        print(f"  only cpu success:  {only_cpu}")
        print(f"  only cuda success: {only_cuda}")
    if both_fail:
        print(f"  both fail: {both_fail}")

    print("\n=== timing (all problems) ===")
    print(
        f"{'backend':<6}  {'mean_avg_succ_us':>16}  {'mean_exp/s':>12}  "
        f"{'mean_wall_s':>12}"
    )
    for name, rows in (("cpu", cpu_rows), ("cuda", cuda_rows)):
        print(
            f"{name:<6}  {means(rows, 'avg_succ_us'):16.2f}  "
            f"{means(rows, 'exp_per_s'):12.0f}  "
            f"{means(rows, 'wall_s'):12.3f}"
        )
    # exp/s = expansions / solve_time (wall_s), already per-row.
    speedup = means(cpu_rows, "exp_per_s") / means(cuda_rows, "exp_per_s")
    print(f"\nexp/s speedup cuda/cpu ≈ {1.0 / speedup:.2f}x")
    print(
        f"avg_succ_us speedup cpu/cuda ≈ "
        f"{means(cpu_rows, 'avg_succ_us') / means(cuda_rows, 'avg_succ_us'):.2f}x"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("cpu", "cuda", "both", "compare"))
    p.add_argument("--n-problems", type=int, default=50)
    p.add_argument("--vmax", type=float, default=None, help="override BEST vmax")
    p.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=None,
        help="override BEST action_radii (e.g. --radii 50)",
    )
    p.add_argument(
        "--cpu-out",
        default=os.path.join(CONTENT, "pusht_qs_ab_cpu_ndir24.csv"),
    )
    p.add_argument(
        "--cuda-out",
        default=os.path.join(CONTENT, "pusht_qs_ab_cuda_ndir24.csv"),
    )
    args = p.parse_args()
    radii = tuple(args.radii) if args.radii is not None else None

    if args.backend == "compare":
        compare(args.cpu_out, args.cuda_out)
        return

    if args.backend == "both":
        # Separate processes — pybind IGHAStar name clash if both load in-process.
        script = os.path.abspath(__file__)
        for b, out in (("cpu", args.cpu_out), ("cuda", args.cuda_out)):
            cmd = [
                sys.executable,
                script,
                "--backend",
                b,
                "--n-problems",
                str(args.n_problems),
                "--cpu-out",
                args.cpu_out,
                "--cuda-out",
                args.cuda_out,
            ]
            if args.vmax is not None:
                cmd += ["--vmax", str(args.vmax)]
            if radii is not None:
                cmd += ["--radii", *[str(r) for r in radii]]
            subprocess.check_call(cmd)
        compare(args.cpu_out, args.cuda_out)
        return

    out = args.cpu_out if args.backend == "cpu" else args.cuda_out
    run_backend(
        args.backend, out, args.n_problems, vmax=args.vmax, radii=radii
    )


if __name__ == "__main__":
    main()
