#!/usr/bin/env python3
"""Collect / plot Omega-vs-expansion curves (analysis only; not required to run the planner).

Suite YAML lists named curves (planner config + pkl dir). Example:

  max_expansions: 50000
  problems: Content/standalone/pusht_dino_problems.npz
  out_dir: Content/standalone/curve_plots
  curves:
    - name: qs_ndir24
      label: QS ndir=24
      config: examples/standalone/Configs/pusht.yml
      pkl_dir: Content/standalone/curves/qs_ndir24

  python utils/cost_curves.py collect --suite utils/suites/example_pusht.yml
  python utils/cost_curves.py plot    --suite utils/suites/example_pusht.yml

Requires: pip install 'ighastar[analysis]'  (matplotlib, scipy)
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
COST_INF = 1e4  # Omega sentinel: no solution yet


def _repo_path(p: Union[str, Path]) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO / path)


def load_suite(path: Union[str, Path]) -> Dict[str, Any]:
    with open(_repo_path(path)) as f:
        suite = yaml.safe_load(f)
    assert suite and "curves" in suite and suite["curves"], "suite needs non-empty curves:"
    return suite


def pkl_name(problem_idx: int) -> str:
    return f"p{problem_idx:03d}.pkl"


def expected_pkls(pkl_dir: Path, n: int) -> List[Path]:
    return [pkl_dir / pkl_name(i) for i in range(n)]


def assert_pkls_present(pkl_dir: Path, n: int) -> None:
    missing = [p.name for p in expected_pkls(pkl_dir, n) if not p.is_file()]
    assert not missing, (
        f"{pkl_dir}: missing {len(missing)}/{n} pkls "
        f"(e.g. {missing[:5]}{'…' if len(missing) > 5 else ''}). "
        "Run: python utils/cost_curves.py collect --suite …"
    )


def load_best_cost(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f)
    assert "best_cost" in data, f"{path}: expected key 'best_cost'"
    return np.asarray(data["best_cost"], dtype=np.float64)


def pad_best_cost(raw: np.ndarray, max_exp: int) -> np.ndarray:
    """Match plot_cost_playback: invalid → nan, hold last value to max_exp."""
    out = np.full(max_exp, np.nan, dtype=np.float64)
    n = min(len(raw), max_exp)
    if n == 0:
        return out
    chunk = np.asarray(raw[:n], dtype=np.float64).copy()
    chunk[chunk >= COST_INF] = np.nan
    out[:n] = chunk
    last = chunk[n - 1]
    if np.isfinite(last):
        out[n:] = last
    return out


def first_solution_exp(best_cost: np.ndarray) -> float:
    idx = np.where(np.isfinite(best_cost) & (best_cost < COST_INF))[0]
    return float(idx[0]) if len(idx) else float("inf")


def load_curve_costs(
    pkl_dir: Path, n_problems: int, max_exp: int
) -> List[np.ndarray]:
    assert_pkls_present(pkl_dir, n_problems)
    return [pad_best_cost(load_best_cost(pkl_dir / pkl_name(i)), max_exp) for i in range(n_problems)]


# ---------------------------------------------------------------------------
# Stats (same policy as IGHAStar_private plot_cost_playback)
# ---------------------------------------------------------------------------


def bernoulli_ci(successes: int, total: int, confidence: float = 0.95):
    import scipy.stats as st

    if total == 0:
        return np.nan, np.nan, np.nan
    p = successes / total
    z = float(st.norm.ppf(1 - (1 - confidence) / 2))
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def success_vs_expansions(
    curves: Dict[str, List[np.ndarray]], max_exp: int
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Success = finite cost at expansion e. Returns name → (rate, lo, hi)."""
    xs = np.arange(max_exp)
    out = {}
    for name, lists in curves.items():
        n = len(lists)
        rate = np.zeros(max_exp)
        lo = np.zeros(max_exp)
        hi = np.zeros(max_exp)
        for e in range(max_exp):
            k = sum(1 for c in lists if np.isfinite(c[e]))
            p, a, b = bernoulli_ci(k, n)
            rate[e], lo[e], hi[e] = p, a, b
        out[name] = (rate, lo, hi)
    return out


def cost_vs_expansions(
    curves: Dict[str, List[np.ndarray]],
    max_exp: int,
    min_exp_threshold: Optional[int] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Mean±95% t-CI of cost, only after >50% success, over earliest-solving 50%.

    Problems must have first solution < min_exp_threshold in *all* curves
    (intersection), same as plot_cost_playback.
    """
    import scipy.stats as st

    if min_exp_threshold is None:
        min_exp_threshold = max_exp
    names = list(curves.keys())
    n = len(next(iter(curves.values())))
    valid = []
    for i in range(n):
        if all(first_solution_exp(curves[nm][i]) < min_exp_threshold for nm in names):
            valid.append(i)
    print(f"common problems with first sol < {min_exp_threshold}: {len(valid)}/{n}")

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in names:
        if not valid:
            out[name] = (np.full(max_exp, np.nan), np.full(max_exp, np.nan))
            continue
        mats = np.stack([curves[name][i] for i in valid], axis=0)  # [P, E]
        p = mats.shape[0]
        firsts = np.array([first_solution_exp(mats[i]) for i in range(p)])
        n50 = int(np.ceil(p * 0.5))
        earliest = set(np.argsort(firsts)[:n50].tolist())

        mean = np.full(max_exp, np.nan)
        ci = np.full(max_exp, np.nan)
        for e in range(max_exp):
            col = mats[:, e]
            succ_rate = np.mean(np.isfinite(col))
            if succ_rate <= 0.5:
                continue
            vals = [col[i] for i in earliest if np.isfinite(col[i])]
            if len(vals) > 1:
                m = float(np.mean(vals))
                se = float(st.sem(vals))
                if np.isfinite(se) and se > 0:
                    a, _b = st.t.interval(0.95, len(vals) - 1, loc=m, scale=se)
                    mean[e], ci[e] = m, m - a
                else:
                    mean[e], ci[e] = m, 0.0
            elif len(vals) == 1:
                mean[e], ci[e] = float(vals[0]), 0.0
        out[name] = (mean, ci)
        above = np.where(np.mean(np.isfinite(mats), axis=0) > 0.5)[0]
        if len(above):
            print(f"  {name}: >50% success at exp {above[0]}")
    return out


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------


def _load_problems(path: Path) -> Dict[str, Any]:
    data = dict(np.load(path))
    assert "starts" in data and "goals" in data, f"{path}: need starts, goals"
    return data


def _world_tensor(cfg: Dict[str, Any]) -> torch.Tensor:
    if "map" not in cfg:
        return torch.zeros(1, dtype=torch.float32)
    # Map envs: reuse standalone helpers.
    sys.path.insert(0, str(REPO / "examples" / "standalone"))
    from utils import get_map, resolve_map_dir  # type: ignore

    mi = cfg["map"]
    bitmap = get_map(
        mi["name"],
        map_dir=resolve_map_dir(mi["dir"]),
        map_size=mi["size"],
        node_info=cfg["experiment_info_default"]["node_info"],
    )
    return bitmap


def collect_curve(curve: Dict[str, Any], suite: Dict[str, Any], overwrite: bool) -> None:
    from ighastar.scripts.common_utils import create_planner

    cfg_path = _repo_path(curve["config"])
    pkl_dir = _repo_path(curve["pkl_dir"])
    pkl_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    info = cfg["experiment_info_default"]
    max_exp = int(suite.get("max_expansions", info["max_expansions"]))
    info["max_expansions"] = max_exp
    hyst = int(info["hysteresis"])

    problems = _load_problems(_repo_path(suite["problems"]))
    n = len(problems["starts"])
    world = _world_tensor(cfg)
    planner = create_planner(cfg, bidirectional=False)

    print(f"[{curve.get('label', curve['name'])}] → {pkl_dir}  ({n} problems, max_exp={max_exp})")
    for i in range(n):
        out = pkl_dir / pkl_name(i)
        if out.is_file() and not overwrite:
            print(f"  skip {out.name}")
            continue
        start = torch.as_tensor(problems["starts"][i], dtype=torch.float32)
        goal = torch.as_tensor(problems["goals"][i], dtype=torch.float32)
        t0 = time.perf_counter()
        planner.search(start, goal, world, max_exp, hyst, True)
        wall_s = time.perf_counter() - t0
        cost_exp = np.asarray(planner.get_profiler_info()[9], dtype=np.float64)
        payload = {
            "best_cost": cost_exp,
            "wall_s": float(wall_s),
            "problem_idx": i,
            "seed": int(problems["seeds"][i]) if "seeds" in problems else -1,
        }
        with open(out, "wb") as f:
            pickle.dump(payload, f)
        sol = np.where(cost_exp < COST_INF)[0]
        print(
            f"  {out.name}: wall={wall_s:.3f}s  expansions={len(cost_exp)}  "
            f"first_sol={int(sol[0]) if sol.size else -1}"
        )


def cmd_collect(args: argparse.Namespace) -> None:
    # One process per curve: each JIT backend registers IGHAStar globally.
    import multiprocessing

    suite = load_suite(args.suite)
    ctx = multiprocessing.get_context("spawn")
    for curve in suite["curves"]:
        proc = ctx.Process(
            target=collect_curve, args=(curve, suite, args.overwrite)
        )
        proc.start()
        proc.join()
        if proc.exitcode != 0:
            raise SystemExit(proc.exitcode or 1)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def cmd_plot(args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib required. Install: pip install 'ighastar[analysis]'"
        ) from e

    suite = load_suite(args.suite)
    max_exp = int(suite["max_expansions"])
    problems = _load_problems(_repo_path(suite["problems"]))
    n = len(problems["starts"])

    curves: Dict[str, List[np.ndarray]] = {}
    labels: Dict[str, str] = {}
    for c in suite["curves"]:
        name = c["name"]
        labels[name] = c.get("label", name)
        curves[name] = load_curve_costs(_repo_path(c["pkl_dir"]), n, max_exp)

    out_dir = _repo_path(suite.get("out_dir", "Content/standalone/curve_plots"))
    out_dir.mkdir(parents=True, exist_ok=True)
    xs = np.arange(max_exp)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Success vs expansions
    succ = success_vs_expansions(curves, max_exp)
    fig, ax = plt.subplots(figsize=(4, 3))
    for i, name in enumerate(curves):
        rate, lo, hi = succ[name]
        color = colors[i % len(colors)]
        ax.plot(xs, rate * 100, label=labels[name], color=color, lw=2)
        ax.fill_between(xs, lo * 100, hi * 100, color=color, alpha=0.2)
    ax.set_xlabel("Expansions")
    ax.set_ylabel("Success (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "success_vs_expansions.png", dpi=150)
    fig.savefig(out_dir / "success_vs_expansions.pdf")
    print(f"wrote {out_dir / 'success_vs_expansions.png'}")

    # Cost vs expansions
    cost = cost_vs_expansions(curves, max_exp, min_exp_threshold=max_exp)
    fig, ax = plt.subplots(figsize=(4, 3))
    for i, name in enumerate(curves):
        mean, ci = cost[name]
        mask = np.isfinite(mean) & np.isfinite(ci)
        if not np.any(mask):
            print(f"  {name}: no cost curve (never >50% success on common set)")
            continue
        color = colors[i % len(colors)]
        ax.plot(xs[mask], mean[mask], label=labels[name], color=color, lw=2)
        ax.fill_between(
            xs[mask], mean[mask] - ci[mask], mean[mask] + ci[mask], color=color, alpha=0.2
        )
    ax.set_xlabel("Expansions")
    ax.set_ylabel("Cost")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "cost_vs_expansions.png", dpi=150)
    fig.savefig(out_dir / "cost_vs_expansions.pdf")
    print(f"wrote {out_dir / 'cost_vs_expansions.png'}")
    if args.show:
        plt.show()


def _selfcheck() -> None:
    """ponytail: smallest check that pad + success stats break if logic regresses."""
    max_exp = 10
    a = pad_best_cost(np.array([1e9, 1e9, 5.0, 4.0]), max_exp)
    assert np.isnan(a[0]) and a[2] == 5.0 and a[9] == 4.0
    b = pad_best_cost(np.array([1e9] * 3), max_exp)
    assert np.all(np.isnan(b))
    curves = {"A": [a, pad_best_cost(np.array([1e9, 3.0]), max_exp)]}
    succ = success_vs_expansions(curves, max_exp)
    assert succ["A"][0][0] == 0.0
    assert succ["A"][0][2] == 1.0
    print("cost_curves selfcheck ok")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="run planner, write p{idx:03d}.pkl with best_cost")
    c.add_argument("--suite", required=True)
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=cmd_collect)

    pl = sub.add_parser("plot", help="assert pkls exist, write success/cost vs expansions")
    pl.add_argument("--suite", required=True)
    pl.add_argument("--show", action="store_true")
    pl.set_defaults(func=cmd_plot)

    sc = sub.add_parser("selfcheck", help="assert pad/success helpers")
    sc.set_defaults(func=lambda _a: _selfcheck())

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
