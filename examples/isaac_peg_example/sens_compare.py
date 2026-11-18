"""Compare sens_iters.py outputs. UNREVIEWED.

    python sens_compare.py --a r36_a.npz --a2 r36_b.npz --b r8.npz [--config config.yml]

Reports, per start state and for edge end states: (a) a vs a2 = run-to-run noise at the same setting, (b) a vs b = the effect
of the setting. Features are the planner's hashed 16; "exceeds tolerance" means the feature differs by more than the
config's per-feature tolerance (the planner would treat the two end states as different)."""
import argparse
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--a2", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--config", default=str(HERE / "config.yml"))
args = ap.parse_args()
A, A2, B = (np.load(p, allow_pickle=True) for p in (args.a, args.a2, args.b))
tol = np.asarray(yaml.safe_load(open(args.config))["task"]["tolerance"], np.float64)
res = np.asarray(yaml.safe_load(open(args.config))["task"]["resolution"], np.float64)
F = int(A["F"])
names = [str(x) for x in A["feature_names"]]
print(f"a = robot iters {int(A['robot_iters'])} (default 36 if -1), a2 = repeat of a, b = robot iters {int(B['robot_iters'])}")
for key in [k for k in A.files if "|" not in k and k not in ("F", "feature_names", "robot_iters")]:
    a, a2, b = A[key][:, -1, :F], A2[key][:, -1, :F], B[key][:, -1, :F]
    va, va2, vb = (~np.isnan(x).any(axis=1) for x in (a, a2, b))
    print(f"\n=== {key}: {len(a)} edges; invalid rows: a {int((~va).sum())}, a2 {int((~va2).sum())}, b {int((~vb).sum())}; "
          f"validity differs a/a2 {int((va != va2).sum())}, a/b {int((va != vb).sum())}")
    for label, x, y, ok in (("noise  (a vs a2)", a, a2, va & va2), ("effect (a vs b) ", a, b, va & vb)):
        d = np.abs(x[ok] - y[ok])
        exceed = (d > tol).any(axis=1).mean()
        print(f"  {label}: rows compared {int(ok.sum())} | median |diff| over rows: hand pos {np.median(np.linalg.norm(x[ok][:, :3] - y[ok][:, :3], axis=1)) * 1e3:.2f} mm, "
              f"peg pos {np.median(np.linalg.norm(x[ok][:, 9:12] - y[ok][:, 9:12], axis=1)) * 1e3:.2f} mm "
              f"| p95: hand {np.percentile(np.linalg.norm(x[ok][:, :3] - y[ok][:, :3], axis=1), 95) * 1e3:.2f} mm, "
              f"peg {np.percentile(np.linalg.norm(x[ok][:, 9:12] - y[ok][:, 9:12], axis=1), 95) * 1e3:.2f} mm "
              f"| rows with any hashed feature over tolerance: {exceed:.1%}")
    d = np.abs(a[va & vb] - b[va & vb])
    worst = np.argsort(-(d / tol).max(axis=0))[:3]
    print("  features most affected (a vs b, max |diff|/tolerance): " + ", ".join(f"{names[i]} {float((d[:, i] / tol[i]).max()):.1f}" for i in worst))
    gd = np.abs(a[va & vb][:, F - 1] - b[va & vb][:, F - 1]).max() if (va & vb).any() else 0
    print(f"  gripper feature max |diff| {gd:.2f}")
