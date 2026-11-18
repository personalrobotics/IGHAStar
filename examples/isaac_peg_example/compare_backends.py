"""Compare edge outcomes of two newton_probe.py --save runs (same start node, same controls). UNREVIEWED.

    python compare_backends.py ref.npz other.npz
"""
import sys

import numpy as np

A, B = (np.load(p) for p in sys.argv[1:3])
F = int(A["F"])
assert np.allclose(A["U"], B["U"]) and np.allclose(A["S"], B["S"], atol=1e-5), "different controls / start nodes"
a, b = A["out"][:, -1, :F], B["out"][:, -1, :F]
ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
S0 = A["S"][0, :F]
print(f"edges valid: ref {int(np.isfinite(a).all(1).sum())}/{len(a)}, other {int(np.isfinite(b).all(1).sum())}/{len(b)}, both {int(ok.sum())}")
d_hand = np.linalg.norm(a[ok, :3] - b[ok, :3], axis=1) * 1e3
mv_a, mv_b = np.linalg.norm(a[ok, :3] - S0[:3], axis=1) * 1e3, np.linalg.norm(b[ok, :3] - S0[:3], axis=1) * 1e3
print(f"hand displacement per edge [mm]: ref median {np.median(mv_a):.1f} p90 {np.percentile(mv_a, 90):.1f} | other median {np.median(mv_b):.1f} p90 {np.percentile(mv_b, 90):.1f}")
print(f"hand end-point difference ref vs other [mm]: median {np.median(d_hand):.1f} p90 {np.percentile(d_hand, 90):.1f} max {d_hand.max():.1f}  (relative to ref displacement: median {np.median(d_hand / np.maximum(mv_a, 1e-3)):.2f})")
cor = np.corrcoef(mv_a, mv_b)[0, 1]
print(f"correlation of hand displacement magnitudes: {cor:.3f}")
r6 = np.linalg.norm(a[ok, 3:9] - b[ok, 3:9], axis=1)
print(f"hand 6D-rotation difference: median {np.median(r6):.3f} p90 {np.percentile(r6, 90):.3f}  (ref's own change from start: median {np.median(np.linalg.norm(a[ok, 3:9] - S0[3:9], axis=1)):.3f})")
dp = np.linalg.norm(a[ok, 9:12] - b[ok, 9:12], axis=1) * 1e3
pm = np.linalg.norm(a[ok, 9:12] - S0[9:12], axis=1) * 1e3
print(f"peg end-point difference [mm]: median {np.median(dp):.1f} p90 {np.percentile(dp, 90):.1f} max {dp.max():.1f}; ref peg displacement median {np.median(pm):.1f} p90 {np.percentile(pm, 90):.1f}")
tol = np.array([0.02] * 3 + [0.03] * 6 + [0.02] * 3 + [0.03] * 3 + [0.02])[:F]
print(f"rows whose hashed features all agree within the planner's tolerance: {np.mean((np.abs(a[ok] - b[ok]) <= tol).all(1)):.1%}")
