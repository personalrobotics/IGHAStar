"""Success-vs-expansions and cost-vs-expansions plots (same statistics as IGHAStar/utils/cost_curves.py).
python plot_curves.py OUT_DIR A.pkl[:label] B.pkl[:label] ...   (pickles from run_planner.py, same problems and budget)"""
import os, pickle, sys, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as st

warnings.filterwarnings("ignore")   # all-NaN columns before the first solution
INF = 1e4        # planner sentinel for "no solution yet"
out, specs = sys.argv[1], sys.argv[2:]
os.makedirs(out, exist_ok=True)
curves, E = {}, None
for s in specs:
    f, _, label = s.partition(":")
    runs = pickle.load(open(f, "rb"))
    E = E or max(len(r["best_cost"]) for r in runs)
    c = np.full((len(runs), E), np.nan)
    for i, r in enumerate(runs):
        b = np.where(r["best_cost"][:E] >= INF, np.nan, r["best_cost"][:E])
        c[i, :len(b)] = b
        if len(b) and np.isfinite(b[-1]):
            c[i, len(b):] = b[-1]                  # hold the last cost after the search ends
    curves[label or os.path.basename(f)[:-4]] = c
xs = np.arange(E)
first = lambda c: np.where(np.isfinite(c).any(1), np.isfinite(c).argmax(1), E + 1)     # first expansion with a solution
common = np.all([first(c) < E for c in curves.values()], axis=0)                       # solved within budget by every config

fig, ax = plt.subplots(figsize=(5.2, 3.6))
for name, c in curves.items():
    k, n = np.isfinite(c).sum(0), len(c)
    p = k / n; z = 1.96; d = 1 + z**2 / n
    mid = (p + z**2 / (2 * n)) / d; m = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / d       # Wilson interval
    ax.plot(xs, 100 * p, lw=2, label=name); ax.fill_between(xs, 100 * (mid - m), 100 * (mid + m), alpha=0.2)
ax.set(xlabel="Expansions", ylabel="Success (%)", ylim=(0, 100)); ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(f"{out}/success_vs_expansions.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 3.6))
for name, c in curves.items():
    c = c[common]
    ok = np.isfinite(c).mean(0) > 0.5                                                  # only where more than half of the common problems are solved
    c = c[np.argsort(first(c))[: int(np.ceil(0.5 * len(c)))]]                          # average the earliest-solving half of them
    mean, se, n = np.nanmean(c, 0), st.sem(c, 0, nan_policy="omit"), np.isfinite(c).sum(0)
    ci = st.t.ppf(0.975, np.maximum(n - 1, 1)) * se
    ok &= np.isfinite(mean) & np.isfinite(ci)
    ax.plot(xs[ok], mean[ok], lw=2, label=name); ax.fill_between(xs[ok], (mean - ci)[ok], (mean + ci)[ok], alpha=0.2)
ax.set(xlabel="Expansions", ylabel="Cost"); ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(f"{out}/cost_vs_expansions.png", dpi=150); plt.close(fig)
print(f"{common.sum()}/{len(common)} searches solved within budget by every config; wrote {out}/")
