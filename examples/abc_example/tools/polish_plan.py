"""UNREVIEWED: written by Claude, not yet read or reviewed by a human. Treat as experimental.

Polish coarse-step plans with segment MPPI and check them under abc's 2 ms physics.

For each plan episode (plan_abc.py output, planned with a coarse --plan-substeps):
  none : replay the planner's actions as they are in abc's 2 ms CPU env,
  h<H> : refine every control period with MPPI against the coarse plan's knots (horizon H
         periods), then replay the refined actions open-loop in the same env.
Reports abc's evaluator success and the mover/objects' final position error versus the
state the coarse planner predicted.

    python tools/polish_plan.py DIR_OR_NPZ... [--horizons 1 3] [--K 256] [--iters 4]
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

EX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EX))
import mujoco  # noqa: E402
from abc_refine import PlanData, build_env, evaluate, polish, replay_cpu  # noqa: E402


def free_adrs(m):
    return [int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]


def metrics(env, plan, q_replay):
    g = env.eval_goal if hasattr(env, "eval_goal") else env._task_evaluator._goal
    m = g["mover_adr"]
    pos = lambda a, adr: a[:, adr:adr + 3]  # noqa: E731
    fin, anyv = evaluate(env, q_replay)
    mover_end = float(np.linalg.norm(pos(q_replay, m)[-1] - pos(plan.qpos, m)[-1]))
    obj_end = max(float(np.linalg.norm(pos(q_replay, a)[-1] - pos(plan.qpos, a)[-1])) for a in free_adrs(env.model))
    mover_path = float(np.linalg.norm(pos(q_replay, m) - pos(plan.qpos, m), axis=1).max())
    return dict(final=fin, any=anyv, mover_end_cm=100 * mover_end, obj_end_cm=100 * obj_end, mover_max_cm=100 * mover_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--sigma-arm", type=float, default=0.15)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    files = []
    for p in map(Path, args.paths):
        files += sorted(p.glob("put_relative_seed*.npz")) if p.is_dir() else [p]
    files = [f for f in files if not f.name.endswith(("_tree.npz",))]
    rows, env = [], None
    for f in files:
        plan = PlanData(f)
        env = build_env(plan, env)
        r = {"seed": plan.seed, "steps": len(plan.actions)}
        r["none"] = metrics(env, plan, replay_cpu(env, plan, plan.actions))
        for H in args.horizons:
            t0 = time.perf_counter()
            U, qf, info = polish(plan, env, horizon=H, K=args.K, iters=args.iters, sigma_arm=args.sigma_arm,
                                 nthread=args.threads)
            dt = time.perf_counter() - t0
            r[f"h{H}"] = metrics(env, plan, replay_cpu(env, plan, U))
            r[f"h{H}"]["secs"] = dt
            r[f"h{H}"]["du_max"] = float(np.abs(U - plan.actions).max())
        rows.append(r)
        print(f"seed {r['seed']:2d} ({r['steps']} periods): " + "  |  ".join(
            f"{k}: {'OK ' if v['final'] else 'FAIL'} mover end {v['mover_end_cm']:.1f} cm, objs {v['obj_end_cm']:.1f} cm"
            + (f" ({v['secs']:.0f}s, max du {v['du_max']:.2f})" if 'secs' in v else "") for k, v in r.items() if isinstance(v, dict)), flush=True)
    keys = [k for k in rows[0] if isinstance(rows[0][k], dict)]
    print(f"\n{len(rows)} plans")
    print(f"{'variant':8s} {'success final':>14s} {'success any':>12s} {'mover end err cm med/p90/max':>30s} {'<1cm':>6s}")
    for k in keys:
        e = np.array([r[k]["mover_end_cm"] for r in rows])
        print(f"{k:8s} {sum(r[k]['final'] for r in rows):>9d}/{len(rows):<4d} {sum(r[k]['any'] for r in rows):>7d}/{len(rows):<4d} "
              f"{np.median(e):>9.1f} /{np.quantile(e, .9):>6.1f} /{e.max():>6.1f} {np.mean(e < 1) * 100:>5.0f}%")


if __name__ == "__main__":
    main()
