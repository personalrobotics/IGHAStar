"""How large can the physics step be before open-loop replay of abc's sim demos breaks?

The policy's action is sampled once per control period (34 ms = 17 x 2 ms); the PD
controller and physics run underneath. This replays each demo's recorded actions in
abc's CPU MuJoCo env with the same 34 ms control period but a coarser physics step
(``timestep = 34 ms / n``, ``control_decimation = n``) and compares, per episode:

* success of abc's evaluator on the replay (final frame, and at any frame) against the
  success of the recorded demo itself and of the 2 ms replay,
* object positions against the demo's recorded scene_qpos (xyz error per free joint),
  and against the 2 ms replay of the same actions.

    python tools/dt_sweep_demos.py --episodes 60 [--root ~/abc_src/abc/cache/train_sim]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

CONTROL_PERIOD = 0.034
DECIMATIONS = (17, 10, 8, 5, 4, 3, 2, 1)  # 2, 3.4, 4.25, 6.8, 8.5, 11.3, 17, 34 ms


def free_adrs(m):
    return [int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]


def build(ep: Path):
    import abc_sim
    from abc_minimal.episode_io import load_episode, load_scene_qpos, load_scene_xml, seed_initial_state

    metadata, states, actions = load_episode(ep)
    scene_qpos = load_scene_qpos(ep, metadata, len(states))
    task = metadata.get("task") or metadata.get("task_name")
    xml, _ = load_scene_xml(ep, metadata, Path(os.environ["ABC_ASSETS_DIR"]) if os.environ.get("ABC_ASSETS_DIR") else None)
    env = abc_sim.make_env(task=task, render_cameras=False, scene_xml_string=xml, enable_task_randomizer=xml is None)
    env.reset(seed=0, randomize=xml is None)
    seed_initial_state(env, ep, metadata, states)
    ev = env._task_evaluator
    if getattr(ev, "_goal", None) is None:
        ev.configure_from_randomization(env.model, json.loads((ep / "randomization.json").read_text()))
    return env, ev, states, actions, scene_qpos


def success_frames(ev, qpos):
    ev.reset(nworld=len(qpos))
    return np.asarray(ev.evaluate_qpos_batch(np.asarray(qpos, np.float32)).success, bool)


def replay(env, init_qpos, actions, n_dec):
    m, d = env.model, env.data
    m.opt.timestep = CONTROL_PERIOD / n_dec
    env._physics_dt, env._control_decimation = CONTROL_PERIOD / n_dec, n_dec
    d.qpos[:] = init_qpos
    d.qvel[:] = 0.0
    d.qacc_warmstart[:] = 0.0
    d.ctrl[:] = 0.0
    mujoco.mj_forward(m, d)
    out = [d.qpos.copy()]
    for a in actions[1:]:  # states[t] is the response to actions[t]
        env.step(np.asarray(a), render_obs=False)
        out.append(d.qpos.copy())
    return np.stack(out)


def worker(ep_str):
    ep = Path(ep_str)
    try:
        env, ev, states, actions, scene_qpos = build(ep)
        g = ev._goal
        adrs = free_adrs(env.model)
        init = scene_qpos[0].astype(np.float64)
        demo_ok = success_frames(ev, scene_qpos)
        res = {"episode": ep.name, "T": len(actions), "demo_final": bool(demo_ok[-1]), "demo_any": bool(demo_ok.any()),
               "variants": {}}
        base = None
        for n in DECIMATIONS:
            q = replay(env, init, actions, n)
            if not np.isfinite(q).all():
                res["variants"][n] = {"diverged": True}
                continue
            ok = success_frames(ev, q)
            pos = lambda arr, a: arr[:, a:a + 3]  # noqa: E731
            err_demo = np.stack([np.linalg.norm(pos(q, a) - pos(scene_qpos, a), axis=1) for a in adrs], 1)  # [T, objs]
            v = {"final": bool(ok[-1]), "any": bool(ok.any()),
                 "mover_err_demo_final": float(np.linalg.norm(pos(q, g["mover_adr"])[-1] - pos(scene_qpos, g["mover_adr"])[-1])),
                 "max_err_demo_final": float(err_demo[-1].max()), "max_err_demo_any": float(err_demo.max())}
            if base is None:
                base = q
            err_base = np.stack([np.linalg.norm(pos(q, a) - pos(base, a), axis=1) for a in adrs], 1)
            v["mover_err_base_final"] = float(np.linalg.norm(pos(q, g["mover_adr"])[-1] - pos(base, g["mover_adr"])[-1]))
            v["max_err_base_final"] = float(err_base[-1].max())
            res["variants"][n] = v
        env.close()
        return res
    except Exception as e:  # noqa: BLE001
        return {"episode": ep.name, "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "abc_src/abc/cache/train_sim"))
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    eps = sorted(Path(args.root).glob("episode_*"))
    rng = np.random.default_rng(0)
    eps = [eps[i] for i in sorted(rng.choice(len(eps), min(args.episodes, len(eps)), replace=False))]
    with mp.get_context("fork").Pool(args.workers) as pool:
        results = pool.map(worker, [str(e) for e in eps], chunksize=1)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
    good = [r for r in results if "error" not in r]
    for r in results:
        if "error" in r:
            print("ERR", r["episode"], r["error"])
    print(f"{len(good)} episodes replayed ({len(results) - len(good)} errors); "
          f"demo success: final {sum(r['demo_final'] for r in good)}/{len(good)}, any {sum(r['demo_any'] for r in good)}/{len(good)}")
    print(f"{'dt [ms]':>8s} {'success final':>14s} {'success any':>12s} {'same as 2ms':>12s} "
          f"{'mover err vs demo [cm] med/p90':>32s} {'<1cm vs demo':>13s} {'<1cm vs 2ms':>12s} {'diverged':>9s}")
    for n in DECIMATIONS:
        vs = [(r, r["variants"].get(n, r["variants"].get(str(n)))) for r in good]
        div = sum(bool(v.get("diverged")) for _, v in vs)
        vs = [(r, v) for r, v in vs if not v.get("diverged")]
        if not vs:
            print(f"{1000 * CONTROL_PERIOD / n:8.2f} all diverged")
            continue
        fin = sum(v["final"] for _, v in vs)
        anyv = sum(v["any"] for _, v in vs)
        base = [r["variants"].get(17, r["variants"].get("17")) for r, _ in vs]
        same = sum(v["final"] == b["final"] for (_, v), b in zip(vs, base) if not b.get("diverged"))
        e = np.array([v["mover_err_demo_final"] for _, v in vs]) * 100
        eb = np.array([v["mover_err_base_final"] for _, v in vs]) * 100
        print(f"{1000 * CONTROL_PERIOD / n:8.2f} {fin:>8d}/{len(vs):<5d} {anyv:>7d}/{len(vs):<4d} {same:>7d}/{len(vs):<4d} "
              f"{np.median(e):>14.1f} / {np.quantile(e, .9):<13.1f} {np.mean(e < 1) * 100:>12.0f}% {np.mean(eb < 1) * 100:>11.0f}% {div:>9d}")


if __name__ == "__main__":
    main()
