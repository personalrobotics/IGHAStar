#!/usr/bin/env python3
"""Plan abc_sim tasks with IGHA* (generic env + MJWarp rollouts) and export episodes.

    cd examples/abc_example
    python plan_abc.py --task put_relative --seeds 0 1 2 --video

For every seed: reset the abc task (its own randomizer), build the generic
problem, search, then check the plan two ways:

* ``warp_replay``: re-simulate the plan's edges from the start state in MJWarp
  (the planner's own physics). Should reproduce the plan.
* ``cpu_replay``: feed the plan's 30 Hz actions open-loop to abc's CPU MuJoCo
  env (``env.step``) from the same start state and score it with abc's own task
  evaluator. Different engine, so contact-rich plans can drift.

Each solved seed writes ``<out>/<task>_seed<k>.npz`` (30 Hz policy-space states
and actions, full qpos/qvel, prompt, metadata) and, with ``--video``, an mp4.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import abc_sim  # noqa: E402
from abc_problem import PROBLEMS, ProblemConfig  # noqa: E402
from ighastar.scripts.common_utils import create_planner  # noqa: E402


def to_numpy(x):
    return x.detach().cpu().numpy()


def decode_path(problem, path: torch.Tensor):
    """Planner path (goal-first, last col g) -> chronological dense states incl. start."""
    dense = torch.flip(path[:, :-1], dims=[0])
    g = torch.flip(path[:, -1], dims=[0])
    states = torch.cat([problem.start[None], dense], 0)
    g = torch.cat([torch.zeros(1), g])
    return states, g


def edges_from_states(problem, states):
    """Recover (ctrl_from, ctrl_to) per edge from the dense ctrl column."""
    T = problem.cfg.ctrl_steps_per_edge
    ctrl = problem.ctrl(states)
    n_edges = (states.shape[0] - 1) // T
    assert n_edges * T == states.shape[0] - 1, "dense path length is not a multiple of T"
    froms = torch.stack([ctrl[e * T] for e in range(n_edges)])
    tos = torch.stack([ctrl[(e + 1) * T] for e in range(n_edges)])
    return froms, tos


def warp_replay(problem, states):
    """Re-simulate the plan edge by edge in MJWarp from the start state."""
    T = problem.cfg.ctrl_steps_per_edge
    froms, tos = edges_from_states(problem, states)
    s = problem.start[None]
    out = [s]
    for c0, c1 in zip(froms, tos):
        u = (c1 - problem.ctrl(s)[0])[None]
        nxt = problem.dynamics_joint(s, u)  # [1, T, n]; u is the realized joint delta
        out.append(nxt[0])
        s = nxt[:, -1]
    replay = torch.cat(out, 0)
    return replay


def cpu_replay(env, problem, actions: np.ndarray):
    """Open-loop replay of policy-space actions in abc's CPU MuJoCo env."""
    import mujoco

    q0 = to_numpy(problem.qpos(problem.start)).astype(np.float64)
    v0 = to_numpy(problem.qvel(problem.start)).astype(np.float64)
    env.data.qpos[:] = q0
    env.data.qvel[:] = v0
    env.data.qacc_warmstart[:] = 0.0
    env.data.ctrl[:] = to_numpy(problem.policy_to_actuator(problem.ctrl(problem.start)[None])[0])
    mujoco.mj_forward(env.model, env.data)
    qs, ok_any, info = [q0.copy()], False, {}
    for a in actions:
        _, _, _, _, info = env.step(a, render_obs=False)
        qs.append(np.asarray(env.data.qpos).copy())
        ok_any |= bool(info.get("task_success", False))
    return np.stack(qs), bool(info.get("task_success", False)), ok_any, info


def render_video(env, qpos_seq: np.ndarray, out_mp4: Path, camera: str, fps: int, size=(360, 480)):
    import imageio.v2 as imageio
    import mujoco

    model = env.model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=size[0], width=size[1])
    frames = []
    for q in qpos_seq:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())
    renderer.close()
    imageio.mimsave(out_mp4, frames, fps=fps)


def mover_summary(problem, states):
    mover, ref, along, perp = problem._cone(states)
    return {
        "along": float(along[-1]),
        "perp": float(perp[-1]),
        "mover_z": float(mover[-1, 2]),
        "mover_disp": float((mover[-1] - problem.mover_start).norm()),
    }


def run_seed(env, spec, task, seed, cfg, args):
    t_reset = time.perf_counter()
    env.reset(seed=seed, randomize=True)
    overrides = {k: getattr(args, k) for k in ("sampler", "tcp_site", "tcp_step", "tcp_rot_step", "grasp_fraction", "plan_substeps")
                 if getattr(args, k) is not None}
    pcfg = ProblemConfig.from_dict({**cfg.get("problem", {}), **overrides, "seed": seed, "log_tree": args.save_tree})
    problem = PROBLEMS[task](env, pcfg, cfg["task"][task])
    t_build = time.perf_counter() - t_reset
    print(f"\n=== {task} seed {seed}: '{env.prompt}'  arms={problem.active_arms} "
          f"n={problem.n} (F={problem.F}, nq={problem.nq}, nv={problem.nv})")

    start = problem.start
    v0 = problem.validity(start[None])[0].item()
    h0 = problem.heuristic(start[None])[0].item()
    print(f"start valid={bool(v0)}  h={h0:.2f}s  already_goal={problem.goal_test(start)}  "
          f"(problem build {t_build:.1f}s)")
    print(f"admissible h: max object step {problem.max_object_step:.3f} m per {problem.edge_dt:.3f}s edge "
          f"(= {problem.max_object_step / problem.edge_dt:.2f} m/s)")

    planner_cfg = dict(cfg["planner"])
    if args.max_expansions is not None:
        planner_cfg["max_expansions"] = args.max_expansions
    for key in ("resolution", "tolerance", "division_factor", "max_level"):
        if getattr(args, key) is not None:
            planner_cfg[key] = getattr(args, key)
    if args.save_tree and planner_cfg.get("preemptive_expansion", {}).get("enabled"):
        print("--save-tree: switching preemptive expansion off for this run (the tree log needs it)")
        planner_cfg["preemptive_expansion"] = {**planner_cfg["preemptive_expansion"], "enabled": False}
    configs = problem.planner_config(planner_cfg)
    info = configs["experiment_info_default"]
    print("hash (resolution/tolerance): " + ", ".join(
        f"{n}={r:g}/{t:g}" for n, r, t in zip(problem.feature_names, info["resolution"], info["tolerance"]))
        + f"  division_factor={info['division_factor']} max_level={info['max_level']}")
    planner = create_planner(configs, bidirectional=False)
    goal = torch.zeros(problem.n)
    world = torch.zeros(1)

    t0 = time.perf_counter()
    ok = planner.search(start, goal, world, int(info["max_expansions"]), int(info["hysteresis"]), True)
    search_s = time.perf_counter() - t0
    prof = planner.get_profiler_info()
    expansions = prof[7]
    p = problem.profile
    rate = problem.rollout.world_steps / max(p["dyn_time"], 1e-9)
    rec = {
        "task": task, "seed": seed, "prompt": env.prompt, "success": bool(ok),
        "search_s": search_s, "expansions": int(expansions),
        "preemptive_expansions": int(planner.get_preemptive_expansions()),
        "dyn_calls": p["dyn_calls"], "dyn_rows": p["dyn_rows"], "dyn_time_s": p["dyn_time"],
        "sim_ctrl_steps_per_s": rate, "overflows": p["overflows"],
        "active_arms": list(problem.active_arms), "h0": h0,
        "max_object_step_bound": problem.max_object_step,
        "max_mover_step_seen": p.get("max_mover_step", 0.0),
        "min_tcp_mover_dist": p.get("min_tcp_mover_dist", float("nan")),
        "sampler": problem.cfg.sampler,
        "ik_fail_frac": p.get("ik_fail", 0) / max(p.get("ik_rows", 0), 1),
        "grasp_fraction": problem.cfg.grasp_fraction if problem.cfg.sampler == "tcp" else 0.0,
        "grasp_pool": int(len(getattr(problem, "grasp_pool", []))),
        "plan_substeps": problem.cfg.plan_substeps or int(env._control_decimation),
    }
    cost_curve = [float(c) for c in prof[9]]
    first = next((i for i, c in enumerate(cost_curve) if c < 1e5), None)  # 1e5 = planner's 'no solution' sentinel
    rec["first_solution_expansion"] = None if first is None else first + 1
    rec["first_solution_cost"] = None if first is None else cost_curve[first]
    rec["cost_vs_expansion"] = cost_curve[:: max(1, len(cost_curve) // 200)]
    print(f"search: success={ok} in {search_s:.1f}s, {expansions} expansions "
          f"({rec['preemptive_expansions']} preemptive), {p['dyn_calls']} rollout batches, "
          f"{p['dyn_rows']} edges simulated, dynamics {p['dyn_time']:.1f}s "
          f"(~{rate:.0f} world-ctrl-steps/s), overflows={p['overflows']}")
    print(f"admissibility check: largest mover step seen {rec['max_mover_step_seen']:.3f} m "
          f"vs bound {rec['max_object_step_bound']:.3f} m; closest active TCP to mover {rec['min_tcp_mover_dist']:.3f} m"
          + ("  ** BOUND VIOLATED **" if rec['max_mover_step_seen'] > rec['max_object_step_bound'] else ""))
    tree_path = None
    if args.save_tree:
        problem.cfg.log_tree = False  # replays below must not extend the log
        tree = problem.tree_arrays()
        sol_ids = np.zeros(0, np.int64)
        if ok:
            st, _ = decode_path(problem, planner.get_best_path())
            T = problem.cfg.ctrl_steps_per_edge
            sol_ids = np.asarray(problem.child_ids_of(st[T::T]), np.int64)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        tree_path = out_dir / f"{task}_seed{seed}_tree.npz"
        np.savez_compressed(
            tree_path, **tree, solution_child_ids=sol_ids,
            start_qpos=to_numpy(problem.qpos(problem.start)),
            start_feat=to_numpy(problem.start[: problem.F]),
            feature_names=np.asarray(problem.feature_names),
            active_arms=np.asarray(problem.active_arms),
            task=task, seed=seed, prompt=env.prompt,
            edge_dt=problem.edge_dt, ctrl_steps_per_edge=problem.cfg.ctrl_steps_per_edge,
            mover_adr=getattr(problem, "mover_adr", -1), ref_adr=getattr(problem, "ref_adr", -1),
        )
        n_child = len(tree.get("child_valid", []))
        print(f"tree: {len(tree.get('parent_call', []))} expansions, {n_child} children -> {tree_path}")
    rec["tree"] = None if tree_path is None else str(tree_path)
    if ok:
        print(f"first solution at expansion {rec['first_solution_expansion']} (cost "
              f"{rec['first_solution_cost']:.2f}); final cost {cost_curve[-1]:.2f}")
    if not ok:
        problem.close()
        return rec

    path = planner.get_best_path()
    states, g = decode_path(problem, path)
    actions = to_numpy(problem.ctrl(states[1:]))  # action t takes state t -> t+1
    duration = (states.shape[0] - 1) * problem.dt_ctrl
    rec.update(cost=float(g[-1]), plan_steps=int(states.shape[0] - 1), plan_duration_s=duration,
               plan_goal=bool(problem.goal_mask(states[-1:])[0]), **{f"plan_{k}": v for k, v in mover_summary(problem, states).items()})
    print(f"plan: {rec['plan_steps']} steps @30Hz = {duration:.2f}s, cost {rec['cost']:.2f}, "
          f"final along={rec['plan_along']:.3f} perp={rec['plan_perp']:.3f} z={rec['plan_mover_z']:.3f}")

    wr = warp_replay(problem, states)
    mv = problem.qpos(wr)[:, problem.mover_adr:problem.mover_adr + 3]
    mp = problem.qpos(states)[:, problem.mover_adr:problem.mover_adr + 3]
    rec["warp_replay_goal"] = bool(problem.goal_mask(wr[-1:])[0])
    rec["warp_replay_max_mover_err"] = float((mv - mp).norm(dim=1).max())
    print(f"warp replay: goal={rec['warp_replay_goal']}  max mover deviation "
          f"{rec['warp_replay_max_mover_err'] * 100:.2f} cm")

    cq, cpu_final, cpu_any, cpu_info = cpu_replay(env, problem, actions)
    cpu_mover = cq[-1, problem.mover_adr:problem.mover_adr + 3]
    rec["cpu_replay_success_final"] = cpu_final
    rec["cpu_replay_success_any"] = cpu_any
    rec["cpu_replay_mover_err"] = float(np.linalg.norm(cpu_mover - to_numpy(mp[-1])))
    te = cpu_info.get("task_eval", {})
    print(f"cpu replay (abc evaluator): success_final={cpu_final} ever={cpu_any} "
          f"along={te.get('mover_along_m')} perp={te.get('mover_perp_m')} "
          f"mover end err {rec['cpu_replay_mover_err'] * 100:.2f} cm")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    qpos = to_numpy(problem.qpos(states))
    from abc_sim.env import project_policy_state_batch

    obs_state = project_policy_state_batch(qpos, env._qpos_indices, env._gripper_indices)
    meta = env._last_randomization.metadata if env._last_randomization is not None else {}
    np.savez_compressed(
        out / f"{task}_seed{seed}.npz",
        state=obs_state.astype(np.float32),  # [T+1, 14] measured joints, policy space
        action=actions.astype(np.float32),  # [T, 14] commanded targets, policy space
        qpos=qpos.astype(np.float32),
        qvel=to_numpy(problem.qvel(states)).astype(np.float32),
        cpu_replay_qpos=cq.astype(np.float32),
        g=to_numpy(g),
        prompt=env.prompt,
        metadata=json.dumps(meta, default=str),
        record=json.dumps(rec),
    )
    if args.video:
        cam = cfg.get("video", {}).get("camera", "overhead")
        render_video(env, qpos, out / f"{task}_seed{seed}_plan.mp4", cam, fps=30)
        render_video(env, cq, out / f"{task}_seed{seed}_cpu_replay.mp4", cam, fps=30)
        print(f"videos -> {out}")
    problem.close()
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="put_relative")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--config", default=str(HERE / "config.yml"))
    ap.add_argument("--max-expansions", type=int, default=None)
    ap.add_argument("--resolution", type=float, nargs="+", default=None,
                    help="one value per hashed feature (put_relative: 11), overrides config")
    ap.add_argument("--tolerance", type=float, nargs="+", default=None,
                    help="one value per hashed feature, overrides config")
    ap.add_argument("--division-factor", dest="division_factor", type=float, default=None)
    ap.add_argument("--max-level", dest="max_level", type=int, default=None)
    ap.add_argument("--out", default=str(HERE.parent.parent / "Content" / "abc_example"))
    ap.add_argument("--sampler", choices=["joint", "tcp"], default=None,
                    help="control sampler, overrides problem.sampler (tcp = grasp-site deltas + ssik IK)")
    ap.add_argument("--tcp-site", dest="tcp_site", choices=["flange", "grasp"], default=None)
    ap.add_argument("--tcp-step", dest="tcp_step", type=float, default=None, help="max TCP translation per edge [m]")
    ap.add_argument("--tcp-rot-step", dest="tcp_rot_step", type=float, default=None, help="max TCP rotation per edge [rad]")
    ap.add_argument("--grasp-fraction", dest="grasp_fraction", type=float, default=None,
                    help="tcp sampler: fraction of controls that reach toward a sampled grasp of the object (0 = random only)")
    ap.add_argument("--plan-substeps", dest="plan_substeps", type=int, default=None,
                    help="physics substeps per 34 ms control period while planning (17 = abc's 2 ms; 8 = 4.25 ms); "
                         "the CPU replay check always uses abc's 2 ms")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--save-tree", action="store_true", help="log every expansion to <task>_seed<k>_tree.npz")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    spec = abc_sim.get_task_spec(args.task)
    task = spec.env_task
    if task not in PROBLEMS:
        raise SystemExit(f"no IGHA* problem for abc task {task!r}; have {sorted(PROBLEMS)}")
    env = abc_sim.make_env(task=task, prompt=spec.prompt, render_cameras=False)

    records = [run_seed(env, spec, task, s, cfg, args) for s in args.seeds]
    env.close()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{task}_summary.json", "w") as f:
        json.dump(records, f, indent=2)
    n_ok = sum(r["success"] for r in records)
    n_cpu = sum(r.get("cpu_replay_success_final", False) for r in records)
    print(f"\nsolved {n_ok}/{len(records)}; abc CPU-replay success {n_cpu}/{len(records)}")


if __name__ == "__main__":
    main()
