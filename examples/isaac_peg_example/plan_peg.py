"""Plan UWLab OmniReset UR5e peg insertion with IGHA* (Isaac Sim physics).

UNREVIEWED: written by Claude, not yet read or reviewed by a human.

    python plan_peg.py --seeds 0 1 2 --max-expansions 3000 [--num-envs 256] [--selftest]

Start states: the "object anywhere, end-effector anywhere" reset dataset of OmniReset
(seed k = k-th start state of the dataset). Goal: OmniReset's own success test.
Headless by default (Isaac Lab 3.0); run inside the env_uwlab venv.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))  # repo root: ighastar

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--max-expansions", type=int, default=None)
ap.add_argument("--num-envs", type=int, default=None, help="Isaac envs = nodes per batch * num_controls")
ap.add_argument("--config", default=str(HERE / "config.yml"))
ap.add_argument("--out", default=str(HERE / "out"))
ap.add_argument("--backend", default="physx", choices=["physx", "newton"],
                help="physics engine; newton = MuJoCo-Warp via newton_robot_usd.py (one start state per process)")
ap.add_argument("--selftest", action="store_true")
ap.add_argument("--reset-type", default="ObjectAnywhereEEAnywhere",
                help="OmniReset reset dataset used for start states (e.g. ObjectPartiallyAssembledEEGrasped)")
ap.add_argument("--start-idx", type=int, nargs="+", default=None,
                help="dataset index per seed (default: seed %% M)")
ap.add_argument("--scan-out", default=None, help="with --scan: save all rows (idx, dist, angle, goal, grip, h0) as .npy")
ap.add_argument("--save-tree", action="store_true",
                help="log every expansion to <out>/peg_seed<k>_tree.npz (+ scene_meshes.npz) for viser_peg_tree.py; "
                     "(works with preemptive expansion: pre-expanded vertices that were never popped are in the tree too)")
ap.add_argument("--tree-no-preemptive", action="store_true", help="with --save-tree: turn preemptive expansion off (old behaviour)")
ap.add_argument("--scan", type=int, default=0, help="print alignment/grasp stats of the first N start states and exit")
AppLauncher.add_app_launcher_args(ap)
args, hydra_args = ap.parse_known_args()
for ov in ("env.scene.insertive_object=peg", "env.scene.receptive_object=peghole"):
    if not any(h.startswith(ov.split("=")[0] + "=") for h in hydra_args):
        hydra_args.append(ov)
if not any("reset_from_reset_states.params.reset_types" in h for h in hydra_args):
    hydra_args += [f"env.events.reset_from_reset_states.params.reset_types=[{args.reset_type}]",
                   "env.events.reset_from_reset_states.params.probs=[1.0]"]
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import uwlab_tasks  # noqa: E402,F401
import yaml  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import newton_robot_usd  # noqa: E402
from peg_problem import (NCTRL, PegConfig, PegProblem, apply_arm_sysid, apply_osc_config,  # noqa: E402
                         disable_domain_randomization)

TASK = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0"


def load_start_states(env):
    """The ObjectAnywhereEEAnywhere reset dataset for this peg/hole pair (dict of [M,...] tensors)."""
    from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils as ou
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import sample_state_data_set

    pair = ou.compute_pair_dir(env.scene["insertive_object"].cfg.spawn.usd_path,
                               env.scene["receptive_object"].cfg.spawn.usd_path)
    f = f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset_isaaclab3/Resets/{pair}/resets_{args.reset_type}.pt"
    data = torch.load(ou.safe_retrieve_file_path(f))
    data.pop("quat_convention", None)
    M = len(data["initial_state"]["articulation"]["robot"]["joint_position"])
    return sample_state_data_set(data, torch.arange(M), env.device), M


def put_start(env, ds, idx):
    """Write dataset start state ``idx`` into every env (all envs identical)."""
    sc = env.scene
    st = {k: {n: {f: v.clone() for f, v in d.items()} for n, d in grp.items()}
          for k, grp in sc.get_state(is_relative=True).items()}
    ini = ds["initial_state"]
    for grp in ("articulation", "rigid_object"):
        for name, fields in ini.get(grp, {}).items():
            if name in st[grp]:
                for f, v in fields.items():
                    st[grp][name][f] = v[idx:idx + 1].expand(env.num_envs, *v.shape[1:]).clone().to(env.device)
    sc.reset_to(st, env_ids=None, is_relative=True)
    sc.write_data_to_sim()
    sc.update(env.physics_dt)


def replay_plan(env, prob, ds, start_idx, start, states, U):
    """Re-run a found plan edge by edge and check the goal with OmniReset's own success term. UNREVIEWED.

    A: restore every planner node exactly (what the planner assumed) and compare the edge's end state with the planner's.
    B: one continuous rollout from the start state with the same controls (no restores between edges), as a real
       execution would run; reports the goal after every edge and at the end."""
    print("\n=== replay of the plan (%d edges) ===" % len(U))
    N, T = prob.N, prob.T
    U = torch.as_tensor(U, dtype=torch.float32)
    print("A) restore each node, apply the edge's control:")
    for e in range(len(U)):
        s0 = states[e * T].to(env.device)[None].repeat(N, 1)
        out = prob.dynamics(s0.cpu(), U[e][None].repeat(N, 1))[0, -1]
        want = states[(e + 1) * T]
        print(f"  edge {e}: end-state |diff| feat {float((out[:prob.F] - want[:prob.F]).abs().max()):.4f} "
              f"peg pos {float((out[prob.o_peg:prob.o_peg + 3] - want[prob.o_peg:prob.o_peg + 3]).norm()):.4f} m  "
              f"goal(mine)={bool(prob.goal_mask(out[None])[0])}")
    print("B) continuous rollout from the start state:")
    put_start(env, ds, start_idx)
    s = prob.set_start(None, 0).to(env.device)[None].repeat(N, 1)
    for e in range(len(U)):
        act, cmd = prob._resolve(s, U[e][None].repeat(N, 1).to(env.device))
        for _ in range(T):
            env.step(act)
        own = bool(prob.prog.success[0])
        s = prob._read_states(cmd)
        pp = s[0, prob.o_peg:prob.o_peg + 3].cpu()
        print(f"  edge {e}: peg pos {[round(float(x), 3) for x in pp]}  goal(mine)={bool(prob.goal_mask(s[:1].cpu())[0])}  "
              f"OmniReset success={own}")
    print(f"  final: goal(mine)={bool(prob.goal_mask(s[:1].cpu())[0])}  OmniReset success={bool(prob.prog.success[0])}")


def selftest(env, prob, ds, M):
    print("\n=== selftest ===")
    put_start(env, ds, 0)
    s0 = prob.set_start(None, 0)
    print("start node dim", s0.shape, "finite", bool(torch.isfinite(s0).all()))
    print("start goal?", prob.goal_test(s0), " h =", float(prob.heuristic(s0[None])[0]),
          " alignment", [float(x) for x in prob.alignment(s0[None, prob.o_peg:prob.o_peg + 7].to(env.device))])
    # 1. restore round trip
    S = s0[None].repeat(prob.N, 1).to(env.device)
    prob._write_states(S)
    back = prob._read_states(S[:, prob.o_grip]).cpu()
    print("round-trip max |err| per block: q", float((back[:, prob.o_q:prob.o_q + prob.J] - s0[prob.o_q:prob.o_q + prob.J]).abs().max()),
          "peg", float((back[:, prob.o_peg:prob.o_peg + 7] - s0[prob.o_peg:prob.o_peg + 7]).abs().max()))
    # 2. dynamics: determinism + batch invariance (same node, same control, rows 0 and N-1 and a repeat)
    u = prob.sample_controls()
    B = prob.N
    S = s0[None].repeat(B, 1)
    U = u[:B] if len(u) >= B else u.repeat(B // len(u) + 1, 1)[:B]
    a = prob.dynamics(S, U)
    b = prob.dynamics(S, U)
    print("dyn finite rows", int(torch.isfinite(a).all(dim=(1, 2)).sum()), "/", B,
          "repeat max |diff|", float(torch.nan_to_num(a - b).abs().max()))
    blocks = {"feat": (0, prob.F), "joint_pos": (prob.o_q, prob.o_q + prob.J), "joint_vel": (prob.o_qd, prob.o_qd + prob.J),
              "peg_pose": (prob.o_peg, prob.o_peg + 7), "peg_vel": (prob.o_pegv, prob.n)}
    d = torch.nan_to_num(a - b).abs()
    print("repeat diff by block (max over rows,t):", {k: round(float(d[..., lo:hi].max()), 5) for k, (lo, hi) in blocks.items()})
    print("row spread within one batch (same node, same control): ",
          {k: round(float(a[0:1, :, lo:hi].sub(a[:1, :, lo:hi]).abs().max()), 5) for k, (lo, hi) in list(blocks.items())[:1]})
    same = a[torch.arange(B)[(U == U[0]).all(dim=1)]]
    if len(same) > 1:
        print("rows with identical controls: n =", len(same), "max spread", float((same - same[0]).abs().max()))
    sub = prob.dynamics(S[:5], U[:5])
    print("batch-size-5 vs full max |diff|", float(torch.nan_to_num(sub - a[:5]).abs().max()))
    # 3. goal test vs OmniReset's own success buffer after env.step
    prob._write_states(S.to(env.device))
    env.step(torch.zeros(prob.N, 7, device=env.device))
    own = prob.prog.success.cpu()
    mine = prob.goal_mask(prob._read_states(S[:, prob.o_grip].to(env.device)).cpu())
    print("goal test agrees with progress_context.success:", bool((own == mine).all()))
    # 4. speed
    for reps in (3,):
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(reps):
            prob.dynamics(S, U)
        torch.cuda.synchronize(); dt = time.perf_counter() - t
        print(f"speed: {reps} batches of {B} edges ({prob.T} steps) in {dt:.2f}s -> {reps * B / dt:.1f} edge-sims/s")
    # 5. do the target-directed rows move the hand toward their targets?
    S1 = s0[None].repeat(B, 1)
    U1 = torch.zeros(B, NCTRL)
    U1[:, 7] = torch.where(torch.arange(B) % 2 == 0, 1.0, 2.0)  # even rows: peg point, odd rows: hole
    U1[:, 8] = torch.randint(0, 2 ** 30, (B,)).float()
    act, cmd = prob._resolve(S1.to(env.device), U1.to(env.device))
    dbg = prob._dbg
    out = prob.dynamics(S1, U1)
    hand0, hand1 = S1[:, prob.i_hand], out[:, -1, prob.i_hand]
    moved = (hand1 - hand0)
    want = dbg["delta_w"].cpu()
    cosine = torch.nn.functional.cosine_similarity(moved, want, dim=1)
    d0 = (dbg["target"].cpu() - hand0).norm(dim=1)
    d1 = (dbg["target"].cpu() - hand1).norm(dim=1)
    for name, m in (("peg-point rows", U1[:, 7] == 1), ("hole rows", U1[:, 7] == 2)):
        print(f"{name}: commanded |action[:3]| mean {float(act[m.to(env.device), :3].norm(dim=1).mean()):.2f}, "
              f"hand moved {float(moved[m].norm(dim=1).mean()) * 100:.1f} cm (wanted {float(want[m].norm(dim=1).mean()) * 100:.1f} cm to target), "
              f"cos(moved, wanted) {float(cosine[m].mean()):.2f}, dist to target {float(d0[m].mean()) * 100:.1f} -> {float(d1[m].mean()) * 100:.1f} cm")
    print("grip cmd peg rows", float(cmd[(U1[:, 7] == 1).to(env.device)].mean()), " start grip cmd", float(s0[prob.o_grip]))
    print("max_peg_step so far", prob.profile["max_peg_step"], "bound", prob.max_object_step)


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    cfg = yaml.safe_load(open(args.config))
    pcfg0 = PegConfig.from_dict(cfg.get("problem", {}))
    planner_cfg = dict(cfg["planner"])
    if args.save_tree and args.tree_no_preemptive:
        planner_cfg["preemptive_expansion"] = {**planner_cfg.get("preemptive_expansion", {}), "enabled": False}
        print("--save-tree --tree-no-preemptive: preemptive expansion off")
    pre = planner_cfg.get("preemptive_expansion", {})
    n_envs = args.num_envs or pcfg0.num_controls * (1 + (int(pre.get("max_preemptive", 0)) if pre.get("enabled", True) else 0))
    env_cfg.scene.num_envs = n_envs
    apply_osc_config(env_cfg, cfg.get("osc"))
    if not cfg.get("domain_randomization", True):
        print("[plan_peg] domain randomization off:", disable_domain_randomization(env_cfg))
    env_cfg.episode_length_s = 1.0e6
    env_cfg.terminations.time_out = None
    env_cfg.terminations.abnormal_robot = None
    if args.backend == "newton":
        if not args.start_idx or len(set(args.start_idx)) != 1:
            raise SystemExit("--backend newton needs --start-idx with a single start state (the fixed robot base is baked into the USD)")
        usd = newton_robot_usd.configure_newton(env_cfg, args.reset_type, args.start_idx[0], Path(args.out) / "newton_usd")
        print("[plan_peg] Newton robot USD:", usd)
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()
    apply_arm_sysid(env, bool(cfg.get("arm_sysid", False)), float(cfg.get("arm_friction_scale", 1.0)))
    ph = env.sim.cfg.physics
    if args.backend == "newton":
        print(f"[plan_peg] physics: Newton MJWarp, dt={env.sim.cfg.dt:.5f} decimation={env.cfg.decimation}, Newton collision pipeline")
    else:
      print(f"[plan_peg] physics: dt={env.sim.cfg.dt:.5f} decimation={env.cfg.decimation} solver_type={ph.solver_type} "
          f"pos_iters<={ph.max_position_iteration_count} vel_iters<={ph.max_velocity_iteration_count} "
          f"bounce_thr={ph.bounce_threshold_velocity} fric_offset_thr={ph.friction_offset_threshold} "
          f"fric_corr_dist={ph.friction_correlation_distance}")
    ds, M = load_start_states(env)
    print(f"[plan_peg] {n_envs} Isaac envs; {M} start states in the dataset")
    if args.scan:
        prob = PegProblem(env, pcfg0, cfg.get("task"))
        rows = []
        for k in range(min(args.scan, M)):
            put_start(env, ds, k)
            s0 = prob.set_start(None, 0)
            d, a = [float(x) for x in prob.alignment(s0[None, prob.o_peg:prob.o_peg + 7].to(env.device))][:2]
            rows.append((k, d, a, bool(prob.goal_test(s0)), float(s0[prob.i_grip]), float(prob.heuristic(s0[None])[0])))
        if args.scan_out:
            np.save(args.scan_out, np.array(rows, dtype=np.float64))
        rows.sort(key=lambda r: r[1])
        print("idx  dist_m  angle  goal  grip  h0")
        for r in rows[:25]:
            print("%4d %7.3f %6.3f %5s %5.2f %6.2f" % r)
        ng = [r for r in rows if not r[3]]
        print("goal states: %d of %d; non-goal (closest 15 shown):" % (len(rows) - len(ng), len(rows)))
        for r in ng[:15]:
            print("%4d %7.3f %6.3f %5s %5.2f %6.2f" % r)
        print("not-goal states with grip>0.5:", sum(1 for r in rows if not r[3] and r[4] > 0.5), "of", len(rows))
        env.close()
        return
    if args.selftest:
        put_start(env, ds, 0)
        prob = PegProblem(env, pcfg0, cfg.get("task"))
        selftest(env, prob, ds, M)
        env.close()
        return
    from ighastar.scripts.common_utils import create_planner

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for i, seed in enumerate(args.seeds):
        put_start(env, ds, (args.start_idx[i] if args.start_idx else seed) % M)
        pcfg = PegConfig.from_dict({**cfg.get("problem", {}), "seed": seed, "log_tree": args.save_tree})
        prob = PegProblem(env, pcfg, cfg.get("task"))
        start = prob.set_start(None, 0)
        if args.save_tree:
            from scene_export import export_scene_meshes
            nm = export_scene_meshes(list(prob.robot.body_names), str(out / "scene_meshes.npz"))
            print("exported scene meshes:", nm)
        print(f"\n=== peg seed {seed} (start state {(args.start_idx[i] if args.start_idx else seed) % M}/{M}, {args.reset_type}): h0={float(prob.heuristic(start[None])[0]):.2f}s "
              f"already_goal={prob.goal_test(start)}  edge={prob.edge_dt:.2f}s  bound={prob.max_object_step:.3f} m/edge")
        pc = dict(planner_cfg)
        if args.max_expansions is not None:
            pc["max_expansions"] = args.max_expansions
        configs = prob.planner_config(pc)
        info = configs["experiment_info_default"]
        planner = create_planner(configs, bidirectional=False)
        t0 = time.perf_counter()
        ok = planner.search(start, torch.zeros(prob.n), torch.zeros(1), int(info["max_expansions"]),
                            int(info["hysteresis"]), True)
        dt = time.perf_counter() - t0
        prof = planner.get_profiler_info()
        p = prob.profile
        cost_curve = [float(c) for c in prof[9]]
        first = next((i for i, c in enumerate(cost_curve) if c < 1e5), None)
        rec = {"seed": seed, "success": bool(ok), "search_s": dt, "expansions": int(prof[7]),
               "preemptive": int(planner.get_preemptive_expansions()), "dyn_calls": p["dyn_calls"],
               "dyn_rows": p["dyn_rows"], "dyn_time_s": p["dyn_time"], "bad_rows": p["bad_rows"], "accel_rows": p.get("accel_rows", 0),
               "max_peg_step": p["max_peg_step"], "max_hand_step": p["max_hand_step"], "bound": prob.max_object_step, "edge_s": prob.edge_dt,
               "first_solution_expansion": None if first is None else first + 1,
               "avg_succ_us": float(prof[0]), "preemptive_hits": int(planner.get_preemptive_hits()),
               **prob.closest_summary()}
        cl = prob.closest
        print(f"closest peg approach (valid edge end states): distance {cl['dist'][0] * 1e3:.1f} mm (angle {cl['dist'][1]:.3f} rad, call {cl['dist'][2]}); "
              f"within angle thr {prob.rot_thr:.3f} rad: {cl['dist_in_angle'][0] * 1e3:.1f} mm (call {cl['dist_in_angle'][2]}); "
              f"smallest angle {cl['angle'][1]:.3f} rad at {cl['angle'][0] * 1e3:.1f} mm; goal thr {prob.pos_thr * 1e3:.1f} mm / {prob.rot_thr:.3f} rad")
        np.savez_compressed(out / f"closest_seed{seed}.npz", best_dist_per_call=np.asarray(prob.closest_curve, np.float32),
                            **{f"state_{k}": v.numpy() for k, v in prob.closest_state.items()},
                            feature_names=np.asarray(prob.feature_names), o_peg=prob.o_peg, record=json.dumps(rec))
        print(f"avg_succ={rec['avg_succ_us']:.0f} us, preemptive hits {rec['preemptive_hits']} of {rec['preemptive']} preemptive")
        print(f"search: success={ok} in {dt:.1f}s, {rec['expansions']} expansions ({rec['preemptive']} preemptive), "
              f"{p['dyn_rows']} edges simulated ({p['dyn_rows'] / max(p['dyn_time'], 1e-9):.0f}/s), bad rows {p['bad_rows']} (of which peg-accel {p.get('accel_rows', 0)}); "
              f"largest peg step {p['max_peg_step']:.3f} m, hand step {p['max_hand_step']:.3f} m vs bound {prob.max_object_step:.3f} m"
              + ("  ** BOUND VIOLATED **" if max(p["max_peg_step"], p["max_hand_step"]) > prob.max_object_step else ""))
        if prob.flung:
            torch.save({"states": torch.stack([f[0] for f in prob.flung]), "controls": torch.stack([f[1] for f in prob.flung]),
                        "peg_step": torch.tensor([f[2] for f in prob.flung]), "bad": torch.tensor([f[3] for f in prob.flung]),
                        "normal_states": torch.stack([f[0] for f in prob.normal]) if prob.normal else None,
                        "normal_controls": torch.stack([f[1] for f in prob.normal]) if prob.normal else None,
                        "normal_peg_step": torch.tensor([f[2] for f in prob.normal]),
                        "start_idx": (args.start_idx[i] if args.start_idx else seed), "reset_type": args.reset_type},
                       out / f"flung_seed{seed}.pt")
        if args.save_tree:
            tree = prob.tree_arrays()
            hp = prob.hole_pose.to(prob.device)[None]
            ha_p, ha_q = prob.hole_offset.combine(hp[:, :3], hp[:, 3:7])
            sol = np.zeros(0, np.int64)
            if ok:
                path_ = planner.get_best_path()
                st_ = torch.cat([start[None], torch.flip(path_[:, :start.shape[0]], dims=[0])], 0)
                sol = np.asarray([prob._tree_index.get(prob._key(r), -1) for r in st_[prob.T::prob.T]], np.int64)
            np.savez_compressed(
                out / f"peg_seed{seed}_tree.npz", **tree, solution_child_ids=sol,
                start_feat=start[:prob.F].numpy(), feature_names=np.asarray(prob.feature_names), start_peg=start[prob.o_peg:prob.o_peg + 7].numpy(),
                start_bp=prob.start_bp.numpy(), body_names=np.asarray(list(prob.robot.body_names)),
                hole_pose=prob.hole_pose.cpu().numpy(), goal_pos=ha_p[0].cpu().numpy(), goal_quat=ha_q[0].cpu().numpy(),
                pos_thr=prob.pos_thr, rot_thr=prob.rot_thr, edge_dt=prob.edge_dt, steps_per_edge=prob.T,
                step_dt=float(env.step_dt), seed=seed, start_idx=(args.start_idx[i] if args.start_idx else seed),
                reset_type=args.reset_type)
            print(f"tree: {len(tree['parent_call'])} expansions, {len(tree['child_valid'])} children -> {out / f'peg_seed{seed}_tree.npz'}")
            if ok and (sol >= 0).all():
                replay_plan(env, prob, ds, (args.start_idx[i] if args.start_idx else seed) % M, start, st_, tree["child_u"][sol])
        if prob.diag:
            np.save(out / f"edge_diag_seed{seed}.npy", torch.cat(prob.diag).numpy())
        if ok:
            path = planner.get_best_path()  # goal-first rows: [state (n_dims), time, cost-to-come]
            dense = torch.flip(path[:, :start.shape[0]], dims=[0])
            states = torch.cat([start[None], dense], 0)
            E = (states.shape[0] - 1) // prob.T
            print(f"first solution at expansion {rec['first_solution_expansion']}; plan = {E} edges = {E * prob.edge_dt:.1f}s")
            rec.update(plan_edges=E, plan_goal=bool(prob.goal_mask(states[-1:])[0]))
            np.savez_compressed(out / f"peg_seed{seed}.npz", states=states.numpy(), hole_pose=prob.hole_pose.cpu().numpy(),
                                record=json.dumps(rec))
        records.append(rec)
    (out / "summary.json").write_text(json.dumps(records, indent=1))
    print(f"\nsolved {sum(r['success'] for r in records)}/{len(records)}")
    env.close()


main()
app.close()
