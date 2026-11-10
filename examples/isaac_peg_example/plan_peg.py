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
ap.add_argument("--selftest", action="store_true")
AppLauncher.add_app_launcher_args(ap)
args, hydra_args = ap.parse_known_args()
for ov in ("env.scene.insertive_object=peg", "env.scene.receptive_object=peghole"):
    if not any(h.startswith(ov.split("=")[0] + "=") for h in hydra_args):
        hydra_args.append(ov)
if not any("reset_from_reset_states.params.reset_types" in h for h in hydra_args):
    hydra_args += ["env.events.reset_from_reset_states.params.reset_types=[ObjectAnywhereEEAnywhere]",
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

from peg_problem import NCTRL, PegConfig, PegProblem  # noqa: E402

TASK = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0"


def load_start_states(env):
    """The ObjectAnywhereEEAnywhere reset dataset for this peg/hole pair (dict of [M,...] tensors)."""
    from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils as ou
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import sample_state_data_set

    pair = ou.compute_pair_dir(env.scene["insertive_object"].cfg.spawn.usd_path,
                               env.scene["receptive_object"].cfg.spawn.usd_path)
    f = f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset_isaaclab3/Resets/{pair}/resets_ObjectAnywhereEEAnywhere.pt"
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
    hand0, hand1 = S1[:, 6:9], out[:, -1, 6:9]
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
    pre = planner_cfg.get("preemptive_expansion", {})
    n_envs = args.num_envs or pcfg0.num_controls * (1 + int(pre.get("max_preemptive", 0)))
    env_cfg.scene.num_envs = n_envs
    env_cfg.episode_length_s = 1.0e6
    env_cfg.terminations.time_out = None
    env_cfg.terminations.abnormal_robot = None
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()
    ds, M = load_start_states(env)
    print(f"[plan_peg] {n_envs} Isaac envs; {M} start states in the dataset")
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
    for seed in args.seeds:
        put_start(env, ds, seed % M)
        pcfg = PegConfig.from_dict({**cfg.get("problem", {}), "seed": seed})
        prob = PegProblem(env, pcfg, cfg.get("task"))
        start = prob.set_start(None, 0)
        print(f"\n=== peg seed {seed} (start state {seed % M}/{M}): h0={float(prob.heuristic(start[None])[0]):.2f}s "
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
               "dyn_rows": p["dyn_rows"], "dyn_time_s": p["dyn_time"], "bad_rows": p["bad_rows"],
               "max_peg_step": p["max_peg_step"], "max_hand_step": p["max_hand_step"], "bound": prob.max_object_step, "edge_s": prob.edge_dt,
               "first_solution_expansion": None if first is None else first + 1}
        print(f"search: success={ok} in {dt:.1f}s, {rec['expansions']} expansions ({rec['preemptive']} preemptive), "
              f"{p['dyn_rows']} edges simulated ({p['dyn_rows'] / max(p['dyn_time'], 1e-9):.0f}/s), bad rows {p['bad_rows']}; "
              f"largest peg step {p['max_peg_step']:.3f} m, hand step {p['max_hand_step']:.3f} m vs bound {prob.max_object_step:.3f} m"
              + ("  ** BOUND VIOLATED **" if max(p["max_peg_step"], p["max_hand_step"]) > prob.max_object_step else ""))
        if ok:
            path = planner.get_best_path()  # goal-first, last col = cost-to-come
            dense = torch.flip(path[:, :-1], dims=[0])
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
