"""Replay saved peg edges with contact sensors, to see whether 'flung' edges show a force spike.

UNREVIEWED diagnostic (not part of the planner). Input: flung_seed*.pt written by plan_peg.py (edges whose peg moved
> 0.1 m, or were marked bad, plus a ~2% baseline sample of ordinary edges). Each edge is restored exactly and replayed
with the unmodified env.step; contact sensors on the peg (vs hole, net) and on the four gripper finger bodies (vs peg)
are sampled after every physics step (60 per edge) by wrapping scene.update.

    python diag_contacts.py --flung out/flung_seed0.pt --num-envs 128 --out out/contacts
"""
import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

ap = argparse.ArgumentParser()
ap.add_argument("--flung", required=True)
ap.add_argument("--num-envs", type=int, default=128)
ap.add_argument("--max-edges", type=int, default=512, help="max flung and max baseline edges to replay")
ap.add_argument("--config", default=str(HERE / "config.yml"))
ap.add_argument("--out", default=str(HERE / "out_contacts"))
AppLauncher.add_app_launcher_args(ap)
args, hydra_args = ap.parse_known_args()
for ov in ("env.scene.insertive_object=peg", "env.scene.receptive_object=peghole"):
    if not any(h.startswith(ov.split("=")[0] + "=") for h in hydra_args):
        hydra_args.append(ov)
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import uwlab_tasks  # noqa: E402,F401
import yaml  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from peg_problem import PegConfig, PegProblem  # noqa: E402

TASK = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0"


def load_start_states(env, reset_type):
    from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils as ou
    from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import sample_state_data_set

    pair = ou.compute_pair_dir(env.scene["insertive_object"].cfg.spawn.usd_path,
                               env.scene["receptive_object"].cfg.spawn.usd_path)
    f = f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset_isaaclab3/Resets/{pair}/resets_{reset_type}.pt"
    data = torch.load(ou.safe_retrieve_file_path(f))
    data.pop("quat_convention", None)
    M = len(data["initial_state"]["articulation"]["robot"]["joint_position"])
    return sample_state_data_set(data, torch.arange(M), env.device), M


def put_start(env, ds, idx):
    sc = env.scene
    st = {k: {n: {f: v.clone() for f, v in d.items()} for n, d in grp.items()}
          for k, grp in sc.get_state(is_relative=True).items()}
    for grp in ("articulation", "rigid_object"):
        for name, fields in ds["initial_state"].get(grp, {}).items():
            if name in st[grp]:
                for f, v in fields.items():
                    st[grp][name][f] = v[idx:idx + 1].expand(env.num_envs, *v.shape[1:]).clone().to(env.device)
    sc.reset_to(st, env_ids=None, is_relative=True)
    sc.write_data_to_sim()
    sc.update(env.physics_dt)


def pct(x, q=(50, 90, 99, 100)):
    x = np.asarray(x, dtype=np.float64)
    return " ".join(f"p{p}={np.percentile(x, p):.3g}" for p in q) if len(x) else "-"


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    cfg = yaml.safe_load(open(args.config))
    pcfg = PegConfig.from_dict(cfg.get("problem", {}))
    d = torch.load(args.flung)
    print(f"[contacts] {len(d['states'])} flung/bad edges, "
          f"{0 if d['normal_states'] is None else len(d['normal_states'])} baseline edges, start {d['start_idx']} ({d['reset_type']})")

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = 1.0e6
    env_cfg.terminations.time_out = None
    env_cfg.terminations.abnormal_robot = None
    env_cfg.scene.robot.spawn.activate_contact_sensors = True
    env_cfg.scene.insertive_object.spawn.activate_contact_sensors = True
    env_cfg.scene.receptive_object.spawn.activate_contact_sensors = True
    env_cfg.scene.peg_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject", filter_prim_paths_expr=["{ENV_REGEX_NS}/ReceptiveObject"], history_length=0)
    env_cfg.scene.finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_finger", filter_prim_paths_expr=["{ENV_REGEX_NS}/InsertiveObject"], history_length=0)
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()
    print("[contacts] physics handles decimation:", getattr(env, "_physics_handles_decimation", "n/a"),
          "| finger bodies:", env.scene["finger_contact"].body_names, "| peg bodies:", env.scene["peg_contact"].body_names)

    ds, M = load_start_states(env, d["reset_type"])
    put_start(env, ds, d["start_idx"] % M)
    prob = PegProblem(env, pcfg, cfg.get("task"))
    prob.set_start(None, 0)  # fixes hole pose etc. for this start state
    N, T = prob.N, prob.T
    peg_s, fing_s, peg_obj = env.scene["peg_contact"], env.scene["finger_contact"], env.scene["insertive_object"]

    rec = {"on": False, "peg_net": [], "peg_hole": [], "peg_fing": [], "peg_speed": [], "peg_vel": []}
    orig_update = env.scene.update

    def update(dt, *a, **k):
        r = orig_update(dt, *a, **k)
        if rec["on"]:
            rec["peg_net"].append(peg_s.data.net_forces_w.torch[:, 0].norm(dim=-1))                 # [N]
            rec["peg_hole"].append(peg_s.data.force_matrix_w.torch[:, 0, 0].norm(dim=-1))           # [N]
            rec["peg_fing"].append(fing_s.data.force_matrix_w.torch[:, :, 0].norm(dim=-1).sum(dim=1))  # [N] sum of 4 fingers
            rec["peg_speed"].append(peg_obj.data.root_lin_vel_w.torch.norm(dim=-1))                 # [N]
            rec["peg_vel"].append(peg_obj.data.root_lin_vel_w.torch.clone())                         # [N,3]
        return r

    env.scene.update = update

    def replay(S, U):
        out = {k: [] for k in ("vel_t", "net", "hole", "fing", "speed", "t_force", "t_speed", "pstep", "grip", "net_t", "fing_t", "hole_t")}
        for c0 in range(0, len(S), N):
            s = S[c0:c0 + N].to(env.device, torch.float32)
            u = U[c0:c0 + N].to(env.device, torch.float32)
            nb = len(s)
            if nb < N:
                s = torch.cat([s, s[:1].expand(N - nb, -1)])
                u = torch.cat([u, u[:1].expand(N - nb, -1)])
            act, cmd = prob._resolve(s, u)
            prob._write_states(s)
            for k in ("peg_net", "peg_hole", "peg_fing", "peg_speed", "peg_vel"):
                rec[k].clear()
            rec["on"] = True
            for _ in range(T):
                env.step(act)
            rec["on"] = False
            net, hole, fing, spd = (torch.stack(rec[k], 1) for k in ("peg_net", "peg_hole", "peg_fing", "peg_speed"))  # [N, T*dec]
            cur = prob._read_states(cmd)
            pstep = (cur[:, prob.o_peg:prob.o_peg + 3] - s[:, prob.o_peg:prob.o_peg + 3]).norm(dim=1)
            tot = net.cpu().numpy()[:nb]
            out["net"].append(net.amax(1)[:nb].cpu()); out["hole"].append(hole.amax(1)[:nb].cpu())
            out["fing"].append(fing.amax(1)[:nb].cpu()); out["speed"].append(spd.amax(1)[:nb].cpu())
            out["t_speed"].append(spd.argmax(1)[:nb].cpu()); out["t_force"].append(net.argmax(1)[:nb].cpu())
            out["net_t"].append(net[:nb].cpu()); out["fing_t"].append(fing[:nb].cpu()); out["hole_t"].append(hole[:nb].cpu())
            out["vel_t"].append(torch.stack(rec["peg_vel"], 1)[:nb].cpu())
            out["pstep"].append(pstep[:nb].cpu()); out["grip"].append(u[:nb, 6].cpu())
        return {k: torch.cat(v).numpy() for k, v in out.items()}

    n = args.max_edges
    F = replay(d["states"][:n], d["controls"][:n])
    B = replay(d["normal_states"][:n], d["normal_controls"][:n]) if d["normal_states"] is not None else None

    print("\n== replay fidelity (replayed peg step vs recorded in planner run) ==")
    rf = np.abs(F["pstep"] - d["peg_step"][:len(F["pstep"])].numpy())
    print("flung: |dpeg_step| ", pct(rf), "  (large = replay is not reproducing the planner's edge)")
    print(f"\n== peak forces over each 0.5 s edge (N), {len(F['net'])} flung vs {0 if B is None else len(B['net'])} baseline edges ==")
    for name, key in (("peg net (any contact)", "net"), ("peg vs hole", "hole"), ("fingers vs peg (sum of 4)", "fing")):
        print(f"{name:26s} flung   : {pct(F[key])}")
        if B is not None:
            print(f"{'':26s} baseline: {pct(B[key])}")
    print(f"peak peg speed (m/s)       flung   : {pct(F['speed'])}")
    if B is not None:
        print(f"{'':26s} baseline: {pct(B['speed'])}")
        for name, key in (("net", "net"), ("hole", "hole"), ("fing", "fing")):
            thr = np.percentile(B[key], 99)
            print(f"flung edges with peak {name} force > baseline p99 ({thr:.3g} N): {(F[key] > thr).mean():.0%}")
    big = F["speed"] > 2.0
    if big.any():
        print(f"\nflung edges with peak peg speed > 2 m/s: {int(big.sum())}; "
              f"substep of peak force minus substep of peak speed: median {np.median(F['t_force'][big] - F['t_speed'][big]):.0f} "
              f"(negative = force spike comes first)")
    print("\ngripper command on flung edges (open = u>=0, close = u<0):",
          {"open": int((F['grip'] >= 0).sum()), "close": int((F['grip'] < 0).sum())},
          "| baseline:", None if B is None else {"open": int((B['grip'] >= 0).sum()), "close": int((B['grip'] < 0).sum())})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "contacts.npz", **{f"flung_{k}": v for k, v in F.items()},
                        **({} if B is None else {f"base_{k}": v for k, v in B.items()}),
                        recorded_peg_step=d["peg_step"].numpy())
    env.close()


if __name__ == "__main__":
    main()
    app.close()
