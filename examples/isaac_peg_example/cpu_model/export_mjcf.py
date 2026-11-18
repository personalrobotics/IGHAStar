"""Newton's MuJoCo model of the planning scene as MJCF (newton_scene.xml, the geometry/joint source for build_static.py)
plus Newton's per-pair contact parameters (newton_contact_params.npz). Nominal physics (domain randomization off).
UNREVIEWED.   python cpu_model/export_mjcf.py --out <dir> [--start-idx 3261]  (or build_cpu_model.sh)
"""
import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

EX = Path(__file__).resolve().parent.parent  # examples/isaac_peg_example
sys.path.insert(0, str(EX))
sys.path.insert(0, str(EX.parent.parent))

ap = argparse.ArgumentParser()
ap.add_argument("--reset-type", default="ObjectPartiallyAssembledEEGrasped")
ap.add_argument("--start-idx", type=int, default=3261)
ap.add_argument("--config", default=str(EX / "config.yml"))
ap.add_argument("--out", required=True, help="output directory (shared by the cpu_model scripts)")
AppLauncher.add_app_launcher_args(ap)
args, hydra_args = ap.parse_known_args()
for ov in ("env.scene.insertive_object=peg", "env.scene.receptive_object=peghole"):
    if not any(h.startswith(ov.split("=")[0] + "=") for h in hydra_args):
        hydra_args.append(ov)
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import uwlab_tasks  # noqa: E402,F401
import yaml  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import newton_robot_usd  # noqa: E402
from diag_contacts_lib import load_start_states, put_start  # noqa: E402
from peg_problem import NCTRL, PegConfig, PegProblem, apply_arm_sysid, apply_osc_config  # noqa: E402

TASK = "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0"
NE = 64


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    cfg = yaml.safe_load(open(args.config))
    env_cfg.scene.num_envs = NE
    env_cfg.episode_length_s = 1.0e6
    env_cfg.terminations.time_out = None
    env_cfg.terminations.abnormal_robot = None
    apply_osc_config(env_cfg, cfg.get("osc"))
    from peg_problem import disable_domain_randomization
    print("[cpu_model] randomization off:", disable_domain_randomization(env_cfg))
    newton_robot_usd.configure_newton(env_cfg, args.reset_type, args.start_idx, OUT / "usd")
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()
    ds, M = load_start_states(env, args.reset_type)
    put_start(env, ds, args.start_idx % M)
    apply_arm_sysid(env, bool(cfg.get("arm_sysid", False)), float(cfg.get("arm_friction_scale", 1.0)))
    pc = {**cfg.get("problem", {}), "num_controls": 64, "seed": 0, "edge_graph": False}
    prob = PegProblem(env, PegConfig.from_dict(pc), cfg.get("task"))
    print(f"[rec] osc_targets {prob.cfg.osc_targets} N {prob.N} T {prob.T} decimation {env.cfg.decimation} dt {env.physics_dt}")
    from isaaclab_newton.physics import NewtonManager as NM
    sol = NM._solver
    m, mw, dw = sol.mj_model, sol.mjw_model, sol.mjw_data
    nq, nv = m.nq, m.nv

    s0 = prob.set_start(None, 0)
    U = torch.zeros(NE, NCTRL); U[:, 6] = -1.0
    prob.dynamics(s0[None].repeat(NE, 1), U)  # one GPU edge so Isaac-side properties reach the Newton model
    # --- per-contact parameters MuJoCo-Warp actually used (Newton pipeline + force-space solref), per geom pair
    dw = NM._solver.mjw_data
    rows = {}
    UA = prob.sample_controls()[:NE]
    st = s0[None].repeat(NE, 1)
    for rep in range(6):
        orr = prob._read_states
        def rd(cmd, _o=orr):
            out = _o(cmd)
            n = int(dw.nacon.numpy()[0])
            if n:
                c = dw.contact
                g = c.geom.numpy()[:n]; w = c.worldid.numpy()[:n]
                sr, si, fr, srf, im, dim = (c.solref.numpy()[:n], c.solimp.numpy()[:n], c.friction.numpy()[:n],
                                            c.solreffriction.numpy()[:n], c.includemargin.numpy()[:n], c.dim.numpy()[:n])
                for i in range(n):
                    k = (int(min(g[i])), int(max(g[i])))
                    rows.setdefault(k, []).append(np.concatenate([sr[i], si[i], fr[i], srf[i], [im[i], dim[i]]]))
            return out
        prob._read_states = rd
        tr = prob.dynamics(st, UA if rep % 2 == 0 else prob.sample_controls()[:NE])
        prob._read_states = orr
        ok = torch.isfinite(tr[:, -1]).all(1)
        st = torch.where(ok[:, None], tr[:, -1], st)
    tab = {}
    for k, v in sorted(rows.items()):
        v = np.array(v)
        tab[k] = v.mean(0)
        print(f"[exp] pair {k} n {len(v):5d} solref {np.round(v[:, 0:2].mean(0), 5).tolist()} (sd {np.round(v[:, 0:2].std(0), 5).tolist()}) "
              f"solimp {np.round(v[:, 2:7].mean(0), 4).tolist()} friction {np.round(v[:, 7:12].mean(0), 4).tolist()} "
              f"solreffric {np.round(v[:, 12:14].mean(0), 4).tolist()} includemargin {np.round(v[:, 14].mean(), 5)} dim {np.unique(v[:, 15]).tolist()}")
    np.savez(OUT / "newton_contact_params.npz", pairs=np.array(list(tab.keys())), params=np.array(list(tab.values())))
    nm = NM._model
    print("[exp] shape_material_ke", np.unique(np.round(nm.shape_material_ke.numpy(), 3)).tolist(),
          "kd", np.unique(np.round(nm.shape_material_kd.numpy(), 3)).tolist(),
          "solref_mode", np.unique(nm.mujoco.solref_mode.numpy()).tolist() if hasattr(nm, "mujoco") and hasattr(nm.mujoco, "solref_mode") else None)
    from newton.solvers import SolverMuJoCo
    out = str(OUT / "newton_scene.xml")
    try:
        SolverMuJoCo(NM._model, separate_worlds=True, use_mujoco_contacts=False, save_to_mjcf=out)
    except Exception as e:  # noqa: BLE001
        print("[exp] separate_worlds export failed:", repr(e))
        raise
    print("[exp] wrote", out)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
