"""Start-state helpers shared by diagnostics (copied from diag_contacts.py). UNREVIEWED."""
import torch


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
