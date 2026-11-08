#!/usr/bin/env python3
"""Play a planned abc_sim episode in Viser (no search tree needed).

Reads the ``<task>_seed<k>.npz`` that ``plan_abc.py`` writes for every solved seed
(works for runs with preemptive expansion on), rebuilds the same abc scene, and
plays the plan:

* ``plan (MJWarp)``: the qpos sequence the planner simulated,
* ``abc CPU replay``: the same actions replayed open-loop in abc's CPU MuJoCo,
* the grasp-site paths of both arms (black = active arm) and the mover's path (orange),
* abc's goal region for the mover (green), and abc's own evaluator verdict
  (along / perp / placed / success) for the frame on screen.

    python viser_abc_episode.py --episode ../../Content/abc_example/put_relative_seed13.npz --port 8090
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import viser

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import abc_sim  # noqa: E402
from mjviser import ViserMujocoScene  # noqa: E402

DT_CTRL = 1.0 / 30.0  # abc control period


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    z = np.load(args.episode, allow_pickle=False)
    rec = json.loads(str(z["record"]))
    task, seed = rec["task"], int(rec["seed"])
    seqs = {"plan (MJWarp)": z["qpos"], "abc CPU replay": z["cpu_replay_qpos"]}
    n_frames = len(z["qpos"])

    spec = abc_sim.get_task_spec(task)
    env = abc_sim.make_env(task=spec.env_task, prompt=spec.prompt, render_cameras=False)
    env.reset(seed=seed, randomize=True)
    model = env.model
    ev = env._task_evaluator
    goal = ev._goal
    data = mujoco.MjData(model)
    sites = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_grasp_site") for s in ("left", "right")}
    mover_adr = int(goal["mover_adr"])
    arms = list(rec.get("active_arms", ["left"]))
    print(f"{task} seed {seed}: '{rec['prompt']}'  {n_frames} frames ({(n_frames - 1) * DT_CTRL:.2f} s), arms={arms}")

    def sites_xyz(qpos_seq):
        out = {s: [] for s in sites}
        for q in qpos_seq:
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            for s, i in sites.items():
                out[s].append(data.site_xpos[i].copy())
        return {s: np.array(v) for s, v in out.items()}

    paths = {name: sites_xyz(q) for name, q in seqs.items()}
    verdict = {}
    for name, q in seqs.items():  # abc's own evaluator, one world per frame
        ev.reset(nworld=len(q))
        verdict[name] = ev.evaluate_qpos_batch(np.asarray(q, np.float32))

    server = viser.ViserServer(port=args.port)
    scene = ViserMujocoScene(server, model, num_envs=1)
    scene.camera_tracking_enabled = False
    server.scene.remove_by_name("/fixed_bodies/world/table_plane")

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.position = (-0.45, 0.0, 1.65)
        client.camera.look_at = (0.55, 0.0, 0.85)

    ref_adr = int(goal["reference_adr"])
    ref = np.asarray(seqs["plan (MJWarp)"][0][ref_adr:ref_adr + 3], float)  # reference at the start
    ax = np.asarray(goal["axis"], float)
    c = 0.5 * (ev.along_min_m + ev.along_max_m)
    yaw = np.arctan2(ax[1], ax[0])
    server.scene.add_box(
        "/goal_region", color=(40, 200, 60),
        dimensions=(ev.along_max_m - ev.along_min_m, 2 * ev.perp_max_m, 0.004),
        position=(ref[0] + ax[0] * c, ref[1] + ax[1] * c, 0.752),
        wxyz=(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)), opacity=0.4,
    )

    with server.gui.add_folder("Episode"):
        server.gui.add_markdown(f"**{rec['prompt']}**  \nplanner goal reached: {rec.get('plan_goal')}, "
                                f"abc CPU replay success: {rec.get('cpu_replay_success_final')}")
        dd = server.gui.add_dropdown("source", list(seqs), initial_value="plan (MJWarp)")
        s_frame = server.gui.add_slider("frame (30 Hz)", 0, n_frames - 1, 1, 0)
        cb_play = server.gui.add_checkbox("play", False)
        s_speed = server.gui.add_slider("speed x", 0.1, 4.0, 0.1, 1.0)
        cb_paths = server.gui.add_checkbox("show paths", True)
        info = server.gui.add_markdown("")

    def draw_paths(_=None):
        for name in seqs:
            for k in ("/paths/" + name + "/mover",):
                server.scene.remove_by_name(k)
            for s in sites:
                server.scene.remove_by_name(f"/paths/{name}/{s}")
        if not cb_paths.value:
            return
        name = dd.value
        q = seqs[name]
        mover = q[:, mover_adr:mover_adr + 3]
        server.scene.add_line_segments(f"/paths/{name}/mover", np.stack([mover[:-1], mover[1:]], 1), (255, 140, 0), thickness=0.006)
        for s in arms:
            p = paths[name][s]
            server.scene.add_line_segments(f"/paths/{name}/{s}", np.stack([p[:-1], p[1:]], 1), (0, 0, 0), thickness=0.004)

    def show(_=None):
        name, k = dd.value, int(s_frame.value)
        data.qpos[:] = seqs[name][k]
        mujoco.mj_forward(model, data)
        scene.update_from_mjdata(data)
        m = verdict[name].metrics
        info.content = (f"t = {k * DT_CTRL:.2f} s  \nalong {float(m['mover_along_m'][k]):.3f} m "
                        f"(need {ev.along_min_m}-{ev.along_max_m}), perp {float(m['mover_perp_m'][k]):.3f} m "
                        f"(<= {ev.perp_max_m}), in cone: {bool(m['mover_in_cone'][k])}, "
                        f"placed: {bool(m['mover_placed'][k])}  \n**abc success: {bool(verdict[name].success[k])}**")

    dd.on_update(lambda _: (draw_paths(), show()))
    s_frame.on_update(show)
    cb_paths.on_update(draw_paths)
    draw_paths()
    show()
    print(f"open http://localhost:{args.port}")

    def loop():
        while True:
            if cb_play.value:
                s_frame.value = (int(s_frame.value) + 1) % n_frames
                time.sleep(DT_CTRL / float(s_speed.value))
            else:
                time.sleep(0.05)

    threading.Thread(target=loop, daemon=True).start()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
