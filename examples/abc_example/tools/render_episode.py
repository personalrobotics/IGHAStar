"""Render the plan and CPU-replay videos of a saved episode npz (plan_abc.py output).

    python tools/render_episode.py <task>_seed<k>.npz [--camera top] [--out DIR]

Use this when plan_abc.py's --video failed (e.g. the configured camera does not exist in
the scene): it reads qpos / cpu_replay_qpos from the npz and re-renders, no planning.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plan_abc import render_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--camera", default="top")
    ap.add_argument("--out", default=None)
    ap.add_argument("--size", type=int, nargs=2, default=[480, 640], metavar=("H", "W"))
    args = ap.parse_args()
    import abc_sim

    env = None
    for path in map(Path, args.npz):
        z = np.load(path, allow_pickle=False)
        import json
        rec = json.loads(str(z["record"]))
        if env is None:
            spec = abc_sim.get_task_spec(rec["task"])
            env = abc_sim.make_env(task=spec.env_task, prompt=spec.prompt, render_cameras=False)
        env.reset(seed=int(rec["seed"]), randomize=True)
        out = Path(args.out) if args.out else path.parent
        out.mkdir(parents=True, exist_ok=True)
        for key, name in (("qpos", "plan"), ("cpu_replay_qpos", "cpu_replay")):
            f = out / f"{path.stem}_{name}_{args.camera}.mp4"
            render_video(env, z[key].astype(np.float64), f, args.camera, fps=30, size=tuple(args.size))
            print(f)
    env.close()


if __name__ == "__main__":
    main()
