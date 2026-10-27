"""Run the planner with a yaml on a problem set; save success + per-expansion best cost for every search.
python run_planner.py CONFIG.yml PROBLEMS.json OUT.pkl [--repeats 3] [--max-exp N]
Maps are loaded from each problem's "map_file", relative to the problems file."""
import argparse, copy, json, os, pickle
import numpy as np, torch, yaml
from ighastar.scripts.common_utils import create_planner

ap = argparse.ArgumentParser()
ap.add_argument("config"); ap.add_argument("problems"); ap.add_argument("out")
ap.add_argument("--repeats", type=int, default=1); ap.add_argument("--max-exp", type=int, default=None)
a = ap.parse_args()

conf = yaml.safe_load(open(a.config))
info = conf["experiment_info_default"]
if a.max_exp:
    info["max_expansions"] = a.max_exp
planner = create_planner(copy.deepcopy(conf), bidirectional=info.get("bidirectional", True))

root = os.path.dirname(a.problems)
runs = []
for p in json.load(open(a.problems)):
    cost = np.load(os.path.join(root, p["map_file"]))                      # [N, N] cost map, 255 free / 0 blocked
    bitmap = torch.zeros(*cost.shape, 2)
    bitmap[..., 0] = torch.from_numpy(cost)
    off = cost.shape[0] * info["node_info"]["map_res"] / 2                 # planner frame = problem frame + half the window
    start = torch.tensor([p["start"][0] + off, p["start"][1] + off] + p["start"][2:], dtype=torch.float32)
    goal = torch.tensor([p["goal"][0] + off, p["goal"][1] + off] + p["goal"][2:], dtype=torch.float32)
    for rep in range(a.repeats):
        ok = bool(planner.search(start, goal, bitmap, info["max_expansions"], info["hysteresis"], True))
        prof = planner.get_profiler_info()
        runs.append(dict(problem_id=p["problem_id"], rep=rep, success=ok, expansions=int(prof[7]), best_cost=np.asarray(prof[9], dtype=np.float64)))
pickle.dump(runs, open(a.out, "wb"))
print(f"{a.out}: {sum(r['success'] for r in runs)}/{len(runs)} searches succeeded")
