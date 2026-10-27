"""Sweep planner-yaml values. Each --set is KEY=v1;v2;... (KEY is a dotted path under experiment_info_default, values are yaml).
python sweep.py BASE.yml PROBLEMS.json OUT_DIR --set node_info.K_strafe="0.3;2.0" --set node_info.min_vel="-1.55;-0.6" [--mode grid|oat] [--repeats 1]
grid = every combination, oat = one key at a time (base value elsewhere). One yaml + pickle per run in OUT_DIR, plus summary.csv."""
import argparse, copy, csv, itertools, os, pickle, subprocess, sys
import numpy as np, yaml

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("problems"); ap.add_argument("out")
ap.add_argument("--set", action="append", default=[]); ap.add_argument("--mode", default="grid", choices=["grid", "oat"])
ap.add_argument("--repeats", type=int, default=1); ap.add_argument("--max-exp", type=int, default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
base = yaml.safe_load(open(a.base))
axes = {k: [yaml.safe_load(v) for v in vs.split(";")] for k, vs in (s.split("=", 1) for s in a.set)}

def get(d, path):
    for p in path.split("."): d = d[p]
    return d
def put(d, path, v):
    *ks, last = path.split("."); [d := d[k] for k in ks]; d[last] = v

if a.mode == "grid":
    combos = [dict(zip(axes, vals)) for vals in itertools.product(*axes.values())]
else:
    combos = [{}] + [{k: v} for k, vs in axes.items() for v in vs if v != get(base["experiment_info_default"], k)]
rows = []
for c in combos:
    name = ",".join(f"{k.split('.')[-1]}={v}" for k, v in c.items()) or "base"
    conf = copy.deepcopy(base)
    for k, v in c.items(): put(conf["experiment_info_default"], k, v)
    f = os.path.join(a.out, name)
    yaml.safe_dump(conf, open(f + ".yml", "w"))
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_planner.py"), f + ".yml", a.problems, f + ".pkl", "--repeats", str(a.repeats)]
    if a.max_exp: cmd += ["--max-exp", str(a.max_exp)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    runs = pickle.load(open(f + ".pkl", "rb"))
    ok = [r for r in runs if r["success"]]
    rows.append(dict(config=name, searches=len(runs), success=len(ok) / len(runs), median_first_solution_exp=float(np.median([np.argmax(r["best_cost"] < 1e4) for r in ok])) if ok else np.nan,
                     median_final_cost=float(np.median([r["best_cost"][-1] for r in ok])) if ok else np.nan))
    print(rows[-1], flush=True)
with open(os.path.join(a.out, "summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
