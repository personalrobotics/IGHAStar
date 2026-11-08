# How to: IGHA* on abc_sim tasks (picking up the work on a new machine)

This branch (`abc_example`) adds `examples/abc_example/`: amazon-far/abc's
simulated tasks plugged into IGHA* through the **generic** (Python-callback)
environment, with batched MJWarp rollouts on the GPU. The IGHA* search code is
untouched. The new code is all under `examples/abc_example/`, plus this file.

This file covers:
1. what was set up on the original machine (versions),
2. how to rebuild it from scratch,
3. how to rerun everything done so far,
4. what was found, and what's still open.

---

## 1. Reference machine

| | |
| --- | --- |
| OS | Ubuntu 22.04.5 |
| GPU | RTX 4090, driver 575.57.08 |
| Compiler | gcc 11.4, `ninja` on `PATH` (IGHA* JIT-compiles its C++ via `torch.utils.cpp_extension`) |
| Python | 3.12, a **uv**-managed venv owned by the abc repo (uv 0.11.2) |
| abc | `https://github.com/amazon-far/abc` @ `d0832d1` |
| Key packages | torch 2.11.0+cu128, mujoco 3.8.1, mujoco-warp 3.8.0.3, warp-lang 1.13.0, numpy 2.2.6, viser 1.1.1, mjviser 0.0.16, ssik 8.2.0 |

The full pinned package list of that venv is in
`examples/abc_example/tools/abc_venv_freeze.txt`.

Directory layout used originally (any layout works; paths below use these
names):

```
~/abc_src/abc                      # amazon-far/abc clone, holds the venv (.venv)
~/ros2_ws/src/IGHAStar             # this repo, branch abc_example
~/ssik_src/ssik                    # optional: personalrobotics/ssik clone (source only, not needed to run)
```

There's no separate conda env. Everything (abc, IGHA*, ssik, viser) lives in
**abc's** venv, because abc pins Python 3.12 and its own
torch/mujoco/mujoco-warp versions.

---

## 2. Install from scratch

### 2.1 System prerequisites

```bash
sudo apt install -y build-essential git git-lfs
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv
pip install --user ninja   # or: uv pip install ninja into the venv below
nvidia-smi                 # need a working NVIDIA driver (CUDA 12.8-capable)
```

A CUDA toolkit isn't required, because torch/warp ship their own runtime. The
reference machine had CUDA 12.9 installed anyway, which nsys (§3.5) uses.

### 2.2 abc and its venv

```bash
mkdir -p ~/abc_src && cd ~/abc_src
git clone https://github.com/amazon-far/abc.git && cd abc
git checkout d0832d1          # the commit everything was tested against
uv python pin 3.12
uv sync                       # creates .venv
uv run prepare.py --sim-task put_relative   # scene assets for put_relative (~137 MB, into abc_sim/models/assets)
```

abc gitignores its `uv.lock`, so `uv sync` resolves fresh versions. If anything
behaves differently, pin to the exact versions used originally:

```bash
uv pip install --python .venv/bin/python -r ~/ros2_ws/src/IGHAStar/examples/abc_example/tools/abc_venv_freeze.txt
```

### 2.3 IGHA* (this branch) into abc's venv

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone -b abc_example https://github.com/personalrobotics/IGHAStar.git
cd ~/abc_src/abc
uv pip install --python .venv/bin/python -e ~/ros2_ws/src/IGHAStar
```

The first planner construction JIT-compiles the generic environment for this
problem's dimensions: `ighastar_generic_<state>_<14>_<11>_...`, cached under
`~/.cache/torch_extensions/py312_cu128/`. The build fetches robin_hood (a git
clone) if `ighastar/third_party/robin_hood` is missing, so it needs network
access once. It takes about a minute. Later runs print `ninja: no work to do.`

### 2.4 frax (IK for the TCP-delta sampler) and ssik (probing only)

```bash
git clone https://github.com/StanfordASL/frax.git ~/frax_src/frax   # tested at f5e87ab (0.2.0)
uv pip install --python ~/abc_src/abc/.venv/bin/python -e "$HOME/frax_src/frax[ik]"
# adds jax 0.12.0.1 (CPU), elastiqp, lxml; abc's pins are untouched
uv pip install --python ~/abc_src/abc/.venv/bin/python ssik==8.2.0   # optional: only for the old IK probes
```

frax runs on CPU JAX (`tcp_ik.py` sets `JAX_PLATFORMS=cpu`; the GPU stays with MJWarp).
There is no YAM URDF on disk; `tcp_ik.arm_urdf` generates one from abc's MJCF.

### 2.5 Smoke tests

```bash
PY=~/abc_src/abc/.venv/bin/python
$PY -c "import torch, mujoco, mujoco_warp, warp; print(torch.cuda.is_available(), mujoco.__version__)"
$PY -c "import abc_sim; e=abc_sim.make_env(task='put_relative', render_cameras=False); e.reset(seed=0, randomize=True); print(e.prompt)"
$PY tools/tcp_ik_check.py   # frax frames vs MuJoCo (1e-11), IK timing; run from examples/abc_example
```

Set `MUJOCO_GL=egl` on headless machines. The scripts set it themselves.

---

## 3. Reproduce what was done

All commands run from `examples/abc_example` with `PY=~/abc_src/abc/.venv/bin/python`.
The example's own README (`examples/abc_example/README.md`) has the full
formulation: state layout, controls, cost, admissible heuristic, goal, validity,
and output format.

### 3.1 Plan put_relative

```bash
cd ~/ros2_ws/src/IGHAStar/examples/abc_example
$PY plan_abc.py --task put_relative --seeds 13 19 3 10 --max-expansions 2500
# add --video for plan + CPU-replay mp4s of solved seeds
```

Outputs go to `Content/abc_example/` (`--out` to change): per-seed episode `.npz`,
videos, and `put_relative_summary.json`. `.npz` files are gitignored.

Hash resolution and tolerance (11 values: mover xyz, left TCP xyz, right TCP xyz,
left grip, right grip) can be set in three places. The first of these that's
set wins:

1. `--resolution ... --tolerance ...` on the CLI,
2. `planner:` in `config.yml`,
3. `task.put_relative` in `config.yml`.

The run prints the effective values. Preemptive expansion is **on**
(`planner.preemptive_expansion.enabled: true`; `plan_abc.py --save-tree` switches it off by itself). `active_only: true` keeps inactive (dominated) vertices out of the preemptive set (new C++ option, default false). Heuristics must
stay **admissible**.

### 3.2 Save and view a search tree

```bash
$PY plan_abc.py --task put_relative --seeds 13 --max-expansions 2500 --save-tree
$PY viser_abc_tree.py --tree ../../Content/abc_example/put_relative_seed13_tree.npz --port 8090
# open http://localhost:8090  (over ssh: ssh -L 8090:localhost:8090 <host>)
```

The viewer shows:
- TCP paths coloured by expansion order, with invalid edges in red,
- mover paths in orange and the goal region in green,
- a slider to scrub the robot through the expanded nodes,
- branch playback: solution, closest TCP to mover, lowest h, deepest, or most mover motion.

`--save-tree` needs preemptive expansion off.

### 3.3 Profile an expansion

```bash
NEXP=150 $PY tools/profile_expansion.py   # callback timing + MJWarp capacity sweep + per-stage timing
$PY tools/graph_fusion_check.py                 # 10x17-substep graphs vs one 170-substep graph; full dynamics() call
nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node -t cuda \
    -o step64 -f true $PY tools/nsys_step.py
nsys stats -r cuda_gpu_kern_sum step64.nsys-rep
```

---

## 4. Where things stand

### Setup choices
- **Task:** `put_relative`, the shortest-horizon static abc task. The mean demo
  is 14.9 s. `conveyor_pick` is shorter but time-varying.
- **Easiest seeds:** 13, 19, 3 and 10, with the mover 6–7 cm from the goal cone.
  Seed 28 already starts in the goal. The median gap across seeds is 0.29 m.
- **Timing:** one control period is 34 ms (2 ms physics × 17 substeps, ~30 Hz).
  One edge is 10 periods = 0.34 s = 170 physics substeps. Controls are 64
  uniform joint-space deltas on the active arm, with the zero action first.

### Results
- **0 seeds solved so far:**
  - seeds 13/19/3/10: 2500 expansions each, about 245 s each;
  - seed 0: 3000 expansions, with preemption on at the time.
- **Admissibility:** the per-edge mover step never exceeded the bound.
- **Contact buffers:** no overflows (`nconmax=128`, `njmax=1200`, both per world).
- **Seed-13 tree:** depth reaches only 5 edges, and the search fans out
  breadth-first over the whole arm workspace. h is about 0 near the start,
  because the admissible bound (max arm-point step 0.8 m per edge) is about 5×
  loose, and the reference slack cancels small gaps. The closest TCP-to-mover
  approach was 11–22 cm.
- **TCP motion per edge (joint-space sampling):** median 4.0 cm, p90 7.4 cm,
  p99 11.8 cm, max 23.7 cm.

### Speed (about 10 expansions/s)
- **Planner overhead is negligible:** 99% of time is `dynamics_fn`, which takes
  about 93 ms per expansion.
  - About 72 ms of that is pure GPU stepping.
  - About 20 ms is torch glue: ctrl ramp, per-period recording, overflow check,
    pack, `.cpu()`.
- **No rendering and no per-substep CPU↔GPU traffic.** Each control period is
  one CUDA graph launch. One 170-substep graph is no faster than ten 17-substep
  graphs.
- **The step is latency-bound, not compute-bound:** about 135 small kernels per
  substep, about 3 µs each, and 64 worlds don't fill a 4090.
  - Throughput per launch: 256 worlds cost the same as 64, and 1024 cost about
    3× for 16× the rollouts.
  - Per control period at 64 worlds: 7.5–11 ms, depending on the contact state.
  - The solver averages 1.2 iterations. Collision narrowphase is about 4% of
    GPU time.

### TCP-delta sampling with frax IK (done; `--sampler tcp --tcp-site grasp`)
- Controls are grasp-site motions (+ gripper delta); `tcp_ik.py` turns them into joint
  targets with `frax.ik` inside `dynamics()`. No per-edge joint cap, only abc's joint
  limits. Default step 15 cm/edge, from `tools/tcp_reach.py` (the PD arm saturates around
  13-17 cm per 0.34 s edge).
- The admissible bound in tcp mode comes from the TCP step (0.37 m/edge vs 0.11-0.16 m
  largest mover step seen).
- Results at 1000 expansions: **seeds 13 and 19 both solved**, abc CPU replay 2/2
  (joint sampling: 0/4 at 2500). About 147 s per 1000 expansions; IK adds about 35 ms of
  the ~147 ms per expansion. Joint-space sampling with ssik-style caps (4 cm, 0.2-0.3 rad)
  was worse than random joint sampling.
- Near abc's home pose even 4 cm TCP moves need median max-joint moves of 0.5-0.8 rad
  (singular home, condition number 342); frax stays on-branch, so that is real.
- Grasp-targeted controls (`--grasp-fraction`, default 0.5; see the README): 4 seeds, 1000 expansions,
  preemption off. With 32 grasp rows: seeds 13 and 19 solved with 19-24% cheaper plans and a closer
  TCP-to-mover approach on every seed; seed 10 solved only without grasps; seed 3 unsolved both ways.
- IK failures (no pose within 1 mm / 0.01 rad; single solve, no retries) just make that control invalid.
- Earlier ssik findings (exact FK match to 6.5 um, abc's `tcp_site` is the flange,
  `grasp_site` is +0.1347 m along link_6) still hold.

### Physics step size vs abc's demos (`tools/dt_sweep_demos.py`)
Replays the recorded actions of abc's `put_relative` sim demos (download: `prepare.py --sim-data put_relative`)
open-loop in abc's CPU env with the same 34 ms control period but physics step 34 ms / n, and compares
abc's evaluator and the mover's final position against the demo (300 random episodes, 40 s on 28 cores).
The 2 ms replay reproduces the demos (285/300 succeed vs 290/300 for the demos themselves). Coarser steps:
3.4 ms 276, 4.25 ms 278, 6.8 ms 262, 8.5 ms 267, 11.3 ms 256, 17 ms 248, 34 ms 161 of 300 succeed;
mover final position within 1 cm of the demo: 94 % (2 ms), 85, 83, 77, 76, 73, 67, 19 %.
Only steps that divide 34 ms are possible without changing the control period.

### Open decisions (not yet chosen)
1. ~~TCP-space sampling~~: done, see above.
2. **A tighter admissible heuristic,** for example
   max(object term, arm-to-object distance / v_max), or simply a bigger budget
   (about 25k expansions).
3. **Throughput** (CPU MuJoCo is competitive: `tools/cpu_rollout_bench.py` gets 11.9 / 13.5
   expansions/s at 16 / 32 threads vs ~10 on the GPU; `tools/speed_sweep.py` shows
   solver/integrator settings do not matter, only substep count does):
   - more controls per node (256 costs about the same as 64);
   - batch several seeds' searches into one launch;
   - ~~re-enable preemptive expansion~~ (now on);
   - trim the 20 ms of torch glue;
   - a coarser physics step, such as 6.8 ms × 5 substeps (3.4× fewer substeps).
     This must be checked against abc's 2 ms replay. abc's 2 ms is MuJoCo's
     default rather than a stated requirement; the arm PD gains are soft
     (ω·dt ≤ 0.12 at 2 ms).
4. **Kernel fusion (a megakernel per world)** would need a rewrite of MJWarp's
   step. It's possible in principle, but a big project.

### Not carried over
These were scratch files on the original machine and weren't kept:
- the seed-13 tree `.npz` (regenerate with §3.2),
- the ssik probe scripts,
- the seed-0 scene image.
