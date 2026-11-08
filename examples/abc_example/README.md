# IGHA* on abc_sim tasks (dataset generation)

Plans tasks from [amazon-far/abc](https://github.com/amazon-far/abc) with IGHA*,
treating IGHA* as a dependency: the search code is untouched. The task plugs in
through the **generic** (Python-callback) environment described in
[`ighastar/src/Environments/README.md`](../../ighastar/src/Environments/README.md)
/ [`GENERIC.md`](../../ighastar/src/Environments/GENERIC.md). Rollouts run as
batched MJWarp physics on the GPU, the same engine abc uses for its batched
evaluation.

| File | What |
| --- | --- |
| `warp_rollout.py` | `WarpBatchRollout`: loads B full sim states into MJWarp worlds, ramps actuator targets over N 30 Hz control periods (each one a captured CUDA graph of 17 physics substeps), records every period |
| `abc_problem.py` | `AbcGenericProblem` (state layout, dynamics, generic validity, planner config) and `PutRelativeProblem` (features, admissible heuristic, goal = abc's evaluator) |
| `plan_abc.py` | resets an abc task by seed, searches, replays the plan in MJWarp and in abc's CPU MuJoCo env (scored by abc's evaluator), writes the episode `.npz` and videos |
| `config.yml` | planner, problem and task settings |

## Setup

abc wants Python 3.12 and its own pinned torch/mujoco/mujoco-warp, so use abc's venv
and install IGHA* into it in editable mode:

```bash
git clone https://github.com/amazon-far/abc.git && cd abc
uv python pin 3.12 && uv sync
uv pip install -e /path/to/IGHAStar
uv run prepare.py --sim-task put_relative     # scene assets (~137 MB)
```

## Run

```bash
cd examples/abc_example
/path/to/abc/.venv/bin/python plan_abc.py --task put_relative --seeds 0 1 2 --video
```

Outputs go to `Content/abc_example/` (`--out` to change): `put_relative_seed<k>.npz`,
`*_plan.mp4`, `*_cpu_replay.mp4`, `put_relative_summary.json`.

## Search-tree viewer

`export_search_tree()` would return every node's full 135-D pose and rollout as
Python lists (several GB for a few thousand expansions), so `--save-tree` logs
the tree compactly from inside the callbacks instead. It needs preemptive
expansion off, so that each `dynamics` call is one expansion. Per expansion, it
records the parent's features and qpos. Per child edge, it records the 10-step
feature rollout (TCP, mover, grippers), final qpos, validity, cost and h.

```bash
python plan_abc.py --task put_relative --seeds 13 --save-tree
python viser_abc_tree.py --tree ../../Content/abc_example/put_relative_seed13_tree.npz --port 8090
```

The viewer shows the abc scene (mjviser) and every simulated TCP path, coloured
by expansion order, with invalid edges in red. Mover paths are orange and the
goal region is green. You can scrub the robot through the expanded nodes and play
back a branch: the solution, the node closest to the mover, the lowest-h node, or
the deepest node.

## Formulation

**State** `[features (F, hashed) | ctrl (14) | qpos (nq) | qvel (nv)]`. Features are
the leading `hash_dims`, and the full MuJoCo state follows, so any node can be
re-simulated from its vector. `ctrl` is the current position target in abc's
14-D policy space `[left j1..6, left grip, right j1..6, right grip]`.

**Control**: a 14-D delta on `ctrl`, sampled uniformly in
`±[joint_delta, gripper_delta]` for the active arm(s) (`active_arms: auto` picks
the arm on the mover's side). The first sample is the zero (hold) action. One
edge ramps the target linearly over `ctrl_steps_per_edge` 30 Hz periods. Each
period is stored as a path state, so the dense path is a 30 Hz
(state, action) episode in abc's action convention.

**Cost**: edge duration + `joint_motion_weight * |Δjoints|`, so every edge costs at
least `edge_dt`.

**Heuristic (admissible)**: `goal_distance / max_object_step * edge_dt`.
`goal_distance` is the exact distance from the mover to abc's goal region (the
directional cone around the reference, capped at the placed height), minus the
most the reference could still move (it must stay within `disturb_tol` of where it
started). `max_object_step` is an upper bound on how far any point of an active
arm can travel in one edge, `Σ_i joint_delta_i · R_i + finger travel`, where
`R_i` is the anchor-to-anchor link length below joint *i* plus the hand radius.
That sum is configuration independent, then multiplied by `speed_bound_margin`
to cover PD overshoot. The run log prints the largest per-edge mover
displacement seen during search next to the bound, as an empirical check.

**Goal** (put_relative): abc's `PutRelativeEvaluator` criterion on the live state,
plus the mover at rest (`goal_max_speed`).

**Validity**: finite state, bounded qvel, objects on the table, TCPs above the
table, and non-mover objects within `disturb_tol` of their start.

## Episode `.npz`

`state [T+1, 14]` (measured, policy space), `action [T, 14]` (commanded targets),
`qpos`, `qvel`, `cpu_replay_qpos`, `g`, `prompt`, `metadata` (abc randomization),
`record` (search stats and replay checks).
