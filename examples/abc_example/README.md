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

## Viewing a plan (no tree needed)

Every solved seed's episode `.npz` is enough, so this works with preemptive expansion on:

```bash
python viser_abc_episode.py --episode ../../Content/abc_example/put_relative_seed13.npz --port 8090
```

It plays the plan's qpos sequence or abc's CPU replay (dropdown), draws the grasp-site
and mover paths and the goal region, and shows abc's own evaluator verdict (along / perp
/ placed / success) for the frame on screen.

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

**TCP-delta sampler** (`problem.sampler: tcp`, or `--sampler tcp`): the planner's control
is still 14-D, but the active arm's 7 slots hold `[dx dy dz rx ry rz grip]`, a rigid
motion of the grasp site (world frame; metres / axis-angle radians) and a gripper
delta. Inside `dynamics()` (`tcp_ik.py`) each node's commanded joints seed
[frax](https://github.com/StanfordASL/frax) differential IK (`frax.ik`, joint-limit
constrained, run to convergence on the moved pose, batched on CPU JAX). Being local it
stays on the current joint branch. The joints are just carried physics state: there is
**no per-edge joint cap** (optional `tcp_joint_cap`, off by default). A row is solved
when the converged pose is within 1 mm / 0.01 rad of the target; otherwise the control
is simply invalid (no retries; sampling many controls anticipates failures). The realised joint delta ends up in the path's `ctrl`, so replay is unchanged
(`warp_replay` uses `dynamics_joint`). The URDF frax needs is generated from abc's MJCF
(`tcp_ik.arm_urdf`). Sampling (our choice, not an abc constraint): row 0 is zero, rows
1-6 are +-x/y/z full steps (`tcp_step`, default 15 cm, about what the PD arm achieves in
one 0.34 s edge: `tools/tcp_reach.py`), the rest are random directions at 25-100 %
magnitude; `tcp_rot_prob` of them also rotate (<= `tcp_rot_step`); the gripper holds 70 %
of the time. `tcp_site: grasp` hashes the point between the fingertips instead of abc's
`tcp_site` (the wrist flange). `tools/tcp_ik_check.py` checks the frames against MuJoCo
and times the IK. In tcp mode the admissible bound below comes from the TCP step
(`tcp_step + tcp_rot_step * hand radius + finger travel`, times `speed_bound_margin`).

**Grasp-targeted controls** (tcp sampler, object-centric tasks; `grasp_fraction`, default 0.5):
the last half of the K controls reach toward a sampled antipodal grasp of the task's object
(put_relative: the mover) instead of a random TCP delta, and IK decides which are feasible.
`grasp_gen.py` samples up to `grasp_pool` (256) parallel-jaw grasps on the object's collision
geometry once per problem, stored as grasp-site poses in the object frame (opening 1-8.5 cm,
30 degree friction cone, approach not from below; gripper geometry measured from abc's MJCF).
The control carries only a candidate id (column 14, so tcp-mode `control_dim` is 15); the world
target is resolved in `dynamics()` from each node's own object pose. The hand moves toward it
by up to `tcp_step` / `grasp_rot_step` per edge with the gripper open; within tolerance the next
edge closes the gripper. If the object already looks grasped (gripper mostly closed, site within
6 cm) the row keeps its random delta. `--grasp-fraction 0` gives random TCP deltas only.
`tools/grasp_check.py` reaches, closes and lifts for sampled candidates as a physical check.

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

**Goal** (put_relative): exactly abc's `PutRelativeEvaluator` criterion on the live
state (cone around the reference + placed height), nothing added.

**Validity**: finite state, bounded qvel, objects on the table, TCPs above the
table, and non-mover objects within `disturb_tol` of their start.

## Episode `.npz`

`state [T+1, 14]` (measured, policy space), `action [T, 14]` (commanded targets),
`qpos`, `qvel`, `cpu_replay_qpos`, `g`, `prompt`, `metadata` (abc randomization),
`record` (search stats and replay checks).

## MPC polish of coarse plans (UNREVIEWED, experimental)

Written by Claude and **not yet reviewed by a human**. Planning stays at the default 2 ms
physics step; `--plan-substeps N` (coarser step) is optional and the polish below is the
experiment that goes with it.

`segment_mppi.py` (generic segment-wise MPPI), `abc_refine.py` (abc adapter: CPU MuJoCo at
2 ms, knot cost) and `tools/polish_plan.py` re-optimise each 34 ms control period of a
coarse plan under the fine model, targeting the coarse plan's knot states, then replay in
abc's evaluator. Early results: 4.25 ms plans 5/7 -> 7/7 success after polish; 6.8 ms plans
0/7 -> 4-5/7. Not validated beyond a handful of seeds.
