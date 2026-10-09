# Generic environment (Python callbacks)

Use the **generic** environment when you define dynamics, cost, validity, heuristic, and goal test in Python. You do not write a C++ `Environment` class.

The planner builds a Torch extension with your state and control sizes. Your callbacks run on batched `torch.Tensor` values.

For a C++ environment instead, see [README.md](README.md).

## Glossary

| Term | Meaning |
|---|---|
| `generic` | Environment type set by `node_info.node_type: "generic"` |
| callback | Python function the planner calls during search |
| edge | One control held from one state to the next state |
| `g` | Path cost from the start to a node |

## When to use generic

Use generic when:

- You own the dynamics model in Python (or JAX, NumPy, MuJoCo, and similar).
- You want to change cost or validity without a C++ rebuild of those rules.
- Your world model lives in Python closures (not a map tensor).

Use a built-in C++ environment (`kinematic`, `kinodynamic`, `simple`, `holonomic_6D`) when you need the packaged car or map pipelines.

## Required callbacks

Attach these six callables on the config dict. YAML cannot store them. Attach them in Python after you load the YAML.

| Key | Signature | Role |
|---|---|---|
| `sample_controls_fn` | `() -> Tensor[K, n_cont]` | Returns the control set for one expansion |
| `dynamics_fn` | `(states[B, n], controls[B, n_cont]) -> [B, n]` or `[B, T, n]` | Rolls each state with each control |
| `cost_fn` | `(states, controls, next_states) -> [B]` | Edge cost for the **final** state of each rollout |
| `validity_fn` | `(states[B, n]) -> [B]` | Nonzero means valid |
| `heuristic_fn` | `(states[B, n]) -> [B]` | Cost-to-go. Encode goal knowledge here |
| `goal_test_fn` | `(state[n]) -> bool` | True when the state is in the goal region |

### Dynamics shapes

- Return `[B, n]` for a single step.
- Return `[B, T, n]` for a multi-step rollout. The node state is the last step. The planner stores intermediate steps on the path.
- `cost_fn`, `validity_fn`, and `heuristic_fn` always see the **final** state.

### Goal and world tensors

For generic search, `goal_test_fn` and `heuristic_fn` own the goal. The planner does not use the `goal` tensor for that test. The planner does not use the `world` tensor. Capture maps or simulators inside your callbacks.

## Config keys

Set these under `experiment_info_default` (see `examples/standalone/Configs/generic_integrator.yml`):

| Key | Role |
|---|---|
| `state_dim` | State size (`n`). Compiled as `-DN_DIMS` |
| `control_dim` | Control size (`n_cont`). Compiled as `-DN_CONT` |
| `hash_dims` | Leading dims used in the grid hash (default: all state dims) |
| `num_controls` | Expected branching factor `K` |
| `resolution` | Start grid size per dim (length `state_dim`) |
| `tolerance` | Dedup tolerance per dim |
| `bounds_lower` / `bounds_upper` | State bounds (length `state_dim`) |
| `max_level` | Number of resolution levels |
| `division_factor` | Resolution shrink factor per level |
| `max_expansions` | Search budget |
| `hysteresis` | Resolution-switch threshold |
| `node_info.node_type` | Must be `"generic"` |
| `node_info.timesteps` | Hint only. Multi-step `T` comes from `dynamics_fn` output |

`create_planner` rebuilds the extension when `(state_dim, control_dim, hash_dims)` change. Keep those three stable across runs when you can.

## How to plan

1. Load a YAML config with `node_type: "generic"`.
2. Attach the six callbacks on the config dict.
3. Call `create_planner(configs, bidirectional=False)` for IGHA\*, or `bidirectional=True` for BiIGHA\*.
4. Call `planner.search(start, goal, world, max_expansions, hysteresis, ignore_goal_validity)`.
5. If search returns true, read the path with `get_best_path()` or `get_best_path_with_controls()`.

Minimal example:

```python
import torch
import yaml
from ighastar.scripts.common_utils import create_planner

with open("examples/standalone/Configs/generic_integrator.yml") as f:
    configs = yaml.safe_load(f)

configs["sample_controls_fn"] = sample_controls
configs["dynamics_fn"] = dynamics
configs["cost_fn"] = cost
configs["validity_fn"] = validity
configs["heuristic_fn"] = heuristic
configs["goal_test_fn"] = goal_test

planner = create_planner(configs, bidirectional=False)
info = configs["experiment_info_default"]
start = torch.tensor([1.0, 1.0], dtype=torch.float32)
goal = torch.zeros(info["state_dim"], dtype=torch.float32)  # unused by goal_test
world = torch.zeros(1, dtype=torch.float32)                 # unused by generic

ok = planner.search(
    start, goal, world, info["max_expansions"], info["hysteresis"], True
)
if ok:
    path = planner.get_best_path()  # [path_len, state_dim + 1], last column is g
```

Run the full point-mass demo:

```bash
python3 examples/standalone/generic_integrator.py
```

## Path output

- `get_best_path()` returns `[path_len, state_dim + 1]`. The last column is `g`.
- Rows are **goal-first**, start-last. Solution cost is `path[0, -1]`.
- `get_best_path_with_controls()` also returns the control sequence along that path.

## Batching rules

The planner expands several nodes at once. For `M` nodes and `K` controls it builds `B = M * K` rows:

1. Call `sample_controls_fn()` once. Expect shape `[K, n_cont]`.
2. Call `dynamics_fn(states[B, n], controls[B, n_cont])`.
3. Call `cost_fn`, `validity_fn`, and `heuristic_fn` on the final states.

Write every callback to accept a batch on dim 0. Do not assume `B == 1`.

## JAX and other backends

Callbacks must accept and return `torch.Tensor` (CPU float32 is enough). Convert as needed:

1. `tensor.detach().cpu().numpy()`
2. Run your model (JAX, NumPy, MuJoCo, …).
3. `torch.from_numpy(...).float()`

For a multi-step edge with per-step rewards, return `[B, T, n]` from `dynamics_fn`. Put the scalar edge cost in `cost_fn` for the final state. You may store intermediate costs in a closure. See a heavier pattern in an external Sumo Spot runner if you use that stack.

## Bidirectional search

Pass `bidirectional=True` to `create_planner`. BiIGHA\* uses the same six callbacks. Near-meet distance for generic is Euclidean on the full state. Tune `tolerance` and `resolution` if meet detection is too strict or too loose.

## Working examples

| File | What it shows |
|---|---|
| `examples/standalone/generic_integrator.py` | Minimal 2D integrator |
| `examples/standalone/Configs/generic_integrator.yml` | Generic YAML keys |
| `examples/standalone/generic_ur5e_mjx.py` | Skeleton for a MuJoCo MJX arm |
| `ighastar/src/Environments/include/generic.h` | Callback contract in the C++ header |

## Related

- C++ `Environment` interface: [README.md](README.md)
- Standalone usage notes: [examples/standalone/README.md](../../../../examples/standalone/README.md)
