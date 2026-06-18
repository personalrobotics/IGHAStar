# Planner API

The supported entry point is `create_planner()` in [ighastar/scripts/common_utils.py](../ighastar/scripts/common_utils.py). It loads the C++/CUDA extension and returns either an `IGHAStar` or `BiIGHAStar` instance.

## Minimal example

```python
import torch
import yaml
from ighastar.scripts.common_utils import create_planner

with open("examples/standalone/Configs/kinodynamic_example.yml") as f:
    configs = yaml.safe_load(f)

# For generic env: attach callbacks before create_planner (see generic_environment.md)
# configs["dynamics_fn"] = ...
# configs["sample_controls_fn"] = ...

planner = create_planner(configs, bidirectional=False)

start = torch.tensor([5.6, 8.9, 1.58, 4.94, 0.0], dtype=torch.float32)
goal = torch.tensor([21.1, 46.1, 0.61, 4.52, 0.0], dtype=torch.float32)
world = ...  # costmap tensor from your map loader (see examples/standalone/utils.py)

expansion_limit = configs["experiment_info_default"]["max_expansions"]
hysteresis = configs["experiment_info_default"]["hysteresis"]

success = planner.search(
    start, goal, world,
    expansion_limit,
    hysteresis,
    ignore_goal_validity=False,  # set True for generic env when goal is tested in callbacks
)

if success:
    path = planner.get_best_path()  # torch.Tensor [N, n_dims + 1]
    path_np = path.numpy()
```

## `create_planner(configs, bidirectional=False)`

| Argument | Type | Description |
|----------|------|-------------|
| `configs` | `dict` | Full YAML config; must contain `experiment_info_default` with `node_info.node_type` |
| `bidirectional` | `bool` | If `True`, returns `BiIGHAStar`; otherwise `IGHAStar` |

`node_type` values: `"kinematic"`, `"kinodynamic"`, `"simple"`, `"generic"`.

For `"generic"`, attach Python callbacks to `configs` before calling `create_planner` (see [generic_environment.md](generic_environment.md)).

## `search(start, goal, world, expansion_limit, hysteresis, ignore_goal_validity=False)`

| Argument | Type | Description |
|----------|------|-------------|
| `start` | `torch.Tensor` | Start state, shape `[n_dims]` |
| `goal` | `torch.Tensor` | Goal state, shape `[n_dims]` (may be unused by generic `goal_test_fn`) |
| `world` | `torch.Tensor` | Map/world representation (costmap for car envs; placeholder for generic) |
| `expansion_limit` | `int` | Maximum node expansions |
| `hysteresis` | `int` | Resolution-switch hysteresis threshold |
| `ignore_goal_validity` | `bool` | Skip goal validity check (useful for generic env) |

Returns `bool`: whether a solution was found.

## Path and control output

### `get_best_path() -> torch.Tensor`

Returns the best path as `[N, n_dims + 1]`. The last column is `g * time_direction` (cost-from-start, signed for bidirectional segments: positive = forward search, negative = backward).

Path order is **goal-first to start-last**.

### `get_best_path_with_controls() -> (torch.Tensor, torch.Tensor)`

Returns `(path, controls)` where `controls` has shape `[N-1, n_cont]` (or environment-specific control dims). Available for built-in car environments and generic env.

## Profiler

### `get_profiler_info() -> tuple`

Returns a 10-tuple:

| Index | Name | Description |
|-------|------|-------------|
| 0 | `avg_successor_time` | Mean successor generation time (µs) |
| 1 | `avg_goal_check_time` | Mean goal-check time (µs) |
| 2 | `avg_overhead_time` | Mean overhead per resolution switch (µs) |
| 3 | `avg_g_update_time` | Mean cost-update time (µs) |
| 4 | `switches` | Number of resolution-level switches |
| 5 | `max_level_profile` | Deepest resolution level reached |
| 6 | `Q_v_size` | Final active-queue size (BiIGHA*: combined forward+backward) |
| 7 | `expansion_counter` | Total expansions |
| 8 | `expansion_list` | Per-expansion profile data |
| 9 | `cost_exp_list` | Best cost after each expansion (`< 1e4` indicates a solution) |

Use `cost_exp_list` to plot anytime cost-vs-expansion curves.

### `get_preemptive_expansions() -> int`

Number of vertices expanded preemptively (batched) during the last search.

## BiIGHAStar-only methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_total_expansions()` | `int` | Combined forward + backward expansions |
| `get_best_cost()` | `float` | Cost of the best solution found |

## Method availability

| Method | IGHAStar | BiIGHAStar |
|--------|----------|------------|
| `search(...)` | yes | yes |
| `get_best_path()` | yes | yes |
| `get_best_path_with_controls()` | yes | yes |
| `get_profiler_info()` | yes | yes |
| `get_preemptive_expansions()` | yes | yes |
| `get_total_expansions()` | — | yes |
| `get_best_cost()` | — | yes |

## Notes

- Do not call `torch.utils.cpp_extension.load` directly; use `create_planner()`.
- Changing `state_dim` / `control_dim` for generic env triggers a separate extension build (cached by dimension tuple).
- See [example.py](../examples/standalone/example.py) and [generic_integrator.py](../examples/standalone/generic_integrator.py) for complete working scripts.
