# Generic environment

The **generic environment** lets you define a kinodynamic planning problem entirely in Python—no new `.cu` or `.h` files. It follows an OMPL-style callback pattern: you provide dynamics, cost, validity, heuristic, goal test, and a control sampling function; IGHA* handles the search.

**Reference implementation:** [examples/standalone/generic_integrator.py](../examples/standalone/generic_integrator.py)  
**Config template:** [examples/standalone/Configs/generic_integrator.yml](../examples/standalone/Configs/generic_integrator.yml)

## When to use which integration path

| Path | Best for |
|------|----------|
| **Generic env** (this doc) | Custom dynamics in Python; manipulation scaffolds; latent-space planning |
| Built-in `kinematic` / `kinodynamic` | GPU costmap car on 2D maps (fastest path to off-road / street demos) |
| Hand-written C++ env | Maximum performance with a fixed representation—see [Environments README](../ighastar/src/Environments/README.md) |

## Setup

1. Copy [generic_integrator.yml](../examples/standalone/Configs/generic_integrator.yml) and set `state_dim`, `control_dim`, `hash_dims`, resolutions, and bounds.
2. Implement the six callbacks below in Python.
3. Attach them to the config dict **before** `create_planner()`:

```python
configs["sample_controls_fn"] = sample_controls
configs["dynamics_fn"] = dynamics
configs["cost_fn"] = cost
configs["validity_fn"] = validity
configs["heuristic_fn"] = heuristic
configs["goal_test_fn"] = goal_test

planner = create_planner(configs, bidirectional=False)
```

4. Call `search()` with `ignore_goal_validity=True` if your goal is defined only in `goal_test_fn`:

```python
success = planner.search(start, goal, world, expansion_limit, hysteresis, True)
```

The `world` tensor is unused by the generic env; world state should be captured in your callback closures (e.g. an MJX model handle).

## Callback contract

All tensor callbacks receive and return **CPU float32** tensors unless you convert internally. Batched calls use leading dimension `B`.

| Config key | Signature | Description |
|------------|-----------|-------------|
| `sample_controls_fn` | `() -> Tensor [K, n_cont]` | Control set for one expansion (fixed or sampled each call) |
| `dynamics_fn` | `(states [B, n], controls [B, n_cont]) -> [B, n]` | One propagation step |
| `cost_fn` | `(states, controls, next_states) -> [B]` | Non-negative edge cost |
| `validity_fn` | `(states [B, n]) -> [B]` | Nonzero = valid, zero = invalid |
| `heuristic_fn` | `(states [B, n]) -> [B]` | Cost-to-go estimate (encodes goal knowledge; goal arg ignored) |
| `goal_test_fn` | `(state [n]) -> bool` | Terminal test for a single state |

### Example: 2D point integrator

```python
def sample_controls() -> torch.Tensor:
    return _CONTROLS  # [8, 2] fixed directions

def dynamics(states, controls):
    return states + controls * DT

def cost(states, controls, next_states):
    return torch.norm(next_states - states, dim=1)

def validity(states):
    in_bounds = (states >= LOWER).all(dim=1) & (states <= UPPER).all(dim=1)
    clear = torch.norm(states - OBSTACLE, dim=1) > OBSTACLE_RADIUS
    return (in_bounds & clear).float()

def heuristic(states):
    return torch.norm(states - GOAL, dim=1)

def goal_test(state):
    return bool(torch.norm(state - GOAL).item() < GOAL_RADIUS)
```

## YAML parameters (generic-specific)

See [generic_integrator.yml](../examples/standalone/Configs/generic_integrator.yml):

- `state_dim`, `control_dim` — compiled into the extension (`-DN_DIMS`, `-DN_CONT`)
- `hash_dims` — dimensions used for grid hashing (`-DHASH_DIMS`); can be **less than** `state_dim`
- `num_controls` — branching factor K
- `resolution`, `tolerance` — per-dimension lists
- `bounds_lower`, `bounds_upper` — state limits
- `max_level`, `division_factor`, `max_expansions`, `hysteresis` — search settings

## `hash_dims < state_dim` (latent / high-D planning)

Plan and deduplicate in a **low-dimensional subspace** while running dynamics in the full state:

- `state_dim=64`, `hash_dims=8` → IGHA* grids only the first 8 (whitened PCA) dimensions
- `heuristic_fn` and `goal_test_fn` operate in that subspace
- `dynamics_fn` still integrates the full 64-D state

Advanced example: [generic_latent_planning_smoke.py](../examples/standalone/generic_latent_planning_smoke.py) (placeholder world model; swap for your encoder + learned dynamics).

## Compile-time dimensions

Changing `state_dim`, `control_dim`, or `hash_dims` triggers a **new** JIT build. Extension names include the dimension tuple (see [common_utils.py](../ighastar/scripts/common_utils.py)), so builds do not collide in the PyTorch extension cache.

## Performance notes

- Callbacks execute on **CPU** (Python / PyTorch). Keep them vectorized over batch dimension `B`.
- Enable `preemptive_expansion` only after callbacks handle large batches efficiently.
- For sim-backed validity (MuJoCo, MJX), JIT + `vmap` your FK/collision checks—see the skeleton below.

## UR5e manipulation skeleton

[generic_ur5e_mjx.py](../examples/standalone/generic_ur5e_mjx.py) is a **runnable scaffold**, not a finished tutorial. It shows batched FK/collision via MuJoCo/MJX for a 6-DoF arm. Contributions to complete this example are welcome.

## Bidirectional search

BiIGHA* is supported for generic env. You may need to tune `LCR`, `backward_epsilon`, and `goal_sampling_config`—see [configuration.md](configuration.md).

## Further reading

- [api.md](api.md) — `search`, `get_best_path`, profiler
- [extending.md](extending.md) — decision tree for integration paths
- [generic.h](../ighastar/src/Environments/include/generic.h) — C++ implementation details
