# Configuration reference

Planner parameters live under `experiment_info_default` in YAML config files. Map-specific settings (start, goal, map path) are under `map` for standalone examples.

Example configs:

- [kinodynamic_example.yml](../examples/standalone/Configs/kinodynamic_example.yml)
- [kinematic_example.yml](../examples/standalone/Configs/kinematic_example.yml)
- [simple_example.yml](../examples/standalone/Configs/simple_example.yml)
- [generic_integrator.yml](../examples/standalone/Configs/generic_integrator.yml)

## Core search parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `resolution` | float or list | Starting grid resolution for state discretization. Scalar for car envs; per-dimension list for generic env. |
| `tolerance` | float or list | Minimum separation between vertices (map/perception resolution should be ≤ this). |
| `max_level` | int | Maximum resolution refinement level (used to size hash tables). |
| `division_factor` | float | Factor by which resolution tightens each level (typically `2`). |
| `max_expansions` | int | Search budget passed to `search()`. |
| `hysteresis` | int | Threshold controlling when IGHA* switches resolution levels (IGHA*-H). |
| `epsilon` | list | Goal region tolerances `[ate, cte, heading, vel]` for car envs (along-track, cross-track, heading, velocity). |

## BiIGHA* parameters

Used when `create_planner(..., bidirectional=True)` or `--bidirectional` flag.

| Parameter | Type | Description |
|-----------|------|-------------|
| `backward_epsilon` | list | Goal region for backward search (same layout as `epsilon`). |
| `LCR` | list | Local controllability radius per dimension—region where forward and backward trees can meet. |
| `near_meet_config.enabled` | bool | Enable near-meet connection strategy. |
| `near_meet_config.num_interpolation_points` | int | Interpolation points for validity along a meet edge. |
| `near_meet_config.num_perturbations` | int | Perturbation attempts when connecting trees (0–3). |
| `near_meet_config.perturbation_scale` | float | Scale for meet perturbations. |
| `goal_sampling_config.enabled` | bool | Sample states near goal for backward search (needed when exact goal is unreachable). |
| `goal_sampling_config.space_type` | str | `"SE2"` (rotation manifold) or `"R_N"` (Euclidean). |
| `goal_sampling_config.angular_dims` | list | State indices that are angular (e.g. `[2]` for heading). |
| `goal_sampling_config.num_samples` | int | Number of goal-region samples. |
| `goal_sampling_config.sigma` | list | Sampling std per dimension (defaults to `LCR` if omitted). |

## Performance: preemptive expansion

Batches multiple vertex expansions per GPU kernel launch.

| Parameter | Type | Description |
|-----------|------|-------------|
| `preemptive_expansion.enabled` | bool | Enable batched preemptive expansion. |
| `preemptive_expansion.min_preemptive` | int | Minimum stash size before batching kicks in. |
| `preemptive_expansion.max_preemptive` | int | Max vertices expanded per batched launch. |

## `node_info` — built-in car environments

| Parameter | Description |
|-----------|-------------|
| `node_type` | `"kinematic"`, `"kinodynamic"`, or `"simple"` |
| `length`, `width` | Vehicle footprint (m) |
| `map_res` | Meters per map pixel |
| `dt` | Integration timestep |
| `timesteps` | Intermediate states stored per edge |
| `step_size` | Spatial step per primitive |
| `steering_list` | Discrete steering angles (degrees) |
| `throttle_list` | Discrete throttle values (negative = reverse) |
| `del_theta`, `max_theta` | Steering rate and limit (degrees) |
| `del_vel`, `min_vel`, `max_vel` | Velocity change and limits (kinodynamic) |
| `RI` | Rolling resistance |
| `max_vert_acc` | Max vertical acceleration (terrain limit) |
| `gear_switch_time` | Time penalty multiplier for reverse segments |

## `node_info` — generic environment

| Parameter | Description |
|-----------|-------------|
| `node_type` | `"generic"` |
| `timesteps` | Intermediate pose storage (typically `1`) |
| `state_dim` | State dimension (`-DN_DIMS` at compile time) |
| `control_dim` | Control dimension (`-DN_CONT`) |
| `hash_dims` | Leading dims used for grid hash (`-DHASH_DIMS`; can be `< state_dim`) |
| `num_controls` | Branching factor K (controls per expansion) |
| `bounds_lower`, `bounds_upper` | State-space bounds (lists, length `state_dim`) |

Python callbacks are attached to the config dict in code, not YAML—see [generic_environment.md](generic_environment.md).

## Map section (standalone examples)

| Parameter | Description |
|-----------|-------------|
| `map.name` | Map filename |
| `map.dir` | Directory under `examples/standalone/` |
| `map.size` | `[width, height]` in pixels |
| `map.res` | Meters per pixel |
| `map.start`, `map.goal` | Default poses |
| `map.test_cases` | Named alternate start/goal pairs |

## Tuning guide

1. **Start with a shipped YAML** for your environment type; only change start/goal first.
2. **Search fails or times out** → try `--bidirectional` (often needs far fewer expansions).
3. **Still no solution** → enable `goal_sampling_config` for BiIGHA*; widen `epsilon` slightly.
4. **Too slow on GPU** → enable `preemptive_expansion`; reduce `max_expansions` for anytime behavior.
5. **Generic env slow** → vectorize callbacks; keep `preemptive_expansion` off until callbacks are batch-friendly.

Bidirectional-specific parameter details also appear in [examples/BeamNG/README.md](../examples/BeamNG/README.md#bidirectional-search-parameters).
