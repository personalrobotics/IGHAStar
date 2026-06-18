# Examples catalog

Examples are tiered by setup friction. **Complete Tier 1 before attempting Tier 3 or 4.**

## Tier 1 — Low friction (start here)

### `example.py` + YAML configs

Built-in car environments on shipped maps. Uni- and bidirectional search, profiler output, visualization.

| Config | Environment | Map |
|--------|-------------|-----|
| `Configs/kinodynamic_example.yml` | Kinodynamic car | Off-road `race-2` |
| `Configs/kinematic_example.yml` | Kinematic car | Street `Berlin_0_1024.png` |
| `Configs/simple_example.yml` | 2D grid | Generated bottleneck |

```bash
python examples/standalone/example.py --config Configs/kinodynamic_example.yml
python examples/standalone/example.py --config Configs/kinodynamic_example.yml --bidirectional
python examples/standalone/example.py --config Configs/kinodynamic_example.yml --test-case case2
```

**Copy into your stack:** `create_planner` → load map tensor → `search` → `get_best_path`. See [api.md](api.md).

Docs: [examples/standalone/README.md](../examples/standalone/README.md)

### `generic_integrator.py`

Custom 2D point-mass problem via Python callbacks. No map files, no C++.

```bash
python examples/standalone/generic_integrator.py
```

**Copy into your stack:** callback definitions + [generic_integrator.yml](../examples/standalone/Configs/generic_integrator.yml). See [generic_environment.md](generic_environment.md).

## Tier 2 — Medium friction (advanced patterns)

### `generic_ur5e_mjx.py` (skeleton)

6-DoF UR5e arm planning scaffold with MuJoCo/MJX batched FK. **Not a finished tutorial**—contributions welcome.

```bash
# Requires mujoco; optional mujoco-mjx + jax for MJX backend
python examples/standalone/generic_ur5e_mjx.py
```

**Copy into your stack:** pattern for sim-backed `validity_fn` with batched collision checks.

### `generic_latent_planning_smoke.py`

Latent-space planning with `hash_dims < state_dim` (PCA subspace hashing, full-D dynamics). Placeholder world model—swap for your encoder + learned dynamics.

```bash
python examples/standalone/generic_latent_planning_smoke.py
```

## Tier 3 — High friction (ROS integration stencil)

### `examples/ROS/`

In-the-loop planner on ROS topics (costmap, odometry, waypoints). Not a full ROS package—use as a stencil.

**Prerequisites:** ROS Noetic, `pip install -e .`, rosbags from [ROS README](../examples/ROS/README.md).

```bash
roscore
python examples/ROS/example.py
# In another terminal: rosbag play ...
```

**Copy into your stack:** topic wiring, map tensor assembly, replan loop.

## Tier 4 — Very high friction (full autonomy demo)

### `examples/BeamNG/`

IGHA* global planner + MPPI tracking in BeamNG via [BeamNGRL](https://github.com/prl-mushr/BeamNGRL/tree/devel).

**Prerequisites:** BeamNGRL devel branch, BeamNG.sim, tuned MPC parameters.

```bash
cd examples/BeamNG
python example.py --remote True --host_IP <beamng_host>
```

**Copy into your stack:** multiprocess planner wrapper ([IGHAStarMP.py](../examples/BeamNG/IGHAStarMP.py)), costmap from BEV, closed-loop tracking.

Bidirectional YAML reference: [BeamNG README](../examples/BeamNG/README.md#bidirectional-search-parameters).

## Future

**Isaac Lab** integration is planned; no example in this release yet.

## Tests

```bash
python tests/unit/environments/test_environment.py --env kinodynamic
python tests/unit/environments/test_environment.py --env kinematic --cpu
```

See [tests/README.md](../tests/README.md).
