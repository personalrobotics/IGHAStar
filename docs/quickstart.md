# Quickstart

## Prerequisites

- Ubuntu 20.04 (primary test platform)
- Python 3.8+ with PyTorch
- NVIDIA GPU + CUDA (recommended; CPU fallback available)

## Install

From the repository root:

```bash
conda env create -f ighastar.yml
conda activate ighastar
pip install -e .
```

## Run built-in car examples

From the repository root:

```bash
# Kinodynamic planning on off-road map
python examples/standalone/example.py --config Configs/kinodynamic_example.yml

# Bidirectional search (often fewer expansions)
python examples/standalone/example.py --config Configs/kinodynamic_example.yml --bidirectional

# Kinematic planning on street map (default config)
python examples/standalone/example.py

# Simple 2D grid planning
python examples/standalone/example.py --config Configs/simple_example.yml

# Alternate start/goal from YAML test_cases
python examples/standalone/example.py --config Configs/kinodynamic_example.yml --test-case case2
```

## Run a custom problem (generic environment)

No new C++ required—define dynamics and collision in Python:

```bash
python examples/standalone/generic_integrator.py
```

See [generic_environment.md](generic_environment.md) for the callback API.

## Expected output

After the first successful run you should see something like:

```
Search completed, success: True
Search took 0.23 seconds          # varies by GPU and problem
Expansion counter: 14689
First solution expansion: 5052
Best solution cost: 7.0
✓ Optimal path found!
Saved to: .../Content/standalone/race-2_kinodynamic_IGHAStar.png
```

The generic integrator prints `success: True`, expansion count, and a short path trace.

## First-run compile warning

The planner is JIT-compiled on first use via `torch.utils.cpp_extension.load`. **The first run can take several minutes** while C++/CUDA sources build. Subsequent runs reuse the cached extension.

If compilation fails, check that your PyTorch build matches your CUDA toolkit and that `ninja` is installed.

## CPU fallback

If `torch.cuda.is_available()` is false, [common_utils.py](../ighastar/scripts/common_utils.py) automatically selects CPU environment implementations (`kinematic_cpu`, `kinodynamic_cpu`). Planning will be slower but functionally equivalent for the built-in car environments.

## Next steps

- [api.md](api.md) — use the planner from your own script
- [configuration.md](configuration.md) — tune search parameters
- [when_to_use.md](when_to_use.md) — decide if IGHA* fits your problem
- [examples.md](examples.md) — full example catalog
