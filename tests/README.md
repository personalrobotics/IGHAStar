# IGHAStar Environment Unit Tests

This directory contains comprehensive unit tests for all IGHAStar environment types. The test framework validates core environment functions including validity checking, successor generation, goal reaching, and heuristic calculations.

## Overview

The test framework consists of:
- **`test_environment.py`**: Python orchestrator that manages compilation and test execution
- **`test.cpp`**: Generic C++ test implementation that works with all environment types
- **Programmatically generated test maps**: No external file dependencies

## Supported Environments

### Environment Types
- **Simple**: 2D states `[x, y]` with header-only implementation
- **Kinematic**: 3D states `[x, y, theta, unused]` with GPU/CPU versions
- **Kinodynamic**: 4D states `[x, y, theta, velocity]` with GPU/CPU versions

### Compilation Modes
- **GPU/CUDA**: Automatic when CUDA is available
- **CPU**: Forced with `--cpu` flag or automatic fallback
- **Cross-platform**: Linux and macOS support with proper include paths

## Test Coverage

Each environment is tested for:

1. **Validity Checking**
   - Invalid map: All positions should be marked invalid (return 0)
   - Valid map: All positions should be marked valid (return 2)

2. **Successor Generation**
   - Ensures at least one successor differs from start state
   - Validates motion primitive generation

3. **Goal Reaching**
   - Close distance: Should reach goal region
   - Far distance: Should not reach goal region

4. **Heuristic & Distance Calculations**
   - Non-negative values
   - Zero distance/heuristic for same position

## Usage

### Command Line Options
- `--env {simple,kinematic,kinodynamic}`: Environment type to test (default: kinematic)
- `--cpu`: Force CPU compilation even when CUDA is available

```bash
# Test specific environment (GPU version)
python3 tests/unit/environments/test_environment.py --env kinematic
python3 tests/unit/environments/test_environment.py --env kinodynamic

# Force CPU compilation
python3 tests/unit/environments/test_environment.py --env kinematic --cpu
python3 tests/unit/environments/test_environment.py --env kinodynamic --cpu

# Always on CPU
python3 tests/unit/environments/test_environment.py --env simple

```

## Generic environments (`generic` and `generic_torch`)

The callback-driven environments have their own pytest suites. Both suites use the problem definitions in `tests/unit/generic_problems.py`, which is the 2D integrator from `examples/standalone/generic_integrator.py` plus a few variants.

| File | What it covers |
|---|---|
| `unit/environments/test_generic_environment.py` + `test_generic.cpp` | Environment-level checks, mirroring `test.cpp`: validity (single and batched), successors (pruning, batched vs single), goal reaching, heuristic, multi-step rollouts. Every check runs against both `generic` and `generic_torch`. Parity tests compare the two modes on identical inputs, including a seeded random control sampler. |
| `unit/planner/test_generic_torch_planner.py` | Full IGHA\* / BiIGHA\* searches. Covers parity with `generic` (identical expansion count, path, controls and cost), proof that the search makes no Python calls (counting/raising callbacks), GIL release during search, rollouts, random samplers, fixed control tensors, the compile cache, export error messages, and CUDA (skipped without a GPU). |
| `unit/environments/test_external_env.py` | Smoke for `IGHASTAR_ENV_HEADER`: builds `ighastar.cpp` against an out-of-tree stub header and constructs `IGHAStar`. |

`tests/conftest.py` puts this checkout first on `sys.path`, points `IGHASTAR_SRC_DIR` at this checkout's sources, and adds the interpreter's `bin/` to `PATH` so `ninja` is found.

```bash
# From the repository root (requires pytest)
python3 -m pytest tests/unit/environments/test_generic_environment.py \
                  tests/unit/planner/test_generic_torch_planner.py -v

# Or one suite at a time
python3 -m pytest tests/unit/environments/test_generic_environment.py -v
python3 -m pytest tests/unit/planner/test_generic_torch_planner.py -v
```

Runtime: about 7 s when everything is cached. A cold run (empty `~/.cache/torch_extensions` and `~/.cache/ighastar/generic_torch`) compiles about 11 AOTInductor packages (4 of them for CUDA) and 4 C++ extensions, which takes about 2 minutes on a 16-core machine. Set `IGHASTAR_AOTI_CACHE=/some/dir` to keep compiled test packages separate from your own.


## Example Output

```
Running Kinematic Environment Function Tests
============================================================
=============================================================
Running Environment Unit Tests
=============================================================
=== DIAGNOSTIC TESTS ===
Test 1: Validity checking with invalid map...
✓ All positions correctly identified as invalid
Test 2: Validity checking with valid map...
✓ All positions correctly identified as valid
Test 3: Successor generation...
✓ Generated 10 successors, at least one different from start
Test 4: Goal reaching (close distance)...
✓ Close goal correctly reached
Test 5: Goal reaching (far distance)...
✓ Far goal correctly not reached
Test 6: Heuristic and distance calculations...
✓ Heuristic and distance calculations work correctly
=============================================================
✅ All tests passed!
=============================================================
```