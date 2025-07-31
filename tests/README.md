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