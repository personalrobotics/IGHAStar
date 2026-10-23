# Memory leak: preemptive expansion keeps unused speculated nodes alive

With preemptive expansion enabled, every node IGHA* expands speculatively but never pops stays in
memory after its search, and after its planner is destroyed. Memory grows with every search and is
never returned. Found while benchmarking a GPU environment for
[jaxmaze](https://github.com/personalrobotics/jaxmaze); measured and fixed on `main` (`6941f2b`).

## Impact

- Only searches with `preemptive_expansion.enabled: true` leak. That's the batching mode meant for
  GPU environments; it's off by default.
- Both `IGHAStar` and `BiIGHAStar` (which owns two `IGHAStar` searches), for every environment.
- The leak grows with the number of speculated nodes, so with problem size and `max_preemptive`.
  It was about 3.5 MB per search on the 2D integrator below, and about 64 MB per search on a
  5-dimensional car problem in a maze (jaxmaze's extended bicycle, `max_preemptive=8`). A
  long-running planner, or a process that builds many planners, eventually runs out of memory.

## Root cause

A reference cycle between a speculated node and its cached successors:

- `Node::parent` is a `std::shared_ptr<Node>`, so every child keeps its parent alive
  (`Environments/include/generic_common.h`; the other environments' `Node`s are the same).
- When a node is expanded preemptively, its successors are stored on it in
  `Node::cached_succ`, a `std::vector<std::shared_ptr<Node>>` (`IGHAStar::expand_with_preemption`,
  `p->cached_succ = std::move(results[idx])`). Now the parent also keeps each child alive.
- The cache is only emptied when that node is popped (`std::move(v->cached_succ)` then
  `v->cached_succ.clear()`). `reset()` and `naive()` clear `unexpanded_stash` and the start node's
  cache, and the destructor only deletes the environment.

Preemptive candidates are taken from the stash first-in-first-out rather than by `f`, so most of
them are never popped. Each of those keeps its cycle, parent and every cached child, after the
search's queues and maps release their references, so `shared_ptr` reference counting can never
free them.

## Reproduction

Run from the repository root (needs torch and psutil). It uses the 2D integrator from
`tests/unit/generic_problems.py` in `generic` mode, and measures resident memory over repeated
fresh planners, each searched once and destroyed:

```python
import gc
import sys

import psutil

sys.path.insert(0, "tests/unit")
sys.path.insert(0, ".")
import generic_problems as gp
from ighastar.scripts.common_utils import create_planner

proc = psutil.Process()
for enabled in (False, True):
    cfg = gp.make_configs(
        "generic",
        gp.callbacks(),
        preemptive_expansion={"enabled": enabled, "min_preemptive": 8, "max_preemptive": 32},
        max_expansions=3000,
    )
    planner = create_planner(cfg)  # Warm-up: build the extension, one search.
    gp.run_search(planner, cfg)
    del planner
    gc.collect()
    for n in (20, 60):
        before = proc.memory_info().rss
        for _ in range(n):
            planner = create_planner(cfg)
            gp.run_search(planner, cfg)
            del planner
            gc.collect()
        print(f"preemptive={enabled}: {n} searches -> {(proc.memory_info().rss - before) / 1e6:.1f} MB")
```

## Results

Every search finds the same solution in 548 expansions; with preemption on, 1,740 nodes per search
are expanded speculatively.

| | preemption off | preemption on |
|---|---:|---:|
| 20 fresh planners, one search each | 0.6 MB | 68.6 MB |
| 60 fresh planners, one search each | – | 207.9 MB |
| 20 searches on one planner | 0.4 MB | 69.0 MB |
| **with the fix:** 20 / 60 fresh planners | – | 0.8 / 0.0 MB |
| **with the fix:** 20 searches on one planner | – | 0.0 MB |

Without the fix the growth is linear, about 3.5 MB per search; with it, memory stays flat (the
remaining differences are allocator noise, not growing with the number of searches).

## Fix

In `ighastar/src/ighastar.cpp` (uncommitted on `main`):

- `IGHAStar` keeps `preexpanded_nodes`, every node it stores cached successors on.
- A new `release_preemptive_caches()` empties those caches (and the stash), which breaks the
  cycles of the caches no search consumed.
- It runs in `reset()` (so at the start of every search), in `naive()`, and in `~IGHAStar()`.
  `BiIGHAStar` deletes its two searches, so it's covered too.

Search behavior is unchanged: the caches it drops belong to the previous search, or to a planner
being destroyed. The node list holds no node any longer than the search's own queues already do.

Verified with the reproduction above (results table), and `tests/unit` passes (49 passed; the CUDA
test skips without a GPU).

## Note: no test covers preemptive expansion

None of the tests in `tests/unit` enables `preemptive_expansion`, so neither this leak nor
preemption's correctness (the same search with it on and off) is exercised. Worth adding at least
a search with preemption on that checks the result against preemption off.
