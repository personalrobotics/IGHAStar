# When to use IGHA*

Search-based kinodynamic planning pays off when **global structure and optimality** matter more than **planning latency**, and when simpler methods struggle with combinatorial structure (bottlenecks, gear changes, tight non-convex corridors).

IGHA* is **not** a drop-in replacement for every motion-planning or control problem.

## Use IGHA* / BiIGHA* when

- You need a **global kinodynamic path** with a discrete control structure (steering/throttle sets, integrator steps, joint deltas).
- The free space is **non-convex** and greedy local methods (pure MPC, naive sampling) loop, collide, or miss narrow passages.
- The problem has **combinatorial structure**: reverse segments, multi-phase maneuvers, off-road corridors with velocity/terrain constraints.
- You can afford roughly **0.1–10 seconds** of planning time in exchange for substantially better global paths.
- **Optimality or constraint-aware global structure** has high payoff relative to planning time (e.g. expensive replans, safety-critical global routes, long horizons).

IGHA* is **anytime**: you can stop at a budget and still improve with more expansions. BiIGHA* often finds feasible solutions under tighter budgets.

## Prefer other methods when

| Method | Typical use | Why not IGHA* |
|--------|-------------|---------------|
| **Convex MPC / QP** | Local trajectory tracking, smooth dynamics | No global combinatorial search needed |
| **MPPI / sampling MPC** | Fast reactive control, short horizon | Does not optimize global path structure |
| **Model-free RL** | High-rate body-frame control (e.g. quadruped twist policies) | No explicit global kinodynamic optimality |
| **OMPL (RRT*, PRM, …)** | Moderate-D geometric planning, library ecosystems | IGHA* targets kinodynamic trees with GPU-batched car dynamics or custom callbacks |
| **Trajectory optimization** | Smooth paths from a good initial guess | Expensive global search unnecessary if convexification works |

## Practical guidance

1. **Start with standalone examples** ([quickstart.md](quickstart.md)) on a problem structurally similar to yours.
2. **Prototype custom problems** with the [generic environment](generic_environment.md)—lowest friction.
3. **Use ROS / BeamNG examples** only as integration stencils after standalone works—they add sim, middleware, and tuning overhead ([examples.md](examples.md)).
4. If your problem is "track a reference path at 50 Hz," use a tracker (MPC/MPPI), not a global searcher.
5. If your problem is "find a feasible global route through rough terrain with gear changes," IGHA* is in scope.

## Reproducibility note

Paper benchmark comparisons (IGHA* vs multi-resolution Hybrid A*) used an earlier research codebase with OMPL baselines. This release demonstrates the shipped planner via examples and profiler output—not a reproduction of those exact plots. See the [website](https://personalrobotics.github.io/IGHAStar/) for paper figures.
