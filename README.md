<div align="center">

# IGHA*: Incremental Generalized Hybrid A* Search

## ([Website](https://personalrobotics.github.io/IGHAStar/) · [Paper](https://personalrobotics.cs.washington.edu/publications/talia2025ighastar.pdf) · IEEE RA-L, Nov 2025)

</div>

<figure align="center">
  <img src="Content/IGHAStar_main_fig.png" alt="IGHAStar Main Figure" width="1000"/>
  <figcaption><b>Fig. 1:</b> The issue with Hybrid A* Search. Too coarse grid resolution risks failure (a), while too fine leads to excessive expansions and slow planning (e).</figcaption>
</figure>
<br>
<br>
We address the problem of efficiently organizing
search over very large trees, which arises in many applications
such as autonomous driving, aerial vehicles, and so on; here, we
are motivated by off-road autonomy, where real-time planning is
essential.
Classical approaches use graphs of motion primitives
and exploit dominance to mitigate the curse of dimensionality
and prune expansions efficiently. However, for complex dynamics,
repeatedly solving two-point boundary-value problems makes
graph construction too slow for fast kinodynamic planning.
Hybrid A* (HA*) addressed this challenge by searching over a
tree of motion primitives and introducing approximate pruning
using a grid-based dominance check.
However, choosing the grid
resolution is difficult: too coarse risks failure (Fig. 1(a)), while too fine
leads to excessive expansions and slow planning (Fig. 1(e)).
To overcome this, we propose Incremental Generalized Hybrid A* (IGHA*), an
anytime tree-search framework that dynamically organizes vertex expansions
without rigid pruning.
IGHA* provably matches or outperforms HA*, and has been tested in both simulation (Fig. 2, left) and in the real world on a
small scale off-road platform (Fig. 2, right). We also provide **BiIGHA***, a bidirectional variant that searches from both start and goal simultaneously, further reducing expansions in many scenarios.

<p align="center">

<table>
  <tr>
    <td><img src="Content/ighastar_sim.gif" alt="IGHAStar Simulation" width="470"/></td>
    <td><img src="Content/ighastar_real.gif" alt="IGHAStar Real-World" width="470"/></td>
  </tr>
  <tr>
    <td colspan="2" align="center"><b>Fig. 2:</b> IGHA* in simulation (left) and real-world testing on a small-scale off-road platform (right).</td>
  </tr>
</table>

</p>
Generally speaking, IGHA* will find better paths given the same compute budget,
compared to a smart multi-resolution version of HA* (HA*M),
which produces suboptimal (looping) paths, visible in the simulation comparison.<br><br>

### Bidirectional IGHA* (BiIGHA*)
<figure align="center">
  <img src="Content/BiIGHAStar_main_fig.png" alt="BiIGHAStar Main Figure" width="1000"/>
  <figcaption><b>Fig. 3:</b> Comparison of IGHA* (Uni) and BiIGHA* (Bi). At the base resolution, IGHA* fails (a), whereas BiIGHA*'s backward search succeeds (b). IGHA* must refine the resolution to obtain first a suboptimal (c) then optimal solution (e). In contrast, BiIGHA* recovers equivalent-cost solutions at lower resolutions via near-meets (d, f), using far fewer expansions.</figcaption>
</figure>
<br>
<br>
IGHA*'s key innovation is *freezing* vertices for later iterations rather than pruning them outright. However, these frozen vertices can temporarily hide solution-supporting vertices from the search. BiIGHA* extends IGHA* with bidirectional search to mitigate this effect. Beyond the classical benefit of reduced search depth, searching from both start and goal fundamentally reduces the impact of frozen vertices obscuring solutions (Fig. 4, Left). BiIGHA* preserves IGHA*'s guarantees on monotonic cost improvement and termination, while empirically requiring significantly fewer expansions across a range of planning problems.

<p align="center">

<table>
  <tr>
    <td><img src="Content/biigha_comparison_bench.gif" alt="BiIGHA* vs IGHA* Benchmark" width="470"/></td>
    <td><img src="Content/biigha_comparison_sim.gif" alt="BiIGHA* vs IGHA* Simulation" width="470"/></td>
  </tr>
  <tr>
    <td colspan="2" align="center"><b>Fig. 4:</b> (Left) 3-DoF benchmark showing how bidirectional search mitigates the frozen vertex barrier, (Right) BiIGHA* (top) vs IGHA* (bottom) in simulation.</td>
  </tr>
</table>

</p>

In practice this translates to the planner producing good solutions under tighter computational budgets, where unidirectional IGHA* may not find any solution at all (Fig. 4, right). The examples below run with far fewer expansions when you pass `--bidirectional`; the per-thread expansion limit is printed in green on the command line.

The core is C++/CUDA ([ighastar/src/ighastar.cpp](ighastar/src/ighastar.cpp)), exposed to Python via [PyTorch C++/CUDA](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html#cpp-custom-ops-tutorial) JIT compilation. The first run compiles and caches the extension; CUDA is used when available, with CPU fallback otherwise.

For extended algorithm discussion and additional figures, see the [project website](https://personalrobotics.github.io/IGHAStar/). For API, configuration, and integration guides, see [Documentation](#documentation) below.

## Quickstart

```bash
git clone --single-branch --branch main https://github.com/personalrobotics/IGHAStar.git
cd IGHAStar
conda env create -f ighastar.yml && conda activate ighastar
pip install -e .

# Built-in car environment (first run compiles CUDA; may take several minutes)
python examples/standalone/example.py --config Configs/kinodynamic_example.yml

# Custom problem via Python callbacks (no new C++ required)
python examples/standalone/generic_integrator.py
```

See [docs/quickstart.md](docs/quickstart.md) for expected output, CPU fallback, and bidirectional usage.

## Documentation

| Guide | Description |
|-------|-------------|
| [docs/quickstart.md](docs/quickstart.md) | Install, run examples, first-run notes |
| [docs/api.md](docs/api.md) | Programmatic planner API |
| [docs/configuration.md](docs/configuration.md) | YAML parameter reference |
| [docs/generic_environment.md](docs/generic_environment.md) | **Custom problems via Python callbacks** |
| [docs/extending.md](docs/extending.md) | Choose generic vs built-in vs C++ env |
| [docs/when_to_use.md](docs/when_to_use.md) | When IGHA* vs MPC / MPPI / RL |
| [docs/examples.md](docs/examples.md) | Tiered example catalog |
| [Algorithm & paper](https://personalrobotics.github.io/IGHAStar/) | Method, figures, benchmarks (website) |

## Example tiers

- **Start here:** [examples/standalone/](examples/standalone/) — car-on-map demos and generic integrator (low friction).
- **Integration stencils:** [examples/ROS/](examples/ROS/) (in-the-loop on topics) and [examples/BeamNG/](examples/BeamNG/) (full sim + MPPI). Complete standalone examples first.
- **Planned:** Isaac Lab integration.

## Installation

Tested on Ubuntu 20.04. Use a virtual environment or conda.

```bash
conda env create -f ighastar.yml
conda activate ighastar
pip install -e .
```

**Compute:** CUDA is used when available; otherwise the planner falls back to CPU ([common_utils.py](ighastar/scripts/common_utils.py)). The first run JIT-compiles the extension and caches it for later use.

## Reproducibility

Benchmark figures in the paper (IGHA* vs HA*M, expansion curves) were produced with an earlier research codebase that integrated OMPL baselines. **This release** demonstrates the shipped implementation via standalone examples and built-in profiler output—it is not a paper reproduction artifact. For qualitative comparisons, see the [website](https://personalrobotics.github.io/IGHAStar/).

## Citation

```bibtex
@ARTICLE{11297782,
  author={Talia, Sidharth and Salzman, Oren and Srinivasa, Siddhartha},
  journal={IEEE Robotics and Automation Letters},
  title={Incremental Generalized Hybrid {A*}},
  year={2025},
  doi={10.1109/LRA.2025.3643271}
}
```

## Project structure

```
IGHAStar/
├── ighastar/              # Package: C++/CUDA core, environments, create_planner()
├── examples/
│   ├── standalone/        # Start here
│   ├── ROS/               # ROS integration stencil
│   └── BeamNG/            # BeamNG + MPPI demo
├── docs/                  # Practitioner guides + project website (HTML)
├── tests/                 # Unit tests
└── Content/               # Figures and media
```

## Contributing

1. Fork the repository and create a feature branch.
2. Format C++ with [clang-format](https://clang.llvm.org/docs/ClangFormat.html), Python with [black](https://black.readthedocs.io/).
3. Use [PEP 484](https://peps.python.org/pep-0484/) type hints for Python.
4. Test on CPU and GPU where applicable.
5. Submit a pull request.
