<div align="center">

# IGHA*: Incremental Generalized Hybrid A* Search

## ([Website](https://personalrobotics.github.io/IGHAStar/) · [Paper](https://personalrobotics.cs.washington.edu/publications/talia2025ighastar.pdf) · IEEE RA-L, Nov 2025)

</div>

**IGHA*** is an anytime kinodynamic tree-search planner that dynamically organizes vertex expansions without rigid grid-based pruning. **BiIGHA*** extends it with bidirectional search, often finding good solutions with far fewer expansions. The core is C++/CUDA ([ighastar/src/ighastar.cpp](ighastar/src/ighastar.cpp)), exposed to Python via PyTorch JIT compilation.

For algorithm details, figures, and paper results, see the [project website](https://personalrobotics.github.io/IGHAStar/).

<p align="center">
  <img src="Content/standalone/race-2_kinodynamic.png" alt="Kinodynamic planning example" width="500"/>
  <br>
  <em>Kinodynamic planning on off-road terrain (standalone example).</em>
</p>

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
