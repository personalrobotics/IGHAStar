# IGHAStar practitioner documentation

Algorithm details, paper figures, and BibTeX live on the [project website](https://personalrobotics.github.io/IGHAStar/) ([docs/index.html](index.html) in this folder). The guides below are for **using and extending** the released code.

## Guides

| Document | Purpose |
|----------|---------|
| [quickstart.md](quickstart.md) | Install, run examples, expected output |
| [api.md](api.md) | Programmatic planner API (`create_planner`, `search`, paths) |
| [configuration.md](configuration.md) | YAML parameter reference and tuning |
| [generic_environment.md](generic_environment.md) | Custom problems via Python callbacks (**start here to extend**) |
| [extending.md](extending.md) | Generic vs built-in car env vs hand-written C++ |
| [when_to_use.md](when_to_use.md) | When IGHA* makes sense vs MPC, MPPI, RL |
| [examples.md](examples.md) | Tiered example catalog (standalone → ROS → BeamNG) |

## Related READMEs

- [examples/standalone/README.md](../examples/standalone/README.md) — map configs and test cases
- [ighastar/src/Environments/README.md](../ighastar/src/Environments/README.md) — C++ environment interface
- [tests/README.md](../tests/README.md) — running unit tests
