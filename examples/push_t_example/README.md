# Closed-loop Push-T

QS or Chipmunk IGHA* plans; MPPI tracks the path in gym-pusht, or `--open-loop` plays the planner actions.

## Dependencies

| Need | Where |
| --- | --- |
| `ighastar` | `pip install -e .` from the repo root |
| Chipmunk 7 | `pusht_ext/Chipmunk2D` next to this repo, or `PUSHT_CHIPMUNK_DIR` |
| `gym-pusht` | `pip install gym-pusht` |
| MPPI | sibling `../MPPI` (cloned from https://github.com/sidtalia/MPPI.git on first closed-loop run) |

Chipmunk is checked when the planner extension builds. A missing `gym-pusht` raises on import. A missing MPPI is cloned next to `ighastar` (outside this workspace tree’s package, as a sibling of the repo).

## Run

```bash
cd examples/push_t_example
python example.py                  # Content/push_t_example/push_t_closed_loop.mp4
python example.py --open-loop --dynamics qs
python example.py --open-loop --dynamics chipmunk
python example.py --live
python example.py --no-visuals     # exits 0 on goal, raises on timeout
```
