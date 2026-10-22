# robin_hood (optional hash-map backend)

Header-only [robin-hood-hashing](https://github.com/martinus/robin-hood-hashing) (MIT).
Shipped by default for IGHA* performance; see `LICENSE` in this directory.

- Enabled when `robin_hood.h` is present (default after `pip install`).
- Disable: `IGHA_NO_ROBIN_HOOD=1` (falls back to `std::unordered_*`).
- Override path: `ROBIN_HOOD_DIR=/path/to/dir/containing/robin_hood.h`

If this directory is empty, `pip install` / first planner build will shallow-clone
the upstream repo and copy the header + LICENSE here.
