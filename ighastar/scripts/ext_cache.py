"""Persistent torch cpp_extension cache for Jetson / Docker.

Torch's default ``~/.cache/torch_extensions`` is ephemeral in containers and
rebuilds when absolute source paths change. On Jetson (or when
``IGHASTAR_EXT_CACHE`` is set), we content-hash the inputs, reuse a copied
``.so`` across container starts, and point ``TORCH_EXTENSIONS_DIR`` at the
same durable directory so ninja intermediates survive too.

Disable: ``IGHASTAR_NO_EXT_CACHE=1``.
Override dir: ``IGHASTAR_EXT_CACHE=/path/to/dir``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.cpp_extension import load

_HEADER_SUFFIXES = {".h", ".hpp", ".cuh", ".cu", ".cpp", ".cc", ".c"}


def on_jetson() -> bool:
    if Path("/etc/nv_tegra_release").is_file():
        return True
    try:
        model = Path("/proc/device-tree/model").read_text(errors="ignore").lower()
    except OSError:
        return False
    return "tegra" in model or "jetson" in model


def cache_enabled() -> bool:
    if os.environ.get("IGHASTAR_NO_EXT_CACHE", "").lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("IGHASTAR_EXT_CACHE"):
        return True
    return on_jetson()


def cache_root() -> Path:
    env = os.environ.get("IGHASTAR_EXT_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    home_cache = Path.home() / ".cache" / "ighastar_ext"
    for cand in (
        Path.home() / "colcon_ws" / "cache" / "ighastar_ext",
        Path("/root/colcon_ws/cache/ighastar_ext"),
        home_cache,
    ):
        try:
            if cand.parent.is_dir():
                return cand
        except OSError:
            continue
    return home_cache


def _digest_file(h: "hashlib._Hash", path: Path) -> None:
    # Content only (not absolute path) so Docker remounts still cache-hit.
    h.update(path.name.encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)


def _digest_includes(h: "hashlib._Hash", includes: Iterable[str]) -> None:
    for inc in includes:
        root = Path(inc)
        if not root.is_dir():
            continue
        for f in sorted(root.rglob("*")):
            if f.is_file() and f.suffix in _HEADER_SUFFIXES:
                _digest_file(h, f)


def fingerprint(name: str, sources: Sequence[str], kwargs: dict) -> str:
    h = hashlib.sha256()
    h.update(name.encode())
    h.update(sys.version.encode())
    h.update(torch.__version__.encode())
    h.update((torch.version.cuda or "cpu").encode())
    if torch.cuda.is_available():
        h.update(str(torch.cuda.get_device_capability()).encode())
    for src in sources:
        _digest_file(h, Path(src))
    for key in ("extra_cflags", "extra_cuda_cflags", "extra_ldflags"):
        for flag in kwargs.get(key) or []:
            h.update(str(flag).encode())
    _digest_includes(h, kwargs.get("extra_include_paths") or [])
    return h.hexdigest()[:24]


def _import_so(name: str, so_path: Path) -> Any:
    # PyInit_{name} is keyed off this name (not the file path). Reuse an
    # already-imported module in this process (e.g. right after a cache miss).
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", None):
        return existing
    spec = importlib.util.spec_from_file_location(name, so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import extension from {so_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_extension(name: str, sources: Sequence[str], **kwargs: Any) -> Any:
    """Like ``torch.utils.cpp_extension.load``, with optional durable .so cache."""
    sources = [str(p) for p in sources]
    if not cache_enabled():
        return load(name=name, sources=sources, **kwargs)

    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    # Keep torch's own build tree on the same volume (survives Docker restarts).
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(root / "torch_extensions"))

    key = fingerprint(name, sources, kwargs)
    # ``{name}.so`` basename is required for PyInit_{name}; key is the dir.
    so_path = root / name / key / f"{name}.so"
    if so_path.is_file():
        print(f"[ighastar] ext cache hit: {name}/{key}")
        return _import_so(name, so_path)

    print(f"[ighastar] ext cache miss: building {name}/{key}")
    mod = load(name=name, sources=sources, **kwargs)
    built = getattr(mod, "__file__", None)
    if built and Path(built).is_file():
        so_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = so_path.with_suffix(f".{os.getpid()}.tmp")
        shutil.copy2(built, tmp)
        os.replace(tmp, so_path)
    return mod


def _self_check() -> None:
    # ponytail: fingerprint must be stable for identical inputs.
    here = Path(__file__).resolve()
    a = fingerprint("demo", [str(here)], {"extra_cflags": ["-O3"]})
    b = fingerprint("demo", [str(here)], {"extra_cflags": ["-O3"]})
    assert a == b, (a, b)
    c = fingerprint("demo", [str(here)], {"extra_cflags": ["-O2"]})
    assert a != c, a
    print("ext_cache self-check ok", a[:8])


if __name__ == "__main__":
    _self_check()
