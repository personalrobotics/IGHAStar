"""Fetch / locate robin_hood.h for the optional hash-map backend.

Default install path (in-package, ships with pip):
  ighastar/third_party/robin_hood/robin_hood.h

Override: ROBIN_HOOD_DIR=/dir/containing/robin_hood.h
Disable:  IGHA_NO_ROBIN_HOOD=1
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

ROBIN_HOOD_REPO = "https://github.com/martinus/robin-hood-hashing.git"

# ighastar/scripts/ -> ighastar/
_PKG_ROOT = Path(__file__).resolve().parent.parent
_BUNDLED = _PKG_ROOT / "third_party" / "robin_hood"


def bundled_dir() -> Path:
    return _BUNDLED


def resolve_robin_hood_include() -> Optional[Path]:
    """Directory that contains robin_hood.h, or None if unavailable / disabled."""
    if os.environ.get("IGHA_NO_ROBIN_HOOD", "").lower() in ("1", "true", "yes"):
        return None
    env = os.environ.get("ROBIN_HOOD_DIR")
    if env:
        p = Path(env)
        if (p / "robin_hood.h").is_file():
            return p
        return None
    if (_BUNDLED / "robin_hood.h").is_file():
        return _BUNDLED
    # Legacy sibling clone (dev machines).
    sibling = _PKG_ROOT.parent.parent / "robin-hood-hashing" / "src" / "include"
    if (sibling / "robin_hood.h").is_file():
        return sibling
    return None


def ensure_robin_hood(*, quiet: bool = False) -> Optional[Path]:
    """Ensure bundled robin_hood.h exists (clone+copy if needed). Returns include dir."""
    if os.environ.get("IGHA_NO_ROBIN_HOOD", "").lower() in ("1", "true", "yes"):
        if not quiet:
            print("robin_hood: skipped (IGHA_NO_ROBIN_HOOD)")
        return None

    env = os.environ.get("ROBIN_HOOD_DIR")
    if env:
        p = Path(env)
        if (p / "robin_hood.h").is_file():
            return p
        raise FileNotFoundError(
            f"ROBIN_HOOD_DIR={env} but robin_hood.h not found there"
        )

    if (_BUNDLED / "robin_hood.h").is_file():
        return _BUNDLED

    _BUNDLED.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"robin_hood: cloning {ROBIN_HOOD_REPO} → {_BUNDLED}")
    with tempfile.TemporaryDirectory(prefix="robin_hood_") as tmp:
        tmp_path = Path(tmp)
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--quiet",
                ROBIN_HOOD_REPO,
                str(tmp_path / "src"),
            ]
        )
        src = tmp_path / "src"
        hdr = src / "src" / "include" / "robin_hood.h"
        if not hdr.is_file():
            # older layout fallback
            hdr = src / "robin_hood.h"
        if not hdr.is_file():
            raise FileNotFoundError("cloned robin-hood-hashing but robin_hood.h missing")
        shutil.copy2(hdr, _BUNDLED / "robin_hood.h")
        for name in ("LICENSE", "LICENSE.txt", "LICENSE.md"):
            lic = src / name
            if lic.is_file():
                shutil.copy2(lic, _BUNDLED / "LICENSE")
                break
    if not quiet:
        print(f"robin_hood: ready at {_BUNDLED / 'robin_hood.h'}")
    return _BUNDLED


if __name__ == "__main__":
    ensure_robin_hood(quiet=False)
