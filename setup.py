"""setuptools entry: ensure robin_hood is present for install / editable installs."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop


def _ensure_robin_hood() -> None:
    # Load by path so this works before the package is on sys.path.
    path = (
        Path(__file__).resolve().parent
        / "ighastar"
        / "scripts"
        / "fetch_robin_hood.py"
    )
    spec = importlib.util.spec_from_file_location("fetch_robin_hood", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.ensure_robin_hood(quiet=False)


class BuildPy(build_py):
    def run(self):
        _ensure_robin_hood()
        super().run()


class Develop(develop):
    def run(self):
        _ensure_robin_hood()
        super().run()


_cmdclass = {"build_py": BuildPy, "develop": Develop}

# Newer setuptools editable installs.
try:
    from setuptools.command.editable_wheel import editable_wheel

    class EditableWheel(editable_wheel):
        def run(self):
            _ensure_robin_hood()
            super().run()

    _cmdclass["editable_wheel"] = EditableWheel
except ImportError:
    pass


setup(cmdclass=_cmdclass)
