"""pytest setup: make sure *this checkout's* ighastar is imported and built.

- Puts the repository root first on sys.path (ahead of any installed or
  editable ighastar pointing at another checkout).
- Points IGHASTAR_SRC_DIR at this checkout's C++ sources (overriding any
  value from the environment, which would otherwise win in common_utils).
- Adds the interpreter's bin dir to PATH so torch.utils.cpp_extension finds the
  venv's ninja.
"""
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "unit"))
os.environ["IGHASTAR_SRC_DIR"] = str(REPO_ROOT / "ighastar")
_bin = str(pathlib.Path(sys.executable).parent)
if _bin not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")
