#!/usr/bin/env bash
# Set up IGHA* + Isaac Sim 6.1 / Isaac Lab 3.0 / UW Lab (OmniReset peg insertion) on a new machine, inside its OWN
# virtual environment. Nothing is installed into the system Python or into any existing venv / conda env.
# UNREVIEWED.
#
#   ./isaac_install.sh [--root DIR] [--with-cpu-model] [--reuse-venv]
#
#   --root DIR          where everything new goes (default: ~/ighastar_isaac):
#                         DIR/venv               the Python 3.12 venv (uv-managed)
#                         DIR/UWLab              UW Lab clone (pinned), which also holds Isaac Lab in UWLab/_isaaclab
#                         DIR/cpu_model_3261     only with --with-cpu-model
#   --with-cpu-model    also build the CPU MuJoCo model for start 3261 (Isaac Sim on the GPU, a few minutes) and run a
#                       short all-C++ search on it as an end-to-end check
#   --reuse-venv        continue in an existing DIR/venv instead of refusing (e.g. after a failed run)
#
# Touches outside DIR only: uv in ~/.local/bin if uv is missing (installed without editing your shell rc files), uv's
# Python / package caches, Isaac Sim's asset cache under /tmp, and `sudo apt install cmake build-essential` if UW Lab's
# installer finds cmake missing. This IGHAStar checkout is installed into the venv in editable mode (no copy).
#
# Reference machine: Ubuntu 22.04, RTX 4090, driver 575.57.08, gcc 11.4. Exact package versions of the reference venv:
# examples/isaac_peg_example/tools/isaac_venv_freeze.txt (used here as pip constraints for the extras).
set -euo pipefail

# ---------------------------------------------------------------- pins (bump together; see UW Lab's uwlab.sh)
UWLAB_REPO=https://github.com/UW-Lab/UWLab.git
UWLAB_COMMIT=2b40afb   # UW Lab main, uses Isaac Lab 3.0 EA @ ae37b028 (pinned inside uwlab.sh) and Isaac Sim 6.1
ISAACSIM_VERSION=6.1.0.0
TORCH_VERSION=2.11.0
TORCHVISION_VERSION=0.26.0
TORCH_INDEX=https://download.pytorch.org/whl/cu128
PYTHON_VERSION=3.12

ROOT="$HOME/ighastar_isaac"
WITH_CPU_MODEL=0
REUSE_VENV=0
while [ $# -gt 0 ]; do
    case "$1" in
        --root) ROOT="$2"; shift 2 ;;
        --with-cpu-model) WITH_CPU_MODEL=1; shift ;;
        --reuse-venv) REUSE_VENV=1; shift ;;
        -h|--help) sed -n 2,22p "$0"; exit 0 ;;
        *) echo "unknown argument: $1 (see --help)" >&2; exit 1 ;;
    esac
done

IGHASTAR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE="$IGHASTAR/examples/isaac_peg_example"
FREEZE="$EXAMPLE/tools/isaac_venv_freeze.txt"
mkdir -p "$ROOT"
ROOT=$(cd "$ROOT" && pwd)
VENV="$ROOT/venv"
UWLAB="$ROOT/UWLab"
log() { echo -e "\n[isaac_install] $*"; }
die() { echo "[isaac_install] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------- 0. system checks (nothing is changed here)
log "checking the system"
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || die "needs Linux x86_64 (Isaac Sim pip wheels)"
glibc=$(ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+$')
python3 -c "import sys; sys.exit(0 if tuple(map(int, '$glibc'.split('.'))) >= (2, 35) else 1)" \
    || die "GLIBC $glibc < 2.35: Isaac Sim's pip install needs Ubuntu 22.04+ (or use the Isaac Sim binaries)"
command -v nvidia-smi >/dev/null || die "no nvidia-smi: install the NVIDIA driver first"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
for tool in git gcc g++ curl; do command -v $tool >/dev/null || die "missing '$tool' (sudo apt install git build-essential curl)"; done
if [ -n "${CONDA_PREFIX:-}" ]; then
    die "a conda env is active ($CONDA_PREFIX); run 'conda deactivate' first so nothing gets installed into it"
fi
if [ -n "${VIRTUAL_ENV:-}" ] && [ "$VIRTUAL_ENV" != "$VENV" ]; then
    die "another venv is active ($VIRTUAL_ENV); run 'deactivate' first so nothing gets installed into it"
fi
echo "IGHAStar checkout: $IGHASTAR ($(git -C "$IGHASTAR" rev-parse --abbrev-ref HEAD) @ $(git -C "$IGHASTAR" rev-parse --short HEAD))"
echo "install root:      $ROOT"

# ---------------------------------------------------------------- 1. uv + venv
if ! command -v uv >/dev/null; then
    if [ -x "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        log "installing uv into ~/.local/bin (shell rc files are not modified)"
        curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi
if [ -d "$VENV" ] && [ "$REUSE_VENV" = 0 ]; then
    die "$VENV already exists. Use --reuse-venv to continue in it, or --root to pick a fresh location."
fi
if [ ! -d "$VENV" ]; then
    log "creating the venv (Python $PYTHON_VERSION, uv-managed interpreter): $VENV"
    uv venv --python "$PYTHON_VERSION" --seed "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
[ "$(python -c 'import sys; print(sys.prefix)')" = "$VENV" ] || die "venv activation failed"

# ---------------------------------------------------------------- 2. Isaac Sim + torch
log "installing Isaac Sim $ISAACSIM_VERSION (large download, ~15 GB on disk)"
uv pip install "isaacsim[all,extscache]==$ISAACSIM_VERSION" --extra-index-url https://pypi.nvidia.com
log "installing torch $TORCH_VERSION (CUDA 12.8)"
uv pip install -U "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX"

# ---------------------------------------------------------------- 3. UW Lab (+ Isaac Lab, Newton, MuJoCo-Warp)
if [ ! -d "$UWLAB/.git" ]; then
    log "cloning UW Lab @ $UWLAB_COMMIT into $UWLAB"
    git clone "$UWLAB_REPO" "$UWLAB"
fi
git -C "$UWLAB" fetch -q origin
git -C "$UWLAB" checkout -q "$UWLAB_COMMIT"
log "installing UW Lab + its pinned Isaac Lab (editable, rsl_rl only) via uwlab.sh"
(cd "$UWLAB" && ./uwlab.sh --install rsl_rl)

# ---------------------------------------------------------------- 4. IGHA* and the peg example's extras
log "installing this IGHAStar checkout (editable, no dependency changes) and the example's extras"
uv pip install --no-deps -e "$IGHASTAR"
# pinned to the reference venv's versions where it has them; viser (tree viewer) is optional
uv pip install -c "$FREEZE" ninja pyyaml scipy imageio imageio-ffmpeg mujoco matplotlib trimesh viser

# ---------------------------------------------------------------- 5. checks
log "checking imports and versions"
python - <<'PY'
import importlib
for m in ("torch", "warp", "mujoco", "mujoco_warp", "newton", "isaaclab", "uwlab_tasks", "ighastar"):
    mod = importlib.import_module(m)
    print(f"  {m:12s} {getattr(mod, '__version__', '')}")
import torch
assert torch.cuda.is_available(), "torch does not see the GPU"
print("  CUDA device:", torch.cuda.get_device_name(0))
PY
log "building the C++ MuJoCo planner test module (checks compiler, ninja, MuJoCo headers/libs)"
(cd "$EXAMPLE" && python -c "import mujoco_cpp; m = mujoco_cpp.build_test_module(); print('  built', m.__name__)")

if [ "$WITH_CPU_MODEL" = 1 ]; then
    OUT="$ROOT/cpu_model_3261"
    log "building the CPU MuJoCo model for start 3261 -> $OUT (runs Isaac Sim headless; the first run accepts/asks for
the Isaac Sim EULA and downloads assets)"
    "$EXAMPLE/cpu_model/build_cpu_model.sh" "$OUT" 3261
    log "short all-C++ search (300 expansions) on it"
    (cd "$EXAMPLE" && python plan_peg_cpp.py --consts "$OUT/cpp_problem.npz" --model "$OUT/static_scene.mjb" \
        --max-expansions 300 --no-preemptive --out "$OUT/plan_smoke" | tail -12)
fi

cat <<EOF

[isaac_install] done.

Activate the environment (in any new shell):
    source $VENV/bin/activate

Isaac Sim asks you to accept its EULA on the first run (or set OMNI_KIT_ACCEPT_EULA=YES once you have read it).

Peg example ($EXAMPLE):
    # GPU planner on Isaac (PhysX default, or --backend newton)
    python plan_peg.py --start-idx 3261
    # all-C++ CPU planner: build the MuJoCo model once per start state, then search
    cpu_model/build_cpu_model.sh <model_dir> 3261
    python plan_peg_cpp.py --consts <model_dir>/cpp_problem.npz --model <model_dir>/static_scene.mjb --out <plan_dir>
    # render a plan in Isaac by state replay (front / side / wrist cameras)
    python render_plan.py --plan <plan_dir>/plan.npz --consts <model_dir>/cpp_problem.npz --out <video_dir>
EOF
