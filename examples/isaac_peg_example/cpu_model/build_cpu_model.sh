#!/usr/bin/env bash
# Build the CPU MuJoCo model + problem constants that plan_peg_cpp.py (the all-C++ planner, cpp/peg_mujoco.h) needs,
# for one OmniReset start state. Runs Isaac Sim headless on the GPU (Newton backend) three times, then MuJoCo on the CPU.
# UNREVIEWED.
#
#   ./build_cpu_model.sh <out_dir> [start_idx=3261] [reset_type=ObjectPartiallyAssembledEEGrasped]
#
# Writes into <out_dir>:
#   cpu_scene.mjb, gpu_edges.npz     record_gpu.py: Newton's MuJoCo model (world-0 nominal physics) + GPU reference edges
#   newton_scene.xml                 export_mjcf.py: the scene as MJCF (+ newton_contact_params.npz)
#   cpp_problem.npz                  export_cpp_problem.py: PegProblem constants and Isaac<->MuJoCo reference states
#   static_scene.mjb, static_map.npz build_static.py: the planner's model (hole / table / support as mocap bodies)
# Then:
#   python ../plan_peg_cpp.py --consts <out_dir>/cpp_problem.npz --model <out_dir>/static_scene.mjb --out <plan_dir>
set -euo pipefail

OUT=${1:?usage: build_cpu_model.sh <out_dir> [start_idx] [reset_type]}
START=${2:-3261}
RESET=${3:-ObjectPartiallyAssembledEEGrasped}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "[build_cpu_model] activate the Isaac venv first (isaac_install.sh prints the command)" >&2
    exit 1
fi

for step in record_gpu export_mjcf export_cpp_problem; do
    echo "[build_cpu_model] $step (log: $OUT/$step.log)"
    python "$HERE/$step.py" --out "$OUT" --start-idx "$START" --reset-type "$RESET" > "$OUT/$step.log" 2>&1 || {
        echo "[build_cpu_model] $step failed; last lines of $OUT/$step.log:" >&2
        tail -20 "$OUT/$step.log" >&2
        exit 1
    }
done
echo "[build_cpu_model] build_static"
python "$HERE/build_static.py" --dir "$OUT" > "$OUT/build_static.log" 2>&1 || { tail -20 "$OUT/build_static.log" >&2; exit 1; }
echo "[build_cpu_model] done: $OUT/static_scene.mjb, $OUT/cpp_problem.npz"
