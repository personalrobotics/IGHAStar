#!/usr/bin/env bash
# Run the PushT hyperparam sweep as NUM_SHARDS parallel workers.
# Default: 3 workers × 8 OpenMP threads = 24 hardware threads.
#
#   bash examples/standalone/pusht_hyperparam_sweep_parallel.sh
#   NUM_SHARDS=3 OMP_THREADS=8 bash examples/standalone/pusht_hyperparam_sweep_parallel.sh
#   EXTRA_ARGS='--n-problems 10 --no-traj' bash ...
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
SWEEP=examples/standalone/pusht_hyperparam_sweep.py
CONTENT=Content/standalone
NUM_SHARDS="${NUM_SHARDS:-3}"
OMP_THREADS="${OMP_THREADS:-8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$CONTENT"

# Materialize shared problem set once (before workers race on it).
"$PYTHON" - "$ROOT/$SWEEP" <<'PY'
import importlib.util, os, sys
sweep_py = sys.argv[1]
spec = importlib.util.spec_from_file_location("pusht_sweep", sweep_py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
path = os.path.join(mod.CONTENT, "pusht_dino_problems.npz")
if not os.path.isfile(path):
    mod.generate_problem_set(path)
else:
    print(f"problems ok: {path}")
PY

echo "launching ${NUM_SHARDS} shards × OMP_NUM_THREADS=${OMP_THREADS}"
pids=()
for ((i = 0; i < NUM_SHARDS; i++)); do
  log="$CONTENT/pusht_sweep_shard${i}.log"
  echo "  shard $i → $log"
  # shellcheck disable=SC2086
  OMP_NUM_THREADS="$OMP_THREADS" \
    "$PYTHON" "$SWEEP" \
      --omp-threads "$OMP_THREADS" \
      --shard-id "$i" \
      --num-shards "$NUM_SHARDS" \
      $EXTRA_ARGS \
    >"$log" 2>&1 &
  pids+=($!)
done

ec=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "shard $i failed (pid ${pids[$i]}); see $CONTENT/pusht_sweep_shard${i}.log" >&2
    ec=1
  else
    echo "shard $i done"
  fi
done

# Merge whatever shard CSVs this run produced (match --out stem from EXTRA_ARGS or default).
OUT_STEM="${OUT_STEM:-pusht_sweep_qs_oat_50k_results}"
OUT="$CONTENT/${OUT_STEM}.csv"
shopt -s nullglob
mapfile -t shards < <(printf '%s\n' "$CONTENT"/${OUT_STEM}_shard*.csv | sort)
if ((${#shards[@]} == 0)); then
  echo "no shard CSVs matching ${OUT_STEM}_shard*.csv" >&2
  exit 1
fi
head -n 1 "${shards[0]}" >"$OUT"
for f in "${shards[@]}"; do
  tail -n +2 "$f" >>"$OUT"
done
echo "merged ${#shards[@]} shards → $OUT  ($(wc -l <"$OUT") lines incl. header)"
echo "per-shard traj dirs: ${CONTENT}/pusht_sweep_qs_oat_50k_traj_shard*/"
exit "$ec"
