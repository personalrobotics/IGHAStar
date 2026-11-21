#!/usr/bin/env bash
# Run a sweep script as N processes of T OpenMP threads.
# N defaults to floor(cpus / T). cpus is $SLURM_CPUS_PER_TASK, else nproc.
#
#   bash examples/standalone/sweep.sh \
#     --script examples/standalone/pusht_condition_sweep.py \
#     --config examples/standalone/Configs/pusht.yml \
#     --out Content/standalone/curve_plots/example \
#     --require Content/standalone/pusht_dino_problems.npz \
#     --repo /path/to/ighastar \
#     -- --problems ... --n 10 --max-expansions 1000
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
ORIG=("$0" "$@")

SCRIPT=""
CONFIG=""
OUT=""
THREADS=8
WORKERS=""
WAIT=20
SAVE_TREES=0
SKIP_CHECKS=0
MIN_FREE_GB=8
MIN_DISK_GB=5
REQUIRES=()
REPOS=()
PY_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --script) SCRIPT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --wait) WAIT="$2"; shift 2 ;;
    --save-trees) SAVE_TREES=1; shift ;;
    --skip-checks) SKIP_CHECKS=1; shift ;;
    --min-free-gb) MIN_FREE_GB="$2"; shift 2 ;;
    --min-free-disk-gb) MIN_DISK_GB="$2"; shift 2 ;;
    --require) REQUIRES+=("$2"); shift 2 ;;
    --repo) REPOS+=("$2"); shift 2 ;;
    --) shift; PY_ARGS=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

abspath() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf '%s\n' "$p"
  elif [[ -e "$p" ]]; then
    printf '%s\n' "$(cd "$(dirname "$p")" && pwd)/$(basename "$p")"
  else
    printf '%s\n' "$(pwd)/$p"
  fi
}

[[ -n "$SCRIPT" && -n "$CONFIG" && -n "$OUT" ]] || {
  echo "need --script, --config, and --out" >&2
  exit 1
}
((${#REPOS[@]} > 0)) || { echo "pass at least one --repo" >&2; exit 1; }
(( THREADS >= 1 && WAIT >= 0 && MIN_FREE_GB >= 1 && MIN_DISK_GB >= 1 )) || {
  echo "threads, wait, and free-space floors must be non-negative (floors >= 1)" >&2
  exit 1
}

SCRIPT="$(abspath "$SCRIPT")"
CONFIG="$(abspath "$CONFIG")"
OUT="$(abspath "$OUT")"
[[ -f "$SCRIPT" ]] || { echo "sweep script not found: $SCRIPT" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 1; }

if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  CPUS="$SLURM_CPUS_PER_TASK"
  CPU_SRC="SLURM_CPUS_PER_TASK"
else
  CPUS="$(nproc)"
  CPU_SRC="nproc"
fi
[[ "$CPUS" =~ ^[0-9]+$ && "$CPUS" -ge 1 ]] || {
  echo "bad cpu count from $CPU_SRC: ${CPUS:-empty}" >&2
  exit 1
}
OMP_THREADS="$THREADS"
if (( CPUS < OMP_THREADS )); then
  OMP_THREADS="$CPUS"
fi
if [[ -z "$WORKERS" ]]; then
  WORKERS=$(( CPUS / OMP_THREADS ))
  (( WORKERS >= 1 )) || WORKERS=1
fi
[[ "$WORKERS" =~ ^[0-9]+$ && "$WORKERS" -ge 1 ]] || {
  echo "bad --workers: $WORKERS" >&2
  exit 1
}

mkdir -p "$OUT"
probe="$OUT/.write_probe"
: >"$probe"
rm -f "$probe"

for req in "${REQUIRES[@]}"; do
  req="$(abspath "$req")"
  [[ -s "$req" ]] || { echo "missing or empty input: $req" >&2; exit 1; }
done

avail_ram="$(free -b | awk '/^Mem:/ {print $7}')"
[[ "$avail_ram" =~ ^[0-9]+$ ]] || { echo "could not read available RAM" >&2; exit 1; }
min_ram=$(( MIN_FREE_GB * 1024 * 1024 * 1024 ))
if (( avail_ram < min_ram )); then
  echo "available RAM ${avail_ram} bytes is below ${MIN_FREE_GB} GiB" >&2
  exit 1
fi
avail_disk="$(df -B1 --output=avail "$OUT" | awk 'NR==2 {print $1}')"
[[ "$avail_disk" =~ ^[0-9]+$ ]] || { echo "could not read free disk for $OUT" >&2; exit 1; }
min_disk=$(( MIN_DISK_GB * 1024 * 1024 * 1024 ))
if (( avail_disk < min_disk )); then
  echo "free disk ${avail_disk} bytes on $OUT is below ${MIN_DISK_GB} GiB" >&2
  exit 1
fi

repo_notes=()
for repo in "${REPOS[@]}"; do
  repo="$(abspath "$repo")"
  if ! rev="$(git -C "$repo" rev-parse HEAD 2>/dev/null)"; then
    echo "not a git checkout: $repo" >&2
    exit 1
  fi
  dirty="$(git -C "$repo" status --porcelain)"
  if [[ -n "$dirty" ]]; then
    state="dirty"
  else
    state="clean"
  fi
  repo_notes+=("$repo $rev $state")
  if [[ -n "$dirty" ]]; then
    while IFS= read -r line; do
      repo_notes+=("  $line")
    done <<<"$dirty"
  fi
done

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
COMMON=("$PYTHON" -u "$SCRIPT")
if (( SAVE_TREES == 1 )); then
  COMMON+=(--save-trees)
fi
COMMON+=("${PY_ARGS[@]}" --config "$CONFIG" --out "$OUT")

echo "cpus=$CPUS source=$CPU_SRC workers=$WORKERS omp=$OMP_THREADS ram_avail_bytes=$avail_ram disk_avail_bytes=$avail_disk"
dry="$OUT/dry_run.txt"
if ! "${COMMON[@]}" --dry-run >"$dry" 2>&1; then
  cat "$dry" >&2
  echo "dry-run failed" >&2
  exit 1
fi
cat "$dry"
TOTAL="$(awk '/^TOTAL / {t=$2} END {print t}' "$dry")"
[[ "$TOTAL" =~ ^[0-9]+$ && "$TOTAL" -ge 1 ]] || {
  echo "dry-run did not print TOTAL" >&2
  exit 1
}

manifest="$OUT/run_manifest.txt"
{
  echo "date $(date -Is)"
  echo "cpus $CPUS source $CPU_SRC"
  echo "workers $WORKERS omp $OMP_THREADS"
  echo "ram_avail_bytes $avail_ram"
  echo "disk_avail_bytes $avail_disk"
  echo "save_trees $SAVE_TREES"
  echo "total_results $TOTAL"
  echo "config_copy config.yml"
  echo "script $SCRIPT"
  printf 'argv'
  printf ' %q' "${ORIG[@]}"
  printf '\n'
  echo "python_args ${PY_ARGS[*]}"
  echo "repos:"
  printf '%s\n' "${repo_notes[@]}"
} >"$manifest"
cp "$CONFIG" "$OUT/config.yml"
echo "manifest $manifest"
cat "$manifest"

if (( SAVE_TREES == 0 )); then
  printf '\033[1;33mWARNING: TREES ARE NOT BEING SAVED\033[0m\n'
fi
if (( WAIT > 0 )); then
  echo "checks begin in ${WAIT}s (Ctrl-C to stop)"
  sleep "$WAIT"
fi

if (( SKIP_CHECKS == 0 )); then
  {
    echo "=== smoke ==="
    bash "$ROOT/examples/standalone/smoke.sh"
    echo "=== environment tests ==="
    "$PYTHON" "$ROOT/tests/unit/environments/test_environment.py" --env simple
    "$PYTHON" "$ROOT/tests/unit/environments/test_environment.py" --env kinematic --cpu
    "$PYTHON" "$ROOT/tests/unit/environments/test_environment.py" --env kinodynamic --cpu
    echo "=== pytest ==="
    # generic_torch compiles with AOTInductor, which this repo requires torch>=2.6 for.
    # Older torch still runs the generic (pybind) half of the environment tests.
    if "$PYTHON" -c "import torch; raise SystemExit(0 if hasattr(torch._inductor, 'aoti_compile_and_package') else 1)"; then
      "$PYTHON" -m pytest \
        "$ROOT/tests/unit/environments/test_generic_environment.py" \
        "$ROOT/tests/unit/planner/test_generic_torch_planner.py"
    else
      echo "torch has no AOTInductor; skipping generic_torch tests"
      "$PYTHON" -m pytest \
        "$ROOT/tests/unit/environments/test_generic_environment.py" \
        -k "not generic_torch and not parity and not random_sampler"
    fi
  } 2>&1 | tee "$OUT/preflight.log"
fi

echo "=== warmup ==="
"${COMMON[@]}" --warmup

pids=()
for ((i = 0; i < WORKERS; i++)); do
  log="$OUT/shard_${i}.log"
  {
    cat "$manifest"
    echo "--- dry-run ---"
    cat "$dry"
    echo "--- shard $i ---"
  } >"$log"
  OMP_NUM_THREADS="$OMP_THREADS" \
    "${COMMON[@]}" --shard-id "$i" --num-shards "$WORKERS" \
      >>"$log" 2>&1 &
  pids+=($!)
  echo "shard $i pid ${pids[$i]} → $log"
done

monitor="$OUT/monitor.log"
: >"$monitor"
SECONDS=0
last_line="rss_gb=0 cpu_pct=0 done=0/${TOTAL} eta=unknown"
while :; do
  alive=0
  rss_kb=0
  cpu_pct=0
  if ((${#pids[@]} > 0)); then
    IFS=','; pid_list="${pids[*]}"; unset IFS
    ps_text="$(ps -p "$pid_list" -o rss=,pcpu= 2>/dev/null || true)"
    if [[ -n "$ps_text" ]]; then
      alive=1
      read -r rss_kb cpu_pct < <(awk '{r+=$1; c+=$2} END {printf "%.0f %.0f\n", r+0, c+0}' <<<"$ps_text")
    fi
  fi
  done_n="$(find "$OUT" -mindepth 2 -maxdepth 2 -name 'p*.npz' -type f | wc -l)"
  done_n="${done_n// /}"
  if (( done_n > 0 && done_n < TOTAL )); then
    eta="$(( (SECONDS * (TOTAL - done_n)) / done_n ))s"
  elif (( done_n >= TOTAL && TOTAL > 0 )); then
    eta=0s
  else
    eta=unknown
  fi
  rss_gb="$(awk -v kb="$rss_kb" 'BEGIN {printf "%.2f", kb/1048576}')"
  now="$(date -Is)"
  if (( SECONDS % 10 < 2 || alive == 0 )); then
    last_line="$now rss_gb=$rss_gb cpu_pct=$cpu_pct done=${done_n}/${TOTAL} eta=$eta"
    echo "$last_line" | tee -a "$monitor"
  fi
  if (( alive == 0 )); then
    break
  fi
  sleep 2
done

ec=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "shard $i failed (pid ${pids[$i]}); see $OUT/shard_${i}.log" >&2
    ec=1
  else
    echo "shard $i done"
  fi
  echo "$last_line" >>"$OUT/shard_${i}.log"
done
echo "monitor $monitor"
exit "$ec"
