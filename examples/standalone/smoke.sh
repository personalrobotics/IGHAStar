#!/usr/bin/env bash
# Sequential planner smokes. Extensions that share a .so must not build in parallel.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$(cd ../.. && pwd)/ighastar${PYTHONPATH:+:$PYTHONPATH}"

run() {
  local name="$1"
  shift
  echo "=== ${name} ==="
  set +e
  "$@"
  local status=$?
  set -e
  if [[ "${status}" -ne 0 ]]; then
    echo "FAILED: ${name} (exit ${status})" >&2
    echo "command: $*" >&2
    exit "${status}"
  fi
}

run kinematic python3 -u example.py --no-visuals -c Configs/kinematic_example.yml
run kinodynamic python3 -u example.py --no-visuals -c Configs/kinodynamic_example.yml
run simple python3 -u example.py --no-visuals -c Configs/simple_example.yml
run pusht python3 -u pusht_example.py --no-video
run pusht-qs python3 -u pusht_qs_selfcheck.py
run generic-integrator python3 -u generic_integrator.py
run latent python3 -u generic_latent_planning_smoke.py
echo "=== all smokes passed ==="
