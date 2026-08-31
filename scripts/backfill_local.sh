#!/bin/zsh
# Detached local backfill: scripts/backfill_local.sh <start> <end> [model ...]
#
# One model at a time, one run at a time (--workers 1): the point of the local backfill is to be
# gentle on the upstream mirrors, which throttle bursts of range GETs. A failure in one model is
# logged and the loop continues with the next.
set -uo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 2 ]]; then
  echo "usage: $0 <start YYYY-MM-DD> <end YYYY-MM-DD> [model ...]" >&2
  exit 2
fi
START=$1; END=$2; shift 2
MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=(ifs_hres aifs_single gfs graphcast_ifs graphcast_gfs pangu_ifs pangu_gfs
          fourcastnet_ifs fourcastnet_gfs aurora_ifs aurora_gfs)
fi
failed=()
for m in "${MODELS[@]}"; do
  echo "=== $m $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if ! PYTHONPATH=. .venv/bin/python -m castcheck.cli backfill "$m" "$START" "$END" --workers 1; then
    echo "!!! $m failed" >&2
    failed+=("$m")
  fi
done
echo "backfill done $(date -u +%Y-%m-%dT%H:%M:%SZ)${failed:+ (failed: ${failed[*]})}"
(( ${#failed[@]} == 0 ))
