#!/bin/zsh
# Detached local backfill: scripts/backfill_local.sh <start> <end> [model ...]
set -uo pipefail
cd "$(dirname "$0")/.."
START=$1; END=$2; shift 2
MODELS=("$@"); [[ ${#MODELS[@]} -eq 0 ]] && MODELS=(ifs_hres aifs_single gfs graphcast_ifs graphcast_gfs pangu_ifs pangu_gfs fourcastnet_ifs fourcastnet_gfs aurora_ifs aurora_gfs)
for m in "${MODELS[@]}"; do
  echo "=== $m $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  PYTHONPATH=. .venv/bin/python -m castcheck.cli backfill "$m" "$START" "$END" --workers 1
done
echo "backfill done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
