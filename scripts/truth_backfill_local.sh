#!/bin/zsh
# Detached truth backfill in yearly chunks via IEM: scripts/truth_backfill_local.sh <start> <end>
set -uo pipefail
cd "$(dirname "$0")/.."
START=$1; END=$2
y=${START:0:4}; ye=${END:0:4}
while [[ $y -le $ye ]]; do
  s="$y-01-01"; e="$y-12-31"
  [[ $y == ${START:0:4} ]] && s=$START
  [[ $y == ${END:0:4} ]] && e=$END
  echo "=== truth $s..$e $(date -u +%H:%M:%SZ)"
  PYTHONPATH=. .venv/bin/python -m castcheck.cli truth-backfill "$s" "$e"
  y=$((y+1))
done
echo "truth backfill done $(date -u +%H:%M:%SZ)"
