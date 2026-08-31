#!/bin/zsh
# Detached truth backfill in yearly chunks via IEM: scripts/truth_backfill_local.sh <start> <end>
#
# Yearly chunks keep each upsert (and each parquet rewrite) bounded, and mean a crash costs at most
# one year of re-fetching. A failed year is reported at the end instead of aborting the range.
set -uo pipefail
cd "$(dirname "$0")/.."
if [[ $# -ne 2 ]]; then
  echo "usage: $0 <start YYYY-MM-DD> <end YYYY-MM-DD>" >&2
  exit 2
fi
START=$1; END=$2
y=${START:0:4}; ye=${END:0:4}
failed=()
while [[ $y -le $ye ]]; do
  s="$y-01-01"; e="$y-12-31"
  [[ $y == ${START:0:4} ]] && s=$START
  [[ $y == ${END:0:4} ]] && e=$END
  echo "=== truth $s..$e $(date -u +%H:%M:%SZ)"
  if ! PYTHONPATH=. .venv/bin/python -m castcheck.cli truth-backfill "$s" "$e"; then
    echo "!!! $y failed" >&2
    failed+=("$y")
  fi
  y=$((y+1))
done
echo "truth backfill done $(date -u +%H:%M:%SZ)${failed:+ (failed: ${failed[*]})}"
(( ${#failed[@]} == 0 ))
