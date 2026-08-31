#!/bin/zsh
# Local backup of the GitHub Actions pipeline (launchd). Safe to run anytime: every step is idempotent.
#
# Failure handling: the script does not abort on the first error, because a failed fetch must not stop
# the site from being rebuilt from what we already have. Instead every step records its outcome, and a
# non-empty run writes data/raw/last_failure.txt (read by the status page and by the operator) plus a
# macOS notification. A clean run removes that file.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

LOG_DIR="data/raw"
mkdir -p "$LOG_DIR"
FAILURE_FILE="$LOG_DIR/last_failure.txt"
RUN_LOG="$LOG_DIR/run_daily.log"
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
failures=()

step() {   # step <name> <command...>
  local name=$1; shift
  echo "=== $name $(date -u +%H:%M:%SZ)" >>"$RUN_LOG"
  if "$@" >>"$RUN_LOG" 2>&1; then
    return 0
  fi
  echo "$name FAILED (exit $?)" >>"$RUN_LOG"
  failures+=("$name")
  return 1
}

: >"$RUN_LOG"
git pull --rebase --quiet || true

step fetch-latest .venv/bin/castcheck fetch-latest --workers 2
step truth        .venv/bin/castcheck truth
step truth-instant .venv/bin/castcheck truth-instant
step truth-instant-iem .venv/bin/castcheck truth-instant-backfill "$(date -u -v-10d +%F)" "$(date -u -v-2d +%F)"
step truth-qc     .venv/bin/castcheck truth-qc --start "$(date -u -v-15d +%F)"
step daily        .venv/bin/castcheck daily
step deploy       npx --yes wrangler pages deploy public --project-name castcheck --branch main --commit-dirty=true
step commit       scripts/commit_data.sh "[data] local $STAMP"

if (( ${#failures[@]} )); then
  printf '%s\n' "$STAMP  failed steps: ${failures[*]}" >"$FAILURE_FILE"
  tail -40 "$RUN_LOG" >>"$FAILURE_FILE"
  osascript -e "display notification \"${failures[*]}\" with title \"CastCheck daily failed\"" 2>/dev/null || true
  echo "run_daily: FAILED (${failures[*]}) — see $FAILURE_FILE" >&2
  exit 1
fi

rm -f "$FAILURE_FILE"
echo "run_daily: ok $STAMP"
