#!/usr/bin/env bash
# Commit and push data/ from a workflow, retrying the rebase-push race.
#
# Every data-writing workflow shares the `data-writes` concurrency group, so two of *ours* never run
# at once — but a human push (or a queued run that starts the moment the group frees up) can still
# land between our `git pull --rebase` and our `git push`. The loop below retries a few times;
# parquet files cannot be merged, so a rebase conflict is resolved by keeping *our* freshly written
# shard (upserts are idempotent, so re-running the command reproduces the same file).
#
# Usage: scripts/commit_data.sh "[data] fetch 2026-08-30T05:00Z" [path ...]
set -euo pipefail

MESSAGE=${1:?commit message required}
shift
PATHS=("$@")
[[ ${#PATHS[@]} -eq 0 ]] && PATHS=(data)

git config user.name "castcheck-bot"
git config user.email "bot@castcheck.zifanzhang.com"
git add -- "${PATHS[@]}"
if git diff --cached --quiet; then
  echo "no data changes to commit"
  exit 0
fi
git commit -q -m "$MESSAGE"

for attempt in 1 2 3 4 5; do
  if git push --quiet; then
    echo "pushed on attempt $attempt"
    exit 0
  fi
  echo "push rejected (attempt $attempt); rebasing onto origin"
  git fetch --quiet origin "$(git rev-parse --abbrev-ref HEAD)"
  if ! git pull --rebase --quiet; then
    # Binary shards cannot be merged, so keep what this run just computed. During a rebase the
    # commit being replayed is "theirs" (HEAD is upstream), so --theirs is our own shard — this is
    # the opposite of what it means during a merge, and getting it backwards would silently discard
    # the run's work. Safe either way, because upserts are idempotent: the shard we keep already
    # contains the rows we just fetched, and any rows only upstream had come back on the next run.
    git checkout --theirs -- "${PATHS[@]}" 2>/dev/null || true
    git add -- "${PATHS[@]}"
    GIT_EDITOR=true git rebase --continue || git rebase --abort
  fi
  sleep $((attempt * 5))
done

echo "could not push after 5 attempts" >&2
exit 1
