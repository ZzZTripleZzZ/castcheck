#!/usr/bin/env bash
# Commit and push data/ from a workflow, retrying the rebase-push race.
#
# Every data-writing workflow shares the `data-writes` concurrency group, so two of *ours* never run
# at once — but a human push (or a queued run that starts the moment the group frees up) can still
# land between our `git pull --rebase` and our `git push`. The loop below retries a few times;
# Conflicted parquet shards are union-merged with the store's upsert semantics (castcheck.merge), so
# neither writer's rows are lost.
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
    # Parquet shards conflict as binaries, but both sides only *added* rows, so we union-merge them
    # with the store's upsert semantics (castcheck.merge). During a rebase the upstream version is
    # stage 2 (:2:) and the commit being replayed — our run — is stage 3 (:3:).
    PY=${CASTCHECK_PY:-.venv/bin/python}
    for f in $(git diff --name-only --diff-filter=U); do
      case "$f" in
        *.parquet)
          tmpd=$(mktemp -d)
          git show ":2:$f" > "$tmpd/upstream.parquet" 2>/dev/null || true
          git show ":3:$f" > "$tmpd/ours.parquet" 2>/dev/null || true
          if [[ -s "$tmpd/upstream.parquet" && -s "$tmpd/ours.parquet" ]]; then
            PYTHONPATH=. "$PY" -m castcheck.merge "$tmpd/ours.parquet" "$tmpd/upstream.parquet" "$f"
          else
            git checkout --theirs -- "$f" 2>/dev/null || true   # one side deleted/added: keep ours
          fi
          rm -rf "$tmpd"
          ;;
        *) git checkout --theirs -- "$f" 2>/dev/null || true ;;
      esac
      git add -- "$f"
    done
    GIT_EDITOR=true git rebase --continue || git rebase --abort
  fi
  sleep $((attempt * 5))
done

echo "could not push after 5 attempts" >&2
exit 1
