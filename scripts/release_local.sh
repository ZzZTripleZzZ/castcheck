#!/bin/zsh
# Deploy the already-built public/ to production, publish datasets, commit data. Detached-friendly.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
SNAP=$(mktemp -d)/public; mkdir -p "$SNAP"; rsync -a --delete public/ "$SNAP"/   # deploy a snapshot so a concurrent rebuild cannot pull files from under wrangler
echo "=== deploy $(date -u +%H:%M:%SZ) ($(find "$SNAP" -type f | wc -l | tr -d ' ') files)"; npx --yes wrangler pages deploy "$SNAP" --project-name castcheck --branch "${BRANCH:-main}" --commit-dirty=true 2>&1 | grep -E "Deployment complete|alias URL|Error|error" | tail -2; rm -rf "$(dirname "$SNAP")"
echo "=== hf"; PYTHONPATH=. .venv/bin/python -m castcheck.cli publish hf 2>&1 | grep -E "ok in|ERROR|Error" | tail -1
echo "=== kaggle"; PYTHONPATH=. .venv/bin/python -m castcheck.cli publish kaggle 2>&1 | grep -vE "%\|" | tail -1
echo "=== commit"; git add -A data castcheck scripts && git -c user.name="Zifan Zhang" -c user.email="126985627+ZzZTripleZzZ@users.noreply.github.com" commit -q -m "[data] v0.3 derived tables and scores (restored)" ; git pull -q --rebase && git push -q 2>&1 | tail -1; git status -sb | head -1
echo "=== done $(date -u +%H:%M:%SZ)"
