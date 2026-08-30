#!/bin/zsh
# Local backup of the GitHub Actions pipeline (launchd). Safe to run anytime: every step is idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
git pull --rebase --quiet || true
.venv/bin/castcheck fetch-latest --workers 2
.venv/bin/castcheck truth || true
.venv/bin/castcheck daily
npx --yes wrangler pages deploy public --project-name castcheck --branch main --commit-dirty=true >/dev/null
git add data && (git diff --cached --quiet || git commit -qm "[data] local $(date -u +%Y-%m-%dT%H:%MZ)") && git push --quiet || true
