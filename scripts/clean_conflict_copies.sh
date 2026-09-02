#!/bin/zsh
# Remove macOS/iCloud "conflict copies" ("name 2.ext", "dir 2") anywhere in the repo except .venv.
# They are never legitimate here: shards are content-addressed by path, and store._is_shard ignores
# them on read, but they still bloat the tree, confuse git and can be deployed by mistake.
cd "$(dirname "$0")/.."
find . -path ./.venv -prune -o -name "* *" -print -exec rm -rf {} + 2>/dev/null
