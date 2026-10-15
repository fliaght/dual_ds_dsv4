#!/usr/bin/env bash
# Offline checks only. Does not read cluster.conf or start any model service.
set -euo pipefail
CHECK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CHECK_ROOT"

if [ -n "${CHECK_PYTHON:-}" ]; then
  PYTHON="$CHECK_PYTHON"
elif [ -x .venv/bin/python ]; then
  PYTHON="$CHECK_ROOT/.venv/bin/python"
elif [ -x chat/.venv/bin/python ]; then
  PYTHON="$CHECK_ROOT/chat/.venv/bin/python"
else
  PYTHON=python3
fi
if ! "$PYTHON" -c 'import aiohttp, pyflakes, requests, streamlit' >/dev/null 2>&1; then
  echo 'Missing check dependencies. See Docs/runbooks/checks.md and requirements-dev.txt.' >&2
  exit 1
fi
for CHECK_TOOL in jq rg; do
  command -v "$CHECK_TOOL" >/dev/null || { echo "Required check tool missing: $CHECK_TOOL" >&2; exit 1; }
done

echo 'Checking shell syntax...'
while IFS= read -r file; do bash -n "$file"; done < <(
  rg --files scripts config bench chat -g '*.sh' -g '*.env' -g '!.venv/**'
)
bash -n cluster.conf.example
echo 'Checking Python syntax and imports...'
"$PYTHON" -m compileall -q scripts bench tests chat -x '/(\.venv|__pycache__)/'
mapfile -t PY_FILES < <(rg --files scripts bench tests chat -g '*.py' -g '!.venv/**')
"$PYTHON" -m pyflakes "${PY_FILES[@]}"
"$PYTHON" scripts/check_docs.py

# Ignore inherited live-test/remote-monitor switches: this command is always offline.
export CHAT_LIVE_TESTS=0 CHAT_MONITOR_TEST_HOSTS=""
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" -m unittest discover -s chat/tests -v
echo 'Offline checks passed. No live model tests were run.'
