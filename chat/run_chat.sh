#!/usr/bin/env bash
# Launch the multi-user Streamlit chat UI for the dual-Spark vLLM service (port 9450).
#
#   bash chat/run_chat.sh              # foreground (Ctrl-C to stop)
#   bash chat/run_chat.sh --daemon     # background; log -> $LOG_DIR/chat-ui.log
#   bash chat/run_chat.sh --stop       # stop the daemon
#   bash chat/run_chat.sh --status
#
# Defaults are derived from cluster.conf when present (API_PORT, MODEL, LOG_DIR,
# and the active recipe's MAX_NUM_SEQS -> CHAT_MAX_CONCURRENCY).  Any CHAT_*
# variable already in the environment wins.  See chat/README.md for the full list.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
VENV="$HERE/.venv"

# ---- defaults from cluster.conf ---------------------------------------------
API_PORT=8000; MODEL=""; LOG_DIR="$HOME/dsv4_logs"
source "$ROOT/scripts/_config.sh"
load_cluster_config optional
export CHAT_API_BASE="${CHAT_API_BASE:-http://127.0.0.1:${API_PORT}/v1}"
export CHAT_MODEL="${CHAT_MODEL:-$MODEL}"
export CHAT_MAX_CONCURRENCY="${CHAT_MAX_CONCURRENCY:-$MAX_NUM_SEQS}"
if [ "$RUNTIME" = v41 ]; then
  export CHAT_TITLE="${CHAT_TITLE:-DeepSeek-V4.1-Flash EXL3 on 2× DGX Spark}"
  export CHAT_CONTEXT_TOKEN_BUDGET="${CHAT_CONTEXT_TOKEN_BUDGET:-$((MAX_MODEL_LEN - 16384))}"
  # Fixed KV pool: reply length costs time (~24 tok/s), not memory.  A ~500K-token
  # uncached prefill sends nothing for ~10+ min, so wait longer than the 900 s default.
  export CHAT_MAX_TOKENS_CAP="${CHAT_MAX_TOKENS_CAP:-65536}"
  export CHAT_READ_TIMEOUT="${CHAT_READ_TIMEOUT:-1800}"
fi
export CHAT_LOG_DIR="${CHAT_LOG_DIR:-$LOG_DIR/chat}"
# Shown in the page's bottom-left corner: repository VERSION and commit ("-dirty": chat/ differs from it).
if [ -z "${CHAT_VERSION:-}" ]; then
  CHAT_VERSION="v$(cat "$ROOT/VERSION" 2>/dev/null || echo '?')"
  if commit=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null); then
    git -C "$ROOT" diff --quiet HEAD -- chat 2>/dev/null || commit="$commit-dirty"
    CHAT_VERSION="$CHAT_VERSION · $commit"
  fi
  export CHAT_VERSION
fi
# Performance panel: every node in cluster.conf (head first), same ssh key + watchdog line.
export CHAT_MONITOR_HOSTS="${CHAT_MONITOR_HOSTS:-${NODES:-}}"
export CHAT_MONITOR_SSH_KEY="${CHAT_MONITOR_SSH_KEY:-${SSH_KEY:-}}"
export CHAT_MONITOR_WATCHDOG_MIB="${CHAT_MONITOR_WATCHDOG_MIB:-${WATCHDOG_THRESHOLD_MIB:-4096}}"
CHAT_BIND="${CHAT_BIND:-0.0.0.0}"
CHAT_PORT="${CHAT_PORT:-9450}"
mkdir -p "$LOG_DIR"
PIDFILE="$LOG_DIR/chat-ui.pid"
UILOG="$LOG_DIR/chat-ui.log"

# ---- venv ---------------------------------------------------------------------
ensure_venv() {
  if [ ! -x "$VENV/bin/python" ]; then
    echo "Creating venv at $VENV ..."
    python3 -m venv "$VENV"
  fi
  if ! "$VENV/bin/python" -c 'import streamlit, requests' 2>/dev/null; then
    echo "Installing requirements ..."
    "$VENV/bin/python" -m pip install -q --upgrade pip
    "$VENV/bin/python" -m pip install -q -r "$HERE/requirements.txt"
  fi
}

is_running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-}" in
  --stop)
    if is_running; then
      kill "$(cat "$PIDFILE")" && rm -f "$PIDFILE" && echo "chat UI stopped."
    else
      rm -f "$PIDFILE"; echo "chat UI not running."
    fi
    exit 0 ;;
  --status)
    if is_running; then echo "chat UI running (pid $(cat "$PIDFILE")) on http://${CHAT_BIND}:${CHAT_PORT}  log: $UILOG"
    else echo "chat UI not running"; fi
    exit 0 ;;
  --daemon)
    ensure_venv
    if is_running; then echo "already running (pid $(cat "$PIDFILE"))"; exit 0; fi
    cd "$HERE"
    nohup "$VENV/bin/streamlit" run app.py \
        --server.address "$CHAT_BIND" --server.port "$CHAT_PORT" --server.headless true \
        > "$UILOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    sleep 3
    if is_running; then
      echo "chat UI started (pid $(cat "$PIDFILE"))"
      echo "  URL:  http://$(hostname -I | awk '{print $1}'):${CHAT_PORT}   (bound to ${CHAT_BIND})"
      echo "  API:  $CHAT_API_BASE   model: ${CHAT_MODEL:-<auto>}   queue concurrency: $CHAT_MAX_CONCURRENCY"
      echo "  conversation log: $CHAT_LOG_DIR   ui log: $UILOG"
    else
      echo "failed to start — see $UILOG" >&2; tail -20 "$UILOG" >&2; exit 1
    fi
    exit 0 ;;
  ""|--run)
    ensure_venv
    cd "$HERE"
    echo "chat UI: http://${CHAT_BIND}:${CHAT_PORT}  API=$CHAT_API_BASE  concurrency=$CHAT_MAX_CONCURRENCY  logs=$CHAT_LOG_DIR"
    exec "$VENV/bin/streamlit" run app.py \
        --server.address "$CHAT_BIND" --server.port "$CHAT_PORT" --server.headless true ;;
  *)
    echo "usage: $0 [--run | --daemon | --stop | --status]" >&2; exit 2 ;;
esac
