#!/usr/bin/env bash
# Stop the vLLM service on all nodes.
#   bash scripts/04-stop.sh         # SIGTERM→SIGKILL vllm, leave containers
#   bash scripts/04-stop.sh --rm    # also docker rm -f the containers (recommended:
#                                     drops PyTorch allocator state for a clean restart)

source "$(dirname "$0")/_lib.sh"

RM_CONTAINER=false
[ "${1:-}" = "--rm" ] && RM_CONTAINER=true

# Kill host-side watchdog.
pkill -f "scripts/oom-watchdog.sh" 2>/dev/null || true

section "Stopping vllm on all nodes"
for_each_node "docker exec '$CONTAINER_NAME' bash -lc '
pkill -TERM -f vllm 2>/dev/null || true
sleep 5
pkill -9 -f vllm 2>/dev/null || true
echo stopped' 2>&1 | tail -1"

if $RM_CONTAINER; then
  section "Removing containers"
  for_each_node "docker rm -f '$CONTAINER_NAME' 2>&1 | tail -1"
fi

echo
echo "Verifying free memory + port released..."
sleep 3
for_each_node "free -h | awk 'NR==2 {print \"Mem used=\" \$3 \" available=\" \$7}'"
ss -ltn 2>/dev/null | grep -q ":$API_PORT " && echo "port $API_PORT still LISTEN" || echo "port $API_PORT freed"

echo "✅ Stop complete."
