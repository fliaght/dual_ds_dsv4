#!/usr/bin/env bash
# Best-effort cluster memory watcher. Polls MemAvailable, not MemFree.
# This is not an allocation limit: polling/SSH can miss fast UMA peaks.
# See Docs/kedb/memory.md. Auto-started by 03-start-serve.sh.

source "$(dirname "$0")/_lib.sh"

THRESH="${WATCHDOG_THRESHOLD_MIB:-4096}"
SLEEP_S="${SLEEP_S:-15}"
LAST_REPORT=0

echo "$(date +%T) watchdog: threshold=${THRESH} MiB across $N_NODES nodes (every ${SLEEP_S}s)"

while true; do
  declare -A FREE
  for node in "${NODE_ARR[@]}"; do
    FREE[$node]=$(ssh_node "$node" "free -m | awk '/^Mem:/{print \$7}'" 2>/dev/null || echo "?")
  done

  NOW=$(date +%s)
  if [ $(( NOW - LAST_REPORT )) -ge 30 ]; then
    line="$(date +%T) available MiB:"
    for node in "${NODE_ARR[@]}"; do line+=" ${node##*.}=${FREE[$node]}"; done
    echo "$line"
    LAST_REPORT=$NOW
  fi

  for node in "${NODE_ARR[@]}"; do
    v="${FREE[$node]}"
    if [ "$v" != "?" ] && [ "$v" -lt "$THRESH" ] 2>/dev/null; then
      echo "$(date +%T) !!! WATCHDOG TRIPPED on $node: available=${v}MiB < ${THRESH}MiB — killing vllm cluster-wide"
      for n in "${NODE_ARR[@]}"; do
        ssh_node "$n" "docker exec '$CONTAINER_NAME' pkill -9 -f vllm 2>/dev/null" >/dev/null 2>&1 || true
      done
      exit 1
    fi
  done

  sleep "$SLEEP_S"
done
