#!/usr/bin/env bash
# Defeat vm.panic_on_oom=1 by hard-killing vllm cluster-wide BEFORE free memory
# hits the floor.  DGX Spark uses unified memory (121 GiB CPU+GPU shared); an OOM
# there triggers a kernel panic that reboots the node within ~50 ms — no userland
# log, and because both ranks do collective ops, one node's panic wedges the
# other.  Losing both nodes costs a ~5 min cold start, so we pre-empt.
#
# Auto-started by 03-start-serve.sh.  Run standalone in a spare terminal too if
# you like.  Polls every $SLEEP_S; trips below $WATCHDOG_THRESHOLD_MIB on any node.

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
    line="$(date +%T) free MiB:"
    for node in "${NODE_ARR[@]}"; do line+=" ${node##*.}=${FREE[$node]}"; done
    echo "$line"
    LAST_REPORT=$NOW
  fi

  for node in "${NODE_ARR[@]}"; do
    v="${FREE[$node]}"
    if [ "$v" != "?" ] && [ "$v" -lt "$THRESH" ] 2>/dev/null; then
      echo "$(date +%T) !!! WATCHDOG TRIPPED on $node: free=${v}MiB < ${THRESH}MiB — killing vllm cluster-wide"
      for n in "${NODE_ARR[@]}"; do
        ssh_node "$n" "docker exec '$CONTAINER_NAME' pkill -9 -f vllm 2>/dev/null" >/dev/null 2>&1 || true
      done
      exit 1
    fi
  done

  sleep "$SLEEP_S"
done
