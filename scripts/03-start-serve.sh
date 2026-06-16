#!/usr/bin/env bash
# Boot the cross-node vLLM service.  Run from the HEAD node.
#
# Order matters: workers (--headless) must be up and waiting at
# master-addr:HEAD_PORT BEFORE rank 0 connects, or the rendezvous can race.
# So this launches every worker first, then rank 0 locally.
#
# After this:
#   - API at http://127.0.0.1:$API_PORT/v1/chat/completions on the head
#   - Per-rank logs tee'd to $LOG_DIR/vllm-dsv4-rank<N>.log
#   - OOM watchdog running in the background (kills vllm if free MiB < threshold)
#   - NCCL transport verified to be RDMA (via NET/IB), not socket fallback

source "$(dirname "$0")/_lib.sh"

if ! is_leader; then
  echo "ERROR: run this from the HEAD node ($LEADER); this host isn't it." >&2
  exit 1
fi

HEAD_LOG="$LOG_DIR/vllm-dsv4-rank0.log"

section "Sync run-node.sh + config to workers (REMOTE_STAGE=$REMOTE_STAGE)"
for node in "${NODE_ARR[@]:1}"; do
  push_node "$node"
  echo "[$node] staged"
done

section "Tear down any stale $CONTAINER_NAME on all nodes"
for_each_node "docker rm -f '$CONTAINER_NAME' 2>&1 | tail -1 || true"

# Start the OOM watchdog before we allocate memory.
TS=$(date +%Y%m%d_%H%M%S)
WATCHDOG_LOG="$LOG_DIR/watchdog-${TS}.log"
pkill -f "scripts/oom-watchdog.sh" 2>/dev/null || true
nohup bash "$SCRIPT_DIR/oom-watchdog.sh" > "$WATCHDOG_LOG" 2>&1 &
echo "Watchdog started → $WATCHDOG_LOG (kills vllm if free MiB < $WATCHDOG_THRESHOLD_MIB)"

section "Boot WORKERS first (ranks $((N_NODES-1))..1)"
for (( rank=N_NODES-1; rank>=1; rank-- )); do
  node="${NODE_ARR[$rank]}"
  echo "── rank $rank on $node"
  ssh_node "$node" "NODE_RANK=$rank RECIPE='$RECIPE' bash '$REMOTE_STAGE/scripts/run-node.sh'"
done

section "Boot HEAD (rank 0) locally"
NODE_RANK=0 RECIPE="$RECIPE" bash "$SCRIPT_DIR/run-node.sh"

section "Waiting for API on :$API_PORT (cold start ~5-6 min)"
deadline=$(( $(date +%s) + 600 ))
ready=false
while [ "$(date +%s)" -lt "$deadline" ]; do
  if grep -q "Application startup complete" "$HEAD_LOG" 2>/dev/null \
     && ss -ltn 2>/dev/null | grep -q ":$API_PORT "; then
    ready=true; break
  fi
  if grep -qE "Segmentation fault|exit code 139|RuntimeError|CUDA error|Engine.*died|raise .*Error" "$HEAD_LOG" 2>/dev/null; then
    echo "❌ Startup error in $HEAD_LOG:"; tail -25 "$HEAD_LOG"
    echo "(worker log on the other node: $REMOTE_STAGE-side /host_logs/vllm-dsv4-rank1.log)"
    exit 1
  fi
  sleep 5
done
$ready || { echo "❌ Timeout. Tail of head log:"; tail -25 "$HEAD_LOG"; exit 1; }
echo "✅ Service ready: http://127.0.0.1:$API_PORT"

section "Verify cross-node TP is on RDMA (via NET/IB, not socket fallback)"
# Count only the unambiguous fallback signal: a channel routed `via NET/Socket`.
# Do NOT match bare 'MNNVL'/'gdaki' — healthy boots print benign status lines like
# 'NCCL_MNNVL_ENABLE set by environment to 0' and 'comm ... MNNVL 0', which are
# the features being DISABLED (what we want), not fallback. A real gdaki failure
# segfaults and is already caught by the startup-error grep above.
ib=$(grep -cE 'via NET/IB/[01]' "$HEAD_LOG" 2>/dev/null || echo 0)
sock=$(grep -cE 'via NET/Socket' "$HEAD_LOG" 2>/dev/null || echo 0)
echo "  via NET/IB channels: $ib   |   via NET/Socket (fallback): $sock"
if [ "$ib" -gt 0 ] && [ "$sock" -eq 0 ]; then
  echo "  ✅ PASS — cross-node TP is on dual RoCE."
else
  echo "  ⚠️  Expected via NET/IB/0 + via NET/IB/1 and zero socket fallback — inspect $HEAD_LOG."
fi

section "Pre-warm (kills first-request JIT/autotune spikes; best-effort)"
python3 "$SCRIPT_DIR/prewarm.py" --url "http://127.0.0.1:$API_PORT/v1/chat/completions" --model "$MODEL" \
  || echo "  (prewarm skipped/failed — service is still up)"

echo
echo "Try: bash bench/smoke.sh        # 5-prompt sanity (expects 391)"
echo "Or:  bash bench/run_full_bench.sh"
