#!/usr/bin/env bash
# Pre-flight checks before any container starts.  Verifies, on every node:
#   - SSH reachability from the head
#   - Docker daemon reachable
#   - All RoCE NICs in $IFACES present and Up
#   - RoCE addressing: each IFACES NIC has an IPv4, and each NODES IP is bound to
#     a RoCE iface (NOT a management NIC) — the load-bearing NCCL assumption
#   - RoCE NIC names match across nodes
#   - NCCL master port ($HEAD_PORT) free on the head + fabric reachable
#   - Model snapshot present under $HF_CACHE
#   - Shared SSH key readable
#   - Enough free UMA headroom
# Exit 0 iff every check passes on every node.
#
# New to a pair and don't know NODES/IFACES?  Run `bash scripts/discover.sh` first.

source "$(dirname "$0")/_lib.sh"

section "Pre-flight across $N_NODES nodes (leader=$LEADER, model=$MODEL, recipe=$RECIPE)"
ok=true

section "SSH reachability"
for node in "${NODE_ARR[@]}"; do
  if ssh_node "$node" 'echo OK' &>/dev/null; then echo "[$node] ssh OK"
  else echo "[$node] ssh FAILED"; ok=false; fi
done

section "Docker daemon"
for_each_node 'docker info --format "Server v{{.ServerVersion}}, {{.NCPU}} CPUs"' || ok=false

section "RoCE NIC status (need: ${IFACES})"
for node in "${NODE_ARR[@]}"; do
  out=$(ssh_node "$node" "ibdev2netdev | awk '/Up\\)/{print \$5}' | tr -d '()'") \
    || { echo "[$node] ibdev2netdev FAILED"; ok=false; continue; }
  for iface in ${IFACES//,/ }; do
    if grep -qx "$iface" <<< "$out"; then echo "[$node] $iface UP"
    else echo "[$node] $iface MISSING/DOWN — saw: $(echo "$out" | tr '\n' ' ')"; ok=false; fi
  done
done

section "RoCE addressing: each IFACES NIC has an IP, and NODES IP is on one"
# This catches the #1 new-machine footgun: putting a management/SSH IP in NODES.
# NCCL binds transports to the IFACES NIC NAMES (config/nccl-env.sh) while ranks
# advertise the NODES IPs (run-node.sh VLLM_HOST_IP / --master-addr).  If a NODES
# IP isn't on a RoCE iface, NCCL silently socket-falls-back (~10x slower) or hangs.
for i in "${!NODE_ARR[@]}"; do
  node="${NODE_ARR[$i]}"
  table=$(ssh_node "$node" "ip -4 -br addr show 2>/dev/null | awk '{for(j=3;j<=NF;j++) print \$1, \$j}'") \
    || { echo "[$node] ip addr FAILED"; ok=false; continue; }
  roce_ips=""
  for rnic in ${IFACES//,/ }; do
    a=$(awk -v n="$rnic" '$1==n{print $2}' <<< "$table" | head -1 | cut -d/ -f1)
    if [ -n "$a" ]; then echo "[$node] $rnic = $a"; roce_ips+=" $a"
    else echo "[$node] ✗ $rnic has NO IPv4 (RDMA link up but unaddressed)"; ok=false; fi
  done
  match=false
  for a in $roce_ips; do [ "$a" = "$node" ] && match=true; done
  if $match; then
    echo "[$node] ✓ NODES IP $node is on a RoCE iface"
  else
    onif=$(awk -v ip="$node" '$2 ~ ("^" ip "/"){print $1}' <<< "$table" | head -1)
    echo "[$node] ✗ NODES IP $node is NOT on a RoCE iface — found on '${onif:-<not on this node>}'."
    echo "          NCCL will socket-fallback (~10x slower) or hang.  Set NODES to the RoCE IP"
    echo "          (run: bash scripts/discover.sh)."
    ok=false
  fi
done

section "RoCE NIC names match across nodes"
ref_set=""
for node in "${NODE_ARR[@]}"; do
  s=$(ssh_node "$node" "ibdev2netdev 2>/dev/null | awk '/Up\\)/{print \$5}' | tr -d '()' | sort | paste -sd, -") || s=""
  echo "[$node] Up RoCE NICs: ${s:-<none>}"
  if [ -z "$ref_set" ]; then ref_set="$s"
  elif [ "$s" != "$ref_set" ]; then
    echo "          ✗ differs from $LEADER ($ref_set) — this stack binds one IFACES list to both ranks; names must match"
    ok=false
  fi
done

section "NCCL master port $HEAD_PORT free on head + fabric reachable"
if ssh_node "$LEADER" "ss -ltn 2>/dev/null | grep -q ':$HEAD_PORT '"; then
  echo "[$LEADER] ✗ port $HEAD_PORT already in use (stale container? stop it, or pick another HEAD_PORT)"; ok=false
else
  echo "[$LEADER] port $HEAD_PORT free"
fi
for node in "${NODE_ARR[@]:1}"; do
  if ssh_node "$node" "ping -c1 -W2 $LEADER >/dev/null 2>&1"; then
    echo "[$node] reaches head $LEADER over fabric (ping OK)"
  else
    echo "[$node] ⚠ could not ping head $LEADER (ICMP may be blocked; if NCCL later hangs, check RoCE routing)"
  fi
done

section "Model snapshot present (~158 GB)"
HF_MODEL_DIR="$HF_CACHE/hub/models--$(echo "$MODEL" | sed 's|/|--|g')"
for node in "${NODE_ARR[@]}"; do
  out=$(ssh_node "$node" "ls -1d $HF_MODEL_DIR/snapshots/* 2>/dev/null | head -1") || true
  if [ -n "$out" ]; then
    sz=$(ssh_node "$node" "du -sh $HF_MODEL_DIR 2>/dev/null | cut -f1") || true
    echo "[$node] snapshot OK ($sz): $out"
  else
    echo "[$node] $HF_MODEL_DIR MISSING — run: bash scripts/01-download-weights.sh"; ok=false
  fi
done

section "Shared SSH key"
if [ -n "${SSH_KEY:-}" ]; then
  for node in "${NODE_ARR[@]}"; do
    if ssh_node "$node" "[ -r $SSH_KEY ]"; then echo "[$node] $SSH_KEY readable"
    else echo "[$node] $SSH_KEY MISSING — provision the shared key first"; ok=false; fi
  done
else
  echo "(SSH_KEY unset in cluster.conf — skipping)"
fi

section "Image present?"
for node in "${NODE_ARR[@]}"; do
  if ssh_node "$node" "docker images --format '{{.Repository}}:{{.Tag}}' | grep -qx '$DOCKER_IMAGE'"; then
    echo "[$node] image present"
  else
    echo "[$node] image NOT pulled yet — run: bash scripts/02-pull-image.sh"
  fi
done

section "Memory headroom (need ~120 GiB UMA / node; weights take ~74 GiB/node at TP=2)"
for_each_node "free -h | awk 'NR==2 {print \"Mem total=\" \$2 \" available=\" \$7}'"

echo
if $ok; then
  echo "✅ All pre-flight checks passed.  Next: bash scripts/02-pull-image.sh"
else
  echo "❌ Some checks failed.  Fix them before continuing." >&2
  exit 1
fi
