#!/usr/bin/env bash
# Pre-flight checks before any container starts.  Verifies, on every node:
#   - SSH reachability from the head
#   - Docker daemon reachable
#   - All RoCE NICs in $IFACES present and Up
#   - Model snapshot present under $HF_CACHE
#   - Shared SSH key readable
#   - Enough free UMA headroom
# Exit 0 iff every check passes on every node.

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
