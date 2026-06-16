# Common helpers sourced by every orchestration script.  Loads cluster.conf and
# provides for_each_node / ssh_node / is_leader / push_node / section.
#
# Usage:
#   source "$(dirname "$0")/_lib.sh"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -f "$ROOT_DIR/cluster.conf" ]; then
  echo "ERROR: $ROOT_DIR/cluster.conf missing.  Copy from cluster.conf.example and edit." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$ROOT_DIR/cluster.conf"

# Required vars
: "${NODES:?NODES not set in cluster.conf}"
: "${IFACES:?IFACES not set in cluster.conf}"
: "${DOCKER_IMAGE:?DOCKER_IMAGE not set}"
: "${CONTAINER_NAME:?CONTAINER_NAME not set}"
: "${MODEL:?MODEL not set}"
: "${HF_CACHE:?HF_CACHE not set}"
: "${RECIPE:?RECIPE not set}"
: "${HEAD_PORT:?HEAD_PORT not set}"
: "${API_PORT:?API_PORT not set}"
: "${LOG_DIR:?LOG_DIR not set}"
: "${REMOTE_STAGE:?REMOTE_STAGE not set}"
: "${WATCHDOG_THRESHOLD_MIB:?WATCHDOG_THRESHOLD_MIB not set}"

# shellcheck disable=SC2206
NODE_ARR=($NODES)
LEADER="${NODE_ARR[0]}"
N_NODES=${#NODE_ARR[@]}
TP_SIZE="${TP_SIZE:-$N_NODES}"

# Resolve the active recipe to an absolute path.
RECIPE_FILE="$ROOT_DIR/config/recipes/${RECIPE}.env"
[ -f "$RECIPE_FILE" ] || { echo "ERROR: recipe '$RECIPE' -> $RECIPE_FILE not found." >&2; exit 1; }

mkdir -p "$LOG_DIR"

# True if $1 is one of this host's own IPs.
_is_local() {
  ip -br addr show 2>/dev/null | awk '{for(i=3;i<=NF;i++)print $i}' | cut -d/ -f1 | grep -qx "$1"
}

# Run a command string on a node.  Local node runs directly; remote via ssh.
ssh_node() {
  local node="$1"; shift
  if _is_local "$node"; then
    bash -c "$*"
  else
    # accept-new: auto-trust the host key on first connect over the trusted
    # direct-attach link, so a fresh pair doesn't fail with a confusing
    # "ssh FAILED" that's really an unknown-host-key TOFU prompt.
    ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$node" "$*"
  fi
}

# Run a command on every node in parallel, prefixing each output line with [node].
for_each_node() {
  local pids=() node
  for node in "${NODE_ARR[@]}"; do
    (
      local rc=0
      out=$(ssh_node "$node" "$@" 2>&1) || rc=$?
      while IFS= read -r line; do echo "[$node] $line"; done <<< "$out"
      exit "$rc"
    ) &
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
  return $failed
}

# Are we running on the leader (rank 0) node?
is_leader() { _is_local "$LEADER"; }

# Sync the files a worker needs (run-node.sh, cluster.conf, config/) into
# $REMOTE_STAGE on a remote node, preserving repo layout so run-node.sh can
# resolve its own ROOT_DIR there.  No-op for the local node.
push_node() {
  local node="$1"
  _is_local "$node" && return 0
  ssh_node "$node" "mkdir -p '$REMOTE_STAGE/scripts' '$REMOTE_STAGE/config/recipes'"
  scp -q "$ROOT_DIR/cluster.conf"            "$node:$REMOTE_STAGE/cluster.conf"
  scp -q "$ROOT_DIR/scripts/run-node.sh"     "$node:$REMOTE_STAGE/scripts/run-node.sh"
  scp -q "$ROOT_DIR/config/nccl-env.sh"      "$node:$REMOTE_STAGE/config/nccl-env.sh"
  scp -q "$ROOT_DIR/config/recipes/${RECIPE}.env" "$node:$REMOTE_STAGE/config/recipes/${RECIPE}.env"
}

section() { echo; echo "═══ $* ═══"; }
