#!/usr/bin/env bash
# Download DeepSeek-V4-Flash (~158 GB) to the HF cache on every node.
#
# Default (sequential): hf download on the head, then rsync to each worker over
# the 201.x subnet ($TRANSFER_PEER) so bulk transfer never touches the 200.x
# NCCL fabric.  Use `parallel` to download independently on every node (doubles
# HF mirror load), or `rsync-only` to just (re)push the head's cache.
#
#   bash scripts/01-download-weights.sh [sequential|parallel|rsync-only]
#
# Pre-check: needs >= $NEED_GIB free where the HF cache lives.

source "$(dirname "$0")/_lib.sh"

MODE="${1:-sequential}"
NEED_GIB="${NEED_GIB:-200}"
HF_SUBDIR="hub/models--$(echo "$MODEL" | sed 's|/|--|g')"

precheck_space() {
  local node="$1" avail
  avail=$(ssh_node "$node" "df -BG '$HF_CACHE' 2>/dev/null | awk 'NR==2{print \$4}' | tr -dc 0-9 \
                            || df -BG \$HOME | awk 'NR==2{print \$4}' | tr -dc 0-9")
  if [ -z "$avail" ] || (( avail < NEED_GIB )); then
    echo "[$node] FAIL: ${avail:-?} GiB free at $HF_CACHE; need >= ${NEED_GIB}" >&2; return 1
  fi
  echo "[$node] OK: ${avail} GiB free"
}

download_on() {
  local node="$1"
  echo ">>> hf download $MODEL on $node"
  ssh_node "$node" "HF_HUB_ENABLE_HF_TRANSFER=1 hf download '$MODEL' --max-workers 8"
}

rsync_head_to() {
  local peer="$1"
  local src="${HF_CACHE}/${HF_SUBDIR}/"
  echo ">>> rsync head:$src -> $peer (over 201.x QSFP)"
  ssh "$peer" "mkdir -p '${HF_CACHE}/${HF_SUBDIR}/'"
  rsync -avP --inplace "$src" "${peer}:${HF_CACHE}/${HF_SUBDIR}/"
}

case "$MODE" in
  sequential)
    section "Space precheck"
    for node in "${NODE_ARR[@]}"; do precheck_space "$node"; done
    download_on "$LEADER"
    section "Replicating to workers over 201.x ($TRANSFER_PEER)"
    rsync_head_to "$TRANSFER_PEER"
    ;;
  parallel)
    section "Space precheck"
    for node in "${NODE_ARR[@]}"; do precheck_space "$node"; done
    pids=()
    for node in "${NODE_ARR[@]}"; do download_on "$node" & pids+=("$!"); done
    for p in "${pids[@]}"; do wait "$p"; done
    ;;
  rsync-only)
    precheck_space "$LEADER"
    rsync_head_to "$TRANSFER_PEER"
    ;;
  *)
    echo "usage: $0 [sequential|parallel|rsync-only]" >&2; exit 1 ;;
esac

echo "✅ Weights present on all nodes.  Next: bash scripts/02-pull-image.sh"
