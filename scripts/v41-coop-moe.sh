#!/usr/bin/env bash
# Build the cooperative MoE decode kernel (MiaAI-Lab PR #8, AGPL-3.0; ExLlamaV3 headers MIT)
# from pinned upstream source. Nothing upstream is copied into this repository: sources
# are fetched into $V41_CACHE_DIR/coop-src and the artifact lands in $V41_CACHE_DIR/coop.
#
#   bash scripts/v41-coop-moe.sh build [node]  # CPU-only nvcc, memory-capped container (default: worker)
#   bash scripts/v41-coop-moe.sh push    # copy the artifact to the worker
#
# Selecting it at runtime is separate: V41_COOP_MOE=1 in cluster.conf (both ranks).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
[ "$RUNTIME" = v41 ] || { echo 'Select a V4.1 recipe' >&2; exit 1; }

RUNTIME_REPO=https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks
RUNTIME_PIN=6f7d1590ad49a2b8995188e45d7b9db31e677452
EXL3_REPO=https://github.com/turboderp-org/exllamav3
EXL3_PIN=02aef45cd681b960a00afcd0749a4ab99e6c1bfe
SRC="$V41_CACHE_DIR/coop-src"
OUT="$V41_CACHE_DIR/coop"

fetch() {  # repo pin dir: shallow fetch of exactly one commit
  [ -d "$3/.git" ] || git init -q "$3"
  git -C "$3" fetch -q --depth 1 "$1" "$2"
  git -C "$3" checkout -q --detach "$2"
  [ "$(git -C "$3" rev-parse HEAD)" = "$2" ]
}

case "${1:-}" in
  build)
    # Build where UMA headroom is largest; the worker carries no API server (default).
    node="${2:-${NODE_ARR[1]}}"
    mkdir -p "$SRC"
    fetch "$RUNTIME_REPO" "$RUNTIME_PIN" "$SRC/runtime"
    fetch "$EXL3_REPO" "$EXL3_PIN" "$SRC/exllamav3"
    rm -rf "$SRC/tree" "$OUT.new" && mkdir -p "$SRC/tree" "$OUT.new"
    git -C "$SRC/exllamav3" archive "$EXL3_PIN" exllamav3/exllamav3_ext | tar -x -C "$SRC/tree"
    ext="$SRC/runtime/extensions/cooperative_moe"
    # nvcc only, no GPU. The cap keeps a compile from eating the UMA headroom of a live model.
    run="docker run --rm --network none --memory 2g --memory-swap 2g --cpus 4 \
      --user \$(id -u):\$(id -g) -e HOME=/tmp -v '$SRC:/src:ro' -v '$OUT.new:/out' \
      --entrypoint bash '$DOCKER_IMAGE' \
      -c 'bash /src/runtime/extensions/cooperative_moe/build.sh /src/tree /out/build'"
    if _is_local "$node"; then
      bash -c "$run"
    else
      ssh_node "$node" "rm -rf '$SRC' '$OUT.new' && mkdir -p '$SRC/runtime/extensions' '$OUT.new'"
      scp -qr "$ext" "$node:$SRC/runtime/extensions/"
      scp -qr "$SRC/tree" "$node:$SRC/"
      ssh_node "$node" "$run"
      scp -qr "$node:$OUT.new/build" "$OUT.new/"
      ssh_node "$node" "rm -rf '$OUT.new'"
    fi
    so="$OUT.new/build/cooperative_moe.so"
    digest=$(sha256sum "$so" | cut -d' ' -f1)
    # The adapter refuses any binary but the one it pins; pin our own reviewed build.
    sed "s/^SHA256 = \".*\"$/SHA256 = \"$digest\"/" "$OUT.new/build/runtime.py" > "$OUT.new/runtime.py"
    grep -q "^SHA256 = \"$digest\"$" "$OUT.new/runtime.py"
    cp "$so" "$OUT.new/cooperative_moe.so"
    cp "$ext/test_cuda_integration.py" "$OUT.new/"
    printf '%s\n' "runtime $RUNTIME_PIN" "exllamav3 $EXL3_PIN" "so $digest" "built-on $node" > "$OUT.new/PIN"
    rm -rf "$OUT" && mv "$OUT.new" "$OUT"
    cat "$OUT/PIN" ;;
  push)
    test -s "$OUT/PIN"
    worker="${NODE_ARR[1]}"
    ssh_node "$worker" "mkdir -p '$V41_CACHE_DIR/coop'"
    scp -q "$OUT/cooperative_moe.so" "$OUT/runtime.py" "$OUT/PIN" "$OUT/test_cuda_integration.py" \
      "$worker:$V41_CACHE_DIR/coop/"
    ssh_node "$worker" "cd '$V41_CACHE_DIR/coop' && sha256sum cooperative_moe.so" ;;
  *)
    echo "usage: $0 build|push" >&2; exit 2 ;;
esac
