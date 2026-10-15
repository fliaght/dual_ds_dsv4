#!/usr/bin/env bash
# V4.1 adapter: shared configuration/SSH remain in _config.sh and _lib.sh.
source "$(dirname "$0")/_lib.sh"
[ "$RUNTIME" = v41 ] || { echo 'Select a V4.1 recipe, e.g. RECIPE=v41-416k' >&2; exit 1; }
[ "$N_NODES" = 2 ] || { echo 'V4.1 requires exactly two nodes' >&2; exit 1; }
ACTION="${1:-status}"

coop_digest() {  # both ranks must load the identical kernel binary
  [ "$V41_COOP_MOE" = 1 ] || return 0
  sha256sum "$V41_CACHE_DIR/coop/cooperative_moe.so" 2>/dev/null | cut -d' ' -f1 || echo missing
}

settings_fingerprint() {
  # Environment overrides on the head must not silently diverge from the worker.
  printf '%s\n' "$MODEL" "$MODEL_REVISION" "$ENGRAM_REVISION" "$MAX_MODEL_LEN" \
    "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS" "$GPU_MEM_UTIL" \
    "$KV_CACHE_MEMORY_BYTES" "$BLOCK_SIZE" "$SPEC_METHOD" "$DSPARK_TOKENS" \
    "$V41_IO_THREADS" "$V41_CUDAGRAPH_SIZES" "$V41_PREFIX_RETENTION_INTERVAL" \
    "$V41_EMPTY_CACHE_MEMAVAIL_GIB" "$V41_MALLOC_ARENA_MAX" "$V41_PROFILER_DIR" "$V41_COOP_MOE" \
    "$V41_LOAD_RECLAIM_MIB" "$V41_NCCL_MAX_NCHANNELS" "$V41_KV_GROUP_FLOOR" "$V41_KV_AUDIT" \
    "$(coop_digest)" | sha256sum
}

local_precheck() {
  python3 "$ROOT_DIR/scripts/v41_assets.py" verify --directory "$V41_ENGRAM_DIR" \
    --revision "$ENGRAM_REVISION" --engram
  local budget available required rank_mib free required_free
  budget=$(python3 "$ROOT_DIR/scripts/v41_assets.py" verify --directory "$V41_MODEL_DIR" \
    --revision "$MODEL_REVISION")
  echo "$budget"
  rank_mib=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["conservative_rank_mib"])' "$budget")
  required=$((rank_mib + V41_BOOT_MARGIN_MIB))
  required_free=$((rank_mib + WATCHDOG_THRESHOLD_MIB))
  available=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)
  free=$(awk '/^MemFree:/{print int($2/1024)}' /proc/meminfo)
  echo "MemAvailable=${available} MiB, required=${required} MiB; MemFree=${free} MiB, required=${required_free} MiB"
  (( available >= required )) || { echo 'Insufficient UMA headroom' >&2; return 1; }
  (( free >= required_free )) || {
    echo 'Insufficient physical free pages for GB10 CUDA allocation; reclaim idle file cache before starting' >&2
    return 1
  }
  local active
  active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  [ -z "$active" ] || { echo "GPU is already in use: $active" >&2; return 1; }
  # A GPU container may still be importing Python before it appears in NVML.
  local container requests
  for container in $(docker ps -q); do
    requests=$(docker inspect "$container" --format '{{json .HostConfig.DeviceRequests}}')
    if [[ "$requests" == *'"gpu"'* ]]; then
      echo "GPU container already exists: $container; stop it before deployment" >&2
      return 1
    fi
  done
  docker image inspect "$DOCKER_IMAGE" --format '{{.Id}} {{.Architecture}}'
  command -v systemd-run >/dev/null
  systemctl --user show-environment >/dev/null
  [ "$(loginctl show-user "$(id -un)" -p Linger --value)" = yes ] || {
    echo 'Enable user service persistence first: loginctl enable-linger' >&2; return 1;
  }
}

stop_cluster() {
  for_each_node "systemctl --user stop dsv41-guard.service 2>/dev/null || true; docker rm -f '$CONTAINER_NAME' 2>/dev/null || true"
}

failed_start_cleanup() {
  local rc=$? stamp
  [ "$rc" = 0 ] && return 0
  # Release ranks first, then retain their error evidence before removing them.
  for_each_node "docker kill '$CONTAINER_NAME' 2>/dev/null || true" || true
  stamp=$(date +%Y%m%d-%H%M%S)
  for_each_node "docker logs '$CONTAINER_NAME' > '$LOG_DIR/v41-failed-$stamp.log' 2>&1 || true; \
    docker inspect '$CONTAINER_NAME' > '$LOG_DIR/v41-failed-$stamp.inspect.json' 2>&1 || true; \
    cp '$LOG_DIR/v41-guard.json' '$LOG_DIR/v41-failed-$stamp.guard.json' 2>/dev/null || true" || true
  stop_cluster || true
  return "$rc"
}

case "$ACTION" in
  local-settings) settings_fingerprint; exit ;;
  local-precheck) local_precheck; exit ;;
  local-pack)
    : "${V41_PACKED_DIR:?Set V41_PACKED_DIR in cluster.conf first}"
    : "${NODE_RANK:?}"
    [[ "$NODE_RANK" =~ ^[01]$ ]]
    local_precheck
    mkdir -p "$V41_PACKED_DIR"
    # CPU-only preprocessing, with bounded chunks and cgroup memory. The image
    # owns the format and TP layout; serving reads the result through O_DIRECT.
    docker run --rm --name dsv41-pack --network none --memory=2g --memory-swap=2g \
      --user "$(id -u):$(id -g)" \
      -v "$V41_ENGRAM_DIR:/engram-src:ro" -v "$V41_PACKED_DIR:/engram-packed" \
      --entrypoint python3 "$DOCKER_IMAGE" /opt/dsv41/pack_engram.py \
      --model /engram-src --out /engram-packed --rank "$NODE_RANK" --tp 2 --chunk-rows 65536
    exit ;;
  local-guard)
    # ExecStopPost also kills the rank if the guard exits unexpectedly. A monitor
    # failure therefore fails closed; cgroup RAM limits alone cannot bound GB10 GPU RAM.
    systemctl --user stop dsv41-guard.service 2>/dev/null || true
    systemctl --user reset-failed dsv41-guard.service 2>/dev/null || true
    peer="${NODE_ARR[0]}"
    if is_leader; then peer="${NODE_ARR[1]}"; fi
    cache_args=(--cache-dir "$V41_MODEL_DIR" --cache-dir "$V41_ENGRAM_DIR")
    [ -z "${V41_PACKED_DIR:-}" ] || cache_args+=(--cache-dir "$V41_PACKED_DIR")
    # Clean cache from this user's own tools (clones, builds, logs) is otherwise only reclaimed near
    # the kernel's ~168 MB watermark; the guard asks for it back once MemFree drops below 16 GiB.
    cache_args+=(--reclaim-cgroup "/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service")
    rm -f "$LOG_DIR/v41-guard.json"
    systemd-run --user --unit=dsv41-guard --collect -p MemoryMax=128M -p MemorySwapMax=0 \
      -p "ExecStopPost=-/usr/bin/docker kill $CONTAINER_NAME" \
      -p "StandardOutput=append:$LOG_DIR/v41-guard.log" -p "StandardError=append:$LOG_DIR/v41-guard.log" \
      python3 "$ROOT_DIR/scripts/memory_guard.py" --container "$CONTAINER_NAME" \
      --reserve-mib "$WATCHDOG_THRESHOLD_MIB" --state "$LOG_DIR/v41-guard.json" --peer "$peer" \
      "${cache_args[@]}"
    for ((i=0;i<20;i++)); do
      [ ! -s "$LOG_DIR/v41-guard.json" ] || exit 0
      sleep 0.1
    done
    echo 'Guard failed to become ready' >&2; exit 1 ;;
esac
is_leader || { echo "Run $ACTION on head $LEADER" >&2; exit 1; }

case "$ACTION" in
  pack)
    push_node "${NODE_ARR[1]}"
    for rank in 0 1; do
      script="$REMOTE_STAGE/scripts/v41-service.sh"
      [ "$rank" != 0 ] || script="$ROOT_DIR/scripts/v41-service.sh"
      ssh_node "${NODE_ARR[$rank]}" "NODE_RANK=$rank RECIPE='$RECIPE' bash '$script' local-pack"
    done
    ;;
  download)
    # This command is intentionally separate from start; never fetch huge assets
    # while the model is resident. Each download process has a 4 GiB host limit.
    for_each_node 'test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"'
    mkdir -p "$V41_MODEL_DIR" "$V41_ENGRAM_DIR"
    python3 "$ROOT_DIR/scripts/v41_assets.py" manifest --directory "$V41_MODEL_DIR" --repo "$MODEL" --revision "$MODEL_REVISION"
    python3 "$ROOT_DIR/scripts/v41_assets.py" manifest --directory "$V41_ENGRAM_DIR" --repo "$ENGRAM_MODEL" --revision "$ENGRAM_REVISION" --engram
    free_gib=$(df -BG "$V41_MODEL_DIR" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')
    (( free_gib >= 420 )) || { echo 'Need 420 GiB free for staging' >&2; exit 1; }
    HF_BIN=$(command -v hf)
    systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 \
      env HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_PROGRESS_BARS=1 \
      "$HF_BIN" download "$MODEL" --revision "$MODEL_REVISION" --local-dir "$V41_MODEL_DIR" --max-workers 2
    systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 \
      env HF_HUB_DISABLE_XET=0 HF_XET_HIGH_PERFORMANCE=0 HF_XET_FIXED_DOWNLOAD_CONCURRENCY=4 \
      HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE=256mb HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE=64mb \
      HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT=512mb HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER=128mb \
      HF_HUB_DISABLE_PROGRESS_BARS=1 \
      "$HF_BIN" download "$ENGRAM_MODEL" model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
      model.safetensors.index.json config.json --revision "$ENGRAM_REVISION" --local-dir "$V41_ENGRAM_DIR" --max-workers 1
    python3 "$ROOT_DIR/scripts/v41_assets.py" verify --directory "$V41_MODEL_DIR" --revision "$MODEL_REVISION" --sha256
    python3 "$ROOT_DIR/scripts/v41_assets.py" verify --directory "$V41_ENGRAM_DIR" --revision "$ENGRAM_REVISION" --engram --sha256
    worker="${NODE_ARR[1]}"
    ssh_node "$worker" "mkdir -p '$V41_MODEL_DIR' '$V41_ENGRAM_DIR'"
    peer="${V41_TRANSFER_PEER:-$worker}"
    rsync -a --partial --exclude=.cache -e 'ssh -o BatchMode=yes -o ConnectTimeout=5' \
      "$V41_MODEL_DIR/" "$peer:$V41_MODEL_DIR/"
    rsync -a --partial --exclude=.cache -e 'ssh -o BatchMode=yes -o ConnectTimeout=5' \
      "$V41_ENGRAM_DIR/" "$peer:$V41_ENGRAM_DIR/"
    push_node "$worker"
    ssh_node "$worker" "python3 '$REMOTE_STAGE/scripts/v41_assets.py' verify --directory '$V41_MODEL_DIR' --revision '$MODEL_REVISION' --sha256"
    ssh_node "$worker" "python3 '$REMOTE_STAGE/scripts/v41_assets.py' verify --directory '$V41_ENGRAM_DIR' --revision '$ENGRAM_REVISION' --engram --sha256"
    ;;
  precheck|start)
    push_node "${NODE_ARR[1]}"
    head_settings=$(settings_fingerprint)
    worker_settings=$(ssh_node "${NODE_ARR[1]}" "RECIPE='$RECIPE' bash '$REMOTE_STAGE/scripts/v41-service.sh' local-settings")
    [ "$head_settings" = "$worker_settings" ] || {
      echo 'Inference settings differ between nodes; put shared overrides in cluster.conf' >&2; exit 1;
    }
    local_precheck
    ssh_node "${NODE_ARR[1]}" "RECIPE='$RECIPE' bash '$REMOTE_STAGE/scripts/v41-service.sh' local-precheck"
    # Docker's classic and containerd image stores expose different .Id values
    # for identical content. Compare layers, execution config and platform.
    image_format='{{json .RootFS.Layers}} {{json .Config}} {{.Architecture}} {{.Os}}'
    head_id=$(docker image inspect "$DOCKER_IMAGE" --format "$image_format" | sha256sum)
    worker_id=$(ssh_node "${NODE_ARR[1]}" "set -o pipefail; docker image inspect '$DOCKER_IMAGE' --format '$image_format' | sha256sum")
    [ "$head_id" = "$worker_id" ] || { echo 'Image content differs between nodes' >&2; exit 1; }
    if ss -ltnH | awk '{print $4}' | grep -Eq ":($API_PORT|$HEAD_PORT)$"; then
      echo 'API/master port is occupied' >&2; exit 1
    fi
    [ "$ACTION" = start ] || exit 0
    # Any failed start removes BOTH ranks and stops guards. No automatic retries.
    trap failed_start_cleanup EXIT
    trap 'exit 130' INT TERM
    for_each_node "docker rm '$CONTAINER_NAME' 2>/dev/null || true"
    RECIPE="$RECIPE" bash "$ROOT_DIR/scripts/v41-service.sh" local-guard
    ssh_node "${NODE_ARR[1]}" "RECIPE='$RECIPE' bash '$REMOTE_STAGE/scripts/v41-service.sh' local-guard"
    ssh_node "${NODE_ARR[1]}" "NODE_RANK=1 RECIPE='$RECIPE' bash '$REMOTE_STAGE/scripts/v41-node.sh'"
    NODE_RANK=0 RECIPE="$RECIPE" bash "$ROOT_DIR/scripts/v41-node.sh"
    deadline=$((SECONDS + V41_READY_TIMEOUT))
    until curl -fsS --max-time 3 "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; do
      (( SECONDS < deadline )) || { echo 'V4.1 startup timeout' >&2; exit 1; }
      for_each_node "test \"\$(docker inspect '$CONTAINER_NAME' --format '{{.State.Running}}')\" = true && systemctl --user is-active --quiet dsv41-guard.service" || exit 1
      sleep 5
    done
    docker logs "$CONTAINER_NAME" > "$LOG_DIR/vllm-dsv41-rank0.log" 2>&1
    ssh_node "${NODE_ARR[1]}" "docker logs '$CONTAINER_NAME' > '$LOG_DIR/vllm-dsv41-rank1.log' 2>&1"
    echo "V4.1 ready on :$API_PORT. Capacity remains unvalidated until live tests pass."
    ;;
  stop) stop_cluster ;;
  status)
    for_each_node "docker ps --filter name='$CONTAINER_NAME' --format '{{.Names}} {{.Status}}'; cat '$LOG_DIR/v41-guard.json' 2>/dev/null || true; free -m | head -2"
    ;;
  *) echo "Usage: $0 {download|pack|precheck|start|stop|status}" >&2; exit 2 ;;
esac
