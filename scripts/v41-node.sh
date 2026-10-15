#!/usr/bin/env bash
# Single-rank launch; called only after the cluster preflight and guard are ready.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
: "${NODE_RANK:?}"
[ "$RUNTIME" = v41 ] && [ "$N_NODES" = 2 ]
[[ "$NODE_RANK" =~ ^[01]$ ]]
source "$ROOT_DIR/config/nccl-env.sh"
python3 - "$LOG_DIR/v41-guard.json" <<'PY'
import json,sys,time
s=json.load(open(sys.argv[1]))
assert not s['reason'] and time.time()-s['time'] < 3, 'local memory guard is not healthy'
PY
mkdir -p "$V41_CACHE_DIR"/{vllm,triton,tilelang}
ENV_ARGS=()
for v in $NCCL_ENV_VARS MODEL API_PORT HEAD_PORT NODE_RANK MAX_MODEL_LEN MAX_NUM_SEQS \
  MAX_NUM_BATCHED_TOKENS GPU_MEM_UTIL KV_CACHE_MEMORY_BYTES BLOCK_SIZE SPEC_METHOD DSPARK_TOKENS \
  V41_CUDAGRAPH_SIZES V41_PREFIX_RETENTION_INTERVAL V41_PROFILER_DIR V41_COOP_MOE V41_KV_AUDIT \
  V41_KV_GROUP_FLOOR; do
  ENV_ARGS+=(-e "$v=${!v}")
done
ENV_ARGS+=(
  -e "HEAD_IP=$LEADER" -e "VLLM_HOST_IP=${NODE_ARR[$NODE_RANK]}"
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_NET_PLUGIN=none -e NCCL_NVLS_ENABLE=0
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_BUFFSIZE=1048576 -e NCCL_LL128_BUFFSIZE=262144
  -e 'NCCL_PROTO=^LL128' -e "NCCL_MAX_NCHANNELS=$V41_NCCL_MAX_NCHANNELS"
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 -e DSV41_EXL3_SERIAL_STREAMS=1
  -e DSV41_INDEXER_PREFILL_FACTOR=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
  -e DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192
  -e "DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=$V41_EMPTY_CACHE_MEMAVAIL_GIB"
  -e "DSV41_LOAD_RECLAIM_MIB=$V41_LOAD_RECLAIM_MIB"
  -e DSV41_DROP_PAGE_CACHE=1 -e DSV41_CACHE_GIB=0 -e DSV41_RESIDENT_SCALES=0
  -e "DSV41_IO_THREADS=$V41_IO_THREADS" -e DSV41_SOURCE=/engram-src
  -e DSV41_EXL3_K_MAP=/opt/dsv41/exl3_k_map.json
  -e EXL3_FUSED_MOE=1 -e EXL3_FAT_KERNEL=0 -e EXL3_FAT_GROUPED=1 -e EXL3_TEMP_ROWS_FUSED=16
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
  -e VLLM_CACHE_ROOT=/root/.cache/vllm -e TRITON_CACHE_DIR=/root/.triton/cache
  -e TILELANG_CACHE_DIR=/root/.tilelang/cache -e MAX_JOBS=2
  -e TORCHINDUCTOR_COMPILE_THREADS=1 -e OMP_NUM_THREADS=1
)
[ -z "$V41_MALLOC_ARENA_MAX" ] || ENV_ARGS+=(-e "MALLOC_ARENA_MAX=$V41_MALLOC_ARENA_MAX")
COOP_ARGS=()
if [ "$V41_COOP_MOE" = 1 ]; then
  test -s "$V41_CACHE_DIR/coop/cooperative_moe.so"
  COOP_ARGS=(-v "$V41_CACHE_DIR/coop:/opt/coop:ro")
fi
PACKED_ARGS=()
if [ -n "${V41_PACKED_DIR:-}" ]; then
  for layer in 1 14; do
    test -s "$V41_PACKED_DIR/engram-l${layer}-r${NODE_RANK}of2.bin"
  done
  PACKED_ARGS=(-v "$V41_PACKED_DIR:/engram-packed:ro" -e DSV41_PACKED_DIR=/engram-packed)
fi
docker run -d --name "$CONTAINER_NAME" --gpus all --network host --ipc host \
  --oom-score-adj 1000 --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
  --device /dev/infiniband --stop-timeout 30 --log-opt max-size=100m --log-opt max-file=3 \
  -v "$V41_MODEL_DIR:/model:ro" -v "$V41_ENGRAM_DIR:/engram-src:ro" \
  -v "$V41_CACHE_DIR/vllm:/root/.cache/vllm" -v "$V41_CACHE_DIR/triton:/root/.triton/cache" \
  -v "$V41_CACHE_DIR/tilelang:/root/.tilelang/cache" \
  -v "$ROOT_DIR/scripts/v41-entrypoint.sh:/v41-entrypoint.sh:ro" \
  -v "$ROOT_DIR/scripts/v41_load_reclaim.py:/opt/dsv41/v41_load_reclaim.py:ro" \
  -v "$ROOT_DIR/scripts/v41_coop_patch.py:/opt/dsv41/v41_coop_patch.py:ro" \
  -v "$ROOT_DIR/scripts/v41_kv_audit.py:/opt/dsv41/v41_kv_audit.py:ro" \
  -v "$ROOT_DIR/scripts/v41_kv_grouping.py:/opt/dsv41/v41_kv_grouping.py:ro" \
  "${ENV_ARGS[@]}" "${PACKED_ARGS[@]}" "${COOP_ARGS[@]}" --entrypoint bash "$DOCKER_IMAGE" /v41-entrypoint.sh
