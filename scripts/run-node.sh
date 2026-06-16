#!/usr/bin/env bash
# Launch the vLLM container for ONE node (one TP rank).  Driven by NODE_RANK.
# Self-contained: needs only cluster.conf + config/ alongside it, so it runs
# identically on the local head and on a worker synced into $REMOTE_STAGE.
#
#   NODE_RANK=1 bash run-node.sh    # worker  (boot FIRST)
#   NODE_RANK=0 bash run-node.sh    # head    (boot SECOND; rank 0 serves the API)
#
# Normally you don't call this directly — scripts/03-start-serve.sh orchestrates
# both ranks in the right order.  Override the recipe per-invocation with RECIPE=.

set -euo pipefail

NODE_RANK="${NODE_RANK:?must set NODE_RANK=0 (head) or 1 (worker)}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT_DIR/cluster.conf"

# shellcheck disable=SC2206
NODE_ARR=($NODES)
N_NODES=${#NODE_ARR[@]}
TP_SIZE="${TP_SIZE:-$N_NODES}"
HEAD_IP="${NODE_ARR[0]}"
[ "$NODE_RANK" -lt "$N_NODES" ] || { echo "NODE_RANK $NODE_RANK >= N_NODES $N_NODES" >&2; exit 1; }
VLLM_HOST_IP_VAL="${NODE_ARR[$NODE_RANK]}"

# ---- recipe (tuning knobs); MODEL/IMAGE/etc come from cluster.conf ----
RECIPE_FILE="$ROOT_DIR/config/recipes/${RECIPE}.env"
[ -f "$RECIPE_FILE" ] || { echo "recipe not found: $RECIPE_FILE" >&2; exit 1; }
# shellcheck source=/dev/null
source "$RECIPE_FILE"
: "${MAX_MODEL_LEN:?}" "${GPU_MEM_UTIL:?}" "${MAX_NUM_SEQS:?}" "${MAX_NUM_BATCHED_TOKENS:?}"
: "${BLOCK_SIZE:?}" "${KV_DTYPE:?}" "${CUDAGRAPH_MODE:?}"

# ---- NCCL / transport env (single source of truth) ----
export IFACES
# shellcheck source=/dev/null
source "$ROOT_DIR/config/nccl-env.sh"

# ---- idempotent: tear down stale container on this node ----
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

# ---- rank-specific args ----
if [ "$NODE_RANK" = "0" ]; then
  HEADLESS_FLAG=""
  HOST_FLAGS="--host 0.0.0.0 --port $API_PORT"
else
  HEADLESS_FLAG="--headless"
  HOST_FLAGS=""
fi

# ---- assemble -e flags ----
DOCKER_E=(
  -e "HF_HUB_OFFLINE=1"
  -e "VLLM_HOST_IP=$VLLM_HOST_IP_VAL"
  -e "VLLM_DISABLE_COMPILE_CACHE=1"                     # avoid stale Triton caches across boots
  -e "VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH=1"         # permit sparse-MLA path in compile mode
  -e "VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42"             # force last decoder layer onto DeepGEMM (Marlin garbles it)
  -e "TORCH_CUDA_ARCH_LIST=8.7 8.9 9.0 10.0+PTX 12.0 12.1a"  # include sm_121a in JIT autotune list
  -e "VLLM_ENABLE_CUDA_COMPATIBILITY=0"                 # sm_121 is native; no cross-arch shim
  -e "RAY_memory_monitor_refresh_ms=0"                  # we use the mp executor, not Ray; silence its monitor
)
# Append every NCCL/transport var from nccl-env.sh.
for v in $NCCL_ENV_VARS; do DOCKER_E+=( -e "$v=${!v}" ); done

# ---- optional MTP speculative decoding ----
SPEC_INNER=""
if [ -n "${MTP_CONF:-}" ]; then
  DOCKER_E+=( -e "MTP_CONF=$MTP_CONF" )
  SPEC_INNER='--speculative-config "$MTP_CONF"'
fi

mkdir -p "$LOG_DIR"

# ---- launch detached container ----
# The leading `sed` is an idempotent hot-patch: this image's deepseek_v4_mtp.py
# passes aux_stream_dict= but the decoder layer expects aux_stream_list=. Without
# it, every MTP-enabled boot dies on worker init. No-op once already patched.
docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus '"device=all"' \
  --network host \
  --ipc host \
  --shm-size 10.24g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device /dev/infiniband:/dev/infiniband \
  -v "${HF_CACHE}/:/root/.cache/huggingface/" \
  -v "${LOG_DIR}:/host_logs" \
  "${DOCKER_E[@]}" \
  "$DOCKER_IMAGE" \
  bash -lc "\
    sed -i 's/aux_stream_dict=self\.aux_stream_dict,/aux_stream_list=[torch.cuda.Stream() for _ in range(3)],/' /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v4_mtp.py ; \
    vllm serve $MODEL \
      --trust-remote-code \
      --tensor-parallel-size $TP_SIZE \
      --pipeline-parallel-size 1 \
      --distributed-executor-backend mp \
      --nnodes $N_NODES --node-rank $NODE_RANK \
      --master-addr $HEAD_IP --master-port $HEAD_PORT \
      --kv-cache-dtype $KV_DTYPE \
      --block-size $BLOCK_SIZE \
      --gpu-memory-utilization $GPU_MEM_UTIL \
      --max-model-len $MAX_MODEL_LEN \
      --max-num-seqs $MAX_NUM_SEQS \
      --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
      --enable-expert-parallel \
      --no-enable-flashinfer-autotune \
      --tokenizer-mode deepseek_v4 \
      --tool-call-parser deepseek_v4 \
      --enable-auto-tool-choice \
      --compilation-config '{\"cudagraph_mode\":\"$CUDAGRAPH_MODE\"}' \
      $SPEC_INNER $HEADLESS_FLAG $HOST_FLAGS \
      2>&1 | tee /tmp/vllm.log /host_logs/vllm-dsv4-rank${NODE_RANK}.log"

echo "[rank $NODE_RANK @ $VLLM_HOST_IP_VAL] $(docker ps --filter name="$CONTAINER_NAME" --format '{{.Names}} ({{.Status}})')"
echo "[rank $NODE_RANK] follow: docker exec $CONTAINER_NAME tail -F /tmp/vllm.log"
