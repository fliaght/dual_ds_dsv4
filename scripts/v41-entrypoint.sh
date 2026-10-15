#!/usr/bin/env bash
# Runs inside the dedicated EXL3 image. No V4 patches or KV dtype overrides.
set -euo pipefail
SITE=$(python3 -c 'import pathlib,vllm; print(pathlib.Path(vllm.__file__).resolve().parent)')
export GLM53_DETOKENIZER_PY="$SITE/v1/engine/detokenizer.py"
export GLM53_SCHEDULER_PY="$SITE/v1/core/sched/scheduler.py"
export GLM53_SPINWAIT_TARGET="$SITE/distributed/device_communicators/shm_broadcast.py"
export GLM53_XGRAMMAR_BACKEND_PY="$SITE/v1/structured_output/backend_xgrammar.py"
export GLM53_XGRAMMAR_MANAGER_PY="$SITE/v1/structured_output/__init__.py"
for patch in patch_exl3_packed_names patch_exl3_lm_head patch_engram_secondary \
  patch_engram_file patch_h2d_stage patch_sm120_block64 patch_memory_log; do
  python3 "/opt/dsv41/$patch.py"
done
export PYTHONPATH="/opt/dsv41:${PYTHONPATH:-}"
python3 /opt/dsv41/v41_load_reclaim.py "$SITE/models/deepseek_v4_1/nvidia/vl_model.py"
[ "$V41_KV_AUDIT" != 1 ] || python3 /opt/dsv41/v41_kv_audit.py "$SITE/v1/core/kv_cache_utils.py"
[ "$V41_KV_GROUP_FLOOR" != 1 ] || python3 /opt/dsv41/v41_kv_grouping.py "$SITE/v1/core/kv_cache_utils.py"
[ "$V41_COOP_MOE" != 1 ] ||
  python3 /opt/dsv41/v41_coop_patch.py "$SITE/model_executor/layers/quantization/exl3.py" /opt/coop
[[ "$V41_CUDAGRAPH_SIZES" =~ ^[0-9]+( [0-9]+)*$ ]] || { echo "Bad V41_CUDAGRAPH_SIZES" >&2; exit 1; }
read -r -a GRAPH_SIZES <<< "$V41_CUDAGRAPH_SIZES"
ARGS=(
  --served-model-name "$MODEL" --host 0.0.0.0 --port "$API_PORT"
  --tensor-parallel-size 2 --pipeline-parallel-size 1
  --nnodes 2 --node-rank "$NODE_RANK" --master-addr "$HEAD_IP" --master-port "$HEAD_PORT"
  --distributed-executor-backend mp --quantization exl3
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEM_UTIL" --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES"
  --block-size "$BLOCK_SIZE" --tokenizer-mode deepseek_v41
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41
  --enable-auto-tool-choice --enable-prefix-caching --enable-prompt-tokens-details
  --no-enable-flashinfer-autotune --language-model-only
  --cudagraph-capture-sizes "${GRAPH_SIZES[@]}"
  --hf-overrides '{"engram_table_dir":"/engram-src"}'
  --chat-template /model/chat_template.jinja
)
[ "$V41_PREFIX_RETENTION_INTERVAL" = 0 ] ||
  ARGS+=(--prefix-cache-retention-interval "$V41_PREFIX_RETENTION_INTERVAL")
if [ -n "$V41_PROFILER_DIR" ]; then
  ARGS+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$V41_PROFILER_DIR\",\"torch_profiler_with_stack\":false,\"ignore_frontend\":false}")
fi
if [ "$SPEC_METHOD" = dspark ]; then
  ARGS+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$DSPARK_TOKENS}")
elif [ "$SPEC_METHOD" != none ]; then
  echo "Unsupported SPEC_METHOD=$SPEC_METHOD" >&2; exit 1
fi
[ "$NODE_RANK" = 0 ] || ARGS+=(--headless)
exec vllm serve /model "${ARGS[@]}"
