# Recipes — which operating point to run

A "recipe" is a file under [`config/recipes/`](../config/recipes/) holding only
the engine tuning knobs (context length, util, batch, MTP). The model, image,
IPs, and ports live in `cluster.conf`. Select one by name with `RECIPE=` in
`cluster.conf`, then `bash scripts/03-start-serve.sh`.

## Workload → recipe

| Your workload | Recipe | What you get |
|---|---|---|
| **Interactive chat / agents** (default) | `interactive-200k` | Post-warm short-context 23-25 tok/s; multi-turn TTFT ~1 s after first warm turn; OSL ≤ 8k; combined ISL+OSL up to ~124k. MTP on. |
| **Long-form generation** (code refactors, multi-section reports, ≥8k output) | `long-output` | Post-warm ~12 tok/s; OSL ≤ 12k; no spec-decode ITL variance. MTP off. |
| **Bulk short-prompt throughput** (concurrency 2-4) | `concurrent-32k` | Peak aggregate ~34.9 tok/s at conc=4 (128/128); OSL ≤ 4k; 32k context. MTP off, `MAX_NUM_SEQS=4`. |

## The knobs (and why they're set where they are)

| Knob | interactive-200k | long-output | concurrent-32k | Notes |
|---|---|---|---|---|
| `MAX_MODEL_LEN` | 131072 | 131072 | 32768 | full context vs smaller-for-batch |
| `GPU_MEM_UTIL` | 0.80 | 0.80 | 0.75 | weights need ≥0.61; see TROUBLESHOOTING GOTCHA 5 |
| `MAX_NUM_SEQS` | 1 | 1 | 4 | single-stream vs batched |
| `MAX_NUM_BATCHED_TOKENS` | 8192 | 8192 | 8192 | 32768 cost ~12 GiB workspace for no gain |
| `BLOCK_SIZE` | 256 | 256 | 256 | model-arch requirement |
| `KV_DTYPE` | fp8 | fp8 | fp8 | mandatory; halves KV footprint |
| `CUDAGRAPH_MODE` | NONE | NONE | NONE | DeepGEMM `.item()` sync (GOTCHA 4) |
| `MTP_CONF` | num_spec=2 | (off) | (off) | spec decoding; ~1.6× single-stream |

## Switching recipes safely

Because KV-pool sizing is non-deterministic across restarts (GOTCHA 7), always
do a clean container restart when changing recipes:

```bash
$EDITOR cluster.conf            # set RECIPE="..."
bash scripts/04-stop.sh --rm    # drop containers + PyTorch allocator state
bash scripts/03-start-serve.sh  # boot worker-first with the new recipe
```

`03-start-serve.sh` re-syncs the selected recipe to the worker node, so you only
edit `cluster.conf` on the head.

## Tuning beyond the presets

Start from the closest preset and change **one** knob at a time, watching the
watchdog log:

1. **Need more KV / longer context?** Raise `GPU_MEM_UTIL` 0.80→0.82→0.85 in
   small steps — but past 0.80 you risk watchdog trips on activation surges.
2. **Need more concurrency?** Raise `MAX_NUM_SEQS`, but only with a smaller
   `MAX_MODEL_LEN` (the 32k recipe exists for exactly this) — long context +
   high concurrency saturates the KV pool and per-stream rate collapses.
3. **Need longer single output?** Switch MTP off (`long-output`) for ~12k; the
   hard cliff is 16k (GOTCHA 9) regardless of any knob.
4. **`num_speculative_tokens=1`** (vs 2) is an untried lever — vLLM reports
   higher acceptance at 1, which could flatten the ITL bimodality into a tighter
   band. Worth an experiment if cold-path latency hurts you.
