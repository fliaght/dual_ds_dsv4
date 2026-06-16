# Architecture — model, quantization, topology, executor

## The model: DeepSeek-V4-Flash

From the HF `config.json`:

| Parameter | Value |
|---|---|
| Class | `DeepseekV4ForCausalLM` |
| Total parameters | 284 B |
| Activated parameters / token | 13 B (~4.6%) |
| Decoder layers | 43 |
| Hidden size | 4096 |
| Attention heads | 64 (head dim 512) |
| KV heads | 1 — MLA (multi-head latent attention) |
| Routed experts | 256 (6 active per token) + 1 shared |
| MoE intermediate size | 2048 |
| MTP head | `num_nextn_predict_layers = 1` |
| Quantization | FP4 (MoE experts) + FP8 (rest), UE8M0 scale, block 128×128 |
| On-disk size | 158 GB (46 safetensors shards) |
| Max position embeddings | 1,048,576 (YaRN factor 16 over base 65,536) |
| Sliding window | 128 |

### Hybrid / compressed-sparse attention

V4-Flash claims 1 M context without an exploding KV footprint by combining two
attention paths:

- **Compressed Sparse Attention (CSA)** — token-wise sparse pattern indexed via
  `index_topk = 512` heads.
- **Heavily Compressed Attention (HCA)** — full attention on a small subset of
  layers; the per-layer `compress_ratios` array marks most layers compressed at
  ratio 128, three at ratio 4, and the boundary layers uncompressed.

Measured KV cost is **~400 KB/token at FP8**. With a ~15 GiB KV pool at
`gpu_memory_utilization=0.80`, that is room for ~36k tokens of cached KV per
sequence — more with chunked-prefill pipelining. The practical consequence we
observed: **decode rate is essentially context-independent** (ITL ~87-114 ms
from 128 to 120k tokens of context). The flip side — exact long-context recall
of rare tokens (digits, IDs) degrades past ~16k depth; see
[`PERFORMANCE.md`](PERFORMANCE.md) quality section.

### Multi-Token Prediction (MTP)

The model ships an MTP head used by vLLM as the **draft model** for speculative
decoding (`num_speculative_tokens=2`): each decode step the head proposes 2
candidate tokens, the base model verifies them in one forward pass, accepted
drafts skip a full decode cycle. Measured draft acceptance averages **~70%**
(up to 98% on highly self-similar long context), yielding **~1.6×** single-stream
speedup vs MTP off. The cost is ITL bimodality — ~87 ms on the "hot path"
(drafts accepted) vs ~360 ms when drafts are mostly rejected (structured output
like code/JSON tends toward the cold path).

### Quantization detail

`quantization_config` declares `quant_method=fp8`, but on-disk weights are mixed:
non-expert layers are FP8 (E4M3, block 128×128, UE8M0 scale), expert layers are
FP4. vLLM auto-detects this and routes experts through the MXFP4 handler, which
picks **Marlin** (fast) or **DeepGEMM** (correct, slightly slower) per layer.
**Layer 42 (the last decoder layer) is forced onto DeepGEMM** via
`VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42` — Marlin there produces garbled English
(numerically corrupt). This is also why CUDA graphs are off: DeepGEMM's
`n_valid.item()` CPU↔GPU sync is illegal during graph capture. See
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) GOTCHA 3 & 4.

## Deployment topology

```
              QSFP direct attach (no switch)
        ┌─────────────────────────────────────┐
        │                                     │
   ┌────┴───────────────┐         ┌───────────┴──────┐
   │ HEAD (rank 0)      │         │ WORKER (rank 1)  │
   │ NODES[0]           │◄───────►│ NODES[1]         │
   │ 192.168.200.43     │ RoCE×2  │ 192.168.200.45   │
   │ 192.168.201.43     │         │ 192.168.201.45   │
   │ container          │         │ container        │
   │  vllm-dsv4-mn      │         │  vllm-dsv4-mn    │
   │  --host 0.0.0.0    │         │  --headless      │
   │  --port 8000       │         │  (no API)        │
   │  GB10 121 GiB UMA  │         │  GB10 121 GiB    │
   └────────────────────┘         └──────────────────┘
        ▲   NCCL master = NODES[0]:29519 (HEAD_PORT)
        │   ranks bound to enp1s0f0np0 + enP2p1s0f0np0
        │   GIN/GDAKI/MNNVL disabled (GB10 RoCE doesn't support them)
   ┌────┴───┐
   │ Client │  → :8000/v1/chat/completions  (OpenAI-compatible)
   └────────┘
```

### Cross-node executor — `mp`, not Ray

vLLM's multiprocess (`mp`) executor with explicit
`--nnodes / --node-rank / --master-addr / --master-port`, chosen over Ray because:

1. **Fewer host-network ports** — Ray needs GCS (6379), dashboard (8265), node
   manager ports all exposed; with `--network host` those collide easily.
2. **Community-validated on Spark** — the published V4-Flash Spark recipes use
   `mp`; the Ray path was reported flaky for PP=2 on sm_121.
3. **One less process tree** — only vLLM processes, no Ray layer on top.

**Boot order matters**: workers (`--headless`) must be up and waiting at
`master-addr:HEAD_PORT` before rank 0 connects, or the rendezvous can race.
`scripts/03-start-serve.sh` launches every worker first, then rank 0 locally.

The master endpoint `NODES[0]:29519` is deliberately off the beaten path —
avoids ssh (22), the TRT-LLM sshd (2233), Ray GCS/dashboard (6379/8265), the
vLLM API (8000), and torch-elastic's 29400-29500 range.

### NCCL transport

GB10 RoCE does not support several advanced NCCL transports that this image's
NCCL tries by default (GIN, GDAKI, DMA-BUF, MNNVL). They are all disabled in
[`config/nccl-env.sh`](../config/nccl-env.sh), and every transport is bound to
the two RoCE interfaces. Post-boot, `03-start-serve.sh` greps the head log for
`via NET/IB/0` + `via NET/IB/1` (16 channels/direction × 2 ranks ≈ 64 lines) and
asserts zero `via NET/Socket` / `gdaki` / `MNNVL` fallback. That check is the
gold standard that cross-node TP is actually on RDMA.
