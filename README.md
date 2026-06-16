# dual_ds_dsv4 — DeepSeek-V4-Flash on two stacked DGX Sparks

Serve **`deepseek-ai/DeepSeek-V4-Flash`** — a 284 B-param / 13 B-active MoE with
native 1 M context (~158 GB, mixed FP4+FP8) — across **two QSFP-linked DGX
Sparks** with cross-node tensor parallelism (TP=2) on a community vLLM build for
sm_121 (GB10). Get **~23-25 tok/s** single-stream (warm, with MTP), a **131k**
context window, and an OpenAI-compatible API — out of a model that does **not
fit on one Spark** (121 GiB UMA).

> Why this repo exists: V4-Flash on sm_121 is a minefield of platform-specific
> bugs — a Marlin kernel that garbles the last layer, an MTP loader with a wrong
> kwarg, CUDA-graph capture that fights DeepGEMM, NCCL features GB10 doesn't
> support, and a `panic_on_oom` kernel that reboots both nodes on a single OOM.
> This project pins all of them down behind a turnkey, idempotent deployment.
>
> It's the vLLM/DeepSeek sibling of
> [`stacked-sparks-trtllm`](../stacked-sparks-trtllm) (TRT-LLM + Qwen3), built
> from the `interllm-dsv4` dev log.

## Performance at a glance

| Metric | This repo | Notes |
|---|---:|---|
| Single-stream decode (warm, MTP on) | **23-25 tok/s** | hot path; ~12 without MTP |
| Peak aggregate (conc=4) | **34.9 tok/s** | `concurrent-32k` recipe |
| Max context | **131,072** | full 128k; YaRN auto-applies |
| Max single-shot input | **120,000 tok** | prefill is the cost, not decode |
| Max output | **8k** (MTP on) / **12k** (MTP off) | hard cliff at 16k |
| Multi-turn TTFT (cache warm) | **0.6-1 s** | from 4-10 s cold |
| MTP draft acceptance | **~70%** | up to 98% on repetitive long context |

Full numbers + methodology: [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

## Hardware requirements

- **2× NVIDIA DGX Spark** (GB10, sm_121a, 121 GiB UMA each)
- **QSFP direct attach** with RoCE NICs Up on both ports of each Spark
  (`enp1s0f0np0` on 200.x, `enP2p1s0f0np0` on 201.x — note the uppercase **P**)
- **~158 GB** of HF model weights cached on **every** node
- NCCL cross-node baseline ~23 GB/s

Full list + setup: [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md).

## Quickstart

```bash
cd dual_ds_dsv4
cp cluster.conf.example cluster.conf
$EDITOR cluster.conf                  # set NODES, IFACES, MODEL, paths, RECIPE

# One-time: fetch ~158 GB of weights to every node (head download + rsync to workers)
bash scripts/01-download-weights.sh

# Zero-to-serving: precheck → pull image → start serve (worker-first) → verify NCCL → prewarm
bash scripts/quickstart.sh

# Confirm it works
bash bench/smoke.sh                    # 5-prompt sanity (expects 391)
bash bench/run_full_bench.sh           # perf matrix
```

Run everything **from the head node** (`NODES[0]`). After quickstart the service
is at `http://127.0.0.1:8000` (OpenAI-compatible, head node only).

## Project layout

```
.
├── cluster.conf.example      # All site-specific knobs (IPs, NICs, model, ports, recipe)
├── config/
│   ├── nccl-env.sh           # Conservative NCCL env (the 7 disabled features + NIC binding)
│   └── recipes/
│       ├── interactive-200k.env   # PRODUCTION default: MTP on, 131k ctx, OSL<=8k
│       ├── long-output.env        # MTP off, OSL<=12k
│       └── concurrent-32k.env     # MAX_NUM_SEQS=4, peak aggregate ~35 tok/s
├── scripts/
│   ├── quickstart.sh         # 00 → 02 → 03 in sequence
│   ├── 00-precheck.sh        # NICs Up, ssh, docker, model cache, headroom
│   ├── 01-download-weights.sh# hf download on head + rsync to workers over 201.x
│   ├── 02-pull-image.sh      # pull the vLLM image on all nodes (parallel)
│   ├── 03-start-serve.sh     # boot worker-first, wait ready, verify NCCL/IB, prewarm
│   ├── 04-stop.sh            # graceful stop; --rm also drops containers
│   ├── run-node.sh           # per-node launcher (docker run + sed patch + vllm serve)
│   ├── oom-watchdog.sh       # panic_on_oom guard (auto-started by 03)
│   ├── prewarm.py            # progressive warmup; kills first-call TTFT spike
│   └── _lib.sh               # cluster.conf loader + ssh_node / for_each_node / push_node
├── bench/
│   ├── bench_openai.py       # async OpenAI-protocol client (TTFT, ITL, agg tok/s)
│   ├── smoke.sh              # 5-prompt sanity (en/zh/ja + math + code)
│   ├── run_full_bench.sh     # reproduce the published matrix
│   ├── realprompt_sweep.sh   # realistic (non-degenerate) prompt sweep by ISL
│   └── quality_cases.json    # 10 functional tests (math, code, langs, tool-call, needle)
└── docs/
    ├── REQUIREMENTS.md       # what you need before installing
    ├── ARCHITECTURE.md       # model, quantization, topology, mp executor, NCCL
    ├── PERFORMANCE.md        # reproducible numbers
    ├── TROUBLESHOOTING.md    # ⚠️ 11 sm121/V4-Flash gotchas + per-error lookup
    └── RECIPES.md            # workload → recipe selection
```

## When things go wrong

**Read [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)** — 11 documented
gotchas (`panic_on_oom`, Marlin layer-42 corruption, MTP `aux_stream_dict`,
CUDA-graph vs DeepGEMM, NCCL GDAKI, KV-pool non-determinism, …) with a
per-error-message lookup table at the end.

A few high-frequency ones:
- **Garbled English** → `VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42` not applied (GOTCHA 3)
- **Both nodes reboot, no logs** → `panic_on_oom`; keep the watchdog on (GOTCHA 1)
- **rank 1 segfault `gdaki`** → NCCL conservative env not exported (GOTCHA 6)
- **Won't boot below weight size** → raise `GPU_MEM_UTIL` ≥ 0.70 (GOTCHA 5)

## Graceful stop

```bash
bash scripts/04-stop.sh         # SIGTERM→SIGKILL vllm on all nodes
bash scripts/04-stop.sh --rm    # also docker rm -f (recommended between recipe changes)
```

## How to upgrade

When a newer community image ships:
1. Update `DOCKER_IMAGE` in `cluster.conf`.
2. `bash scripts/04-stop.sh --rm && bash scripts/02-pull-image.sh`.
3. `bash scripts/03-start-serve.sh`.
4. If the image fixed the MTP `aux_stream_dict` bug, delete the `sed` line in
   `scripts/run-node.sh`. If FlashInfer autotune is fixed, drop
   `--no-enable-flashinfer-autotune` for a likely prefill speedup.

## Acknowledgements & lineage

- Hardware/network/UMA/NCCL groundwork shared with
  [`stacked-sparks-trtllm`](../stacked-sparks-trtllm) (TRT-LLM + Qwen3 on the
  same two Sparks).
- Image: community vLLM fork `lmxxf/vllm-deepseek-v4-dgx-spark` for ARM64/sm_121.
- The sm_121 fixes (MLA-sparse Triton kernels, Marlin→DeepGEMM layer-42, MTP
  loader) come from the DGX-Spark vLLM community (`lmxxf`, `eugr`, `jasl9187`).

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
