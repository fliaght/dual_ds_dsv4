# Troubleshooting — DGX Spark / sm121 / GB10 / V4-Flash gotchas

These are the bugs and quirks that the recipes and `scripts/` already work
around. Documented so you know **why** each flag is set and can debug new
failures. A per-error-message lookup table is at the bottom.

---

## GOTCHA 1 — `panic_on_oom` reboots the whole node (often both)

**Symptom**: both DGX Sparks reboot at the same instant, no log entries before
the reboot timestamp, `dmesg` empty afterward. `uptime` / `last reboot` show both
nodes back within ~1 s of each other.

**Root cause**: DGX Spark ships `vm.panic_on_oom=1`. On UMA (121 GiB shared
CPU+GPU), memory pressure triggers a kernel panic + reboot within ~50 ms — faster
than any userland OOM handling. Because rank 0 and rank 1 do collective ops, one
node's panic deadlocks the other on `cudaStreamSynchronize` and it goes down too.

**Mitigation in this repo**:
- `scripts/oom-watchdog.sh` (auto-started by `03-start-serve.sh`) polls `free -m`
  on every node and hard-kills vllm cluster-wide below
  `WATCHDOG_THRESHOLD_MIB=4096`.
- Recipes default `GPU_MEM_UTIL=0.80` (not 0.85+) to keep ~15 GiB idle headroom.

**Permanent fix (optional)**: `sudo sysctl vm.panic_on_oom=0` so the OOM-killer
runs instead — inference fails mid-request but the box stays up.

---

## GOTCHA 2 — MTP fails to load: `aux_stream_dict` kwarg mismatch

**Symptom**: every MTP-enabled boot dies immediately on worker init with
```
TypeError: DeepseekV4DecoderLayer.__init__() got an unexpected keyword argument 'aux_stream_dict'
```

**Root cause**: this image's `deepseek_v4_mtp.py` passes
`aux_stream_dict=self.aux_stream_dict` to the decoder layer, but that
constructor expects `aux_stream_list` (a `list[torch.cuda.Stream]`).

**Fix**: `scripts/run-node.sh` applies an idempotent `sed` hot-patch inside the
container before `vllm serve` starts:
```bash
sed -i 's/aux_stream_dict=self\.aux_stream_dict,/aux_stream_list=[torch.cuda.Stream() for _ in range(3)],/' \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v4_mtp.py
```
Re-running on an already-patched file is a no-op. A newer image with this bug
fixed lets you delete the patch.

---

## GOTCHA 3 — Marlin garbles English in the final MoE layer

**Symptom**: with the default Marlin path, output is numerically corrupt —
e.g. a correct `391` surrounded by `((<!--((` and random Chinese tokens. Chinese
prompts may look coherent while English is garbled (or vice versa).

**Root cause**: the MXFP4 Marlin kernel has a numerical defect in **layer 42**
(the last of 43 decoder layers) on this image. DeepGEMM computes it correctly.

**Fix**: `VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42` (set in `run-node.sh`) forces
layer 42 onto DeepGEMM while Marlin handles 0-41. `bench/smoke.sh` asserts the
math answer is `391` precisely to catch this regression.

---

## GOTCHA 4 — CUDA graphs are incompatible with the DeepGEMM layer-42 path

**Symptom**: with `cudagraph_mode` set to anything but `NONE`:
- `PIECEWISE` → autotune crashes at warmup with
  `cudaErrorStreamCaptureUnsupported`.
- `FULL_AND_PIECEWISE` → appears to work, then hangs after ~6 requests on sm_121.

**Root cause**: DeepGEMM's `m_grouped_fp8_fp4_gemm_nt_contiguous_triton` calls
`n_valid.item()` — a CPU↔GPU sync — *inside* the captured graph, which is illegal
during capture. Since GOTCHA 3 forces layer 42 onto DeepGEMM, graphs can't be
captured there.

**Fix**: all recipes hard-set `CUDAGRAPH_MODE=NONE`. Cost is ~10-20% decode
throughput, but it's the only stable mode for this image. (The theoretical
alternative — Marlin-only MoE *with* graphs — re-introduces GOTCHA 3's garbled
output, so it's not worth it for general chat.)

---

## GOTCHA 5 — `gpu_memory_utilization` semantics on UMA

vLLM's `gpu_memory_utilization` is a fraction of **total** GPU memory, and on UMA
that total is the whole 121 GiB. The TP=2 weight slice alone is ~74 GiB ≈ 0.61 of
UMA, so:

| `gpu_memory_utilization` | Result |
|---:|---|
| 0.45 | **won't even boot** — below the weight footprint |
| 0.70 | boots; ~12 GiB KV pool at 32k context |
| 0.75 | ~12 GiB KV (concurrent-32k default) |
| 0.80 | ~15 GiB KV + ~15 GiB idle headroom (single-stream recipes) |
| 0.85 | bigger KV but only ~7 GiB free at idle → activation surges trip the watchdog |

Raising util past 0.80 trades watchdog-trigger risk for marginal KV. `0.80` is
the sweet spot for single-stream; `0.75` for the smaller-context concurrent
recipe. Bumping `MAX_NUM_BATCHED_TOKENS` from 8192→32768 *cost* ~12 GiB of
workspace and shrank the KV pool — a net loss here; leave it at 8192.

---

## GOTCHA 6 — NCCL GIN/GDAKI/MNNVL unsupported on GB10

**Symptom**: after weights load, before the API comes up, rank 1 segfaults:
```
NCCL error in regResult (.../ncclUtils.cpp): unhandled system error
  transport/net_ib/gdaki/gin_host_gdaki.cc ...
Rank1 worker exit code: 139
```

**Root cause**: this image's NCCL enables GPU-Initiated Networking (GIN),
GDAKI, DMA-BUF registration, and Multi-Node NVLink by default. GB10 / DGX Spark
RoCE exposes none of them.

**Fix**: `config/nccl-env.sh` disables all seven (`NCCL_GIN_ENABLE=0`,
`NCCL_WIN_ENABLE=0`, `NCCL_DMABUF_ENABLE=0`, `NCCL_GRAPH_REGISTER=0`,
`NCCL_LOCAL_REGISTER=0`, `NCCL_MNNVL_ENABLE=0`, `NCCL_IB_MERGE_NICS=0`) and binds
all transports to the two RoCE NICs.

**Verify**: `03-start-serve.sh` greps the head log for `via NET/IB/0` +
`via NET/IB/1` and asserts zero `via NET/Socket` / `gdaki` / `MNNVL`. If you see
socket fallback, decode is ~10× slower.

---

## GOTCHA 7 — KV pool size is non-deterministic across restarts

**Symptom**: same recipe, repeated restarts of `vllm serve`, and the reported KV
pool / effective max length jumps around (the TRT-LLM sibling saw 9920-40288 with
identical config).

**Root cause**: the KV pool is sized from a `torch.cuda.mem_get_info()` probe
after warmup; PyTorch caching-allocator fragmentation makes the same total
occupancy report different "free" each time.

**Fix**:
- **Best**: `docker rm -f` the container between restarts (this repo's
  `04-stop.sh --rm` does it) — drops allocator state. Don't just `pkill vllm`.
- **Gate**: `--max-model-len` caps the pool deterministically; if the probe gives
  less, the smaller wins.
- Keep page cache clean (don't run another big model first).

---

## GOTCHA 8 — FlashInfer autotune crashes on sm_121

**Symptom**: startup crash during FlashInfer autotune (vllm#41524).

**Fix**: `--no-enable-flashinfer-autotune` is always set in `run-node.sh`. If a
newer image fixes the autotune bug, re-enabling it is a likely prefill speedup.

---

## GOTCHA 9 — single-shot output cliff at 16k regardless of util

**Symptom**: `OSL=16384` crashes (watchdog kill) no matter the util setting,
even though 8k (MTP on) / 12k (MTP off) succeed.

**Root cause**: the allocator + workspace growth pattern exceeds the KV pool
budget at ~16k output. It's a hard cliff, not a gradual degradation.

**Fix**: keep single-shot output ≤ 8k (MTP on) or ≤ 12k (`long-output` recipe).
For longer generation, stream and chunk at the application layer.

---

## GOTCHA 10 — PP=2 is buggy on Spark; some CLI flags don't exist

- **`--pipeline-parallel-size 1` is hard-set.** PP=2 is reported buggy on sm_121
  (forum); don't change it. Cross-node parallelism here is TP, not PP.
- **`--reasoning-parser deepseek_v4` does not exist** in this image (only
  `deepseek_v3`). It's intentionally omitted from `run-node.sh`.
- **`--tokenizer-mode deepseek_v4` and `--tool-call-parser deepseek_v4` ARE
  required** for the V4 chat template and OpenAI-style tool calls — they're set.
- **`--trust-remote-code` is required** — V4 ships custom modeling code.

---

## GOTCHA 11 — bench client timeouts too low for long-context prefills

A 120k-token prefill takes ~570 s cold. Default OpenAI-client timeouts (often
120 s) abort it. `bench/bench_openai.py` defaults to 600 s/request and
1800 s/cell; `bench/realprompt_sweep.sh` uses `curl --max-time 1800`.

---

## Diagnosis cheat sheet

| Log line / symptom | Gotcha → fix |
|---|---|
| Garbled English, e.g. `((<!--((` around a correct number | 3 — set `VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42` |
| `got an unexpected keyword argument 'aux_stream_dict'` | 2 — sed hot-patch (run-node.sh applies it) |
| `cudaErrorStreamCaptureUnsupported` at warmup | 4 — `CUDAGRAPH_MODE=NONE` |
| Service hangs after ~6 requests | 4 — you're on `FULL_AND_PIECEWISE`; use `NONE` |
| `transport/net_ib/gdaki/...` segfault, rank exit 139 | 6 — NCCL env vars not exported |
| Decode ~10× slower than expected | 6 — NCCL fell back to `via NET/Socket` |
| Won't boot, KV memory error below weight size | 5 — raise `GPU_MEM_UTIL` to ≥0.70 |
| Watchdog trips at idle / on long output | 5/9 — lower util to 0.80, keep OSL ≤ 8k |
| Effective max length differs every restart | 7 — `04-stop.sh --rm` before restart |
| FlashInfer crash at startup | 8 — `--no-enable-flashinfer-autotune` |
| `OSL=16384` always crashes | 9 — hard cliff; keep OSL ≤ 8k/12k |
| Both nodes reboot, dmesg empty | 1 — keep watchdog on, lower util |
| bench request times out at ~120 s on long prompt | 11 — raise `--timeout` |
