# dual_ds_dsv4

**English** | [中文](README.zh-CN.md)

DeepSeek-V4 / V4.1-Flash serving and a shared chat UI for two QSFP-connected NVIDIA
DGX Sparks with TP=2 and an OpenAI-compatible API. Since 2026-10-15 the website runs on
TensorFold (exact settings: about 1.6x the decode speed and 1.9x the
long-input prefill of the vLLM build); the dedicated vLLM images remain the fallback.
The V4.1 EXL3 site default `v41-416k` configures **425,984 tokens** and has passed
a real 415,014-token input; `v41-256k` (**262,144 tokens**) is the conservative
fallback. A 512K recipe exceeded the head node's memory margin. The older V4
`interactive-128k` recipe remains **131,072 tokens**; `interactive-200k` is its compatibility alias.

Repository baseline: [VERSION](VERSION). Anyone sharing these machines should first
read the machine-wide memory guide at `~/MEMORY.md` on both nodes (Chinese); it is kept
outside this repository so other projects see it too.

## Setup and use

Prepare Docker, SSH, host tools, weights, and `cluster.conf` (start from
`cluster.conf.example`). Do not overwrite an existing configuration.

From the head node, after setup:

```bash
bash scripts/00-precheck.sh       # inspect the cluster
bash scripts/quickstart.sh        # pull image, start ranks, prewarm (changes services)
CHAT_LIVE_TESTS=1 bash bench/smoke.sh  # five real inference requests
bash chat/run_chat.sh --daemon   # optional chat UI, port 9450
bash scripts/04-stop.sh --rm     # stop and remove the model containers
```

The vLLM path binds the API to `0.0.0.0:8000`. On the TensorFold path a single-request gateway
owns port 8000 (open to the LAN at this site, no authentication) and TensorFold itself listens on
`127.0.0.1:8100` only.

## Structure

| Path | Responsibility |
|---|---|
| `scripts/` | Configuration loading, cluster checks and service lifecycle |
| `config/` | Shared NCCL settings and tuning recipes |
| `chat/` | Streamlit app, request queue, client, storage and tests |
| `bench/` | Explicit live inference checks and benchmark clients |
| `tests/` | Offline regression tests for shared scripts and benchmarks |

## Development

```bash
bash scripts/check.sh
```

The check command runs offline validation and tests, including a local fake
API. It does not start model containers or send model inference requests.
Its Python dependencies are listed in `requirements-dev.txt`.

License: [PolyForm Noncommercial 1.0.0](LICENSE). Noncommercial use is free;
commercial use needs separate permission from the author, Yi Luo. Third-party
parts keep their own licenses: the TensorFold recipe patches in `config/tensorfold/`
(Apache-2.0, see its `NOTICE`) and the Inter font in `chat/static/fonts/` (SIL OFL 1.1).
Earlier versions published under Apache-2.0 remain under those terms. The
implementation builds on the `lmxxf` / DGX Spark vLLM community and local
`interllm-dsv4` experiments.
