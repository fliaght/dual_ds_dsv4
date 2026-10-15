# dual_ds_dsv4

**English** | [中文](README.zh-CN.md)

DeepSeek-V4 / V4.1-Flash serving and a shared chat UI for two QSFP-connected NVIDIA
DGX Sparks, using dedicated vLLM images, TP=2 and an OpenAI-compatible API.
The V4.1 EXL3 site default `v41-416k` configures **425,984 tokens** and has passed
a real 415,014-token input; `v41-256k` (**262,144 tokens**) is the conservative
fallback. A 512K recipe exceeded the head node's memory margin. The older V4
`interactive-128k` recipe remains **131,072 tokens**; `interactive-200k` is its compatibility alias.

Repository baseline: [VERSION](VERSION). Operational status and deferred work:
[Docs/current.md](Docs/current.md). Follow the [V4.1 runbook](Docs/runbooks/v41.md)
for the new deployment, UMA protection and staged validation. The status page
records measured input lengths, throughput and memory margins. Anyone sharing these
machines should first read the machine-wide memory guide at `~/MEMORY.md` on both
nodes (Chinese); it is kept outside this repository so other projects see it too.

## Setup and use

Use the [setup runbook](Docs/runbooks/setup.md) to prepare Docker, SSH, host
tools, weights, and `cluster.conf`. Do not overwrite an existing configuration.
The existing image/model and network layouts are in [architecture](Knowledge/architecture.md).

From the head node, after setup:

```bash
bash scripts/00-precheck.sh       # inspect the cluster
bash scripts/quickstart.sh        # pull image, start ranks, prewarm (changes services)
CHAT_LIVE_TESTS=1 bash bench/smoke.sh  # five real inference requests
bash chat/run_chat.sh --daemon   # optional chat UI, port 9450
bash scripts/04-stop.sh --rm     # stop and remove the model containers
```

The API binds to `0.0.0.0:8000` by default. Procedures, expected results and
limitations are in [service operations](Docs/runbooks/service.md).

## Structure and documentation

| Path | Responsibility |
|---|---|
| `scripts/` | Configuration loading, cluster checks and service lifecycle |
| `config/` | Shared NCCL settings and tuning recipes |
| `chat/` | Streamlit app, request queue, client, storage and tests |
| `bench/` | Explicit live inference checks and benchmark clients |
| `tests/` | Offline regression tests for shared scripts and benchmarks |
| `Docs/current.md` | Current state, tasks, blockers and next steps |
| `Docs/runbooks/` | [Setup](Docs/runbooks/setup.md), [service](Docs/runbooks/service.md), [chat](Docs/runbooks/chat.md), [checks](Docs/runbooks/checks.md), [TensorFold (evaluation)](Docs/runbooks/tensorfold.md) |
| `Docs/kedb/` | [Memory](Docs/kedb/memory.md), [runtime](Docs/kedb/runtime.md), [measurement](Docs/kedb/measurement.md) |
| `Docs/playbooks/` | [Startup and network diagnosis](Docs/playbooks/startup.md) |
| `Docs/postmortems/` | [Historical memory-pressure incident](Docs/postmortems/memory-pressure.md) |
| `Knowledge/` | [Architecture](Knowledge/architecture.md), [code](Knowledge/code-layout.md), [recipes](Knowledge/recipes.md), [performance evidence](Knowledge/performance.md) |
| `ArchiveDocs/` | Superseded guides and historical evidence; not operating instructions |

## Development

Follow [AGENTS.md](AGENTS.md); `CLAUDE.md` links to the same file. It imports
[dev-protocol](https://github.com/fliaght/dev-protocol) verbatim at a pinned revision.

```bash
rg -n 'keyword|error' Docs/ Knowledge/
bash scripts/check.sh
```

The check command runs offline validation and tests, including a local fake
API. It does not start model containers or send model inference requests.
Dependencies and explicit live-test commands are in the [check runbook](Docs/runbooks/checks.md).

Apache-2.0: [LICENSE](LICENSE). The implementation builds on the `lmxxf` / DGX
Spark vLLM community and local `interllm-dsv4` experiments. Historical evidence
is retained in [ArchiveDocs](ArchiveDocs/).
