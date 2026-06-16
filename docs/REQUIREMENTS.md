# Hardware & software requirements

## Tested configuration

| Item | Spec |
|---|---|
| Nodes | 2× NVIDIA DGX Spark (GB10, sm_121 / sm_121a) |
| Unified memory per node | 121 GiB (CPU + GPU shared — UMA) |
| GPU compute capability | 12.1a (one GB10 per node) |
| Interconnect | QSFP direct attach (no switch), RoCE v2 |
| RoCE NICs | 2 ports per node: `enp1s0f0np0` (200.x), `enP2p1s0f0np0` (201.x) — note the **uppercase P** in the second name |
| NCCL all-reduce baseline | ~23 GB/s cross-node |
| OS | Ubuntu, kernel 6.17.0-1018-nvidia (aarch64) |
| NVIDIA driver | 580.159.03 |
| CUDA (host) | 13.0 (the container ships its own toolchain) |
| Docker | with `--device /dev/infiniband` working |

A single DGX Spark **cannot** serve this model: the weights are ~158 GB on disk
and ~74 GiB per node even after TP=2 sharding, against 121 GiB of UMA. Two
stacked Sparks are the minimum viable deployment — that is the entire reason this
project exists.

## Model

`deepseek-ai/DeepSeek-V4-Flash` — a 284 B-parameter / 13 B-active MoE with native
1 M context, mixed FP4 (experts) + FP8 (rest) quantization, ~158 GB on disk
(46 safetensors shards). See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full
model breakdown.

## Container image

`lmxxf/vllm-deepseek-v4-dgx-spark:marlin-fix-20260511` — a community vLLM fork
built for ARM64 / sm_121 with patches that upstream vLLM did not have at
deployment time:

- ldmatrix fix for SM120+
- Triton MLA-sparse kernels for sm_121
- Marlin → DeepGEMM fallback for the final MoE layer (numerical-corruption fix)
- DeepSeek-V4-Flash model loader + MTP support

| Property | Value |
|---|---|
| Tag | `marlin-fix-20260511` |
| Size | ~9 GB compressed / ~19-29 GB extracted |
| Arch | linux/arm64 |
| RepoDigest | `sha256:ccff377be2e731cc2c6ef930c84c133eefaa7eaff019d7daa91357ced8fdcff5` |
| vLLM build | `0.1.dev1+g7a34ed538.d20260510` |

Watch `lmxxf/deepseek-v4-deployment-on-dgx-spark` and `eugr/spark-vllm-docker`
for newer builds; a build that fixes the MTP `aux_stream_dict` bug lets you drop
the sed hot-patch in `scripts/run-node.sh`.

## What you must set up before `quickstart.sh`

1. **Model cached on every node** at
   `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/`.
   Use `bash scripts/01-download-weights.sh` (hf download on the head, then rsync
   to workers over the 201.x subnet). Needs ≥ 200 GiB free where the cache lives.
2. **Passwordless SSH** between nodes (default agent/key). The scripts use plain
   `ssh <ip>`; `cluster.conf`'s `SSH_KEY` is only verified for readability by the
   precheck. NVIDIA's `discover-sparks` provisions `~/.ssh/id_ed25519_shared`.
3. **RoCE NICs Up** with IPs on the 200.x and 201.x subnets. Verify with
   `ibdev2netdev | grep -i up` on every node (expect `rocep1s0f0` + `roceP2p1s0f0`).
4. **Docker daemon running** + your user in the `docker` group on every node.
5. **`panic_on_oom` awareness**: DGX Spark ships `vm.panic_on_oom=1`, so an OOM
   reboots the whole node instantly (and, because both ranks do collective ops,
   tends to take the other node down too). `scripts/oom-watchdog.sh` —
   auto-started by `03-start-serve.sh` — is your safety net; keep it running.
   Optional permanent fix: `sudo sysctl vm.panic_on_oom=0` to let the OOM-killer
   run instead (inference fails mid-request, but the box stays up).

## Will it work on a different topology?

The scripts read `NODES` from `cluster.conf` and generalise to N nodes
(`TP_SIZE` defaults to the node count), but **only TP=2 across exactly 2 Sparks
is validated** for V4-Flash. `--pipeline-parallel-size 1` is hard-set: PP=2 is
reported buggy on sm_121. See [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).
