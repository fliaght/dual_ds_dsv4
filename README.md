# dual_ds_dsv4 — DeepSeek-V4-Flash on two stacked DGX Sparks

**English** | [中文 (Chinese)](README.zh-CN.md)

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

> **Factory-fresh pair?** Do **[Step 0 — Bootstrap](#step-0--bootstrap-a-brand-new-pair-first-time-only)**
> first (clone the repo, install tools, log in to HF, wire up SSH). This block
> assumes the repo is already cloned and passwordless SSH to the worker works.

```bash
cd dual_ds_dsv4

# New pair? Auto-discover your RoCE NIC names + IPs (pass the worker's ssh addr too):
bash scripts/discover.sh 192.168.200.45   # prints ready-to-paste NODES/IFACES/TRANSFER_PEER

cp cluster.conf.example cluster.conf
$EDITOR cluster.conf                  # paste NODES/IFACES + set MODEL, paths, RECIPE
                                      # ⚠ NODES must be the RoCE-fabric IPs, NOT management/SSH IPs

# One-time: fetch ~158 GB of weights to every node (head download + rsync to workers)
bash scripts/01-download-weights.sh

# Zero-to-serving: precheck → pull image → start serve (worker-first) → verify NCCL → prewarm
bash scripts/quickstart.sh

# Confirm it works
bash bench/smoke.sh                    # 5-prompt sanity (expects 391)
bash bench/run_full_bench.sh           # perf matrix
```

Run everything **from the head node** (`NODES[0]`). After quickstart the service
is at `http://127.0.0.1:$API_PORT` (default `8000`, OpenAI-compatible, head node
only). `00-precheck.sh` validates your connection config — including that each
`NODES` IP is actually on a RoCE interface — before anything is launched, so a
new-pair misconfig fails in seconds, not minutes into NCCL bring-up.

## Beginner's complete walkthrough

Never deployed a cross-node model before? Follow this top to bottom. You run
**every command from the HEAD node** (the first machine in `NODES`); the scripts
reach the worker over SSH for you.

### The mental model

Two DGX Sparks act as **one** GPU big enough for a 158 GB model. The **head**
(rank 0) serves the OpenAI API on port 8000; the **worker** (rank 1) holds half
the model and has no API. They talk over the RoCE cables. You never log into the
worker by hand — `scripts/` do it.

### Step 0 — Bootstrap a brand-new pair (first time only)

DGX Sparks ship with the NVIDIA driver, CUDA, Docker, and the RDMA tools
(`ibdev2netdev`) already installed. From a **factory-fresh** pair you still need
four things: this repo, a few host tools, a HuggingFace login (the weights are
large), and the RoCE fabric + SSH wired up. Do this once per pair.

**0a — Get this repo on the HEAD node**

```bash
git clone https://github.com/fliaght/dual_ds_dsv4    # or scp/rsync the folder over
cd dual_ds_dsv4
```

`git` is preinstalled on DGX Spark; if not, `sudo apt-get install -y git`.

**0b — Install the host tools the scripts use** (run on the head **and** the worker)

```bash
sudo apt-get update && sudo apt-get install -y git jq rsync bc curl netcat-openbsd iproute2
pip install --user "huggingface_hub[cli,hf_transfer]" aiohttp openai
#   jq → parse JSON   rsync → copy weights   bc → bench timing   hf → download
#   aiohttp → bench_openai.py    openai → the client example in Step 5
```

> ⚠️ `pip --user` installs `hf` into `~/.local/bin`, which is **not** on `PATH`
> for the non-login SSH shells the scripts use — so weight download would fail
> with `hf: command not found`. Put it on PATH on **both** nodes:
> ```bash
> echo 'export PATH=$HOME/.local/bin:$PATH' | tee -a ~/.bashrc ~/.profile
> ```
> And if `docker ps` needs `sudo`, add yourself to the docker group (both nodes):
> `sudo usermod -aG docker $USER && newgrp docker`.

**0c — Log in to HuggingFace** (needed to pull the ~158 GB weights)

```bash
hf auth login        # paste a token from https://huggingface.co/settings/tokens
```

Also open <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash> and **accept the
license / request access** with the same account first — otherwise the download
401s after a successful login.

**0d — Wire up the RoCE fabric + passwordless SSH** (the part unique to a fresh pair)

- **Cable** the two Sparks directly with QSFP: port 1 ↔ port 1 (carries the
  200.x subnet), port 2 ↔ port 2 (201.x). No switch.
- **Give each RoCE NIC a static IP** on a shared subnet — e.g. head
  `192.168.200.43/24` and worker `192.168.200.45/24` on `enp1s0f0np0`, and
  `192.168.201.{43,45}/24` on `enP2p1s0f0np0`. NVIDIA's
  [`dgx-spark-playbooks`](https://github.com/NVIDIA/dgx-spark-playbooks)
  `discover-sparks` automates the cabling discovery **and** the SSH key exchange;
  or set static IPs by hand (netplan / `nmcli`).
- **Passwordless SSH** from head → worker:
  ```bash
  ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_shared -N ''     # skip if you already have a key
  ssh-copy-id -i ~/.ssh/id_ed25519_shared.pub <worker-ip>
  ssh <worker-ip> hostname                                     # must return the worker's name, no password
  ```
- *(Recommended)* the kernel reboots the whole box on any OOM (`vm.panic_on_oom=1`).
  The built-in watchdog protects you, but you may also `sudo sysctl vm.panic_on_oom=0`.

**Verify you're ready** — all of these should hold on **both** nodes:

- [ ] `ibdev2netdev | grep -i up` shows the two RoCE NICs Up (`rocep1s0f0` + `roceP2p1s0f0`)
- [ ] `ssh <worker-ip> hostname` returns the worker's name with **no** password prompt
- [ ] `docker ps` works without `sudo` (your user is in the `docker` group)
- [ ] ~200 GB free disk on each node for the weights

### Step 1 — Tell the project about your machines

```bash
cd dual_ds_dsv4

# Auto-detect your RoCE NIC names and IPs (pass the worker's SSH address):
bash scripts/discover.sh 192.168.200.45
```

It prints something like:

```
  NODES="192.168.200.43 192.168.200.45"
  IFACES="enp1s0f0np0,enP2p1s0f0np0"
  TRANSFER_PEER="192.168.201.45"
  ✅ NIC name sets match across all hosts.
```

Now create your config and paste those three lines in:

```bash
cp cluster.conf.example cluster.conf
nano cluster.conf        # or vim/$EDITOR — paste NODES / IFACES / TRANSFER_PEER
```

> ⚠️ **The #1 mistake:** `NODES` must be the **RoCE-fabric IPs** that
> `discover.sh` printed — *not* the management/Wi-Fi IP you SSH in with. Using the
> wrong IP makes the two GPUs talk over slow TCP (≈10× slower) or hang forever.
> The next step rejects this for you, so just trust `discover.sh`.

### Step 2 — Download the model weights (~158 GB, ~30–45 min, once)

```bash
bash scripts/01-download-weights.sh        # downloads on head, copies to worker over 201.x
```

Skip this if both nodes already have
`~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/`.

### Step 3 — Bring the service up

```bash
bash scripts/quickstart.sh
```

This runs three steps in order and is safe to re-run:

| Stage | What it does | Time |
|---|---|---|
| `00-precheck` | Validates NICs, SSH, RoCE addressing, ports, weights, memory | ~30 s |
| `02-pull-image` | Pulls the 9 GB vLLM image on both nodes (skips if present) | 0–5 min |
| `03-start-serve` | Boots worker, then head; waits for ready; verifies NCCL-over-RoCE; warms up | ~5–6 min |

**What "it worked" looks like** — `03-start-serve` ends with:

```
✅ Service ready: http://127.0.0.1:8000
  via NET/IB channels: 64   |   via NET/Socket (fallback): 0
  ✅ PASS — cross-node TP is on dual RoCE.
```

If it instead times out or errors, read the printed tail and
[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — the error → fix table at
the bottom maps almost every failure to a one-line fix.

### Step 4 — Confirm the model answers

```bash
bash bench/smoke.sh        # sends 5 prompts (en/zh/ja + math + code); expects "391"
```

Success prints `✅ PASS: 5/5 prompts answered, math returned 391.`

### Step 5 — Actually use the model

It speaks the **OpenAI Chat Completions** protocol, so any OpenAI client works —
just point it at `http://127.0.0.1:8000/v1` (run from the head node, or open an
SSH tunnel `ssh -L 8000:127.0.0.1:8000 <head>` from your laptop).

**curl:**

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Flash",
    "messages": [{"role": "user", "content": "Explain TP=2 in one sentence."}],
    "max_tokens": 128
  }' | jq -r '.choices[0].message.content'
```

**Python (official `openai` package):**

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V4-Flash",
    messages=[{"role": "user", "content": "Write a haiku about GPUs."}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Useful endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`
(lists `max_model_len`), `/metrics` (Prometheus, incl. MTP acceptance),
`/health`, `/docs` (interactive API browser).

### Step 6 — Monitor while it runs

```bash
tail -f ~/dsv4_logs/vllm-dsv4-rank0.log        # head server log
tail -f ~/dsv4_logs/watchdog-*.log             # OOM watchdog (auto-started)
docker stats vllm-dsv4-mn                       # live memory/CPU
```

The **watchdog** runs in the background and hard-kills vLLM if free memory gets
dangerously low — this is what protects you from the `panic_on_oom` whole-node
reboot. Leave it running.

### Step 7 — Stop (and switch recipes)

```bash
bash scripts/04-stop.sh --rm                    # clean shutdown + remove containers
```

To change the operating point (e.g. longer outputs, or batch throughput), edit
`RECIPE=` in `cluster.conf`, then stop-with-`--rm` and re-run `03-start-serve.sh`.
See [`docs/RECIPES.md`](docs/RECIPES.md) for the workload → recipe table.

### Moving to a different pair of Sparks

Just repeat Steps 1–3 with the new machines: `discover.sh` finds their NICs/IPs,
you paste them into `cluster.conf`, and `00-precheck.sh` validates the new
connection before anything launches. Nothing else changes — the image, model, and
GPU-architecture settings are identical on every DGX Spark.

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
│   ├── discover.sh           # auto-detect RoCE NIC names + IPs → paste into cluster.conf
│   ├── quickstart.sh         # 00 → 02 → 03 in sequence
│   ├── 00-precheck.sh        # NICs Up + NODES-IP-on-RoCE + port free, ssh, model cache, headroom
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
- **Decode ~10× slow, or rank 1 never connects** → `NODES` holds management IPs,
  not the RoCE-fabric IPs. Run `bash scripts/discover.sh`; `00-precheck.sh` now
  rejects this before launch.

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
