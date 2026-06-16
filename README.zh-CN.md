# dual_ds_dsv4 — 双 DGX Spark 堆叠运行 DeepSeek-V4-Flash

[English](README.md) | **中文**

在**两台 QSFP 直连的 DGX Spark** 上、以跨节点张量并行（TP=2）的方式部署
**`deepseek-ai/DeepSeek-V4-Flash`** 服务——它是一个 284B 参数 / 13B 激活的 MoE 模型，
原生 1M 上下文（约 158 GB，FP4+FP8 混合量化），运行在为 sm_121（GB10）构建的
社区版 vLLM 上。单流（预热后、开启 MTP）可达 **~23-25 tok/s**、**131k** 上下文窗口、
并暴露一个 OpenAI 兼容的 API——而这个模型**单台 Spark（121 GiB 统一内存）根本装不下**。

> 这个仓库为什么存在：V4-Flash 在 sm_121 上是平台特定 bug 的雷区——会把最后一层
> 输出搞乱的 Marlin kernel、传错关键字参数的 MTP 加载器、与 DeepGEMM 冲突的
> CUDA-graph 捕获、GB10 不支持的 NCCL 特性，以及一旦 OOM 就把两台机器一起重启的
> `panic_on_oom` 内核。本项目把这些坑全部固定下来，封装成一套开箱即用、可幂等重跑的
> 部署流程。
>
> 它是 [`stacked-sparks-trtllm`](../stacked-sparks-trtllm)（TRT-LLM + Qwen3）的
> vLLM/DeepSeek 姊妹项目，源自 `interllm-dsv4` 开发日志。

## 性能一览

| 指标 | 本仓库 | 说明 |
|---|---:|---|
| 单流解码（预热、开启 MTP） | **23-25 tok/s** | 热路径；不开 MTP 约 12 |
| 峰值聚合吞吐（并发 4） | **34.9 tok/s** | `concurrent-32k` 配方 |
| 最大上下文 | **131,072** | 完整 128k；YaRN 自动生效 |
| 单次最大输入 | **120,000 token** | 成本在预填充，不在解码 |
| 最大输出 | **8k**（开 MTP）/ **12k**（关 MTP） | 16k 处有硬性悬崖 |
| 多轮 TTFT（缓存命中） | **0.6-1 s** | 冷启动时为 4-10 s |
| MTP 草稿接受率 | **~70%** | 重复性长上下文可达 98% |

完整数据与方法：[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)。

## 硬件要求

- **2× NVIDIA DGX Spark**（GB10，sm_121a，每台 121 GiB 统一内存）
- **QSFP 直连**，每台 Spark 两个端口的 RoCE 网卡都 Up
  （`enp1s0f0np0` 在 200.x，`enP2p1s0f0np0` 在 201.x——注意第二个名字里的大写 **P**）
- 每个节点都缓存有约 **158 GB** 的 HF 模型权重
- NCCL 跨节点基线约 23 GB/s

完整清单与配置：[`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md)。

## 快速开始

> **全新出厂的机器对？** 请先做
> **[步骤 0 — 全新机器对的初始化](#步骤-0--全新机器对的初始化仅第一次)**
> （克隆仓库、装工具、登录 HF、配好 SSH）。下面这段假设仓库已克隆、且到工作节点的
> 免密 SSH 已经可用。

```bash
cd dual_ds_dsv4

# 新机器对？自动发现 RoCE 网卡名与 IP（把工作节点的 ssh 地址也传进去）：
bash scripts/discover.sh 192.168.200.45   # 打印可直接粘贴的 NODES/IFACES/TRANSFER_PEER

cp cluster.conf.example cluster.conf
$EDITOR cluster.conf                  # 粘贴 NODES/IFACES，并设置 MODEL、路径、RECIPE
                                      # ⚠ NODES 必须是 RoCE 网卡的 IP，不能是管理网/SSH 的 IP

# 一次性：把约 158 GB 权重拉到每个节点（主节点下载，再 rsync 给工作节点）
bash scripts/01-download-weights.sh

# 从零到上线：预检 → 拉镜像 → 启动服务（先工作节点）→ 校验 NCCL → 预热
bash scripts/quickstart.sh

# 确认可用
bash bench/smoke.sh                    # 5 条提示的冒烟测试（期望出现 391）
bash bench/run_full_bench.sh           # 性能矩阵
```

**所有命令都在主节点（`NODES[0]`）上运行**；脚本会替你通过 SSH 操作工作节点。
quickstart 完成后，服务位于 `http://127.0.0.1:$API_PORT`（默认 `8000`，OpenAI
兼容，仅主节点暴露）。`00-precheck.sh` 会在拉起任何东西**之前**校验你的连接配置——
包括每个 `NODES` IP 是否真的在 RoCE 网卡上——所以新机器配错会在几秒内失败，而不是
在 NCCL 启动几分钟后才暴露。

## 新手完整教程

第一次部署跨节点模型？从头到尾照着做就行。**所有命令都在主节点**（`NODES` 里的
第一台）运行；脚本会替你通过 SSH 操作工作节点。

### 核心概念

两台 DGX Spark 合起来当成**一块**足以装下 158 GB 模型的大 GPU。**主节点**（rank 0）
在 8000 端口提供 OpenAI API；**工作节点**（rank 1）持有模型的另一半、没有 API。
它们通过 RoCE 直连通信。你永远不需要手动登录工作节点——`scripts/` 会处理。

### 步骤 0 — 全新机器对的初始化（仅第一次）

DGX Spark 出厂自带 NVIDIA 驱动、CUDA、Docker，以及 RDMA 工具（`ibdev2netdev`）。
对一台**全新出厂**的机器对，你还需要四样东西：这个仓库、几个主机工具、一个
HuggingFace 登录（权重很大），以及把 RoCE 网络 + SSH 配好。每对机器只需做一次。

**0a — 在主节点上获取本仓库**

```bash
git clone <your-repo-url> dual_ds_dsv4    # 把 <your-repo-url> 换成你拿到的地址，
cd dual_ds_dsv4                            # 或者直接用 scp/rsync 把目录拷过来
```

DGX Spark 自带 `git`；若没有，`sudo apt-get install -y git`。

**0b — 安装脚本用到的主机工具**（主节点**和**工作节点上都要装）

```bash
sudo apt-get update && sudo apt-get install -y git jq rsync bc curl netcat-openbsd iproute2
pip install --user "huggingface_hub[cli,hf_transfer]" aiohttp openai
#   jq → 解析 JSON   rsync → 拷权重   bc → bench 计时   hf → 下载权重
#   aiohttp → bench_openai.py    openai → 步骤 5 的客户端示例
```

> ⚠️ `pip --user` 会把 `hf` 装到 `~/.local/bin`，而脚本用的非登录 SSH shell 的
> `PATH` 里**没有**它——否则下载权重会报 `hf: command not found`。在**两个**节点上
> 都把它加到 PATH：
> ```bash
> echo 'export PATH=$HOME/.local/bin:$PATH' | tee -a ~/.bashrc ~/.profile
> ```
> 如果 `docker ps` 需要 `sudo`，把自己加进 docker 组（两个节点都做）：
> `sudo usermod -aG docker $USER && newgrp docker`。

**0c — 登录 HuggingFace**（拉取约 158 GB 权重需要）

```bash
hf auth login        # 粘贴 https://huggingface.co/settings/tokens 里的 token
```

还要先用同一个账号打开 <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>
**接受许可 / 申请访问权限**——否则即使登录成功，下载也会 401。

**0d — 配好 RoCE 网络 + 免密 SSH**（全新机器对独有的步骤）

- 用 QSFP 线把两台 Spark **直连**：端口 1 ↔ 端口 1（承载 200.x 子网）、
  端口 2 ↔ 端口 2（201.x）。不用交换机。
- **给每个 RoCE 网卡配一个同子网的静态 IP**——例如主节点
  `192.168.200.43/24`、工作节点 `192.168.200.45/24`（在 `enp1s0f0np0` 上），
  以及 `192.168.201.{43,45}/24`（在 `enP2p1s0f0np0` 上）。NVIDIA 的
  [`dgx-spark-playbooks`](https://github.com/NVIDIA/dgx-spark-playbooks)
  里的 `discover-sparks` 会自动完成线路发现**和** SSH 密钥分发；也可以手动配静态
  IP（netplan / `nmcli`）。
- 主节点 → 工作节点的**免密 SSH**：
  ```bash
  ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_shared -N ''     # 已有密钥可跳过
  ssh-copy-id -i ~/.ssh/id_ed25519_shared.pub <worker-ip>
  ssh <worker-ip> hostname                                     # 应免密返回工作节点名
  ```
- *（推荐）* 内核一旦 OOM 就会重启整台机器（`vm.panic_on_oom=1`）。内置看门狗会保护
  你，但你也可以执行 `sudo sysctl vm.panic_on_oom=0`。

**确认你已就绪**——下面这些在**两个**节点上都应成立：

- [ ] `ibdev2netdev | grep -i up` 显示两个 RoCE 网卡 Up（`rocep1s0f0` + `roceP2p1s0f0`）
- [ ] `ssh <worker-ip> hostname` 免密返回工作节点名
- [ ] `docker ps` 无需 `sudo`（你的用户在 `docker` 组里）
- [ ] 每个节点有约 200 GB 空闲磁盘放权重

### 步骤 1 — 把你的机器信息告诉项目

```bash
cd dual_ds_dsv4

# 自动检测 RoCE 网卡名与 IP（传入工作节点的 SSH 地址）：
bash scripts/discover.sh 192.168.200.45
```

它会打印类似：

```
  NODES="192.168.200.43 192.168.200.45"
  IFACES="enp1s0f0np0,enP2p1s0f0np0"
  TRANSFER_PEER="192.168.201.45"
  ✅ NIC name sets match across all hosts.
```

然后创建配置、把上面三行粘进去：

```bash
cp cluster.conf.example cluster.conf
nano cluster.conf        # 或 vim/$EDITOR——粘贴 NODES / IFACES / TRANSFER_PEER
```

> ⚠️ **头号错误：** `NODES` 必须是 `discover.sh` 打印出来的 **RoCE 网卡 IP**——
> 而**不是**你用来 SSH 登录的管理网/Wi-Fi IP。用错 IP 会让两块 GPU 走慢速 TCP
> 通信（慢约 10 倍）或直接永远卡住。下一步会替你拦截这个错误，所以直接信
> `discover.sh` 就好。

### 步骤 2 — 下载模型权重（约 158 GB，约 30–45 分钟，一次性）

```bash
bash scripts/01-download-weights.sh        # 主节点下载，再经 201.x 拷给工作节点
```

如果两个节点都已有
`~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/`，可跳过本步。

### 步骤 3 — 拉起服务

```bash
bash scripts/quickstart.sh
```

它依次跑三个步骤，且可安全地重复运行：

| 阶段 | 做什么 | 耗时 |
|---|---|---|
| `00-precheck` | 校验网卡、SSH、RoCE 地址绑定、端口、权重、内存 | ~30 s |
| `02-pull-image` | 在两个节点拉取 9 GB 的 vLLM 镜像（已有则跳过） | 0–5 分钟 |
| `03-start-serve` | 先工作节点、再主节点；等待就绪；校验 NCCL 走 RoCE；预热 | ~5–6 分钟 |

**“成功了”长这样**——`03-start-serve` 结尾会打印：

```
✅ Service ready: http://127.0.0.1:8000
  via NET/IB channels: 64   |   via NET/Socket (fallback): 0
  ✅ PASS — cross-node TP is on dual RoCE.
```

如果它超时或报错，请看打印出来的日志末尾，并查阅
[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)——文末的“错误 → 修复”对照表
几乎能把每种失败映射到一行修复。

### 步骤 4 — 确认模型能回答

```bash
bash bench/smoke.sh        # 发 5 条提示（中/英/日 + 数学 + 代码）；期望出现 391
```

成功会打印 `✅ PASS: 5/5 prompts answered, math returned 391.`

### 步骤 5 — 真正使用模型

它讲的是 **OpenAI Chat Completions** 协议，所以任何 OpenAI 客户端都能用——
把地址指到 `http://127.0.0.1:8000/v1` 即可（在主节点上跑，或从你的笔记本开一个
SSH 隧道 `ssh -L 8000:127.0.0.1:8000 <head>`）。

**curl：**

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Flash",
    "messages": [{"role": "user", "content": "Explain TP=2 in one sentence."}],
    "max_tokens": 128
  }' | jq -r '.choices[0].message.content'
```

**Python（官方 `openai` 包）：**

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

常用端点：`/v1/chat/completions`、`/v1/completions`、`/v1/models`
（列出 `max_model_len`）、`/metrics`（Prometheus，含 MTP 接受率）、
`/health`、`/docs`（交互式 API 浏览器）。

### 步骤 6 — 运行时监控

```bash
tail -f ~/dsv4_logs/vllm-dsv4-rank0.log        # 主节点服务日志
tail -f ~/dsv4_logs/watchdog-*.log             # OOM 看门狗（自动启动）
docker stats vllm-dsv4-mn                       # 实时内存/CPU
```

**看门狗**在后台运行，一旦空闲内存危险地下降就会硬杀 vLLM——这正是保护你不被
`panic_on_oom` 整机重启的东西。让它一直开着。

### 步骤 7 — 停止（以及切换配方）

```bash
bash scripts/04-stop.sh --rm                    # 干净停服 + 删除容器
```

要切换运行档位（例如更长输出、或批量吞吐），编辑 `cluster.conf` 里的 `RECIPE=`，
然后用 `--rm` 停服、再重新跑 `03-start-serve.sh`。配方对照表见
[`docs/RECIPES.md`](docs/RECIPES.md)。

### 换一对 Spark 机器

只要用新机器重复步骤 1–3：`discover.sh` 找到它们的网卡/IP，你粘进 `cluster.conf`，
`00-precheck.sh` 在拉起任何东西前校验新连接。其它什么都不用改——镜像、模型、
GPU 架构相关设置在每台 DGX Spark 上都一样。

## 项目结构

```
.
├── cluster.conf.example      # 所有站点相关参数（IP、网卡、模型、端口、配方）
├── config/
│   ├── nccl-env.sh           # 保守的 NCCL 环境（7 个被禁特性 + 网卡绑定）
│   └── recipes/
│       ├── interactive-200k.env   # 生产默认：开 MTP，131k 上下文，OSL<=8k
│       ├── long-output.env        # 关 MTP，OSL<=12k
│       └── concurrent-32k.env     # MAX_NUM_SEQS=4，峰值聚合 ~35 tok/s
├── scripts/
│   ├── discover.sh           # 自动检测 RoCE 网卡名 + IP → 粘进 cluster.conf
│   ├── quickstart.sh         # 依次跑 00 → 02 → 03
│   ├── 00-precheck.sh        # 网卡 Up + NODES-IP 在 RoCE 上 + 端口空闲、ssh、权重、内存
│   ├── 01-download-weights.sh# 主节点 hf 下载 + 经 201.x rsync 给工作节点
│   ├── 02-pull-image.sh      # 在所有节点并行拉镜像
│   ├── 03-start-serve.sh     # 先工作节点、等就绪、校验 NCCL/IB、预热
│   ├── 04-stop.sh            # 优雅停服；--rm 同时删容器
│   ├── run-node.sh           # 单节点启动器（docker run + sed 补丁 + vllm serve）
│   ├── oom-watchdog.sh       # panic_on_oom 守护（由 03 自动启动）
│   ├── prewarm.py            # 渐进式预热；消除首请求 TTFT 尖峰
│   └── _lib.sh               # cluster.conf 加载器 + ssh_node / for_each_node / push_node
├── bench/
│   ├── bench_openai.py       # 异步 OpenAI 协议客户端（TTFT、ITL、聚合 tok/s）
│   ├── smoke.sh              # 5 条提示冒烟（中/英/日 + 数学 + 代码）
│   ├── run_full_bench.sh     # 复现已发布的性能矩阵
│   ├── realprompt_sweep.sh   # 真实（非退化）提示按 ISL 扫描
│   └── quality_cases.json    # 10 个功能测试（数学、代码、多语、工具调用、长上下文检索）
└── docs/
    ├── REQUIREMENTS.md       # 安装前需要准备什么
    ├── ARCHITECTURE.md       # 模型、量化、拓扑、mp 执行器、NCCL
    ├── PERFORMANCE.md        # 可复现的数据
    ├── TROUBLESHOOTING.md    # ⚠️ 11 个 sm121/V4-Flash 坑 + 逐错误对照表
    └── RECIPES.md            # 工作负载 → 配方 选择
```

## 出问题时

**先读 [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)**——里面有 11 个已记录的
坑（`panic_on_oom`、Marlin 第 42 层损坏、MTP `aux_stream_dict`、CUDA-graph 与
DeepGEMM 冲突、NCCL GDAKI、KV 池大小不确定性……），文末附逐条错误信息对照表。

几个高频问题：

- **英文输出乱码** → 没设 `VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42`（坑 3）
- **两个节点都重启、没有日志** → `panic_on_oom`；保持看门狗开启（坑 1）
- **rank 1 段错误 `gdaki`** → NCCL 保守环境变量没传进去（坑 6）
- **低于权重大小就启动失败** → 把 `GPU_MEM_UTIL` 提到 ≥ 0.70（坑 5）
- **解码慢约 10 倍，或 rank 1 一直连不上** → `NODES` 填了管理网 IP，而不是 RoCE 网卡
  IP。跑 `bash scripts/discover.sh`；`00-precheck.sh` 现在会在启动前拦截这个错误。

## 优雅停服

```bash
bash scripts/04-stop.sh         # 在所有节点上 SIGTERM→SIGKILL vllm
bash scripts/04-stop.sh --rm    # 并且 docker rm -f（切换配方时推荐）
```

## 如何升级

当有更新的社区镜像发布时：

1. 更新 `cluster.conf` 里的 `DOCKER_IMAGE`。
2. `bash scripts/04-stop.sh --rm && bash scripts/02-pull-image.sh`。
3. `bash scripts/03-start-serve.sh`。
4. 如果新镜像已修复 MTP `aux_stream_dict` bug，删掉 `scripts/run-node.sh` 里的
   `sed` 那一行。如果 FlashInfer autotune 已修复，去掉
   `--no-enable-flashinfer-autotune`，预填充很可能提速。

## 致谢与渊源

- 硬件/网络/UMA/NCCL 基础与
  [`stacked-sparks-trtllm`](../stacked-sparks-trtllm)（同两台 Spark 上的
  TRT-LLM + Qwen3）共享。
- 镜像：为 ARM64/sm_121 构建的社区 vLLM 分支 `lmxxf/vllm-deepseek-v4-dgx-spark`。
- sm_121 相关修复（MLA-sparse Triton kernel、Marlin→DeepGEMM 第 42 层、MTP 加载器）
  来自 DGX-Spark vLLM 社区（`lmxxf`、`eugr`、`jasl9187`）。

## 许可证

Apache-2.0——见 [`LICENSE`](LICENSE)。
