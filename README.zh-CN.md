# dual_ds_dsv4

[English](README.md) | **中文**

在两台通过 QSFP 连接的 NVIDIA DGX Spark 上运行 DeepSeek-V4 / V4.1-Flash，
使用 TP=2，提供 OpenAI 兼容 API 和共享聊天界面。2026-10-15 起网站以
[TensorFold](Docs/runbooks/tensorfold.md) 为后端（精确设置，解码约为 vLLM 版的 1.6 倍、长输入预填充约 1.9 倍），
专用 vLLM 镜像作为回退。
V4.1 EXL3 的站点默认配方 `v41-416k` 上限为 **425,984 tokens**，已通过
415,014 tokens 实际输入的验证；`v41-256k`（**262,144 tokens**）是保守回退。
512K 配方超出 head 节点的内存余量。旧 V4 的 `interactive-128k` 仍为 **131,072 tokens**；
`interactive-200k` 只是它的兼容别名。

项目基线见 [VERSION](VERSION)，当前状态见 [Docs/current.md](Docs/current.md)。
新部署、UMA 保护与分级验证流程见 [V4.1 部署](Docs/runbooks/v41.md)。
状态页记录实际输入长度、吞吐和内存余量。共用这两台机器前请先读
两台机器上的 `~/MEMORY.md`（内存分配与回收说明，放在仓库外以便其他项目也能看到）。
完整的研究记录（剖析、实验、框架评估、精度原则与结论）见[研究文件组](Knowledge/research/README.md)。

## 安装与使用

先按[环境准备](Docs/runbooks/setup.md)配置 Docker、SSH、主机工具、权重和
`cluster.conf`，不要覆盖已有配置。现有模型、镜像与网络布局见[架构](Knowledge/architecture.md)。

完成准备后，在主节点执行：

```bash
bash scripts/00-precheck.sh       # 检查集群
bash scripts/quickstart.sh        # 拉镜像、启动两端、预热；会改变服务状态
CHAT_LIVE_TESTS=1 bash bench/smoke.sh  # 五条真实推理请求
bash chat/run_chat.sh --daemon   # 可选聊天界面，端口 9450
bash scripts/04-stop.sh --rm     # 停止并移除模型容器
```

vLLM 路径的 API 绑定 `0.0.0.0:8000`；TensorFold 路径由单请求网关占用 8000 端口（本站对局域网开放、无鉴权），
TensorFold 本身只监听 `127.0.0.1:8100`。
操作步骤、成功标准及限制见[服务操作](Docs/runbooks/service.md)与 [TensorFold runbook](Docs/runbooks/tensorfold.md)。

## 项目结构与文档入口

| 路径 | 职责 |
|---|---|
| `scripts/` | 配置加载、集群检查和服务生命周期 |
| `config/` | NCCL 参数及推理配方 |
| `chat/` | 界面、排队、流式客户端、存储和测试 |
| `bench/` | 显式执行的在线检查与性能测试 |
| `tests/` | 公共脚本和基准客户端的离线回归测试 |
| `Docs/current.md` | 当前状态、任务、阻碍及下一步 |
| `Docs/runbooks/` | [准备](Docs/runbooks/setup.md)、[服务](Docs/runbooks/service.md)、[聊天](Docs/runbooks/chat.md)、[检查](Docs/runbooks/checks.md)、[TensorFold](Docs/runbooks/tensorfold.md) |
| `Docs/kedb/` | [内存](Docs/kedb/memory.md)、[运行时](Docs/kedb/runtime.md)、[测量](Docs/kedb/measurement.md)已知问题 |
| `Docs/playbooks/` | [启动与网络诊断](Docs/playbooks/startup.md) |
| `Docs/postmortems/` | [历史内存压力事件复核](Docs/postmortems/memory-pressure.md) |
| `Knowledge/` | [架构](Knowledge/architecture.md)、[模块职责](Knowledge/code-layout.md)、[配方](Knowledge/recipes.md)、[性能证据](Knowledge/performance.md) |
| `ArchiveDocs/` | 被替代的指南与历史证据，不作为当前操作依据 |

## 开发

遵循 [AGENTS.md](AGENTS.md)；`CLAUDE.md` 是指向它的符号链接。
其中原样引入固定版本的 [dev-protocol](https://github.com/fliaght/dev-protocol)。

```bash
rg -n '关键词|错误信息' Docs/ Knowledge/
bash scripts/check.sh
```

检查入口执行离线校验与测试，包含本地模拟 API；不启动模型容器或发送真实推理请求。
依赖与在线测试入口见[检查流程](Docs/runbooks/checks.md)。

许可证：[PolyForm Noncommercial 1.0.0](LICENSE)。非商业用途免费；商业用途须另获作者 Yi Luo 许可。
第三方部分保留各自许可证：`config/tensorfold/` 中的 TensorFold 配方补丁（Apache-2.0，见其 `NOTICE`）、
`chat/static/fonts/` 中的 Inter 字体（SIL OFL 1.1）。此前以 Apache-2.0 发布的版本仍适用原条款。
现有运行时基于 `lmxxf` 等 DGX Spark vLLM 社区实现及本地 `interllm-dsv4` 实验。
