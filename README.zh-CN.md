# dual_ds_dsv4

[English](README.md) | **中文**

在两台通过 QSFP 连接的 NVIDIA DGX Spark 上运行 DeepSeek-V4 / V4.1-Flash，
使用各自的 vLLM 镜像与 TP=2，提供 OpenAI 兼容 API 和共享聊天界面。
V4.1 EXL3 的站点默认配方 `v41-416k` 上限为 **425,984 tokens**，已通过
415,014 tokens 实际输入的验证；`v41-256k`（**262,144 tokens**）是保守回退。
512K 配方超出 head 节点的内存余量。旧 V4 的 `interactive-128k` 仍为 **131,072 tokens**；
`interactive-200k` 只是它的兼容别名。

项目基线见 [VERSION](VERSION)，当前状态见 [Docs/current.md](Docs/current.md)。
新部署、UMA 保护与分级验证流程见 [V4.1 部署](Docs/runbooks/v41.md)。
状态页记录实际输入长度、吞吐和内存余量。共用这两台机器前请先读
两台机器上的 `~/MEMORY.md`（内存分配与回收说明，放在仓库外以便其他项目也能看到）。

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

API 默认绑定 `0.0.0.0:8000`。操作步骤、成功标准及限制见[服务操作](Docs/runbooks/service.md)。

## 项目结构与文档入口

| 路径 | 职责 |
|---|---|
| `scripts/` | 配置加载、集群检查和服务生命周期 |
| `config/` | NCCL 参数及推理配方 |
| `chat/` | 界面、排队、流式客户端、存储和测试 |
| `bench/` | 显式执行的在线检查与性能测试 |
| `tests/` | 公共脚本和基准客户端的离线回归测试 |
| `Docs/current.md` | 当前状态、任务、阻碍及下一步 |
| `Docs/runbooks/` | [准备](Docs/runbooks/setup.md)、[服务](Docs/runbooks/service.md)、[聊天](Docs/runbooks/chat.md)、[检查](Docs/runbooks/checks.md)、[TensorFold（评估）](Docs/runbooks/tensorfold.md) |
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

许可证：[Apache-2.0](LICENSE)。现有运行时基于 `lmxxf` 等 DGX Spark vLLM
社区实现，本地 `interllm-dsv4` 历史测试证据保留在 [ArchiveDocs](ArchiveDocs/)。
