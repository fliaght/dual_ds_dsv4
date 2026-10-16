# dual_ds_dsv4

[English](README.md) | **中文**

在两台通过 QSFP 连接的 NVIDIA DGX Spark 上运行 DeepSeek-V4 / V4.1-Flash，
使用 TP=2，提供 OpenAI 兼容 API 和共享聊天界面。2026-10-15 起网站以
TensorFold 为后端（精确设置，解码约为 vLLM 版的 1.6 倍、长输入预填充约 1.9 倍），
专用 vLLM 镜像作为回退。
V4.1 EXL3 的站点默认配方 `v41-416k` 上限为 **425,984 tokens**，已通过
415,014 tokens 实际输入的验证；`v41-256k`（**262,144 tokens**）是保守回退。
512K 配方超出 head 节点的内存余量。旧 V4 的 `interactive-128k` 仍为 **131,072 tokens**；
`interactive-200k` 只是它的兼容别名。

项目基线见 [VERSION](VERSION)。共用这两台机器前请先读两台机器上的 `~/MEMORY.md`
（内存分配与回收说明，放在仓库外以便其他项目也能看到）。

## 安装与使用

先准备 Docker、SSH、主机工具、权重和 `cluster.conf`（以 `cluster.conf.example` 为模板），
不要覆盖已有配置。

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

## 项目结构

| 路径 | 职责 |
|---|---|
| `scripts/` | 配置加载、集群检查和服务生命周期 |
| `config/` | NCCL 参数及推理配方 |
| `chat/` | 界面、排队、流式客户端、存储和测试 |
| `bench/` | 显式执行的在线检查与性能测试 |
| `tests/` | 公共脚本和基准客户端的离线回归测试 |

## 开发

```bash
bash scripts/check.sh
```

检查入口执行离线校验与测试，包含本地模拟 API；不启动模型容器或发送真实推理请求。
所需 Python 依赖见 `requirements-dev.txt`。

许可证：[PolyForm Noncommercial 1.0.0](LICENSE)。非商业用途免费；商业用途须另获作者 Yi Luo 许可。
第三方部分保留各自许可证：`config/tensorfold/` 中的 TensorFold 配方补丁（Apache-2.0，见其 `NOTICE`）、
`chat/static/fonts/` 中的 Inter 字体（SIL OFL 1.1）。此前以 Apache-2.0 发布的版本仍适用原条款。
现有运行时基于 `lmxxf` 等 DGX Spark vLLM 社区实现及本地 `interllm-dsv4` 实验。
