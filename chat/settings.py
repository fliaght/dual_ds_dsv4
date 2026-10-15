"""All knobs for the chat UI, read from environment variables (CHAT_*).

``chat/run_chat.sh`` derives sensible defaults from cluster.conf (API port,
model id, recipe MAX_NUM_SEQS -> queue concurrency) before launching.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    if isinstance(default, bool):
        return v.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v


@dataclass(frozen=True)
class Settings:
    app_title: str = _env("CHAT_TITLE", "DeepSeek-V4-Flash on 2× DGX Spark")
    api_base: str = _env("CHAT_API_BASE", "http://127.0.0.1:8000/v1")
    model: str = _env("CHAT_MODEL", "")                       # "" -> discover via /v1/models
    log_dir: Path = Path(_env("CHAT_LOG_DIR", os.path.expanduser("~/dsv4_logs/chat")))   # server-side JSONL logs

    # admission queue (see request_queue.py).  max_concurrency should equal the
    # vLLM recipe's MAX_NUM_SEQS: 1 for interactive-128k / long-output, 4 for concurrent-32k.
    max_concurrency: int = _env("CHAT_MAX_CONCURRENCY", 1)
    max_waiting: int = _env("CHAT_MAX_WAITING", 32)
    max_wait_s: float = _env("CHAT_MAX_WAIT_S", 600.0)
    per_user_limit: int = _env("CHAT_PER_USER_LIMIT", 1)

    # browser-side history: newest N messages kept in localStorage (and in context)
    max_local_messages: int = _env("CHAT_MAX_LOCAL_MESSAGES", 400)

    # generation defaults / caps.  8192 is the V4 MTP recipes' OSL ceiling; run_chat.sh raises it for V4.1.
    default_max_tokens: int = _env("CHAT_DEFAULT_MAX_TOKENS", 2048)
    max_tokens_cap: int = _env("CHAT_MAX_TOKENS_CAP", 8192)
    default_temperature: float = _env("CHAT_DEFAULT_TEMPERATURE", 0.7)
    default_top_p: float = _env("CHAT_DEFAULT_TOP_P", 0.95)
    # DeepSeek-V4 thinking mode: "off" | "high" | "max".  Reasoning tokens count
    # toward max_tokens, so thinking replies need a larger reply budget.
    default_thinking: str = _env("CHAT_DEFAULT_THINKING", "off")
    context_token_budget: int = _env("CHAT_CONTEXT_TOKEN_BUDGET", 96000)
    system_prompt: str = _env("CHAT_SYSTEM_PROMPT", "You are a helpful, precise assistant. Answer in the user's language.")

    # sidebar "Performance" panel (hostmon.py): unified memory + GPU of every node.
    # Hosts are space-separated, head first; run_chat.sh fills them from cluster.conf NODES.
    monitor_enabled: bool = _env("CHAT_MONITOR", True)
    monitor_hosts: str = _env("CHAT_MONITOR_HOSTS", "")                  # "" -> this machine only
    monitor_ssh_key: str = _env("CHAT_MONITOR_SSH_KEY", "")
    monitor_interval_s: float = _env("CHAT_MONITOR_INTERVAL_S", 2.0)
    monitor_history_s: float = _env("CHAT_MONITOR_HISTORY_S", 300.0)
    monitor_watchdog_mib: int = _env("CHAT_MONITOR_WATCHDOG_MIB", 4096)  # oom-watchdog kill line

    # HTTP timeouts to the model server (seconds)
    connect_timeout: float = _env("CHAT_CONNECT_TIMEOUT", 5.0)
    read_timeout: float = _env("CHAT_READ_TIMEOUT", 900.0)


def load_settings() -> Settings:
    return Settings()
