"""Process-wide singletons shared by every Streamlit session.

Plain module-level objects, on purpose: ``import runtime`` resolves through
``sys.modules`` so every session (and any test in the same process) gets the
*same* queue/logger/client no matter how ``app.py`` itself is executed.  Using
``st.cache_resource`` would tie the queue's identity to the cache -- a
"Clear cache" click would silently spawn a second queue while tickets from the
first are still running.
"""
from __future__ import annotations

import threading

from hostmon import HostMonitor
from llm_client import LLMClient
from request_queue import RequestQueue
from settings import Settings
from storage import ChatLogger

_lock = threading.Lock()
_queue: RequestQueue | None = None
_logger: ChatLogger | None = None
_client: LLMClient | None = None
_monitor: HostMonitor | None = None


def get_queue(s: Settings) -> RequestQueue:
    global _queue
    with _lock:
        if _queue is None:
            # A silent long prefill is alive until the client's read timeout gives up.
            _queue = RequestQueue(max_concurrency=s.max_concurrency, max_waiting=s.max_waiting,
                                  per_user_limit=s.per_user_limit,
                                  stale_running_s=max(900.0, s.read_timeout))
        return _queue


def get_logger(s: Settings) -> ChatLogger:
    global _logger
    with _lock:
        if _logger is None:
            _logger = ChatLogger(s.log_dir)
        return _logger


def get_client(s: Settings) -> LLMClient:
    global _client
    with _lock:
        if _client is None:
            _client = LLMClient(s.api_base, model=s.model or None,
                                connect_timeout=s.connect_timeout, read_timeout=s.read_timeout)
        return _client


def get_monitor(s: Settings) -> HostMonitor:
    global _monitor
    with _lock:
        if _monitor is None:
            _monitor = HostMonitor(s.monitor_hosts.split(), ssh_key=s.monitor_ssh_key,
                                   interval_s=s.monitor_interval_s, history_s=s.monitor_history_s).start()
        return _monitor
