"""Minimal streaming client for the vLLM OpenAI-compatible endpoint.

Deliberately built on ``requests`` + hand-parsed SSE rather than the ``openai``
SDK: fewer moving parts, no SDK-version drift, and full control over the two
timeouts that matter here (connect vs. time-between-bytes -- a 120k-token
prefill can legitimately take minutes before the first token arrives).

Events yielded by :meth:`LLMClient.stream_chat`::

    {"type": "content",   "text": "..."}          # visible answer delta
    {"type": "reasoning", "text": "..."}          # thinking delta (if the model emits any)
    {"type": "finish",    "reason": "stop"|"length"|...}
    {"type": "usage",     "usage": {"prompt_tokens": n, "completion_tokens": n, ...}}

Thinking mode (``thinking="high"|"max"``) asks DeepSeek-V4 to reason inside
``<think>...</think>`` before answering. When the server returns inline
``reasoning</think>answer`` in ``content``, :class:`ThinkSplitter` turns it
into reasoning/content events. If the server separates it
(``reasoning_content`` deltas), those are used as-is and no splitting happens.
"""
from __future__ import annotations

import json
from typing import Iterator, Optional

import requests

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
THINKING_LEVELS = ("high", "max")        # what the DeepSeek-V4 chat template accepts


class LLMError(Exception):
    """The server rejected the request or the stream broke."""


class ThinkSplitter:
    """Split an inline ``reasoning</think>answer`` content stream into events.

    Everything before the first ``</think>`` is reasoning, everything after is
    the answer.  A tag cut across two deltas is held back until it resolves.
    """

    def __init__(self) -> None:
        self.in_reasoning = True
        self._buf = ""
        self._started = False

    def feed(self, text: str) -> list[dict]:
        if not self.in_reasoning:
            return [{"type": "content", "text": text}] if text else []
        self._buf += text
        if not self._started:
            stripped = self._buf.lstrip()
            if THINK_OPEN.startswith(stripped):          # maybe a leading <think>, wait for more
                return []
            if stripped.startswith(THINK_OPEN):
                self._buf = stripped[len(THINK_OPEN):]
            self._started = True
        out: list[dict] = []
        if THINK_CLOSE in self._buf:
            reasoning, answer = self._buf.split(THINK_CLOSE, 1)
            self.in_reasoning, self._buf = False, ""
            if reasoning:
                out.append({"type": "reasoning", "text": reasoning})
            if answer.lstrip("\n"):
                out.append({"type": "content", "text": answer.lstrip("\n")})
            return out
        keep = next((k for k in range(min(len(THINK_CLOSE) - 1, len(self._buf)), 0, -1)
                     if self._buf.endswith(THINK_CLOSE[:k])), 0)
        emit, self._buf = self._buf[:len(self._buf) - keep], self._buf[len(self._buf) - keep:]
        if emit:
            out.append({"type": "reasoning", "text": emit})
        return out

    def flush(self) -> list[dict]:
        """End of stream: anything held back was reasoning (cut off before ``</think>``)."""
        rest, self._buf = self._buf, ""
        if not rest:
            return []
        return [{"type": "reasoning" if self.in_reasoning else "content", "text": rest}]


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: Optional[str] = None,
        connect_timeout: float = 5.0,
        read_timeout: float = 900.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self._max_model_len: Optional[int] = None

    # ------------------------------------------------------------ discovery
    @property
    def root_url(self) -> str:
        return self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.root_url}/health", timeout=self.connect_timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def models(self) -> list[dict]:
        r = requests.get(f"{self.base_url}/models", timeout=self.connect_timeout)
        r.raise_for_status()
        return r.json().get("data", [])

    def resolve_model(self) -> str:
        """Return the model id, discovering it from /v1/models on first use."""
        if self.model and self._max_model_len is not None:
            return self.model
        try:
            data = self.models()
        except requests.RequestException as e:
            if self.model:
                return self.model
            raise LLMError(f"cannot reach {self.base_url}/models: {e}") from e
        if not data:
            raise LLMError("server lists no models")
        chosen = None
        if self.model:
            chosen = next((m for m in data if m.get("id") == self.model), None)
        if chosen is None:
            chosen = data[0]
            self.model = chosen.get("id")
        self._max_model_len = chosen.get("max_model_len")
        return self.model

    @property
    def max_model_len(self) -> Optional[int]:
        if self._max_model_len is None:
            try:
                self.resolve_model()
            except LLMError:
                return None
        return self._max_model_len

    # ------------------------------------------------------------- streaming
    def stream_chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        thinking: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> Iterator[dict]:
        model = self.resolve_model()
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        splitter = None
        is_v41 = 'deepseek-v4.1' in model.lower()
        if thinking:
            if thinking not in THINKING_LEVELS:
                raise ValueError(f"thinking must be one of {THINKING_LEVELS} or None, got {thinking!r}")
            thinking_key = 'enable_thinking' if is_v41 else 'thinking'
            payload["chat_template_kwargs"] = {thinking_key: True, "reasoning_effort": thinking}
            splitter = ThinkSplitter()
        elif is_v41:
            # V4.1 defaults to thinking=True when this argument is omitted.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if extra:
            payload.update(extra)
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                stream=True,
                timeout=(self.connect_timeout, self.read_timeout),
            )
        except requests.RequestException as e:
            raise LLMError(f"cannot reach the model server: {e}") from e
        if resp.status_code != 200:
            body = resp.text[:600]
            resp.close()
            raise LLMError(f"server returned HTTP {resp.status_code}: {body}")
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    yield {"type": "usage", "usage": obj["usage"]}
                for ch in obj.get("choices") or []:
                    delta = ch.get("delta") or {}
                    r = delta.get("reasoning_content") or delta.get("reasoning")
                    if r:
                        splitter = None          # server separates reasoning itself
                        yield {"type": "reasoning", "text": r}
                    c = delta.get("content")
                    if c:
                        if splitter:
                            yield from splitter.feed(c)
                        else:
                            yield {"type": "content", "text": c}
                    if ch.get("finish_reason"):
                        if splitter:
                            yield from splitter.flush()
                        yield {"type": "finish", "reason": ch["finish_reason"]}
        except requests.RequestException as e:
            raise LLMError(f"stream interrupted: {e}") from e
        finally:
            resp.close()
