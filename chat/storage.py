"""Conversation model + server-side audit log.

Where things live
-----------------
* The **conversation** belongs to the visitor's browser: the app hands it to the
  ``local_store`` component, which keeps it in ``localStorage``.  On the next
  visit the component hands it back.  Because that data comes from the client
  it is treated as untrusted and re-validated by :meth:`Conversation.from_untrusted`.
* The **server** keeps only append-only JSONL logs under ``log_dir``: one line
  per turn (every visitor) and one line per notable event.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_ID_RE = re.compile(r"^[0-9a-f]{8,64}$")
ROLES = ("user", "assistant")
STATUSES = ("ok", "interrupted", "error")
META_KEYS = {"queued_s", "ttft_s", "gen_s", "total_s", "prompt_tokens", "completion_tokens", "tok_s", "finish_reason"}
MAX_CONTENT_CHARS = 400_000


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def clean_id(value, fallback: Optional[str] = None) -> str:
    """Accept a hex id from the client, or mint a fresh one."""
    if isinstance(value, str) and _ID_RE.match(value):
        return value
    return fallback or uuid.uuid4().hex


@dataclass
class Conversation:
    id: str
    created: str = field(default_factory=now_iso)
    system_prompt: str = ""
    messages: list = field(default_factory=list)   # [{role, content, ts, reasoning?, meta?, status?, error?}]

    @staticmethod
    def new(system_prompt: str = "") -> "Conversation":
        return Conversation(id=uuid.uuid4().hex, system_prompt=system_prompt)

    def to_dict(self) -> dict:
        return asdict(self)

    def trim(self, max_messages: int) -> int:
        """Keep only the newest ``max_messages``; returns how many were dropped."""
        extra = len(self.messages) - max_messages
        if extra > 0:
            del self.messages[:extra]
            return extra
        return 0

    @staticmethod
    def from_untrusted(obj, max_messages: int, max_chars: int = MAX_CONTENT_CHARS) -> Optional["Conversation"]:
        """Rebuild a conversation from browser-supplied JSON, keeping only well-typed fields."""
        if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list):
            return None
        out = []
        for m in obj["messages"][-max_messages:]:
            if not isinstance(m, dict) or m.get("role") not in ROLES or not isinstance(m.get("content"), str):
                continue
            rec = {"role": m["role"], "content": m["content"][:max_chars]}
            if isinstance(m.get("ts"), str):
                rec["ts"] = m["ts"][:40]
            if isinstance(m.get("reasoning"), str) and m["reasoning"]:
                rec["reasoning"] = m["reasoning"][:max_chars]
            if m.get("status") in STATUSES:
                rec["status"] = m["status"]
            if isinstance(m.get("error"), str) and m["error"]:
                rec["error"] = m["error"][:300]
            if isinstance(m.get("meta"), dict):
                rec["meta"] = {k: v for k, v in m["meta"].items()
                               if k in META_KEYS and isinstance(v, (int, float, str, type(None)))}
            out.append(rec)
        sp = obj.get("system_prompt")
        created = obj.get("created")
        return Conversation(
            id=clean_id(obj.get("id")),
            created=created[:40] if isinstance(created, str) else now_iso(),
            system_prompt=sp[:4000] if isinstance(sp, str) else "",
            messages=out,
        )


class ChatLogger:
    """Append-only JSONL: ``chat-YYYY-MM-DD.jsonl`` (one line per turn, all visitors)
    and ``events-YYYY-MM-DD.jsonl`` (sessions, clears, queue rejections)."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.Lock()

    def _append(self, path: Path, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"   # never let a log line break a request
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)

    def _daily(self, prefix: str) -> Path:
        return self.log_dir / f"{prefix}-{time.strftime('%Y-%m-%d')}.jsonl"

    def log_turn(self, rec: dict) -> None:
        self._append(self._daily("chat"), {"ts": now_iso(), "kind": "turn", **rec})

    def log_event(self, event: str, **fields) -> None:
        self._append(self._daily("events"), {"ts": now_iso(), "kind": "event", "event": event, **fields})
