"""The visitor's chat list and saved settings: the pure parts, kept out of the Streamlit page.

The browser keeps every conversation in IndexedDB and the generation settings in localStorage
(``components/browser_store.js``).  What it hands back is untrusted: the list and the settings are
re-validated here, conversations by :meth:`storage.Conversation.from_untrusted`.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone, tzinfo

from storage import Conversation, is_id

TITLE_CHARS = 60
MAX_CHATS = 5000                      # list entries accepted from one browser

# Generation settings: the sliders' ranges, also used to validate what the browser saved.
MAX_TOKENS_MIN, MAX_TOKENS_STEP = 64, 64
TEMPERATURE = (0.0, 1.5, 0.05)        # min, max, step
TOP_P = (0.1, 1.0, 0.05)
SYSTEM_PROMPT_CHARS = 4000


def title_of(conv: Conversation) -> str:
    """The chat's name in the list: its first question, on one line."""
    for m in conv.messages:
        text = " ".join(m["content"].split()) if m["role"] == "user" else ""
        if text:
            return text if len(text) <= TITLE_CHARS else text[:TITLE_CHARS - 1].rstrip() + "…"
    return "New chat"


def meta_of(conv: Conversation) -> dict:
    """The chat's entry in the browser's list (stored beside the conversation itself)."""
    return {"id": conv.id, "title": title_of(conv), "created": conv.created, "updated": conv.updated,
            "messages": len(conv.messages)}


def _text(value, limit: int) -> str:
    return value[:limit] if isinstance(value, str) else ""


def clean_index(obj) -> list[dict]:
    """The browser's chat list: well-formed entries only, one per id, newest first."""
    out: dict[str, dict] = {}
    for e in obj[:MAX_CHATS] if isinstance(obj, list) else []:
        if not isinstance(e, dict) or not is_id(e.get("id")):
            continue
        n = e.get("messages")
        out[e["id"]] = {"id": e["id"], "title": _text(e.get("title"), TITLE_CHARS + 1) or "Untitled",
                        "created": _text(e.get("created"), 40), "updated": _text(e.get("updated"), 40),
                        "messages": n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else 0}
    return sorted(out.values(), key=lambda e: e["updated"], reverse=True)


def upsert(index: list[dict], entry: dict) -> list[dict]:
    return sorted([e for e in index if e["id"] != entry["id"]] + [entry], key=lambda e: e["updated"], reverse=True)


def _when(ts: str, tz: tzinfo) -> datetime | None:
    try:
        t = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).astimezone(tz)


def group_by_day(index: list[dict], now: datetime, tz: tzinfo) -> list[tuple[str, list[dict]]]:
    """Newest-first list -> [(Today | Yesterday | Previous 7 days | Previous 30 days | Older, entries)]."""
    today = now.astimezone(tz).date()
    groups: dict[str, list[dict]] = {}
    for e in index:
        t = _when(e["updated"], tz)
        age = (today - t.date()).days if t else None
        label = ("Older" if age is None or age > 30 else "Today" if age <= 0 else "Yesterday" if age == 1
                 else "Previous 7 days" if age <= 7 else "Previous 30 days")
        groups.setdefault(label, []).append(e)
    order = ("Today", "Yesterday", "Previous 7 days", "Previous 30 days", "Older")
    return [(label, groups[label]) for label in order if label in groups]


def _number(value, lo: float, hi: float, step: float) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return None
    return min(hi, max(lo, round(round(value / step) * step, 4)))


def clean_settings(obj, defaults: dict, *, max_tokens_cap: int, thinking_options) -> dict:
    """Settings saved by the browser, each one checked against the current ranges; anything
    missing or malformed keeps its default (so a lowered server cap simply clamps)."""
    out = dict(defaults)
    if not isinstance(obj, dict):
        return out
    if obj.get("thinking") in thinking_options:
        out["thinking"] = obj["thinking"]
    n = _number(obj.get("max_tokens"), MAX_TOKENS_MIN, max_tokens_cap, MAX_TOKENS_STEP)
    if n is not None:
        out["max_tokens"] = min(int(n), max_tokens_cap)
    for name, (lo, hi, step) in (("temperature", TEMPERATURE), ("top_p", TOP_P)):
        v = _number(obj.get(name), lo, hi, step)
        if v is not None:
            out[name] = v
    if isinstance(obj.get("system_prompt"), str):
        out["system_prompt"] = obj["system_prompt"][:SYSTEM_PROMPT_CHARS]
    return out
