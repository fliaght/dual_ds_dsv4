"""Build the message list sent to the model, keeping it inside the context window.

We don't ship the tokenizer, so token counts are *estimated*: each CJK character
counts as one token, everything else as ~1 token per 3.5 characters.  That is
pessimistic for English and about right for Chinese/Japanese, which is the safe
direction -- the budget is the model's 131k window minus the reply budget minus
a margin, so a small over-estimate just trims one more old turn.
"""
from __future__ import annotations

import math
import re
from typing import Callable

_CJK = re.compile(r"[　-ヿ㐀-䶿一-鿿豈-﫿＀-￯가-힯]")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    rest = len(text) - cjk
    return cjk + int(rest / 3.5) + 1


def reasoning_meta(reasoning: str, thinking: bool, count: Callable[[str], int | None]) -> dict:
    """This reply's thinking tokens (its thinking text counted by the server's tokenizer, ``count``; estimated
    when that fails), or nothing when thinking was off and nothing was thought."""
    if not (thinking or reasoning):
        return {}
    n = count(reasoning)
    return {"reasoning_tokens": n if n is not None else estimate_tokens(reasoning), "reasoning_exact": n is not None}


def _count(value) -> int | None:
    """An int from a (browser-restored, untrusted) meta field, else None."""
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    return int(value) if ok else None


def token_caption(meta: dict) -> list[str]:
    """One reply's own numbers: ``input N tok`` (everything sent for it: system prompt, kept history, question;
    the server's count) and ``output M tok (thinking K)`` (generated, thinking included). Not the conversation's
    size: that is the context gauge (:func:`context_gauge`). ``≈`` marks an estimate."""
    prompt, reply = _count(meta.get("prompt_tokens")), _count(meta.get("completion_tokens"))
    bits = [f"input {prompt:,} tok"] if prompt is not None else []
    if reply is not None:
        part = f"output {'' if prompt is not None else '≈'}{reply:,} tok"
        thought = _count(meta.get("reasoning_tokens"))
        if thought is not None:
            part += f" (thinking {'' if meta.get('reasoning_exact') else '≈'}{thought:,})"
        bits.append(part)
    return bits


def context_gauge(tokens: int, exact: bool, limit: int, dropped: int, reply_tokens: int) -> tuple[float, str, list[str]]:
    """The conversation's context gauge: (bar fraction, label, warnings). ``tokens`` is what the next question
    would be sent with, before the question itself; ``reply_tokens`` is the current max reply length."""
    share = tokens / limit if limit else 0.0
    pct = "<0.1" if 0 < share < 0.001 else f"{100 * share:.1f}"
    label = f"Context in use: {'' if exact else '≈'}{tokens:,} / {limit:,} tokens ({pct}%)"
    notes = []
    if dropped:
        notes.append(f"{dropped} oldest message{'s no longer fit and are' if dropped > 1 else ' no longer fits and is'}"
                     " not sent to the model")
    if tokens + reply_tokens > limit:
        notes.append(f"with replies up to {reply_tokens:,} tokens the next question will not fit: "
                     "start a new chat or lower Max reply tokens")
    elif share >= 0.8:
        notes.append("context nearly full: start a new chat to keep earlier messages in view")
    return min(share, 1.0), label, notes


def build_context(messages: list[dict], system_prompt: str, budget_tokens: int) -> tuple[list[dict], int, int]:
    """Return (api_messages, dropped_count, estimated_tokens).

    Keeps the system prompt and the newest turns; drops oldest turns first.
    Only ``role``/``content`` are sent -- reasoning, timings, etc. stay local.
    Messages with empty content (e.g. an interrupted reply) are skipped.
    """
    api = []
    if system_prompt.strip():
        api.append({"role": "system", "content": system_prompt.strip()})
    base = sum(estimate_tokens(m["content"]) for m in api)

    turns = [{"role": m["role"], "content": m["content"]} for m in messages
             if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()]
    kept: list[dict] = []
    total = base
    for m in reversed(turns):
        cost = estimate_tokens(m["content"])
        if kept and total + cost > budget_tokens:
            break
        if not kept and total + cost > budget_tokens:
            # even the newest message alone doesn't fit
            return api, len(turns), total + cost
        kept.append(m)
        total += cost
    kept.reverse()
    # never start the history with an assistant turn
    while kept and kept[0]["role"] == "assistant":
        kept.pop(0)
    return api + kept, len(turns) - len(kept), total
