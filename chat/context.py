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


def token_meta(usage: dict | None, text: str, reasoning: str, est_prompt: int, thinking: bool,
               count: Callable[[str], int | None], limit: int | None) -> dict:
    """A finished reply's two token numbers for its status line.

    ``context_tokens``: the conversation as the model held it for this reply -- the server's prompt tokens
    (system prompt, kept history, new question) plus this reply's tokens, thinking included; estimated when the
    server sent no usage. Earlier replies' thinking is never sent again (``build_context`` sends role/content
    only, and DeepSeek-V4.1's template drops it too), so it is not part of later totals.
    ``reasoning_tokens``: this reply's thinking text counted by the server's tokenizer (``count``), estimated
    when that fails; absent when thinking was off and nothing was thought.
    """
    u = usage or {}
    prompt, completion = u.get("prompt_tokens"), u.get("completion_tokens")
    exact = all(isinstance(v, int) and not isinstance(v, bool) for v in (prompt, completion))
    total = prompt + completion if exact else est_prompt + estimate_tokens(text) + estimate_tokens(reasoning)
    meta = {"context_tokens": total, "context_exact": exact, "context_limit": limit}
    if thinking or reasoning:
        n = count(reasoning)
        meta.update(reasoning_tokens=n if n is not None else estimate_tokens(reasoning), reasoning_exact=n is not None)
    return meta


def _count(value) -> int | None:
    """An int from a (browser-restored, untrusted) meta field, else None."""
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    return int(value) if ok else None


def token_caption(meta: dict) -> list[str]:
    """``conversation N tok (P% of context)`` and ``thinking M tok`` (``≈`` marks an estimate); replies saved
    before these fields existed show their completion tokens as before."""
    total = _count(meta.get("context_tokens"))
    if total is None:
        old = _count(meta.get("completion_tokens"))
        return [f"{old:,} tok"] if old is not None else []
    part = f"conversation {'' if meta.get('context_exact') else '≈'}{total:,} tok"
    limit = _count(meta.get("context_limit"))
    if limit:
        share = 100 * total / limit
        part += f" ({'<0.1' if share < 0.1 else f'{share:.1f}'}% of context)"
    bits = [part]
    thought = _count(meta.get("reasoning_tokens"))
    if thought is not None:
        bits.append(f"thinking {'' if meta.get('reasoning_exact') else '≈'}{thought:,} tok")
    return bits


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
