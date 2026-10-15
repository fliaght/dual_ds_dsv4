"""Build the message list sent to the model, keeping it inside the context window.

We don't ship the tokenizer, so token counts are *estimated*: each CJK character
counts as one token, everything else as ~1 token per 3.5 characters.  That is
pessimistic for English and about right for Chinese/Japanese, which is the safe
direction -- the budget is the model's 131k window minus the reply budget minus
a margin, so a small over-estimate just trims one more old turn.
"""
from __future__ import annotations

import re

_CJK = re.compile(r"[　-ヿ㐀-䶿一-鿿豈-﫿＀-￯가-힯]")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    rest = len(text) - cjk
    return cjk + int(rest / 3.5) + 1


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
