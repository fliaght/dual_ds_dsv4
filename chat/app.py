"""Streamlit chat UI for the dual-Spark DeepSeek-V4-Flash service -- no accounts.

    bash chat/run_chat.sh --daemon   # http://<head-ip>:9450

How it works, in one paragraph: every visitor lands straight in a chat.  Their
conversation is kept in *their own browser* (localStorage, via the tiny
``components/local_store`` component) and restored on the next visit; the
server stores no per-user data.  A prompt joins the process-wide FIFO queue
(request_queue.py) -- waiting visitors see their position -- then the reply
streams from vLLM.  Every turn is appended to the server-side audit log
(``$CHAT_LOG_DIR/chat-YYYY-MM-DD.jsonl``), even if the visitor hits Stop.
"""
from __future__ import annotations

import importlib
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from context import build_context, context_gauge, estimate_tokens, reasoning_meta, token_caption  # noqa: E402
from llm_client import THINKING_LEVELS, LLMError  # noqa: E402
from request_queue import QueueFull, QueueTimeout, UserBusy  # noqa: E402
from runtime import get_client, get_logger, get_monitor, get_queue  # noqa: E402
from settings import Settings, load_settings  # noqa: E402
from storage import Conversation, clean_id, now_iso  # noqa: E402

S: Settings = load_settings()

THINKING_OPTIONS = ("off", *THINKING_LEVELS)
THINKING_LABELS = {"off": "Off", "high": "High", "max": "Max"}
DEFAULT_THINKING = S.default_thinking if S.default_thinking in THINKING_OPTIONS else "off"

st.set_page_config(page_title=S.app_title, page_icon="💬", layout="wide")

# One queue / logger / client per server process, shared by all sessions (runtime.py).
Q = get_queue(S)
LOG = get_logger(S)
LLM = get_client(S)
MON = get_monitor(S) if S.monitor_enabled else None

# Bidirectional component that reads/writes the chat in the browser's localStorage.
_HERE = os.path.dirname(os.path.abspath(__file__))
local_store = components.declare_component("dsv4_local_store", path=os.path.join(_HERE, "components", "local_store"))
STORE = "conversation"


# ------------------------------------------------------------ session state
def ss_init() -> None:
    ss = st.session_state
    if "session_id" in ss:
        return
    ss.session_id = uuid.uuid4().hex[:12]
    ss.client_id = None            # stable per-browser id, supplied by localStorage
    ss.restored = False            # browser has answered the "load" request
    ss.conv = None                 # Conversation
    ss.gen = {"max_tokens": S.default_max_tokens, "temperature": S.default_temperature,
              "top_p": S.default_top_p, "system_prompt": S.system_prompt, "thinking": DEFAULT_THINKING}
    LOG.log_event("session_start", session_id=ss.session_id, **client_info())


def client_info() -> dict:
    """Best-effort request metadata for the audit log (IP, user agent)."""
    info = {"ip": None, "user_agent": None}
    try:
        h = st.context.headers
        xff = h.get("X-Forwarded-For")
        ip = (xff.split(",")[0].strip() if isinstance(xff, str) else "") or st.context.ip_address
        ua = h.get("User-Agent")
        info["ip"] = ip if isinstance(ip, str) and ip else None          # only real strings (AppTest mocks these)
        info["user_agent"] = ua[:200] if isinstance(ua, str) and ua else None
    except Exception:  # noqa: BLE001 - headers unavailable
        pass
    return info


def client_id() -> str:
    if not st.session_state.client_id:
        st.session_state.client_id = uuid.uuid4().hex
    return st.session_state.client_id


def current_conv() -> Conversation:
    if st.session_state.conv is None:
        st.session_state.conv = Conversation.new(system_prompt=st.session_state.gen["system_prompt"])
    return st.session_state.conv


def restore_from_browser() -> None:
    """Ask the browser for its saved chat.  The component answers on its first
    render, which triggers a rerun; until then we carry on with an empty chat."""
    ss = st.session_state
    payload = local_store(action="load", store=STORE, key="ls_load", default=None)
    if not (isinstance(payload, dict) and payload.get("loaded")):
        return
    ss.client_id = clean_id(payload.get("client_id"), fallback=ss.client_id)
    saved = Conversation.from_untrusted(payload.get("conv"), max_messages=S.max_local_messages)
    if saved is not None and (ss.conv is None or not ss.conv.messages):
        ss.conv = saved              # don't clobber a chat the visitor already started this session
    ss.restored = True
    LOG.log_event("history_restored", client_id=ss.client_id, session_id=ss.session_id,
                  messages=len(saved.messages) if saved else 0, stored_chars=payload.get("chars"))


def persist_to_browser(conv: Conversation, slot: str) -> None:
    """Hand the current chat to the browser to store (empty chat -> entry removed)."""
    local_store(action="save", store=STORE, conv=conv.to_dict(), key=f"ls_save_{slot}", default=None)


def cb_clear_chat() -> None:
    ss = st.session_state
    old = ss.conv
    LOG.log_event("chat_cleared", client_id=ss.client_id, session_id=ss.session_id,
                  conv_id=old.id if old else None, messages=len(old.messages) if old else 0)
    ss.conv = Conversation.new(system_prompt=ss.gen["system_prompt"])


# ---------------------------------------------------------------- sidebar
def render_sidebar(conv: Conversation) -> None:
    with st.sidebar:
        st.button("🗑 Clear chat", on_click=cb_clear_chat, use_container_width=True,
                  disabled=not conv.messages, key="clear_chat")
        st.caption("Your chat history is stored **only in this browser** (localStorage) and comes back "
                   "when you return on the same device. Clearing removes it from this browser.")

        with st.expander("Generation settings"):
            g = st.session_state.gen
            g["system_prompt"] = st.text_area("System prompt", value=g["system_prompt"], height=90,
                                              help="Applies from the next new chat (after Clear).")
            g["thinking"] = st.radio(
                "Thinking", THINKING_OPTIONS, index=THINKING_OPTIONS.index(g.get("thinking", "off")),
                format_func=THINKING_LABELS.get, horizontal=True,
                help="High: the model reasons step by step before answering (shown under 'Reasoning'). "
                     "Max: deepest reasoning, much slower. Reasoning tokens count toward Max reply tokens, "
                     "so raise that slider when thinking is on.")
            g["max_tokens"] = st.slider("Max reply tokens", 64, S.max_tokens_cap, int(g["max_tokens"]), 64)
            g["temperature"] = st.slider("Temperature", 0.0, 1.5, float(g["temperature"]), 0.05)
            g["top_p"] = st.slider("Top-p", 0.1, 1.0, float(g["top_p"]), 0.05)

        with st.expander("Server & queue", expanded=True):
            st_ = Q.stats()
            up = LLM.health()
            st.markdown(("🟢 model server up" if up else "🔴 model server unreachable") +
                        f"  \n`{LLM.model or S.model or '?'}`" +
                        (f" · ctx {LLM.max_model_len:,}" if up and LLM.max_model_len else ""))
            c1, c2, c3 = st.columns(3)
            c1.metric("Running", f"{st_['running']}/{st_['max_concurrency']}")
            c2.metric("Waiting", st_["waiting"])
            c3.metric("Served", st_["served"])
            if st_["served"]:
                st.caption(f"avg queue wait {st_['avg_wait_s']:.1f}s · rejected {st_['rejected']}")
            st.caption("Requests are served strictly first-come-first-served; one in-flight request per visitor.")


# ------------------------------------------------------------ performance
# Categorical slots in fixed order (dataviz reference palette): head, worker, ...
SERIES_COLORS = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
GIB = 1024 ** 3


def _host_label(snap: dict, i: int, n: int) -> str:
    return "head" if snap["role"] == "head" else ("worker" if n == 2 else f"worker {i}")


def _theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # noqa: BLE001 - no browser context (AppTest)
        return "light"


def _trend_chart(rows: list[dict], field: str, title: str, labels: list[str]):
    # altair reads sys.modules['pandas'] directly; a real import waits for another session's half-done one
    # (sessions reconnecting together after a restart: Docs/kedb/runtime.md)
    importlib.import_module("pandas")
    import altair as alt
    colors = SERIES_COLORS[_theme()][:len(labels)]
    hover = alt.selection_point(fields=["t"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    base = alt.Chart(alt.Data(values=rows)).encode(
        x=alt.X("t:T", title=None, axis=alt.Axis(format="%H:%M", tickCount=3, grid=False)),
        y=alt.Y(f"{field}:Q", title=None, scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(values=[0, 50, 100])),
        color=alt.Color("host:N", scale=alt.Scale(domain=labels, range=colors),
                        legend=alt.Legend(orient="bottom", title=None)),
    )
    line = base.mark_line(strokeWidth=2)
    dots = base.mark_point(size=64, filled=True).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("host:N", title="node"), alt.Tooltip("t:T", title="time", format="%H:%M:%S"),
                 alt.Tooltip(f"{field}:Q", title=title, format=".0f")],
    ).add_params(hover)
    return (line + dots).properties(title=title, height=120)


def draw_monitor(slot) -> None:
    """Render every node's latest sample + 5-minute trend into ``slot`` (an st.empty)."""
    snaps = MON.snapshot()
    labels = [_host_label(s, i, len(snaps)) for i, s in enumerate(snaps)]
    wd = S.monitor_watchdog_mib * 1024 ** 2
    with slot.container():
        st.caption("DGX Spark memory is unified: the GPU allocates from the same pool as the CPU.")
        for label, s in zip(labels, snaps):
            name = s["hostname"] or s["host"]
            st.markdown(f"**{label}** · `{name}`" + (f" · {s['host']}" if s["hostname"] else ""))
            L = s["latest"]
            if L is None or MON.is_stale(s):
                st.caption("🔴 no data" + (f" — {s['error']}" if s["error"] else (" yet…" if L is None else "")))
                continue
            pct = L["mem_used_pct"] or 0.0
            st.progress(min(pct / 100, 1.0),
                        text=f"Memory {L['mem_used'] / GIB:.1f} / {L['mem_total'] / GIB:.1f} GiB · {pct:.0f}%")
            mem_bits = [f"available {L['mem_avail'] / GIB:.1f} GiB"]
            if L.get("gpu_proc_mem") is not None:
                mem_bits.insert(0, f"GPU processes {L['gpu_proc_mem'] / GIB:.1f} GiB")
            st.caption(" · ".join(mem_bits))
            if L["mem_avail"] < wd:
                st.error(f"Available memory is below the OOM-watchdog line ({S.monitor_watchdog_mib / 1024:.0f} GiB) "
                         f"— the watchdog kills vLLM.", icon="⛔")
            elif L["mem_avail"] < 2 * wd:
                st.warning("Low memory headroom.", icon="⚠️")
            util = L.get("gpu_util")
            st.progress(min((util or 0) / 100, 1.0), text=f"GPU {util:.0f}%" if util is not None else "GPU n/a")
            gpu_bits = [f"{L['gpu_temp']:.0f} °C" if L.get("gpu_temp") is not None else None,
                        f"{L['gpu_power']:.1f} W" if L.get("gpu_power") is not None else None,
                        f"{L['gpu_clock'] / 1000:.2f} GHz" if L.get("gpu_clock") is not None else None]
            st.caption(" · ".join(b for b in gpu_bits if b))

        rows = [{"t": int(t * 1000), "host": label, "mem": m, "gpu": g}
                for label, s in zip(labels, snaps) for t, m, g in s["history"]]
        if rows:
            st.caption(f"Last {S.monitor_history_s / 60:.0f} min")
            st.altair_chart(_trend_chart([r for r in rows if r["mem"] is not None], "mem", "Memory used %", labels),
                            width="stretch")
            st.altair_chart(_trend_chart([r for r in rows if r["gpu"] is not None], "gpu", "GPU utilization %", labels),
                            width="stretch")


@st.fragment(run_every=S.monitor_interval_s)
def monitor_panel():
    """Refreshes itself every interval without rerunning the page.  Timed fragment reruns
    never preempt a running script, so while a reply streams, handle_prompt redraws the
    returned slot instead."""
    with st.expander("Performance", expanded=True):
        slot = st.empty()
    draw_monitor(slot)
    return slot


# -------------------------------------------------------------- chat view
def _meta_caption(meta: dict) -> str:
    bits = token_caption(meta)
    if meta.get("tok_s"):
        bits.append(f"{meta['tok_s']:.1f} tok/s")
    if meta.get("ttft_s") is not None:
        bits.append(f"TTFT {meta['ttft_s']:.1f}s")
    if meta.get("queued_s"):
        bits.append(f"queued {meta['queued_s']:.0f}s")
    if meta.get("finish_reason") == "length":
        bits.append("⚠ cut off at max tokens")
    return " · ".join(bits)


def render_message(m: dict) -> None:
    with st.chat_message(m["role"]):
        if m.get("reasoning"):
            with st.expander("Reasoning", expanded=False):
                st.markdown(m["reasoning"])
        if m.get("content"):
            st.markdown(m["content"])
        status = m.get("status")
        if status == "interrupted":
            st.caption("⏹ stopped before the reply finished")
        elif status == "error":
            st.caption(f"❌ {m.get('error', 'error')}")
        elif m.get("meta"):
            st.caption(_meta_caption(m["meta"]))


def handle_prompt(conv: Conversation, prompt: str, mon_slot=None) -> None:
    ss = st.session_state
    g = ss.gen
    cid = client_id()

    # 1. keep the visitor's turn: in memory now, in their browser right away
    conv.messages.append({"role": "user", "content": prompt, "ts": now_iso()})
    conv.trim(S.max_local_messages)
    render_message(conv.messages[-1])
    persist_to_browser(conv, "pre")

    # 2. build the context window
    _, budget = context_limits(g)
    api_messages, dropped, est = build_context(conv.messages, conv.system_prompt, budget)
    if len(api_messages) == 0 or api_messages[-1]["role"] != "user":
        with st.chat_message("assistant"):
            st.error(f"This message is too long for the context window (~{est:,} tokens estimated, "
                     f"budget {budget:,}). Please shorten it.")
        return

    # 3. queue, then stream
    record = {
        "client_id": cid, "session_id": ss.session_id, **client_info(),
        "conv_id": conv.id, "turn_id": uuid.uuid4().hex[:12], "model": LLM.model or S.model,
        "params": {"max_tokens": int(g["max_tokens"]), "temperature": float(g["temperature"]),
                   "top_p": float(g["top_p"]), "thinking": g["thinking"]},
        "context": {"messages_sent": len(api_messages), "dropped_turns": dropped, "est_prompt_tokens": est},
        "prompt": prompt,
    }
    with st.chat_message("assistant"):
        status = st.status("Joining the queue…", expanded=False)
        reasoning_slot = st.empty()
        body = st.empty()
        foot = st.empty()

        try:
            ticket = Q.submit(cid)
        except (UserBusy, QueueFull) as e:
            status.update(label=f"{e} Your message was not sent.", state="error")
            LOG.log_event("rejected", client_id=cid, session_id=ss.session_id, conv_id=conv.id,
                          reason=type(e).__name__, queue=Q.stats())
            conv.messages.pop()          # don't keep an unanswered turn the visitor must resend
            return

        text, reasoning = "", ""
        usage, finish = None, None
        t_submit = time.monotonic()
        t_start = t_first = None
        done, err = False, None
        reasoning_box = None
        last_label = None
        finalized: dict = {}
        t_mon = time.monotonic()

        def refresh_monitor() -> None:
            nonlocal t_mon
            if mon_slot is not None and time.monotonic() - t_mon >= S.monitor_interval_s:
                t_mon = time.monotonic()
                draw_monitor(mon_slot)

        def on_tick(pos: int, waited: float, stats: dict) -> None:
            nonlocal last_label
            refresh_monitor()
            label = (f"In queue · position {pos} of {stats['waiting']} · waited {waited:.0f}s · "
                     f"{stats['running']}/{stats['max_concurrency']} running")
            if label != last_label:
                status.update(label=label)
                last_label = label

        try:
            Q.wait(ticket, timeout=S.max_wait_s, on_tick=on_tick)
            status.update(label="Generating…", state="running")
            t_start = time.monotonic()
            thinking = None if g["thinking"] == "off" else g["thinking"]
            for ev in LLM.stream_chat(api_messages, max_tokens=int(g["max_tokens"]),
                                      temperature=float(g["temperature"]), top_p=float(g["top_p"]),
                                      thinking=thinking):
                Q.heartbeat(ticket)
                refresh_monitor()
                kind = ev["type"]
                if kind == "content":
                    if t_first is None:
                        t_first = time.monotonic()
                    if not text and reasoning:
                        status.update(label="Generating…", state="running")
                    text += ev["text"]
                    body.markdown(text + " ▌")
                elif kind == "reasoning":
                    if t_first is None:
                        t_first = time.monotonic()
                    reasoning += ev["text"]
                    if reasoning_box is None:
                        status.update(label="Thinking…", state="running")
                        reasoning_box = reasoning_slot.expander("Reasoning", expanded=True).empty()
                    reasoning_box.markdown(reasoning)
                elif kind == "usage":
                    usage = ev["usage"]
                elif kind == "finish":
                    finish = ev["reason"]
            done = True
        except (QueueTimeout, LLMError) as e:
            err = str(e)
        finally:
            # Runs on success, on our own errors, AND when Streamlit stops the script
            # (Stop button, a new prompt, tab closed -> StopException, a BaseException):
            # free the slot and record whatever was generated.  No st.* calls in here.
            Q.release(ticket)
            finalized["meta"] = _finish_turn(conv, record, ticket, text, reasoning, usage, finish,
                                             t_submit, t_start, t_first, done=done, err=err)

        meta = finalized["meta"]
        if err:
            status.update(label="Failed — your message was not answered", state="error")
            body.error(err)
        else:
            status.update(label=f"Done · {_meta_caption(meta)}", state="complete")
            body.markdown(text)
            foot.caption(_meta_caption(meta))


def _finish_turn(conv, record, ticket, text, reasoning, usage, finish, t_submit, t_start, t_first,
                 *, done: bool, err) -> dict:
    """Append the assistant turn (even partial) to the chat and write the audit-log line."""
    now = time.monotonic()
    gen_s = (now - t_start) if t_start else 0.0
    ctoks = (usage or {}).get("completion_tokens")
    if ctoks is None and (text or reasoning):
        ctoks = estimate_tokens(text) + estimate_tokens(reasoning)
    meta = {
        "queued_s": round(ticket.waited_s, 2),
        "ttft_s": round(t_first - t_start, 2) if (t_first and t_start) else None,
        "gen_s": round(gen_s, 2),
        "total_s": round(now - t_submit, 2),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": ctoks,
        "tok_s": round(ctoks / gen_s, 1) if (ctoks and gen_s > 0) else None,
        "finish_reason": finish,
    }
    if done:                 # a stopped reply shows no status line: no /tokenize call for it
        meta.update(reasoning_meta(reasoning, record["params"]["thinking"] != "off", LLM.count_tokens))
    status = "ok" if done else ("error" if err else "interrupted")
    msg = {"role": "assistant", "content": text, "ts": now_iso(), "status": status, "meta": meta}
    if reasoning:
        msg["reasoning"] = reasoning
    if err:
        msg["error"] = err
    conv.messages.append(msg)
    conv.trim(S.max_local_messages)
    LOG.log_turn({**record, "ticket_id": ticket.id, "status": status, "error": err,
                  "response": text, "reasoning": reasoning or None, "usage": usage, "timing": meta})
    return meta


def context_limits(g: dict) -> tuple[int, int]:
    """(model context, estimated history budget): history beyond the budget is left out of a request."""
    max_len = LLM.max_model_len or (S.context_token_budget + int(g["max_tokens"]) + 2048)
    return max_len, min(S.context_token_budget, max_len - int(g["max_tokens"]) - 2048)


def context_usage(conv: Conversation) -> dict | None:
    """What the next question would be sent with (system prompt + the history that fits), counted by the
    server's chat template; earlier replies' thinking is not resent. Recounted only when the conversation or
    the settings that shape the request change."""
    if not conv.messages:
        return None
    g = st.session_state.gen
    limit, budget = context_limits(g)
    api, dropped, est = build_context(conv.messages, conv.system_prompt, budget)
    last = conv.messages[-1]
    key = (conv.id, len(conv.messages), last.get("ts"), len(last.get("content") or ""), conv.system_prompt,
           g["thinking"], len(api), dropped)
    cached = st.session_state.get("ctx_usage")
    if cached is None or cached["key"] != key:
        thinking = None if g["thinking"] == "off" else g["thinking"]
        # an empty question renders the history exactly as the next request will (thinking dropped)
        n = LLM.count_chat_tokens(api + [{"role": "user", "content": ""}], thinking=thinking)
        cached = {"key": key, "tokens": n if n is not None else est, "exact": n is not None, "dropped": dropped}
        st.session_state.ctx_usage = cached
    return dict(cached, limit=limit, reply_tokens=int(g["max_tokens"]))


def draw_context_gauge(slot, conv: Conversation) -> None:
    u = context_usage(conv)
    with slot.container():
        if u is None:
            return
        fraction, label, notes = context_gauge(u["tokens"], u["exact"], u["limit"], u["dropped"], u["reply_tokens"])
        st.progress(fraction, text=label)
        st.caption(" · ".join(f"⚠ {n}" for n in notes) or "What the model reads with your next message",
                   help="System prompt plus the conversation so far, counted by the model's tokenizer; your next "
                        "question and the reply add to it. Thinking from earlier replies is not sent again, so it "
                        "does not count. When the conversation no longer fits, the oldest messages are left out; "
                        "start a new chat to keep everything in view. The numbers under each reply are that "
                        "reply's own input and output.")


def render_chat(conv: Conversation, mon_slot=None):
    """Draw the chat; returns the context gauge's slot so main() can redraw it after a reply."""
    st.title(S.app_title)
    gauge = st.empty()
    draw_context_gauge(gauge, conv)
    if not conv.messages:
        st.caption("Ask anything. Your conversation stays in this browser; the server keeps an anonymous "
                   "log of requests for operations.")
    for m in conv.messages:
        render_message(m)
    prompt = st.chat_input("Message the model…", key="chat_box")
    if prompt and prompt.strip():
        handle_prompt(conv, prompt.strip(), mon_slot)
    return gauge


# ------------------------------------------------------------------- main
def main() -> None:
    ss_init()
    if not st.session_state.restored:
        restore_from_browser()
    conv = current_conv()
    # The monitor goes first so it is live (not a stale copy) while a reply streams.
    mon_slot = None
    if MON is not None:
        with st.sidebar:
            mon_slot = monitor_panel()
    gauge = render_chat(conv, mon_slot)
    render_sidebar(conv)
    draw_context_gauge(gauge, current_conv())      # after a reply and with this run's settings
    # Save AFTER the browser has answered the load (or once the visitor has typed
    # something) -- never before, or an empty chat would overwrite the stored one.
    if st.session_state.restored or conv.messages:
        persist_to_browser(conv, "post")


if __name__ == "__main__":
    main()
