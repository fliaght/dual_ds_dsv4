"""Streamlit chat UI for the dual-Spark DeepSeek service -- no accounts.

    bash chat/run_chat.sh --daemon   # http://<head-ip>:9450

How it works, in one paragraph: every visitor lands straight in a chat.  Their
chats and generation settings are kept in *their own browser* (IndexedDB and
localStorage, via ``browser.py``) and come back on the next visit; the server
stores no per-user data.  A prompt joins the process-wide FIFO queue
(request_queue.py) -- waiting visitors see their position -- then the reply
streams from the model server.  Every turn is appended to the server-side audit
log (``$CHAT_LOG_DIR/chat-YYYY-MM-DD.jsonl``), even if the visitor hits Stop.
"""
from __future__ import annotations

import html
import importlib
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st  # noqa: E402

import browser  # noqa: E402
from context import build_context, context_gauge, estimate_tokens, reasoning_meta, token_caption  # noqa: E402
from history import (MAX_TOKENS_MIN, MAX_TOKENS_STEP, SYSTEM_PROMPT_CHARS, TEMPERATURE, TOP_P,  # noqa: E402
                     clean_index, clean_settings, group_by_day, meta_of, upsert)
from llm_client import THINKING_LEVELS, LLMError  # noqa: E402
from request_queue import QueueFull, QueueTimeout, UserBusy  # noqa: E402
from runtime import get_client, get_logger, get_monitor, get_queue  # noqa: E402
from settings import Settings, load_settings  # noqa: E402
from storage import MAX_CONTENT_CHARS, Conversation, clean_id, now_iso  # noqa: E402

S: Settings = load_settings()

THINKING_OPTIONS = ("off", *THINKING_LEVELS)
THINKING_LABELS = {"off": "Off", "high": "High", "max": "Max"}
DEFAULT_THINKING = S.default_thinking if S.default_thinking in THINKING_OPTIONS else "off"
DEFAULTS = {"thinking": DEFAULT_THINKING, "max_tokens": S.default_max_tokens, "temperature": S.default_temperature,
            "top_p": S.default_top_p, "system_prompt": S.system_prompt}
SETTING_KEYS = {name: f"set_{name}" for name in DEFAULTS}       # the settings widgets' session-state keys
HISTORY_PAGE = 40

st.set_page_config(page_title=S.app_title, page_icon="🟥", layout="centered")

# One queue / logger / client per server process, shared by all sessions (runtime.py).
Q = get_queue(S)
LOG = get_logger(S)
LLM = get_client(S)
MON = get_monitor(S) if S.monitor_enabled else None

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "components", "page.css"), encoding="utf-8") as _f:
    PAGE_CSS = _f.read()


# ------------------------------------------------------------ session state
def ss_init() -> None:
    ss = st.session_state
    if "session_id" in ss:
        return
    ss.session_id = uuid.uuid4().hex[:12]
    ss.client_id = None            # stable per-browser id, supplied by the browser store
    ss.restored = False            # the browser has answered the "load" request
    ss.store_error = None          # why this browser cannot keep chats (from the load answer)
    ss.conv = None                 # the open Conversation
    ss.index = []                  # the browser's chat list (history.meta_of entries), newest first
    ss.deleted = []                # chats deleted in this session; the browser is told every run
    ss.switch = None               # {"nonce", "id"}: waiting for the browser to save this chat and hand over the next
    ss.legacy = False              # the open chat came from the pre-IndexedDB localStorage key
    ss.editing = None              # (conv id, message index) of the reply being edited
    ss.regen = None                # (conv id, message index) to regenerate on the next run
    ss.notice = None               # one-off message for the visitor
    ss.history_limit = HISTORY_PAGE
    ss.nonce = 0
    for name, value in DEFAULTS.items():
        ss[SETTING_KEYS[name]] = value
    LOG.log_event("session_start", session_id=ss.session_id, **client_info())


def gen() -> dict:
    """The generation settings, as the sidebar widgets hold them."""
    return {name: st.session_state[key] for name, key in SETTING_KEYS.items()}


def nonce() -> str:
    st.session_state.nonce += 1
    return f"{st.session_state.session_id}-{st.session_state.nonce}"


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
        st.session_state.conv = Conversation.new(system_prompt=gen()["system_prompt"])
    return st.session_state.conv


def _event(name: str, **fields) -> None:
    ss = st.session_state
    LOG.log_event(name, client_id=ss.client_id, session_id=ss.session_id, **fields)


# ------------------------------------------------- the browser's copy (browser.py)
def chat_list(conv: Conversation) -> list[dict]:
    """The browser's chats, with the open one's current title and time."""
    index = [e for e in st.session_state.index if e["id"] not in st.session_state.deleted]
    return upsert(index, meta_of(conv)) if conv.messages else index


def store_data(conv: Conversation) -> dict:
    """What the browser should hold after this run (browser_store.js applies each part once)."""
    ss = st.session_state
    data = {"deletes": list(ss.deleted),
            "note": f"Not saved in this browser: {ss.store_error}" if ss.store_error
            else "Chats and settings are saved in this browser only."}
    if conv.messages and conv.id not in ss.deleted:
        data["conv"] = {"conv": conv.to_dict(), "meta": meta_of(conv), "legacy": ss.legacy}
    if ss.restored:            # never overwrite saved settings with defaults before they were read
        data["prefs"] = {"settings": gen(), "active": conv.id if conv.messages else None}
    else:
        data["load"] = f"{ss.session_id}-load"
    if ss.switch:
        data["open"] = ss.switch
    data["ask"] = ss.switch["nonce"] if ss.switch else data.get("load")
    return data


def apply_loaded(p) -> bool:
    """The browser's saved state, once per session: settings, the chat list and the chat open last time."""
    ss = st.session_state
    if ss.restored or not isinstance(p, dict) or p.get("nonce") != f"{ss.session_id}-load":
        return False
    ss.restored = True
    ss.client_id = clean_id(p.get("client_id"), fallback=ss.client_id)
    ss.store_error = None if p.get("ok") else str(p.get("error") or "unknown error")[:200]
    ss.index = clean_index(p.get("index"))
    settings = clean_settings(p.get("settings"), DEFAULTS, max_tokens_cap=S.max_tokens_cap,
                              thinking_options=THINKING_OPTIONS)
    for name, value in settings.items():
        ss[SETTING_KEYS[name]] = value
    saved = Conversation.from_untrusted(p.get("conv"), max_messages=S.max_local_messages)
    legacy = Conversation.from_untrusted(p.get("legacy"), max_messages=S.max_local_messages)
    if saved is None and legacy is not None and legacy.messages:
        saved, ss.legacy = legacy, True            # the single chat an older version kept: moved on first save
    if saved is not None and saved.messages and (ss.conv is None or not ss.conv.messages):
        ss.conv = saved                            # don't clobber a chat the visitor already started
    _event("history_restored", chats=len(ss.index), messages=len(saved.messages) if saved else 0,
           store_ok=bool(p.get("ok")), migrated=ss.legacy)
    return True


def start_switch(target: str | None) -> None:
    """Open chat ``target`` (None: a new chat).  The browser first saves the open chat, then answers."""
    ss = st.session_state
    ss.editing = None
    if ss.restored:
        ss.switch = {"nonce": nonce(), "id": target}
    elif target is None:       # no browser store (JavaScript never answered): nothing to save or fetch
        finish_switch(None, None)


def finish_switch(target: str | None, stored) -> None:
    ss = st.session_state
    old = current_conv()
    if old.messages and old.id not in ss.deleted:
        ss.index = upsert(ss.index, meta_of(old))
    conv = Conversation.from_untrusted(stored, max_messages=S.max_local_messages) if target else None
    if target and (conv is None or conv.id != target):
        ss.index = [e for e in ss.index if e["id"] != target]
        ss.notice = "That chat is no longer stored in this browser."
        conv = None
    ss.conv = conv or Conversation.new(system_prompt=gen()["system_prompt"])
    ss.legacy = False
    _event("chat_opened" if conv else "chat_new", conv_id=ss.conv.id, messages=len(ss.conv.messages))


def apply_opened(p) -> bool:
    ss = st.session_state
    if not (ss.switch and isinstance(p, dict) and p.get("nonce") == ss.switch["nonce"]):
        return False
    target, ss.switch = ss.switch["id"], None
    finish_switch(target, p.get("conv"))
    return True


def cb_loaded() -> None:
    apply_loaded(browser.trigger("store", "loaded"))


def cb_opened() -> None:
    apply_opened(browser.trigger("store", "opened"))


def cb_new_chat() -> None:
    if current_conv().messages:
        start_switch(None)


def cb_open_chat(chat_id: str) -> None:
    if chat_id != current_conv().id:
        start_switch(chat_id)


def cb_delete_chat() -> None:
    ss = st.session_state
    conv = current_conv()
    ss.deleted.append(conv.id)
    ss.index = [e for e in ss.index if e["id"] != conv.id]
    _event("chat_deleted", conv_id=conv.id, messages=len(conv.messages))
    if ss.restored:
        start_switch(None)
    else:
        ss.conv = Conversation.new(system_prompt=gen()["system_prompt"])


def cb_more_history() -> None:
    st.session_state.history_limit += HISTORY_PAGE


def cb_reset_settings() -> None:
    for name, value in DEFAULTS.items():
        st.session_state[SETTING_KEYS[name]] = value


# ------------------------------------------------------- reply actions
def _reply_at(conv_id: str, i: int) -> dict | None:
    conv = current_conv()
    ok = conv.id == conv_id and 0 < i < len(conv.messages) and conv.messages[i]["role"] == "assistant"
    return conv.messages[i] if ok else None


def cb_reply_action(key: str, conv_id: str, i: int) -> None:
    action = browser.trigger(key, "action")
    if _reply_at(conv_id, i) is None:
        return
    if action == "edit":
        st.session_state.editing = (conv_id, i)
    elif action == "regen":
        st.session_state.editing = None
        st.session_state.regen = (conv_id, i)


def cb_edit_save(conv_id: str, i: int, key: str) -> None:
    ss = st.session_state
    text = ss.pop(key, None)
    ss.editing = None
    m = _reply_at(conv_id, i)
    if m is None or not isinstance(text, str) or not text.strip() or text == m["content"]:
        return
    _event("reply_edited", conv_id=conv_id, index=i, chars_before=len(m["content"]), chars_after=len(text))
    m["content"] = text[:MAX_CONTENT_CHARS]
    m["edited"] = True
    current_conv().touch()


def cb_edit_cancel(key: str) -> None:
    st.session_state.pop(key, None)
    st.session_state.editing = None


def take_regenerate(conv: Conversation) -> list[dict] | None:
    """A requested regenerate: take that reply and everything after it out of the chat (returned, so a refused
    request can put them back); the question before it is asked again."""
    ss = st.session_state
    req, ss.regen = ss.regen, None
    if not req or _reply_at(*req) is None or conv.messages[req[1] - 1]["role"] != "user":
        return None
    i = req[1]
    _event("reply_regenerated", conv_id=conv.id, index=i, removed=len(conv.messages) - i)
    removed = conv.messages[i:]
    del conv.messages[i:]
    conv.touch()
    return removed


# ---------------------------------------------------------------- sidebar
def _md_escape(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|~<>$])", r"\\\1", text)


def _browser_tz():
    try:
        return ZoneInfo(st.context.timezone or "UTC")
    except Exception:  # noqa: BLE001 - unknown zone name or no browser context (AppTest)
        return timezone.utc


def render_history(conv: Conversation) -> None:
    ss = st.session_state
    entries = chat_list(conv)
    if not entries:
        st.container(key="history-empty").caption("Your chats will appear here.")
    with st.container(key="history"):
        shown = entries[:ss.history_limit]
        for label, group in group_by_day(shown, datetime.now(timezone.utc), _browser_tz()):
            st.caption(label)
            for e in group:
                here = e["id"] == conv.id
                st.button(_md_escape(e["title"]), key=f"{'chat-here' if here else 'chat'}-{e['id']}", type="tertiary",
                          on_click=cb_open_chat, args=(e["id"],), width="stretch",
                          disabled=bool(ss.switch))
        if len(entries) > len(shown):
            st.button(f"Show {min(HISTORY_PAGE, len(entries) - len(shown))} more", type="tertiary",
                      on_click=cb_more_history, key="chat-more")
    answer = browser.store(store_data(conv), key="store", on_loaded=cb_loaded, on_opened=cb_opened)
    if apply_loaded(answer.loaded) or apply_opened(answer.opened):
        st.rerun()             # an answer whose callback Streamlit skipped: redraw with it


def render_settings() -> None:
    with st.expander("Generation settings"):
        st.segmented_control(
            "Thinking", THINKING_OPTIONS, format_func=THINKING_LABELS.get, key=SETTING_KEYS["thinking"],
            required=True, width="stretch",
            help="High: the model reasons step by step before answering (shown under 'Thought'). "
                 "Max: deepest reasoning, much slower. Reasoning tokens count toward Max reply tokens, "
                 "so raise that slider when thinking is on.")
        st.slider("Max reply tokens", MAX_TOKENS_MIN, S.max_tokens_cap, step=MAX_TOKENS_STEP,
                  key=SETTING_KEYS["max_tokens"])
        st.slider("Temperature", TEMPERATURE[0], TEMPERATURE[1], step=TEMPERATURE[2], key=SETTING_KEYS["temperature"])
        st.slider("Top-p", TOP_P[0], TOP_P[1], step=TOP_P[2], key=SETTING_KEYS["top_p"])
        st.text_area("System prompt", key=SETTING_KEYS["system_prompt"], height=100, max_chars=SYSTEM_PROMPT_CHARS,
                     help="Used for chats started after the change.")
        st.button("Reset to defaults", type="tertiary", on_click=cb_reset_settings, key="settings-reset")
        st.caption("Settings are remembered in this browser.")


def render_server() -> None:
    with st.expander("Server"):
        stats = Q.stats()
        up = LLM.health()
        st.markdown(f"{':green[●] Online' if up else ':red[●] Unreachable'}  \n`{LLM.model or S.model or '?'}`")
        if up and LLM.max_model_len:
            st.caption(f"Context window {LLM.max_model_len:,} tokens")
        line = (f"Running {stats['running']}/{stats['max_concurrency']} · waiting {stats['waiting']} · "
                f"served {stats['served']}")
        if stats["served"]:
            line += f" · average wait {stats['avg_wait_s']:.1f} s · refused {stats['rejected']}"
        st.caption(line)
        st.caption("Requests are served first come, first served; one request at a time per visitor.")


def render_sidebar(conv: Conversation):
    """Everything in the sidebar; returns the monitor's slot (redrawn while a reply streams)."""
    with st.sidebar:
        st.button("New chat", type="primary", icon=":material/add:", width="stretch", on_click=cb_new_chat,
                  key="new-chat")
        render_history(conv)
        render_settings()
        render_server()
        return monitor_panel() if MON is not None else None


# ------------------------------------------------------------ performance
# Categorical slots in fixed order: head, worker, ... (black, then the accent red; the theme's chart colours)
SERIES_COLORS = {
    "light": ["#161616", "#D52B1E", "#8C8A84", "#3F6E9A", "#B58A00", "#2F7D4F"],
    "dark": ["#EDEDEA", "#F0574B", "#A3A19A", "#6F9CC7", "#D6A920", "#4FA673"],
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
    line = base.mark_line(strokeWidth=1.5)
    dots = base.mark_point(size=48, filled=True).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("host:N", title="node"), alt.Tooltip("t:T", title="time", format="%H:%M:%S"),
                 alt.Tooltip(f"{field}:Q", title=title, format=".0f")],
    ).add_params(hover)
    return (line + dots).properties(title=title, height=110)


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
                st.caption(":red[●] no data" + (f" — {s['error']}" if s["error"] else (" yet…" if L is None else "")))
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
                         f"— the watchdog kills the model server.")
            elif L["mem_avail"] < 2 * wd:
                st.warning("Low memory headroom.")
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
    never preempt a running script, so while a reply streams, generate_reply redraws the
    returned slot instead."""
    with st.expander("Performance"):
        slot = st.empty()
    draw_monitor(slot)
    return slot


# -------------------------------------------------------------- chat view
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
    g = gen()
    limit, budget = context_limits(g)
    api, dropped, est = build_context(conv.messages, conv.system_prompt, budget)
    key = (conv.id, conv.updated, len(conv.messages), conv.system_prompt, g["thinking"], len(api), dropped)
    cached = st.session_state.get("ctx_usage")
    if cached is None or cached["key"] != key:
        thinking = None if g["thinking"] == "off" else g["thinking"]
        # an empty question renders the history exactly as the next request will (thinking dropped)
        n = LLM.count_chat_tokens(api + [{"role": "user", "content": ""}], thinking=thinking)
        cached = {"key": key, "tokens": n if n is not None else est, "exact": n is not None, "dropped": dropped}
        st.session_state.ctx_usage = cached
    return dict(cached, limit=limit, reply_tokens=int(g["max_tokens"]))


def draw_context_gauge(conv: Conversation) -> None:
    u = context_usage(conv)
    if u is None:
        return
    fraction, label, notes = context_gauge(u["tokens"], u["exact"], u["limit"], u["dropped"], u["reply_tokens"])
    with st.container(key="gauge"):
        st.progress(fraction, text=label)
        st.caption(" · ".join(f"⚠ {n}" for n in notes) or "What the model reads with your next message",
                   help="System prompt plus the conversation so far, counted by the model's tokenizer; your next "
                        "question and the reply add to it. Thinking from earlier replies is not sent again, so it "
                        "does not count. When the conversation no longer fits, the oldest messages are left out; "
                        "start a new chat to keep everything in view. The numbers under each reply are that "
                        "reply's own input and output.")


def render_header(conv: Conversation) -> None:
    with st.container(horizontal=True, vertical_alignment="bottom", key="masthead"):
        st.title(S.app_title, anchor=False)
        if conv.messages:
            with st.popover("Delete", type="tertiary", icon=":material/delete:", key="delete-chat"):
                st.markdown("Delete this chat from this browser?")
                st.button("Delete chat", type="primary", on_click=cb_delete_chat, key="delete-chat-confirm")
                st.caption("The server's anonymous request log is not affected.")
    if st.session_state.notice:
        st.info(st.session_state.notice)
        st.session_state.notice = None
    draw_context_gauge(conv)


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


def _thought_label(m: dict) -> str:
    n = (m.get("meta") or {}).get("reasoning_tokens")
    return f"Thought · {n:,} tokens" if isinstance(n, int) and not isinstance(n, bool) else "Thought"


def _reply_note(m: dict) -> str:
    status = m.get("status")
    if status == "interrupted":
        note = "Stopped before the reply finished"
    elif status == "error":
        note = f"Failed: {m.get('error', 'error')}"
    else:
        note = _meta_caption(m.get("meta") or {})
    return " · ".join(b for b in (note, "edited" if m.get("edited") else "") if b)


def render_editor(conv: Conversation, i: int, m: dict) -> None:
    key = f"edit-{conv.id[:12]}-{i}"
    lines = m["content"].count("\n") + len(m["content"]) // 70 + 2
    with st.form(key=f"form-{key}", border=False):
        st.text_area("Edit the reply", value=m["content"], key=key, height=min(560, max(140, 24 * lines)),
                     label_visibility="collapsed")
        with st.container(horizontal=True):
            st.form_submit_button("Save", type="primary", on_click=cb_edit_save, args=(conv.id, i, key))
            st.form_submit_button("Cancel", type="tertiary", on_click=cb_edit_cancel, args=(key,))
        st.caption("The next questions see the edited reply.")


def render_message(conv: Conversation, i: int) -> None:
    m = conv.messages[i]
    with st.chat_message(m["role"]):
        if m["role"] == "assistant" and st.session_state.editing == (conv.id, i):
            render_editor(conv, i, m)
            return
        if m.get("reasoning"):
            with st.expander(_thought_label(m), type="compact"):
                st.markdown(m["reasoning"])
        if m.get("content"):
            st.markdown(m["content"])
        if m["role"] == "assistant":
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                key = f"act-{conv.id[:12]}-{i}"
                browser.reply_actions(m.get("content", ""), len(conv.messages) - 1 - i, key=key,
                                      on_action=lambda: cb_reply_action(key, conv.id, i))
                note = _reply_note(m)
                if note:
                    st.caption(note)


def render_chat(conv: Conversation) -> None:
    render_header(conv)
    if not conv.messages:
        with st.container(key="welcome"):
            st.markdown("## How can I help?")
            st.caption("Chats and settings stay in this browser. The server keeps an anonymous log of requests "
                       "for operations.")
    for i in range(len(conv.messages)):
        render_message(conv, i)


def handle_prompt(conv: Conversation, prompt: str, mon_slot=None) -> bool:
    """Append the visitor's question and answer it; True when the page should be redrawn from state."""
    conv.messages.append({"role": "user", "content": prompt, "ts": now_iso()})
    conv.trim(S.max_local_messages)
    conv.touch()
    with st.chat_message("user"):
        st.markdown(prompt)
    return generate_reply(conv, mon_slot)


def generate_reply(conv: Conversation, mon_slot=None, *, replaced: list[dict] | None = None) -> bool:
    """Answer the last (user) message: queue, then stream.  ``replaced``: the messages a regenerate took out.
    True when the page should be redrawn from state (a reply was added, or the request was refused)."""
    ss = st.session_state
    g = gen()
    cid = client_id()
    prompt = conv.messages[-1]["content"]

    # 1. build the context window
    _, budget = context_limits(g)
    api_messages, dropped, est = build_context(conv.messages, conv.system_prompt, budget)
    if len(api_messages) == 0 or api_messages[-1]["role"] != "user":
        with st.chat_message("assistant"):
            st.error(f"This message is too long for the context window (~{est:,} tokens estimated, "
                     f"budget {budget:,}). Please shorten it.")
        return False

    # 2. queue, then stream
    record = {
        "client_id": cid, "session_id": ss.session_id, **client_info(),
        "conv_id": conv.id, "turn_id": uuid.uuid4().hex[:12], "model": LLM.model or S.model,
        "params": {"max_tokens": int(g["max_tokens"]), "temperature": float(g["temperature"]),
                   "top_p": float(g["top_p"]), "thinking": g["thinking"]},
        "context": {"messages_sent": len(api_messages), "dropped_turns": dropped, "est_prompt_tokens": est},
        "prompt": prompt,
        "regenerated": replaced is not None,
    }
    with st.chat_message("assistant"):
        status = st.status("Joining the queue…", type="compact")
        body = st.empty()

        try:
            ticket = Q.submit(cid)
        except (UserBusy, QueueFull) as e:
            LOG.log_event("rejected", client_id=cid, session_id=ss.session_id, conv_id=conv.id,
                          reason=type(e).__name__, queue=Q.stats())
            if replaced is not None:
                conv.messages.extend(replaced)               # keep the reply that was to be replaced
                ss.notice = f"{e} The reply was not regenerated."
            else:
                conv.messages.pop()                          # back into the input box, not left unanswered
                ss.restore_prompt = prompt
                ss.notice = f"{e} Your message was not sent."
            conv.touch()
            return True
        # keep the question in the browser while the reply runs (the sidebar copy predates it)
        browser.store({"conv": {"conv": conv.to_dict(), "meta": meta_of(conv), "legacy": ss.legacy}},
                      key="store-pending")

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
            status.update(label=":shimmer[Waiting for the model]")
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
                        status.update(label=f"Thought for {time.monotonic() - t_start:.0f}s", state="complete",
                                      expanded=False)
                    elif not text:
                        status.update(label=":shimmer[Writing]")
                    text += ev["text"]
                    body.markdown(text + " ▌")
                elif kind == "reasoning":
                    if t_first is None:
                        t_first = time.monotonic()
                    reasoning += ev["text"]
                    if reasoning_box is None:
                        status.update(label=":shimmer[Thinking]", expanded=True)
                        reasoning_box = status.empty()
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

        if err:
            status.update(label="Failed — your message was not answered", state="error")
            body.error(err)
        else:
            body.markdown(text)
        return True


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
    conv.touch()
    LOG.log_turn({**record, "ticket_id": ticket.id, "status": status, "error": err,
                  "response": text, "reasoning": reasoning or None, "usage": usage, "timing": meta})
    return meta


def render_footer() -> None:
    st.html(f'<div class="app-version">{html.escape(S.version)}</div>')


# ------------------------------------------------------------------- main
def main() -> None:
    ss_init()
    st.markdown(f"<style>{PAGE_CSS}</style>", unsafe_allow_html=True)
    conv = current_conv()
    replaced = take_regenerate(conv)
    if "restore_prompt" in st.session_state:         # a refused question goes back into the input box
        st.session_state.chat_box = st.session_state.pop("restore_prompt")
    with st.bottom:
        prompt = st.chat_input("Message the model…", key="chat_box", submit_mode="stop")
        st.caption(S.disclaimer, text_alignment="center")
    # The sidebar goes first: its monitor stays live while a reply streams, and its settings apply to it.
    mon_slot = render_sidebar(conv)
    render_chat(conv)
    render_footer()
    answered = False
    if prompt and prompt.strip():
        answered = handle_prompt(conv, prompt.strip(), mon_slot)
    elif replaced is not None:
        answered = generate_reply(conv, mon_slot, replaced=replaced)
    if answered:
        st.rerun()             # redraw from state: actions under the new reply, chat list, gauge, browser copy


if __name__ == "__main__":
    main()

