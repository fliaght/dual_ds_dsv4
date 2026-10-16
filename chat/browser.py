"""The browser half of the chat page: Streamlit custom components v2 (inline, no iframes).

* :func:`store` -- the visitor's chats (IndexedDB) and settings (localStorage), ``components/browser_store.js``.
* :func:`reply_actions` -- Copy / Edit / Regenerate under a reply, ``components/reply_actions.js``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import streamlit as st
from streamlit.runtime import Runtime

_DIR = Path(__file__).resolve().parent / "components"


def _asset(name: str) -> str:
    return (_DIR / name).read_text(encoding="utf-8")


# Registered once per Streamlit runtime (AppTest starts a new one per test).  Multi-line strings are
# inline content, not file paths.
_SPECS = {
    "store": dict(
        name="dsv4_browser_store",
        html='<p class="note"></p>\n',
        css="""
.note { margin: 0; font-family: var(--st-font); font-size: 0.75rem; line-height: 1.45; color: #8c8a84; }
.note.error { color: var(--st-primary-color); }
""",
        js=_asset("browser_store.js"),
    ),
    "actions": dict(
        name="dsv4_reply_actions",
        html="""
<div class="bar" role="toolbar" aria-label="Reply actions">
  <button type="button" data-a="copy">Copy</button>
  <button type="button" data-a="edit">Edit</button>
  <button type="button" data-a="regen">Regenerate</button>
</div>
""",
        css=_asset("reply_actions.css"),
        js=_asset("reply_actions.js"),
    ),
}
_mounts: dict[int, dict] = {}


def _mount(kind: str):
    runtime = id(Runtime.instance())
    if runtime not in _mounts:
        _mounts.clear()
        _mounts[runtime] = {k: st.components.v2.component(**spec) for k, spec in _SPECS.items()}
    return _mounts[runtime][kind]


def store(data: dict, *, key: str, on_loaded: Callable[[], None] | None = None,
          on_opened: Callable[[], None] | None = None):
    """Hand the browser the state it should hold (see browser_store.js for ``data``).  Its answers arrive
    as the triggers ``loaded`` and ``opened``: in the callbacks (read with :func:`trigger`) and on the
    returned result.  Streamlit skips a callback when the run before it was interrupted ahead of this
    element, so callers check the result too."""
    callbacks = {}
    if on_loaded:
        callbacks["on_loaded_change"] = on_loaded
    if on_opened:
        callbacks["on_opened_change"] = on_opened
    return _mount("store")(key=key, data=data, **callbacks)


def reply_actions(text: str, later: int, *, key: str, on_action: Callable[[], None]) -> None:
    """``later``: messages after this reply (regenerating it removes them, so it asks first)."""
    _mount("actions")(key=key, data={"text": text, "later": later}, on_action_change=on_action, width="content")


def trigger(key: str, name: str):
    """A component's trigger value inside its callback (None when absent)."""
    state = st.session_state.get(key)
    if state is None:
        return None
    return state.get(name) if isinstance(state, dict) else getattr(state, name, None)
