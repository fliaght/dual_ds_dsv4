#!/usr/bin/env python3
"""Real-browser check of the chat page, fully offline: a fake model server (fake_openai.py), a private
Streamlit instance of chat/app.py on 127.0.0.1, and headless Chromium driven by Playwright.

    python -m pip install playwright && python -m playwright install chromium     # once, any venv
    python chat/tests/browser_check.py [--shots DIR]

Checks: chats and settings survive a reload (IndexedDB / localStorage); New chat and the chat list;
opening an earlier chat; Copy (Clipboard API, and the plain-http fallback); Edit; Regenerate, with the
confirmation for an earlier reply; Delete; another browser sees nothing; the single chat saved by the
previous version is moved into the chat list.  Never contacts the real model server.  Exit 0 on success.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright

HERE = Path(__file__).resolve().parent
CHAT = HERE.parent
sys.path.insert(0, str(HERE))
from fake_openai import FakeOpenAI  # noqa: E402

TIMEOUT_MS = 30_000
LEGACY = {"id": "0123456789abcdef0123456789abcdef", "created": "2026-10-01T09:00:00+00:00", "system_prompt": "",
          "messages": [{"role": "user", "content": "An old question from the single saved chat"},
                       {"role": "assistant", "content": "An old answer.", "status": "ok"}]}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_app(api_base: str, log_dir: str) -> tuple[subprocess.Popen, str]:
    port = free_port()
    env = dict(os.environ, CHAT_API_BASE=api_base, CHAT_LOG_DIR=log_dir, CHAT_MONITOR="0",
               CHAT_TITLE="DeepSeek-V4.1-Flash EXL3 on 2× DGX Spark", CHAT_MAX_TOKENS_CAP="65536",
               CHAT_VERSION="v0.0.0 · check", CHAT_CONTEXT_TOKEN_BUDGET="409600")
    streamlit = CHAT / ".venv" / "bin" / "streamlit"
    proc = subprocess.Popen([str(streamlit), "run", "app.py", "--server.address", "127.0.0.1",
                             "--server.port", str(port), "--server.headless", "true"],
                            cwd=CHAT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            if urllib.request.urlopen(url + "/_stcore/health", timeout=1).status == 200:
                return proc, url
        except OSError:
            time.sleep(0.25)
    proc.terminate()
    raise RuntimeError("the Streamlit test instance did not start")


class Check:
    def __init__(self, url: str, fake: FakeOpenAI, shots: Path | None) -> None:
        self.url, self.fake, self.shots, self.failed = url, fake, shots, []

    def ok(self, cond: bool, what: str) -> None:
        print(("PASS " if cond else "FAIL ") + what)
        if not cond:
            self.failed.append(what)

    def shot(self, pg: Page, name: str, full: bool = False) -> None:
        if self.shots:
            pg.screenshot(path=str(self.shots / f"{name}.png"), full_page=full)

    # ------------------------------------------------------------ helpers
    def open(self, pg: Page) -> None:
        pg.goto(self.url)
        self.ready(pg)

    @staticmethod
    def ready(pg: Page) -> None:
        pg.locator("textarea[data-testid='stChatInputTextArea']").wait_for(timeout=TIMEOUT_MS)
        pg.get_by_text("saved in this browser only").wait_for(timeout=TIMEOUT_MS)
        pg.wait_for_timeout(600)

    @staticmethod
    def copies(pg: Page) -> int:
        return pg.locator("button[data-a='copy']").count()

    def send(self, pg: Page, text: str) -> None:
        n = self.copies(pg)
        box = pg.locator("textarea[data-testid='stChatInputTextArea']")
        box.fill(text)
        box.press("Enter")
        pg.get_by_text(f"to: {text}").last.wait_for(timeout=TIMEOUT_MS)
        expect(pg.locator("button[data-a='copy']")).to_have_count(n + 1, timeout=TIMEOUT_MS)   # redrawn with actions
        pg.wait_for_timeout(400)

    @staticmethod
    def chats(pg: Page) -> list[str]:
        return [t.strip() for t in pg.locator(".st-key-history button").all_inner_texts()]

    @staticmethod
    def stored(pg: Page) -> dict:
        return pg.evaluate("""() => new Promise((resolve) => {
          const req = indexedDB.open("dsv4chat", 1);
          req.onsuccess = () => {
            const tx = req.result.transaction(["meta"], "readonly");
            const all = tx.objectStore("meta").getAll();
            all.onsuccess = () => resolve({meta: all.result, ls: {
              settings: localStorage.getItem("dsv4chat:settings"), active: localStorage.getItem("dsv4chat:active"),
              legacy: localStorage.getItem("dsv4chat:conversation")}});
          };
        })""")

    @staticmethod
    def sidebar(pg: Page):
        return pg.locator("[data-testid='stSidebar']")

    # ------------------------------------------------------------- checks
    def run(self, pw) -> None:
        browser = pw.chromium.launch()
        pg = None
        try:
            ctx = browser.new_context(viewport={"width": 1440, "height": 960},
                                      permissions=["clipboard-read", "clipboard-write"])
            pg = ctx.new_page()
            errors: list[str] = []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            self.steps(pg, browser, errors)
        except Exception:
            if pg is not None:                       # where it stopped, for the person reading the failure
                self.shot(pg, "failure", full=True)
                print("browser store:", pg.evaluate("() => JSON.stringify(window.__dsv4store && "
                                                    "{done: [...window.__dsv4store.done], error: window.__dsv4store.error})"))
            raise
        finally:
            browser.close()

    def steps(self, pg: Page, browser, errors: list[str]) -> None:
        if True:
            self.open(pg)
            self.shot(pg, "01-empty")
            self.ok(pg.get_by_text("How can I help?").is_visible(), "empty chat shows the welcome")
            self.ok(pg.get_by_text("Large language models can make mistakes.").is_visible(), "disclaimer under the input")
            self.ok(pg.locator(".app-version").inner_text() == "v0.0.0 · check", "version in the corner")

            # settings: thinking High, temperature +0.1 -> saved in this browser
            self.sidebar(pg).get_by_text("Generation settings").click()
            self.sidebar(pg).get_by_text("High", exact=True).click()
            slider = pg.locator("[data-testid='stSlider']").filter(has_text="Temperature").get_by_role("slider")
            slider.focus()
            slider.press("ArrowRight")
            slider.press("ArrowRight")
            pg.wait_for_timeout(800)
            self.shot(pg, "02-settings")

            self.send(pg, "First question with markdown")
            self.ok(self.fake.chats()[-1].get("chat_template_kwargs", {}).get("enable_thinking") is True,
                    "thinking setting reaches the request")
            self.ok(abs(self.fake.chats()[-1]["temperature"] - 0.8) < 1e-9, "temperature setting reaches the request")
            self.ok(pg.get_by_text("Thought").first.is_visible(), "reply shows its thinking toggle")
            self.send(pg, "Second question")
            self.shot(pg, "03-two-replies", full=True)
            self.ok(self.chats(pg) == ["First question with markdown"], "chat list names the chat by its first question")

            # Copy (Clipboard API on 127.0.0.1, a secure context)
            last = pg.locator("button[data-a='copy']").last
            last.click()
            expect(last).to_have_text("Copied", timeout=5000)
            clip = pg.evaluate("() => navigator.clipboard.readText()")
            self.ok(clip.startswith("Reply ") and clip.endswith("to: Second question"), "Copy puts the reply's text on the clipboard")

            # Edit the last reply
            pg.locator("button[data-a='edit']").last.click()
            editor = pg.get_by_label("Edit the reply")
            editor.wait_for(timeout=TIMEOUT_MS)
            self.shot(pg, "04-edit")
            editor.fill("An edited answer.")
            pg.get_by_role("button", name="Save").click()
            pg.get_by_text("An edited answer.").wait_for(timeout=TIMEOUT_MS)
            self.ok(pg.get_by_text("edited", exact=False).last.is_visible(), "edited reply is marked")
            n_before = len(self.fake.chats())
            self.send(pg, "Third question")
            sent = [m["content"] for m in self.fake.chats()[-1]["messages"]]
            self.ok("An edited answer." in sent and len(self.fake.chats()) == n_before + 1,
                    "the next question carries the edited reply")

            # Regenerate the first reply: asks first (4 later messages), then drops them and answers again
            first = pg.locator("button[data-a='regen']").first
            first.click()
            expect(first).to_contain_text("remove the 4 later messages", timeout=5000)
            self.shot(pg, "05-regen-confirm")
            first.click()
            expect(pg.locator("button[data-a='copy']")).to_have_count(1, timeout=TIMEOUT_MS)
            self.ok(not pg.get_by_text("Second question").first.is_visible()
                    and self.fake.chats()[-1]["messages"][-1]["content"] == "First question with markdown",
                    "regenerating an earlier reply removes what followed and asks again")
            # Regenerate the last reply: no confirmation
            text_before = pg.locator("[data-testid='stChatMessage']").last.inner_text()
            pg.locator("button[data-a='regen']").last.click()
            pg.wait_for_function("t => [...document.querySelectorAll('[data-testid=stChatMessage]')].pop()"
                                 ".innerText !== t", arg=text_before, timeout=TIMEOUT_MS)
            expect(pg.locator("button[data-a='copy']")).to_have_count(1, timeout=TIMEOUT_MS)
            self.ok(True, "regenerating the last reply replaces it")

            # New chat, a second chat, open the first again
            pg.get_by_role("button", name="New chat").click()
            pg.get_by_text("How can I help?").wait_for(timeout=TIMEOUT_MS)
            self.send(pg, "A question in a second chat")
            self.ok(self.chats(pg) == ["A question in a second chat", "First question with markdown"],
                    "New chat starts a second chat; the list shows both, newest first")
            self.sidebar(pg).get_by_role("button", name="First question with markdown").click()
            pg.get_by_text("to: First question with markdown").first.wait_for(timeout=TIMEOUT_MS)
            self.ok(not pg.get_by_text("to: A question in a second chat").is_visible(), "an earlier chat opens from the list")
            self.shot(pg, "06-history")

            # Reload: the open chat, the list and the settings come back
            pg.reload()
            self.ready(pg)
            pg.get_by_text("to: First question with markdown").first.wait_for(timeout=TIMEOUT_MS)
            self.ok(len(self.chats(pg)) == 2, "after a reload: the chat list is back")
            saved = json.loads(self.stored(pg)["ls"]["settings"])
            self.ok(saved["thinking"] == "high" and abs(saved["temperature"] - 0.8) < 1e-9, "settings saved in localStorage")
            self.send(pg, "After the reload")
            self.ok(self.fake.chats()[-1].get("chat_template_kwargs", {}).get("enable_thinking") is True
                    and abs(self.fake.chats()[-1]["temperature"] - 0.8) < 1e-9, "after a reload: the settings are used")

            # Delete the open chat
            pg.get_by_role("button", name="Delete").click()
            pg.get_by_role("button", name="Delete chat").click()
            pg.get_by_text("How can I help?").wait_for(timeout=TIMEOUT_MS)
            pg.wait_for_timeout(800)
            self.ok(self.chats(pg) == ["A question in a second chat"], "Delete removes the chat from the list")
            pg.reload()
            self.ready(pg)
            self.ok(self.chats(pg) == ["A question in a second chat"] and len(self.stored(pg)["meta"]) == 1,
                    "after a reload: the deleted chat stays deleted")

            # plain http (no Clipboard API): the textarea fallback still copies
            ctx_http = browser.new_context(permissions=["clipboard-read", "clipboard-write"])
            ctx_http.add_init_script("Object.defineProperty(window, 'isSecureContext', {get: () => false});")
            pg2 = ctx_http.new_page()
            self.open(pg2)
            self.ok(self.chats(pg2) == [], "another browser sees no chats")
            self.send(pg2, "Copy over plain http")
            pg2.locator("button[data-a='copy']").last.click()
            expect(pg2.locator("button[data-a='copy']").last).to_have_text("Copied", timeout=5000)
            self.ok(pg2.evaluate("() => navigator.clipboard.readText()").endswith("to: Copy over plain http"),
                    "Copy works without the Clipboard API (plain http)")
            ctx_http.close()

            # the previous version's single saved chat moves into the list
            ctx_old = browser.new_context(viewport={"width": 390, "height": 844})
            ctx_old.add_init_script("if (!sessionStorage.getItem('seeded')) { sessionStorage.setItem('seeded', '1');"
                                    f" localStorage.setItem('dsv4chat:conversation', {json.dumps(json.dumps(LEGACY))}); }}")
            pg3 = ctx_old.new_page()
            self.open(pg3)
            pg3.get_by_text("An old answer.").wait_for(timeout=TIMEOUT_MS)
            pg3.wait_for_timeout(800)
            st3 = self.stored(pg3)
            self.ok([m["id"] for m in st3["meta"]] == [LEGACY["id"]] and st3["ls"]["legacy"] is None,
                    "the old single saved chat is moved to the chat list")
            self.shot(pg3, "07-phone")
            ctx_old.close()
            self.ok(not errors, f"no page errors {errors[:3]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shots", type=Path, help="save screenshots here")
    args = ap.parse_args()
    if args.shots:
        args.shots.mkdir(parents=True, exist_ok=True)
    fake = FakeOpenAI(delay_s=0.01).start()
    with tempfile.TemporaryDirectory(prefix="dsv4chat-browser-") as logs:
        proc, url = start_app(fake.base_url, logs)
        try:
            check = Check(url, fake, args.shots)
            with sync_playwright() as pw:
                check.run(pw)
        finally:
            proc.terminate()
            proc.wait(timeout=20)
            fake.stop()
    print("BROWSER CHECK", "FAIL" if check.failed else "PASS")
    return 1 if check.failed else 0


if __name__ == "__main__":
    sys.exit(main())
