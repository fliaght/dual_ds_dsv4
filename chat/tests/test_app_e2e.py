"""End-to-end through Streamlit's AppTest against the LIVE model server.

AppTest executes no browser JavaScript, so the localStorage component never
answers here: these tests cover the server side (no-login chat, multi-turn
context, audit log, queueing, busy-visitor refusal, clear).  The real-browser
round-trip (save -> reload -> restore) is covered by tests/browser_check.py.

Opt in with CHAT_LIVE_TESTS=1. Otherwise no model server is contacted, even
when one is running. Uses a throw-away CHAT_LOG_DIR.
"""
import json, os, shutil, sys, tempfile, threading, time, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "app.py"
sys.path.insert(0, str(HERE.parent))

TMP = tempfile.mkdtemp(prefix="dsv4chat-e2e-")
os.environ["CHAT_LOG_DIR"] = TMP
os.environ.setdefault("CHAT_MAX_CONCURRENCY", "1")

from llm_client import LLMClient  # noqa: E402

LIVE = os.environ.get("CHAT_LIVE_TESTS") == "1" and LLMClient(
    os.environ.get("CHAT_API_BASE", "http://127.0.0.1:8000/v1")
).health()
MATH_Q = "What is 17 * 23? Reply with just the number."


def _at():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    return at


def _last_assistant(at):
    msgs = [m for m in at.chat_message if m.name == "assistant"]
    assert msgs, "no assistant message rendered"
    return "\n".join(md.value for md in msgs[-1].markdown)


def _turns():
    return [json.loads(l) for p in Path(TMP).glob("chat-*.jsonl") for l in p.read_text(encoding="utf-8").splitlines()]


def _events():
    return [json.loads(l) for p in Path(TMP).glob("events-*.jsonl") for l in p.read_text(encoding="utf-8").splitlines()]


def tearDownModule():
    if not os.environ.get("CHAT_E2E_KEEP"):
        shutil.rmtree(TMP, ignore_errors=True)


@unittest.skipUnless(LIVE, "requires CHAT_LIVE_TESTS=1 and a reachable model server")
class E2E(unittest.TestCase):
    def test_chat_without_login_multiturn_and_audit_log(self):
        at = _at()
        self.assertFalse(at.exception, at.exception)
        self.assertTrue(at.chat_input, "no chat box on first load (login page?)")
        self.assertFalse(at.text_input, "unexpected text inputs (login form?)")
        sid = at.session_state["session_id"]
        self.assertTrue(any(e["event"] == "session_start" and e["session_id"] == sid for e in _events()))

        at.chat_input[0].set_value(MATH_Q).run()
        self.assertFalse(at.exception, at.exception)
        self.assertIn("391", _last_assistant(at))
        at.chat_input[0].set_value("Now double that number. Reply with just the number.").run()
        self.assertFalse(at.exception, at.exception)
        self.assertIn("782", _last_assistant(at))                     # earlier turn was in the context

        conv = at.session_state["conv"]
        self.assertEqual([m["role"] for m in conv.messages], ["user", "assistant", "user", "assistant"])
        self.assertEqual(conv.messages[1]["status"], "ok")
        self.assertIsNotNone(conv.messages[1]["meta"]["completion_tokens"])

        mine = [r for r in _turns() if r["session_id"] == sid]
        self.assertEqual(len(mine), 2)
        self.assertEqual(mine[0]["prompt"], MATH_Q); self.assertIn("391", mine[0]["response"])
        self.assertEqual(mine[1]["context"]["messages_sent"], 4)         # system + 3 prior turns... + this one
        for r in mine:
            for k in ("client_id", "conv_id", "turn_id", "ticket_id", "usage", "timing", "params", "status"):
                self.assertIn(k, r)
            self.assertEqual(r["status"], "ok"); self.assertEqual(r["conv_id"], conv.id)

    def test_clear_chat(self):
        at = _at()
        at.chat_input[0].set_value(MATH_Q).run()
        self.assertEqual(len(at.session_state["conv"].messages), 2)
        old_id = at.session_state["conv"].id
        at.button(key="clear_chat").click().run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(at.session_state["conv"].messages, [])
        self.assertNotEqual(at.session_state["conv"].id, old_id)
        self.assertTrue(any(e["event"] == "chat_cleared" and e["conv_id"] == old_id and e["messages"] == 2
                            for e in _events()))

    def test_second_visitor_waits_for_a_busy_slot(self):
        """Someone else holds the only slot; the UI must queue, show progress, then answer.

        (AppTest can't run two sessions in parallel threads -- it tears down a
        shared fake runtime -- so the other visitor is simulated by taking a
        ticket directly from the process-wide queue the app uses.)
        """
        from runtime import get_queue
        from settings import load_settings
        q = get_queue(load_settings())
        self.assertEqual(q.max_concurrency, 1)
        base_served = q.stats()["served"]
        other = q.submit("someone-else")
        self.assertTrue(q.try_acquire(other))
        HOLD_S = 2.0
        released = []
        def free_later():
            time.sleep(HOLD_S); released.append(time.monotonic()); q.release(other)
        threading.Thread(target=free_later, daemon=True).start()

        at = _at()
        t0 = time.monotonic()
        at.chat_input[0].set_value(MATH_Q).run()
        self.assertFalse(at.exception, at.exception)
        self.assertIn("391", _last_assistant(at))
        self.assertGreaterEqual(time.monotonic() - t0, HOLD_S - 0.1)
        self.assertEqual(len(released), 1)
        rec = [r for r in _turns() if r["session_id"] == at.session_state["session_id"]][-1]
        self.assertGreaterEqual(rec["timing"]["queued_s"], HOLD_S - 0.5, rec["timing"])
        self.assertEqual(q.stats()["served"] - base_served, 2)
        self.assertEqual((q.stats()["running"], q.stats()["waiting"]), (0, 0))

    def test_rejected_when_visitor_already_busy(self):
        from runtime import get_queue
        from settings import load_settings
        q = get_queue(load_settings())
        at = _at()
        at.chat_input[0].set_value("warm up: say hi").run()          # establishes this session's client_id
        cid = at.session_state["client_id"]
        mine = q.submit(cid); self.assertTrue(q.try_acquire(mine))    # same visitor busy in another tab
        try:
            n_before = len(at.session_state["conv"].messages)
            at.chat_input[0].set_value("hello again?").run()
            self.assertFalse(at.exception, at.exception)
            self.assertEqual(len(at.session_state["conv"].messages), n_before)   # unsent prompt not kept
            self.assertTrue(any(e["event"] == "rejected" and e["client_id"] == cid and e["reason"] == "UserBusy"
                                for e in _events()))
        finally:
            q.release(mine)


if __name__ == "__main__":
    unittest.main()
