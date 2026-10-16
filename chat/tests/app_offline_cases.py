"""The page through Streamlit's AppTest against a fake model server (fake_openai.py) -- offline.
Run by test_app_offline.py in a process of its own (this module points the process-wide client, queue
and settings at the fake server); directly: cd chat/tests && ../.venv/bin/python -m unittest app_offline_cases

AppTest runs no browser JavaScript, so the browser store never answers: this covers the server side of
New chat, Edit, Regenerate, Delete, a refused question and the settings reset.  The browser half
(IndexedDB, Copy, the chat list across reloads) is tests/browser_check.py.
"""
import json, os, shutil, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from fake_openai import FakeOpenAI  # noqa: E402

FAKE = FakeOpenAI().start()
TMP = tempfile.mkdtemp(prefix="dsv4chat-offline-")
os.environ.update(CHAT_API_BASE=FAKE.base_url, CHAT_LOG_DIR=TMP, CHAT_MONITOR="0", CHAT_MAX_CONCURRENCY="1",
                  CHAT_MAX_TOKENS_CAP="65536", CHAT_CONTEXT_TOKEN_BUDGET="409600", CHAT_VERSION="v9.9.9 · test")


def tearDownModule():
    FAKE.stop()
    shutil.rmtree(TMP, ignore_errors=True)


def _events(name):
    return [json.loads(l) for p in Path(TMP).glob("events-*.jsonl") for l in p.read_text(encoding="utf-8").splitlines()
            if json.loads(l)["event"] == name]


def _turns():
    return [json.loads(l) for p in Path(TMP).glob("chat-*.jsonl") for l in p.read_text(encoding="utf-8").splitlines()]


class OfflinePage(unittest.TestCase):
    def setUp(self):
        from streamlit.testing.v1 import AppTest
        self.at = AppTest.from_file(str(HERE.parent / "app.py"), default_timeout=60)
        self.at.run()
        self.assertFalse(self.at.exception, self.at.exception)

    def ask(self, text):
        self.at.chat_input[0].set_value(text).run()
        self.assertFalse(self.at.exception, self.at.exception)
        return self.at.session_state["conv"]

    def test_first_visit_and_a_reply(self):
        md = " ".join(m.value for m in self.at.markdown)
        self.assertIn("How can I help?", md)
        self.assertIn("v9.9.9 · test", " ".join(h.proto.body for h in self.at.get("html")))
        self.assertIn("Large language models can make mistakes.", " ".join(c.value for c in self.at.caption))
        conv = self.ask("First question")
        self.assertEqual([m["role"] for m in conv.messages], ["user", "assistant"])
        self.assertTrue(conv.messages[1]["content"].endswith("to: First question"))
        self.assertGreater(conv.updated, conv.created)
        self.assertTrue(self.at.button(key=f"chat-here-{conv.id}"))                  # in the chat list, marked open

    def test_edit_then_the_next_question_sees_it(self):
        conv = self.ask("Question to edit")
        self.at.session_state["editing"] = (conv.id, 1)                            # what the Edit action sets
        self.at.run()
        key = f"edit-{conv.id[:12]}-1"
        self.at.text_area(key=key).set_value("A better answer.")
        self.at.button(key=f"FormSubmitter:form-{key}-Save").click().run()
        self.assertFalse(self.at.exception, self.at.exception)
        m = self.at.session_state["conv"].messages[1]
        self.assertEqual((m["content"], m["edited"]), ("A better answer.", True))
        self.assertIsNone(self.at.session_state["editing"])
        self.assertEqual(_events("reply_edited")[-1]["chars_after"], len("A better answer."))
        self.ask("And now?")
        self.assertIn("A better answer.", [m["content"] for m in FAKE.chats()[-1]["messages"]])

    def test_regenerate_replaces_the_reply_and_what_followed(self):
        conv = self.ask("Regenerate me")
        first = conv.messages[1]["content"]
        self.ask("A later question")
        self.at.session_state["regen"] = (conv.id, 1)                              # what the Regenerate action sets
        self.at.run()
        self.assertFalse(self.at.exception, self.at.exception)
        msgs = self.at.session_state["conv"].messages
        self.assertEqual(len(msgs), 2)
        self.assertNotEqual(msgs[1]["content"], first)
        self.assertTrue(msgs[1]["content"].endswith("to: Regenerate me"))
        self.assertEqual(_events("reply_regenerated")[-1]["removed"], 3)
        self.assertTrue(_turns()[-1]["regenerated"])

    def test_new_chat_and_delete_without_a_browser_store(self):
        old = self.ask("Something to forget")
        self.at.button(key="new-chat").click().run()
        self.assertEqual(self.at.session_state["conv"].messages, [])
        self.assertNotEqual(self.at.session_state["conv"].id, old.id)
        self.assertEqual(_events("chat_new")[-1]["messages"], 0)
        doomed = self.ask("Delete me")
        self.at.button(key="delete-chat-confirm").click().run()
        self.assertEqual(self.at.session_state["conv"].messages, [])
        self.assertIn(doomed.id, self.at.session_state["deleted"])
        self.assertEqual(_events("chat_deleted")[-1]["conv_id"], doomed.id)

    def test_a_refused_question_goes_back_into_the_input(self):
        from runtime import get_queue
        from settings import load_settings
        self.ask("warm up")                                                         # fixes this visitor's id
        q = get_queue(load_settings())
        mine = q.submit(self.at.session_state["client_id"])
        self.assertTrue(q.try_acquire(mine))                                        # the same visitor, another tab
        try:
            conv = self.ask("Not now?")
            self.assertEqual(len(conv.messages), 2)
            self.assertIn("not sent", " ".join(i.value for i in self.at.info))
            box = self.at.chat_input[0].proto
            self.assertTrue(box.set_value and box.value == "Not now?", box)          # back in the input box
        finally:
            q.release(mine)

    def test_settings_reset(self):
        self.at.slider(key="set_temperature").set_value(1.2).run()
        self.assertEqual(self.at.session_state["set_temperature"], 1.2)
        self.at.button(key="settings-reset").click().run()
        self.assertEqual(self.at.session_state["set_temperature"], 0.7)


if __name__ == "__main__":
    unittest.main()
