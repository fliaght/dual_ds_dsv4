import json, os, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from storage import ChatLogger, Conversation, clean_id
from context import build_context, context_gauge, estimate_tokens, reasoning_meta, token_caption


class ConversationTests(unittest.TestCase):
    def test_from_untrusted_keeps_only_well_typed_fields(self):
        bad = {"id": "../../etc", "system_prompt": 42, "messages": [
            {"role": "user", "content": "hi", "ts": "2026-09-15T00:00:00Z", "meta": {"tok_s": 20.5, "evil": "x"}},
            {"role": "system", "content": "injected"},            # role not allowed
            {"role": "assistant", "content": 123},                # content not a string
            "garbage",
            {"role": "assistant", "content": "ok", "status": "weird", "error": 5, "reasoning": "r" * 10},
        ]}
        c = Conversation.from_untrusted(bad, max_messages=100)
        self.assertIsNotNone(c)
        self.assertRegex(c.id, r"^[0-9a-f]{32}$")               # bad id replaced
        self.assertEqual(c.system_prompt, "")
        self.assertEqual([m["role"] for m in c.messages], ["user", "assistant"])
        self.assertEqual(c.messages[0]["meta"], {"tok_s": 20.5})
        self.assertNotIn("status", c.messages[1]); self.assertNotIn("error", c.messages[1])
        self.assertEqual(c.messages[1]["reasoning"], "r" * 10)
        self.assertIsNone(Conversation.from_untrusted("nope", 10))
        self.assertIsNone(Conversation.from_untrusted({"messages": "nope"}, 10))
        long = {"messages": [{"role": "user", "content": "x" * 10} for _ in range(50)]}
        self.assertEqual(len(Conversation.from_untrusted(long, max_messages=7).messages), 7)
        self.assertEqual(len(Conversation.from_untrusted(long, max_messages=7, max_chars=3).messages[0]["content"]), 3)

    def test_roundtrip_and_trim(self):
        c = Conversation.new("sys")
        c.messages += [{"role": "user", "content": f"m{i}"} for i in range(5)]
        d = json.loads(json.dumps(c.to_dict()))
        c2 = Conversation.from_untrusted(d, max_messages=100)
        self.assertEqual((c2.id, c2.system_prompt, c2.messages), (c.id, "sys", c.messages))
        self.assertEqual(c.trim(3), 2); self.assertEqual([m["content"] for m in c.messages], ["m2", "m3", "m4"])
        self.assertEqual(c.trim(3), 0)
        self.assertEqual(clean_id("abcdef0123"), "abcdef0123")
        self.assertRegex(clean_id("ABC"), r"^[0-9a-f]{32}$"); self.assertEqual(clean_id(None, "fallback1"), "fallback1")


class LoggerTests(unittest.TestCase):
    def test_turn_and_event_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = ChatLogger(Path(tmp) / "chat")
            log.log_turn({"client_id": "c1", "session_id": "s1", "prompt": "p", "response": "r", "status": "ok"})
            log.log_turn({"client_id": "c2", "session_id": "s2", "prompt": "你好", "response": "r2", "status": "ok"})
            log.log_event("session_start", session_id="s1", ip="10.0.0.1")
            chat = list((Path(tmp) / "chat").glob("chat-*.jsonl")); ev = list((Path(tmp) / "chat").glob("events-*.jsonl"))
            self.assertEqual((len(chat), len(ev)), (1, 1))
            recs = [json.loads(l) for l in chat[0].read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["client_id"] for r in recs], ["c1", "c2"])       # all visitors, one file
            self.assertEqual(recs[1]["prompt"], "你好")                             # utf-8, not escaped
            self.assertTrue(all(r["kind"] == "turn" and "ts" in r for r in recs))
            self.assertEqual(json.loads(ev[0].read_text())["event"], "session_start")
            self.assertEqual(oct(os.stat(Path(tmp) / "chat").st_mode & 0o777), "0o700")


class TokenMetaTests(unittest.TestCase):
    def test_reply_numbers_are_this_replys_input_and_output(self):
        seen = []
        meta = {"prompt_tokens": 18000, "completion_tokens": 452,
                **reasoning_meta("plan", True, lambda t: seen.append(t) or 236)}
        self.assertEqual(seen, ["plan"])
        self.assertEqual(token_caption(meta), ["input 18,000 tok", "output 452 tok (thinking 236)"])
        self.assertEqual(reasoning_meta("", False, lambda t: self.fail("thinking off: no count")), {})
        self.assertEqual(reasoning_meta("", True, lambda t: 0), {"reasoning_tokens": 0, "reasoning_exact": True})
        guess = reasoning_meta("思考", True, lambda t: None)
        self.assertEqual(token_caption({"completion_tokens": 9, **guess}), ["output ≈9 tok (thinking ≈3)"])

    def test_old_and_untrusted_meta_never_break_the_caption(self):
        self.assertEqual(token_caption({"completion_tokens": 1234}), ["output ≈1,234 tok"])
        self.assertEqual(token_caption({"prompt_tokens": "9", "completion_tokens": -1}), [])
        self.assertEqual(token_caption({"prompt_tokens": 10, "completion_tokens": 2, "reasoning_tokens": float("inf")}),
                         ["input 10 tok", "output 2 tok"])
        kept = Conversation.from_untrusted({"messages": [{"role": "assistant", "content": "a", "meta": {
            "prompt_tokens": 5, "completion_tokens": 3, "reasoning_tokens": 2, "reasoning_exact": False,
            "context_tokens": 8}}]}, 10)
        self.assertNotIn("context_tokens", kept.messages[0]["meta"])        # the old, misleading total
        self.assertEqual(token_caption(kept.messages[0]["meta"]), ["input 5 tok", "output 3 tok (thinking ≈2)"])

    def test_context_gauge(self):
        self.assertEqual(context_gauge(18452, True, 425984, 0, 2048),
                         (18452 / 425984, "Context in use: 18,452 / 425,984 tokens (4.3%)", []))
        frac, label, notes = context_gauge(40, False, 425984, 0, 2048)
        self.assertEqual(label, "Context in use: ≈40 / 425,984 tokens (<0.1%)")
        self.assertEqual(notes, [])
        _, _, notes = context_gauge(350000, True, 425984, 2, 2048)
        self.assertEqual(notes, ["2 oldest messages no longer fit and are not sent to the model",
                                 "context nearly full: start a new chat to keep earlier messages in view"])
        frac, _, notes = context_gauge(420000, True, 425984, 1, 65536)
        self.assertLessEqual(frac, 1.0)
        self.assertEqual(notes[0], "1 oldest message no longer fits and is not sent to the model")
        self.assertIn("will not fit", notes[1])


class ContextTests(unittest.TestCase):
    def test_estimate(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertGreater(estimate_tokens("你好世界"), 4)                 # CJK ~1 tok/char
        self.assertLess(estimate_tokens("hello world"), 6)

    def test_trimming_keeps_newest_and_system(self):
        msgs = []
        for i in range(10):
            msgs.append({"role": "user", "content": f"question {i} " + "x" * 300})
            msgs.append({"role": "assistant", "content": f"answer {i} " + "y" * 300})
        msgs.append({"role": "user", "content": "final question"})
        api, dropped, est = build_context(msgs, "SYS", budget_tokens=400)
        self.assertEqual(api[0], {"role": "system", "content": "SYS"})
        self.assertEqual(api[-1]["content"], "final question")
        self.assertGreater(dropped, 0)
        self.assertEqual(api[1]["role"], "user")                         # never starts with assistant
        self.assertLessEqual(est, 400)
        api2, dropped2, _ = build_context(msgs, "", budget_tokens=10_000)
        self.assertEqual(dropped2, 0); self.assertEqual(len(api2), 21)
        api3, dropped3, _ = build_context([{"role": "user", "content": "z" * 5000}], "", budget_tokens=100)
        self.assertEqual(api3, []); self.assertEqual(dropped3, 1)
        api4, _, _ = build_context([{"role": "user", "content": "q"}, {"role": "assistant", "content": ""},
                                    {"role": "user", "content": "q2"}], "", 1000)
        self.assertEqual([m["role"] for m in api4], ["user", "user"])


if __name__ == "__main__":
    unittest.main()
