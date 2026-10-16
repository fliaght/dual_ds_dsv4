"""Chat list, titles, date groups and saved settings (history.py) -- offline, no Streamlit."""
import os, sys, unittest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from history import clean_index, clean_settings, group_by_day, meta_of, title_of, upsert
from storage import Conversation

DEFAULTS = {"thinking": "off", "max_tokens": 2048, "temperature": 0.7, "top_p": 0.95, "system_prompt": "S"}


def entry(i, updated):
    return {"id": f"{i:032x}", "title": f"chat {i}", "created": updated, "updated": updated, "messages": 2}


class TitleTests(unittest.TestCase):
    def test_title_is_the_first_question_on_one_line(self):
        c = Conversation.new()
        self.assertEqual(title_of(c), "New chat")
        c.messages += [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "  What is\n17 × 23?  "}]
        self.assertEqual(title_of(c), "What is 17 × 23?")
        c.messages[1]["content"] = "x" * 100
        self.assertEqual(len(title_of(c)), 60)
        self.assertTrue(title_of(c).endswith("…"))
        m = meta_of(c)
        self.assertEqual((m["id"], m["messages"], m["updated"]), (c.id, 2, c.updated))


class IndexTests(unittest.TestCase):
    def test_untrusted_list_is_cleaned_deduplicated_and_sorted(self):
        raw = [entry(1, "2026-10-01T10:00:00+00:00"), entry(2, "2026-10-03T10:00:00+00:00"),
               {"id": "../x", "title": "bad id"}, "junk", {"id": f"{1:032x}", "updated": "2026-10-02T00:00:00+00:00",
               "title": 5, "messages": True}]
        out = clean_index(raw)
        self.assertEqual([e["id"][-1] for e in out], ["2", "1"])
        self.assertEqual((out[1]["title"], out[1]["messages"]), ("Untitled", 0))   # later duplicate wins, cleaned
        self.assertEqual(clean_index("nope"), [])
        newer = dict(entry(1, "2026-10-04T00:00:00+00:00"), title="renamed")
        self.assertEqual([e["title"] for e in upsert(out, newer)], ["renamed", "chat 2"])

    def test_groups_follow_the_visitors_calendar(self):
        now = datetime(2026, 10, 16, 1, 30, tzinfo=timezone.utc)                 # 03:30 in Berlin
        idx = clean_index([entry(1, "2026-10-15T23:00:00+00:00"),                 # 01:00 on the 16th in Berlin
                           entry(2, "2026-10-15T12:00:00+00:00"), entry(3, "2026-10-10T12:00:00+00:00"),
                           entry(4, "2026-09-25T12:00:00+00:00"), entry(5, "2025-01-01T00:00:00+00:00"),
                           dict(entry(6, ""), updated="not a date")])
        groups = group_by_day(idx, now, ZoneInfo("Europe/Berlin"))
        self.assertEqual([(g, [e["id"][-1] for e in es]) for g, es in groups],
                         [("Today", ["1"]), ("Yesterday", ["2"]), ("Previous 7 days", ["3"]),
                          ("Previous 30 days", ["4"]), ("Older", ["6", "5"])])
        first = [e for e in idx if e["id"].endswith("1")]
        self.assertEqual(group_by_day(first, now, timezone.utc)[0][0], "Yesterday")   # 23:00 UTC on the 15th


class SettingsTests(unittest.TestCase):
    def test_saved_settings_are_checked_and_clamped(self):
        kw = {"max_tokens_cap": 65536, "thinking_options": ("off", "high", "max")}
        self.assertEqual(clean_settings(None, DEFAULTS, **kw), DEFAULTS)
        got = clean_settings({"thinking": "max", "max_tokens": 70000, "temperature": 0.83, "top_p": 0.02,
                              "system_prompt": "x" * 5000}, DEFAULTS, **kw)
        self.assertEqual((got["thinking"], got["max_tokens"], got["temperature"], got["top_p"]), ("max", 65536, 0.85, 0.1))
        self.assertEqual(len(got["system_prompt"]), 4000)
        bad = clean_settings({"thinking": "ultra", "max_tokens": True, "temperature": float("nan"), "top_p": "0.5",
                              "system_prompt": 3}, DEFAULTS, **kw)
        self.assertEqual(bad, DEFAULTS)
        self.assertEqual(clean_settings({"max_tokens": 100}, DEFAULTS, **kw)["max_tokens"], 128)   # slider step 64
        self.assertEqual(clean_settings({"max_tokens": 9000}, DEFAULTS, max_tokens_cap=8192,
                                        thinking_options=("off",))["max_tokens"], 8192)


class ConversationVersionTests(unittest.TestCase):
    def test_updated_and_edited_survive_the_browser(self):
        c = Conversation.new()
        self.assertEqual(c.updated, c.created)
        c.messages.append({"role": "assistant", "content": "x", "edited": True})
        c.touch()
        back = Conversation.from_untrusted(c.to_dict(), 10)
        self.assertEqual((back.updated, back.messages[0]["edited"]), (c.updated, True))
        old = Conversation.from_untrusted({"created": "2026-01-01T00:00:00+00:00", "messages": [
            {"role": "assistant", "content": "y", "edited": "yes"}]}, 10)
        self.assertEqual(old.updated, "2026-01-01T00:00:00+00:00")                 # older copies: created
        self.assertNotIn("edited", old.messages[0])


if __name__ == "__main__":
    unittest.main()
