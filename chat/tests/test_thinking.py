"""Thinking mode in llm_client -- offline: requests.post is faked, no model server needed."""
import json, os, sys, unittest
from unittest import mock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import llm_client
from llm_client import LLMClient, ThinkSplitter


def _split(chunks):
    sp, out = ThinkSplitter(), []
    for c in chunks:
        out += sp.feed(c)
    out += sp.flush()
    joined = {"reasoning": "", "content": ""}
    for ev in out:
        joined[ev["type"]] += ev["text"]
    return joined


class SplitterTests(unittest.TestCase):
    def test_inline_reasoning_then_answer(self):
        self.assertEqual(_split(["Let me ", "think.</think>", "The answer is 4."]),
                         {"reasoning": "Let me think.", "content": "The answer is 4."})

    def test_tag_inside_one_delta(self):   # MTP can emit several tokens per delta
        self.assertEqual(_split(["2+2 is easy.</think>\n\n4"]), {"reasoning": "2+2 is easy.", "content": "4"})

    def test_tag_cut_across_deltas(self):
        self.assertEqual(_split(["abc</th", "ink>xyz"]), {"reasoning": "abc", "content": "xyz"})

    def test_leading_think_tag_stripped(self):
        self.assertEqual(_split(["<thi", "nk>hmm</think>ok"]), {"reasoning": "hmm", "content": "ok"})

    def test_cut_off_before_close_is_all_reasoning(self):
        self.assertEqual(_split(["still thinking </t"]), {"reasoning": "still thinking </t", "content": ""})

    def test_lt_that_is_not_a_tag(self):
        self.assertEqual(_split(["a < b", "</think>yes"]), {"reasoning": "a < b", "content": "yes"})


class _FakeResp:
    status_code = 200

    def __init__(self, deltas):
        lines = [f"data: {json.dumps({'choices': [{'delta': d, 'finish_reason': None}]})}" for d in deltas]
        lines.append(f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}")
        lines.append("data: [DONE]")
        self._lines = lines

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)

    def close(self):
        pass


def _run(deltas, model="m", **kw):
    cli = LLMClient(model=model)
    cli._max_model_len = 1000
    with mock.patch.object(llm_client.requests, "post", return_value=_FakeResp(deltas)) as post:
        events = list(cli.stream_chat([{"role": "user", "content": "hi"}], **kw))
    return post.call_args.kwargs["json"], events


class StreamChatTests(unittest.TestCase):
    def test_v41_off_explicitly_disables_template_default(self):
        model = 'dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw'
        payload, _ = _run([{'content': '323'}], model=model)
        self.assertEqual(payload['chat_template_kwargs'], {'enable_thinking': False})
        for level in ('high', 'max'):
            payload, _ = _run([], model=model, thinking=level)
            self.assertEqual(payload['chat_template_kwargs'],
                             {'enable_thinking': True, 'reasoning_effort': level})

    def test_off_sends_no_template_kwargs_and_passes_content_through(self):
        payload, events = _run([{"content": "a</think>b"}])
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertEqual([e for e in events if e["type"] == "content"], [{"type": "content", "text": "a</think>b"}])

    def test_high_and_max_payload(self):
        for level in ("high", "max"):
            payload, _ = _run([{"content": "x"}], thinking=level)
            self.assertEqual(payload["chat_template_kwargs"], {"thinking": True, "reasoning_effort": level})

    def test_inline_reasoning_is_split(self):
        _, events = _run([{"content": "plan"}, {"content": "</think>"}, {"content": "done"}], thinking="high")
        kinds = [(e["type"], e.get("text")) for e in events]
        self.assertEqual(kinds, [("reasoning", "plan"), ("content", "done"), ("finish", None)])

    def test_server_side_reasoning_disables_split(self):
        _, events = _run([{"reasoning_content": "plan"}, {"content": "done"}], thinking="max")
        kinds = [(e["type"], e.get("text")) for e in events]
        self.assertEqual(kinds, [("reasoning", "plan"), ("content", "done"), ("finish", None)])

    def test_bad_level_rejected(self):
        with self.assertRaises(ValueError):
            _run([], thinking="medium")


if __name__ == "__main__":
    unittest.main()
