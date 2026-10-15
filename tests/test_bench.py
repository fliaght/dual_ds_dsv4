"""Exercise the actual HTTP/SSE benchmark against a local fake API, never a model."""
import asyncio
import json
import sys
import unittest
from pathlib import Path

import aiohttp
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from bench_openai import one_request, run_cell  # noqa: E402


class BenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mode = "normal"
        self.payloads = []

        async def handler(request):
            self.payloads.append(await request.json())
            if self.mode == "http_error":
                return web.json_response({"error": "busy"}, status=503)
            if self.mode == "slow":
                await asyncio.sleep(0.3)
            records = [
                {"choices": [{"delta": {"content": "several tokens together"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "another group"}, "finish_reason": "length"}]},
            ]
            if self.mode != "missing_usage":
                records.append({"choices": [], "usage": {"prompt_tokens": 23, "completion_tokens": 9}})
            body = "".join(f"data: {json.dumps(record)}\n\n" for record in records)
            if self.mode != "truncated":
                body += "data: [DONE]\n\n"
            return web.Response(text=body, content_type="text/event-stream")

        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = self.runner.addresses[0][1]
        self.url = f"http://127.0.0.1:{port}/v1/chat/completions"

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def request(self):
        async with aiohttp.ClientSession() as session:
            return await one_request(session, self.url, "fake", "hi", 16, 1)

    async def test_speculative_chunks_use_actual_usage(self):
        result = await self.request()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["n_tok"], 9)
        self.assertEqual(result["n_chunks"], 2)
        self.assertEqual(result["prompt_tokens"], 23)
        self.assertTrue(self.payloads[0]["stream_options"]["include_usage"])

    async def test_bad_responses_do_not_become_measurements(self):
        for mode, error in (("missing_usage", "usage"), ("truncated", "incomplete"), ("http_error", "503")):
            with self.subTest(mode=mode):
                self.mode = mode
                result = await self.request()
                self.assertFalse(result["ok"])
                self.assertIn(error, result["err"])

    async def test_aggregate_counts_tokens_and_labels_chunk_timing(self):
        result = await run_cell(self.url, "fake", 128, 16, 2, 3, 1, 2)
        self.assertEqual(result["n_ok"], 3)
        self.assertEqual(result["total_tokens"], 27)
        self.assertEqual(result["total_chunks"], 6)
        self.assertEqual(result["prompt_tokens_min"], 23)
        self.assertIn("inter_chunk_p50_ms", result)
        self.assertNotIn("itl_p50_ms", result)

    async def test_cell_timeout_includes_queued_requests(self):
        self.mode = "slow"
        result = await run_cell(self.url, "fake", 128, 16, 1, 3, 10, 0.05)
        self.assertEqual(result["n_ok"], 0)
        self.assertEqual(result["n_err"], 3)
        self.assertTrue(all(r["err"] == "cell timeout" for r in result["errors"]))
        self.assertLess(result["wall_s"], 1)



class TfEvalTests(unittest.TestCase):
    """bench/tf_eval.py offline: opt-in, needle prompts and the admission gate; no server, no SSH."""

    def setUp(self):
        import tf_eval
        self.tf_eval = tf_eval

    def test_refuses_without_live_opt_in(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"CHAT_LIVE_TESTS": ""}):
            with self.assertRaises(SystemExit) as ctx:
                self.tf_eval.main(["web", "--peer", "x", "--guard-state", "/nonexistent", "--out", "/nonexistent"])
        self.assertIn("CHAT_LIVE_TESTS=1", str(ctx.exception))

    def test_needle_prompt_is_unique_and_answerable(self):
        import random
        rng = random.Random(1)
        first, code = self.tf_eval.needle_doc(rng, 2000)
        second, _ = self.tf_eval.needle_doc(rng, 2000)
        self.assertIn(f"is {code}.", first)
        self.assertTrue(first.endswith("Reply with just the number."))
        self.assertNotEqual(first, second)

    def test_health_accepts_vllm_empty_body(self):
        import types
        from unittest import mock
        run = self.tf_eval.Run.__new__(self.tf_eval.Run)
        run.a = types.SimpleNamespace(base="http://127.0.0.1:9")
        empty = mock.Mock(status_code=200, raise_for_status=mock.Mock(), json=mock.Mock(side_effect=ValueError))
        tf = mock.Mock(status_code=200, raise_for_status=mock.Mock(), json=mock.Mock(return_value={"inflight": 0}))
        with mock.patch.object(self.tf_eval.requests, "get", side_effect=[empty, tf]):
            self.assertEqual(run.health(), {})
            self.assertEqual(run.health(), {"inflight": 0})

    def test_admission_waits_for_both_nodes_then_gives_up(self):
        import types
        from unittest import mock
        run = self.tf_eval.Run.__new__(self.tf_eval.Run)
        run.a = types.SimpleNamespace(admission_mib=9216, admission_free_mib=5120)
        low = [{"available_mib": 20000, "free_mib": 9000}, {"available_mib": 20000, "free_mib": 5000}]
        ok = [{"available_mib": 20000, "free_mib": 9000}, {"available_mib": 9216, "free_mib": 5120}]
        run.guards = mock.Mock(side_effect=[low, ok])
        with mock.patch.object(self.tf_eval.time, "sleep"):
            self.assertEqual(run.admit(), [[20000, 9000], [9216, 5120]])
        run.guards = mock.Mock(return_value=low)
        with mock.patch.object(self.tf_eval.time, "sleep"), \
                mock.patch.object(self.tf_eval.time, "time", side_effect=[0, 1000]):
            with self.assertRaises(SystemExit):
                run.admit(wait_s=180)


if __name__ == "__main__":
    unittest.main()
