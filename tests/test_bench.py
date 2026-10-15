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


if __name__ == "__main__":
    unittest.main()
