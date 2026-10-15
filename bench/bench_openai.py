#!/usr/bin/env python3
"""Streaming OpenAI benchmark; actual token counts come from server usage.

ISL is an approximate prompt target; OSL is a request budget, not an assertion
about actual generated length. Timings describe SSE chunks, which can contain
multiple tokens. See Knowledge/performance.md for metric definitions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time

import aiohttp


def make_prompt(input_tokens: int) -> str:
    return ("benchmark " * max(1, int(input_tokens / 1.3))).strip()


async def one_request(session, url, model, prompt, max_tokens, req_timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
    }
    started = time.perf_counter()
    ttft = None
    last_chunk = None
    intervals = []
    n_chunks = 0
    usage = None
    finish_reason = None
    seen_done = False
    try:
        async with session.post(url, json=payload,
                                timeout=aiohttp.ClientTimeout(total=req_timeout)) as resp:
            if resp.status != 200:
                return {"ok": False, "err": f"HTTP {resp.status}: {(await resp.text())[:200]}"}
            async for line in resp.content:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    seen_done = True
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    return {"ok": False, "err": f"stream error: {str(chunk['error'])[:200]}"}
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    if not (delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")):
                        continue
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - started
                    if last_chunk is not None:
                        intervals.append(now - last_chunk)
                    last_chunk = now
                    n_chunks += 1
    except asyncio.TimeoutError:
        return {"ok": False, "err": "request timeout"}
    except (aiohttp.ClientError, ValueError, TypeError, AttributeError) as exc:
        return {"ok": False, "err": str(exc)[:200]}
    if not seen_done or not finish_reason:
        return {"ok": False, "err": "incomplete SSE stream (missing finish_reason or [DONE])"}
    if not isinstance(usage, dict) or any(
        type(usage.get(key)) is not int or usage[key] < 0
        for key in ("prompt_tokens", "completion_tokens")
    ):
        return {"ok": False, "err": "missing/invalid usage; cannot measure exact token throughput"}
    return {
        "ok": True, "ttft": ttft, "inter_chunk_s": intervals,
        "n_chunks": n_chunks, "n_tok": usage["completion_tokens"],
        "prompt_tokens": usage["prompt_tokens"], "finish_reason": finish_reason,
        "total": time.perf_counter() - started,
    }


async def run_cell(url, model, isl, osl, conc, n_req, req_timeout, cell_timeout):
    prompt = make_prompt(isl)
    sem = asyncio.Semaphore(conc)
    started = time.perf_counter()
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=conc * 2)) as session:
        async def worker():
            async with sem:
                return await one_request(session, url, model, prompt, osl, req_timeout)

        tasks = [asyncio.create_task(worker()) for _ in range(n_req)]
        try:
            done, pending = await asyncio.wait(tasks, timeout=cell_timeout)
            results = [task.result() for task in done]
            results.extend({"ok": False, "err": "cell timeout"} for _ in pending)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    wall = time.perf_counter() - started
    ok = [r for r in results if r["ok"]]
    errors = [r for r in results if not r["ok"]]
    row = {
        "isl": isl, "osl": osl, "concurrency": conc, "n_req": n_req,
        "n_ok": len(ok), "n_err": len(errors), "wall_s": wall, "errors": errors[:3],
        "metric_version": 2, "token_count_source": "server_usage",
    }
    if not ok:
        return row
    ttfts = sorted(r["ttft"] for r in ok if r["ttft"] is not None)
    intervals = [v for r in ok for v in r["inter_chunk_s"]]
    total_tokens = sum(r["n_tok"] for r in ok)
    row.update({
        "ttft_p50": statistics.median(ttfts) if ttfts else None,
        "ttft_p95": ttfts[int(0.95 * (len(ttfts) - 1))] if ttfts else None,
        "inter_chunk_p50_ms": 1000 * statistics.median(intervals) if intervals else None,
        "tok_per_s_per_stream_p50": statistics.median(r["n_tok"] / r["total"] for r in ok),
        "throughput_tok_per_s": total_tokens / wall,
        "total_tokens": total_tokens, "total_chunks": sum(r["n_chunks"] for r in ok),
        "prompt_tokens_min": min(r["prompt_tokens"] for r in ok),
        "prompt_tokens_max": max(r["prompt_tokens"] for r in ok),
        "completion_tokens_min": min(r["n_tok"] for r in ok),
        "completion_tokens_max": max(r["n_tok"] for r in ok),
    })
    return row


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_float(value):
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def display_ms(value):
    return "n/a" if value is None else f"{value * 1000:.1f} ms"


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=f"http://127.0.0.1:{os.environ.get('API_PORT', '8000')}/v1/chat/completions")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    parser.add_argument("--isl", nargs="+", type=positive_int, default=[128, 1024])
    parser.add_argument("--osl", nargs="+", type=positive_int, default=[128])
    parser.add_argument("--conc", nargs="+", type=positive_int, default=[1, 4])
    parser.add_argument("--n", type=positive_int, default=8)
    parser.add_argument("--timeout", type=positive_float, default=600.0)
    parser.add_argument("--cell-timeout", type=positive_float, default=1800.0)
    parser.add_argument("--out")
    args = parser.parse_args()
    rows = []
    print(f"# bench {args.model} via {args.url}; exact tokens from usage; rates include prefill")
    for isl in args.isl:
        for osl in args.osl:
            for conc in args.conc:
                print(f">>> ISL target={isl} OSL budget={osl} CONC={conc} N={args.n}", flush=True)
                row = await run_cell(args.url, args.model, isl, osl, conc, args.n,
                                     args.timeout, args.cell_timeout)
                print(f"    ok={row['n_ok']}/{args.n} wall={row['wall_s']:.1f}s")
                if row["n_ok"]:
                    print(f"    TTFT p50={display_ms(row['ttft_p50'])}; "
                          f"chunk interval p50={row['inter_chunk_p50_ms']} ms; "
                          f"aggregate={row['throughput_tok_per_s']:.2f} tok/s")
                if row["n_err"]:
                    print(f"    errors={row['errors']}")
                rows.append(row)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(rows, handle, indent=2)
        print(f"wrote {args.out}")
    return int(any(row["n_err"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
