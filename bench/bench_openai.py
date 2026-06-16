#!/usr/bin/env python3
"""Async benchmark client for OpenAI-compatible /v1/chat/completions.

Streaming sweep over (input_len x output_len x concurrency). For each cell it
fires N requests and records TTFT (time-to-first-token), ITL (inter-token
latency), per-stream tok/s and aggregate tok/s.

Timeouts default to 600 s/request and 1800 s/cell — high enough for V4-Flash's
multi-minute cold prefills at 65k-120k context. Lower them with --timeout /
--cell-timeout for short-context runs.

    python3 bench/bench_openai.py --isl 128 1024 --osl 128 --conc 1 4 --n 8
"""
import argparse, asyncio, json, statistics, sys, time
from typing import Optional
try:
    import aiohttp
except ImportError:
    print("install aiohttp first: pip install --user aiohttp", file=sys.stderr); sys.exit(1)


def make_prompt(input_tokens: int) -> str:
    word = "benchmark "
    n_words = max(1, int(input_tokens / 1.3))
    return (word * n_words).strip()


async def one_request(session, url, model, prompt, max_tokens, req_timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    t_send = time.perf_counter()
    ttft: Optional[float] = None
    last_t = t_send
    itls = []
    n_tok = 0
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=req_timeout)) as resp:
            if resp.status != 200:
                txt = await resp.text()
                return {"ok": False, "err": f"HTTP {resp.status}: {txt[:200]}"}
            async for line in resp.content:
                if not line: continue
                line = line.strip()
                if not line.startswith(b"data: "): continue
                data = line[6:]
                if data == b"[DONE]": break
                try:
                    chunk = json.loads(data)
                except Exception:
                    continue
                ch = chunk.get("choices") or [{}]
                delta = ch[0].get("delta", {}) or {}
                content = delta.get("content") or delta.get("reasoning_content") or ""
                if not content: continue
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t_send
                else:
                    itls.append(now - last_t)
                last_t = now
                n_tok += 1
    except asyncio.TimeoutError:
        return {"ok": False, "err": "timeout"}
    except Exception as e:
        return {"ok": False, "err": str(e)[:200]}
    total = time.perf_counter() - t_send
    return {"ok": True, "ttft": ttft, "itls": itls, "n_tok": n_tok, "total": total}


async def run_cell(url, model, isl, osl, conc, n_req, req_timeout, cell_timeout):
    prompt = make_prompt(isl)
    sem = asyncio.Semaphore(conc)
    results = []
    timeout = aiohttp.ClientTimeout(total=cell_timeout)
    connector = aiohttp.TCPConnector(limit=conc * 2)

    async def worker():
        async with sem:
            return await one_request(session, url, model, prompt, osl, req_timeout)

    t_global_start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [asyncio.create_task(worker()) for _ in range(n_req)]
        for t in asyncio.as_completed(tasks):
            results.append(await t)
    t_global_end = time.perf_counter()

    ok = [r for r in results if r["ok"]]
    err = [r for r in results if not r["ok"]]
    if not ok:
        return {"isl": isl, "osl": osl, "concurrency": conc, "n_req": n_req,
                "n_ok": 0, "n_err": len(err), "errors": err[:3]}
    ttfts = sorted(r["ttft"] for r in ok if r["ttft"] is not None)
    all_itls = [v for r in ok for v in r["itls"]]
    tps_per_stream = [r["n_tok"] / r["total"] for r in ok]
    total_tokens = sum(r["n_tok"] for r in ok)
    wall = t_global_end - t_global_start
    return {
        "isl": isl, "osl": osl, "concurrency": conc,
        "n_ok": len(ok), "n_err": len(err),
        "ttft_p50": statistics.median(ttfts) if ttfts else None,
        "ttft_p95": ttfts[int(0.95 * (len(ttfts) - 1))] if ttfts else None,
        "itl_p50_ms": 1000 * statistics.median(all_itls) if all_itls else None,
        "tok_per_s_per_stream_p50": statistics.median(tps_per_stream),
        "throughput_tok_per_s": total_tokens / wall,
        "total_tokens": total_tokens,
        "wall_s": wall,
        "errors": err[:1],
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    p.add_argument("--isl", nargs="+", type=int, default=[128, 1024])
    p.add_argument("--osl", nargs="+", type=int, default=[128])
    p.add_argument("--conc", nargs="+", type=int, default=[1, 4])
    p.add_argument("--n", type=int, default=8, help="requests per cell")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="per-request timeout in seconds (high for long-context prefills)")
    p.add_argument("--cell-timeout", type=float, default=1800.0,
                   help="overall cell timeout in seconds")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    print(f"# bench {args.model} via {args.url}")
    print(f"# cells: ISL={args.isl} OSL={args.osl} CONC={args.conc} N={args.n}")
    print(f"# timeouts: per-req={args.timeout}s cell={args.cell_timeout}s")
    rows = []
    for isl in args.isl:
        for osl in args.osl:
            for conc in args.conc:
                print(f"\n>>> ISL={isl} OSL={osl} CONC={conc} N={args.n} ...", flush=True)
                t0 = time.time()
                r = await run_cell(args.url, args.model, isl, osl, conc, args.n,
                                   args.timeout, args.cell_timeout)
                print(f"    done in {time.time()-t0:.1f}s")
                print(f"    ok={r['n_ok']}/{r['n_ok']+r['n_err']}", end="")
                if r['n_ok']:
                    print(f"  TTFT p50={r['ttft_p50']*1000:.0f}ms p95={r['ttft_p95']*1000:.0f}ms"
                          f"  ITL p50={r['itl_p50_ms']:.1f}ms"
                          f"  per-stream={r['tok_per_s_per_stream_p50']:.1f} tok/s"
                          f"  agg={r['throughput_tok_per_s']:.1f} tok/s")
                else:
                    print(f"  ERRORS: {r['errors']}")
                rows.append(r)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
