#!/usr/bin/env python3
"""Pre-warm the vLLM service to kill first-request TTFT spikes.

Triton/Inductor compile some kernels lazily the first time each prompt-shape
regime is seen. We send progressive ISL/OSL probes so the JIT cache covers what
real traffic will hit. Without this the first user request eats 4-12 s of
one-off autotune latency. Auto-run by scripts/03-start-serve.sh.

Stages: short (warm tokenizer/sampler/MTP draft), mid (1-2 chunked-prefill
iterations), long (long-chunk + sliding-window attention), rewarm (confirm
short is now warm: expect < ~1.5 s wall, > ~20 tok/s).
"""
import argparse, json, sys, time
import urllib.request

ALICE = (
    'Alice was beginning to get very tired of sitting by her sister on the bank, '
    'and of having nothing to do: once or twice she had peeped into the book her '
    'sister was reading, but it had no pictures or conversations in it. ' * 50
)

STAGES = [
    ("short ", 100,   32),
    ("mid   ", 4000,  64),
    ("long  ", 30000, 64),
    ("rewarm", 100,   32),  # confirm short is warm now
]


def call(url, model, label, isl_chars, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": ALICE[:isl_chars]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            data = json.loads(r.read())
    except Exception as e:
        return {"label": label, "ok": False, "err": str(e)[:200]}
    wall = time.time() - t0
    u = data["usage"]
    return {"label": label, "ok": True, "wall_s": round(wall, 2),
            "ptoks": u["prompt_tokens"], "ctoks": u["completion_tokens"],
            "tok_s": round(u["completion_tokens"] / wall, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    args = p.parse_args()

    print(f"# prewarm via {args.url}")
    for label, isl_chars, osl in STAGES:
        r = call(args.url, args.model, label, isl_chars, osl)
        if r["ok"]:
            print(f"  {label} ptoks={r['ptoks']:>6} ctoks={r['ctoks']:>4} "
                  f"wall={r['wall_s']:.2f}s tok/s={r['tok_s']}")
        else:
            print(f"  {label} ERROR  {r['err']}")
            sys.exit(1)
    print("prewarm done")


if __name__ == "__main__":
    main()
