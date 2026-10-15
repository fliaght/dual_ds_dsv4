#!/usr/bin/env python3
"""Torch-profiler capture and kernel-time summary for the V4.1 service.

capture   needs CHAT_LIVE_TESTS=1 and a server started with V41_PROFILER_DIR set
          (maintenance only: it adds /start_profile and /stop_profile).
summarize reads the gzipped Chrome traces the ranks wrote; offline.
"""
import argparse
import collections
import gzip
import json
import os
from pathlib import Path
import time

# First matching substring wins; order puts the specific EXL3/NCCL names first.
GROUPS = (('exl3_moe', 'moe'), ('exl3', 'exl3'), ('nccl', 'nccl'), ('allreduce', 'nccl'),
          ('flashinfer', 'attention'), ('mla', 'attention'), ('index', 'indexer'),
          ('topk', 'indexer'), ('tilelang', 'mhc'), ('gemv', 'gemm'), ('gemm', 'gemm'),
          ('memcpy', 'memcpy'), ('memset', 'memcpy'))


def group(name):
    lower = name.lower()
    return next((label for key, label in GROUPS if key in lower), 'other')


def summarize(path, steps):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        events = json.load(stream)['traceEvents']
    kernels = [e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset') and 'dur' in e]
    if not kernels:
        raise SystemExit(f'{path}: no GPU events')
    start = min(e['ts'] for e in kernels)
    end = max(e['ts'] + e['dur'] for e in kernels)
    by_group, by_name = collections.Counter(), collections.Counter()
    for e in kernels:
        by_group[group(e['name'])] += e['dur']
        by_name[e['name'][:90]] += e['dur']
    busy = sum(by_group.values())
    host = sum(e['dur'] for e in events if e.get('cat') == 'cuda_runtime'
               and 'LaunchHostFunc' in e.get('name', ''))
    per = max(steps, 1) * 1000.0
    print(f'{path}: window {(end - start) / per:.2f} ms/step, GPU busy {busy / per:.2f} ms/step, '
          f'{len(kernels) / max(steps, 1):.0f} GPU ops/step, host-func launches {host / per:.3f} ms/step')
    for label, dur in by_group.most_common():
        print(f'  {label:<10} {dur / per:8.3f} ms/step  {100 * dur / busy:5.1f}%')
    for name, dur in by_name.most_common(15):
        print(f'    {dur / per:8.3f} ms  {name}')


def capture(base, model, prompt_words, max_tokens):
    import requests
    if os.environ.get('CHAT_LIVE_TESTS') != '1':
        raise SystemExit('Set CHAT_LIVE_TESTS=1 explicitly for real inference')
    prompt = 'Summarize this text in one paragraph. ' + ' '.join(f'word{i % 997}' for i in range(prompt_words))
    payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': max_tokens,
               'temperature': 0, 'ignore_eos': True, 'chat_template_kwargs': {'enable_thinking': False}}
    requests.post(base + '/start_profile', timeout=60).raise_for_status()
    started = time.perf_counter()
    try:
        response = requests.post(base + '/v1/chat/completions', json=payload, timeout=(15, 1800))
        response.raise_for_status()
        usage = response.json()['usage']
    finally:
        requests.post(base + '/stop_profile', timeout=600).raise_for_status()
    print(json.dumps({'usage': usage, 'elapsed_s': time.perf_counter() - started}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    cap = sub.add_parser('capture')
    cap.add_argument('--base', default='http://127.0.0.1:8000')
    cap.add_argument('--model', required=True)
    cap.add_argument('--prompt-words', type=int, default=16)
    cap.add_argument('--max-tokens', type=int, default=32)
    summ = sub.add_parser('summarize')
    summ.add_argument('traces', type=Path, nargs='+')
    summ.add_argument('--steps', type=int, default=1, help='decode steps captured, for per-step figures')
    args = parser.parse_args()
    if args.action == 'capture':
        capture(args.base, args.model, args.prompt_words, args.max_tokens)
    else:
        for trace in args.traces:
            summarize(trace, args.steps)


if __name__ == '__main__':
    main()
