#!/usr/bin/env python3
"""Explicit live V4.1 text checks, with real token counts and staged admission."""
import argparse
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import time

import requests


def guards(path, peer, minimum_mib, minimum_free_mib=6144):
    samples = []
    if peer:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                                 peer, 'cat', shlex.quote(str(path))], check=True, capture_output=True,
                                text=True, timeout=10)
        samples.append(json.loads(result.stdout))
    # Read local state after SSH, so a slow connection cannot age our local
    # snapshot into an apparent monitor failure.
    samples.insert(0, json.loads(path.read_text()))
    for sample in samples:
        if (sample['reason'] or time.time() - sample['time'] > 5
                or sample['available_mib'] < minimum_mib
                or sample['free_mib'] < minimum_free_mib):
            raise RuntimeError(f'unsafe/stale memory admission: {sample}')
    return samples


def token_count(base, model, messages):
    response = requests.post(base + '/tokenize', json={
        'model': model, 'messages': messages, 'add_generation_prompt': True,
        'chat_template_kwargs': {'enable_thinking': False}}, timeout=120)
    response.raise_for_status()
    return response.json()['count']


def context_prompt(base, model, target, seed):
    rng = random.Random(seed)
    key = f'CHECK-{rng.getrandbits(48):012x}'
    value = f'{rng.getrandbits(64):016x}'
    # Varied records exercise actual Engram I/O. A repeated one-token prompt
    # would give unrepresentative prefill/cache numbers.
    records = ''.join(f'Record {i:06d}: region {rng.randrange(500)}; '
                      f'reading {rng.getrandbits(48):012x}; note normal observation.\n'
                      for i in range(max(100, target // 10)))

    def messages(length):
        body = records[:length]
        mid = len(body) // 2
        content = (f'Archive batch {seed}. Read the records and answer the question at the end.\n'
                   + body[:mid] + f'\nLookup entry {key} has value {value}.\n' + body[mid:]
                   + f'\nWhat is the value of lookup entry {key}? Reply only with its value.')
        return [{'role': 'user', 'content': content}]

    lo, hi = 0, len(records)
    if token_count(base, model, messages(hi)) < target:
        raise RuntimeError('insufficient generated text')
    best = None
    while lo <= hi:
        length = (lo + hi) // 2
        result = messages(length)
        count = token_count(base, model, result)
        if target <= count <= target + 32:
            return result, value, count
        if count < target:
            lo = length + 1
        else:
            best = result, value, count
            hi = length - 1
    if best is None:
        raise RuntimeError('could not construct requested context')
    return best


def request(base, model, messages, output_tokens, **extra):
    started = time.perf_counter()
    first = None
    usage = None
    answer = ''
    finish = None
    done = False
    payload = {'model': model, 'messages': messages, 'max_tokens': output_tokens,
               'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True},
               'chat_template_kwargs': {'enable_thinking': False}, **extra}
    with requests.post(base + '/v1/chat/completions', json=payload, stream=True,
                       timeout=(15, 1800)) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith(b'data:'):
                continue
            raw = line[5:].strip()
            if raw == b'[DONE]':
                done = True
                break
            chunk = json.loads(raw)
            if chunk.get('error'):
                raise RuntimeError(str(chunk['error']))
            usage = chunk.get('usage') or usage
            for choice in chunk.get('choices', []):
                finish = choice.get('finish_reason') or finish
                delta = choice.get('delta', {})
                text = delta.get('content') or ''
                if text or delta.get('reasoning') or delta.get('reasoning_content'):
                    first = first or time.perf_counter()
                answer += text
    end = time.perf_counter()
    if not done or finish not in ('stop', 'length') or not usage or first is None:
        raise RuntimeError('incomplete response or missing server token usage')
    completion = usage['completion_tokens']
    return dict(usage=usage, finish_reason=finish, answer=answer, elapsed_s=end-started,
                ttft_s=first-started, request_tok_s=completion/(end-started),
                post_ttft_tok_s_estimate=completion/(end-first))


def main():
    if os.environ.get('CHAT_LIVE_TESTS') != '1':
        raise SystemExit('Set CHAT_LIVE_TESTS=1 explicitly for real inference')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8000')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--guard-state', type=Path, required=True)
    parser.add_argument('--peer', required=True)
    parser.add_argument('--sizes', default='1024,32768,131072,210000,250000')
    parser.add_argument('--speed-repeats', type=int, default=3)
    parser.add_argument('--admission-mib', type=int, default=10240)
    parser.add_argument('--long-speed', action='store_true',
                        help='after the ladder, decode 1024 tokens on the largest (now cached) context')
    parser.add_argument('--long-output-tokens', type=int, default=0,
                        help='finally force one short-prompt reply to exactly this many tokens (ignore_eos)')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def run_case(kind, messages, max_tokens, expected=None, target=0, **extra):
        before = guards(args.guard_state, args.peer, args.admission_mib)
        result = request(args.base, args.model, messages, max_tokens, **extra)
        result.update(kind=kind, time=time.time(), model=args.model, memory_before=before,
                      memory_after=guards(args.guard_state, args.peer, 8192, 4096))
        result['capacity_ok'] = result['usage']['prompt_tokens'] >= target
        result['answer_ok'] = expected is None or expected in result['answer']
        with args.output.open('a') as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'answer'}, ensure_ascii=False), flush=True)
        if not result['capacity_ok']:
            raise RuntimeError('actual prompt length did not reach target')
        return result

    smoke = run_case('arithmetic', [{'role': 'user', 'content': 'What is 17*19? Reply only with the integer.'}], 32, '323')
    if smoke['answer'].strip() != '323':
        raise RuntimeError('arithmetic smoke failed')
    largest = None
    for target in [int(s) for s in args.sizes.split(',') if s]:
        guards(args.guard_state, args.peer, args.admission_mib)
        messages, expected, count = context_prompt(args.base, args.model, target, target + 17)
        print(f'context target={target} tokenized={count}', flush=True)
        run_case('context', messages, 128, expected, target)
        largest = messages, expected, target
    if args.long_speed and largest:
        messages, expected, target = largest
        messages = [{'role': 'user', 'content': messages[0]['content'] + '\nAfter stating that lookup value, '
                     'write a numbered list of 200 practical Python programming tips, with one full sentence '
                     'explaining each tip. Start immediately.'}]
        run_case('long_speed', messages, 1024, expected, target)
    for repeat in range(args.speed_repeats):
        messages = [{'role': 'user', 'content': f'Exercise {repeat}. Write a numbered list of 200 practical '
                     'Python programming tips, with one full sentence explaining each tip. Start immediately.'}]
        run_case('speed', messages, 1024)
    if args.long_output_tokens:
        messages = [{'role': 'user', 'content': 'Write a long, detailed technical handbook on Python programming, '
                     'chapter by chapter. Start immediately.'}]
        result = run_case('long_output', messages, args.long_output_tokens, ignore_eos=True)
        if result['usage']['completion_tokens'] != args.long_output_tokens:
            raise RuntimeError('long output stopped before its token budget')


if __name__ == '__main__':
    main()
