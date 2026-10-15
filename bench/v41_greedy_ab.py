#!/usr/bin/env python3
"""Greedy A/B: record fixed-prompt outputs on one build, compare another build against them.

  record  --out ref.json      needs CHAT_LIVE_TESTS=1; temperature 0, thinking off
  compare --ref ref.json --out new.json   same prompts; reports first divergence per prompt
  agree   --ref ref.json                  teacher-forced top-1 share (prefill path; confounded by
                                          re-tokenization, kept only as a rough screen)
  diff    A.json B.json                   offline: first token divergence per prompt and whether it
                                          was a near-tie (top-1/top-2 margin) in A; mean |dlogprob| before it
A kernel that is not bit-exact may diverge late; divergence within the first few
tokens on most prompts means a numerical or routing bug, not rounding noise.
"""
import argparse
import json
import os
import sys
import time

import requests

PROMPTS = (
    'Explain how a hash map handles collisions and resizing. Be thorough.',
    'Write a Python function that merges overlapping intervals, with a docstring and tests.',
    '请用中文详细介绍长城的历史，包括修建原因、主要朝代和现状。',
)


def run(base, model, max_tokens):
    results = []
    for prompt in PROMPTS:
        payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                   'max_tokens': max_tokens, 'temperature': 0, 'logprobs': True, 'top_logprobs': 5,
                   'return_tokens_as_token_ids': True,
                   'chat_template_kwargs': {'enable_thinking': False}}
        started = time.perf_counter()
        response = requests.post(base + '/v1/chat/completions', json=payload, timeout=(15, 1800))
        response.raise_for_status()
        body = response.json()
        steps = [{'id': int(e['token'].split(':')[1]), 'logprob': e['logprob'],
                  'top': [[int(t['token'].split(':')[1]), t['logprob']] for t in e['top_logprobs']]}
                 for e in body['choices'][0]['logprobs']['content']]
        results.append({'prompt': prompt, 'text': body['choices'][0]['message']['content'], 'steps': steps,
                        'completion_tokens': body['usage']['completion_tokens'],
                        'elapsed_s': time.perf_counter() - started})
    return results


def tokenize(base, model, text):
    response = requests.post(base + '/tokenize', json={'model': model, 'prompt': text}, timeout=120)
    response.raise_for_status()
    return response.json()['tokens']


def agreement(base, model, record):
    """Teacher-force the reference reply and count positions where it is this build's argmax."""
    messages = [{'role': 'user', 'content': record['prompt']}]
    response = requests.post(base + '/tokenize', json={
        'model': model, 'messages': messages, 'add_generation_prompt': True,
        'chat_template_kwargs': {'enable_thinking': False}}, timeout=120)
    response.raise_for_status()
    head = response.json()['tokens']
    reply = requests.post(base + '/tokenize', json={'model': model, 'prompt': record['text'],
                                                     'add_special_tokens': False}, timeout=120)
    reply.raise_for_status()
    tail = reply.json()['tokens']
    response = requests.post(base + '/v1/completions', json={
        'model': model, 'prompt': head + tail, 'max_tokens': 1, 'temperature': 0,
        'prompt_logprobs': 1}, timeout=(15, 600))
    response.raise_for_status()
    scored = response.json()['choices'][0]['prompt_logprobs'][len(head):]
    hits = sum(1 for token, entry in zip(tail, scored) if entry and entry[str(token)]['rank'] == 1)
    return hits, len(tail)


def diff(path_a, path_b):
    for a, b in zip(json.load(open(path_a)), json.load(open(path_b))):
        sa, sb = a['steps'], b['steps']
        i = next((k for k, (x, y) in enumerate(zip(sa, sb)) if x['id'] != y['id']), None)
        prefix = sa[:i] if i is not None else sa
        dlp = [abs(x['logprob'] - y['logprob']) for x, y in zip(prefix, sb)]
        row = {'prompt': a['prompt'][:40], 'tokens': len(sa), 'first_divergence': i,
               'mean_abs_dlogprob_before': round(sum(dlp) / max(len(dlp), 1), 5)}
        if i is not None:
            top = sa[i]['top']
            row['margin_top1_top2_in_A'] = round(top[0][1] - top[1][1], 4)
            row['B_choice_rank_in_A'] = next((r + 1 for r, (t, _) in enumerate(top) if t == sb[i]['id']), None)
        print(json.dumps(row, ensure_ascii=False))


def main():
    if len(sys.argv) == 4 and sys.argv[1] == 'diff':
        return diff(sys.argv[2], sys.argv[3])
    if os.environ.get('CHAT_LIVE_TESTS') != '1':
        raise SystemExit('Set CHAT_LIVE_TESTS=1 explicitly for real inference')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('record', 'compare', 'agree'))
    parser.add_argument('--base', default='http://127.0.0.1:8000')
    parser.add_argument('--model', required=True)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--ref')
    parser.add_argument('--out')
    args = parser.parse_args()
    if args.action == 'agree':
        with open(args.ref) as stream:
            for record in json.load(stream):
                hits, total = agreement(args.base, args.model, record)
                print(json.dumps({'prompt': record['prompt'][:40], 'top1_agreement': round(hits / total, 4),
                                  'tokens': total}, ensure_ascii=False))
        return
    results = run(args.base, args.model, args.max_tokens)
    with open(args.out, 'w') as stream:
        json.dump(results, stream, ensure_ascii=False, indent=1)
    if args.action == 'compare':
        with open(args.ref) as stream:
            reference = json.load(stream)
        for ref, new in zip(reference, results):
            a, b = tokenize(args.base, args.model, ref['text']), tokenize(args.base, args.model, new['text'])
            same = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print(json.dumps({'prompt': ref['prompt'][:40], 'tokens': [len(a), len(b)],
                              'identical_prefix_tokens': same, 'identical': a == b}, ensure_ascii=False))
    for result in results:
        print(json.dumps({'prompt': result['prompt'][:40], 'completion_tokens': result['completion_tokens'],
                          'elapsed_s': round(result['elapsed_s'], 2)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
