#!/usr/bin/env python3
"""Live serving checks used to evaluate TensorFold (any OpenAI-compatible server on the site works).

Every request goes through chat/llm_client.LLMClient (the website's client) or plain HTTP, strictly one at a time
(the website queue's concurrency), after both guards report available/free above --admission-mib/--admission-free-mib.

  CHAT_LIVE_TESTS=1 python bench/tf_eval.py web   --peer WORKER --guard-state ~/dsv4_logs/tf-guard.json --out DIR
  CHAT_LIVE_TESTS=1 python bench/tf_eval.py soak  ... --minutes 120 --huge 250000@30 415000@75
  CHAT_LIVE_TESTS=1 python bench/tf_eval.py queue ...   # upstream TensorFold #13: requests queued server-side

web:   thinking off/high/max, length cap, over-context 400, Stop during decode and at the first token after a long
       prefill, an API client dropping the socket mid-prefill; each Stop is followed at once by the next request and a
       50 ms /health poll records the overlap (inflight 2 = queued in the server).
soak:  mixed sequential traffic (short, code, thinking, growing multi-turn, needle documents, long output, Stops,
       dropped sockets, non-streamed) plus scheduled huge needle prompts; a 10 s monitor of /health and both guards.
queue: 1 and 2 requests queued behind a 4,096-token decode, 1 behind a 131K prefill. Can stop both ranks (#13).
Results are JSON lines in --out; nothing goes into the repository.
"""
import argparse
import http.client
import json
import os
from pathlib import Path
import random
import shlex
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'chat'))
from llm_client import LLMClient, LLMError  # noqa: E402

WORDS = ('amber basin cobalt drift ember fable grove harbor island jetty kelp lumen marsh north orbit pine '
         'quay ridge salt tide umber vale willow xenon yarrow zephyr').split()
SYS = {'role': 'system', 'content': "You are a helpful, precise assistant. Answer in the user's language."}
MATH = 'What is 17 * 23? Reply with just the number.'
ESSAY = 'Write a very long, detailed history of lighthouses, chapter by chapter.'
SHORT = ['Explain why the sky is blue in two paragraphs.', 'Give three tips for writing clear emails.',
         '用三句话介绍长城。', 'What is the difference between TCP and UDP? Be brief.',
         'Summarize the plot of Hamlet in five sentences.', '请解释什么是统一内存架构，100 字以内。',
         'List five prime numbers greater than 100.', 'Translate "knowledge is power" into French and German.']
CODE = ['Write a Python function that merges two sorted lists, with a docstring and two doctests.',
        'Implement an LRU cache class in Python with get / put in O(1). Code only.',
        'Write a bash script that rotates log files older than 7 days into a tar.gz archive.',
        'Write a C function that reverses a singly linked list in place.']
FOLLOW = ['Continue with one more detail.', 'Now summarize everything so far in one sentence.',
          'What would be a good follow-up question?', 'Give a concrete example.', 'Rewrite that more formally.']
SOAK_KINDS = [('short', 24), ('code', 10), ('thinking', 9), ('multiturn', 15), ('doc_qa', 13), ('long_out', 5),
              ('cancel', 12), ('disconnect', 4), ('nonstream', 8)]


class Run:
    def __init__(self, args):
        self.a = args
        self.out = args.out
        self.out.mkdir(parents=True, exist_ok=True)
        self.client = LLMClient(f'{args.base}/v1', read_timeout=1800)
        self.model = self.client.resolve_model()

    def log(self, name, rec):
        rec['time'] = time.strftime('%H:%M:%S')
        with (self.out / f'{name}.jsonl').open('a') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(json.dumps(rec, ensure_ascii=False)[:600], flush=True)

    def guards(self):
        """Both guard states; a missing, stale or tripped guard is an error."""
        remote = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', self.a.peer, 'cat',
                                 shlex.quote(str(self.a.guard_state))], check=True, capture_output=True, text=True,
                                timeout=10).stdout
        states = [json.loads(self.a.guard_state.read_text()), json.loads(remote)]
        for s in states:
            if s.get('reason') or time.time() - s['time'] > 10:
                raise RuntimeError(f'guard tripped or stale: {s}')
        return states

    def admit(self, wait_s=180):
        """Wait until both nodes are above the admission lines; give up (SystemExit) after wait_s."""
        end = time.time() + wait_s
        while True:
            states = self.guards()
            mem = [[s['available_mib'], s['free_mib']] for s in states]
            if all(a >= self.a.admission_mib and f >= self.a.admission_free_mib for a, f in mem):
                return mem
            if time.time() > end:
                raise SystemExit(f'admission: below {self.a.admission_mib}/{self.a.admission_free_mib} MiB: {mem}')
            time.sleep(5)

    def health(self):
        """TensorFold's /health JSON (inflight, errors, ...); vLLM answers 200 with no body -> {}."""
        resp = requests.get(f'{self.a.base}/health', timeout=5)
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            return {}

    def stream(self, messages, cancel_at=None, **kw):
        """One streamed reply as the website reads it; cancel_at=n closes it after n events, as Stop does.

        Streamlit's Stop is cooperative (raised at the next st.* call), so during a prefill the website cannot
        stop before the first token: a website Stop there is cancel_at=1."""
        r = {'content': '', 'reasoning_chars': 0, 'events': 0, 'finish': None, 'usage': None, 'ttft': None,
             'error': None, 'cancelled': False, 't0': time.time()}
        gen = self.client.stream_chat(messages, **kw)
        try:
            for ev in gen:
                r['events'] += 1
                if r['ttft'] is None and ev['type'] in ('content', 'reasoning'):
                    r['ttft'] = round(time.time() - r['t0'], 2)
                if ev['type'] == 'content':
                    r['content'] += ev['text']
                elif ev['type'] == 'reasoning':
                    r['reasoning_chars'] += len(ev['text'])
                elif ev['type'] == 'finish':
                    r['finish'] = ev['reason']
                elif ev['type'] == 'usage':
                    r['usage'] = ev['usage']
                if cancel_at is not None and r['events'] >= cancel_at:
                    r['cancelled'] = True
                    gen.close()
                    break
        except LLMError as e:
            r['error'] = str(e)[:300]
        r['t1'] = time.time()
        r['total_s'] = round(r['t1'] - r['t0'], 2)
        u = r['usage'] or {}
        if r['ttft'] is not None and u.get('completion_tokens', 0) > 1 and r['total_s'] > r['ttft']:
            r['decode_tps'] = round((u['completion_tokens'] - 1) / (r['total_s'] - r['ttft']), 2)
        return r

    def drop_socket(self, messages, after_s):
        """An API client that drops its socket mid-prefill (closed laptop, killed script)."""
        url = urlparse(self.a.base)
        conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=30)
        body = json.dumps({'model': self.model, 'messages': messages, 'max_tokens': 1024, 'stream': True,
                           'chat_template_kwargs': {'enable_thinking': False}})
        t0 = time.time()
        conn.request('POST', '/v1/chat/completions', body=body, headers={'Content-Type': 'application/json'})
        time.sleep(after_s)
        try:
            conn.sock.shutdown(socket.SHUT_RDWR)
        finally:
            conn.close()
        t1 = time.time()
        return {'t0': t0, 't1': t1, 'events': 0, 'ttft': None, 'total_s': round(t1 - t0, 2), 'cancelled': True,
                'error': None}

    def post(self, messages, max_tokens, temperature=0.0, ignore_eos=False):
        t0 = time.time()
        try:
            resp = requests.post(f'{self.a.base}/v1/chat/completions', timeout=(5, 1800), json={
                'model': self.model, 'messages': messages, 'max_tokens': max_tokens, 'temperature': temperature,
                'ignore_eos': ignore_eos, 'chat_template_kwargs': {'enable_thinking': False}})
            if resp.status_code != 200:
                return {'status': resp.status_code, 'error': resp.text[:300], 's': round(time.time() - t0, 1)}
            j = resp.json()
            c = j['choices'][0]
            return {'status': 200, 'content': c['message'].get('content') or '', 'finish': c.get('finish_reason'),
                    'usage': j.get('usage'), 'error': None, 's': round(time.time() - t0, 1)}
        except Exception as e:  # noqa: BLE001 - a dead server is a result here
            return {'status': None, 'error': repr(e)[:300], 's': round(time.time() - t0, 1)}


class HealthPoll(threading.Thread):
    def __init__(self, run, every=0.05):
        super().__init__(daemon=True)
        self.run_, self.every, self.samples, self.stop = run, every, [], False

    def run(self):
        while not self.stop:
            try:
                h = self.run_.health()
                self.samples.append((time.time(), h.get('inflight'), h.get('errors')))
            except Exception:  # noqa: BLE001
                self.samples.append((time.time(), None, None))
            time.sleep(self.every)


def needle_doc(rng, tokens):
    """~tokens of filler with one 'secret code' sentence; returns (prompt, code). Unique per call: no cache hits.

    This vocabulary tokenizes at ~1.42 tokens a word on V4.1 (measured: 187,500 words -> 266,756 tokens), so a
    target stays a few percent under the request (a 415K target must fit a 425,984 context)."""
    words = [rng.choice(WORDS) for _ in range(int(tokens * 0.70))]
    key, code = rng.choice(WORDS) + str(rng.randint(10, 99)), str(rng.randint(100000, 999999))
    words.insert(rng.randint(len(words) // 10, len(words) - 1), f'. The secret code for {key} is {code}. ')
    return ' '.join(words) + f'\nWhat is the secret code for {key}? Reply with just the number.', code


def cmd_web(run):
    rng = random.Random(20261015)
    for label, kw in (('thinking_off', {}), ('thinking_high', {'thinking': 'high'}),
                      ('thinking_max', {'thinking': 'max'})):
        mem = run.admit()
        r = run.stream([SYS, {'role': 'user', 'content': MATH}], max_tokens=8192 if kw else 256, temperature=0.0,
                       top_p=1.0, **kw)
        thought = r['reasoning_chars'] > 0
        run.log('web', {'case': label, 'ok': '391' in r['content'] and not r['error'] and thought == bool(kw),
                        'mem': mem, **{k: r[k] for k in ('content', 'reasoning_chars', 'finish', 'usage', 'error')}})
    run.admit()
    r = run.stream([SYS, {'role': 'user', 'content': ESSAY}], max_tokens=64, temperature=0.7, top_p=0.95)
    run.log('web', {'case': 'length_cap', 'ok': r['finish'] == 'length'
                    and (r['usage'] or {}).get('completion_tokens') == 64, 'usage': r['usage'], 'error': r['error']})
    run.admit()
    r = run.stream([SYS, {'role': 'user', 'content': needle_doc(rng, 30000)[0]}], max_tokens=400000)
    run.log('web', {'case': 'over_context', 'ok': bool(r['error']) and 'HTTP 400' in r['error'], 'error': r['error']})

    poll = HealthPoll(run)
    poll.start()

    def stop_then_ask(label, first):
        mem = run.admit()
        a1 = first()
        a2 = run.stream([SYS, {'role': 'user', 'content': MATH}], max_tokens=64, temperature=0.0, top_p=1.0)
        time.sleep(0.5)
        after = [s for s in poll.samples if a1['t1'] <= s[0] <= a2['t1'] + 0.5]
        queued = [round(s[0] - a1['t1'], 2) for s in after if (s[1] or 0) >= 2]
        busy = [round(s[0] - a1['t1'], 2) for s in after if (s[1] or 0) >= 1]
        run.log('web', {'case': label, 'ok': a1['cancelled'] and '391' in a2['content'] and not a2['error'],
                        'first': {k: a1[k] for k in ('events', 'ttft', 'total_s', 'error')},
                        'next': {k: a2[k] for k in ('content', 'ttft', 'total_s', 'error')},
                        'queued_s_after_stop': queued[-1:] or [0], 'busy_s_after_stop': busy[-1:] or [0],
                        'errors_counter': [after[0][2], after[-1][2]] if after else None, 'mem': mem})

    essay = [SYS, {'role': 'user', 'content': ESSAY}]
    for n in (5, 20, 60, 150, 400):
        stop_then_ask(f'stop_decode_after_{n}_events',
                      lambda n=n: run.stream(essay, cancel_at=n, max_tokens=2048, temperature=0.7, top_p=0.95))
    for size in run.a.prefill_tokens:
        doc = [SYS, {'role': 'user', 'content': needle_doc(rng, size)[0]}]
        stop_then_ask(f'web_stop_at_first_token_{size}', lambda doc=doc: run.stream(doc, cancel_at=1, max_tokens=256))
        doc = [SYS, {'role': 'user', 'content': needle_doc(rng, size)[0]}]
        stop_then_ask(f'api_drop_mid_prefill_{size}', lambda doc=doc: run.drop_socket(doc, 8.0))
    poll.stop = True
    h = run.health()
    run.log('web', {'case': 'final_health', 'ok': h.get('inflight', 0) == 0 and not h.get('fatal'),
                    'health': {k: h.get(k) for k in ('inflight', 'errors', 'requests', 'fatal')}})


class Monitor(threading.Thread):
    """/health and both guards every 10 s; sets fatal on a tripped/stale guard or two failed health reads."""
    def __init__(self, run):
        super().__init__(daemon=True)
        self.run_, self.stop, self.fatal, self.low, self.misses = run, False, None, {}, 0

    def sample(self):
        s = {'t': round(time.time(), 1)}
        try:
            h = self.run_.health()
            s['health'] = {k: h.get(k) for k in ('inflight', 'errors', 'requests', 'fatal', 'uptime_s')}
            self.misses = 0
            if h.get('fatal'):
                self.fatal = f'server fatal: {s["health"]}'
        except Exception as e:  # noqa: BLE001
            self.misses += 1
            s['health_error'] = repr(e)[:200]
            if self.misses >= 2:
                self.fatal = f'health unreachable: {s["health_error"]}'
        try:
            states = self.run_.guards()
            s['mem'] = [[g['available_mib'], g['free_mib']] for g in states]
            for node, (a, f) in zip(('head', 'worker'), s['mem']):
                lo = self.low.setdefault(node, [a, f])
                self.low[node] = [min(lo[0], a), min(lo[1], f)]
        except Exception as e:  # noqa: BLE001
            self.fatal = f'guard: {e!r}'[:300]
        with (self.run_.out / 'monitor.jsonl').open('a') as f:
            f.write(json.dumps(s) + '\n')
        return s

    def run(self):
        while not self.stop:
            self.sample()
            time.sleep(10)


def cmd_soak(run):
    a = run.a
    mon = Monitor(run)
    mon.start()
    rng = random.Random(a.seed)
    huge = sorted((int(m) * 60, int(n)) for n, m in (h.split('@') for h in a.huge))
    convo, t_start, n, stop_reason = [SYS], time.time(), 0, None
    kinds, weights = zip(*SOAK_KINDS)
    while time.time() - t_start < a.minutes * 60:
        if mon.fatal:
            stop_reason = mon.fatal
            break
        try:
            mem = run.admit()
        except SystemExit as e:
            stop_reason = str(e)
            break
        elapsed = time.time() - t_start
        if huge and elapsed >= huge[0][0]:
            kind, size = 'huge', huge.pop(0)[1]
        else:
            kind, size = rng.choices(kinds, weights)[0], None
        rec = {'n': n, 'kind': kind, 'start_min': round(elapsed / 60, 2), 'mem_before': mem}
        if kind == 'short':
            r = run.stream([SYS, {'role': 'user', 'content': rng.choice(SHORT)}], max_tokens=512)
            r['check'] = bool(r['content'].strip()) and r['finish'] in ('stop', 'length')
        elif kind == 'code':
            r = run.stream([SYS, {'role': 'user', 'content': rng.choice(CODE)}], max_tokens=1500,
                           temperature=rng.choice([0.0, 0.7]))
            r['check'] = bool(r['content'].strip())
        elif kind == 'thinking':
            r = run.stream([SYS, {'role': 'user', 'content': rng.choice(SHORT)}], max_tokens=4096, thinking='high')
            r['check'] = r['reasoning_chars'] > 0 and (bool(r['content'].strip()) or r['finish'] == 'length')
        elif kind == 'multiturn':
            convo.append({'role': 'user', 'content': rng.choice(FOLLOW if len(convo) > 1 else SHORT)})
            r = run.stream(convo, max_tokens=1024)
            convo.append({'role': 'assistant', 'content': r['content']})
            u = r['usage'] or {}
            r['turn'] = len(convo) // 2
            r['cached'] = (u.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
            # TensorFold snapshots a prompt (>= 64 tokens), never a reply: reuse starts at turn 3.
            r['check'] = bool(r['content'].strip()) and (r['turn'] < 3 or r['cached'] > 0)
            if (u.get('prompt_tokens') or 0) > 40000 or r['error']:
                convo = [SYS]
        elif kind in ('doc_qa', 'huge'):
            size = size or rng.choice([8192, 16384, 32768, 65536])
            text, code = needle_doc(rng, size)
            r = run.stream([SYS, {'role': 'user', 'content': text}], max_tokens=64, temperature=0.0, top_p=1.0)
            r['size'], r['check'] = size, code in r['content']
            r['cached'] = ((r['usage'] or {}).get('prompt_tokens_details') or {}).get('cached_tokens')
        elif kind == 'long_out':
            r = run.stream([SYS, {'role': 'user', 'content': ESSAY}], max_tokens=8192)
            r['check'] = r['finish'] in ('stop', 'length') and bool(r['content'].strip())
        elif kind == 'cancel':
            r = run.stream([SYS, {'role': 'user', 'content': rng.choice(SHORT + CODE)}], max_tokens=2048,
                           cancel_at=rng.randint(1, 200))
            r['check'] = r['cancelled'] or r['finish'] is not None
        elif kind == 'disconnect':
            r = run.drop_socket([SYS, {'role': 'user', 'content': needle_doc(rng, 32768)[0]}], rng.uniform(2, 10))
            r['check'] = True
        else:
            r = run.post([SYS, {'role': 'user', 'content': rng.choice(SHORT)}], 512, temperature=0.7)
            r['check'] = bool((r.get('content') or '').strip())
        if kind not in ('doc_qa', 'huge'):
            r.pop('content', None)
        rec.update({k: v for k, v in r.items() if k not in ('t0', 't1')})
        rec['ok'] = not r.get('error') and bool(r.get('check'))
        run.log('requests', rec)
        n += 1
    mon.stop = True
    time.sleep(1)
    final = mon.sample()
    recs = [json.loads(line) for line in (run.out / 'requests.jsonl').open()]
    by = {}
    for r in recs:
        b = by.setdefault(r['kind'], {'n': 0, 'ok': 0, 'errors': 0})
        b['n'] += 1
        b['ok'] += bool(r['ok'])
        b['errors'] += bool(r.get('error'))
    tps = [r['decode_tps'] for r in recs if r.get('decode_tps')
           and (r.get('usage') or {}).get('completion_tokens', 0) >= 256]
    summary = {'minutes': round((time.time() - t_start) / 60, 1), 'requests': len(recs), 'stop_reason': stop_reason,
               'by_kind': by, 'failed': [r['n'] for r in recs if not r['ok']], 'low_mem_mib': mon.low,
               'decode_tps_first5_last5': [tps[:5], tps[-5:]], 'final': final}
    (run.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0 if not stop_reason and not summary['failed'] else 1


def cmd_queue(run):
    def scenario(name, lead, lead_tokens, n_queued):
        mem = run.admit()
        res, threads = {}, []

        def go(key, *args, **kw):
            res[key] = run.post(*args, **kw)
        threads.append(threading.Thread(target=go, args=('lead', lead, lead_tokens), kwargs={
            'temperature': 0.7, 'ignore_eos': True}))
        threads[0].start()
        time.sleep(5)
        for i in range(n_queued):
            t = threading.Thread(target=go, args=(f'q{i}', [SYS, {'role': 'user', 'content': MATH}], 32))
            t.start()
            threads.append(t)
            time.sleep(1)
        time.sleep(2)
        inflight = run.health().get('inflight')
        for t in threads:
            t.join()
        try:
            run.guards()
            h = run.health()
            alive = True
        except Exception as e:  # noqa: BLE001
            h, alive = {'error': repr(e)[:200]}, False
        ok = alive and all(v.get('status') == 200 for v in res.values()) and all(
            '391' in res[f'q{i}'].get('content', '') for i in range(n_queued))
        for v in res.values():
            v['content'] = (v.get('content') or '')[:40]
        run.log('queue', {'scenario': name, 'ok': ok, 'alive': alive, 'inflight_while_queued': inflight,
                          'results': res, 'mem_before': mem,
                          'health_after': {k: h.get(k) for k in ('inflight', 'errors', 'fatal', 'error')}})
        return alive

    essay = [SYS, {'role': 'user', 'content': ESSAY}]
    if not scenario('one_queued_behind_4096_decode', essay, 4096, 1):
        return 2
    if not scenario('two_queued_behind_4096_decode', essay, 4096, 2):
        return 2
    doc = [SYS, {'role': 'user', 'content': needle_doc(random.Random(1316), 131072)[0]}]
    return 0 if scenario('one_queued_behind_131k_prefill', doc, 64, 1) else 2


def main(argv=None):
    if os.environ.get('CHAT_LIVE_TESTS') != '1':
        raise SystemExit('Set CHAT_LIVE_TESTS=1 explicitly for real inference')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('web', 'soak', 'queue'))
    p.add_argument('--base', default='http://127.0.0.1:8000')
    p.add_argument('--peer', required=True, help='worker ssh host (reads its guard state)')
    p.add_argument('--guard-state', type=Path, required=True, help='same path on both nodes')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--admission-mib', type=int, default=9216)
    p.add_argument('--admission-free-mib', type=int, default=5120)
    p.add_argument('--prefill-tokens', type=int, nargs='*', default=[65536, 131072], help='web: Stop/drop sizes')
    p.add_argument('--minutes', type=float, default=120, help='soak')
    p.add_argument('--huge', nargs='*', default=['250000@30', '415000@75'], help='soak: TOKENS@MINUTE')
    p.add_argument('--seed', type=int, default=1015)
    a = p.parse_args(argv)
    run = Run(a)
    return {'web': cmd_web, 'soak': cmd_soak, 'queue': cmd_queue}[a.command](run) or 0


if __name__ == '__main__':
    sys.exit(main())
