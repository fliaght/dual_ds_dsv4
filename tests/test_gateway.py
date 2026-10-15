"""Single-request gateway (scripts/tf_gateway.py) against a fake TensorFold server; loopback only, no model."""
import asyncio
import contextlib
import json
import os
import random
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tf_gateway  # noqa: E402


def peer_closed(conn):
    """TensorFold's socket check (server/cancellation.py): readable and a peeked read returns nothing."""
    try:
        poller = select.poll()
        poller.register(conn, select.POLLIN | select.POLLPRI)
        if not poller.poll(0):
            return False
        return conn.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
    except BlockingIOError:
        return False
    except OSError:
        return True


class FakeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _json(self, payload, status=200):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        srv = self.server
        if self.path == "/health":
            with srv.lock:
                self._json({"ok": True, "inflight": srv.inflight + srv.extra_inflight})
        elif self.path == "/v1/models":
            self._json({"object": "list", "data": [{"id": "fake", "max_model_len": 4096}]})
        else:
            self._json({"error": "not found"}, 404)

    def gone_during(self, seconds):
        end = time.monotonic() + seconds
        while not peer_closed(self.connection):
            if time.monotonic() >= end:
                return False
            time.sleep(0.01)
        return True

    def do_POST(self):
        srv = self.server
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with srv.lock:
            srv.active += 1
            srv.max_active = max(srv.max_active, srv.active)
            srv.events.append(("start", body["id"]))
        outcome = "done"
        try:
            time.sleep(body.get("prep", 0))            # tokenizing: not in flight yet, no socket check
            if peer_closed(self.connection):
                outcome = "cancelled-prep"
                return
            with srv.lock:
                srv.inflight += 1
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                for i in range(body.get("chunks", 3)):
                    if self.gone_during(body.get("gap", 0.02)):
                        outcome = "cancelled"
                        break
                    try:
                        self.wfile.write(f"data: {json.dumps({'id': body['id'], 'i': i})}\n\n".encode())
                        self.wfile.flush()
                    except OSError:
                        outcome = "cancelled"
                        break
                else:
                    self.wfile.write(b"data: [DONE]\n\n")
            finally:
                with srv.lock:
                    srv.inflight -= 1
            if outcome == "cancelled":
                time.sleep(body.get("post", 0))        # after /health says idle, before the handler returns
        finally:
            self.close_connection = True
            with srv.lock:
                srv.active -= 1
                srv.events.append(("end", body["id"], outcome))


class FakeTF(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), FakeHandler)
        self.lock = threading.Lock()
        self.active = self.max_active = self.inflight = self.extra_inflight = 0
        self.events = []
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"

    def started(self):
        with self.lock:
            return [e[1] for e in self.events if e[0] == "start"]

    def ends(self):
        with self.lock:
            return {e[1]: e[2] for e in self.events if e[0] == "end"}

    def close(self):
        self.shutdown()
        self.server_close()


class GatewayThread:
    def __init__(self, upstream, **kw):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.gateway = tf_gateway.Gateway(upstream, poll=0.05, **kw)
        self.port = self.call(self.gateway.start("127.0.0.1", 0))

    def call(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(10)

    def status(self):
        async def get():
            return self.gateway.status()
        return self.call(get())

    def close(self):
        self.call(self.gateway.stop())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        self.loop.close()


def send(port, method, path, payload=None, source="127.0.0.1", extra_headers=""):
    sock = socket.create_connection(("127.0.0.1", port), timeout=15, source_address=(source, 0))
    body = json.dumps(payload).encode() if payload is not None else b""
    sock.sendall(f"{method} {path} HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\n"
                 f"Content-Length: {len(body)}\r\n{extra_headers}\r\n".encode() + body)
    return sock


def read_all(sock):
    data = b""
    while chunk := sock.recv(65536):
        data += chunk
    sock.close()
    head, _, body = data.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    headers = {k.lower(): v.strip() for k, _, v in (line.partition(":") for line in lines[1:])}
    return int(lines[0].split()[1]), headers, body


def read_until(sock, marker):
    data = b""
    while marker not in data:
        chunk = sock.recv(65536)
        if not chunk:
            raise AssertionError(f"closed before {marker!r}: {data!r}")
        data += chunk
    return data


def in_thread(fn, *args):
    out = {}
    thread = threading.Thread(target=lambda: out.setdefault("result", fn(*args)), daemon=True)
    thread.start()
    return thread, out


def wait_for(predicate, timeout=5.0):
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > end:
            raise AssertionError("condition not reached")
        time.sleep(0.01)


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.tf = FakeTF()
        self.addCleanup(self.tf.close)

    def gateway(self, upstream=None, **kw):
        gw = GatewayThread(upstream or self.tf.url, **kw)
        self.addCleanup(gw.close)
        return gw

    def test_one_request_reaches_the_server_at_a_time(self):
        gw = self.gateway()
        runs = [in_thread(lambda i=i: read_all(send(gw.port, "POST", "/v1/chat/completions",
                                                    {"id": f"r{i}", "chunks": 4, "gap": 0.03})))
                for i in range(4)]
        wait_for(lambda: self.tf.active == 1)
        t0 = time.monotonic()
        status, _, body = read_all(send(gw.port, "GET", "/v1/models"))     # not queued behind completions
        self.assertEqual(status, 200)
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertEqual(json.loads(body)["data"][0]["max_model_len"], 4096)
        for thread, out in runs:
            thread.join(10)
            status, headers, body = out["result"]
            self.assertEqual(status, 200)
            self.assertEqual(body.count(b"data: {"), 4)
            self.assertTrue(body.endswith(b"data: [DONE]\n\n"))
            self.assertEqual(headers["connection"], "close")
            self.assertIn("x-gateway-wait", headers)
        self.assertEqual(self.tf.max_active, 1)
        self.assertEqual(gw.status()["counts"]["ok"], 4)

    def test_cancel_mid_reply_holds_the_turn_until_the_server_has_finished(self):
        gw = self.gateway()
        a = send(gw.port, "POST", "/v1/chat/completions", {"id": "a", "chunks": 200, "gap": 0.02, "post": 0.4})
        read_until(a, b"data: {")
        b_run = in_thread(lambda: read_all(send(gw.port, "POST", "/v1/chat/completions", {"id": "b"})))
        wait_for(lambda: gw.status()["queued"] == 1)
        a.close()
        b_run[0].join(10)
        self.assertEqual(b_run[1]["result"][0], 200)
        self.assertEqual(self.tf.max_active, 1)      # b waited for a's handler, not only for /health
        self.assertEqual(self.tf.started(), ["a", "b"])
        self.assertEqual(self.tf.ends()["a"], "cancelled")

    def test_cancel_before_the_engine_starts_waits_for_the_handler(self):
        gw = self.gateway()
        a = send(gw.port, "POST", "/v1/chat/completions", {"id": "a", "prep": 0.6})
        wait_for(lambda: self.tf.active == 1)
        b_run = in_thread(lambda: read_all(send(gw.port, "POST", "/v1/chat/completions", {"id": "b"})))
        wait_for(lambda: gw.status()["queued"] == 1)
        a.close()                                     # /health says idle throughout: only the close tells
        b_run[0].join(10)
        self.assertEqual(b_run[1]["result"][0], 200)
        self.assertEqual(self.tf.max_active, 1)
        self.assertEqual(self.tf.ends()["a"], "cancelled-prep")

    def test_a_waiting_client_that_leaves_never_reaches_the_server(self):
        gw = self.gateway()
        a_run = in_thread(lambda: read_all(send(gw.port, "POST", "/v1/chat/completions",
                                                {"id": "a", "chunks": 60, "gap": 0.03})))
        wait_for(lambda: self.tf.active == 1)
        b = send(gw.port, "POST", "/v1/chat/completions", {"id": "b"})
        wait_for(lambda: gw.status()["queued"] == 1)
        b.close()
        wait_for(lambda: gw.status()["queued"] == 0, timeout=1.0)     # dropped at once, not at a's end
        self.assertEqual(self.tf.active, 1)
        c_run = in_thread(lambda: read_all(send(gw.port, "POST", "/v1/chat/completions", {"id": "c"})))
        for thread, out in (a_run, c_run):
            thread.join(10)
            self.assertEqual(out["result"][0], 200)
        self.assertEqual(self.tf.started(), ["a", "c"])
        self.assertEqual(gw.status()["counts"]["client_gone"], 1)

    def test_addresses_take_turns(self):
        gw = self.gateway()
        runs = [in_thread(lambda: read_all(send(gw.port, "POST", "/v1/chat/completions",
                                                {"id": "hold", "chunks": 10, "gap": 0.05})))]
        wait_for(lambda: self.tf.active == 1)
        for n, (ident, source) in enumerate([("a1", "127.0.0.1"), ("a2", "127.0.0.1"), ("a3", "127.0.0.1"),
                                             ("b1", "127.0.0.2")]):
            runs.append(in_thread(lambda i=ident, s=source: read_all(
                send(gw.port, "POST", "/v1/chat/completions", {"id": i}, source=s))))
            wait_for(lambda: gw.status()["queued"] == n + 1)
        for thread, _ in runs:
            thread.join(10)
        self.assertEqual(self.tf.started(), ["hold", "a1", "b1", "a2", "a3"])

    def test_waits_while_the_server_reports_other_work(self):
        gw = self.gateway()
        self.tf.extra_inflight = 1                    # a request that bypassed the gateway
        run = in_thread(lambda: read_all(send(gw.port, "POST", "/v1/chat/completions", {"id": "a"})))
        time.sleep(0.4)
        self.assertEqual(self.tf.started(), [])
        self.assertEqual(gw.status()["current"]["phase"], "server-busy")
        self.tf.extra_inflight = 0
        run[0].join(10)
        status, headers, _ = run[1]["result"]
        self.assertEqual(status, 200)
        self.assertGreaterEqual(float(headers["x-gateway-wait"]), 0.35)

    def test_mixed_clients_never_overlap_upstream(self):
        gw = self.gateway(queue_limit=64)
        rng = random.Random(7)

        def client(ident, source, mode, extra):
            sock = send(gw.port, "POST", "/v1/chat/completions", {"id": ident, **extra}, source=source)
            if mode < 0.15:                           # leaves while waiting (or just after starting)
                time.sleep(mode * 3)
                sock.close()
                return "left"
            if mode < 0.3:                            # Stop after the first chunk
                with contextlib.suppress(AssertionError):
                    read_until(sock, b"data: {")
                sock.close()
                return "cut"
            return read_all(sock)[0]

        runs = []
        for i in range(40):
            extra = {"chunks": rng.randint(1, 8), "gap": 0.01, "prep": rng.choice([0, 0, 0.05]), "post": 0.05}
            runs.append(in_thread(client, f"r{i}", f"127.0.0.{1 + i % 3}", rng.random(), extra))
            time.sleep(0.01)
        for thread, _ in runs:
            thread.join(30)
        wait_for(lambda: not gw.status()["busy"], 10)
        self.assertEqual(self.tf.max_active, 1)
        self.assertEqual({out["result"] for _, out in runs} - {"left", "cut"}, {200})
        counts = gw.status()["counts"]
        self.assertEqual(counts.get("ok", 0) + counts.get("client_gone", 0), 40)
        self.assertEqual(set(self.tf.started()), set(self.tf.ends()))     # every handler finished

    def test_refusals_are_openai_errors(self):
        gw = self.gateway(queue_limit=1)
        hold = in_thread(lambda: read_all(send(gw.port, "POST", "/v1/chat/completions",
                                               {"id": "hold", "chunks": 10, "gap": 0.05})))
        wait_for(lambda: self.tf.active == 1)
        queued = in_thread(lambda: read_all(send(gw.port, "POST", "/v1/chat/completions", {"id": "q"})))
        wait_for(lambda: gw.status()["queued"] == 1)
        status, headers, body = read_all(send(gw.port, "POST", "/v1/chat/completions", {"id": "x"}))
        self.assertEqual((status, headers["retry-after"]), (503, "30"))
        self.assertEqual(json.loads(body)["error"]["code"], "queue_full")
        for thread, out in (hold, queued):
            thread.join(10)
            self.assertEqual(out["result"][0], 200)
        down = self.gateway("http://127.0.0.1:9")    # nothing listens there
        for method, path in (("POST", "/v1/chat/completions"), ("GET", "/health")):
            status, _, body = read_all(send(down.port, method, path, {"id": "d"} if method == "POST" else None))
            self.assertEqual(status, 502)
            self.assertEqual(json.loads(body)["error"]["code"], "upstream_unavailable")

    def test_chunked_body_with_expect_continue(self):
        gw = self.gateway()
        sock = socket.create_connection(("127.0.0.1", gw.port), timeout=10)
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nTransfer-Encoding: chunked\r\n"
                     b"Expect: 100-continue\r\n\r\n")
        self.assertTrue(read_until(sock, b"\r\n\r\n").startswith(b"HTTP/1.1 100 Continue"))
        body = json.dumps({"id": "chunked", "chunks": 1}).encode()
        sock.sendall(b"%x\r\n%s\r\n0\r\n\r\n" % (len(body), body))
        status, _, reply = read_all(sock)
        self.assertEqual(status, 200)
        self.assertIn(b'"id": "chunked"', reply)
        status, _, _ = read_all(send(gw.port, "POST", "/v1/chat/completions", None,
                                     extra_headers="Transfer-Encoding: chunked\r\n"))
        self.assertEqual(status, 400)                 # both framings at once

    def test_sigterm_cancels_the_request_in_flight(self):
        proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "tf_gateway.py"), "--port", "0",
                                 "--upstream", self.tf.url], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        self.addCleanup(proc.stderr.close)
        self.addCleanup(proc.stdout.close)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        line = proc.stdout.readline()
        port = int(line.split("listening on 127.0.0.1:")[1].split()[0])
        sock = send(port, "POST", "/v1/chat/completions", {"id": "long", "chunks": 500, "gap": 0.02})
        read_until(sock, b"data: {")
        os.kill(proc.pid, signal.SIGTERM)
        self.assertEqual(proc.wait(10), 0)
        wait_for(lambda: self.tf.ends().get("long") == "cancelled")
        log = [json.loads(x) for x in proc.stderr.read().splitlines()]
        self.assertEqual([e["event"] for e in log], ["start", "request", "stop"])
        self.assertEqual(log[1]["outcome"], "cancelled")
        sock.close()


if __name__ == "__main__":
    unittest.main()
