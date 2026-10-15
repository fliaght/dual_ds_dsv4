#!/usr/bin/env python3
"""Single-request gateway in front of TensorFold's Python server.

TensorFold's Python batcher can stop both ranks when a completion waits inside the server behind another one
(upstream #13, Docs/kedb/runtime.md). Every client therefore talks to this gateway, which keeps the server at one
request at a time:

- POST requests wait here, round-robin between client addresses, and go upstream one at a time. GET requests
  (/health, /v1/models, /metrics), /tokenize and GET /gateway/status are answered without waiting.
- A request goes upstream only when the server's /health reports nothing in flight (this also covers traffic that
  bypassed the gateway), and its turn ends only when the server has closed the connection, i.e. its handler has
  returned. A cancelled request is therefore completely gone before the next one starts.
- A client that leaves while waiting never reaches the server. One that leaves mid-request makes the gateway
  half-close the upstream connection: TensorFold's socket check reads end-of-file and cancels the request.

Standard library only. One JSON line per completion request in --log (addresses, paths, timings; never bodies).
Operations: Docs/runbooks/tensorfold.md.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
import json
import logging
import logging.handlers
import signal
import socket
import sys
import time
from dataclasses import dataclass
from http import HTTPStatus
from urllib.parse import urlsplit

HEAD_LIMIT = 64 * 1024
CHUNK = 64 * 1024
HOP_BY_HOP = {"connection", "keep-alive", "proxy-connection", "te", "trailer", "transfer-encoding", "upgrade",
              "expect", "content-length", "host"}
PASS_POST = {"/tokenize", "/v1/tokenize"}       # TensorFold tokenizes on the host: no engine, no lock


class Refused(Exception):
    """An answer from the gateway itself: HTTP status, OpenAI-style error code, message."""

    def __init__(self, status: int, code: str, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.status, self.code, self.retry_after = status, code, retry_after


class ClientGone(Exception):
    """The client closed its connection, it broke, or the client stopped reading."""


@dataclass
class Request:
    method: str
    target: str
    path: str
    headers: list[tuple[str, str]]
    body: bytes


class FairQueue:
    """One turn at a time. Waiters queue per client address; addresses take turns round-robin."""

    def __init__(self, limit: int):
        self.limit = limit
        self.busy = False
        self.waiting: collections.OrderedDict[str, collections.deque] = collections.OrderedDict()

    def __len__(self) -> int:
        return sum(len(q) for q in self.waiting.values())

    def by_client(self) -> dict[str, int]:
        return {client: len(q) for client, q in self.waiting.items()}

    async def acquire(self, client: str) -> None:
        if not self.busy:                      # nobody waits while the turn is free
            self.busy = True
            return
        if len(self) >= self.limit:
            raise Refused(503, "queue_full", f"{self.limit} requests are already waiting for the model", 30)
        turn = asyncio.get_running_loop().create_future()
        self.waiting.setdefault(client, collections.deque()).append(turn)
        try:
            await turn
        except asyncio.CancelledError:
            if turn.done() and not turn.cancelled():
                self.release()                 # granted just as we were cancelled: pass it on
            else:
                self._forget(client, turn)
            raise

    def _forget(self, client: str, turn: asyncio.Future) -> None:
        q = self.waiting.get(client)
        if q is None:
            return
        with contextlib.suppress(ValueError):
            q.remove(turn)
        if not q:
            del self.waiting[client]

    def release(self) -> None:
        while self.waiting:
            client, q = next(iter(self.waiting.items()))
            del self.waiting[client]
            turn = q.popleft()
            if q:
                self.waiting[client] = q       # this address goes to the back of the rotation
            if not turn.done():
                turn.set_result(None)          # busy stays True: the turn is handed over
                return
        self.busy = False


def error_response(exc: Refused) -> bytes:
    body = json.dumps({"error": {"message": str(exc), "type": "gateway_error", "code": exc.code}}).encode()
    head = [f"HTTP/1.1 {exc.status} {HTTPStatus(exc.status).phrase}", "Content-Type: application/json",
            f"Content-Length: {len(body)}", "Connection: close"]
    if exc.retry_after:
        head.append(f"Retry-After: {exc.retry_after}")
    return ("\r\n".join(head) + "\r\n\r\n").encode() + body


def json_response(payload: dict) -> bytes:
    body = json.dumps(payload).encode()
    return (f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n").encode() + body


async def read_chunked(reader: asyncio.StreamReader, max_body: int) -> bytes:
    body = bytearray()
    while True:
        line = await reader.readuntil(b"\r\n")
        try:
            size = int(line.split(b";", 1)[0].strip(), 16)
        except ValueError:
            raise Refused(400, "bad_request", "malformed chunk size") from None
        if size == 0:
            while await reader.readuntil(b"\r\n") != b"\r\n":      # trailers
                pass
            return bytes(body)
        if len(body) + size > max_body:
            raise Refused(413, "body_too_large", f"request body over {max_body >> 20} MiB")
        body += await reader.readexactly(size)
        if await reader.readexactly(2) != b"\r\n":
            raise Refused(400, "bad_request", "malformed chunk")


async def read_request(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, max_body: int,
                       head_timeout: float, body_timeout: float) -> Request | None:
    """The client's request, or None when it left (or stalled) before sending one."""

    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), head_timeout)
    except asyncio.IncompleteReadError as exc:
        if exc.partial.strip():
            raise Refused(400, "bad_request", "incomplete request head") from None
        return None
    except asyncio.LimitOverrunError:
        raise Refused(431, "header_too_large", f"request head over {HEAD_LIMIT >> 10} KiB") from None
    except TimeoutError:
        return None
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
        raise Refused(400, "bad_request", "malformed request line")
    method, target, _ = parts
    headers = []
    for line in lines[1:]:
        if not line:
            continue
        name, sep, value = line.partition(":")
        if not sep or not name or name != name.strip():
            raise Refused(400, "bad_request", "malformed header")
        headers.append((name, value.strip()))
    values = lambda key: [v for k, v in headers if k.lower() == key]      # noqa: E731
    chunked = ",".join(values("transfer-encoding")).strip().lower()
    lengths = set(values("content-length"))
    if chunked and lengths:
        raise Refused(400, "bad_request", "both Transfer-Encoding and Content-Length")
    if chunked and chunked != "chunked":
        raise Refused(400, "bad_request", f"unsupported Transfer-Encoding: {chunked}")
    if len(lengths) > 1 or (lengths and not next(iter(lengths)).isdigit()):
        raise Refused(400, "bad_request", "invalid Content-Length")
    length = int(next(iter(lengths))) if lengths else 0
    if length > max_body:
        raise Refused(413, "body_too_large", f"request body over {max_body >> 20} MiB")
    if (chunked or length) and "100-continue" in ",".join(values("expect")).lower():
        writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
    try:
        if chunked:
            body = await asyncio.wait_for(read_chunked(reader, max_body), body_timeout)
        else:
            body = await asyncio.wait_for(reader.readexactly(length), body_timeout) if length else b""
    except (asyncio.IncompleteReadError, TimeoutError):
        return None
    except asyncio.LimitOverrunError:
        raise Refused(400, "bad_request", "malformed chunked body") from None
    path = target.split("?", 1)[0].rstrip("/") or "/"
    return Request(method.upper(), target, path, headers, body)


def upstream_request(req: Request, netloc: str) -> bytes:
    """The request as the server gets it: one request a connection, the body re-framed by Content-Length."""

    listed = {n.strip().lower() for k, v in req.headers if k.lower() == "connection" for n in v.split(",")}
    lines = [f"{req.method} {req.target} HTTP/1.1", f"Host: {netloc}"]
    lines += [f"{k}: {v}" for k, v in req.headers if k.lower() not in HOP_BY_HOP and k.lower() not in listed]
    if req.body or req.method in ("POST", "PUT", "PATCH"):
        lines.append(f"Content-Length: {len(req.body)}")
    lines += ["Connection: close", "", ""]
    return "\r\n".join(lines).encode("latin-1") + req.body


def client_head(head: bytes, wait_s: float | None) -> tuple[int, bytes]:
    """The server's status and headers for the client: framing kept (the body is relayed as is), connection closed."""

    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ", 2)
    if len(parts) < 2 or not parts[0].startswith("HTTP/") or not parts[1].isdigit():
        raise ValueError(f"bad status line from the server: {lines[0][:80]!r}")
    kept = [line for line in lines[1:]
            if line and line.split(":", 1)[0].strip().lower() not in ("connection", "keep-alive", "proxy-connection")]
    if wait_s is not None:
        kept.append(f"X-Gateway-Wait: {wait_s:.3f}")
    return int(parts[1]), "\r\n".join([lines[0], *kept, "Connection: close", "", ""]).encode("latin-1")


async def until_gone(reader: asyncio.StreamReader) -> None:
    """Return when the client closes its side or the connection breaks; anything it sends meanwhile is ignored."""

    with contextlib.suppress(OSError):
        while await reader.read(CHUNK):
            pass


async def drain_to_eof(reader: asyncio.StreamReader) -> None:
    while await reader.read(CHUNK):
        pass


def keepalive(writer: asyncio.StreamWriter) -> None:
    """Notice a client machine that vanished without closing (no bytes flow while a request waits or prefills)."""

    sock = writer.get_extra_info("socket")
    if sock is None or sock.family not in (socket.AF_INET, socket.AF_INET6):
        return
    with contextlib.suppress(OSError, AttributeError):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 6)


class Gateway:
    def __init__(self, upstream: str, *, queue_limit: int = 32, queue_timeout: float = 3600.0,
                 idle_timeout: float = 3600.0, settle_timeout: float = 600.0, connect_timeout: float = 5.0,
                 head_timeout: float = 30.0, body_timeout: float = 300.0, max_body: int = 64 << 20,
                 poll: float = 0.2, log: logging.Logger | None = None):
        url = urlsplit(upstream)
        if url.scheme != "http" or not url.hostname:
            raise ValueError(f"upstream must be http://host:port, got {upstream!r}")
        self.upstream = upstream.rstrip("/")
        self.up_host, self.up_port, self.up_netloc = url.hostname, url.port or 80, url.netloc
        self.queue = FairQueue(queue_limit)
        self.queue_timeout, self.idle_timeout, self.settle_timeout = queue_timeout, idle_timeout, settle_timeout
        self.connect_timeout, self.head_timeout, self.body_timeout = connect_timeout, head_timeout, body_timeout
        self.max_body, self.poll = max_body, poll
        self.log = log or logging.getLogger("tf_gateway")
        self.counts: collections.Counter[str] = collections.Counter()
        self.current: dict | None = None
        self.started = time.time()
        self.server: asyncio.Server | None = None
        self.tasks: set[asyncio.Task] = set()

    # -- lifecycle -------------------------------------------------------------------------------------------------

    async def start(self, host: str, port: int) -> int:
        self.server = await asyncio.start_server(self.handle, host, port, limit=HEAD_LIMIT, backlog=256)
        return self.server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        """Stop accepting and drop every connection; an upstream request in flight is cancelled by the close."""

        if self.server is not None:
            self.server.close()
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def event(self, name: str, **fields) -> None:
        self.log.info(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": name, **fields},
                                 separators=(",", ":")))

    def status(self) -> dict:
        current = None
        if self.current:
            current = dict(self.current, elapsed_s=round(time.monotonic() - self.current["since"], 1))
            del current["since"]
        return {"ok": True, "upstream": self.upstream, "busy": self.queue.busy, "queued": len(self.queue),
                "queued_by_client": self.queue.by_client(), "current": current, "counts": dict(self.counts),
                "uptime_s": round(time.time() - self.started, 1)}

    # -- one connection, one request -------------------------------------------------------------------------------

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        self.tasks.add(task)
        keepalive(writer)
        client = str((writer.get_extra_info("peername") or ("?",))[0])
        try:
            await self.serve(reader, writer, client)
        except (ClientGone, OSError, asyncio.CancelledError):
            pass
        except Exception as exc:               # a gateway bug: keep serving, leave a trace
            self.event("error", client=client, error=f"{type(exc).__name__}: {exc}"[:300])
        finally:
            self.tasks.discard(task)
            writer.close()

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, client: str) -> None:
        t0 = time.monotonic()
        try:
            req = await read_request(reader, writer, self.max_body, self.head_timeout, self.body_timeout)
        except Refused as exc:
            writer.write(error_response(exc))
            self.counts[exc.code] += 1
            return
        if req is None:
            return
        if req.method == "GET" and req.path == "/gateway/status":
            writer.write(json_response(self.status()))
            return
        generate = req.method == "POST" and req.path not in PASS_POST
        rec: dict = {"kind": "generate" if generate else "pass", "client": client, "method": req.method,
                     "path": req.path}
        gone = asyncio.ensure_future(until_gone(reader))
        try:
            if generate:
                await self.take_turn(req, writer, gone, rec)
            else:
                await self.relay(req, writer, gone, rec, turn=False)
        except Refused as exc:
            rec.update(status=exc.status, outcome=exc.code)
            writer.write(error_response(exc))
        except ClientGone:
            rec.setdefault("outcome", "client_gone")
        finally:
            gone.cancel()
            rec["total_s"] = round(time.monotonic() - t0, 3)
            outcome = rec.setdefault("outcome", "cancelled")
            self.counts[outcome if generate else f"pass_{outcome}"] += 1
            if generate or outcome != "ok" or (rec.get("status") or 0) >= 500:
                self.event("request", **rec)

    async def take_turn(self, req: Request, writer: asyncio.StreamWriter, gone: asyncio.Future, rec: dict) -> None:
        client = rec["client"]
        rec["ahead"] = len(self.queue) + int(self.queue.busy)
        t0 = time.monotonic()
        deadline = t0 + self.queue_timeout
        acquire = asyncio.ensure_future(self.queue.acquire(client))
        try:
            done, _ = await asyncio.wait({acquire, gone}, timeout=self.queue_timeout,
                                         return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            acquire.cancel()
            raise
        if acquire not in done:
            acquire.cancel()
            try:
                await acquire
            except asyncio.CancelledError:
                pass
            else:
                self.queue.release()           # it was granted before the cancel landed
            rec["wait_s"] = round(time.monotonic() - t0, 3)
            if gone in done:
                raise ClientGone()
            raise Refused(503, "queue_timeout", "timed out waiting for the model", 30)
        acquire.result()                       # Refused when the queue is full
        try:
            self.current = {"client": client, "path": req.path, "phase": "checking", "since": time.monotonic()}
            await self.wait_idle(gone, deadline)
            rec["wait_s"] = round(time.monotonic() - t0, 3)
            await self.relay(req, writer, gone, rec, turn=True)
        finally:
            self.current = None
            self.queue.release()

    async def upstream_inflight(self) -> int | None:
        """The server's /health ``inflight``: None when it cannot say (no such field, or nothing listens), -1 when it
        does not answer in time (a hung server must not get another request)."""

        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(self.up_host, self.up_port),
                                                    self.connect_timeout)
        except TimeoutError:
            return -1
        except OSError:
            return None
        try:
            writer.write(f"GET /health HTTP/1.1\r\nHost: {self.up_netloc}\r\nConnection: close\r\n\r\n".encode())
            raw = await asyncio.wait_for(reader.read(-1), self.connect_timeout)
        except TimeoutError:
            return -1
        except OSError:
            return None
        finally:
            writer.close()
        try:
            value = json.loads(raw.partition(b"\r\n\r\n")[2]).get("inflight")
        except (ValueError, AttributeError):
            return None
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    async def wait_idle(self, gone: asyncio.Future, deadline: float) -> None:
        while True:
            inflight = await self.upstream_inflight()
            if not inflight:
                return
            self.current["phase"] = "server-busy" if inflight > 0 else "server-silent"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Refused(503, "server_busy", "the model server stayed busy with another request", 30)
            await asyncio.wait({gone}, timeout=min(self.poll, remaining))
            if gone.done():
                raise ClientGone()

    async def race(self, aw, gone: asyncio.Future, timeout: float):
        """``aw``'s result, unless the client leaves first (ClientGone) or ``timeout`` passes (TimeoutError)."""

        task = asyncio.ensure_future(aw)
        try:
            done, _ = await asyncio.wait({task, gone}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            task.cancel()
            raise
        if task in done:
            return task.result()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        raise ClientGone() if gone in done else TimeoutError()

    async def send(self, writer: asyncio.StreamWriter, data: bytes, gone: asyncio.Future) -> None:
        writer.write(data)
        try:
            await self.race(writer.drain(), gone, self.idle_timeout)
        except OSError:                        # reset, broken pipe, or a client that stopped reading
            raise ClientGone() from None

    async def relay(self, req: Request, writer: asyncio.StreamWriter, gone: asyncio.Future, rec: dict, *,
                    turn: bool) -> None:
        t0 = time.monotonic()
        try:
            up_r, up_w = await asyncio.wait_for(
                asyncio.open_connection(self.up_host, self.up_port, limit=HEAD_LIMIT), self.connect_timeout)
        except (OSError, TimeoutError) as exc:
            raise Refused(502, "upstream_unavailable",
                          f"the model server is not reachable ({exc or type(exc).__name__})", 30) from None
        if turn:
            self.current["phase"] = "running"
        sent = 0
        try:
            up_w.write(upstream_request(req, self.up_netloc))
            await self.race(up_w.drain(), gone, self.idle_timeout)
            head = await self.race(up_r.readuntil(b"\r\n\r\n"), gone, self.idle_timeout)
            rec["status"], head = client_head(head, rec.get("wait_s"))
            await self.send(writer, head, gone)
            sent = len(head)
            while data := await self.race(up_r.read(CHUNK), gone, self.idle_timeout):
                await self.send(writer, data, gone)
                sent += len(data)
            rec["outcome"] = "ok"
        except ClientGone:
            rec["outcome"] = "client_gone"
            await self.settle(up_r, up_w, rec, turn)
            raise
        except TimeoutError:
            rec["outcome"] = "upstream_timeout"
            await self.settle(up_r, up_w, rec, turn)
            if not sent:
                raise Refused(504, "upstream_timeout", "the model server sent nothing in time") from None
        except (OSError, ValueError, asyncio.IncompleteReadError, asyncio.LimitOverrunError) as exc:
            rec.update(outcome="upstream_error", error=f"{type(exc).__name__}: {exc}"[:200])
            if not sent:
                raise Refused(502, "upstream_error", "the model server closed the connection") from None
        finally:
            up_w.close()
            rec.update(run_s=round(time.monotonic() - t0, 3), bytes=sent)

    async def settle(self, up_r: asyncio.StreamReader, up_w: asyncio.StreamWriter, rec: dict, turn: bool) -> None:
        """Half-close the upstream connection; TensorFold reads end-of-file on its socket check (every 250 ms) and
        cancels. Holding the turn, wait until it closes its side too: its handler has returned."""

        if turn:
            self.current["phase"] = "cancelling"
        with contextlib.suppress(OSError):
            up_w.write_eof()
        if not turn:
            return
        t0 = time.monotonic()
        try:
            await asyncio.wait_for(drain_to_eof(up_r), self.settle_timeout)
        except TimeoutError:
            rec["settle_timeout"] = True       # the next turn still waits for /health to show it gone
        except (OSError, asyncio.IncompleteReadError):
            pass
        rec["settle_s"] = round(time.monotonic() - t0, 3)


async def run(args: argparse.Namespace, log: logging.Logger) -> None:
    gateway = Gateway(args.upstream, queue_limit=args.queue_limit, queue_timeout=args.queue_timeout,
                      idle_timeout=args.idle_timeout, settle_timeout=args.settle_timeout,
                      max_body=args.max_body_mib << 20, log=log)
    port = await gateway.start(args.host, args.port)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    gateway.event("start", host=args.host, port=port, upstream=gateway.upstream)
    print(f"tf_gateway listening on {args.host}:{port} -> {gateway.upstream}", flush=True)
    await stop.wait()
    await gateway.stop()
    gateway.event("stop", counts=dict(gateway.counts))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Single-request gateway in front of TensorFold's Python server.")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 opens the API to the LAN")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--upstream", default="http://127.0.0.1:8100", help="TensorFold, loopback only")
    p.add_argument("--queue-limit", type=int, default=32, help="waiting requests before 503 queue_full")
    p.add_argument("--queue-timeout", type=float, default=3600, help="seconds a request may wait for its turn")
    p.add_argument("--idle-timeout", type=float, default=3600,
                   help="seconds without bytes from the server (a non-streamed 64K reply sends none for ~30 min)")
    p.add_argument("--settle-timeout", type=float, default=600, help="seconds to wait for a cancelled request")
    p.add_argument("--max-body-mib", type=int, default=64)
    p.add_argument("--log", help="JSON lines, rotated at 20 MiB (default: stderr)")
    args = p.parse_args(argv)
    log = logging.getLogger("tf_gateway")
    handler = (logging.handlers.RotatingFileHandler(args.log, maxBytes=20 << 20, backupCount=2) if args.log
               else logging.StreamHandler(sys.stderr))
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    asyncio.run(run(args, log))


if __name__ == "__main__":
    main()
