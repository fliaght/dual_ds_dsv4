"""A stand-in for the model server (offline tests): /health, /v1/models, /tokenize and streamed chat completions.

Replies are numbered, so a regenerated reply differs from the one it replaces: ``Reply 3 to: <question>``.
With thinking on, a short reasoning text streams first.  ``delay_s`` paces the stream (Stop tests).
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"


class FakeOpenAI:
    def __init__(self, *, max_model_len: int = 425984, delay_s: float = 0.0) -> None:
        self.max_model_len = max_model_len
        self.delay_s = delay_s
        self.requests: list[tuple[str, dict]] = []
        self.replies = 0
        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._srv.server_address[1]}/v1"

    def start(self) -> "FakeOpenAI":
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()

    def chats(self) -> list[dict]:
        return [b for p, b in self.requests if p.endswith("/chat/completions")]

    def _handler(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, obj):
                b = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path.endswith("/models"):
                    return self._json({"data": [{"id": MODEL, "max_model_len": fake.max_model_len}]})
                self._json({"ok": True})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append((self.path, body))
                if self.path.endswith("/tokenize"):
                    text = body.get("prompt") or "".join(m["content"] for m in body.get("messages", []))
                    return self._json({"count": 10 + len(text) // 4})
                fake.replies += 1
                question = body["messages"][-1]["content"]
                think = (body.get("chat_template_kwargs") or {}).get("enable_thinking")
                answer = f"Reply {fake.replies} to: {question}"
                if "markdown" in question:
                    answer += "\n\n**Bold point.** A list:\n\n- first item\n- second item\n\n> A quoted line."
                chunks = ([{"reasoning_content": "Let me think about it."}] if think else [])
                chunks += [{"content": w} for w in answer.replace(" ", "\x00 ").split("\x00")]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    for d in chunks:
                        self.wfile.write(f"data: {json.dumps({'choices': [{'delta': d, 'finish_reason': None}]})}\n\n".encode())
                        self.wfile.flush()
                        time.sleep(fake.delay_s)
                    done = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
                    usage = {"choices": [], "usage": {"prompt_tokens": 30 + len(question) // 4,
                                                      "completion_tokens": len(chunks)}}
                    self.wfile.write(f"data: {json.dumps(done)}\n\ndata: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True

        return H
