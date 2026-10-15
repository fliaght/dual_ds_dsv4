"""Performance panel -- offline.  The model server is a local fake; the probe only reads
/proc/meminfo and nvidia-smi.  CHAT_MONITOR_TEST_HOSTS="<ip> ..." also probes those nodes over ssh."""
import json, os, subprocess, sys, tempfile, threading, time, unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hostmon import PROBE, HostMonitor

HERE = Path(__file__).resolve().parent


class IngestTests(unittest.TestCase):
    def test_used_is_total_minus_available(self):
        mon = HostMonitor(["localhost"], interval_s=2.0)
        h = mon._hosts[0]
        mon.ingest(h, {"hostname": "n1", "mem_total": 100 * 2**30, "mem_avail": 25 * 2**30, "gpu_util": 40.0})
        s = mon.snapshot()[0]
        self.assertEqual(s["latest"]["mem_used"], 75 * 2**30)
        self.assertAlmostEqual(s["latest"]["mem_used_pct"], 75.0)
        self.assertEqual(s["hostname"], "n1")
        self.assertEqual(s["history"][-1][1:], (75.0, 40.0))
        self.assertFalse(mon.is_stale(s))

    def test_stale_and_roles_and_commands(self):
        mon = HostMonitor(["localhost", "10.9.9.9"], ssh_key="/nonexistent", interval_s=2.0)
        snaps = mon.snapshot()
        self.assertEqual([s["role"] for s in snaps], ["head", "worker"])
        self.assertTrue(mon.is_stale(snaps[0]))                      # never sampled
        self.assertEqual(mon.command(mon._hosts[0])[:3], ["python3", "-u", "-"])
        remote = mon.command(mon._hosts[1])
        self.assertEqual(remote[0], "ssh")
        self.assertNotIn("-i", remote)                                # missing key file is skipped
        self.assertEqual(remote[-2:], ["10.9.9.9", "python3 -u - 2.0"])

    def test_history_is_bounded(self):
        mon = HostMonitor(["localhost"], interval_s=1.0, history_s=5.0)
        for i in range(20):
            mon.ingest(mon._hosts[0], {"mem_total": 10, "mem_avail": 5}, now=float(i))
        self.assertEqual(len(mon.snapshot()[0]["history"]), 6)


class ProbeTests(unittest.TestCase):
    def test_probe_emits_json_locally(self):
        p = subprocess.Popen(["python3", "-u", "-", "0.5"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        p.stdin.write(PROBE); p.stdin.close()
        try:
            s = json.loads(p.stdout.readline())
        finally:
            p.kill(); p.stdout.close(); p.wait()
        self.assertGreater(s["mem_total"], s["mem_avail"])
        self.assertIn("gpu_util", s)

    def test_sampler_threads(self):
        hosts = ["localhost", *os.environ.get("CHAT_MONITOR_TEST_HOSTS", "").split()]
        mon = HostMonitor(hosts, ssh_key=os.environ.get("CHAT_MONITOR_SSH_KEY", ""), interval_s=0.5).start()
        deadline = time.time() + 20
        while time.time() < deadline and any(s["latest"] is None for s in mon.snapshot()):
            time.sleep(0.2)
        for s in mon.snapshot():
            self.assertIsNotNone(s["latest"], f"{s['host']}: {s['error']}")
            self.assertIsNotNone(s["latest"]["mem_used_pct"])


class _FakeVLLM(BaseHTTPRequestHandler):
    """Streams ~3 s of tokens so the app's in-stream monitor refresh runs."""
    def log_message(self, *a): pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "fake", "max_model_len": 131072}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(body if self.path.endswith("/models") else b"{}")

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        for i in range(30):
            self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': f'w{i} '}}]})}\n\n".encode())
            self.wfile.flush(); time.sleep(0.1)
        self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


class AppPanelTests(unittest.TestCase):
    def test_panel_renders_and_survives_a_streamed_reply(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeVLLM)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        code = f"""
import os, sys, runpy
os.environ.update(CHAT_API_BASE="http://127.0.0.1:{srv.server_port}/v1", CHAT_LOG_DIR={tempfile.mkdtemp()!r},
                  CHAT_MONITOR_HOSTS="localhost", CHAT_MONITOR_INTERVAL_S="0.5")
sys.path.insert(0, {str(HERE.parent)!r})
from streamlit.testing.v1 import AppTest
at = AppTest.from_file({str(HERE.parent / 'app.py')!r}, default_timeout=60); at.run()
import time; time.sleep(1.5); at.run()
assert not at.exception, at.exception
assert any(e.label == "Performance" for e in at.expander), "no Performance panel"
txt = " ".join(p.proto.text for p in at.get("progress"))
assert "Memory" in txt and "GPU" in txt, txt
at.chat_input[0].set_value("hi").run()
assert not at.exception, at.exception
print("OK", txt)
"""
        try:   # own process: app.py's module-level singletons must see this env
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertIn("OK", out.stdout, out.stdout + out.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
