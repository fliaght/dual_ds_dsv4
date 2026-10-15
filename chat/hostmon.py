"""Live memory + GPU monitor for the DGX Spark nodes (the sidebar "Performance" panel).

GB10 memory is unified: the GPU allocates from the same LPDDR5X pool as the CPU,
and ``nvidia-smi`` reports ``memory.used`` as ``[N/A]``.  So memory comes from
``/proc/meminfo`` -- used = MemTotal - MemAvailable, the same "available" figure
the OOM watchdog trips on -- and the GPU processes' share of it from
``nvidia-smi --query-compute-apps``.

One long-lived probe per host streams a JSON line every interval: the local node
runs it as a subprocess, the others over ssh (``python3 -`` with the probe on
stdin, so nothing is installed on the workers).  The samplers are process-wide
(runtime.py), so the number of open browser tabs doesn't change the load on the
nodes.  A probe that dies (ssh drop, node reboot) is restarted with backoff.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from collections import deque
from typing import Optional

# Runs on every node with the system python3 (stdlib only).  argv[1] = interval s.
PROBE = r'''
import json, socket, subprocess, sys
iv = float(sys.argv[1])

def num(s):
    try:
        return float(s)
    except ValueError:          # "[N/A]" etc.
        return None

def meminfo():
    m = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            m[k] = int(v.split()[0]) * 1024
    return m

def gpu_proc_bytes():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return sum(int(x) for x in out.split() if x.isdigit()) * 1024 * 1024
    except Exception:
        return None

smi = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw,clocks.sm",
                        "--format=csv,noheader,nounits", "-lms", str(int(iv * 1000))],
                       stdout=subprocess.PIPE, text=True)
name, n, procs = socket.gethostname(), 0, None
try:
    for line in smi.stdout:
        f = [x.strip() for x in line.split(",")]
        if len(f) < 4:
            continue
        if n % 5 == 0:              # process allocations move slowly; skip the extra nvidia-smi most ticks
            procs = gpu_proc_bytes()
        n += 1
        m = meminfo()
        print(json.dumps({"hostname": name, "mem_total": m["MemTotal"], "mem_avail": m["MemAvailable"],
                          "gpu_proc_mem": procs, "gpu_util": num(f[0]), "gpu_temp": num(f[1]),
                          "gpu_power": num(f[2]), "gpu_clock": num(f[3])}), flush=True)
finally:
    smi.kill()
'''


def _local_addrs() -> set[str]:
    addrs = {"localhost", "127.0.0.1", "::1", socket.gethostname()}
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5).stdout
        addrs.update(out.split())
    except (OSError, subprocess.SubprocessError):
        pass
    return addrs


class HostMonitor:
    """Background samplers for a list of hosts; HEAD (rank 0) first, like cluster.conf NODES."""

    def __init__(self, hosts: list[str], *, ssh_key: str = "", interval_s: float = 2.0,
                 history_s: float = 300.0) -> None:
        self.interval_s = interval_s
        self.ssh_key = ssh_key
        local = _local_addrs()
        self._lock = threading.Lock()
        self._hosts = [{
            "host": h, "role": "head" if i == 0 else "worker", "local": h in local,
            "hostname": None, "latest": None, "rx": 0.0, "error": None,
            "history": deque(maxlen=int(history_s / interval_s) + 1),
        } for i, h in enumerate(hosts or ["localhost"])]
        self._started = False

    # ------------------------------------------------------------- lifecycle
    def start(self) -> "HostMonitor":
        with self._lock:
            if not self._started:
                self._started = True
                for h in self._hosts:
                    threading.Thread(target=self._run, args=(h,), daemon=True,
                                     name=f"hostmon-{h['host']}").start()
        return self

    def command(self, h: dict) -> list[str]:
        probe = ["python3", "-u", "-", str(self.interval_s)]
        if h["local"]:
            return probe
        ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "ServerAliveInterval=10",
               "-o", "ServerAliveCountMax=3", "-o", "StrictHostKeyChecking=accept-new"]
        if self.ssh_key and os.path.exists(os.path.expanduser(self.ssh_key)):
            ssh += ["-i", os.path.expanduser(self.ssh_key)]
        return ssh + [h["host"], " ".join(probe)]

    def _run(self, h: dict) -> None:
        backoff = self.interval_s
        while True:
            last_err = None
            try:
                proc = subprocess.Popen(self.command(h), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True)
                proc.stdin.write(PROBE)
                proc.stdin.close()
                for line in proc.stdout:
                    try:
                        s = json.loads(line)
                    except json.JSONDecodeError:
                        last_err = line.strip()[:200] or last_err      # ssh / python error text
                        continue
                    self.ingest(h, s)
                    backoff = self.interval_s
                proc.wait()
                err = last_err or f"probe exited ({proc.returncode})"
            except OSError as e:
                err = str(e)
            with self._lock:
                h["error"] = err
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    def ingest(self, h: dict, s: dict, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        total, avail = s.get("mem_total") or 0, s.get("mem_avail") or 0
        s["mem_used"] = max(total - avail, 0)
        s["mem_used_pct"] = 100.0 * s["mem_used"] / total if total else None
        with self._lock:
            h["latest"], h["rx"], h["error"] = s, now, None
            h["hostname"] = s.get("hostname") or h["hostname"]
            h["history"].append((now, s["mem_used_pct"], s.get("gpu_util")))

    # ---------------------------------------------------------------- reading
    def snapshot(self) -> list[dict]:
        """Copy of each host's state.  ``age_s`` = seconds since its last sample (None = never)."""
        now = time.time()
        with self._lock:
            return [{
                "host": h["host"], "role": h["role"], "hostname": h["hostname"],
                "latest": dict(h["latest"]) if h["latest"] else None,
                "age_s": (now - h["rx"]) if h["rx"] else None,
                "error": h["error"], "history": list(h["history"]),
            } for h in self._hosts]

    def is_stale(self, snap: dict) -> bool:
        return snap["age_s"] is None or snap["age_s"] > max(3 * self.interval_s, 10.0)
