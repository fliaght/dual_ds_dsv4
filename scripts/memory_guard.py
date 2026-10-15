#!/usr/bin/env python3
"""Local UMA guard; runs outside Docker and never relies on SSH for a kill."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time


def memory_stats(path=Path('/proc/meminfo')):
    values = dict(line.split(':', 1) for line in path.read_text().splitlines())
    return {key: int(values[key].split()[0]) // 1024 for key in ('MemAvailable', 'MemFree')}


def available_mib(path=Path('/proc/meminfo')):
    return memory_stats(path)['MemAvailable']


def release_shard_cache(directories):
    count = 0
    for directory in directories:
        for pattern in ('*.safetensors', 'engram-*.bin'):
            for path in directory.glob(pattern):
                with path.open('rb') as stream:
                    os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                count += 1
    return count


def sample(container, reserve_mib, read_memory=available_mib, run=subprocess.run,
           read_free=None, minimum_free_mib=4096):
    try:
        available = read_memory()
        reason = 'low_memory' if available < reserve_mib else None
        if not reason and read_free is not None and read_free() < minimum_free_mib:
            reason = 'low_free_memory'
    except (OSError, KeyError, ValueError):
        available, reason = None, 'monitor_failure'
    if reason:
        # Kill the entire rank, including driver allocations, before doing log I/O.
        run(['docker', 'kill', container], timeout=10, capture_output=True, check=False)
    return available, reason


class CacheReclaimer:
    """Slow file-cache eviction must never block the emergency sampling loop."""
    def __init__(self, directories, release=release_shard_cache):
        self.directories = directories
        self.release = release
        self.thread = None
        self.error = None
        self.count = 0
        self.started = 0

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def request(self):
        if self.running:
            return
        self.count += 1
        self.started = time.monotonic()

        def work():
            try:
                self.release(self.directories)
            except Exception as exc:
                self.error = str(exc)

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', required=True)
    parser.add_argument('--reserve-mib', type=int, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--peer', help='best-effort peer cleanup after the local emergency kill')
    parser.add_argument('--cache-dir', type=Path, action='append', default=[],
                        help='owned shard directory whose clean page cache may be released')
    args = parser.parse_args()
    if args.reserve_mib < 4096:
        parser.error('reserve must be at least 4096 MiB')
    low_water = None
    free_low_water = None
    cache = CacheReclaimer(args.cache_dir)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    last_report = 0
    while True:
        snapshot = {}

        def read_available():
            if cache.error is not None:
                raise OSError('cache reclaim failed: ' + cache.error)
            snapshot.update(memory_stats())
            snapshot['observed_free_mib'] = snapshot['MemFree']
            # Start reclaim early, off the sampling thread. A low-memory sample
            # always goes straight to the kill path, even if eviction is slow.
            if (snapshot['MemAvailable'] >= args.reserve_mib
                    and 4096 <= snapshot['MemFree'] < 16384 and args.cache_dir
                    and time.monotonic() - cache.started >= 2):
                cache.request()
            return snapshot['MemAvailable']

        available, reason = sample(args.container, args.reserve_mib, read_memory=read_available,
                                   read_free=lambda: snapshot['MemFree'])
        if available is not None:
            low_water = min(available, low_water) if low_water is not None else available
        free = snapshot.get('MemFree')
        if free is not None:
            observed_free = min(free, snapshot['observed_free_mib'])
            free_low_water = (min(observed_free, free_low_water)
                              if free_low_water is not None else observed_free)
        state = dict(pid=os.getpid(), time=time.time(), available_mib=available,
                     low_water_mib=low_water, free_mib=free, free_low_water_mib=free_low_water,
                     observed_free_mib=snapshot.get('observed_free_mib'),
                     cache_releases=cache.count, cache_reclaim_running=cache.running,
                     reason=reason, reserve_mib=args.reserve_mib, free_reserve_mib=4096)
        temporary = args.state.with_suffix('.tmp')
        temporary.write_text(json.dumps(state) + '\n')
        temporary.replace(args.state)
        if reason or time.monotonic() - last_report >= 10:
            print(json.dumps(state), flush=True)
            last_report = time.monotonic()
        if reason:
            if args.peer:
                try:
                    subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=3',
                                    '-o', 'StrictHostKeyChecking=accept-new',
                                    args.peer, 'docker kill ' + shlex.quote(args.container)],
                                   timeout=5, capture_output=True, check=False)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            return 1
        time.sleep(0.25)


if __name__ == '__main__':
    raise SystemExit(main())
