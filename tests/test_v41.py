"""Safety regressions use tiny files and fake commands, never live models."""
import hashlib
import ctypes
import json
import mmap
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import memory_guard
import v41_assets
import v41_load_reclaim
import v41_validate


class LoadReclaimTests(unittest.TestCase):
    def test_pageout_preserves_clean_and_private_modified_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tiny.safetensors'
            data = b'x' * (8 * 1024**2)
            path.write_bytes(data)
            with path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY) as mapping:
                pointer = ctypes.c_char.from_buffer(mapping)
                address = ctypes.addressof(pointer)
                del pointer
                self.assertEqual(mapping[:], data)
                mapping[0] = ord('y')  # COW data must not revert to the file.
                storage = SimpleNamespace(data_ptr=lambda: address, nbytes=lambda: len(mapping))
                tensor = SimpleNamespace(device=SimpleNamespace(type='cpu'),
                                         numel=lambda: len(mapping), untyped_storage=lambda: storage)
                reclaimer = v41_load_reclaim.MappedWeightReclaimer(threshold=1)
                reclaimer.consumed(tensor)
                self.assertGreater(reclaimer.calls, 0)
                self.assertEqual(mapping[0], ord('y'))
                self.assertEqual(mapping[1:], data[1:])
                self.assertEqual(path.read_bytes(), data)

    def test_wrapper_patch_is_idempotent_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrapper.py'
            path.write_text(v41_load_reclaim.ANCHOR)
            v41_load_reclaim.patch(path)
            first = path.read_text()
            v41_load_reclaim.patch(path)
            self.assertEqual(path.read_text(), first)
            path.write_text('unrecognized loader\n')
            with self.assertRaisesRegex(RuntimeError, 'refusing'):
                v41_load_reclaim.patch(path)


    def test_pinned_staging_pools_are_released(self):
        calls = []
        tensor = SimpleNamespace(numel=lambda: 1024, element_size=lambda: 2)
        pool = {'bf16': tensor}
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: calls.append('sync')),
                                     _C=SimpleNamespace(_host_emptyCache=lambda: calls.append('empty')))
        module = SimpleNamespace(_PIN_POOL=pool)
        saved = {name: sys.modules.get(name) for name in ('torch', v41_load_reclaim.PIN_POOLS[0][0])}
        sys.modules.update({'torch': fake_torch, v41_load_reclaim.PIN_POOLS[0][0]: module})
        try:
            self.assertEqual(v41_load_reclaim.release_pinned_staging(), 2048)
        finally:
            for name, value in saved.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
        self.assertEqual(pool, {})
        self.assertEqual(calls, ['sync', 'empty'])  # Synchronize before freeing pinned pages.


class GuardTests(unittest.TestCase):
    def test_live_client_rejects_low_free_or_stale_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'guard.json'
            state = dict(reason=None, time=time.time(), available_mib=12000, free_mib=6143)
            path.write_text(json.dumps(state))
            with self.assertRaisesRegex(RuntimeError, 'unsafe/stale'):
                v41_validate.guards(path, None, 10240)
            state.update(free_mib=8000, time=time.time() - 10)
            path.write_text(json.dumps(state))
            with self.assertRaisesRegex(RuntimeError, 'unsafe/stale'):
                v41_validate.guards(path, None, 10240)
            state['time'] = time.time()
            path.write_text(json.dumps(state))
            self.assertEqual(v41_validate.guards(path, None, 10240), [state])

    def test_available_is_not_memfree(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'meminfo'
            path.write_text('MemFree: 1024 kB\nMemAvailable: 8388608 kB\n')
            self.assertEqual(memory_guard.available_mib(path), 8192)

    def test_low_memory_kills_whole_local_rank(self):
        calls = []
        result = memory_guard.sample('test-rank', 6144, lambda: 6143,
                                     lambda args, **kwargs: calls.append(args))
        self.assertEqual(result, (6143, 'low_memory'))
        self.assertEqual(calls, [['docker', 'kill', 'test-rank']])
        memory_guard.sample('test-rank', 6144, lambda: 6144,
                            lambda args, **kwargs: calls.append(args))
        self.assertEqual(len(calls), 1)

    def test_unreadable_memory_fails_closed(self):
        def unreadable():
            raise OSError('proc unavailable')
        calls = []
        result = memory_guard.sample('test-rank', 6144, unreadable,
                                     lambda args, **kwargs: calls.append(args))
        self.assertEqual(result, (None, 'monitor_failure'))
        self.assertEqual(calls[0], ['docker', 'kill', 'test-rank'])

    def test_low_memfree_protects_driver_despite_reclaimable_cache(self):
        calls = []
        result = memory_guard.sample('test-rank', 8192, lambda: 12000,
                                     lambda args, **kwargs: calls.append(args),
                                     read_free=lambda: 1024)
        self.assertEqual(result, (12000, 'low_free_memory'))
        self.assertEqual(calls, [['docker', 'kill', 'test-rank']])

    def test_slow_cache_reclaim_does_not_delay_emergency_check(self):
        entered, release = threading.Event(), threading.Event()

        def slow_reclaim(directories):
            entered.set()
            release.wait(2)

        cache = memory_guard.CacheReclaimer([], release=slow_reclaim)
        cache.request()
        try:
            self.assertTrue(entered.wait(1))
            self.assertTrue(cache.running)
            calls = []
            result = memory_guard.sample('test-rank', 8192, lambda: 12000,
                                         lambda args, **kwargs: calls.append(args),
                                         read_free=lambda: 4095)
            self.assertEqual(result, (12000, 'low_free_memory'))
            self.assertEqual(calls, [['docker', 'kill', 'test-rank']])
        finally:
            release.set()
            cache.thread.join(1)


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.shard = self.root / 'model-00001.safetensors'
        name = 'layers.0.ffn.experts.0.weight'
        header = json.dumps({name: {'dtype': 'F16', 'shape': [2], 'data_offsets': [0, 4]}}).encode()
        self.shard.write_bytes(struct.pack('<Q', len(header)) + header + b'1234')
        index = self.root / 'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {name: self.shard.name}}))
        self.manifest = {'revision': 'pinned', 'files': [
            {'rfilename': p.name, 'size': p.stat().st_size,
             'lfs': {'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}}
            for p in (self.shard, index)]}
        (self.root / '.deployment-manifest.json').write_text(json.dumps(self.manifest))

    def test_complete_pinned_files_pass(self):
        self.assertEqual(v41_assets.verify(self.root, 'pinned', True)['tensors'], 1)
        with self.assertRaisesRegex(ValueError, 'revision'):
            v41_assets.verify(self.root, 'changed')

    def test_missing_truncated_or_corrupted_weights_fail(self):
        original = self.shard.read_bytes()
        self.shard.write_bytes(original[:-1])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            v41_assets.verify(self.root, 'pinned')
        self.shard.write_bytes(original[:-1] + b'x')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            v41_assets.verify(self.root, 'pinned', True)
        self.shard.unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            v41_assets.verify(self.root, 'pinned')


if __name__ == '__main__':
    unittest.main()
