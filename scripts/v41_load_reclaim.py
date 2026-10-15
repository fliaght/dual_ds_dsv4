#!/usr/bin/env python3
"""Bound clean mmap residency while the V4.1 wrapper consumes sorted weights.

MADV_PAGEOUT preserves contents, including private dirty pages. Never use
MADV_DONTNEED here: it could discard an in-place modification to a CPU tensor.
This is local to weight loading; the independent host guard remains required.
"""
import bisect
import ctypes
import os
from pathlib import Path
import sys


class MappedWeightReclaimer:
    def __init__(self, threshold=64 * 1024**2):
        self.threshold = threshold
        self.page = os.sysconf('SC_PAGESIZE')
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.libc.madvise.restype = ctypes.c_int
        self.mappings = []
        for line in Path('/proc/self/maps').read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].endswith('.safetensors'):
                lo, hi = (int(x, 16) for x in fields[0].split('-'))
                self.mappings.append((lo, hi))
        self.starts = [lo for lo, _ in self.mappings]
        self.pending = []  # Keep mappings alive until the syscall completes.
        self.ranges = {}
        self.bytes = 0
        self.calls = 0

    def consumed(self, tensor):
        if tensor.device.type != 'cpu' or tensor.numel() == 0:
            return
        storage = tensor.untyped_storage()
        ptr, size = storage.data_ptr(), storage.nbytes()
        index = bisect.bisect_right(self.starts, ptr) - 1
        if index < 0:
            return
        lo, hi = self.mappings[index]
        if not lo <= ptr < hi:
            return
        if ptr + size > hi:
            raise RuntimeError('weight storage crosses its file mapping')
        start = ptr // self.page * self.page
        end = (ptr + size + self.page - 1) // self.page * self.page
        previous = self.ranges.get(index, (start, end))
        self.ranges[index] = (min(start, previous[0]), max(end, previous[1]))
        self.pending.append(tensor)
        self.bytes += size
        if self.bytes >= self.threshold:
            self.flush()

    def flush(self):
        for start, end in self.ranges.values():
            # Linux MADV_PAGEOUT=21: reclaim clean pages, retain dirty COW data.
            if self.libc.madvise(start, end - start, 21) != 0:
                code = ctypes.get_errno()
                raise OSError(code, os.strerror(code))
            self.calls += 1
        self.ranges.clear()
        self.pending.clear()
        self.bytes = 0


# Pinned host staging that the loaders keep for H2D copies (module, dict name).
# Pinned pages are UMA the GPU cannot use; after load they are dead weight.
PIN_POOLS = (('vllm.model_executor.layers.quantization.exl3', '_PIN_POOL'),
             ('vllm.model_executor.model_loader.weight_utils', '_DSV41_PIN_POOL'))


def release_pinned_staging():
    """Drop the staging pools and return cached pinned blocks to the OS."""
    import torch
    freed = 0
    for module, name in PIN_POOLS:
        pool = getattr(sys.modules.get(module), name, None)
        if pool:
            freed += sum(t.numel() * t.element_size() for t in pool.values())
            pool.clear()  # A later copy re-allocates only what it needs.
    torch.cuda.synchronize()  # No DMA may still read a released buffer.
    empty = getattr(torch._C, '_host_emptyCache', None)
    if empty is not None:
        empty()
    print(f'[v41-load-reclaim] released {freed / 2**20:.0f} MiB pinned staging'
          f' (host cache emptied: {empty is not None})', flush=True)
    return freed


def bounded_weights(mapped):
    """Preserve sorted order and drop each consumed tensor's list reference."""
    # Smaller steps flatten the load-time MemFree dip (pages copied to the GPU but not yet
    # paged out); 2026-10-15 lows at 64 MiB reached 4,159 MiB on the head.
    reclaimer = MappedWeightReclaimer(int(os.environ.get('DSV41_LOAD_RECLAIM_MIB', '64')) * 1024**2)
    mapped.reverse()
    try:
        while mapped:
            item = mapped.pop()
            yield item
            reclaimer.consumed(item[1])
            del item
        reclaimer.flush()
        print(f'[v41-load-reclaim] completed {reclaimer.calls} mapped-page reclaims', flush=True)
        release_pinned_staging()
    finally:
        # Do not perform syscalls after an exception: release retained references.
        reclaimer.pending.clear()
        mapped.clear()


ANCHOR = '        mapped = sorted(self.hf_to_vllm_mapper.apply(weights), key=lambda x: x[0])\n'
INSERT = ('        # [v41-bounded-weight-mappings]\n'
          '        from v41_load_reclaim import bounded_weights\n'
          '        mapped = bounded_weights(mapped)\n')


def patch(path):
    text = path.read_text()
    if INSERT in text:
        return
    if text.count(ANCHOR) != 1:
        raise RuntimeError('V4.1 wrapper loader changed; refusing an unverified patch')
    path.write_text(text.replace(ANCHOR, ANCHOR + INSERT))


if __name__ == '__main__':
    patch(Path(sys.argv[1]))
