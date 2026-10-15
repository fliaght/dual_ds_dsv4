#!/usr/bin/env python3
"""Experimental (V41_KV_GROUP_FLOOR=1): let the KV allocator split cache groups to one layer.

With the SM12x 64-token block patch, layer 20 (compress ratio 1) forms its own group,
but the shared pool charges every block at the widest group page (138,240 B) while a
layer-20 block uses 46,080 B: ~44% of the pool can never be written. Dropping the
repeats-per-group floor makes the pages uniform (research-arch/kvcalc: 3,240 -> 1,800
B/token, 435K -> ~787K tokens at 1440 MiB) at the cost of ~3x more cache groups.
vLLM validates the resulting layout at startup and refuses an inexpressible one.
"""
from pathlib import Path
import sys

ANCHOR = '            lower_bound=min_repeats_per_group,\n'
PATCHED = '            lower_bound=1,  # [v41-kv-group-floor]\n'


def patch(path):
    text = path.read_text()
    if PATCHED in text:
        return 'already enabled'
    if text.count(ANCHOR) != 1:
        raise RuntimeError('kv_cache_utils grouping changed; refusing the floor patch')
    path.write_text(text.replace(ANCHOR, PATCHED))
    return 'enabled'


if __name__ == '__main__':
    print(f'[v41-kv-group-floor] {patch(Path(sys.argv[1]))}', flush=True)
