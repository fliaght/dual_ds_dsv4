#!/usr/bin/env python3
"""Opt the installed EXL3 overlay into the cooperative MoE decode kernel.

Runs inside the container at start, only when V41_COOP_MOE=1. Appends the same
footer upstream's prepare_profile.py writes, but only onto the exact stock overlay
it was validated against; the adapter itself pins the binary's SHA-256.
"""
import hashlib
from pathlib import Path
import sys

STOCK_SHA = 'ccdc69bfa04bff4870c3e555736a990fde6448ddb329441c4e0a27d6fc41078d'
MARKER = '# [v41-cooperative-moe]\n'


def footer(root):
    return (MARKER +
            'import runpy as _coop_runpy\nimport sys as _coop_sys\n'
            f'_coop_setup = _coop_runpy.run_path({str(Path(root) / "runtime.py")!r})\n'
            f'_coop_setup["install"](_coop_sys.modules[__name__], library_root={str(root)!r}, enabled=True)\n')


def patch(overlay, root):
    data = overlay.read_bytes()
    if MARKER.encode() in data:
        return 'already enabled'
    if hashlib.sha256(data).hexdigest() != STOCK_SHA:
        raise RuntimeError('EXL3 overlay is not the validated stock version; refusing cooperative MoE')
    for name in ('runtime.py', 'cooperative_moe.so'):
        if not (Path(root) / name).is_file():
            raise RuntimeError(f'cooperative MoE artifact missing: {name}')
    overlay.write_bytes(data + b'\n' + footer(root).encode())
    return 'enabled'


if __name__ == '__main__':
    print(f'[v41-coop-moe] {patch(Path(sys.argv[1]), sys.argv[2])}', flush=True)
