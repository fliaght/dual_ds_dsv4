#!/usr/bin/env python3
"""Diagnostic only (V41_KV_AUDIT=1): log how the KV pool is split across cache groups.

Wraps vllm.v1.core.kv_cache_utils.update_kv_cache_capacity so the startup log shows,
per group, the spec type, layer count, block/page size, per-request bytes and blocks.
The reported "GPU KV cache size" is max_concurrency x max_model_len, so this is the
only way to see which group dominates the bytes per token. No behaviour changes.
"""
from pathlib import Path
import sys

MARKER = '# [v41-kv-audit]\n'
FOOTER = MARKER + '''_v41_update_kv_cache_capacity = update_kv_cache_capacity


def update_kv_cache_capacity(vllm_config, kv_cache_config):
    _v41_update_kv_cache_capacity(vllm_config, kv_cache_config)
    try:
        groups = kv_cache_config.kv_cache_groups
        logger.info("[v41-kv-audit] num_blocks=%d bytes_per_block=%d groups=%d max_model_len=%d",
                    kv_cache_config.num_blocks, _get_kv_cache_bytes_per_block(groups), len(groups),
                    vllm_config.model_config.max_model_len)
        for index, group in enumerate(groups):
            spec = group.kv_cache_spec
            inner = getattr(spec, "kv_cache_specs", None)
            sample = next(iter(inner.values())) if inner else spec
            need = spec.max_memory_usage_bytes(vllm_config)
            logger.info("[v41-kv-audit] group=%d type=%s/%s layers=%d block=%s page=%d "
                        "per_request_bytes=%d blocks=%d tokens_per_state=%s window=%s dtype=%s",
                        index, type(spec).__name__, type(sample).__name__, len(group.layer_names),
                        getattr(sample, "block_size", None), spec.page_size_bytes, need,
                        -(-need // spec.page_size_bytes), getattr(sample, "tokens_per_state", None),
                        getattr(sample, "sliding_window", None), getattr(sample, "dtype", None))
        for tensor in kv_cache_config.kv_cache_tensors:
            logger.info("[v41-kv-audit] tensor size=%d shared_by=%d", tensor.size,
                        len(getattr(tensor, "shared_by", None) or getattr(tensor, "layers", []) or []))
    except Exception as error:  # diagnostics must never break startup
        logger.warning("[v41-kv-audit] failed: %r", error)
'''


def patch(path):
    text = path.read_text()
    if MARKER in text:
        return 'already enabled'
    for anchor in ('def update_kv_cache_capacity(', 'def _get_kv_cache_bytes_per_block('):
        if anchor not in text:
            raise RuntimeError(f'kv_cache_utils changed ({anchor}); refusing the audit patch')
    path.write_text(text + '\n\n' + FOOTER)
    return 'enabled'


if __name__ == '__main__':
    print(f'[v41-kv-audit] {patch(Path(sys.argv[1]))}', flush=True)
