#!/usr/bin/env python3
"""Pinned HF manifests, bounded-memory verification and conservative TP=2 budget."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import urllib.request

ENGRAM_FILES = {'config.json', 'model.safetensors.index.json',
                'model-00047-of-00048.safetensors', 'model-00048-of-00048.safetensors'}


def manifest(repo, revision, engram):
    url = f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'
    with urllib.request.urlopen(url, timeout=60) as response:
        data = json.load(response)
    if data['sha'] != revision:
        raise ValueError('Hub revision mismatch')
    files = [f for f in data['siblings'] if not engram or f['rfilename'] in ENGRAM_FILES]
    if engram and {f['rfilename'] for f in files} != ENGRAM_FILES:
        raise ValueError('Engram manifest incomplete')
    return dict(repo=repo, revision=revision, files=files)


def headers(path):
    with path.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if not 0 < length < 64 * 1024**2:
            raise ValueError(f'invalid safetensors header: {path.name}')
        header = json.loads(stream.read(length))
    entries = {name: value for name, value in header.items() if name != '__metadata__'}
    size = path.stat().st_size
    for value in entries.values():
        start, end = value['data_offsets']
        if not 0 <= start <= end <= size - 8 - length:
            raise ValueError(f'invalid tensor offsets: {path.name}')
    return entries


def verify(directory, revision, sha256=False, engram=False):
    data = json.loads((directory / '.deployment-manifest.json').read_text())
    if data['revision'] != revision:
        raise ValueError('local manifest revision mismatch')
    tensors = {}
    for entry in data['files']:
        path = directory / entry['rfilename']
        if path.parent != directory or not path.is_file() or path.stat().st_size != entry['size']:
            raise ValueError(f'missing/incomplete file: {path.name}')
        if sha256 and 'lfs' in entry:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                while chunk := stream.read(8 * 1024**2):
                    digest.update(chunk)
            if digest.hexdigest() != entry['lfs']['sha256']:
                raise ValueError(f'SHA256 mismatch: {path.name}')
        if path.suffix == '.safetensors':
            for name, value in headers(path).items():
                if name in tensors:
                    raise ValueError(f'duplicate tensor: {name}')
                tensors[name] = (path.name, value)
            # Release our clean shard pages before GPU loading on shared UMA.
            # This is targeted fadvise, not a privileged global drop_caches.
            with path.open('rb') as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    index = json.loads((directory / 'model.safetensors.index.json').read_text())['weight_map']
    expected = {k: v for k, v in index.items() if not engram or v in ENGRAM_FILES}
    if {name: item[0] for name, item in tensors.items()} != expected:
        raise ValueError('safetensors/index tensor map mismatch')
    if engram:
        return dict(files=len(data['files']), tensors=len(tensors), revision=revision)
    # Only indisputably TP-sharded weights are divided. All attention, Engram,
    # vision, router and auxiliary tensors are counted in full on BOTH nodes.
    # This deliberately overestimates the resident weights; workspaces are extra.
    rank_bytes = 0
    for name, (_, value) in tensors.items():
        start, end = value['data_offsets']
        sharded = ('.ffn.experts.' in name or '.ffn.shared_experts.' in name
                   or name.startswith(('embed.', 'head.')))
        rank_bytes += (end - start + 1) // 2 if sharded else end - start
    return dict(files=len(data['files']), tensors=len(tensors), revision=revision,
                conservative_rank_mib=(rank_bytes + 1024**2 - 1) // 1024**2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['manifest', 'verify'])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--repo')
    parser.add_argument('--revision', required=True)
    parser.add_argument('--engram', action='store_true')
    parser.add_argument('--sha256', action='store_true')
    args = parser.parse_args()
    if args.action == 'manifest':
        data = manifest(args.repo, args.revision, args.engram)
        args.directory.mkdir(parents=True, exist_ok=True)
        (args.directory / '.deployment-manifest.json').write_text(json.dumps(data, indent=2) + '\n')
        print(json.dumps(dict(files=len(data['files']), bytes=sum(f['size'] for f in data['files']))))
    else:
        print(json.dumps(verify(args.directory, args.revision, args.sha256, args.engram)))


if __name__ == '__main__':
    main()
