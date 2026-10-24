import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from plyfile import PlyData


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def frames(directory, suffix='.ply', expected=3142):
    rows = []
    for p in Path(directory).glob('*'+suffix):
        m = re.fullmatch(r'(\d+)_ts_(\d+)', p.stem)
        if not m: raise ValueError(f'Invalid frame name: {p.name}')
        rows.append(dict(original_idx=int(m[1]), timestamp_ns=int(m[2]), path=p))
    rows.sort(key=lambda r: r['timestamp_ns'])
    if len(rows) != expected or [r['original_idx'] for r in rows] != list(range(expected)):
        raise ValueError(f'Expected contiguous 0..{expected-1} frames, found {len(rows)}')
    if any(b['timestamp_ns'] <= a['timestamp_ns'] for a,b in zip(rows, rows[1:])):
        raise ValueError('Duplicate or decreasing frame timestamp')
    return rows


def load_ply(path):
    v = PlyData.read(str(path))['vertex']
    xyz = np.column_stack([v[a] for a in ('x','y','z')]).astype(np.float64)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    # A readable but sparse frame is a registration rejection, not an I/O
    # failure: the runner must still emit its prior for this timestamp.
    return xyz


def load_map(path, leaf):
    import pandas as pd
    import cupy as cp
    from .gpu import voxel_mean
    header = []; offset = 0
    with Path(path).open('rb') as f:
        while True:
            line = f.readline().decode('ascii').strip(); header.append(line); offset += 1
            if line.startswith('DATA '): break
            if offset > 100: raise ValueError('Invalid PCD header')
    fields = next(s.split()[1:] for s in header if s.startswith('FIELDS '))
    if header[-1] != 'DATA ascii': raise ValueError('This streaming loader requires ASCII PCD')
    # Explicitly select XYZ; intensity and timestamp-related map fields never enter localization.
    cols = [fields.index(a) for a in ('x', 'y', 'z')]
    pieces, weights = [], []; read_count = 0
    for chunk in pd.read_csv(path, sep=r'\s+', skiprows=offset, header=None,
                             usecols=cols, dtype=np.float32, chunksize=1_000_000):
        p = chunk.to_numpy(); p = p[np.isfinite(p).all(axis=1)]; read_count += len(p)
        mean, count = voxel_mean(p, leaf)
        pieces.append(cp.asnumpy(mean)); weights.append(cp.asnumpy(count))
        if len(pieces) >= 4:
            mean, count = voxel_mean(np.concatenate(pieces), leaf, np.concatenate(weights))
            pieces = [cp.asnumpy(mean)]; weights = [cp.asnumpy(count)]
    if not pieces: raise ValueError('Empty map')
    mean, _ = voxel_mean(np.concatenate(pieces), leaf, np.concatenate(weights))
    return mean, read_count
