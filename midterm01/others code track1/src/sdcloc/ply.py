"""讀 PLY（ASCII 或 binary_little_endian；x y z [r g b]）。回傳 xyz float64 (N,3) 與 rgb uint8 (N,3) 或 None。"""
from pathlib import Path

import numpy as np
import polars as pl

_NP = {'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8', 'uchar': 'u1', 'uint8': 'u1',
       'char': 'i1', 'int8': 'i1', 'ushort': '<u2', 'short': '<i2', 'uint': '<u4', 'int': '<i4'}


def read_ply(path):
    path = Path(path)
    with path.open('rb') as f:
        n = None
        props, types = [], []
        fmt = 'ascii'
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f'{path}: no end_header')
            s = line.decode().strip()
            if s.startswith('format'):
                fmt = s.split()[1]
            elif s.startswith('element vertex'):
                n = int(s.split()[2])
            elif s.startswith('property'):
                _, t, name = s.split()[:3]
                props.append(name); types.append(t)
            elif s == 'end_header':
                break
        if n is None or props[:3] != ['x', 'y', 'z']:
            raise ValueError(f'{path}: unexpected header {props}')
        if fmt == 'ascii':
            df = pl.read_csv(f, separator=' ', has_header=False, new_columns=props,
                             schema_overrides={c: pl.Float64 for c in ('x', 'y', 'z')})
            if df.height != n:
                raise ValueError(f'{path}: expected {n} vertices, got {df.height}')
            xyz = df.select(['x', 'y', 'z']).to_numpy()
            rgb = df.select(['red', 'green', 'blue']).to_numpy().astype(np.uint8) if 'red' in props else None
            return xyz, rgb
        if fmt != 'binary_little_endian':
            raise ValueError(f'{path}: unsupported format {fmt}')
        dt = np.dtype([(p, _NP[t]) for p, t in zip(props, types)])
        rec = np.fromfile(f, dtype=dt, count=n)
    if len(rec) != n:
        raise ValueError(f'{path}: expected {n} vertices, got {len(rec)}')
    xyz = np.column_stack([rec['x'], rec['y'], rec['z']]).astype(np.float64)
    rgb = np.column_stack([rec['red'], rec['green'], rec['blue']]).astype(np.uint8) if 'red' in props else None
    return xyz, rgb
