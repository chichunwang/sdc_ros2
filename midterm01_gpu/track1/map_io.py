"""PCD binary/ASCII 的 XYZ 串流讀取；時間與 intensity 欄不進定位。"""
from pathlib import Path
import numpy as np


def pcd_header(path):
    header = {}
    with Path(path).open('rb') as f:
        for _ in range(100):
            line = f.readline().decode('ascii').strip()
            if line and not line.startswith('#'):
                k, *v = line.split(); header[k] = v
                if k == 'DATA':
                    return header, f.tell()
    raise ValueError('Invalid PCD header')


def xyz_chunks(path, chunk_size=1_000_000):
    header, offset = pcd_header(path)
    fields = header['FIELDS']
    if not all(c in fields for c in ('x', 'y', 'z')):
        raise ValueError('Map requires XYZ')
    mode = header['DATA'][0]
    if mode == 'binary':
        types = {('F', 4): '<f4', ('F', 8): '<f8', ('U', 4): '<u4',
                 ('I', 4): '<i4', ('U', 1): 'u1', ('I', 1): 'i1',
                 ('U', 2): '<u2', ('I', 2): '<i2', ('U', 8): '<u8', ('I', 8): '<i8'}
        count = list(map(int, header.get('COUNT', ['1']*len(fields))))
        specs = []
        for name, typ, size, n in zip(fields, header['TYPE'], map(int, header['SIZE']), count):
            dtype = types[(typ, size)]
            specs.append((name, dtype) if n == 1 else (name, dtype, (n,)))
        dtype = np.dtype(specs)
        n = int(header['POINTS'][0])
        if Path(path).stat().st_size != offset+n*dtype.itemsize:
            raise ValueError('PCD binary payload size mismatch')
        arr = np.memmap(path, mode='r', offset=offset, dtype=dtype, shape=(n,))
        for start in range(0, n, chunk_size):
            a = arr[start:start+chunk_size]
            xyz = np.column_stack([a[c] for c in ('x', 'y', 'z')]).astype(np.float64)
            yield xyz[np.isfinite(xyz).all(axis=1)]
    elif mode == 'ascii':
        import pandas as pd
        cols = [fields.index(c) for c in ('x', 'y', 'z')]
        with Path(path).open('rb') as f:
            f.seek(offset)
            for a in pd.read_csv(f, sep=r'\s+', header=None, usecols=cols, dtype=np.float64, chunksize=chunk_size):
                # pandas sorts usecols by file order; restore the XYZ contract.
                xyz = a[cols].to_numpy()
                yield xyz[np.isfinite(xyz).all(axis=1)]
    else:
        raise ValueError(f'Unsupported PCD DATA {mode}')


def load_gpu_map(path, leaf):
    import cupy as cp
    from track3.gpu import voxel_mean
    points = weights = None
    count = 0
    for xyz in xyz_chunks(path):
        count += len(xyz)
        if not len(xyz):
            continue
        q, w = voxel_mean(xyz, leaf)
        if points is None:
            points, weights = q, w
        else:
            points, weights = voxel_mean(cp.concatenate([points, q]), leaf,
                                         cp.concatenate([weights, w]))
    if points is None:
        raise ValueError('Empty map')
    cp.cuda.get_current_stream().synchronize()
    return points, count
