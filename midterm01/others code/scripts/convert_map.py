"""map.pcd（ASCII, x y z intensity）→ npy（float32 N×4），之後讀取只要 1 秒。"""
import sys, time
from pathlib import Path
import numpy as np
import polars as pl

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
t0 = time.time()
with src.open('rb') as f:
    header = []
    while True:
        line = f.readline().decode()
        header.append(line.strip())
        if line.startswith('DATA'):
            break
    assert 'DATA ascii' in header[-1], header[-1]
    n = int([h for h in header if h.startswith('POINTS')][0].split()[1])
    fields = [h for h in header if h.startswith('FIELDS')][0].split()[1:]
    print('fields', fields, 'points', n, flush=True)
    df = pl.read_csv(f, separator=' ', has_header=False, new_columns=fields,
                     schema_overrides={c: pl.Float32 for c in fields})
arr = df.to_numpy().astype(np.float32)
assert arr.shape == (n, len(fields)), arr.shape
dst.parent.mkdir(parents=True, exist_ok=True)
np.save(dst, arr)
print('saved', dst, arr.shape, f'{time.time()-t0:.0f}s',
      'x', arr[:, 0].min(), arr[:, 0].max(), 'y', arr[:, 1].min(), arr[:, 1].max(),
      'z', arr[:, 2].min(), arr[:, 2].max(), 'i', arr[:, 3].min(), arr[:, 3].max())
