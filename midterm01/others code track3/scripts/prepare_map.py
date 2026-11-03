"""地圖 npy → 分帶降取樣 + 分批法向量，存 npz。用法：prepare_map.py <map npy> <out npz> [--voxel 0.2]"""
import argparse, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sdcloc.mapio import load_map  # noqa: E402
from sdcloc.register import prepare_map  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('src', type=Path); ap.add_argument('dst', type=Path)
ap.add_argument('--voxel', type=float, default=0.2)
a = ap.parse_args()
t0 = time.time()
m = load_map(a.src, mmap=True)                   # mmap，分帶時才讀進來
print('map', m.shape, flush=True)
out = prepare_map(m[:, :3], voxel=a.voxel, extra=m[:, 3])
np.savez(a.dst, **out)
n = out['normals']
print('points', len(out['xyz']), 'ground-like(nz>0.9)', int((n[:, 2] > .9).sum()), 'vertical(|nz|<0.3)', int((np.abs(n[:, 2]) < .3).sum()),
      f'{time.time()-t0:.0f}s', flush=True)
