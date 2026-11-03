"""把 raw_ply 全部讀成 mmap 友善的 npy：<stem>_xyz.npy、<stem>_offsets.npy、<stem>_ts.npy（多程序共用）。"""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sdcloc.frames import load_frames  # noqa: E402
from sdcloc.ply import read_ply  # noqa: E402

ply_dir, dst = Path(sys.argv[1]), Path(sys.argv[2])
ts, _ = load_frames(ply_dir)
files = sorted(ply_dir.glob('*.ply'), key=lambda p: int(p.stem.split('_ts_')[1]))
assert len(files) == len(ts)
xyz_all, rgb_all, offsets = [], [], [0]
t0 = time.time()
for i, f in enumerate(files):
    xyz, rgb = read_ply(f)
    xyz_all.append(xyz.astype(np.float32)); rgb_all.append(rgb); offsets.append(offsets[-1] + len(xyz))
    if i % 500 == 0:
        print(i, f'{time.time()-t0:.0f}s', flush=True)
stem = str(dst)[:-4] if str(dst).endswith('.npz') else str(dst)
np.save(stem + '_xyz.npy', np.vstack(xyz_all)); np.save(stem + '_offsets.npy', np.array(offsets, dtype=np.int64)); np.save(stem + '_ts.npy', ts)
print('saved', dst, 'frames', len(files), 'points', offsets[-1], f'{time.time()-t0:.0f}s')
