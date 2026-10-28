"""第一輪 Sim(3) 結果 → 逐幀尺度的穩健平滑（好 frame 的 running median，再內插到全部），存 npy 並畫圖。"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--reg', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--window', type=int, default=61, help='running median 視窗（幀）')
ap.add_argument('--min-frac', type=float, default=0.5)
ap.add_argument('--fig', type=Path, default=Path('.tmp/fig/scale_smooth.png'))
a = ap.parse_args()
r = np.load(a.reg)
fr, s, info, frac = r['frame'], r['scale_est'], r['scale_info'], r['ng_frac03']
ok = r['ok'] & (frac >= a.min_frac) & np.isfinite(s) & (info > np.nanpercentile(info[r['ok']], 20))
print('good frames for scale', int(ok.sum()), '/', len(fr))
fg, sg = fr[ok], s[ok]
# running median（以幀索引為單位，視窗內取中位數）
half = a.window // 2
med = np.array([np.median(sg[(fg >= f - half) & (fg <= f + half)]) for f in fg])
# 再一次：去掉離 median 太遠的（>5%），重算
keep = np.abs(sg / med - 1) < 0.05
fg2, sg2 = fg[keep], sg[keep]
med2 = np.array([np.median(sg2[(fg2 >= f - half) & (fg2 <= f + half)]) for f in fg2])
all_frames = np.arange(int(fr.max()) + 1)
scale_all = np.interp(all_frames, fg2, med2)
np.save(a.out, scale_all)
print('scale: median %.3f min %.3f max %.3f; outliers removed %d' % (np.median(scale_all), scale_all.min(), scale_all.max(), int((~keep).sum())))
a.fig.parent.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(fr[r['ok']], s[r['ok']], '.', ms=2, color='gray', label='per-frame Sim3 (all)')
ax.plot(fg, sg, '.', ms=3, color='b', label='good frames')
ax.plot(all_frames, scale_all, 'r-', lw=2, label='smoothed')
ax.set_ylim(6, 11); ax.set_xlabel('frame'); ax.set_ylabel('scale'); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(a.fig, dpi=90)
