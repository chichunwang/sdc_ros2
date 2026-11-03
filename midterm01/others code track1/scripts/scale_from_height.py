"""逐幀尺度：假設相機離地高度 H 固定，scale(k) = H / h_ply(k)（h_ply：PLY 原始單位下、相機正前方路面的高度）。
不靠 GNSS、不靠地圖（H 可由任何一段配準的「相機 z − 路面 z」校正）。Track1 實測和 GNSS 管線估的尺度相關 0.87、殘差 MAD 2.5%，
起點那段反而比 61 幀平滑檔準（7.9 vs 8.56）。輸出 npy，一幀一個值，給 track_sequential／register_all --scale-file。"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sdcloc.cloud import PlyCache, ground_level_m  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plys', required=True)
    ap.add_argument('--H', type=float, default=1.573, help='相機離地高（m）；Track1 用 GNSS 管線尺度校正得 1.573')
    ap.add_argument('--nominal', type=float, default=8.6, help='ground_level_m 的公尺窗口要先用一個名義尺度')
    ap.add_argument('--window', type=int, default=7, help='running median 視窗（幀）')
    ap.add_argument('--causal', action='store_true', help='中位數只用當幀和之前的幀（因果版，不碰未來幀）')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    plys = PlyCache(a.plys)
    n = len(plys)
    h = np.full(n, np.nan)
    for k in range(n):
        raw, _ = plys.get(k)
        g = ground_level_m(raw * a.nominal)
        if np.isfinite(g):
            h[k] = -g / a.nominal
        if k % 500 == 0:
            print(k, '/', n, flush=True)
    ok = np.isfinite(h) & (h > 0.02)
    s_raw = np.where(ok, a.H / np.where(ok, h, 1.0), np.nan)
    half = a.window // 2
    if a.causal:
        s = np.array([np.nanmedian(s_raw[max(0, k - a.window + 1):k + 1]) for k in range(n)])
        for k in range(n):                      # 缺的沿用前一幀
            if not np.isfinite(s[k]):
                s[k] = s[k - 1] if k > 0 and np.isfinite(s[k - 1]) else a.H / 0.18
    else:
        s = np.array([np.nanmedian(s_raw[max(0, k - half):k + half + 1]) for k in range(n)])
        bad = ~np.isfinite(s)
        if bad.any():
            idx = np.arange(n)
            s[bad] = np.interp(idx[bad], idx[~bad], s[~bad])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, s)
    print('saved', a.out, 'n', n, 'valid', int(ok.sum()), 'scale med %.3f p10 %.3f p90 %.3f' % (np.median(s), np.percentile(s, 10), np.percentile(s, 90)))


if __name__ == '__main__':
    main()
