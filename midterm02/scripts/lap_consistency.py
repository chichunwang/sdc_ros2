#!/usr/bin/env python3
"""沒有 ground truth 時的局部形狀診斷：同一路段在不同圈的軌跡，Sim(3) 對齊後還差多少。

    python scripts/lap_consistency.py outputs/run/<run> [--poses poses_optimized.csv] [--window 132]

做法：用該 run 已接受的閉環配對（match_kf 在較早的圈、query_kf 在較晚的圈）當作「同一地點」的錨點。
以每個錨點為中心各取一段軌跡（晚的那圈取 window 幀，早的那圈取寬一點），把晚的那段用 Sim(3) 對齊到早的那段的
折線上（最近點迭代＋Umeyama），回報對齊後的 RMSE。

意義與限制：
- 這量的是「重複性」不是精度：兩圈若有相同的系統性變形，這裡看不出來。
- 兩圈的實際行車路線本來就不會完全一樣（車道內偏移），所以數值有非零下限。
- 用途是比較不同演算法版本、以及比較 odometry 與最佳化後的軌跡；window=132 對應 Kaggle Public 的幀數。
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def umeyama(src, dst):
    """求 s, R, t 使 s·R·src + t ≈ dst（最小平方）。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / max((a ** 2).sum() / len(src), 1e-12)
    return s, R, mu_d - s * R @ mu_s


def densify(points, factor=10):
    t = np.linspace(0, len(points) - 1, (len(points) - 1) * factor + 1)
    i = np.minimum(t.astype(int), len(points) - 2)
    f = (t - i)[:, None]
    return points[i] * (1 - f) + points[i + 1] * f


def align_rmse(seg, ref, iterations=30):
    ref = densify(ref)
    s, R, t = 1.0, np.eye(3), np.zeros(3)
    for _ in range(iterations):
        moved = s * seg @ R.T + t
        nearest = ref[np.argmin(((moved[:, None, :] - ref[None, :, :]) ** 2).sum(-1), axis=1)]
        s, R, t = umeyama(seg, nearest)
    moved = s * seg @ R.T + t
    d = np.sqrt(((moved[:, None, :] - ref[None, :, :]) ** 2).sum(-1).min(axis=1))
    return float(np.sqrt(np.mean(d ** 2))), float(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run')
    ap.add_argument('--poses', default='poses_optimized.csv')
    ap.add_argument('--anchors', default=None, help='改用另一個 run 的閉環配對當錨點（比較沒有閉環的版本時用）')
    ap.add_argument('--window', type=int, default=132)
    a = ap.parse_args()
    run = Path(a.run)
    anchors = Path(a.anchors) if a.anchors else run
    rows = list(csv.reader((run / a.poses).open()))[1:]
    xyz = np.array([[float(r[4]), float(r[8]), float(r[12])] for r in rows])
    kf = {int(r['kf_id']): int(r['idx']) for r in csv.DictReader((anchors / 'keyframes.csv').open())}
    half = a.window // 2
    out = []
    for r in csv.DictReader((anchors / 'loop_candidates.csv').open()):
        if r['accepted'] != '1':
            continue
        early, late = kf[int(r['match_kf'])], kf[int(r['query_kf'])]
        lo, hi = late - half, late + half
        e_lo, e_hi = early - half - 40, early + half + 40
        if lo < e_hi or hi > len(xyz) or e_lo < 0:
            continue
        rmse, scale = align_rmse(xyz[lo:hi], xyz[e_lo:e_hi])
        out.append((early, late, rmse, scale))
        print('early %4d  late %4d  rmse %.3f m  scale(late→early) %.3f' % (early, late, rmse, scale))
    if out:
        r = np.array([o[2] for o in out]); s = np.array([o[3] for o in out])
        print('%s %s: n=%d  rmse median %.3f  mean %.3f  max %.3f | scale median %.3f  min %.3f  max %.3f' % (
            run.name, a.poses, len(out), np.median(r), r.mean(), r.max(), np.median(s), s.min(), s.max()))


if __name__ == '__main__':
    main()
