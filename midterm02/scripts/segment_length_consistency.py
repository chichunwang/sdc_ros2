#!/usr/bin/env python3
"""沿行進方向的長度一致性（不需 ground truth）：同一段路在兩圈的里程計路徑長應該相等。

    python scripts/segment_length_consistency.py outputs/run/<run> [--poses poses_odometry.csv] [--min-length 5]

用已接受的閉環當錨點：閉環 (c, q) 表示較晚的 keyframe q 位在較早的 keyframe c 前方 z 公尺（T_rel 的前向分量）。
取同一對圈裡相鄰的兩個錨點 (c1,q1)、(c2,q2)，早的那圈走了 L_early = path(c1→c2)，
晚的那圈走了 L_late = path(q1→q2)，理論上 L_late = L_early + z2 − z1。
回報 log(L_late / (L_early + z2 − z1)) 的分布：標準差越小，兩圈的局部長度（尺度）越一致。
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run')
    ap.add_argument('--poses', default='poses_odometry.csv')
    ap.add_argument('--anchors', default=None)
    ap.add_argument('--min-length', type=float, default=5.0)
    ap.add_argument('--max-length', type=float, default=20.0)
    a = ap.parse_args()
    run = Path(a.run)
    anchors = Path(a.anchors) if a.anchors else run
    rows = list(csv.reader((run / a.poses).open()))[1:]
    xyz = np.array([[float(r[4]), float(r[8]), float(r[12])] for r in rows])
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    kf = {int(r['kf_id']): int(r['idx']) for r in csv.DictReader((anchors / 'keyframes.csv').open())}
    loops = []
    for r in csv.DictReader((anchors / 'loop_candidates.csv').open()):
        if r['accepted'] == '1':
            z = float(r['T_rel'].split()[11])
            loops.append((kf[int(r['match_kf'])], kf[int(r['query_kf'])], z))
    # 依「晚的那圈是第幾圈、早的那圈是第幾圈」分組（用最早閉環的幀距估計一圈的幀數），組內取相隔 min~max 公尺的錨點對
    lap_frames = min(q - c for c, q, _ in loops)
    groups = {}
    for c, q, z in loops:
        groups.setdefault((round(q / lap_frames), round((q - c) / lap_frames)), []).append((c, q, z))
    out = []
    for members in groups.values():
        members.sort(key=lambda t: t[1])
        for i, (c1, q1, z1) in enumerate(members):
            for c2, q2, z2 in members[i + 1:]:
                if c2 <= c1:
                    continue
                early, late = cum[c2] - cum[c1] + z2 - z1, cum[q2] - cum[q1]
                if a.min_length <= early <= a.max_length and late > 0:
                    out.append((c1, c2, q1, q2, early, late, np.log(late / early)))
                    break   # 每個起點只取第一個夠長的區段
    v = np.array([o[6] for o in out])
    for o in out:
        print('early %4d→%4d  late %4d→%4d  L_early %.2f  L_late %.2f  ratio %.3f' % (o[0], o[1], o[2], o[3], o[4], o[5], np.exp(o[6])))
    if len(v):
        print('%s %s: n=%d  ratio median %.3f  log-ratio std %.4f  robust std (MAD) %.4f  min %.3f  max %.3f' % (
            run.name, a.poses, len(v), np.exp(np.median(v)), v.std(), 1.4826 * np.median(np.abs(v - np.median(v))),
            np.exp(v.min()), np.exp(v.max())))


if __name__ == '__main__':
    main()
