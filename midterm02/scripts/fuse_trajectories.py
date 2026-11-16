#!/usr/bin/env python3
"""兩個獨立估計的融合：Sim(3) 對齊後逐幀平均位置（實驗）。

    python scripts/fuse_trajectories.py --base <run A> --other <run B> [<run C> ...] [--track 01]

為什麼：我們的 BA（稀疏特徵）與 DROID（稠密光流）是兩個獨立的估計器，自校正後兩者在 132 幀視窗只差約 3 cm。
若兩者的誤差有一部分彼此獨立、量級相近，最小變異的組合就是等權平均（各自的獨立誤差減為 1/√2）。
做法：其他來源各自以全域 Sim(3)（Umeyama）對齊到 A，逐幀位置取等權平均；旋轉沿用 A（評分只看位置，旋轉只需合法）。
輸出新的 run 目錄（track01_full_*），test_info.json 記錄兩個來源 run 與其 result 檔的 SHA256。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from create_run import create_run, digest  # noqa: E402
from lap_consistency import umeyama  # noqa: E402


def load(fn):
    rows = list(csv.reader(open(fn)))[1:]
    return [r[0] for r in rows], np.array([[float(x) for x in r[1:]] for r in rows]).reshape(-1, 4, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--other', required=True, nargs='+')
    ap.add_argument('--track', default='01')
    a = ap.parse_args()
    name = 'result_%d.csv' % int(a.track)
    fa = ROOT / a.base / name
    ta, A = load(fa)
    pa = A[:, :3, 3]
    aligned, shas, scales = [pa], [], []
    for o in a.other:
        fb = ROOT / o / name
        tb, B = load(fb)
        if ta != tb:
            raise SystemExit('結果的 timestamp 不一致：%s' % o)
        s, R, t = umeyama(B[:, :3, 3], pa)
        aligned.append(s * B[:, :3, 3] @ R.T + t)
        shas.append(digest(fb))
        scales.append(float(s))
    fused = A.copy()
    fused[:, :3, 3] = np.mean(aligned, axis=0)
    pb_al = aligned[1]
    run = create_run('full', a.track)
    with open(run / name, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp'] + ['m%d%d' % (i, j) for i in range(4) for j in range(4)])
        for tsn, T in zip(ta, fused):
            w.writerow([tsn] + ['%.10g' % v for v in T.ravel()])
    import subprocess
    data = ROOT / ('data/data_VSLAM-T1/Track1/data' if a.track == '01' else 'data/data_VSLAM-T2/Track2/data')
    chk = subprocess.run([sys.executable, str(ROOT / 'scripts/check_submission.py'), str(run / name),
                          str(data / 'sample_submission.csv')], capture_output=True, text=True)
    info_path = run / 'test_info.json'
    info = json.loads(info_path.read_text())
    d = np.linalg.norm(pb_al - pa, axis=1)
    info.update(status='success' if chk.returncode == 0 else 'failed', version='fuse-1', experiment='F_fuse_mean',
                parameters=dict(base=a.base, other=a.other, base_sha256=digest(fa), other_sha256=shas,
                                method='Sim(3)-align others to base, equal-weight average of positions, rotations from base',
                                sim3_scale=scales),
                frame_count=len(ta), seed=None,
                summary=dict(base_other_rms_m=float(np.sqrt(np.mean(d ** 2))), base_other_max_m=float(d.max())),
                submission=dict(check_returncode=chk.returncode, check_output=chk.stdout[-1000:] + chk.stderr[-1000:]))
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    print('run:', run)
    print('submission check:', 'OK' if chk.returncode == 0 else 'FAILED')


if __name__ == '__main__':
    main()
