#!/usr/bin/env python3
"""F1：DROID 前端＋gslam 全解析度 SIFT BA 後端（含相機自校正）。

為什麼：DROID 的稠密光流在 0.5 倍解析度（720×464）上估計，對停車、橫越車輛等動態場景很穩，大範圍形狀也已經由
D8（深度先驗＋閉環）與 P1（反向重訪約束）處理好；gslam 的 BA 則在原解析度 1440×928 用 SIFT 特徵軌跡（含跨圈軌跡）
做重投影最佳化，局部精度較高，但 E21 自己的里程計初值在 Track2 會壞掉。
F1 把兩者接起來：BA 初值與相鄰幀邊（鬆，只固定尺度、撐住特徵少的幀）來自 P1；同方向閉環與反向約束也以邊的形式保留，
避免 BA 只看局部重投影時把 P1 修好的大範圍關係放掉。SIFT 特徵與匹配直接用某個 gslam run 存下的 BA 輸入（不重抽）。

    python scripts/droid_ba.py --from outputs/run/<P1 run> --ba-input outputs/run/<gslam run> [--experiment E27_selfcalib_height]
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'vslam'))
from create_run import create_run, digest  # noqa: E402


def load_poses(fn):
    rows = list(csv.reader(open(fn)))[1:]
    return [r[0] for r in rows], np.array([[float(x) for x in r[1:]] for r in rows]).reshape(-1, 4, 4)


def main():
    import gtsam
    from gslam import bundle, params as params_mod
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='src', required=True, help='P1 的 run 目錄（位姿＋debug/edges.npz）')
    ap.add_argument('--ba-input', required=True, help='有 debug/bundle_input.npz 的 gslam run（SIFT 特徵與匹配）')
    ap.add_argument('--experiment', default='E27_selfcalib_height', help='BA 參數取自 gslam/params.py 的這個實驗')
    a = ap.parse_args()
    src = (ROOT / a.src) if not Path(a.src).is_absolute() else Path(a.src)
    bsrc = (ROOT / a.ba_input) if not Path(a.ba_input).is_absolute() else Path(a.ba_input)
    sinfo = json.loads((src / 'test_info.json').read_text())
    track = sinfo['track']
    data = ROOT / ('data/data_VSLAM-T%d/Track%d/data' % (int(track), int(track)))
    result_name = 'result_%d.csv' % int(track)
    ts, T = load_poses(src / result_name)
    p, _ = params_mod.load(a.experiment)
    cfg, g = p['bundle'], p['graph']

    run = create_run('full', track)
    info_path = run / 'test_info.json'
    info = json.loads(info_path.read_text())
    t0 = time.time()
    K, _, kpts, links, _ = bundle.load_input(bsrc / 'debug' / 'bundle_input.npz')
    if len(kpts) != len(T):
        raise SystemExit('BA 輸入 %d 幀與位姿 %d 幀不一致' % (len(kpts), len(T)))
    E = np.load(src / 'debug' / 'edges.npz')
    sig = lambda r, t: np.array([r, r, r, t, t, t], dtype=np.float64)
    odo = gtsam.noiseModel.Diagonal.Sigmas(sig(cfg['odom_rot_sigma_rad'], cfg['odom_trans_sigma_m']))
    robust = lambda s: gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(g['huber_k']),
                                                      gtsam.noiseModel.Diagonal.Sigmas(np.asarray(s, dtype=np.float64)))
    between = [(i - 1, i, np.linalg.inv(T[i - 1]) @ T[i], odo) for i in range(1, len(T))]
    same_noise, opp_noise = robust(E['same_sigma']), robust(E['opp_sigma'])
    between += [(int(i), int(j), Z, same_noise) for (i, j), Z in zip(E['same_ij'], E['same_T'])]
    between += [(int(i), int(j), Z, opp_noise) for (i, j), Z in zip(E['opp_ij'], E['opp_T'])]
    out, binfo = bundle.run(K, list(T), kpts, links, between, cfg)
    t1 = time.time()
    with open(run / result_name, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp'] + ['m%d%d' % (i, j) for i in range(4) for j in range(4)])
        for name, M in zip(ts, out):
            w.writerow([name] + ['%.10g' % v for v in np.asarray(M).ravel()])
    with open(run / 'frame_manifest.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'timestamp_ns', 'image', 'ply'])
        for i, name in enumerate(ts):
            w.writerow([i, name, 'images/%s.jpg' % name, 'raw_ply/%s.ply' % name])
    chk = subprocess.run([sys.executable, str(ROOT / 'scripts/check_submission.py'), str(run / result_name),
                          str(data / 'sample_submission.csv')], capture_output=True, text=True)
    info.update(status='success' if chk.returncode == 0 and binfo.get('status') == 'ok' else 'failed',
                experiment='F1_droid_sift_ba', version='droid_ba-1', frame_count=len(ts),
                parameters=dict(source_run=str(src.relative_to(ROOT)), source_result_sha256=digest(src / result_name),
                                ba_input_run=str(bsrc.relative_to(ROOT)),
                                ba_input_sha256=digest(bsrc / 'debug' / 'bundle_input.npz'),
                                bundle_params_from=a.experiment, bundle=cfg, graph=g),
                timings_seconds=dict(bundle=t1 - t0),
                summary=dict(edges=dict(odometry=len(T) - 1, same_direction=int(len(E['same_ij'])), opposite=int(len(E['opp_ij']))),
                             bundle=binfo),
                submission=dict(check_returncode=chk.returncode, check_output=chk.stdout[-2000:] + chk.stderr[-2000:]))
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2, default=float) + '\n')
    subprocess.run([sys.executable, str(ROOT / 'scripts/analyze_external_run.py'), str(run)])
    print('run:', run)
    print('submission check:', 'OK' if chk.returncode == 0 else 'FAILED', chk.stdout[-300:], chk.stderr[-300:])


if __name__ == '__main__':
    main()
