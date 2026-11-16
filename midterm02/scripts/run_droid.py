#!/usr/bin/env python3
"""DROID-SLAM 對照組／變體的正式執行入口（在 midterm02_droid 容器內，用 docker/droid/run.sh 啟動）。

    docker/droid/run.sh python3 scripts/run_droid.py --experiment D1_droid
    docker/droid/run.sh python3 scripts/run_droid.py --experiment D2_droid_selfcalib --calib-from outputs/run/<E21 的 run>

- 建立 outputs/run/track01_full_YYMMDDHHmm（程式快照、Git 狀態、test_info.json、frame_manifest.csv、result_1.csv）。
- D2：先用自校正估出的主點與徑向畸變（讀 --calib-from 那個 run 的 test_info.json → summary.graph.bundle.calibration）
  校正影像，再以針孔模型交給 DROID。DROID 演算法與參數維持原樣（commit 2dfd39f，預設參數）。
- 權重 outputs/cache/droid/droid.pth（官方 Google Drive，SHA256 46476ef6…）。
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, '/opt/DROID-SLAM/droid_slam')
from create_run import create_run, digest  # noqa: E402

EXPERIMENTS = {
    'D1_droid': dict(scale=0.5, calib=False),            # 原樣：給定的針孔內參
    'D2_droid_selfcalib': dict(scale=0.5, calib=True),   # 影像先用自校正的畸變校正
}
DROID_DEFAULTS = dict(buffer=1024, beta=0.3, filter_thresh=2.4, warmup=8, keyframe_thresh=4.0, frontend_thresh=16.0,
                      frontend_window=25, frontend_radius=2, frontend_nms=1, backend_thresh=22.0, backend_radius=2,
                      backend_nms=3, upsample=False, stereo=False, disable_vis=True, asynchronous=False,
                      frontend_device='cuda', backend_device='cuda')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument('--track', default='01')
    ap.add_argument('--calib-from', default=None, help='讀自校正結果的 run 目錄（D2 需要）')
    ap.add_argument('--weights', default='outputs/cache/droid/droid.pth')
    a = ap.parse_args()
    exp = EXPERIMENTS[a.experiment]
    data = ROOT / ('data/data_VSLAM-T1/Track1/data' if a.track == '01' else 'data/data_VSLAM-T2/Track2/data')
    result_name = 'result_%d.csv' % int(a.track)
    torch.manual_seed(0)
    np.random.seed(0)
    cv2.setRNGSeed(0)

    run = create_run('full', a.track)
    info_path = run / 'test_info.json'
    info = json.loads(info_path.read_text())
    ts = [r[0] for r in list(csv.reader(open(data / 'sample_submission.csv')))[1:]]
    ci = json.loads((data / 'camera_info.json').read_text())
    w0, h0 = ci['width'], ci['height']
    K = np.array([[ci['fx'], 0, ci['cx']], [0, ci['fy'], ci['cy']], [0, 0, 1.0]])
    calib = None
    if exp['calib']:
        if not a.calib_from:
            raise SystemExit('D2 需要 --calib-from（自校正 run 的目錄）')
        src = json.loads((ROOT / a.calib_from / 'test_info.json').read_text())
        calib = src['summary']['graph']['bundle']['calibration']
        K[0, 2], K[1, 2] = calib['cx'], calib['cy']
        dist = np.array([calib['k1'], calib['k2'], 0, 0, 0.0])
        maps = cv2.initUndistortRectifyMap(K, dist, None, K, (w0, h0), cv2.CV_32FC1)
    s = exp['scale']
    w1, h1 = int(w0 * s), int(h0 * s)
    w1, h1 = w1 - w1 % 8, h1 - h1 % 8
    intr = torch.tensor([K[0, 0] * w1 / w0, K[1, 1] * h1 / h0, K[0, 2] * w1 / w0, K[1, 2] * h1 / h0], dtype=torch.float32)

    def stream():
        for t, name in enumerate(ts):
            img = cv2.imread(str(data / 'images' / ('%s.jpg' % name)))
            if calib is not None:
                img = cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR)
            img = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_AREA)
            yield t, torch.as_tensor(img).permute(2, 0, 1)[None], intr.clone()

    with open(run / 'frame_manifest.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'timestamp_ns', 'image', 'ply'])
        for i, name in enumerate(ts):
            w.writerow([i, name, 'images/%s.jpg' % name, 'raw_ply/%s.ply' % name])
    params = dict(experiment=a.experiment, image_scale=s, image_size=[w1, h1], intrinsics=intr.tolist(),
                  calibration=calib, calib_from=a.calib_from, droid=DROID_DEFAULTS, droid_commit='2dfd39f',
                  weights=a.weights, weights_sha256=digest(ROOT / a.weights))
    info.update(status='running', version='droid-2dfd39f', experiment=a.experiment, parameters=params, seed=0,
                frame_count=len(ts), command=' '.join(sys.argv))
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    (run / 'debug').mkdir(exist_ok=True)

    from droid import Droid
    args = argparse.Namespace(weights=str(ROOT / a.weights), image_size=[h1, w1], **DROID_DEFAULTS)
    droid = Droid(args)
    t0 = time.time()
    for t, img, k in stream():
        droid.track(t, img, intrinsics=k)
    n_kf = droid.video.counter.value
    t1 = time.time()
    traj = droid.terminate(stream())    # (N,7) camera-to-world
    t2 = time.time()
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat(traj[:, 3:7]).as_matrix()
    with open(run / result_name, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp'] + ['m%d%d' % (i, j) for i in range(4) for j in range(4)])
        for name, Ri, ti in zip(ts, R, traj[:, :3]):
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = Ri, ti
            w.writerow([name] + ['%.10g' % v for v in T.ravel()])
    import subprocess
    chk = subprocess.run([sys.executable, str(ROOT / 'scripts/check_submission.py'), str(run / result_name),
                          str(data / 'sample_submission.csv')], capture_output=True, text=True)
    info.update(status='success' if chk.returncode == 0 else 'failed',
                timings_seconds=dict(tracking=t1 - t0, global_ba_and_fill=t2 - t1),
                summary=dict(keyframes=int(n_kf)), submission=dict(check_returncode=chk.returncode,
                                                                    check_output=chk.stdout[-2000:] + chk.stderr[-2000:]))
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    print('run:', run)
    print('submission check:', 'OK' if chk.returncode == 0 else 'FAILED', chk.stdout[-300:], chk.stderr[-300:])


if __name__ == '__main__':
    main()
