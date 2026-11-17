#!/usr/bin/env python3
"""DROID-SLAM 對照組／變體的正式執行入口（在 midterm02_droid 容器內，用 docker/droid/run.sh 啟動）。

    docker/droid/run.sh python3 scripts/run_droid.py --experiment D1_droid
    docker/droid/run.sh python3 scripts/run_droid.py --experiment D2_droid_selfcalib --calib-from outputs/run/<E21 的 run>

- 建立 outputs/run/track01_full_YYMMDDHHmm（程式快照、Git 狀態、test_info.json、frame_manifest.csv、result_1.csv）。
- D2：先用自校正估出的主點與徑向畸變（讀 --calib-from 那個 run 的 test_info.json → summary.graph.bundle.calibration）
  校正影像，再以針孔模型交給 DROID。DROID 演算法與參數維持原樣（commit 2dfd39f，預設參數）。
- D5：D2＋LoGeR 深度當 DROID 的 RGB-D 深度先驗（2026-11-17，Track2）。觀察：Track2 是長直路、幾乎不重複，
  單眼 DROID 的尺度沿路漂移（D1 Public 25.1、D2 3.32，baseline 0.658）；LoGeR 每幀各自估深度（約公尺、不累積）。
  點雲是官方針孔內參從 315×196 像素網格反投影而來（dev42_ply_grid.py），所以先用官方內參投影回原始像素，
  再用自校正的畸變校正到 D2 的影像座標，每個 DROID 1/8 網格取深度中位數；50 m 截斷值的點（天空）不用。
  DROID 原本的 RGB-D 模式把它當成視差先驗（ba 的 alpha=0.05），不是硬約束，也不再做單眼的尺度正規化。
- D6：D5 的 LoGeR 深度先用「相機離地高度」正規化（2026-11-17）。觀察：D5 Public 11.46（比 D2 3.32 差）；
  LoGeR／視覺速度比與 LoGeR 估的相機離地高度相關 0.958（dev47_loger_height.py）：相機固定在車上、高度不變，
  所以 LoGeR 高度的起伏（0.8–1.24 倍）就是它的尺度偏差。每幀深度乘 s_i = median(h)/h̃_i，h̃ 為 ±15 幀中位數
  （每幀地面擬合約 ±10% 雜訊，偏差以數百幀尺度變化；擬合失敗的幀用鄰近值）。地面擬合用 gslam.scale.ground_height 與 params 的 scale 設定。
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
    # 同 D2，提高影像解析度（光流更細）。前端的整張相關體積在高解析度超過 int32 上限，
    # 改用官方後端本來就用的 alt 相關（逐點即時計算，同一個相關值，只是不預先存整張體積）
    'D2a_droid_selfcalib_alt': dict(scale=0.5, calib=True, alt_corr=True),   # 驗證：alt 相關在 D2 的解析度應與 D2 幾乎相同
    'D3_droid_selfcalib_s075': dict(scale=0.75, calib=True, alt_corr=True),
    'D4_droid_selfcalib_s100': dict(scale=1.0, calib=True, alt_corr=True, buffer=512),   # 原解析度時 1024 張特徵圖超過 int32 元素上限
    'D5_droid_selfcalib_depth': dict(scale=0.5, calib=True, depth=True),   # D2＋LoGeR 深度先驗（RGB-D 模式）
    'D6_droid_selfcalib_depth_hnorm': dict(scale=0.5, calib=True, depth=True, height_norm=True),   # D5＋相機高度正規化
}
HEIGHT_MEDIAN_HALF = 15   # 相機高度的時間中位數半窗（幀）：壓掉單幀地面擬合的離群值，遠小於 LoGeR 偏差的變化尺度


def _height_job(args):
    path, seed = args
    sys.path.insert(0, str(ROOT / 'vslam'))
    from gslam.params import PARAMS
    from gslam.scale import ground_height
    h, _ = ground_height(read_ply_xyz(path), PARAMS['scale'], seed=seed)
    return np.nan if h is None else h


def height_scales(paths, half=HEIGHT_MEDIAN_HALF):
    """每幀 LoGeR 相機離地高度 → 尺度係數 s_i = median(h̃)/h̃_i（h̃：±half 幀中位數，跳過擬合失敗的幀）。"""
    from multiprocessing import Pool
    with Pool(12) as pool:
        h = np.array(pool.map(_height_job, [(p, i) for i, p in enumerate(paths)], chunksize=20))
    hs = np.array([np.nanmedian(h[max(0, i - half):i + half + 1]) for i in range(len(h))])
    if np.isnan(hs).any():   # 整個視窗都失敗：用最近的有效值
        ok = np.flatnonzero(~np.isnan(hs))
        hs = hs[ok[np.abs(np.arange(len(hs))[:, None] - ok[None]).argmin(1)]]
    return h, hs, np.median(hs) / hs
LOGER_CLIP_M = 49.5   # LoGeR 點雲的深度在 50 m 截斷（最大值 49.9）；截斷處不是真實深度，不當先驗


def read_ply_xyz(path):
    """官方 raw_ply（binary little endian：float x y z、uchar r g b）→ (N,3) float64，相機座標。"""
    raw = open(path, 'rb').read()
    end = raw.index(b'end_header\n') + len(b'end_header\n')
    n = int([l for l in raw[:end].decode().splitlines() if l.startswith('element vertex')][0].split()[-1])
    a = np.frombuffer(raw[end:end + n * 15], dtype=np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                                           ('r', 'u1'), ('g', 'u1'), ('b', 'u1')]))
    return np.stack([a['x'], a['y'], a['z']], 1).astype(np.float64)


def loger_depth(xyz, K0, Kc, dist, size0, size1):
    """LoGeR 點雲 → DROID 輸入解析度的深度圖（只填 DROID 取樣的 [3::8, 3::8] 網格點，其他為 0＝無資料）。

    xyz：相機座標點（官方針孔 K0 從原始像素網格反投影）。Kc、dist：自校正的內參與畸變（D2 的去畸變用同一組）。
    """
    z = xyz[:, 2]
    keep = np.isfinite(z) & (z > 0.1) & (z < LOGER_CLIP_M)
    xyz, z = xyz[keep], z[keep]
    uv = (xyz @ K0.T)[:, :2] / z[:, None]                       # 原始（有畸變）像素
    if dist is not None:
        uv = cv2.undistortPoints(uv.reshape(-1, 1, 2), Kc, dist, P=Kc).reshape(-1, 2)   # → 去畸變後影像的像素
    (w0, h0), (w1, h1) = size0, size1
    u1, v1 = uv[:, 0] * w1 / w0, uv[:, 1] * h1 / h0
    gw, gh = w1 // 8, h1 // 8
    c, r = np.floor(u1 / 8).astype(int), np.floor(v1 / 8).astype(int)
    ok = (c >= 0) & (c < gw) & (r >= 0) & (r < gh)
    cell, z = r[ok] * gw + c[ok], z[ok]
    order = np.lexsort((z, cell))
    cell, z = cell[order], z[order]
    starts = np.flatnonzero(np.r_[True, np.diff(cell) > 0])
    counts = np.diff(np.r_[starts, len(cell)])
    med = z[starts + (counts - 1) // 2]                          # 每格深度中位數
    grid = np.zeros(gh * gw, dtype=np.float32)
    grid[cell[starts]] = med
    depth = np.zeros((h1, w1), dtype=np.float32)
    depth[3::8, 3::8][:gh, :gw] = grid.reshape(gh, gw)
    return depth
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
    ap.add_argument('--buffer', type=int, default=None,
                    help='keyframe 容量（預先配置的儲存空間，不是演算法參數）；Track2 有 4206 幀，keyframe 超過預設 1024')
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
    K0 = K.copy()                       # 官方針孔內參（LoGeR 點雲反投影用的）
    dist = None
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

    hscale = None
    if exp.get('height_norm'):
        h_raw, h_med, hscale = height_scales([data / 'raw_ply' / ('%s.ply' % n) for n in ts])
        (run / 'debug').mkdir(exist_ok=True)
        with open(run / 'debug' / 'loger_height.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['idx', 'timestamp_ns', 'height_raw', 'height_median', 'depth_scale'])
            for i, n in enumerate(ts):
                w.writerow([i, n, h_raw[i], h_med[i], hscale[i]])
        print('LoGeR height: fit ok %.1f%%, scale p5/p95 %.3f/%.3f' % (
            100 * np.isfinite(h_raw).mean(), np.percentile(hscale, 5), np.percentile(hscale, 95)), flush=True)

    def depth_of(t):
        if not exp.get('depth'):
            return None
        xyz = read_ply_xyz(data / 'raw_ply' / ('%s.ply' % ts[t]))
        d = loger_depth(xyz, K0, K, dist, (w0, h0), (w1, h1))   # 截斷判斷用原始深度；投影與尺度無關，最後再乘係數
        return torch.as_tensor(d * np.float32(hscale[t]) if hscale is not None else d)

    with open(run / 'frame_manifest.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'timestamp_ns', 'image', 'ply'])
        for i, name in enumerate(ts):
            w.writerow([i, name, 'images/%s.jpg' % name, 'raw_ply/%s.ply' % name])
    droid_cfg = dict(DROID_DEFAULTS, buffer=a.buffer or exp.get('buffer', DROID_DEFAULTS['buffer']))
    params = dict(experiment=a.experiment, alt_corr=bool(exp.get('alt_corr')), image_scale=s, image_size=[w1, h1], intrinsics=intr.tolist(),
                  calibration=calib, calib_from=a.calib_from,
                  depth_prior=('LoGeR raw_ply, median per 1/8 cell, z < %.1f m' % LOGER_CLIP_M) if exp.get('depth') else None,
                  height_norm=dict(median_half_frames=HEIGHT_MEDIAN_HALF, ground='gslam.scale.ground_height + PARAMS.scale',
                                   file='debug/loger_height.csv') if exp.get('height_norm') else None, droid=droid_cfg, droid_commit='2dfd39f',
                  weights=a.weights, weights_sha256=digest(ROOT / a.weights))
    info.update(status='running', version='droid-2dfd39f', experiment=a.experiment, parameters=params, seed=0,
                frame_count=len(ts), command=' '.join(sys.argv))
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    (run / 'debug').mkdir(exist_ok=True)

    from droid import Droid
    if exp.get('alt_corr'):
        import factor_graph
        init0 = factor_graph.FactorGraph.__init__

        update0 = factor_graph.FactorGraph.update

        def init_alt(self, *args_, **kw):
            kw['corr_impl'] = 'alt'
            init0(self, *args_, **kw)

        def update_alt(self, *args_, **kw):
            # 前端的 update() 會呼叫 self.corr(coords)；alt 模式下改成即時計算 fmaps[ii]·fmaps[jj] 的相關（單眼 rig=1）
            num, rig, ch, ht, wd = self.video.fmaps.shape
            op = factor_graph.AltCorrBlock(self.video.fmaps.view(1, num * rig, ch, ht, wd))
            ii, jj = self.ii, self.jj

            def corr(coords, n=16):   # 分批計算，避免 CUDA kernel 的 int32 索引上限
                return torch.cat([op(coords[:, k:k + n], ii[k:k + n], jj[k:k + n]) for k in range(0, len(ii), n)], 1)
            self.corr = corr
            self.inp = self.video.inps[ii].to(self.device).unsqueeze(0)   # volume 模式在 add_factors 裡存的同一個量
            return update0(self, *args_, **kw)
        factor_graph.FactorGraph.__init__ = init_alt
        factor_graph.FactorGraph.update = update_alt
    args = argparse.Namespace(weights=str(ROOT / a.weights), image_size=[h1, w1], **droid_cfg)
    droid = Droid(args)
    t0 = time.time()
    for t, img, k in stream():
        droid.track(t, img, depth=depth_of(t), intrinsics=k)
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
    # 成功的 full 自動產生唯讀圖表（失敗只記 analysis_status.json，不影響定位結果）
    subprocess.run([sys.executable, str(ROOT / 'scripts/analyze_external_run.py'), str(run)])
    print('run:', run)
    print('submission check:', 'OK' if chk.returncode == 0 else 'FAILED', chk.stdout[-300:], chk.stderr[-300:])


if __name__ == '__main__':
    main()
