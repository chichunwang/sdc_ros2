#!/usr/bin/env python3
"""把跑完的 run（DROID／P1／F1／任何有 result_*.csv 的 run）重播到 RViz，方便除錯。

只有 gslam（E 系列）在執行中會串流到 RViz；DROID、P1、F1 是另外的程式，跑的時候沒有畫面。
這個工具在 run 結束後，依它的位姿逐幀送出：相機影像、該幀的 LoGeR 點雲、累積的點雲地圖、軌跡，
以及閉環／反向重訪配對（綠線，取自 run 或它來源的 debug/edges.npz）。另一條路徑（/vslam/path_odometry）
畫對照 run（預設是 Track2 目前最佳 F1），先 Sim(3) 對齊到這個 run。只讀取檔案，不改任何結果。

    ./scripts/dev.sh replay outputs/run/<run> [--ref outputs/run/<對照 run>] [--step 2] [--fps 8] [--start 1550 --end 1750]

點雲尺度：Track2 的 DROID 系列單位＝相機高度正規化的 LoGeR，所以每幀點雲乘 D8 的 depth_scale，
再乘「這個 run 的路長 ÷ D8 路長」；其他情況只乘路長比（以對照 run 為準），只影響顯示。
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'vslam'))
from lap_consistency import umeyama  # noqa: E402

DEFAULT_REF = {'02': 'outputs/run/track02_full_2611172344', '01': 'outputs/run/track01_full_2611161954'}
HEIGHT_FROM = {'02': 'outputs/run/track02_full_2611172006'}   # 有 debug/loger_height.csv 的 D8


def load(fn):
    rows = list(csv.reader(open(fn)))[1:]
    return [r[0] for r in rows], np.array([[float(x) for x in r[1:]] for r in rows]).reshape(-1, 4, 4)


def path_len(T):
    return float(np.sum(np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1)))


def find_edges(run):
    """run 自己或來源鏈（parameters.source_run）上的第一個 debug/edges.npz。"""
    for _ in range(6):
        if (run / 'debug' / 'edges.npz').exists():
            return run / 'debug' / 'edges.npz'
        src = json.loads((run / 'test_info.json').read_text()).get('parameters', {}).get('source_run')
        if not src:
            return None
        run = ROOT / src
    return None


def main():
    from gslam.dataset import Dataset
    from gslam.viz_stream import VizStream
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--ref', default=None, help='對照 run（畫成另一條路徑）；預設依 Track 取目前最佳')
    ap.add_argument('--step', type=int, default=2, help='每幾幀送一次畫面')
    ap.add_argument('--kf-every', type=int, default=10, help='每幾幀加一份點雲進地圖')
    ap.add_argument('--fps', type=float, default=30.0, help='重播速度（畫面幀／秒；0＝盡快）')
    ap.add_argument('--rviz', default='rviz:5600')
    ap.add_argument('--note', action='append', help='影像上多加一行說明（可重複）')
    ap.add_argument('--start', type=int, default=0, help='只播這一段（frame，含）')
    ap.add_argument('--end', type=int, default=None, help='只播到這一幀（不含）；預設到最後')
    a = ap.parse_args()
    run = (ROOT / a.run) if not Path(a.run).is_absolute() else Path(a.run)
    info = json.loads((run / 'test_info.json').read_text())
    track = '%02d' % int(info['track'])
    data = ROOT / ('data/data_VSLAM-T%d/Track%d/data' % (int(track), int(track)))
    ts, T = load(sorted(run.glob('result_*.csv'))[0])
    ds = Dataset(data)
    if list(ds.timestamps) != list(ts):
        raise SystemExit('run 的 timestamp 與資料不一致')

    a0, a1 = max(a.start, 0), min(a.end or len(T), len(T))
    ref_path = ROOT / (a.ref or DEFAULT_REF.get(track, ''))
    R_ref = None
    if a.ref or track in DEFAULT_REF:
        _, Tr = load(sorted(ref_path.glob('result_*.csv'))[0])
        s, R, t = umeyama(Tr[a0:a1, :3, 3], T[a0:a1, :3, 3])   # 只播一段時，只用這段對齊（看局部形狀差）
        R_ref = Tr.copy()
        R_ref[:, :3, :3] = R @ Tr[:, :3, :3]
        R_ref[:, :3, 3] = s * Tr[:, :3, 3] @ R.T + t

    if a0 > 0 or a1 < len(T):
        # 只播一段：顯示座標平移到這段起點（RViz 預設視角看原點附近），只影響顯示
        o = T[a0, :3, 3].copy()
        T = T.copy()
        T[:, :3, 3] -= o
        if R_ref is not None:
            R_ref[:, :3, 3] -= o
    hs = np.ones(len(T))
    g = 1.0
    if track in HEIGHT_FROM:
        hsrc = ROOT / HEIGHT_FROM[track]
        hs = np.array([float(r['depth_scale']) for r in csv.DictReader(open(hsrc / 'debug' / 'loger_height.csv'))])
        g = path_len(T) / path_len(load(sorted(hsrc.glob('result_*.csv'))[0])[1])
    elif R_ref is not None:
        g = 1.0
    E = None
    ef = find_edges(run)
    if ef is not None:
        E = dict(np.load(ef))
    pairs = np.zeros((0, 2), dtype=np.int64)
    if E is not None:
        pairs = np.concatenate([E['same_ij'], E['opp_ij']]).astype(np.int64)
        pairs = pairs[(pairs.min(1) >= a0) & (pairs.max(1) < a1)]

    viz = VizStream(a.rviz, ds.K, (ds.camera_info['width'], ds.camera_info['height']))
    vcfg = dict(max_depth_m=15.0, frame_point_step=2, keyframe_point_step=6)
    kf_idx = list(range(a0, a1, a.kf_every))
    print('重播 %s（%s，%d 幀，每 %d 幀一張畫面）；對照 %s；配對 %d（%s）' % (
        run.name, info.get('experiment'), len(T), a.step, ref_path.name if R_ref is not None else '無', len(pairs),
        ef.relative_to(ROOT) if ef is not None else '無'), flush=True)
    t_last = time.time()
    sent_kf = 0
    for i in range(a0, a1, a.step):
        xyz, rgb = ds.load_cloud(i)
        img = ds.load_image(i)
        sc = hs[i] * g
        vx, vc = viz.color_cloud(xyz, rgb, vcfg['max_depth_m'], vcfg['frame_point_step'])
        lines = ['%s  %s' % (run.name, info.get('experiment')), 'frame %d / %d' % (i, len(T) - 1)] + (a.note or [])
        ann = VizStream.annotate(img, None, None, None, None, lines) if img is not None else None
        viz.frame(dict(idx=i, tracking_status='replay'), T[i], R_ref[i] if R_ref is not None else T[i], img, ann,
                  vx * sc, vc)
        while sent_kf < len(kf_idx) and kf_idx[sent_kf] <= i:
            k = kf_idx[sent_kf]
            kx, kc = viz.color_cloud(*ds.load_cloud(k), vcfg['max_depth_m'], vcfg['keyframe_point_step'])
            viz.keyframe(sent_kf, kx * hs[k] * g, kc)
            sent_kf += 1
        if i % (a.step * 25) == 0:
            # 地圖與配對線：關鍵幀＝每 kf_every 幀；配對畫到已經走過的部分
            done = pairs[(pairs.max(1) <= i)] if len(pairs) else pairs
            viz.graph(T[kf_idx[:sent_kf]], np.minimum((done - a0) // a.kf_every, max(sent_kf - 1, 0)))
        if a.fps > 0:
            wait = 1.0 / a.fps - (time.time() - t_last)
            if wait > 0:
                time.sleep(wait)
            t_last = time.time()
    viz.graph(T[kf_idx], (pairs - a0) // a.kf_every)
    viz.final((R_ref if R_ref is not None else T)[a0:a1], T[a0:a1])
    print('重播完成：/vslam/path_optimized＝%s，/vslam/path_odometry＝對照 %s（Sim(3) 對齊）' % (
        run.name, ref_path.name if R_ref is not None else '（無）'), flush=True)
    _ = cv2


if __name__ == '__main__':
    main()
