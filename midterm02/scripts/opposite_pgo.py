#!/usr/bin/env python3
"""P1：反向重訪約束（後處理，接在 DROID D8 之後）。

觀察（2026-11-17，Track2，outputs/cache/dev61、dev62）：同一條路來回各經過一次時，D8 的相機高度差中位數 0.24 m、最大 0.81 m
（同方向重訪有閉環約束，只差 0.01 m）。反向看不到同一面，影像特徵接不起來，所以沒有閉環邊。
做法：兩側各取「駛近該點」的 45 幀 LoGeR 點雲（相機高度正規化），用目前位姿拼成子圖，GICP 量測兩次經過的相對位姿。
可信的分量（離線驗證）：
- 垂直：配準的垂直修正與 D8 高度差的相關 −0.96，修正後殘差中位數 0.07 m（≈ 路拱）。
- 橫向：相鄰配對一致（相鄰差 0.46 m）；深度比例誤差讓兩側點都沿各自視線往外推，橫向分量相同、相減抵消。
- 俯仰／側傾：由地面決定。
不可信：沿行進方向（兩側視線相反，深度比例誤差加倍成系統偏差，實測全部偏 +2 m）與航向，給很大的標準差（等於不約束）。
最後做逐幀 SE(3) pose graph；同方向閉環的邊保持 D8 目前的相對位姿（D8 已用稠密光流融合，見 main 內註解）。

    python scripts/opposite_pgo.py --from outputs/run/<D8 run>    （vslam 容器：需要 small_gicp）
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from create_run import create_run, digest  # noqa: E402
import droid_loops as dl  # noqa: E402
from run_droid import read_ply_xyz, PGO_SIGMAS  # noqa: E402

# 反向配對的選取：每 STEP 幀一個，找時間相隔 > MIN_GAP 幀、行進方向相反（夾角 > 150°）、水平距離 < MAX_DIST 的最近一幀
STEP, MIN_GAP, MAX_DIST, MIN_COS = 10, 300, 8.0, -0.87
# 子圖：駛近該點的 APPROACH 幀（每 STRIDE 幀取一張），只留該點 RADIUS 內、深度 < MAX_DEPTH、相機上方 < MAX_ABOVE 的點（去天空與樹冠）
APPROACH, STRIDE, RADIUS, MAX_DEPTH, MAX_ABOVE = 45, 3, 20.0, 30.0, 2.5
# 配準閘門：內點比例下限、修正量上限（超過＝配準跑掉）。small_gicp 的 converged 只代表迭代內收斂，不當閘門（留出驗證：不收斂的配對同樣一致）
MIN_INLIER_FRAC, MAX_ROT_DEG, MAX_VERT, MAX_LAT = 0.03, 5.0, 2.0, 4.0
# 邊的標準差（b 相機座標）：[俯仰 x, 航向 y, 側傾 z] rad、[橫向 x, 垂直 y, 前進 z] 單位
OPP_SIGMA = [0.01, 0.3, 0.01, 0.5, 0.1, 50.0]
# 逐幀 pose graph 的相鄰邊：DROID 的局部位移很準（逐幀平移的誤差遠小於 D8 keyframe 層級的 0.6／√單位），
# 大範圍漂移主要來自航向誤差累積（旋轉沿用 0.0022 rad／√單位：回程約 1000 單位偏斜約 4°）。
# 平移 0.05／√單位：100 單位的路段允許約 0.5% 的長度變化。這樣反向約束的雜訊不會讓軌跡局部扭動，修正以平滑方式分攤。
P1_SIGMAS = dict(PGO_SIGMAS, sigma_trans_per_unit=0.05)


def load_poses(fn):
    rows = list(csv.reader(open(fn)))[1:]
    return [r[0] for r in rows], np.array([[float(x) for x in r[1:]] for r in rows]).reshape(-1, 4, 4)


class Ctx:
    """子程序共用的資料（fork 繼承）。"""
    T = data = ts = hs = K0 = Kc = dist = None
    unit = 1.0


def _cloud(i):
    """第 i 幀的 LoGeR 點（相機高度正規化）→ 世界座標。

    LoGeR 點是用官方針孔 K0 從原始（有畸變）像素反投影的；DROID 位姿屬於自校正的相機（cy 差 11 px ≈ 俯仰 0.63°）。
    先把點投影回原始像素，用自校正的內參與畸變求真實光線，保留深度 z，點雲才和位姿同一個相機座標。
    反向兩側的俯仰誤差方向相反，不修正會變成約 1.3° 的系統相對俯仰。
    """
    x = read_ply_xyz(Ctx.data / 'raw_ply' / ('%s.ply' % Ctx.ts[i]))
    z = x[:, 2]
    keep = np.isfinite(z) & (z > 1.0) & (z < MAX_DEPTH)
    x, z = x[keep], z[keep]
    uv = (x @ Ctx.K0.T)[:, :2] / z[:, None]
    n = cv2.undistortPoints(uv.reshape(-1, 1, 2), Ctx.Kc, Ctx.dist).reshape(-1, 2)
    x = np.c_[n * z[:, None], z] * (Ctx.hs[i] * Ctx.unit)
    x = x[x[:, 1] > -MAX_ABOVE * Ctx.unit]
    return x @ Ctx.T[i, :3, :3].T + Ctx.T[i, :3, 3]


def _submap(c):
    P = np.concatenate([_cloud(i) for i in range(max(0, c - APPROACH), c + 1, STRIDE)])
    return P[np.linalg.norm(P - Ctx.T[c, :3, 3], axis=1) < RADIUS * Ctx.unit]


def _register(pair):
    import small_gicp
    a, b = pair
    ca = Ctx.T[a, :3, 3]
    A, B = _submap(a) - ca, _submap(b) - ca
    r = small_gicp.align(A, B, registration_type='GICP', downsampling_resolution=0.25, max_correspondence_distance=1.0,
                         num_threads=1, max_iterations=30)
    r = small_gicp.align(A, B, r.T_target_source, registration_type='GICP', downsampling_resolution=0.2,
                         max_correspondence_distance=0.6, num_threads=1, max_iterations=30)
    return a, b, r.T_target_source, r.num_inliers / max(len(B), 1), bool(r.converged)


def opposite_pairs(T):
    up = -T[:, :3, 1].mean(0)
    up /= np.linalg.norm(up)
    X = T[:, :3, 3]
    Hh = X - np.outer(X @ up, up)
    fwd = T[:, :3, 2] - np.outer(T[:, :3, 2] @ up, up)
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    idx = np.arange(APPROACH, len(T), STEP)
    D = np.linalg.norm(Hh[idx, None] - Hh[None, idx], axis=2)
    C = fwd[idx] @ fwd[idx].T
    pairs = []
    for k, a in enumerate(idx):
        m = (idx > a + MIN_GAP) & (C[k] < MIN_COS) & (D[k] < MAX_DIST * Ctx.unit)
        if m.any():
            pairs.append((int(a), int(idx[np.flatnonzero(m)[np.argmin(D[k][m])]])))
    return pairs, up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='src', required=True, help='DROID D8 的 run 目錄')
    ap.add_argument('--holdout', action='store_true', help='驗證用：只用偶數編號的反向配對當約束，奇數留著評估（正式結果不用）')
    a = ap.parse_args()
    src = (ROOT / a.src) if not Path(a.src).is_absolute() else Path(a.src)
    sinfo = json.loads((src / 'test_info.json').read_text())
    track = sinfo['track']
    data = ROOT / ('data/data_VSLAM-T%d/Track%d/data' % (int(track), int(track)))
    result_name = 'result_%d.csv' % int(track)
    ts, T = load_poses(src / result_name)
    hs = np.array([float(r['depth_scale']) for r in csv.DictReader(open(src / 'debug' / 'loger_height.csv'))])
    Ctx.T, Ctx.data, Ctx.ts, Ctx.hs = T, data, ts, hs
    # DROID 單位 vs 高度正規化 LoGeR 單位：深度先驗讓兩者接近（Track2 D8 實測 0.99–1.04），取 1
    Ctx.unit = 1.0

    run = create_run('test' if a.holdout else 'full', track)
    info_path = run / 'test_info.json'
    info = json.loads(info_path.read_text())
    t0 = time.time()

    # ---- 同方向閉環（與 D8 相同的 PnP 量測，幀對取自 D8 的 loop_measurements）
    ci = json.loads((data / 'camera_info.json').read_text())
    K0 = np.array([[ci['fx'], 0, ci['cx']], [0, ci['fy'], ci['cy']], [0, 0, 1.0]])
    calib = sinfo['parameters']['calibration']
    K = K0.copy()
    K[0, 2], K[1, 2] = calib['cx'], calib['cy']
    dist = np.array([calib['k1'], calib['k2'], 0, 0, 0.0])
    maps = cv2.initUndistortRectifyMap(K, dist, None, K, (ci['width'], ci['height']), cv2.CV_32FC1)
    Ctx.K0, Ctx.Kc, Ctx.dist = K0, K, dist

    def gray(f):
        img = cv2.remap(cv2.imread(str(data / 'images' / ('%s.jpg' % ts[f]))), maps[0], maps[1], cv2.INTER_LINEAR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    same, same_pnp = [], []
    for k, r in enumerate(csv.DictReader(open(src / 'debug' / 'loop_measurements.csv'))):
        if r['reason'] != 'ok':
            continue
        fi, fj = int(r['frame_a']), int(r['frame_b'])
        depth = dl.depth_cells(read_ply_xyz(data / 'raw_ply' / ('%s.ply' % ts[fj])), K0, K, dist, (ci['width'], ci['height'])) * hs[fj]
        Tm, st = dl.relative_pose_pnp(gray(fi), gray(fj), depth, K, seed=k)
        if Tm is not None:
            # 同方向閉環：D8 的 DROID 全域 BA 已用稠密光流把這些閉環融合（比 PnP 精細）。重新施加 PnP 量測會帶進雜訊
            # （Track1 實測 IMU 一致性 0.1178 → 0.1232），所以邊的量測取 D8 目前的相對位姿（保持現狀），只讓反向約束改變軌跡。
            # PnP 仍然量一次，存起來比較（debug/edges.npz 的 same_T_pnp）
            same_pnp.append(Tm)
            same.append((fi, fj, dl.inv(T[fi]) @ T[fj]))
    t1 = time.time()

    # ---- 反向重訪的點雲配準
    pairs, up = opposite_pairs(T)
    with Pool(12) as pool:
        regs = pool.map(_register, pairs, chunksize=2) if pairs else []
    rows, opp = [], []
    for k, (fa, fb, Tc, frac, conv) in enumerate(regs):
        ca = T[fa, :3, 3]
        S = np.eye(4); S[:3, 3] = ca
        Tw = S @ Tc @ np.linalg.inv(S)                 # 世界座標下「把 b 那側搬到 a 那側」的修正
        Rb = T[fb, :3, :3]
        loc = Rb.T @ Tc[:3, 3]                         # 修正量表示在 b 的相機座標：x 橫向、y 垂直、z 前進
        rot = float(np.degrees(np.arccos(np.clip((np.trace(Tc[:3, :3]) - 1) / 2, -1, 1))))
        ok = frac >= MIN_INLIER_FRAC and rot <= MAX_ROT_DEG and abs(loc[1]) <= MAX_VERT and abs(loc[0]) <= MAX_LAT
        use = ok and (not a.holdout or k % 2 == 0)
        rows.append(dict(frame_a=fa, frame_b=fb, inlier_frac=frac, converged=int(conv), rot_deg=rot, lateral=loc[0],
                         vertical=loc[1], along=loc[2], accepted=int(ok), used=int(use)))
        if use:
            Z = dl.inv(T[fa]) @ Tw @ T[fb]
            opp.append((fa, fb, Z, OPP_SIGMA))
    (run / 'debug').mkdir(exist_ok=True)
    with open(run / 'debug' / 'opposite_registration.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, ['frame_a', 'frame_b', 'inlier_frac', 'converged', 'rot_deg', 'lateral', 'vertical', 'along',
                               'accepted', 'used'])   # Track1 沒有反向重訪：只有表頭
        w.writeheader()
        w.writerows(rows)
    # 量測到的邊另存，供後續（F1 的 BA）直接使用：same＝同方向閉環（預設標準差），opp＝反向約束（各自的標準差）
    np.savez(run / 'debug' / 'edges.npz',
             same_ij=np.array([(i, j) for i, j, _ in same], dtype=np.int64).reshape(-1, 2),
             same_T=np.array([Tm for _, _, Tm in same]).reshape(-1, 4, 4), same_T_pnp=np.array(same_pnp).reshape(-1, 4, 4),
             opp_ij=np.array([(i, j) for i, j, _, _ in opp], dtype=np.int64).reshape(-1, 2),
             opp_T=np.array([Z for _, _, Z, _ in opp]).reshape(-1, 4, 4), opp_sigma=np.array(OPP_SIGMA),
             same_sigma=np.array([PGO_SIGMAS['loop_sigma_rot']] * 3 + [PGO_SIGMAS['loop_sigma_trans']] * 3))
    t2 = time.time()

    # ---- 逐幀 pose graph：相鄰幀（目前估計）＋同方向閉環＋反向約束
    T2, pg = dl.pose_graph(T, same + opp, **P1_SIGMAS, max_nfev=300)
    t3 = time.time()
    with open(run / result_name, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp'] + ['m%d%d' % (i, j) for i in range(4) for j in range(4)])
        for name, M in zip(ts, T2):
            w.writerow([name] + ['%.10g' % v for v in M.ravel()])
    with open(run / 'frame_manifest.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'timestamp_ns', 'image', 'ply'])
        for i, name in enumerate(ts):
            w.writerow([i, name, 'images/%s.jpg' % name, 'raw_ply/%s.ply' % name])
    chk = subprocess.run([sys.executable, str(ROOT / 'scripts/check_submission.py'), str(run / result_name),
                          str(data / 'sample_submission.csv')], capture_output=True, text=True)
    acc = [r for r in rows if r['accepted']]
    info.update(
        status='success' if chk.returncode == 0 else 'failed', experiment='P1_opposite_revisit' + ('_holdout' if a.holdout else ''),
        version='opposite_pgo-2', frame_count=len(ts),
        parameters=dict(source_run=str(src.relative_to(ROOT)), source_result_sha256=digest(src / result_name),
                        step=STEP, min_gap=MIN_GAP, max_dist=MAX_DIST, min_cos=MIN_COS, approach=APPROACH, stride=STRIDE,
                        radius=RADIUS, max_depth=MAX_DEPTH, max_above=MAX_ABOVE, min_inlier_frac=MIN_INLIER_FRAC,
                        max_rot_deg=MAX_ROT_DEG, max_vertical=MAX_VERT, max_lateral=MAX_LAT, opposite_sigma=OPP_SIGMA,
                        pgo_sigmas=P1_SIGMAS, unit=Ctx.unit, holdout=a.holdout, calibration=calib),
        timings_seconds=dict(same_direction_pnp=t1 - t0, opposite_registration=t2 - t1, pose_graph=t3 - t2),
        summary=dict(same_direction_loops=len(same), opposite_pairs=len(rows), opposite_accepted=len(acc), opposite_used=len(opp),
                     opposite_vertical_abs_median=float(np.median([abs(r['vertical']) for r in acc])) if acc else None,
                     opposite_lateral_abs_median=float(np.median([abs(r['lateral']) for r in acc])) if acc else None,
                     pose_graph=pg),
        submission=dict(check_returncode=chk.returncode, check_output=chk.stdout[-2000:] + chk.stderr[-2000:]))
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    subprocess.run([sys.executable, str(ROOT / 'scripts/analyze_external_run.py'), str(run)])
    print('opposite pairs %d accepted %d used %d; same-direction loops %d; pose graph %s' % (
        len(rows), len(acc), len(opp), len(same), json.dumps(pg)))
    print('run:', run)
    print('submission check:', 'OK' if chk.returncode == 0 else 'FAILED', chk.stdout[-300:], chk.stderr[-300:])


if __name__ == '__main__':
    main()
