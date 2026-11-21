#!/usr/bin/env python3
"""F1：DROID 前端＋gslam 全解析度 SIFT BA 後端（含相機自校正）。

為什麼：DROID 的稠密光流在 0.5 倍解析度（720×464）上估計，對停車、橫越車輛等動態場景很穩，大範圍形狀也已經由
D8（深度先驗＋閉環）與 P1（反向重訪約束）處理好；gslam 的 BA 則在原解析度 1440×928 用 SIFT 特徵軌跡（含跨圈軌跡）
做重投影最佳化，局部精度較高，但 E21 自己的里程計初值在 Track2 會壞掉。
F1 把兩者接起來：BA 初值與相鄰幀邊（鬆，只固定尺度、撐住特徵少的幀）來自 P1；同方向閉環與反向約束也以邊的形式保留，
避免 BA 只看局部重投影時把 P1 修好的大範圍關係放掉。SIFT 特徵與匹配直接用某個 gslam run 存下的 BA 輸入（不重抽）。
--experiment 的 bundle.learned 開啟時（例如 E28_selfcalib_height_learned），在同一組影像對上另抽 SuperPoint＋LightGlue 軌跡一起進 BA。

    python scripts/droid_ba.py --from outputs/run/<P1 run> --ba-input outputs/run/<gslam run> [--experiment E27_selfcalib_height]

--low-parallax-depth（droid_ba-8）：從根本處理同一件事。視差 < 1° 的軌跡不丟掉：用 LoGeR 深度（gslam 的 PixelCloud 查表）放初值，
深度因子（bundle.depth 的設定，含逐幀 LoGeR 尺度 s_i）只加在這些軌跡上，其他軌跡完全照舊。靜止／剛起步的幀因此有自己的 SIFT 觀測，
不必借 DROID 的邊（Track1 E18 對所有觀測加深度因子曾讓 IMU 殘差變差，所以這裡只加在原本會被丟掉的軌跡上）。

--weak-odom（droid_ba-7，有效版本）：--gap-odom 只換缺配對的邊沒有效（track02_full_2611200056，折點移到旁邊的幀）。
真正的原因是 BA 會篩掉視差 < 1° 的軌跡：靜止／剛起步的幀即使有上千個配對，篩選後與相鄰幀共同的軌跡數是 0
（Track2 恰好 21 對：1–7、39、1654–1666；門檻 1～20 都是同一組），這些幀在 BA 裡只剩鬆的相鄰幀邊。
--weak-odom 把「BA 篩選後共同軌跡數為 0」的相鄰幀邊改用 DROID 精度的標準差，其他不變。原來的 SIFT 配對其實與 DROID 的相對位姿吻合
（對極誤差中位數 0.04–0.07 px），F1 的位姿卻是 1.2–2.4 px（dev75i）。

--gap-odom（droid_ba-6）：BA 輸入缺相鄰幀 SIFT 配對的地方（Track2：起步 0–38、停車後起步 1653–1666），BA 圖在 SIFT 這一層斷開，
只剩鬆的相鄰幀邊（0.01 rad、0.05）撐著，整個 BA 的張力（閉環、反向約束）就集中在這裡折彎（F1 在 frame 10 折 2.5°、
1657 折 1.6°，DROID 與學習式特徵版都沒有）。這些相鄰幀改用標準差等於「DROID 相鄰相對位姿實際誤差」的邊：
用有配對處 F1 與 P1 相鄰相對位姿的差估計（dev75g：旋轉每軸 std ≤ 3.4e-4 rad、平移 ≤ 0.0085），取最大軸、各軸相同。

F2（再一輪）：F1 的 BA 把 P1 修好的反向垂直一致性放鬆了（重量殘差中位數 0.026 → 0.051 m），而且相機校正也變了
（cy 452.9 → 454.7 px，P1 的量測用的是舊校正）。用 F1 的位姿與校正重量反向約束，再做一次 BA：

    python scripts/droid_ba.py --from outputs/run/<F1 run> --ba-input outputs/run/<gslam run> --remeasure-opposite \
        --edges-from outputs/run/<P1 run> --height-from outputs/run/<D8 run>
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
# 缺 SIFT 配對處的相鄰幀邊標準差（旋轉 rad、平移＝位姿單位）。來源：Track2 有配對的 4194 個相鄰幀，
# F1（track02_full_2611172344）與它的輸入 P1（…2611172340）相鄰相對位姿差的每軸 std，取最大軸（outputs/cache/dev75g_odo_noise.py）
GAP_ODOM_SIGMA = (3.4e-4, 0.0085)
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
    ap.add_argument('--remeasure-opposite', action='store_true',
                    help='F2：用 --from 的位姿與它的 BA 自校正重新量反向重訪約束（否則用 --edges-from 的 edges.npz）')
    ap.add_argument('--edges-from', default=None, help='edges.npz 所在的 run（預設 --from）；remeasure 時只取同方向閉環的幀對')
    ap.add_argument('--height-from', default=None, help='remeasure 用：有 debug/loger_height.csv 的 DROID run（D8）')
    ap.add_argument('--holdout', action='store_true', help='remeasure 驗證用：只用偶數編號的反向配對（產生 test run）')
    ap.add_argument('--opp-sigma-scale', type=float, default=1.0,
                    help='交叉驗證用：反向邊「可信分量」（俯仰、側傾、橫向、垂直）的標準差乘這個倍數；航向與沿線不變')
    ap.add_argument('--gap-odom', action='store_true',
                    help='（無效，留作紀錄）只把缺相鄰 SIFT 配對的相鄰幀邊改用 DROID 實際精度的標準差（GAP_ODOM_SIGMA）')
    ap.add_argument('--low-parallax-depth', action='store_true',
                    help='視差 < min_parallax_deg 的軌跡不丟：用 LoGeR 深度放初值並只對它們加深度因子（droid_ba-8）')
    ap.add_argument('--weak-odom', action='store_true',
                    help='BA 篩選後兩幀沒有任何共同軌跡的相鄰幀邊，改用 DROID 實際精度的標準差（GAP_ODOM_SIGMA）')
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

    # --from 是 test run（例如 P1 的 --holdout）時，結果也是 test run
    run = create_run('test' if a.holdout or '_test_' in src.name else 'full', track)
    info_path = run / 'test_info.json'
    info = json.loads(info_path.read_text())
    t0 = time.time()
    K, _, kpts, links, _ = bundle.load_input(bsrc / 'debug' / 'bundle_input.npz')
    if len(kpts) != len(T):
        raise SystemExit('BA 輸入 %d 幀與位姿 %d 幀不一致' % (len(kpts), len(T)))
    learned = None
    lcfg = cfg.get('learned') or {}
    if lcfg.get('enabled'):
        # 學習式特徵軌跡（gslam/learned_tracks.py）：在 BA 輸入的同一組影像對（相鄰幀＋非相鄰的閉環／跨圈對）上
        # 抽 SuperPoint、LightGlue 配對，與 SIFT 軌跡合併（combine_sift）或取代
        import cv2
        from gslam import learned_tracks
        extra = sorted({(int(x), int(y)) for x, y, _ in links if y - x > 1})
        img = lambda j: cv2.imread(str(data / 'images' / ('%s.jpg' % ts[j])), cv2.IMREAD_COLOR)
        lk, ll, learned = learned_tracks.build(img, len(T), extra, K, lcfg, seed=p['seed'],
                                               log=lambda m: print(m, flush=True))
        kpts, links = learned_tracks.merge(kpts, links, lk, ll) if lcfg['combine_sift'] else (lk, ll)
    esrc = src if a.edges_from is None else ((ROOT / a.edges_from) if not Path(a.edges_from).is_absolute() else Path(a.edges_from))
    E = dict(np.load(esrc / 'debug' / 'edges.npz'))
    remeasure = None
    if a.remeasure_opposite:
        # F2：F1 的 BA 讓位姿與相機校正都變了（cy 452.9 → 454.7 px），用新的位姿拼子圖、新的校正修正 LoGeR 光線，重量反向約束；
        # 同方向閉環的邊改成「保持目前位姿的相對關係」（與 P1 v2 同理）
        import opposite_pgo as op
        hsrc = (ROOT / a.height_from) if not Path(a.height_from).is_absolute() else Path(a.height_from)
        calib = sinfo['summary']['bundle']['calibration']
        ci = json.loads((data / 'camera_info.json').read_text())
        K0 = np.array([[ci['fx'], 0, ci['cx']], [0, ci['fy'], ci['cy']], [0, 0, 1.0]])
        Kc = K0.copy()
        Kc[0, 2], Kc[1, 2] = calib['cx'], calib['cy']
        op.Ctx.data, op.Ctx.ts, op.Ctx.K0, op.Ctx.Kc = data, ts, K0, Kc
        op.Ctx.dist = np.array([calib['k1'], calib['k2'], 0, 0, 0.0])
        op.Ctx.hs = np.array([float(r['depth_scale']) for r in csv.DictReader(open(hsrc / 'debug' / 'loger_height.csv'))])
        rows, opp = op.measure_opposite(T, a.holdout)
        (run / 'debug').mkdir(exist_ok=True)
        op.write_rows(run / 'debug' / 'opposite_registration.csv', rows)
        E['same_T'] = np.array([np.linalg.inv(T[i]) @ T[j] for i, j in E['same_ij']]).reshape(-1, 4, 4)
        E['opp_ij'] = np.array([(i, j) for i, j, _, _ in opp], dtype=np.int64).reshape(-1, 2)
        E['opp_T'] = np.array([Z for _, _, Z, _ in opp]).reshape(-1, 4, 4)
        np.savez(run / 'debug' / 'edges.npz', **E)
        acc = [r for r in rows if r['accepted']]
        remeasure = dict(height_from=str(hsrc.relative_to(ROOT)), calibration=calib, pairs=len(rows), accepted=len(acc),
                         used=len(opp), holdout=a.holdout,
                         vertical_abs_median=float(np.median([abs(r['vertical']) for r in acc])) if acc else None,
                         lateral_abs_median=float(np.median([abs(r['lateral']) for r in acc])) if acc else None)
        print('remeasured opposite: %s' % json.dumps(remeasure), flush=True)
    sig = lambda r, t: np.array([r, r, r, t, t, t], dtype=np.float64)
    odo = gtsam.noiseModel.Diagonal.Sigmas(sig(cfg['odom_rot_sigma_rad'], cfg['odom_trans_sigma_m']))
    robust = lambda s: gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(g['huber_k']),
                                                      gtsam.noiseModel.Diagonal.Sigmas(np.asarray(s, dtype=np.float64)))
    between = [(i - 1, i, np.linalg.inv(T[i - 1]) @ T[i], odo) for i in range(1, len(T))]
    depths, lp_info = None, None
    if a.low_parallax_depth:
        # 每個 SIFT 特徵在該幀 LoGeR 點雲的深度（相機 z；與 gslam pipeline 相同的查表：PixelCloud）
        from gslam.dataset import Dataset
        from gslam import visual as vis_mod
        vcfg = p['visual']
        ds = Dataset(data)
        cache = run / 'debug' / 'feature_depths.npz'
        (run / 'debug').mkdir(exist_ok=True)
        depths = []
        for i in range(len(T)):
            xyz, _ = ds.load_cloud(i)
            pix = vis_mod.PixelCloud(xyz, ds.K, ds.size, vcfg['assoc_stride_px'], vcfg['max_depth_m'])
            pts, ok = pix.lookup(kpts[i])
            depths.append(np.where(ok, pts[:, 2], np.nan))
        np.savez_compressed(cache, **{'f%d' % i: d for i, d in enumerate(depths)})
        cfg = dict(cfg, low_parallax_depth=dict(enabled=True))
        nd = np.concatenate(depths)
        lp_info = dict(features=int(len(nd)), with_depth=int(np.sum(np.isfinite(nd))), max_depth_m=vcfg['max_depth_m'],
                       assoc_stride_px=vcfg['assoc_stride_px'], depth_cfg=cfg.get('depth'))
        print('low-parallax depth: %d / %d 個特徵有 LoGeR 深度' % (lp_info['with_depth'], lp_info['features']), flush=True)
    gaps = []
    if a.weak_odom:
        support = bundle.pair_support(K, list(T), kpts, links, cfg, depths)   # 有 --low-parallax-depth 時，算的是保留低視差軌跡之後
        gaps = [i for i in range(1, len(T)) if support[i] == 0]
    elif a.gap_odom:
        have = {(int(x), int(y)) for x, y, _ in links}
        gaps = [i for i in range(1, len(T)) if (i - 1, i) not in have]
    if gaps:
        tight = gtsam.noiseModel.Diagonal.Sigmas(sig(*GAP_ODOM_SIGMA))
        gs = set(gaps)
        between = [(x, y, Z, tight if (y == x + 1 and y in gs) else n) for x, y, Z, n in between]   # 只換相鄰幀邊
        print('%s odometry: %d 個相鄰幀%s，改用 σ %s：%s' % ('weak' if a.weak_odom else 'gap', len(gaps),
              '在 BA 篩選後沒有共同軌跡' if a.weak_odom else '沒有 SIFT 配對', GAP_ODOM_SIGMA, gaps), flush=True)
    same_noise = robust(E['same_sigma'])
    between += [(int(i), int(j), Z, same_noise) for (i, j), Z in zip(E['same_ij'], E['same_T'])]
    # 反向邊：P1 所有邊共用一組標準差（1 維）；P2 每條邊各自一組（2 維，有沿線量測的邊沿線標準差較小）
    opp_sigma = np.asarray(E['opp_sigma'], dtype=np.float64).copy()
    # Pose3 切空間順序 [rx, ry, rz, tx, ty, tz]（相機座標：x 右、y 下、z 前）→ 可信：rx 俯仰、rz 側傾、tx 橫向、ty 垂直
    opp_sigma[..., [0, 2, 3, 4]] *= a.opp_sigma_scale
    if opp_sigma.ndim == 1:
        opp_noise = robust(opp_sigma)
        between += [(int(i), int(j), Z, opp_noise) for (i, j), Z in zip(E['opp_ij'], E['opp_T'])]
    else:
        between += [(int(i), int(j), Z, robust(s)) for (i, j), Z, s in zip(E['opp_ij'], E['opp_T'], opp_sigma)]
    out, binfo = bundle.run(K, list(T), kpts, links, between, cfg, depths=depths)
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
                experiment=('F2_droid_sift_ba_remeasured' if a.remeasure_opposite else 'F1_droid_sift_ba') + ('_learned' if learned else '') + ('_weakodom' if a.weak_odom else ('_gapodom' if a.gap_odom else '')) + ('_lpdepth' if a.low_parallax_depth else '')
                + ('_holdout' if a.holdout else ''),
                version='droid_ba-8', frame_count=len(ts),
                parameters=dict(source_run=str(src.relative_to(ROOT)), source_result_sha256=digest(src / result_name),
                                ba_input_run=str(bsrc.relative_to(ROOT)),
                                ba_input_sha256=digest(bsrc / 'debug' / 'bundle_input.npz'),
                                edges_from=str(esrc.relative_to(ROOT)), remeasure_opposite=remeasure, opp_sigma_scale=a.opp_sigma_scale,
                                gap_odom=dict(sigma=list(GAP_ODOM_SIGMA), frames=gaps, rule='weak' if a.weak_odom else 'gap') if gaps else None,
                                low_parallax_depth=lp_info,
                                bundle_params_from=a.experiment, bundle=cfg, graph=g),
                timings_seconds=dict(bundle=t1 - t0),
                summary=dict(edges=dict(odometry=len(T) - 1, same_direction=int(len(E['same_ij'])), opposite=int(len(E['opp_ij']))),
                             bundle=binfo, learned=learned),
                submission=dict(check_returncode=chk.returncode, check_output=chk.stdout[-2000:] + chk.stderr[-2000:]))
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2, default=float) + '\n')
    subprocess.run([sys.executable, str(ROOT / 'scripts/analyze_external_run.py'), str(run)])
    print('run:', run)
    print('submission check:', 'OK' if chk.returncode == 0 else 'FAILED', chk.stdout[-300:], chk.stderr[-300:])


if __name__ == '__main__':
    main()
