"""Track03 候選：沒有 GNSS，只用「點雲對地圖」的逐幀配準 + 運動模型做 batch smoother（--gyro 時加 IMU 的相對航向）。
1. 初值：追蹤器（track_sequential）的相機軌跡 → 參考點（相機 + R(車體 yaw)·lever）。
2. 地圖量測：好幀的配準 (x, y, 車體 yaw)，資訊矩陣 = ICP 資訊 /c（c＝過度自信倍率），或 gated 模式（等向 σ）。
3. smoother：等速 + 非完整約束 + 航向加速度平滑（或 gyro 相對航向）+ 地圖量測（Cauchy + GNC）。
4. 輸出 Kaggle CSV；給 --ref 時順便算和參考軌跡的差（Track1 開發用：I_ns 當假真值）。"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sdcloc.bagio import load_sensors  # noqa: E402
from sdcloc.frames import load_frames  # noqa: E402
from sdcloc.smoother import BatchSmoother, SmootherConfig, wrap  # noqa: E402

IDX = [0, 1, 5]


def write_kaggle(path, x, y, yaw):
    # 格式照作者給的 Track03 diagnostic.csv：CRLF 換行、x/y 小數 9 位、yaw 11 位
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        f.write('idx,x,y,yaw\r\n')
        for i, (a, b, c) in enumerate(zip(x, y, yaw)):
            f.write(f'{i},{a:.9f},{b:.9f},{c:.11f}\r\n')


def to_ref(cam_xy, cam_yaw, lever, cam_yaw_offset_deg):
    veh_yaw = cam_yaw - np.deg2rad(cam_yaw_offset_deg)
    c_, s_ = np.cos(veh_yaw), np.sin(veh_yaw)
    ref = cam_xy + np.column_stack([c_ * lever[0] - s_ * lever[1], s_ * lever[0] + c_ * lever[1]])
    return ref, veh_yaw


def gyro_map_yaw(track_dir, frame_t, rotvec, ok, dt=0.0):
    """gyro 積分成「地圖 yaw」。車在地圖座標裡斜的時候 gz 不等於 atan2(R10, R00) 的變化率，用相機姿態 R
    （IMU 和 PLY 都是 x 前 z 上，當成同軸）換算；實測轉彎時比值 Track1 0.998、Track3 1.000，影響很小。
    dt：幀時間 t 取 IMU 在 t+dt 的值（相機對 IMU 的時間差）。"""
    imu = load_sensors(track_dir, need_gnss=False)['imu']
    ft = (frame_t - frame_t[0]) / 1e9
    it = (imu['t'] - frame_t[0]) / 1e9
    fk = ft[ok]
    R = Slerp(fk, Rotation.from_rotvec(rotvec[ok]))(np.clip(it, fk[0], fk[-1])).as_matrix()
    w = imu['gyro']
    d00 = R[:, 0, 1] * w[:, 2] - R[:, 0, 2] * w[:, 1]     # (R [ω]×)_00
    d10 = R[:, 1, 1] * w[:, 2] - R[:, 1, 2] * w[:, 1]     # (R [ω]×)_10
    rate = (R[:, 0, 0] * d10 - R[:, 1, 0] * d00) / (R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
    psi = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(it))])
    return np.interp(ft + dt, it, psi), {'imu_samples': int(len(it)), 'gz_vs_map_rate_ratio_turning':
                                     float(np.median(rate[np.abs(w[:, 2]) > np.deg2rad(3)] / w[np.abs(w[:, 2]) > np.deg2rad(3), 2]))
                                     if np.any(np.abs(w[:, 2]) > np.deg2rad(3)) else None}


def fit_gyro(ft, psi_g, yaw_obs, use, deg=3):
    """配準的車體 yaw − gyro 積分 = 多項式(t)（零偏會隨時間漂：Track1 從 0.34°/s 漂到 0.13°/s，線性修正後還彎 0.9°，
    三次剩 0.2°）＋c·psi_g（轉彎超過 45° 才估刻度）；Huber 1° 的 IRLS。回傳修正後的 gyro 航向。"""
    off = yaw_obs[use] - psi_g[use]
    off = off - 2 * np.pi * np.round((off - np.median(off)) / (2 * np.pi))
    span = float(np.ptp(psi_g[use]))
    tn = ft / ft[-1]                                  # 正規化時間，係數才不會差好幾個數量級
    cols = [tn[use] ** p for p in range(deg + 1)] + ([psi_g[use]] if span > np.deg2rad(45) else [])
    A = np.column_stack(cols)
    x, *_ = np.linalg.lstsq(A, off, rcond=None)
    for _ in range(10):
        r = np.abs(A @ x - off)
        w = np.where(r < np.deg2rad(1), 1.0, np.deg2rad(1) / r)
        x, *_ = np.linalg.lstsq(A * w[:, None], off * w, rcond=None)
    drift = sum(x[p] * tn ** p for p in range(1, deg + 1))
    scale = x[deg + 1] if len(x) > deg + 1 else 0.0
    corr = psi_g + drift + scale * psi_g
    resid = np.degrees(np.abs(A @ x - off))
    bias = np.degrees(np.gradient(drift, ft))
    return corr, {'bias_degps_start_end': [float(bias[0]), float(bias[-1])], 'scale_err': float(scale) if len(x) > deg + 1 else None,
                  'heading_span_deg': float(np.degrees(span)), 'resid_deg_median': float(np.median(resid)),
                  'resid_deg_p90': float(np.percentile(resid, 90))}


def imu_stationary(track_dir, frame_t, win_s=1.0, gyro_std_deg=0.2, acc_std=0.06, erode=15):
    """IMU 判定車子靜止的幀：前後 win_s/2 秒內 gyro 標準差 < gyro_std_deg（°/s）且加速度標準差 < acc_std（m/s²），
    頭尾再各內縮 erode 幀（剛停、剛起步那一下不算）。Track3 停車時 gyro 0.1°/s、加速度 0.01–0.03；行駛中 1°/s、0.3–0.7。"""
    imu = load_sensors(track_dir, need_gnss=False)['imu']
    it = (imu['t'] - frame_t[0]) / 1e9; ft = (frame_t - frame_t[0]) / 1e9
    sig = np.column_stack([np.degrees(imu['gyro']), imu['acc']])
    c1 = np.vstack([np.zeros(6), np.cumsum(sig, 0)]); c2 = np.vstack([np.zeros(6), np.cumsum(sig ** 2, 0)])
    a = np.searchsorted(it, ft - win_s / 2); b = np.searchsorted(it, ft + win_s / 2)
    cnt = np.maximum(b - a, 1)[:, None]
    std = np.sqrt(np.maximum((c2[b] - c2[a]) / cnt - ((c1[b] - c1[a]) / cnt) ** 2, 0))
    still = (b - a > 20) & (std[:, :3].max(1) < gyro_std_deg) & (std[:, 3:].max(1) < acc_std)
    if erode > 0:
        still = np.convolve(still.astype(float), np.ones(2 * erode + 1), 'same') >= 2 * erode + 1 - 1e-9
    return still


def estimate_gyro_dt(ft, head, yaw_obs, use, max_dt=0.3):
    """相機幀時間對 IMU 的時間差：轉彎時（|ω|>3°/s 前後 1 s）配準 yaw 和 gyro 航向 head(t+dt) 殘差最小的 dt。
    彎不夠（<100 幀）或比 dt=0 改善不到 0.1° 就回 0。Track3 的左轉給 +0.14 s（1.91°→1.53°）；Track1 彎太緩量不出來。"""
    rate = np.degrees(np.abs(np.gradient(head, ft)))
    m = use & (np.convolve((rate > 3).astype(float), np.ones(57), 'same') > 0)
    if m.sum() < 100:
        return 0.0, {'turning_frames': int(m.sum())}
    taus = np.arange(-max_dt, max_dt + 1e-9, 0.02)
    rms = []
    for tau in taus:
        r = wrap(yaw_obs[m] - np.interp(ft[m] + tau, ft, head))
        r = wrap(r - np.angle(np.mean(np.exp(1j * r))))
        rms.append(np.sqrt(np.mean(np.minimum(np.abs(r), np.deg2rad(5)) ** 2)))
    rms = np.degrees(rms); i = int(np.argmin(rms)); i0 = int(np.argmin(np.abs(taus)))
    info = {'turning_frames': int(m.sum()), 'best_dt_s': float(taus[i]), 'rms_deg_best': float(rms[i]), 'rms_deg_at_0': float(rms[i0])}
    return (float(taus[i]) if rms[i0] - rms[i] >= 0.1 else 0.0), info


def yaw_offset_from_motion(ft, reg, good, seg_m=100.0, max_turn_deg=20.0):
    """相機 yaw − 車體 yaw：每 ~seg_m 公尺路程比「相機 yaw 沿路積分的位移方向」和「實際位移方向」，取各段中位數。
    長基線讓側向位置誤差影響 <0.1°，換車道也會反映在相機 yaw 裡。Track1 這樣估 −0.42～−0.49°，GNSS 估 −0.4455°。
    段內轉超過 max_turn_deg 的不用（大轉彎時相機不在後軸上，行進方向會偏離車頭，Track3 的 92° 左轉段給 −9°）。"""
    k = np.flatnonzero(good)
    xy = reg['t'][k, :2]; yaw = reg['yaw'][k]
    step = np.hypot(*np.diff(xy, axis=0).T)
    sm = np.convolve(np.r_[step, step[-1]], np.ones(31) / 31, mode='same')     # 停車時的抖動不要累積成路程
    v = sm / np.r_[np.diff(ft[k]), np.median(np.diff(ft[k]))]
    ds = np.where(v > 2.0, sm, 0.0)
    s = np.cumsum(ds)
    segs, a = [], 0
    while True:
        b = int(np.searchsorted(s, s[a] + seg_m))
        if b >= len(k):
            break
        if np.degrees(np.ptp(np.unwrap(yaw[a:b]))) <= max_turn_deg:
            pred = np.sum(ds[a:b] * np.exp(1j * yaw[a:b]))
            d = xy[b] - xy[a]
            segs.append(float(np.degrees(wrap(np.angle(pred) - np.arctan2(d[1], d[0])))))
        a = b
    return (float(np.median(segs)) if segs else None), segs


def compare(x, y, yaw, ref_csv, public_frac=0.3):
    ref = pl.read_csv(ref_csv).to_numpy()
    n = min(len(ref), len(x))
    d = np.hypot(x[:n] - ref[:n, 1], y[:n] - ref[:n, 2])
    dy = np.degrees(wrap(yaw[:n] - ref[:n, 3]))
    out = {}
    for name, sl in (('all', slice(0, n)), ('public_first30', slice(0, int(n * public_frac))), ('private_rest', slice(int(n * public_frac), n))):
        dd, yy = d[sl], dy[sl]
        out[name] = {'xy_rmse': float(np.sqrt(np.mean(dd ** 2))), 'xy_median': float(np.median(dd)), 'xy_p90': float(np.percentile(dd, 90)),
                     'xy_max': float(dd.max()), 'yaw_rmse_deg': float(np.sqrt(np.mean(yy ** 2))), 'yaw_median_deg': float(np.median(np.abs(yy))),
                     'score_like': float(np.sqrt(np.mean(dd ** 2)) + 0.2 * np.sqrt(np.mean(yy ** 2)))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('track_dir', type=Path, help='含 raw_ply 的資料夾（只用來拿幀時間）')
    ap.add_argument('--reg', type=Path, required=True, help='配準 npz（track_sequential 或 register_all 的輸出）')
    ap.add_argument('--init', type=Path, help='初值軌跡 npz（預設＝--reg；失敗幀用內插補）')
    ap.add_argument('--lever', default='0.7086,-0.5995')
    ap.add_argument('--cam-yaw-offset-deg', default='-0.4455',
                    help='相機 yaw − 車體 yaw（度）；auto＝用配準軌跡的長基線行進方向估（沒有 GNSS 可校時用）')
    ap.add_argument('--info-mode', choices=('icp', 'gated'), default='icp')
    ap.add_argument('--overconf', type=float, default=100.0, help='icp 模式：ICP 資訊 /c（Track1 對 I_ns 掃過：100 比 20／50 好）')
    ap.add_argument('--sigma-map', type=float, default=0.3, help='gated 模式：等向 σ（m）')
    ap.add_argument('--sigma-map-yaw-deg', type=float, default=1.0)
    ap.add_argument('--min-ng03', type=float, default=0.4)
    ap.add_argument('--min-inlier', type=float, default=0.5)
    ap.add_argument('--min-xy-info', type=float, default=10.0, help='gated 模式：xy 最小資訊門檻（每 (0.1 m)²）')
    ap.add_argument('--sigma-lat', type=float, default=0.01)
    ap.add_argument('--sigma-acc', type=float, default=1.0)
    ap.add_argument('--sigma-yaw-acc-deg', type=float, default=None,
                    help='航向加速度 σ（度／幀²）；沒 gyro 預設 0.003（比 0.01／0.03／0.1 的 yaw RMSE 都低），有 gyro 預設關（0）')
    ap.add_argument('--gyro', action='store_true', help='加 IMU gyro 的相對航向（Track03 的 bag 有 IMU、沒有 GNSS）')
    ap.add_argument('--sigma-dpsi-deg', type=float, default=0.01, help='gyro 相鄰幀航向差的 σ（度）')
    ap.add_argument('--sigma-dpsi-long-deg', type=float, default=0.05, help='gyro 30 幀長基線的 σ（度）；0＝不用')
    ap.add_argument('--gyro-poly', type=int, default=3, help='gyro 零偏漂移的多項式次數（對配準 yaw 擬合）')
    ap.add_argument('--gyro-dt', default='0', help='幀時間 t 取 IMU 在 t+dt 的值（秒）；auto＝用轉彎段估（Track3 +0.14 s）')
    ap.add_argument('--zupt', action='store_true', help='IMU 判定靜止的幀加「位置、航向不變」的約束（Track3 停車 42 秒）')
    ap.add_argument('--nhc-lever', type=float, default=0.0,
                    help='參考點在後軸前方幾公尺（轉彎時允許側滑 d·Δψ）；Track3 左轉估約 2 m（相機約在後軸前 1.4–2.2 m）')
    ap.add_argument('--map-loss', choices=('huber', 'cauchy', 'gm'), default='cauchy')
    ap.add_argument('--map-sigma', type=float, default=3.0, help='穩健損失的 Mahalanobis 尺度')
    ap.add_argument('--gnc', default='10,3,1', help='Cauchy/GM 的 GNC 尺度倍率序列')
    ap.add_argument('--iters', type=int, default=8)
    ap.add_argument('--ref', type=Path, help='比對用的參考 CSV（例如 I_ns）')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    lever = np.array([float(v) for v in a.lever.split(',')])
    frame_t, _ = load_frames(a.track_dir / 'raw_ply')
    ft = (frame_t - frame_t[0]) / 1e9
    N = len(ft)
    reg = np.load(a.reg)
    init = np.load(a.init) if a.init else reg
    assert len(reg['ok']) == N and len(init['ok']) == N, 'npz 幀數與 raw_ply 不符'
    offset_info = None
    if a.cam_yaw_offset_deg == 'auto':
        okr = reg['ok'] & (reg['ng_frac03'] >= a.min_ng03) & (reg['inlier_frac'] >= a.min_inlier)
        off, segs = yaw_offset_from_motion(ft, reg, okr)
        if off is None:
            raise SystemExit('auto 偏角：行駛路程不到一段，無法估')
        a.cam_yaw_offset_deg = off
        offset_info = {'auto_deg': off, 'segments_deg': [round(v, 3) for v in segs]}
    a.cam_yaw_offset_deg = float(a.cam_yaw_offset_deg)

    # ---- 初值：追蹤軌跡（失敗幀內插）→ 參考點 ----
    iok = init['ok'] & np.isfinite(init['t'][:, 0])
    if not iok.all():
        tt = np.arange(N)
        init_xy = np.column_stack([np.interp(tt, tt[iok], init['t'][iok, d]) for d in range(2)])
        yaw_u = np.unwrap(init['yaw'][iok])
        init_yaw = np.interp(tt, tt[iok], yaw_u)
    else:
        init_xy = init['t'][:, :2].copy(); init_yaw = np.unwrap(init['yaw'])
    init_ref, init_veh_yaw = to_ref(init_xy, init_yaw, lever, a.cam_yaw_offset_deg)

    # ---- 地圖量測 ----
    good = reg['ok'] & (reg['ng_frac03'] >= a.min_ng03) & (reg['inlier_frac'] >= a.min_inlier)
    cam_xy = reg['t'][:, :2]
    ref_xy, veh_yaw = to_ref(cam_xy, reg['yaw'], lever, a.cam_yaw_offset_deg)
    obs_frames, obs_xy, obs_yaw, obs_info = [], [], [], []
    sigy2 = np.deg2rad(a.sigma_map_yaw_deg) ** 2
    n_xy_one = 0
    for k in np.flatnonzero(good):
        if a.info_mode == 'icp':
            I = reg['information'][k][np.ix_(IDX, IDX)] / a.overconf
        else:
            I = np.zeros((3, 3))
            H = reg['information'][k][:2, :2] / 100.0          # 每 (0.1 m)²
            w, v = np.linalg.eigh(H)
            if w[0] >= a.min_xy_info:
                I[:2, :2] = np.eye(2) / a.sigma_map ** 2
            elif w[1] >= a.min_xy_info:
                I[:2, :2] = np.outer(v[:, 1], v[:, 1]) / a.sigma_map ** 2; n_xy_one += 1
            if reg['yaw_info'][k] >= 10:
                I[2, 2] = 1 / sigy2
        if not np.isfinite(I).all() or np.trace(I) <= 0:
            continue
        obs_frames.append(k); obs_xy.append(ref_xy[k]); obs_yaw.append(veh_yaw[k]); obs_info.append(I)
    obs_frames = np.array(obs_frames); obs_xy = np.array(obs_xy); obs_yaw = np.array(obs_yaw); obs_info = np.array(obs_info)

    yaw_acc = a.sigma_yaw_acc_deg if a.sigma_yaw_acc_deg is not None else (0.0 if a.gyro else 0.003)
    gyro_yaw = np.zeros(N); gyro_info = None
    if a.gyro:
        rok = reg['ok'] & np.isfinite(reg['rotvec'][:, 0])
        psi_g, gi = gyro_map_yaw(a.track_dir, frame_t, reg['rotvec'], rok)
        yaw_use = good & (reg['yaw_info'] >= 10)
        if yaw_use.sum() < 50:                      # yaw 資訊整體偏低（例如配準有加權）就退回用全部好幀
            yaw_use = good
        yaw_obs = np.where(rok, veh_yaw, 0.0)
        gyro_yaw, gyro_info = fit_gyro(ft, psi_g, yaw_obs, yaw_use, a.gyro_poly)
        dt_info = None
        if a.gyro_dt == 'auto':
            dt, dt_info = estimate_gyro_dt(ft, gyro_yaw, yaw_obs, yaw_use)
        else:
            dt = float(a.gyro_dt)
        if dt != 0.0:
            psi_g, gi = gyro_map_yaw(a.track_dir, frame_t, reg['rotvec'], rok, dt)
            gyro_yaw, gyro_info = fit_gyro(ft, psi_g, yaw_obs, yaw_use, a.gyro_poly)
        gyro_info.update(gi)
        gyro_info.update({'dt_s': dt, 'dt_auto': dt_info})
    cfg = SmootherConfig(use_gnss=False, use_imu=a.gyro, imu_abs=False, sigma_yaw_acc_rad=np.deg2rad(yaw_acc),
                         sigma_dpsi_rad=np.deg2rad(a.sigma_dpsi_deg), sigma_dpsi_long_rad=np.deg2rad(a.sigma_dpsi_long_deg), long_stride=30,
                         sigma_lat_m=a.sigma_lat, nhc_lever_m=a.nhc_lever, sigma_acc_mps2=a.sigma_acc, map_loss=a.map_loss, map_huber_sigma=a.map_sigma,
                         iters=a.iters)
    sm = BatchSmoother(ft, np.zeros(0), np.zeros((0, 2)), gyro_yaw, cfg, rel_yaw_at_frames=gyro_yaw)
    sm.set_initial(np.column_stack([init_ref, init_veh_yaw]))
    sm.set_map_observations(obs_frames, obs_xy, obs_yaw, obs_info)
    zupt_info = None
    if a.zupt:
        still = imu_stationary(a.track_dir, frame_t)
        sm.set_stationary(still)
        edges = np.flatnonzero(np.diff(np.r_[False, still, False]))
        zupt_info = {'frames': int(still.sum()), 'runs': [(int(s0), int(e0 - 1)) for s0, e0 in zip(edges[::2], edges[1::2])]}
    x0 = None
    scales = [float(v) for v in a.gnc.split(',')] if a.map_loss in ('cauchy', 'gm') else [1.0]
    for kk in scales:
        sm.cfg.map_huber_sigma = a.map_sigma * kk
        r = sm.solve(x0=x0)
        x0 = r['state']
    x, y, yaw = r['x'], r['y'], wrap(r['yaw'])
    write_kaggle(a.out, x, y, yaw)
    info = {'frames': int(N), 'good': int(good.sum()), 'map_obs': int(len(obs_frames)), 'xy_one_dir': n_xy_one,
            'init_vs_out_rms_m': float(np.sqrt(np.mean((x - init_ref[:, 0]) ** 2 + (y - init_ref[:, 1]) ** 2))),
            'reg_vs_out_median_m': float(np.median(np.hypot(x[good] - ref_xy[good, 0], y[good] - ref_xy[good, 1]))),
            'chi2': float(r['chi2_hist'][-1]), 'gyro': gyro_info, 'cam_yaw_offset_deg': a.cam_yaw_offset_deg, 'cam_yaw_offset_auto': offset_info, 'zupt': zupt_info}
    if a.ref:
        info['vs_ref'] = compare(x, y, yaw, a.ref)
        info['init_vs_ref'] = compare(init_ref[:, 0], init_ref[:, 1], wrap(init_veh_yaw), a.ref)
    print(json.dumps(info, indent=1))
    a.out.with_suffix('.diag.json').write_text(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
