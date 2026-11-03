import numpy as np

from sdcloc.smoother import BatchSmoother, SmootherConfig


def synth(rng, tau_true=0.0, bias_deg=3.0, n_frames=600, fps=27.8, speed=10.0):
    """直線 → 轉彎 → 直線的真實軌跡，GNSS 4 Hz 帶 0.2 m 雜訊與延遲 tau，IMU yaw 帶常數偏差。"""
    t = np.arange(n_frames) / fps
    yaw_rate = np.where((t > 7) & (t < 12), np.deg2rad(8.0), 0.0)
    psi = np.cumsum(yaw_rate / fps)
    x = np.cumsum(speed * np.cos(psi) / fps)
    y = np.cumsum(speed * np.sin(psi) / fps)
    gt_t = np.arange(0.1, t[-1] - 0.1, 0.25)
    gx = np.interp(gt_t, t, x) + rng.normal(0, .2, len(gt_t))
    gy = np.interp(gt_t, t, y) + rng.normal(0, .2, len(gt_t))
    # GNSS 落後 tau：真位置在 t 的樣本被標成 t + tau... 這裡定義 tau 為「要加到 GNSS 時間戳上的量」，所以標記時間 = t - tau
    imu_yaw = psi - np.deg2rad(bias_deg) + rng.normal(0, np.deg2rad(.02), n_frames).cumsum() * 0
    return t, x, y, psi, gt_t - tau_true, np.column_stack([gx, gy]), imu_yaw


def test_smoother_recovers_trajectory():
    rng = np.random.default_rng(0)
    t, x, y, psi, gt_t, gxy, imu_yaw = synth(rng)
    sm = BatchSmoother(t, gt_t, gxy, imu_yaw)
    r = sm.solve()
    err = np.hypot(r['x'] - x, r['y'] - y)
    assert np.sqrt(np.mean(err ** 2)) < 0.12          # 4 Hz、0.2 m 雜訊、σ_acc=3 → 約 10 cm（實測 9.7）
    assert np.degrees(np.abs(np.arctan2(np.sin(r['yaw'] - psi), np.cos(r['yaw'] - psi)))).max() < 0.5
    assert abs(np.degrees(np.median(r['bias'])) - 3.0) < 0.3


def test_scan_tau_finds_offset():
    rng = np.random.default_rng(1)
    t, x, y, psi, gt_t, gxy, imu_yaw = synth(rng, tau_true=0.08)
    sm = BatchSmoother(t, gt_t, gxy, imu_yaw)
    scan = sm.scan_tau(np.arange(-0.2, 0.201, 0.02))
    best = min(scan, key=lambda d: d['chi2'])
    assert abs(best['tau'] - 0.08) <= 0.021


def test_nhc_lever_allows_slip():
    """後軸以 5 m/s、20°/s 轉彎，參考點在後軸前 2 m（轉彎時側滑約 8°）。沒有 GNSS、gyro 給完美相對航向、
    地圖每 10 幀給一次位置（σ 0.3 m）：nhc_lever_m=2 要貼回真值，設 0 會被非完整約束拉偏。"""
    fps, n, d = 27.8, 600, 2.0
    t = np.arange(n) / fps
    w = np.where((t > 6) & (t < 14), np.deg2rad(20.0), 0.0)
    psi = np.cumsum(w / fps)
    rx = np.cumsum(5.0 * np.cos(psi) / fps); ry = np.cumsum(5.0 * np.sin(psi) / fps)
    x, y = rx + d * np.cos(psi), ry + d * np.sin(psi)
    fr = np.arange(0, n, 10)
    info = np.tile(np.diag([1 / .3 ** 2, 1 / .3 ** 2, 1 / np.deg2rad(1.0) ** 2]), (len(fr), 1, 1))
    errs = {}
    for lever in (0.0, d):
        cfg = SmootherConfig(use_gnss=False, use_imu=True, imu_abs=False, sigma_dpsi_rad=np.deg2rad(0.01),
                             sigma_lat_m=0.01, sigma_acc_mps2=1.0, nhc_lever_m=lever, map_loss='huber')
        sm = BatchSmoother(t, np.zeros(0), np.zeros((0, 2)), psi, cfg, rel_yaw_at_frames=psi)
        sm.set_initial(np.column_stack([x, y, psi]))
        sm.set_map_observations(fr, np.column_stack([x[fr], y[fr]]), psi[fr], info)
        r = sm.solve()
        errs[lever] = np.sqrt(np.mean((r['x'] - x) ** 2 + (r['y'] - y) ** 2))
    assert errs[d] < 0.15 and errs[d] < 0.3 * errs[0.0]   # 實測 0.12 vs 0.59（剩下的是等速先驗擋住向心加速度）


def test_zupt_averages_stop():
    """開 10 s、停 10 s、再開 10 s，地圖量測每幀都有、帶 0.3 m 雜訊。停車那段加零速約束後，位置等於全部量測一起平均。"""
    rng = np.random.default_rng(5)
    fps = 27.8
    t = np.arange(int(30 * fps)) / fps
    v = np.where((t > 10) & (t < 20), 0.0, 5.0)
    x = np.cumsum(v / fps); y = np.zeros_like(x); psi = np.zeros_like(x)
    n = len(t); fr = np.arange(n)
    mxy = np.column_stack([x, y]) + rng.normal(0, 0.3, (n, 2))
    info = np.tile(np.diag([1 / .3 ** 2, 1 / .3 ** 2, 1 / np.deg2rad(1.0) ** 2]), (n, 1, 1))
    stop = (t > 10.5) & (t < 19.5)
    errs = {}
    for use in (False, True):
        cfg = SmootherConfig(use_gnss=False, use_imu=False, sigma_yaw_acc_rad=np.deg2rad(0.003), sigma_lat_m=0.01,
                             sigma_acc_mps2=1.0, map_loss='huber')
        sm = BatchSmoother(t, np.zeros(0), np.zeros((0, 2)), np.zeros(n), cfg)
        sm.set_initial(np.column_stack([x, y, psi]))
        sm.set_map_observations(fr, mxy, psi, info)
        if use:
            sm.set_stationary(stop)
        r = sm.solve()
        errs[use] = np.sqrt(np.mean((r['x'][stop] - x[stop]) ** 2 + (r['y'][stop] - y[stop]) ** 2))
    assert errs[True] < 0.05 and errs[True] < 0.5 * errs[False]


def test_gnss_bias_bridges_gap():
    """GNSS 帶慢變偏差（沿 x 從 +1 m 線性飄到 −1 m），地圖量測只在前 1/3 與後 1/3；
    開 GNSS 偏差狀態時，中段沒地圖的幀應該靠偏差內插而不是貼回 GNSS。"""
    rng = np.random.default_rng(4)
    t, x, y, psi, gt_t, gxy, imu_yaw = synth(rng)
    bias = 1.0 - 2.0 * gt_t / gt_t[-1]
    gxy_b = gxy + np.column_stack([bias, np.zeros_like(bias)])
    n = len(t)
    fr = np.r_[np.arange(0, n // 3, 10), np.arange(2 * n // 3, n, 10)]
    info = np.tile(np.diag([1 / .05 ** 2, 1 / .05 ** 2, 1 / np.deg2rad(.2) ** 2]), (len(fr), 1, 1))
    mid = np.arange(n // 3, 2 * n // 3)
    errs = {}
    for use_bias in (False, True):
        sm = BatchSmoother(t, gt_t, gxy_b, imu_yaw, SmootherConfig(sigma_gnss_m=0.3, use_gnss_bias=use_bias,
                                                                  sigma_gnss_bias_rw_m=0.3))
        sm.set_map_observations(fr, np.column_stack([x[fr], y[fr]]), psi[fr], info)
        r = sm.solve()
        errs[use_bias] = np.sqrt(np.mean((r['x'][mid] - x[mid]) ** 2 + (r['y'][mid] - y[mid]) ** 2))
    assert errs[True] < 0.15 and errs[True] < 0.7 * errs[False]   # 有偏差狀態：中段靠偏差內插（實測 0.11 vs 0.18）


def test_no_gnss_no_imu_mode():
    """Track03 模式：只有（帶雜訊的）地圖量測＋NHC＋等速＋航向平滑，從追蹤初值出發，應該能把雜訊壓下來。"""
    rng = np.random.default_rng(6)
    t, x, y, psi, gt_t, gxy, imu_yaw = synth(rng)
    n = len(t)
    obs_xy = np.column_stack([x, y]) + rng.normal(0, .3, (n, 2))
    obs_yaw = psi + rng.normal(0, np.deg2rad(1.0), n)
    cfg = SmootherConfig(use_gnss=False, use_imu=False, sigma_yaw_acc_rad=np.deg2rad(0.05), sigma_lat_m=0.01, sigma_acc_mps2=2.0)
    sm = BatchSmoother(t, gt_t, gxy, np.zeros(n), cfg)
    sm.set_initial(np.column_stack([obs_xy, obs_yaw]))
    info = np.tile(np.diag([1 / .3 ** 2, 1 / .3 ** 2, 1 / np.deg2rad(1.0) ** 2]), (n, 1, 1))
    sm.set_map_observations(np.arange(n), obs_xy, obs_yaw, info)
    r = sm.solve()
    err = np.hypot(r['x'] - x, r['y'] - y)
    assert np.sqrt(np.mean(err ** 2)) < 0.12                      # 雜訊 0.3 → 平滑後 <0.12
    yaw_err = np.degrees(np.abs(np.arctan2(np.sin(r['yaw'] - psi), np.cos(r['yaw'] - psi))))
    assert np.sqrt(np.mean(yaw_err ** 2)) < 0.4                    # 雜訊 1° → <0.4°


def test_map_observations_pull_trajectory():
    rng = np.random.default_rng(2)
    t, x, y, psi, gt_t, gxy, imu_yaw = synth(rng)
    # GNSS 整體偏 1 m（像地圖與 GNSS 不一致），地圖量測每 20 幀一筆、正確、很有把握
    sm = BatchSmoother(t, gt_t, gxy + [1.0, 0.0], imu_yaw, SmootherConfig(sigma_gnss_m=0.3))
    fr = np.arange(0, len(t), 20)
    info = np.tile(np.diag([1 / .05 ** 2, 1 / .05 ** 2, 1 / np.deg2rad(.2) ** 2]), (len(fr), 1, 1))
    sm.set_map_observations(fr, np.column_stack([x[fr], y[fr]]), psi[fr], info)
    r = sm.solve()
    err = np.hypot(r['x'] - x, r['y'] - y)
    assert np.sqrt(np.mean(err ** 2)) < 0.1
