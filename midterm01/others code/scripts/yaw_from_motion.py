"""把候選 CSV 的 yaw 換成「行進方向」：Track1 排行榜證實 GT 的 yaw 就是車子的行進方向（差 0.09° 以內），
而 12 m 配準的 yaw 絕對水平很弱（Track3 的 0.49 那份在計分段比自己的行進方向偏 −0.45° 還一路飄到 −1.4°）。
做法：移動中的幀用 ±win 秒位移的方向；gyro 積分當形狀把停車、低速段接起來（「行進方向 − gyro 積分」做慢速 running median 當偏移）。
用法：python -I scripts/yaw_from_motion.py <Track 資料夾> --csv 候選.csv --out 新候選.csv [--min-speed 2 --win 0.5 --offset-win 10]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sdcloc.bagio import load_sensors  # noqa: E402
from sdcloc.frames import load_frames  # noqa: E402
from sdcloc.smoother import wrap  # noqa: E402


def running_median(t, y, win, t_out, min_samples=5):
    out = np.full(len(t_out), np.nan)
    for i, ti in enumerate(t_out):
        m = (t >= ti - win / 2) & (t <= ti + win / 2)
        if m.sum() >= min_samples:
            out[i] = np.median(y[m])
    ok = np.isfinite(out)
    return np.interp(t_out, t_out[ok], out[ok])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('track_dir', type=Path)
    ap.add_argument('--csv', type=Path, required=True)
    ap.add_argument('--min-speed', type=float, default=5.0, help='低於這個車速（m/s）不用行進方向（低速時位移短、方向吵）')
    ap.add_argument('--max-rate', type=float, default=3.0, help='視窗內平均 yaw 變化率超過這個值（°/s）＝轉彎，不用行進方向')
    ap.add_argument('--min-samples', type=int, default=150,
                    help='偏移視窗裡至少要這麼多移動幀才更新偏移（約 5 s）；停車後剛起步是從路邊併回車道，行進方向≠車頭，少量樣本會把水平帶偏 3°')
    ap.add_argument('--win', type=float, default=1.0, help='位移方向的半視窗（秒）：前後各 win 秒')
    ap.add_argument('--offset-win', type=float, default=20.0, help='median 模式：「行進方向 − gyro 積分」慢偏移的 running median 視窗（秒）')
    ap.add_argument('--fit', choices=('median', 'linear'), default='median',
                    help='median：running median 當慢偏移、靜止段沿用停車前的值（Track1 對 I_ns 0.21°）；'
                         'linear：每段行駛一條直線（Track1 沒停車、100 s 一段，零偏漂移不是線性 → 後段差 2°，不建議）')
    ap.add_argument('--causal', action='store_true',
                    help='只用過去的幀：位移方向取過去 2·win 秒、慢偏移取過去 offset_win 秒的中位數、靜止判定只看過去 1 s；形狀預設用 CSV 自己的 yaw')
    ap.add_argument('--shape', choices=('gyro', 'csv'), default=None,
                    help='yaw 的形狀來源：gyro＝原始 gyro 積分（零偏用靜止段估，會看到整段；給 --bias-file 就用每幀的零偏估計，因果）；'
                         'csv＝輸入檔自己的 yaw（因果 EKF 的航向，轉彎時會被配準 yaw 拉 1.5°）。預設：離線 gyro、因果 csv')
    ap.add_argument('--bias-file', type=Path, default=None,
                    help='build_causal 輸出的 .state.npz（每幀 gyro 零偏估計，rad/s）；給了就 shape=gyro 且零偏逐幀用它（只用過去的估計）')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    shape = 'gyro' if a.bias_file else (a.shape or ('csv' if a.causal else 'gyro'))
    frame_t, _ = load_frames(a.track_dir / 'raw_ply')
    ft = (frame_t - frame_t[0]) / 1e9
    c = pl.read_csv(a.csv).to_numpy()
    assert len(c) == len(ft), 'CSV 列數和幀數不符'
    xy = c[:, 1:3]
    N = len(ft)
    # gyro 積分（原始 gz，零偏先用靜止段或整段中位數粗估；殘差由慢偏移吸收）
    imu = load_sensors(a.track_dir, need_gnss=False)['imu']
    it = (imu['t'] - frame_t[0]) / 1e9
    gz = imu['gyro'][:, 2]
    sd = np.array([np.degrees(gz[max(0, i - 50):i + 50].std()) for i in range(0, len(gz), 50)])
    still_blocks = np.flatnonzero(sd < 0.2) * 50
    bias = float(np.mean(np.concatenate([gz[s:s + 50] for s in still_blocks]))) if len(still_blocks) else float(np.median(gz))
    if a.bias_file:
        # 逐幀零偏（EKF 線上估的）：第 k 幀到第 k+1 幀的 gyro 積分扣第 k 幀的估計
        bk = np.load(a.bias_file)['gyro_bias']
        assert len(bk) == N
        dt_i = np.gradient(it)
        s_idx = np.searchsorted(it, ft)
        dpsi_k = np.array([np.sum((gz[s_idx[k]:s_idx[k + 1]] - bk[k]) * dt_i[s_idx[k]:s_idx[k + 1]]) for k in range(N - 1)])
        psi_f = np.concatenate([[0.0], np.cumsum(dpsi_k)])
        bias = float(bk[-1])
    else:
        psi = np.concatenate([[0.0], np.cumsum(0.5 * (gz[1:] + gz[:-1] - 2 * bias) * np.diff(it))])
        psi_f = np.interp(ft, it, psi) if shape == 'gyro' else np.unwrap(c[:, 3])
    # 車停著就是不轉：IMU 判定靜止（1 s 內 gyro 標準差 <0.2°/s、加速度 <0.06；離線前後各 0.5 s，因果只看過去 1 s）的幀之間 yaw 不動，
    # 不然 42 秒的停車會把零偏的小誤差累積成 1°
    sig = np.column_stack([np.degrees(imu['gyro']), imu['acc']])
    c1 = np.vstack([np.zeros(6), np.cumsum(sig, 0)]); c2 = np.vstack([np.zeros(6), np.cumsum(sig ** 2, 0)])
    back, fwd = (1.0, 0.0) if a.causal else (0.5, 0.5)
    lo = np.searchsorted(it, ft - back); hi = np.searchsorted(it, ft + fwd); cnt = np.maximum(hi - lo, 1)[:, None]
    std = np.sqrt(np.maximum((c2[hi] - c2[lo]) / cnt - ((c1[hi] - c1[lo]) / cnt) ** 2, 0))
    still = (hi - lo > 20) & (std[:, :3].max(1) < 0.2) & (std[:, 3:].max(1) < 0.06)
    dpsi = np.diff(psi_f)
    dpsi[still[1:] & still[:-1]] = 0.0
    psi_f = np.concatenate([[psi_f[0]], psi_f[0] + np.cumsum(dpsi)])
    # 行進方向：離線＝前後各 win 秒的位移；因果＝過去 2·win 秒的位移
    W = int(round(a.win / np.median(np.diff(ft))))
    if a.causal:
        k = np.arange(2 * W, N); k0, k1 = k - 2 * W, k
    else:
        k = np.arange(W, N - W); k0, k1 = k - W, k + W
    d = xy[k1] - xy[k0]
    speed = np.hypot(*d.T) / (ft[k1] - ft[k0])
    mdir = np.arctan2(d[:, 1], d[:, 0])
    # 轉彎時位移方向落後車頭（視窗內車頭一直在轉），不能拿來定水平：形狀的 yaw 變化率超過 max_rate 的幀排除，
    # 轉彎段的 yaw 就完全照形狀（gyro／EKF）走、偏移沿用轉彎前的值
    rate = np.degrees(np.abs(psi_f[k1] - psi_f[k0]) / (ft[k1] - ft[k0]))
    moving = (speed >= a.min_speed) & (rate < a.max_rate)
    off = wrap(mdir[moving] - psi_f[k[moving]])
    off = off - 2 * np.pi * np.round((off - np.median(off)) / (2 * np.pi))
    tm = ft[k[moving]]
    edges = np.flatnonzero(np.diff(np.r_[False, still, False]))
    bounds = sorted({0, N, *(int(e) for e in edges)})          # 靜止／移動段的邊界（最後一幀靜止時 N 會重複，用 set 去掉）
    if a.fit == 'median' and a.causal:
        # 只看過去 offset_win 秒的中位數；還沒有任何樣本時沿用輸入檔的 yaw（偏移＝輸入 yaw − 形狀）
        off_s = np.zeros(N)
        csv_off = np.unwrap(c[:, 3]) - psi_f
        last = None
        for i in range(N):
            m = (tm >= ft[i] - a.offset_win) & (tm <= ft[i])
            if m.sum() >= a.min_samples:
                last = float(np.median(off[m]))
            off_s[i] = last if last is not None else csv_off[i]
        for s_, e_ in zip(bounds[:-1], bounds[1:]):
            if still[s_] and s_ > 0:
                off_s[s_:e_] = off_s[s_ - 1]
    elif a.fit == 'median':
        off_s = running_median(tm, off, a.offset_win, ft, a.min_samples)
        # 靜止段：沿用停車前最後一個移動幀的偏移（不跨停車內插；停車前後的偏移差是 gyro 零偏漂移，不該算在停車裡）
        for s_, e_ in zip(bounds[:-1], bounds[1:]):
            if still[s_] and s_ > 0:
                off_s[s_:e_] = off_s[s_ - 1]
    else:
        # 行駛段：靜止幀把時間軸切成幾段；每段用該段的高速幀做 a + b·t 的 Huber（1°）擬合；靜止段沿用前一段末端的值
        off_s = np.full(N, np.nan)
        for s_, e_ in zip(bounds[:-1], bounds[1:]):
            if still[s_]:
                continue
            m = (k[moving] >= s_) & (k[moving] < e_)
            if m.sum() < 30:
                continue
            A = np.column_stack([np.ones(m.sum()), tm[m] - tm[m].mean()])
            x, *_ = np.linalg.lstsq(A, off[m], rcond=None)
            for _ in range(10):
                r = np.abs(A @ x - off[m]); w = np.where(r < np.deg2rad(1), 1.0, np.deg2rad(1) / r)
                x, *_ = np.linalg.lstsq(A * w[:, None], off[m] * w, rcond=None)
            off_s[s_:e_] = x[0] + x[1] * (ft[s_:e_] - tm[m].mean())
        ok = np.isfinite(off_s)
        idx = np.arange(N)
        for s_, e_ in zip(bounds[:-1], bounds[1:]):          # 靜止段（或太短的段）：接前一段的末端，沒有前一段就接後一段的開頭
            if not np.isfinite(off_s[s_]):
                prev = idx[:s_][ok[:s_]]
                off_s[s_:e_] = off_s[prev[-1]] if len(prev) else off_s[idx[ok][0]]
    yaw = wrap(psi_f + off_s)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', newline='') as f:
        f.write('idx,x,y,yaw\r\n')
        for i in range(N):
            f.write(f'{i},{xy[i, 0]:.9f},{xy[i, 1]:.9f},{yaw[i]:.11f}\r\n')
    resid = np.degrees(wrap(mdir[moving] - yaw[k[moving]]))
    dy = np.degrees(wrap(yaw - c[:, 3]))
    info = {'frames': N, 'causal': bool(a.causal), 'shape': shape, 'moving_frames': int(moving.sum()), 'still_frames': int(still.sum()),
            'gyro_bias_degps': float(np.degrees(bias)), 'still_blocks': int(len(still_blocks)),
            'motion_minus_out_deg': {'rms': float(np.sqrt(np.mean(resid ** 2))), 'p90': float(np.percentile(np.abs(resid), 90))},
            'offset_deg_range': [float(np.degrees(off_s.min())), float(np.degrees(off_s.max()))],
            'new_minus_old_yaw_deg': {'mean': float(dy.mean()), 'rms': float(np.sqrt(np.mean(dy ** 2))), 'max': float(np.abs(dy).max())}}
    print(json.dumps(info, indent=1))
    a.out.with_suffix('.diag.json').write_text(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
