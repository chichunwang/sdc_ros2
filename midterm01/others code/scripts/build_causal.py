"""因果版候選：每一幀只用「到這一幀為止」的資料（可以在車上即時跑），不用後面的幀回頭修。
- 狀態 [x, y, ψ, gyro 零偏]。預測：航向用 gyro 積分扣零偏；位置用最近幾幀接受的位姿差分出的車速，沿車頭方向推進。
- 更新：當幀的配準 (x, y, ψ)，資訊 = ICP 資訊／c，Cauchy 穩健權重（配準跳掉時拉力變小）；配準的 ψ 會順便修零偏。
- 靜止（IMU 判定，只看過去 1 s）：位置、航向幾乎不動（Q 很小），gyro 平均當零偏的量測；配準照樣更新 → 等於把停車期間的配準平均。
- 配準輸入用 register_all 的單幀結果（不疊合前後幀）；尺度用 scale_from_height --causal；追蹤用 --backfill 0。
輸出相機位姿（lever 0、偏角 0）。"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sdcloc.bagio import load_sensors  # noqa: E402
from sdcloc.frames import load_frames  # noqa: E402
from sdcloc.smoother import wrap  # noqa: E402

IDX = [0, 1, 5]


def write_kaggle(path, x, y, yaw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        f.write('idx,x,y,yaw\r\n')
        for i, (a, b, c) in enumerate(zip(x, y, yaw)):
            f.write(f'{i},{a:.9f},{b:.9f},{c:.11f}\r\n')


def rot2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('track_dir', type=Path)
    ap.add_argument('--reg', type=Path, required=True, help='單幀配準 npz（register_all，不疊合）')
    ap.add_argument('--gyro-dt', type=float, default=0.0, help='幀時間 t 取 IMU 在 t+dt 的值（Track3 用 0.14）')
    ap.add_argument('--overconf', type=float, default=100.0, help='ICP 資訊 ÷ c')
    ap.add_argument('--yaw-info-scale', type=float, default=1.0,
                    help='配準 yaw 的資訊再乘這個數（<1＝少信配準的 yaw、航向形狀跟著 gyro 走；配準 yaw 有場景偏差，Track3 轉彎段比 gyro 多轉 1.4°）')
    ap.add_argument('--sigma-v', type=float, default=2.0, help='每幀車速預測的 σ（m/s）；Track1 對 I_ns：0.5 0.64 m、1.0 0.62、2.0 0.56（配 hist 20）、3.0 同 2.0')
    ap.add_argument('--max-reject', type=int, default=3, help='配準連續被拒幾幀就強制接受（濾波器偏掉時拉回來）')
    ap.add_argument('--sigma-lat', type=float, default=0.02, help='每幀側向位移的 σ（m）')
    ap.add_argument('--sigma-gyro-deg', type=float, default=0.02, help='每幀 gyro 積分航向差的 σ（度）')
    ap.add_argument('--sigma-bias-rw-degps', type=float, default=0.02, help='gyro 零偏隨機漫步 σ（°/s 每 √s；Track1 零偏 100 s 漂 0.2°/s）')
    ap.add_argument('--cauchy', type=float, default=10.0, help='配準更新的 Cauchy 尺度（Mahalanobis）；Track1 掃過 1.5／3／10 → 10 最好（少拒絕就少誤跳）')
    ap.add_argument('--hist', type=int, default=20, help='估車速用最近幾幀接受的位姿（中位數）；Track1：5 0.63 m、10 0.62、20 0.56、30 0.54')
    ap.add_argument('--min-ng03', type=float, default=0.4)
    ap.add_argument('--min-inlier', type=float, default=0.5)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()

    frame_t, _ = load_frames(a.track_dir / 'raw_ply')
    ft = (frame_t - frame_t[0]) / 1e9
    N = len(ft)
    reg = np.load(a.reg)
    assert len(reg['ok']) == N
    imu = load_sensors(a.track_dir, need_gnss=False)['imu']
    it = (imu['t'] - frame_t[0]) / 1e9
    gz = imu['gyro'][:, 2]
    # 靜止判定（1 s 視窗內的標準差；視窗只看過去 1 s，因果）
    win = 1.0
    sig = np.column_stack([np.degrees(imu['gyro']), imu['acc']])
    c1 = np.vstack([np.zeros(6), np.cumsum(sig, 0)]); c2 = np.vstack([np.zeros(6), np.cumsum(sig ** 2, 0)])

    def still_at(t):
        b = np.searchsorted(it, t + a.gyro_dt); s0 = np.searchsorted(it, t + a.gyro_dt - win)
        if b - s0 < 20:
            return False
        m = (c1[b] - c1[s0]) / (b - s0); v = (c2[b] - c2[s0]) / (b - s0) - m ** 2
        sd = np.sqrt(np.maximum(v, 0))
        return bool(sd[:3].max() < 0.2 and sd[3:].max() < 0.06)

    good = reg['ok'] & (reg['ng_frac03'] >= a.min_ng03) & (reg['inlier_frac'] >= a.min_inlier)
    dit = np.gradient(it)
    # 狀態 [x, y, ψ, b]：b 是 gyro 零偏，起始 0、σ 1°/s（開頭車不一定是停的：Track3 一開始就在動，拿開頭 0.8 s 當零偏會差 3°/s）
    k0 = int(np.flatnonzero(good)[0])
    x = np.array([reg['t'][k0, 0], reg['t'][k0, 1], reg['yaw'][k0], 0.0])
    P = np.diag([0.3 ** 2, 0.3 ** 2, np.deg2rad(1.0) ** 2, np.deg2rad(1.0) ** 2])
    H3 = np.zeros((3, 4)); H3[0, 0] = H3[1, 1] = H3[2, 2] = 1.0
    Hb = np.array([[0.0, 0.0, 0.0, 1.0]])
    out = np.zeros((N, 3)); bias_hist = np.zeros(N); accepted = []; n_upd = 0; n_still = 0; n_rej = 0; n_forced = 0; consec_rej = 0
    for k in range(N):
        t = ft[k]
        still = still_at(t)
        if k > 0:
            # gyro 積分（上一幀到這一幀），扣目前估的零偏
            s0 = np.searchsorted(it, ft[k - 1] + a.gyro_dt); s1 = np.searchsorted(it, t + a.gyro_dt)
            T = float(np.sum(dit[s0:s1])) if s1 > s0 else 0.0
            dpsi = float(np.sum(gz[s0:s1] * dit[s0:s1])) - x[3] * T if s1 > s0 else 0.0
            if still:
                n_still += 1
                dpsi = 0.0; v = 0.0
            else:
                # 車速：最近接受的濾波位姿差分（中位數；只看過去）。用配準本身差分試過（單幀配準抖 0.3 m）更差
                H = np.array(accepted[-a.hist:]) if len(accepted) >= 3 else None
                if H is not None:
                    dk = np.diff(H[:, 0]); d = np.diff(H[:, 1:3], axis=0); ym = H[:-1, 3]
                    along = (np.cos(ym) * d[:, 0] + np.sin(ym) * d[:, 1]) / dk
                    v = float(np.median(along))
                else:
                    v = 0.0
            pm = x[2] + 0.5 * dpsi
            dt = t - ft[k - 1]
            x = np.array([x[0] + v * np.cos(pm), x[1] + v * np.sin(pm), x[2] + dpsi, x[3]])
            # 預測共變異數：沿車向 σ_v·dt、側向 σ_lat、航向 σ_gyro、零偏隨機漫步（靜止時位置、航向幾乎不動）
            Q = np.zeros((4, 4))
            if still:
                Q[0, 0] = Q[1, 1] = 0.005 ** 2; Q[2, 2] = np.deg2rad(0.005) ** 2
            else:
                Qb = np.diag([(a.sigma_v * dt) ** 2, a.sigma_lat ** 2])
                R2 = rot2(pm)
                Q[:2, :2] = R2 @ Qb @ R2.T; Q[2, 2] = np.deg2rad(a.sigma_gyro_deg) ** 2
            Q[3, 3] = (np.deg2rad(a.sigma_bias_rw_degps) * np.sqrt(dt)) ** 2
            F = np.eye(4); F[0, 2] = -v * np.sin(pm); F[1, 2] = v * np.cos(pm)
            if not still:
                F[2, 3] = -T                       # ψ 對零偏的導數
            P = F @ P @ F.T + Q
        # 更新 1：靜止時 gyro 平均就是零偏的量測（視窗只看過去 1 s）
        if still:
            s0 = np.searchsorted(it, t + a.gyro_dt - win); s1 = np.searchsorted(it, t + a.gyro_dt)
            if s1 - s0 >= 20:
                zb = float(np.mean(gz[s0:s1])); rb = np.deg2rad(0.05) ** 2
                Sb = float(P[3, 3]) + rb
                Kb = P @ Hb.T / Sb
                x = x + (Kb * (zb - x[3])).ravel(); P = (np.eye(4) - Kb @ Hb) @ P
        # 更新 2：當幀配準（靜止時只更新位置：Q 很小，等於把停車期間的配準平均起來；
        # yaw 靜止時不更新——配準 yaw 在停車處有場景偏差，1200 幀累積曾把 yaw 拉走 2°，gyro 說根本沒轉）
        if good[k]:
            z = np.array([reg['t'][k, 0], reg['t'][k, 1], reg['yaw'][k]])
            I = reg['information'][k][np.ix_(IDX, IDX)] / a.overconf
            if np.isfinite(I).all() and np.trace(I) > 0:
                if a.yaw_info_scale != 1.0:
                    S_ = np.diag([1.0, 1.0, np.sqrt(a.yaw_info_scale)]); I = S_ @ I @ S_
                Hm, zm, Im = (H3[:2], z[:2], I[:2, :2]) if still else (H3, z, I)
                r = zm - Hm @ x
                if not still:
                    r[2] = wrap(r[2])
                Rm = np.linalg.inv(Im + 1e-9 * np.eye(len(zm)))
                S = Hm @ P @ Hm.T + Rm
                md = float(np.sqrt(r @ np.linalg.solve(S, r)))
                w = 1.0 / (1.0 + (md / a.cauchy) ** 2)
                if w < 0.05 and consec_rej < a.max_reject:
                    n_rej += 1; consec_rej += 1
                else:
                    if w < 0.05:
                        n_forced += 1; w = 1.0; P[:3, :3] += np.diag([1.0, 1.0, np.deg2rad(5.0) ** 2])   # 連拒太多：信配準、把不確定放大
                    consec_rej = 0
                    K = P @ Hm.T @ np.linalg.inv(Hm @ P @ Hm.T + Rm / w)
                    x = x + K @ r; x[2] = wrap(x[2])
                    P = (np.eye(4) - K @ Hm) @ P
                    n_upd += 1
                    if not still:
                        accepted.append((k, x[0], x[1], x[2]))
        out[k] = x[:3]; bias_hist[k] = x[3]
    write_kaggle(a.out, out[:, 0], out[:, 1], wrap(out[:, 2]))
    np.savez(a.out.with_suffix('.state.npz'), xy=out[:, :2], yaw=wrap(out[:, 2]), gyro_bias=bias_hist)   # 每幀的零偏估計給 yaw_from_motion --causal 用
    info = {'frames': N, 'updates': n_upd, 'rejected': n_rej, 'forced_accepts': n_forced, 'still_frames': n_still,
            'bias_degps_final': float(np.degrees(x[3])), 'bias_sigma_degps_final': float(np.degrees(np.sqrt(P[3, 3]))), 'gyro_dt': a.gyro_dt}
    print(json.dumps(info, indent=1))
    a.out.with_suffix('.diag.json').write_text(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
