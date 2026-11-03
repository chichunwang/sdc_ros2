"""整段軌跡一起解的 batch smoother（Gauss-Newton + IRLS robust 權重，稀疏正規方程）。

狀態：每個 frame k 的 (x_k, y_k, ψ_k)，加上 heading bias 的分段常數 b_j（每 block_s 秒一段）。
因子：
  GNSS   位置：線性內插到 GNSS 時刻（可帶時間差 tau），Huber 權重。
  IMU    相對航向：ψ_{k+1} − ψ_k = Δψ_imu，σ_dpsi。
  IMU    絕對航向：ψ_k = ψ_imu(k) + b_j，σ_psi（寬鬆，讓 bias 吸收）。
  bias   平滑：b_{j+1} − b_j ≈ 0，σ_b。
  NHC    車體側向位移 ≈ 0：(p_{k+1} − p_k) 在中點航向的側向分量，σ_lat。
  CV     等速平滑：p_{k+1} − 2p_k + p_{k−1} ≈ 0，σ_acc·dt²。
  MAP    （可選）每 frame 的 (x, y, ψ) 絕對量測，帶 3×3 資訊矩陣與 Huber。
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


@dataclass
class SmootherConfig:
    sigma_gnss_m: float = 0.3
    gnss_huber_m: float = 0.5
    sigma_dpsi_rad: float = np.deg2rad(0.03)
    sigma_dpsi_long_rad: float = 0.0         # >0 時加 10 幀長基線的相對 yaw 約束（讓慢彎很貴）
    long_stride: int = 10
    sigma_psi_rad: float = np.deg2rad(1.0)
    sigma_bias_rad_per_block: float = np.deg2rad(0.5)   # IMU 航向偏差實測會在 100 s 內飄 ±3°，要放寬
    block_s: float = 2.0
    sigma_lat_m: float = 0.03
    sigma_acc_mps2: float = 3.0
    map_huber_sigma: float = 3.0
    map_loss: str = 'huber'       # 'huber' 或 'cauchy'（整群一致但錯誤的量測，拉力會被壓住）
    iters: int = 8
    # Track03 模式：沒有 GNSS／IMU 時關掉對應因子，航向只靠地圖量測＋航向加速度平滑
    use_gnss: bool = True
    use_imu: bool = True
    imu_abs: bool = True                     # False：只用 gyro 的相對航向（Track03 沒 GNSS 可校 AHRS 的偏角）
    nhc_lever_m: float = 0.0                 # 參考點在後軸前方幾公尺：轉彎時這個點本來就側滑 d·Δψ（Track3 急左轉才看得出來）
    zupt_sigma_m: float = 0.002              # 靜止幀之間的位置差 σ（set_stationary 給了才用）
    zupt_sigma_rad: float = np.deg2rad(0.002)        # 要比 gyro 相對航向（0.01°）緊，停車時航向才不會跟著殘餘零偏飄
    sigma_yaw_acc_rad: float = 0.0           # >0 時加 ψ_{k+1} − 2ψ_k + ψ_{k−1} ≈ 0（弧度／幀²）
    # GNSS 慢變偏差（2D，每 block_s 一段的 random walk）；關掉時 GNSS 視為無偏
    use_gnss_bias: bool = False
    sigma_gnss_bias_rw_m: float = 0.15      # 每段可以飄多少
    sigma_gnss_bias_prior_m: float = 3.0    # 很弱的歸零先驗，沒地圖量測時定 gauge


class BatchSmoother:
    def __init__(self, frame_t, gnss_t, gnss_xy, imu_yaw_at_frames, cfg=None, rel_yaw_at_frames=None):
        """frame_t, gnss_t：秒（同一時基）；imu_yaw_at_frames：已 unwrap 的 IMU yaw（弧度），給絕對航向（帶偏差狀態）。
        rel_yaw_at_frames：相對航向用的 yaw 序列（例如 gyro 積分）；不給就用 imu_yaw。"""
        self.ft = np.asarray(frame_t, dtype=float)
        self.gt = np.asarray(gnss_t, dtype=float)
        self.gxy = np.asarray(gnss_xy, dtype=float)
        self.imu_yaw = np.asarray(imu_yaw_at_frames, dtype=float)
        self.rel_yaw = self.imu_yaw if rel_yaw_at_frames is None else np.asarray(rel_yaw_at_frames, dtype=float)
        self.init_xy_yaw = None          # (N,3) 初值；Track03 模式由 set_initial 給
        self.cfg = cfg or SmootherConfig()
        self.N = len(self.ft)
        self.nb = int(np.ceil((self.ft[-1] - self.ft[0]) / self.cfg.block_s)) + 1
        self.block_of = np.minimum(((self.ft - self.ft[0]) / self.cfg.block_s).astype(int), self.nb - 1)
        self.map_obs = None
        self.stationary = None

    # 索引：frame 狀態 3N、heading bias nb、GNSS bias 2·nb（開啟時）
    def ix(self, k): return 3 * k
    def iy(self, k): return 3 * k + 1
    def ip(self, k): return 3 * k + 2
    def ib(self, j): return 3 * self.N + j
    def ig(self, j, d): return 3 * self.N + self.nb + 2 * j + d
    @property
    def n_unknowns(self): return 3 * self.N + self.nb + (2 * self.nb if self.cfg.use_gnss_bias else 0)

    def set_initial(self, xy_yaw):
        self.init_xy_yaw = np.asarray(xy_yaw, dtype=float)

    def set_stationary(self, mask):
        """mask (N,) bool：IMU 判定車子靜止的幀；相鄰兩幀都靜止就加位置、航向不變的約束（零速更新）。"""
        self.stationary = np.asarray(mask, bool)

    def set_map_observations(self, frames, xy, yaw, info3):
        """frames: 索引陣列；xy (M,2)；yaw (M,)；info3 (M,3,3) 為 [x,y,ψ] 的資訊矩陣（1/σ²）。"""
        self.map_obs = (np.asarray(frames), np.asarray(xy, float), np.asarray(yaw, float), np.asarray(info3, float))

    def initial(self):
        x = np.zeros(self.n_unknowns)
        if self.init_xy_yaw is not None:
            # Track03：用追蹤結果當初值
            x[0::3][:self.N] = self.init_xy_yaw[:, 0]; x[1::3][:self.N] = self.init_xy_yaw[:, 1]
            x[2::3][:self.N] = np.unwrap(self.init_xy_yaw[:, 2])
            return x
        # 位置：GNSS 線性內插；航向：IMU yaw + 用 GNSS course 粗估的常數 bias
        for d in range(2):
            x[d::3][:self.N] = np.interp(self.ft, self.gt, self.gxy[:, d])
        dxy = np.diff(self.gxy, axis=0)
        sp_ = np.linalg.norm(dxy, axis=1) / np.diff(self.gt)
        course = np.arctan2(dxy[:, 1], dxy[:, 0])
        tm = 0.5 * (self.gt[1:] + self.gt[:-1])
        off = wrap(course - np.interp(tm, self.ft, self.imu_yaw))
        b0 = float(np.median(off[sp_ > 3])) if np.any(sp_ > 3) else 0.0
        x[2::3][:self.N] = self.imu_yaw + b0
        x[3 * self.N:3 * self.N + self.nb] = b0
        return x

    def solve(self, tau=0.0, x0=None, verbose=False, margin=0.0):
        """tau：GNSS 時間戳要加的秒數（正 = GNSS 落後）。margin：掃 τ 時用固定的樣本集（原時間戳離兩端 ≥ margin 秒）。
        回傳 dict(x, y, yaw, bias, chi2 分項, residuals)。"""
        c = self.cfg
        x = self.initial() if x0 is None else x0.copy()
        gt = self.gt + tau
        # GNSS 內插係數（固定，不隨迭代變）；兩端允許約 0.3 s 外推
        kk = np.clip(np.searchsorted(self.ft, gt) - 1, 0, self.N - 2)
        alpha = (gt - self.ft[kk]) / (self.ft[kk + 1] - self.ft[kk])
        use = (gt >= self.ft[0] - 0.3) & (gt <= self.ft[-1] + 0.3)
        if margin > 0:
            use = (self.gt >= self.ft[0] + margin) & (self.gt <= self.ft[-1] - margin)
        kk, alpha, gxy = kk[use], alpha[use], self.gxy[use]
        gblk = np.clip(((gt[use] - self.ft[0]) / c.block_s).astype(int), 0, self.nb - 1)
        dpsi = np.diff(self.rel_yaw)
        Ls = c.long_stride
        dpsi_long = self.rel_yaw[Ls:] - self.rel_yaw[:-Ls] if (c.sigma_dpsi_long_rad > 0 and self.N > Ls) else None
        dt = np.diff(self.ft)
        hist = []
        for it in range(c.iters):
            rows, cols, vals, res, wts = [], [], [], [], []
            r0 = 0

            def add(rr, cc, vv, resid, weight):
                nonlocal r0
                rows.extend(np.asarray(rr) + r0); cols.extend(cc); vals.extend(vv)
                res.append(resid); wts.append(weight); r0 += len(resid)

            X, Y, P = x[0::3][:self.N], x[1::3][:self.N], x[2::3][:self.N]
            B = x[3 * self.N:3 * self.N + self.nb]
            # GNSS：量測 = 真值 + 偏差（開啟時）
            for d, S in ((0, X), (1, Y)) if c.use_gnss else ():
                pred = (1 - alpha) * S[kk] + alpha * S[kk + 1]
                r = pred - gxy[:, d]
                m = len(r)
                idx = np.arange(m)
                if c.use_gnss_bias:
                    gb = x[self.ig(gblk, d)]
                    r = r + gb
                    add(np.r_[idx, idx, idx], np.r_[3 * kk + d, 3 * (kk + 1) + d, self.ig(gblk, d)],
                        np.r_[1 - alpha, alpha, np.ones(m)], r, self._huber(r, c.sigma_gnss_m, c.gnss_huber_m))
                else:
                    add(np.r_[idx, idx], np.r_[3 * kk + d, 3 * (kk + 1) + d], np.r_[1 - alpha, alpha], r,
                        self._huber(r, c.sigma_gnss_m, c.gnss_huber_m))
            if c.use_gnss_bias and c.use_gnss:
                for d in (0, 1):
                    gb = x[self.ig(np.arange(self.nb), d)]
                    if self.nb > 1:
                        idx = np.arange(self.nb - 1)
                        add(np.r_[idx, idx], np.r_[self.ig(idx, d), self.ig(idx + 1, d)],
                            np.r_[-np.ones(self.nb - 1), np.ones(self.nb - 1)], gb[1:] - gb[:-1],
                            np.full(self.nb - 1, 1 / c.sigma_gnss_bias_rw_m ** 2))
                    idx = np.arange(self.nb)
                    add(idx, self.ig(idx, d), np.ones(self.nb), gb, np.full(self.nb, 1 / c.sigma_gnss_bias_prior_m ** 2))
            if c.use_imu:
                # IMU 相對航向（相鄰幀；可選長基線）
                r = (P[1:] - P[:-1]) - dpsi
                idx = np.arange(self.N - 1)
                add(np.r_[idx, idx], np.r_[3 * idx + 2, 3 * (idx + 1) + 2], np.r_[-np.ones(self.N - 1), np.ones(self.N - 1)],
                    r, np.full(self.N - 1, 1 / c.sigma_dpsi_rad ** 2))
                if dpsi_long is not None:
                    r = (P[Ls:] - P[:-Ls]) - dpsi_long
                    idx = np.arange(self.N - Ls)
                    add(np.r_[idx, idx], np.r_[3 * idx + 2, 3 * (idx + Ls) + 2], np.r_[-np.ones(self.N - Ls), np.ones(self.N - Ls)],
                        r, np.full(self.N - Ls, 1 / c.sigma_dpsi_long_rad ** 2))
            if c.use_imu and c.imu_abs:
                # IMU 絕對航向 + bias
                r = wrap(P - self.imu_yaw - B[self.block_of])
                idx = np.arange(self.N)
                add(np.r_[idx, idx], np.r_[3 * idx + 2, 3 * self.N + self.block_of], np.r_[np.ones(self.N), -np.ones(self.N)],
                    r, np.full(self.N, 1 / c.sigma_psi_rad ** 2))
            # bias 平滑（沒有 IMU 時 bias 狀態沒有量測，弱先驗把它固定住）
            if self.nb > 1:
                r = B[1:] - B[:-1]
                idx = np.arange(self.nb - 1)
                add(np.r_[idx, idx], np.r_[3 * self.N + idx, 3 * self.N + idx + 1], np.r_[-np.ones(self.nb - 1), np.ones(self.nb - 1)],
                    r, np.full(self.nb - 1, 1 / c.sigma_bias_rad_per_block ** 2))
            if not (c.use_imu and c.imu_abs):
                idx = np.arange(self.nb)
                add(idx, 3 * self.N + idx, np.ones(self.nb), B, np.full(self.nb, 1.0))
            # 航向加速度平滑（Track03：沒有 gyro 時讓 yaw 不要逐幀亂跳）
            if c.sigma_yaw_acc_rad > 0 and self.N > 2:
                idx = np.arange(1, self.N - 1)
                r = P[idx + 1] - 2 * P[idx] + P[idx - 1]
                j = np.arange(len(idx))
                add(np.r_[j, j, j], np.r_[3 * (idx - 1) + 2, 3 * idx + 2, 3 * (idx + 1) + 2],
                    np.r_[np.ones(len(idx)), -2 * np.ones(len(idx)), np.ones(len(idx))], r, np.full(len(idx), 1 / c.sigma_yaw_acc_rad ** 2))
            # NHC：側向位移
            pm = 0.5 * (P[1:] + P[:-1])
            dx_, dy_ = X[1:] - X[:-1], Y[1:] - Y[:-1]
            s_, c_ = np.sin(pm), np.cos(pm)
            L_ = c.nhc_lever_m
            r = -s_ * dx_ + c_ * dy_ - L_ * (P[1:] - P[:-1])
            # 對 ψ 的導數：d/dψ_mid = -c dx - s dy，分一半給兩端
            dpsi_ = -c_ * dx_ - s_ * dy_
            idx = np.arange(self.N - 1)
            add(np.r_[idx, idx, idx, idx, idx, idx],
                np.r_[3 * idx, 3 * (idx + 1), 3 * idx + 1, 3 * (idx + 1) + 1, 3 * idx + 2, 3 * (idx + 1) + 2],
                np.r_[s_, -s_, -c_, c_, 0.5 * dpsi_ + L_, 0.5 * dpsi_ - L_], r, np.full(self.N - 1, 1 / c.sigma_lat_m ** 2))
            # 零速：相鄰兩幀都靜止 → 位置、航向不變
            if self.stationary is not None:
                idx = np.flatnonzero(self.stationary[1:] & self.stationary[:-1])
                if len(idx):
                    j = np.arange(len(idx))
                    for d, S, sig in ((0, X, c.zupt_sigma_m), (1, Y, c.zupt_sigma_m), (2, P, c.zupt_sigma_rad)):
                        add(np.r_[j, j], np.r_[3 * idx + d, 3 * (idx + 1) + d], np.r_[-np.ones(len(idx)), np.ones(len(idx))],
                            S[idx + 1] - S[idx], np.full(len(idx), 1 / sig ** 2))
            # 等速平滑
            if self.N > 2:
                idx = np.arange(1, self.N - 1)
                sig = c.sigma_acc_mps2 * (0.5 * (dt[:-1] + dt[1:])) ** 2
                for d, S in ((0, X), (1, Y)):
                    r = S[idx + 1] - 2 * S[idx] + S[idx - 1]
                    j = np.arange(len(idx))
                    add(np.r_[j, j, j], np.r_[3 * (idx - 1) + d, 3 * idx + d, 3 * (idx + 1) + d],
                        np.r_[np.ones(len(idx)), -2 * np.ones(len(idx)), np.ones(len(idx))], r, 1 / sig ** 2)
            # 地圖量測
            if self.map_obs is not None:
                fr, mxy, myaw, info = self.map_obs
                for i, k in enumerate(fr):
                    L = np.linalg.cholesky(info[i] + 1e-12 * np.eye(3))      # info = L Lᵀ → whitened residual Lᵀ r
                    rv = np.array([X[k] - mxy[i, 0], Y[k] - mxy[i, 1], wrap(P[k] - myaw[i])])
                    wr = L.T @ rv
                    # robust：以 Mahalanobis 距離做 Huber 或 Cauchy
                    md = np.linalg.norm(wr)
                    if c.map_loss == 'cauchy':
                        w = 1.0 / (1.0 + (md / c.map_huber_sigma) ** 2)
                    elif c.map_loss == 'gm':      # Geman-McClure：成本有上限，整群錯的量測拉力有限
                        w = 1.0 / (1.0 + (md / c.map_huber_sigma) ** 2) ** 2
                    else:
                        w = 1.0 if md <= c.map_huber_sigma else c.map_huber_sigma / md
                    J = L.T
                    rr = np.repeat(np.arange(3), 3)
                    cc = np.tile([3 * k, 3 * k + 1, 3 * k + 2], 3)
                    add(rr, cc, J.ravel(), wr, np.full(3, w))
            res = np.concatenate(res); wts = np.concatenate(wts)
            J = sp.csr_matrix((vals, (rows, cols)), shape=(len(res), self.n_unknowns))
            W = sp.diags(wts)
            H = (J.T @ W @ J).tocsc()
            g = J.T @ (W @ res)
            H = H + sp.identity(self.n_unknowns) * 1e-9
            delta = spla.spsolve(H, -g)
            x = x + delta
            chi2 = float((wts * res ** 2).sum())
            hist.append(chi2)
            if verbose:
                print(f'iter {it} chi2 {chi2:.1f} max|delta| {np.abs(delta).max():.2e}')
            if np.abs(delta).max() < 1e-6:
                break
        X, Y, P = x[0::3][:self.N], x[1::3][:self.N], x[2::3][:self.N]
        pred = np.column_stack([(1 - alpha) * X[kk] + alpha * X[kk + 1], (1 - alpha) * Y[kk] + alpha * Y[kk + 1]])
        gres = pred - gxy
        gnss_bias = None
        if c.use_gnss_bias:
            gnss_bias = np.column_stack([x[self.ig(np.arange(self.nb), 0)], x[self.ig(np.arange(self.nb), 1)]])
        return {'x': X, 'y': Y, 'yaw': wrap(P), 'bias': x[3 * self.N:3 * self.N + self.nb], 'chi2_hist': hist,
                'gnss_residual': gres, 'gnss_rms': float(np.sqrt(np.mean(np.sum(gres ** 2, axis=1)))),
                'gnss_used': use, 'state': x, 'gnss_bias': gnss_bias}

    @staticmethod
    def _huber(r, sigma, delta):
        a = np.abs(r)
        w = np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-12))
        return w / sigma ** 2

    def scan_tau(self, taus, verbose=False):
        """對每個 tau 解一次，回傳 GNSS 殘差 RMS（NHC＋IMU 航向會讓錯的 tau 殘差變大）。"""
        out = []
        for tau in taus:
            r = self.solve(tau=tau, margin=max(0.4, float(np.max(np.abs(taus))) + 0.05))
            out.append({'tau': float(tau), 'gnss_rms': r['gnss_rms'], 'chi2': r['chi2_hist'][-1]})
            if verbose:
                print(out[-1])
        return out
