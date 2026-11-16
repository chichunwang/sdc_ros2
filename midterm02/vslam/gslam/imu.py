"""IMU 加速度計的「平均速度變化」量測：檢查與約束軌跡的局部尺度（只有 Track01 有 IMU）。

為什麼需要：BA 的轉角已與陀螺儀吻合（doc/03），剩下的誤差在「沿行進方向每段路有多長」。影像量不出公尺，
LoGeR 深度的尺度會飄；加速度計是唯一公制、與影像無關的量測。位置要積分兩次，誤差會累積，
所以只積分一次，比較相鄰兩個 1 秒窗的「平均速度」差。

模型（世界座標 = 軌跡的座標；λ = 軌跡單位換成公尺的整體尺度）：
  真實加速度  a(t) = R_wc(t)·(f(t) − b) + g        f：加速度計讀數（已轉到相機座標）、b：偏差、g：重力向量
  相機中心    c(t) = IMU 位置 + R_wc(t)·r            r：IMU→相機的槓桿臂（相機座標；只估水平兩軸，垂直分量不可觀）
  幀 k 的平均速度 v̄_k = (p_{k+h} − p_{k−h}) / (t_{k+h} − t_{k−h})，等於相機速度在窗 I_k 的平均。
  兩個窗相減，未知的初速消掉：
      λ·(v̄_2 − v̄_1) = ΔF − ΔB·b + Δt̄·g + ΔQ·r
  ΔF、ΔB、Δt̄ 是 ∫R f、∫R、t 在兩個窗的平均之差，ΔQ 是 (R_{k+h} − R_{k−h})/窗長 之差。
  旋轉取自目前的軌跡（固定）；λ、b、g、r 對目前的位置做線性最小平方（nuisance）。
用途：
  consistency()：擬合後的水平殘差 RMS（m/s）——軌跡的局部長度越符合 IMU，越小（outputs/cache/dev19_imu_accel.py 的同一模型）。
  velocity_factors()：把上式當成 BA 的因子（只作用在位置），λ、b、g、r 每次求解前依目前軌跡重新擬合。
"""
import numpy as np


def load(path, R_cam_imu):
    """回傳 (timestamp_ns int64, 加速度計讀數 (N,3) 已轉到相機座標)，依時間排序。"""
    lines = [r for r in open(path).read().split('\n')[1:] if r]
    t = np.array([int(r.split(',', 1)[0]) for r in lines], dtype=np.int64)
    raw = np.array([[float(x) for x in r.split(',')[8:11]] for r in lines], dtype=np.float64)
    o = np.argsort(t, kind='stable')
    return t[o], raw[o] @ np.asarray(R_cam_imu, dtype=np.float64).T


def _cumtrapz(y, t):
    out = np.zeros_like(y)
    dt = np.diff(t).reshape((-1,) + (1,) * (y.ndim - 1))
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dt, axis=0)
    return out


def _slerp(tc, R, ti):
    from scipy.spatial.transform import Rotation, Slerp
    return Slerp(tc, Rotation.from_matrix(R))(ti).as_matrix()


class Windows:
    """固定一組幀時間與 IMU 資料；依目前的位姿計算各窗的量測項。"""

    def __init__(self, frame_ns, imu_t_ns, f_cam, half_window, offset_s=0.0, lever_axes=(0, 2)):
        self.ct = np.asarray(frame_ns, dtype=np.int64)
        self.tc = (self.ct - self.ct[0]) / 1e9
        ti = (np.asarray(imu_t_ns, dtype=np.int64) - self.ct[0]) / 1e9 - offset_s   # camera time + offset = imu time
        m = (ti >= self.tc[0]) & (ti <= self.tc[-1])
        self.ti, self.f = ti[m], np.asarray(f_cam)[m]
        self.h = int(half_window)
        self.axes = list(lever_axes)

    def terms(self, R):
        """每一幀 k（h ≤ k < N−h）的窗平均：F̄ (n,3)、B̄ (n,3,3)、t̄ (n,)、Q (n,3,3)、窗長 T (n,)。"""
        h, tc, ti = self.h, self.tc, self.ti
        Rw = _slerp(tc, R, ti)
        F = _cumtrapz(np.einsum('nij,nj->ni', Rw, self.f), ti)
        B = _cumtrapz(Rw, ti)
        k = np.arange(h, len(tc) - h)
        a, b = tc[k - h], tc[k + h]
        T = b - a

        def wmean(Y):
            C = _cumtrapz(Y, ti).reshape(len(ti), -1)
            ia = np.stack([np.interp(a, ti, C[:, j]) for j in range(C.shape[1])], 1)
            ib = np.stack([np.interp(b, ti, C[:, j]) for j in range(C.shape[1])], 1)
            return ((ib - ia) / T[:, None]).reshape((len(k),) + Y.shape[1:])
        Q = (R[k + h] - R[k - h]) / T[:, None, None]
        return dict(k=k, F=wmean(F), B=wmean(B), t=wmean(ti[:, None])[:, 0], Q=Q, T=T)

    def system(self, p, R, k1, k2, terms=None):
        """方程 λ·dv − dQ·r + dB·b − dt·g = dF 的係數。回傳 (A, y, dv, parts)；未知數順序 [λ, b(3), r(水平), g(3)]。"""
        tm = terms or self.terms(R)
        i1, i2 = k1 - self.h, k2 - self.h   # 在 terms 陣列中的索引
        h = self.h
        v1 = (p[k1 + h] - p[k1 - h]) / tm['T'][i1][:, None]
        v2 = (p[k2 + h] - p[k2 - h]) / tm['T'][i2][:, None]
        dv = v2 - v1
        dF = tm['F'][i2] - tm['F'][i1]
        dB = tm['B'][i2] - tm['B'][i1]
        dQ = (tm['Q'][i2] - tm['Q'][i1])[:, :, self.axes]
        dt = tm['t'][i2] - tm['t'][i1]
        A = np.hstack([dv.reshape(-1, 1), dB.reshape(-1, 3), -dQ.reshape(-1, len(self.axes)),
                       np.kron(-dt[:, None], np.eye(3))])
        return A, dF.reshape(-1), dv, dict(dF=dF, dB=dB, dQ=dQ, dt=dt)

    def fit(self, p, R, k1, k2, terms=None):
        """最小平方擬合 λ、b、r、g；回傳 dict（含殘差、水平／垂直 RMS）。"""
        A, y, dv, parts = self.system(p, R, k1, k2, terms)
        x, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = (y - A @ x).reshape(-1, 3)
        na = len(self.axes)
        g = x[4 + na:7 + na]
        up = -g / np.linalg.norm(g)
        rv = res @ up
        rh = res - rv[:, None] * up
        lever = np.zeros(3)
        lever[self.axes] = x[4:4 + na]
        return dict(lam=float(x[0]), b=x[1:4], lever=lever, g=g, up=up, parts=parts,
                    rms_h=float(np.sqrt(np.mean(np.sum(rh ** 2, 1)))), rms_v=float(np.sqrt(np.mean(rv ** 2))),
                    res=res, res_h=rh)


def split_poses(poses):
    P = np.stack([np.asarray(T, dtype=np.float64) for T in poses])
    return P[:, :3, 3].copy(), P[:, :3, :3].copy()


def eval_pairs(n_frames, h, k_sep, stride, mask=None):
    """評估用的窗對（與 dev19 相同：h=5、相隔 20 幀、每 2 幀一組）；mask(k1,k2)→bool 可只取部分。"""
    k1 = np.arange(h, n_frames - h - k_sep, stride)
    k2 = k1 + k_sep
    if mask is not None:
        keep = mask(k1 - h, k2 + h)
        k1, k2 = k1[keep], k2[keep]
    return k1, k2


def consistency(win, poses, k_sep=20, stride=2, mask=None):
    """IMU 一致性指標：擬合後的水平殘差 RMS（m/s）。mask 用來只看保留下來（未進最佳化）的時段。"""
    p, R = split_poses(poses)
    k1, k2 = eval_pairs(len(p), win.h, k_sep, stride, mask)
    return win.fit(p, R, k1, k2)


def factor_pairs(n_frames, h, mask=None):
    """因子用的窗對：相鄰、不重疊的窗（中心相隔 2h），每段 IMU 資料只用一次。"""
    k1 = np.arange(h, n_frames - 3 * h, 2 * h)
    k2 = k1 + 2 * h
    if mask is not None:
        keep = mask(k1 - h, k2 + h)
        k1, k2 = k1[keep], k2[keep]
    return k1, k2


def velocity_factors(win, poses, k1, k2, sigma_m, huber_k, fit=None):
    """回傳 (fn(graph, n_poses), info)：每組窗對一個因子，約束 v̄_2 − v̄_1 = 目標（軌跡單位/s）。

    目標 = (ΔF − ΔB·b + Δt̄·g + ΔQ·r) / λ，λ、b、g、r 由 fit（目前軌跡的擬合）給定、在這次求解中固定。
    殘差只與三個位置有關（p_{k1−h}、p_{k1+h}=p_{k2−h}、p_{k2+h}），對旋轉的導數為 0。
    標準差 sigma_m（m/s，每軸）換成軌跡單位：sigma_m / λ。
    """
    import gtsam
    p, R = split_poses(poses)
    fit = fit or win.fit(p, R, k1, k2)
    parts, lam, h = fit['parts'], fit['lam'], win.h
    na = len(win.axes)
    rr = fit['lever'][win.axes]
    target = (parts['dF'] - np.einsum('nij,j->ni', parts['dB'], fit['b']) + parts['dt'][:, None] * fit['g'][None, :]
              + np.einsum('nij,j->ni', parts['dQ'], rr)) / lam
    T = win.tc
    model = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(huber_k),
                                           gtsam.noiseModel.Isotropic.Sigma(3, sigma_m / lam))
    specs = []
    for a, b, c in zip(k1.tolist(), k2.tolist(), target):
        T1 = T[a + h] - T[a - h]
        T2 = T[b + h] - T[b - h]
        coef = {a - h: 1.0 / T1, a + h: -1.0 / T1}        # −v̄_1
        coef[b - h] = coef.get(b - h, 0.0) - 1.0 / T2
        coef[b + h] = coef.get(b + h, 0.0) + 1.0 / T2
        keys = [k for k in coef if coef[k] != 0.0]
        specs.append((keys, [coef[k] for k in keys], np.asarray(c, dtype=np.float64)))

    def make(keys, cs, c):
        def error(this, values, H):
            r = -c.copy()
            for q, (key, w) in enumerate(zip(keys, cs)):
                pose = values.atPose3(key)
                r += w * pose.translation()
                if H is not None:
                    J = np.zeros((3, 6))
                    J[:, 3:] = w * pose.rotation().matrix()
                    H[q] = J
            return r
        return error

    def add(graph, n_poses):
        for keys, cs, c in specs:
            graph.add(gtsam.CustomFactor(model, gtsam.KeyVector(keys), make(keys, cs, c)))
    info = dict(factors=len(specs), lam=lam, lever=fit['lever'].tolist(), gravity_norm=float(np.linalg.norm(fit['g'])),
                accel_bias=fit['b'].tolist(), fit_rms_h=fit['rms_h'], fit_rms_v=fit['rms_v'], sigma_m=float(sigma_m))
    return add, info


def block_mask(block_frames, keep_even):
    """驗證用：把時間切成 block_frames 幀一塊，只保留偶數（或奇數）塊；窗對整段落在保留塊內才算。"""
    def mask(lo, hi):
        b_lo, b_hi = lo // block_frames, hi // block_frames
        same = b_lo == b_hi
        return same & ((b_lo % 2 == 0) == keep_even)
    return mask
