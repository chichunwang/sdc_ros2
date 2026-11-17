"""DROID 變體用的閉環工具（D8 起）：閉環相對位姿量測（SIFT＋LoGeR 深度＋PnP）與 SE(3) pose graph。

為什麼需要（2026-11-17，Track2）：DROID 的全域 BA 只能修正「光流初值找得到對應」的閉環。Track2 的起點與終點
（影像確認是同一處）在 DROID 估計中相差約 34 m，光流初值太遠接不上。先用量測到的相對位姿做 pose graph 把軌跡粗修到
閉環附近，再讓 DROID 的全域 BA（含閉環光流邊）細修。

座標：Twc（camera-to-world，4×4）；相對位姿 T_ab = Twc_a⁻¹ · Twc_b（b 在 a 相機座標下的位姿）。
"""
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


def inv(T):
    out = np.repeat(np.eye(4)[None], len(T), 0) if T.ndim == 3 else np.eye(4)
    R, t = T[..., :3, :3], T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -(Rt @ t[..., None])[..., 0]
    return out


# ---------------------------------------------------------------- 閉環量測
def depth_cells(xyz, K0, Kc, dist, size, cell=4):
    """LoGeR 點雲 → 去畸變全解析度影像上的深度網格（每 cell×cell 像素取中位數，0＝無資料）。

    點雲是官方針孔 K0 從原始像素網格反投影（見 run_droid.loger_depth）；先投影回原始像素，再用自校正畸變校正到 Kc 影像。
    """
    z = xyz[:, 2]
    keep = np.isfinite(z) & (z > 0.1) & (z < 49.5)
    xyz, z = xyz[keep], z[keep]
    uv = (xyz @ K0.T)[:, :2] / z[:, None]
    if dist is not None:
        uv = cv2.undistortPoints(uv.reshape(-1, 1, 2), Kc, dist, P=Kc).reshape(-1, 2)
    w, h = size
    gw, gh = w // cell, h // cell
    c, r = np.floor(uv[:, 0] / cell).astype(int), np.floor(uv[:, 1] / cell).astype(int)
    ok = (c >= 0) & (c < gw) & (r >= 0) & (r < gh)
    idx, z = r[ok] * gw + c[ok], z[ok]
    order = np.lexsort((z, idx))
    idx, z = idx[order], z[order]
    starts = np.flatnonzero(np.r_[True, np.diff(idx) > 0])
    counts = np.diff(np.r_[starts, len(idx)])
    grid = np.zeros(gh * gw)
    grid[idx[starts]] = z[starts + (counts - 1) // 2]
    return grid.reshape(gh, gw)


def relative_pose_pnp(img_a, img_b, depth_b, Kc, cell=4, ratio=0.8, min_inliers=20, reproj_px=2.0, seed=0):
    """影像 a、b（已去畸變的灰階）＋ b 的深度網格 → T_ab（b 在 a 相機座標下），失敗回傳 None。

    b 的特徵點用深度反投影成 3D（b 相機座標），在 a 上做 PnP：X_a = R X_b + t，即 T_ab = [R|t]。
    """
    sift = cv2.SIFT_create(nfeatures=4000)
    ka, da = sift.detectAndCompute(img_a, None)
    kb, db = sift.detectAndCompute(img_b, None)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return None, dict(reason='few_features')
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(db, da, k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < min_inliers:
        return None, dict(reason='few_matches', matches=len(good))
    pb = np.array([kb[g.queryIdx].pt for g in good])
    pa = np.array([ka[g.trainIdx].pt for g in good])
    gh, gw = depth_b.shape
    c = np.clip((pb[:, 0] / cell).astype(int), 0, gw - 1)
    r = np.clip((pb[:, 1] / cell).astype(int), 0, gh - 1)
    z = depth_b[r, c]
    ok = z > 0
    if ok.sum() < min_inliers:
        return None, dict(reason='few_depth', matches=len(good))
    Kinv = np.linalg.inv(Kc)
    X = (np.c_[pb[ok], np.ones(ok.sum())] @ Kinv.T) * z[ok, None]
    cv2.setRNGSeed(seed)
    found, rvec, tvec, inl = cv2.solvePnPRansac(X, pa[ok], Kc, None, iterationsCount=2000, reprojectionError=reproj_px,
                                                confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not found or inl is None or len(inl) < min_inliers:
        return None, dict(reason='pnp_failed', matches=len(good), inliers=0 if inl is None else len(inl))
    inl = inl[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(X[inl], pa[ok][inl], Kc, None, rvec, tvec)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec[:, 0]
    return T, dict(reason='ok', matches=len(good), inliers=int(len(inl)))


# ---------------------------------------------------------------- pose graph
def _log(T):
    """SE(3) 誤差 → 6 維（旋轉向量, 平移）。小誤差下平移部分直接用 t（pose graph 的常見近似）。"""
    return np.concatenate([Rotation.from_matrix(T[:, :3, :3]).as_rotvec(), T[:, :3, 3]], axis=1)


def _exp(x):
    T = np.repeat(np.eye(4)[None], len(x), 0)
    T[:, :3, :3] = Rotation.from_rotvec(x[:, :3]).as_matrix()
    T[:, :3, 3] = x[:, 3:]
    return T


def pose_graph(Twc, loops, sigma_rot_per_unit, sigma_trans_per_unit, loop_sigma_rot, loop_sigma_trans,
               floor_unit=0.05, huber_k=3.0, max_nfev=50):
    """SE(3) pose graph：相鄰節點的邊＝目前估計的相對位姿（標準差 ∝ √移動距離，隨機漫步漂移），閉環邊＝量測的 T_ab。
    sigma_*_per_unit 是每 √單位 的標準差。

    Twc：(N,4,4) 初值。loops：[(a, b, T_ab)] 或 [(a, b, T_ab, σ6)]（σ6：這條邊自己的 6 維標準差，
    順序 [旋轉 x,y,z, 平移 x,y,z]，表示在 b 的相機座標；沒給就用 loop_sigma_*）。回傳 (修正後 Twc, info)。節點 0 固定。
    參數化：Twc_i = Twc_i⁰ · Exp(δ_i)（右乘局部擾動）；殘差 r = Log(Z⁻¹ · Twc_a⁻¹ · Twc_b) / σ；閉環邊用 Huber。
    """
    N = len(Twc)
    T0 = Twc.copy()
    ia = np.arange(N - 1)
    Z_odo = inv(T0[:-1]) @ T0[1:]
    d = np.sqrt(np.maximum(np.linalg.norm(Z_odo[:, :3, 3], axis=1), floor_unit))   # 隨機漫步：變異數與距離成正比
    s_odo = np.c_[np.repeat((sigma_rot_per_unit * d)[:, None], 3, 1), np.repeat((sigma_trans_per_unit * d)[:, None], 3, 1)]
    la = np.array([e[0] for e in loops], dtype=int)
    lb = np.array([e[1] for e in loops], dtype=int)
    Z_loop = np.stack([e[2] for e in loops]) if loops else np.zeros((0, 4, 4))
    s_default = np.array([loop_sigma_rot] * 3 + [loop_sigma_trans] * 3)
    s_loop = np.array([np.asarray(e[3], dtype=float) if len(e) > 3 else s_default for e in loops]).reshape(-1, 6)
    A = np.r_[ia, la]
    B = np.r_[ia + 1, lb]
    Z = np.concatenate([Z_odo, Z_loop])
    S = np.concatenate([s_odo, s_loop])
    n_odo = len(ia)
    Zi = inv(Z)

    def residual(x):
        Tw = T0.copy()
        Tw[1:] = T0[1:] @ _exp(x.reshape(-1, 6))
        r = _log(Zi @ inv(Tw[A]) @ Tw[B]) / S
        rl = r[n_odo:]
        if len(rl):   # 閉環邊：以邊為單位的 Huber（整條邊一起降權，不是逐分量）
            e = np.linalg.norm(rl, axis=1, keepdims=True)
            w = np.where(e > huber_k, np.sqrt(huber_k / np.maximum(e, 1e-12)), 1.0)
            r = np.concatenate([r[:n_odo], rl * w])
        return r.ravel()

    J = lil_matrix((6 * len(A), 6 * (N - 1)), dtype=np.int8)
    for k, (a, b) in enumerate(zip(A, B)):
        for n in (a, b):
            if n > 0:
                J[6 * k:6 * k + 6, 6 * (n - 1):6 * n] = 1
    r0 = residual(np.zeros(6 * (N - 1)))
    sol = least_squares(residual, np.zeros(6 * (N - 1)), jac_sparsity=J.tocsr(), method='trf', x_scale='jac',
                        max_nfev=max_nfev, verbose=0)
    Tw = T0.copy()
    Tw[1:] = T0[1:] @ _exp(sol.x.reshape(-1, 6))
    rl = _log(Zi[n_odo:] @ inv(Tw[A[n_odo:]]) @ Tw[B[n_odo:]]) / S[n_odo:] if len(la) else np.zeros((0, 6))
    rl0 = r0.reshape(-1, 6)[n_odo:]
    info = dict(nodes=N, loop_edges=int(len(la)), cost_initial=float(0.5 * np.sum(r0 ** 2)), cost_final=float(sol.cost),
                nfev=int(sol.nfev), loop_norm_initial_median=float(np.median(np.linalg.norm(rl0, axis=1))) if len(la) else None,
                loop_norm_final_median=float(np.median(np.linalg.norm(rl, axis=1))) if len(la) else None,
                loop_norm_final_max=float(np.max(np.linalg.norm(rl, axis=1))) if len(la) else None,
                max_node_shift=float(np.max(np.linalg.norm(Tw[:, :3, 3] - T0[:, :3, 3], axis=1))))
    return Tw, info
