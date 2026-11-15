"""GPU 點雲配準：point-to-plane ICP 與 Generalized ICP（plane-to-plane），多尺度、可選尺度自由度。

模型（source 點 p 在相機座標，target 在世界座標）：
    q = s · R · p + t                      （s 預設固定為 1）
左乘小量更新 R ← Exp(ω)R、t ← Exp(ω)t + v、s ← s·exp(σ)，對 δ = (v, ω, σ) 的導數：
    ∂q/∂v = I，∂q/∂ω = −[q]×，∂q/∂σ = q − t
殘差：
    point-to-plane：e = nᵀ(q − m)，n 為最近 target 點 m 的法向
    GICP：          r = q − m，權重矩陣 M = (C_m + s²·R·C_p·Rᵀ)⁻¹，代價 rᵀ M r
兩者都用 Cauchy 權重降低錯誤對應的影響。對應搜尋、殘差與 H、g 累加都在 CUDA；只有 6×6／7×7 線性解在 CPU。
"""
import numpy as np
import cupy as cp

from . import se3
from .gpu_cloud import local_geometry

METHODS = ('icp_plane', 'gicp')


def source_covariances(points, radius, min_neighbors, planar_eps=1e-3):
    """GICP 用：source 點（相機座標）的正則化共變異數；鄰居不足的點回傳 valid=False。"""
    valid, _, cov = local_geometry(points, radius, min_neighbors, planar_eps=planar_eps)
    return valid, cov


def _neg_hat(q):
    """批次 −[q]×，形狀 (N, 3, 3)。"""
    J = cp.zeros((len(q), 3, 3), dtype=cp.float64)
    J[:, 0, 1] = q[:, 2]; J[:, 0, 2] = -q[:, 1]
    J[:, 1, 0] = -q[:, 2]; J[:, 1, 2] = q[:, 0]
    J[:, 2, 0] = q[:, 1]; J[:, 2, 1] = -q[:, 0]
    return J


def _linearize(method, src, src_cov, model, T, scale, max_corr, robust_k):
    R = cp.asarray(T[:3, :3]); t = cp.asarray(T[:3, 3])
    q = scale * (src @ R.T) + t
    idx, _ = model.index.nearest(q, max_corr)
    ok = idx >= 0
    n = int(ok.sum())
    if n < 6:
        return None
    qi = q[ok]; ii = idx[ok]
    r = qi - model.points[ii]
    lever = qi - t
    if method == 'icp_plane':
        nrm = model.normals[ii]
        e = cp.sum(nrm * r, axis=1)
        w = robust_k ** 2 / (robust_k ** 2 + e * e)
        A = cp.concatenate([nrm, cp.cross(qi, nrm), cp.sum(nrm * lever, axis=1, keepdims=True)], axis=1)
        H = (A * w[:, None]).T @ A
        g = A.T @ (w * e)
        unit = 1.0
    else:
        C = model.cov[ii] + (scale ** 2) * (R[None] @ src_cov[ok] @ R.T[None])
        M = cp.linalg.inv(C)
        Mr = cp.einsum('nij,nj->ni', M, r)
        # 平面對平面時 rᵀMr ≈ (法向距離)² / (2·eps)，換回公尺平方後再套 Cauchy
        unit = 1.0 / (2.0 * model.planar_eps)
        e2 = cp.sum(r * Mr, axis=1) / unit
        w = robust_k ** 2 / (robust_k ** 2 + e2)
        J = cp.zeros((n, 3, 7), dtype=cp.float64)
        J[:, 0, 0] = 1.0; J[:, 1, 1] = 1.0; J[:, 2, 2] = 1.0
        J[:, :, 3:6] = _neg_hat(qi)
        J[:, :, 6] = lever
        MJ = cp.einsum('nab,nbj->naj', M, J)
        H = cp.einsum('nai,naj,n->ij', J, MJ, w)
        g = cp.einsum('nai,na,n->i', J, Mr, w)
    return cp.asnumpy(H), cp.asnumpy(g), n, unit


def evaluate(src, model, T, scale, max_corr):
    """配準品質：fitness（有對應的比例）、inlier RMSE [m]、對應數。"""
    R = cp.asarray(T[:3, :3]); t = cp.asarray(T[:3, 3])
    q = scale * (src @ R.T) + t
    idx, d2 = model.index.nearest(q, max_corr)
    ok = idx >= 0
    n = int(ok.sum())
    rmse = float(cp.sqrt(cp.mean(d2[ok]))) if n else float('inf')
    return dict(fitness=n / max(len(src), 1), inlier_rmse=rmse, correspondences=n)


def register(src, model, T_init, method='gicp', stages=None, robust_k=0.1, src_cov=None,
             scale_init=1.0, estimate_scale=False, scale_prior_weight=25.0, min_correspondences=200,
             prior_T=None, prior_weights=None):
    """把 source（相機座標）配準到 target model（世界座標），回傳 dict。

    stages：[{max_corr: 對應距離上限 [m], iterations: 最多迭代次數}, ...]，由粗到細。
    prior_T / prior_weights=(w_t, w_r)：把另一個位姿量測（影像 PnP）當成先驗一起最佳化。
    誤差 ξ 定義為 T = Exp(ξ)·prior_T 的左乘小量：ξ_ω = Log(R·R_pᵀ)，ξ_v = t − R·R_pᵀ·t_p；
    代價多加 n·(w_t·|ξ_v|² + w_r·|ξ_ω|²)，n 為對應數（w 是「每個對應的權重」，見 pipeline 的推導）。
    estimate_scale=True 時多解一個 log 尺度 σ，並以 scale_prior_weight 把它拉向 scale_init
    （每個對應貢獻 weight·σ²，等同「點位雜訊／尺度隨機漫步」變異數比）。
    """
    if method not in METHODS:
        raise ValueError('method 必須是 %s' % (METHODS,))
    if not se3.is_valid(T_init, 1e-4):
        raise ValueError('配準初值不是合法 pose')
    src = cp.asarray(src, dtype=cp.float64)
    if method == 'gicp':
        if src_cov is None:
            raise ValueError('gicp 需要 src_cov（見 source_covariances）')
        src_cov = cp.asarray(src_cov, dtype=cp.float64)
    stages = stages or [dict(max_corr=1.0, iterations=20), dict(max_corr=0.4, iterations=30)]
    T = se3.normalize(np.asarray(T_init, dtype=np.float64))
    scale = float(scale_init)
    dim = 7 if estimate_scale else 6
    iterations = 0
    converged = False
    reason = 'max_iterations'
    H_last = None
    for stage in stages:
        converged = False
        reason = 'max_iterations'
        for _ in range(int(stage['iterations'])):
            data = _linearize(method, src, src_cov, model, T, scale, float(stage['max_corr']), robust_k)
            if data is None or data[2] < min_correspondences:
                reason = 'insufficient_correspondences'
                break
            H, g, n, unit = data
            if estimate_scale:
                prior = n * scale_prior_weight * unit
                H[6, 6] += prior
                g[6] += prior * np.log(scale / scale_init)
            if prior_T is not None:
                dR = T[:3, :3] @ prior_T[:3, :3].T
                wt, wr = n * prior_weights[0] * unit, n * prior_weights[1] * unit
                H[:3, :3] += wt * np.eye(3); g[:3] += wt * (T[:3, 3] - dR @ prior_T[:3, 3])
                H[3:6, 3:6] += wr * np.eye(3); g[3:6] += wr * se3.so3_log(dR)
            Hs, gs = H[:dim, :dim], g[:dim]
            damp = 1e-6 * np.diag(np.maximum(np.diag(Hs), 1e-9))
            try:
                delta = np.linalg.solve(Hs + damp, -gs)
            except np.linalg.LinAlgError:
                reason = 'singular_system'
                break
            # 信賴範圍：單步平移不超過對應距離、旋轉不超過 5 度、尺度不超過 2%
            f = max(1.0, np.linalg.norm(delta[:3]) / float(stage['max_corr']),
                    np.linalg.norm(delta[3:6]) / np.deg2rad(5.0),
                    (abs(delta[6]) / 0.02) if estimate_scale else 0.0)
            delta = delta / f
            T = se3.left_update(T, delta[:6])
            if estimate_scale:
                scale *= float(np.exp(delta[6]))
            iterations += 1
            H_last = H[:6, :6]
            if (np.linalg.norm(delta[:3]) < 1e-4 and np.linalg.norm(delta[3:6]) < 1e-5
                    and (not estimate_scale or abs(delta[6]) < 1e-5)):
                converged = True
                reason = 'converged'
                break
        if reason in ('insufficient_correspondences', 'singular_system'):
            break
    quality = evaluate(src, model, T, scale, float(stages[-1]['max_corr']))
    # 退化指標：把 6×6 資訊矩陣以對角線正規化後的最小特徵值（接近 0 代表某個方向沒有幾何約束）
    min_eig = None
    if H_last is not None:
        d = np.sqrt(np.maximum(np.diag(H_last), 1e-12))
        min_eig = float(np.linalg.eigvalsh(H_last / d[:, None] / d[None, :]).min())
    return dict(T=se3.normalize(T), scale=scale, method=method, converged=bool(converged), reason=reason,
                iterations=iterations, normalized_min_eigenvalue=min_eig, **quality)
