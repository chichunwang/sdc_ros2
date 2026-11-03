"""點雲對地圖的配準：地圖法向量、point-to-plane ICP（6-DOF，Huber），以及每幀的可觀測性資訊。"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .mapio import voxel_downsample_tiled


def estimate_normals(xyz, k=12, workers=-1, chunk=250_000):
    """每點取 k 鄰居做 PCA；回傳 normals (N,3)（z 分量取正）與 planarity (N,)。分批算，省記憶體。"""
    xyz = np.asarray(xyz, dtype=np.float64)
    tree = cKDTree(xyz)
    normals = np.empty((len(xyz), 3), dtype=np.float32)
    planarity = np.empty(len(xyz), dtype=np.float32)
    for s in range(0, len(xyz), chunk):
        e = min(s + chunk, len(xyz))
        _, idx = tree.query(xyz[s:e], k=k, workers=workers)
        nb = xyz[idx]                                   # (n,k,3)
        nb -= nb.mean(axis=1, keepdims=True)
        cov = np.einsum('nki,nkj->nij', nb, nb) / k
        w, v = np.linalg.eigh(cov)                      # 升冪
        n = v[:, :, 0]
        n[n[:, 2] < 0] *= -1
        normals[s:e] = n
        planarity[s:e] = (w[:, 1] - w[:, 0]) / np.maximum(w[:, 2], 1e-12)
    return normals, planarity


def prepare_map(xyz, voxel=0.2, k=12, extra=None):
    ds, ex, counts = voxel_downsample_tiled(xyz, voxel, extra=extra)
    normals, planarity = estimate_normals(ds, k)
    out = {'xyz': ds.astype(np.float32), 'normals': normals.astype(np.float32),
           'planarity': planarity.astype(np.float32), 'counts': counts.astype(np.int32)}
    if ex is not None:
        out['intensity'] = ex[:, 0].astype(np.float32)
    return out


def pose_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def icp_point_to_plane(src, map_xyz, map_normals, tree, R0, t0, max_iter=25, max_corr=1.0, huber=0.3,
                       prior=None, tol_t=1e-4, tol_r=1e-5, src_weights=None, estimate_scale=False, s0=1.0, workers=-1,
                       src_ground=None, map_ground=None):
    """src (N,3) 在感測器座標（公尺，或乘上 s 後為公尺）；R0,t0 初值（map ← sensor）。
    prior: dict(mean_t, mean_yaw, info=6x6[, mean_log_s, info_s]) 可選的高斯先驗（state = [t, θ(, log s)]）。
    estimate_scale：把尺度 s 當第 7 個未知數（p_map = s R p + t），回傳 's' 與 'scale_info'。
    src_ground／map_ground：bool 遮罩；給了就只接受「同類」配對（地面配地面、非地面配非地面），
    地圖裡沒有的車、雜物就配不到東西，不會被拉去貼路面。
    回傳 dict：R, t, iters, residual stats, inliers, information (6x6, 僅資料項), converged。"""
    R, t, s = np.array(R0, dtype=float), np.array(t0, dtype=float), float(s0)
    n_src = len(src)
    w_src = np.ones(n_src) if src_weights is None else np.asarray(src_weights, dtype=float)
    ndof = 7 if estimate_scale else 6
    info = np.zeros((6, 6))
    scale_info = 0.0
    converged = False
    stats = {}
    for it in range(max_iter):
        p = (s * src) @ R.T + t
        d, j = tree.query(p, distance_upper_bound=max_corr, workers=workers)
        ok = np.isfinite(d)
        if src_ground is not None and map_ground is not None:
            jj = np.where(ok, j, 0)
            ok &= (map_ground[jj] == src_ground)
        if ok.sum() < 20:
            stats = {'inliers': int(ok.sum()), 'rmse': float('nan'), 'median': float('nan')}
            break
        n = map_normals[j[ok]]
        q = map_xyz[j[ok]]
        rp = p[ok]
        r = np.einsum('ij,ij->i', n, rp - q)
        # Huber 權重
        ar = np.abs(r)
        w = np.where(ar <= huber, 1.0, huber / np.maximum(ar, 1e-12)) * w_src[ok]
        J = np.empty((ok.sum(), ndof))
        J[:, :3] = n
        J[:, 3:6] = np.cross(rp - t, n)     # d r / d θ，s R p = rp - t
        if estimate_scale:
            J[:, 6] = np.einsum('ij,ij->i', n, rp - t)   # d r / d(log s)
        H = (J * w[:, None]).T @ J
        g = (J * w[:, None]).T @ r
        info = H[:6, :6].copy()
        if estimate_scale:
            scale_info = float(H[6, 6])
        if prior is not None:
            dx = np.zeros(ndof)
            dx[:3] = t - prior['mean_t']
            dx[5] = _wrap(_yaw(R) - prior['mean_yaw'])
            H[:6, :6] += prior['info']
            g[:6] += prior['info'] @ dx[:6]
            if estimate_scale and 'mean_log_s' in prior:
                # 尺度先驗：log s 的高斯
                H[6, 6] += prior['info_s']
                g[6] += prior['info_s'] * (np.log(s) - prior['mean_log_s'])
        try:
            delta = -np.linalg.solve(H + 1e-9 * np.eye(ndof), g)
        except np.linalg.LinAlgError:
            break
        dt, dth = delta[:3], delta[3:6]
        R = Rotation.from_rotvec(dth).as_matrix() @ R
        t = t + dt
        if estimate_scale:
            s *= float(np.exp(np.clip(delta[6], -0.2, 0.2)))
        stats = {'inliers': int(ok.sum()), 'rmse': float(np.sqrt(np.mean(r ** 2))),
                 'median': float(np.median(ar)), 'inlier_frac': float(ok.mean())}
        if np.linalg.norm(dt) < tol_t and np.linalg.norm(dth) < tol_r and (not estimate_scale or abs(delta[6]) < 1e-4):
            converged = True
            break
    stats.update({'R': R, 't': t, 's': s, 'iters': it + 1, 'information': info, 'scale_info': scale_info,
                  'converged': converged})
    return stats


def _yaw(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def observability(info, scale_rot=np.deg2rad(1.0), scale_t=0.1):
    """把資訊矩陣用 [0.1 m, 1°] 的單位正規化後做特徵分解；回傳 xy、yaw 方向的最小資訊與 rank。"""
    S = np.diag([1 / scale_t] * 3 + [1 / scale_rot] * 3)
    A = np.linalg.inv(S) @ info @ np.linalg.inv(S)   # 單位：每 (0.1 m)² / 每 (1°)²
    w, v = np.linalg.eigh(A)
    xy_info = np.linalg.eigvalsh(A[:2, :2])
    return {'eig': w, 'vec': v, 'xy_min': float(xy_info.min()), 'xy_max': float(xy_info.max()),
            'yaw': float(A[5, 5]), 'z': float(A[2, 2]),
            'rank': int((w >= max(1e-6, 0.02 * w.max())).sum())}
