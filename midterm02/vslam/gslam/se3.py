"""SE(3) 小工具（NumPy，CPU）：4×4 pose 的合成、反算、指數映射與插值。"""
import numpy as np


def hat(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def so3_exp(w):
    """旋轉向量 -> 旋轉矩陣（Rodrigues）。"""
    w = np.asarray(w, dtype=np.float64)
    theta = np.linalg.norm(w)
    K = hat(w)
    if theta < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + np.sin(theta) / theta * K + (1 - np.cos(theta)) / theta ** 2 * K @ K


def so3_log(R):
    """旋轉矩陣 -> 旋轉向量。"""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-8:
        return 0.5 * v
    if np.pi - theta < 1e-5:  # 接近 180 度：用對稱部分求軸
        A = (R + np.eye(3)) / 2
        axis = np.sqrt(np.maximum(np.diag(A), 0))
        k = int(np.argmax(axis))
        axis = A[:, k] / max(axis[k], 1e-12)
        if np.dot(axis, v) < 0:
            axis = -axis
        return theta * axis / np.linalg.norm(axis)
    return theta / (2 * np.sin(theta)) * v


def make(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    return make(R.T, -R.T @ t)


def left_update(T, delta):
    """左乘小量更新：R <- Exp(w) R，t <- Exp(w) t + v；delta = (v, w)。"""
    dR = so3_exp(delta[3:6])
    return make(dR @ T[:3, :3], dR @ T[:3, 3] + delta[:3])


def project_rotation(R):
    """把接近旋轉的矩陣投影回 SO(3)（SVD，det=+1）。"""
    U, _, Vt = np.linalg.svd(R)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def normalize(T):
    return make(project_rotation(T[:3, :3]), T[:3, 3])


def rotation_angle(R):
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def relative_error(Ta, Tb):
    """兩個 pose 的差：(平移差 [m]，旋轉差 [rad])。"""
    d = inv(Ta) @ Tb
    return float(np.linalg.norm(d[:3, 3])), rotation_angle(d[:3, :3])


def is_valid(T, tol=1e-6):
    T = np.asarray(T)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    R = T[:3, :3]
    return bool(np.allclose(R.T @ R, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1) < tol
                and np.allclose(T[3], [0, 0, 0, 1], atol=tol))


def interpolate(Ta, Tb, alpha):
    """在 SE(3) 上內插（不平均旋轉矩陣）：旋轉走測地線 Exp(alpha*Log(Ra^T Rb))，平移線性。"""
    Ra, Rb = Ta[:3, :3], Tb[:3, :3]
    R = Ra @ so3_exp(alpha * so3_log(Ra.T @ Rb))
    t = (1 - alpha) * Ta[:3, 3] + alpha * Tb[:3, 3]
    return make(R, t)


def distribute_corrections(odom, kf_indices, kf_opt):
    """把 keyframe 的最佳化結果套用到所有幀。

    odom：每幀的 odometry pose（list of 4×4）；kf_indices：keyframe 的幀索引（遞增）；
    kf_opt：對應的最佳化後 keyframe pose。
    每幀相對前後兩個 keyframe 各有一個候選：opt_A · odom_A⁻¹ · odom_i 與 opt_B · odom_B⁻¹ · odom_i，
    依幀位置在兩者之間做 SE(3) 內插，所以經過 keyframe 時連續、在 keyframe 上等於最佳化值。
    """
    out = [None] * len(odom)
    if not kf_indices:
        return [T.copy() for T in odom]
    for k in range(len(kf_indices) - 1):
        a, b = kf_indices[k], kf_indices[k + 1]
        corr_a = kf_opt[k] @ inv(odom[a])
        corr_b = kf_opt[k + 1] @ inv(odom[b])
        for i in range(a, b):
            alpha = (i - a) / float(b - a)
            out[i] = normalize(interpolate(corr_a @ odom[i], corr_b @ odom[i], alpha))
    first, last = kf_indices[0], kf_indices[-1]
    corr = kf_opt[0] @ inv(odom[first])
    for i in range(0, first):
        out[i] = normalize(corr @ odom[i])
    corr = kf_opt[-1] @ inv(odom[last])
    for i in range(last, len(odom)):
        out[i] = normalize(corr @ odom[i])
    return out
