"""小矩陣資訊分析：尺度先邊際化，map 只提供可觀測的 pose 量測。

這是局部線性／angular-cell 誤差模型，並非已校準的 ICP 真實 covariance。
XYZ 點級殘差與 Hessian 已在 CUDA 累加；此模組不處理點雲。
"""
import numpy as np
from . import parameters as p


def checked_covariance(covariance):
    C = np.asarray(covariance, dtype=float)
    if (C.shape != (3, 3) or not np.isfinite(C).all()
            or not np.allclose(C, C.T, rtol=1e-8, atol=1e-12)
            or np.linalg.eigvalsh(C).min() <= 0):
        raise ValueError('Expected positive-definite map [x,y,yaw] covariance')
    return (C+C.T)*.5


def geometry_observation(data_information, correction_jacobian, prior_covariance, quality):
    H = np.asarray(data_information, dtype=float)
    J = np.asarray(correction_jacobian, dtype=float)
    C = checked_covariance(prior_covariance)
    if H.shape != (4, 4) or J.shape != (3, 4) or not np.isfinite(H).all() or not np.isfinite(J).all():
        raise ValueError('Invalid map information/Jacobian')
    H = (H+H.T)*.5
    # y = [world dx,dy,dyaw,logscale]. Retain slope/scale/XY cross terms.
    A = np.vstack([J, [0., 0., 0., 1.]])
    inverse = np.linalg.inv(A)
    world_H = inverse.T @ H @ inverse
    L = np.linalg.cholesky(C)
    units = np.zeros((4, 4)); units[:3, :3] = L; units[3, 3] = p.MAX_SCALE_LOG_STD
    normalized = units.T @ world_H @ units
    normalized = (normalized+normalized.T)*.5
    # Scale is an uncertain nuisance, NOT a fixed exact calibration. A scalar
    # Schur complement must include even small positive scale information.
    pose_info = normalized[:3, :3].copy()
    if normalized[3, 3] > 0:
        pose_info -= np.outer(normalized[:3, 3], normalized[3, :3])/normalized[3, 3]
    pose_info = (pose_info+pose_info.T)*.5
    values, vectors = np.linalg.eigh(pose_info)
    tolerance = max(1., float(np.linalg.norm(normalized, ord=2)))*1e-8
    if values.min() < -tolerance:
        raise ValueError('Map information is not positive semidefinite after marginalizing scale')
    values = np.maximum(values, 0.)
    threshold = max(p.MAP_MIN_RELATIVE_INFORMATION, float(values.max())*1e-10)
    keep = values >= threshold
    # Measurement coordinates z = V_good.T L^-1 [dx,dy,dyaw]. Null directions
    # have NO measurement row; never use pinv(H) as zero uncertainty there.
    basis = vectors[:, keep]
    observation = np.linalg.solve(L.T, basis).T
    floor = np.diag([p.CORRECTION_MEAS_POSITION_FLOOR_M**2]*2+
                    [np.deg2rad(p.CORRECTION_MEAS_YAW_FLOOR_DEG)**2])
    coverage = float(quality['coverage'])
    median = float(quality['median_plane_residual_m'])
    if not 0 < coverage <= 1 or not np.isfinite([coverage, median]).all() or median < 0:
        raise ValueError('Invalid point-to-plane quality')
    # H already sums ONLY reliable angular-weighted plane observations.
    # A low whole-image plane fraction is not extra noise on those observed
    # modes; inverse(H) already loses information as support decreases.
    inflation = max(1., (median/p.FEATURE_STD_M)**2)
    R = (np.diag(1/values[keep])+observation @ floor @ observation.T)*inflation
    # Independent diagnosis of scale after marginalizing ALL pose directions.
    scale_info = float(normalized[3, 3]-normalized[3, :3] @
                       np.linalg.pinv(normalized[:3, :3], rcond=1e-10) @ normalized[:3, 3])
    scale_info = max(0., scale_info)/p.MAX_SCALE_LOG_STD**2
    projector = L @ basis @ basis.T @ np.linalg.inv(L)
    diag = dict(rank=int(keep.sum()), normalized_eigenvalues=values.tolist(),
                information_threshold=float(threshold), observation_matrix=observation.tolist(),
                observable_projector_world=projector.tolist(),
                pose_information_normalized=pose_info.tolist(),
                measurement_covariance=R.tolist(), quality_inflation=float(inflation),
                scale_information=scale_info,
                scale_log_std=float(1/np.sqrt(max(scale_info, 1e-20))),
                convention='world XY/yaw; current prior whitening; scale marginalized; weak modes omitted')
    return observation, R, diag
