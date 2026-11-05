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
    continuous = p.v515_enabled() or not p.uses_v52_fusion()
    threshold = (max(1e-10, float(values.max())*p.MAP_NUMERICAL_INFORMATION_RCOND)
                 if continuous else max(p.MAP_MIN_RELATIVE_INFORMATION, float(values.max())*1e-10))
    keep = values >= threshold
    # Measurement coordinates z = V_good.T L^-1 [dx,dy,dyaw]. Null directions
    # have NO measurement row; never use pinv(H) as zero uncertainty there.
    basis = vectors[:, keep]
    observation = np.linalg.solve(L.T, basis).T
    strong = values >= max(p.MAP_MIN_RELATIVE_INFORMATION, threshold)
    strong_observation = np.linalg.solve(L.T, vectors[:, strong]).T
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
                strong_rank=int(strong.sum()), strong_information_threshold=p.MAP_MIN_RELATIVE_INFORMATION,
                strong_observation_matrix=strong_observation.tolist(),
                observable_projector_world=projector.tolist(),
                pose_information_normalized=pose_info.tolist(),
                measurement_covariance=R.tolist(), quality_inflation=float(inflation),
                scale_information=scale_info,
                scale_log_std=float(1/np.sqrt(max(scale_info, 1e-20))),
                convention=('world XY/yaw; current prior whitening; scale marginalized; positive weak modes have large variance; only numerical null omitted' if continuous else 'world XY/yaw; current prior whitening; scale marginalized; weak modes omitted'))
    return observation, R, diag


def likelihood_observation(data_information, data_gradient, correction_jacobian,
                           prior_covariance, quality, candidate_correction):
    """Extract a local DATA likelihood, instead of treating a MAP solution as data.

    Linearize at the converged current-frame candidate. Transform both H and g
    to [world dx,dy,dyaw,logscale], Schur-eliminate scale, and solve ONLY the
    retained information modes. No sensor/height prior is subtracted from an
    approximate stationary total gradient; the data gradient is saved directly.
    """
    O, R, diag = geometry_observation(data_information, correction_jacobian,
                                       prior_covariance, quality)
    g = np.asarray(data_gradient, dtype=float)
    c = np.asarray(candidate_correction, dtype=float)
    if g.shape != (4,) or c.shape != (3,) or not np.isfinite(g).all() or not np.isfinite(c).all():
        raise ValueError('Invalid data likelihood gradient/candidate')
    L = np.linalg.cholesky(checked_covariance(prior_covariance))
    A = np.vstack([correction_jacobian, [0., 0., 0., 1.]])
    inverse = np.linalg.inv(A)
    world_H = inverse.T @ np.asarray(data_information) @ inverse
    world_g = inverse.T @ g
    units = np.zeros((4, 4)); units[:3, :3] = L; units[3, 3] = p.MAX_SCALE_LOG_STD
    Hn = units.T @ world_H @ units; gn = units.T @ world_g
    marginal_g = gn[:3].copy()
    if Hn[3, 3] > 0:
        marginal_g -= Hn[:3, 3]*gn[3]/Hn[3, 3]
    # O L = V_good.T; it shares geometry_observation's exact eigenbasis.
    Vt = O @ L
    information = Vt @ np.asarray(diag['pose_information_normalized']) @ Vt.T
    step_modes = -np.linalg.solve(information, Vt @ marginal_g) if len(O) else np.empty(0)
    step_world = L @ Vt.T @ step_modes
    # The Schur-profiled pose center assumes this CONDITIONAL scale optimum.
    # Validate the nonlinear map loss there, not at a mismatched fixed scale.
    nuisance_step = (-(world_g[3]+world_H[3, :3] @ step_world)/world_H[3, 3]
                     if world_H[3, 3] > 0 else 0.)
    measurement = c+step_world
    # This is a LOCAL quadratic center, not a physical pose to wrap by 2*pi.
    # A weak yaw center can be large; only its finite posterior is emitted.
    diag.update(likelihood_gradient_marginalized=marginal_g.tolist(),
                likelihood_step_world=step_world.tolist(),
                likelihood_candidate_correction=c.tolist(),
                likelihood_measurement_correction=measurement.tolist(),
                likelihood_observed_value=(O @ measurement).tolist(),
                likelihood_nuisance_logscale_step=float(nuisance_step),
                likelihood_model='current fixed-association data-only local quadratic; scale marginalized; sensor prior used once')
    return measurement, O, R, diag


def height_observation(data_information, data_gradient, candidate_height, quality):
    """Local data-only height likelihood, marginalizing ALL planar pose modes.

    x=[plane_x,plane_y,plane_yaw,logscale], h=ds*exp(logscale).
    The MAP candidate already used a height prior. Its DATA gradient recovers
    an independent local measurement instead of feeding that prior back twice.
    No GNSS or height prior enters this marginal information/gradient.
    """
    H = np.asarray(data_information, dtype=float)
    g = np.asarray(data_gradient, dtype=float)
    h = float(candidate_height)
    if H.shape != (4, 4) or g.shape != (4,) or not np.isfinite(H).all() or not np.isfinite(g).all() or not np.isfinite(h) or h <= 0:
        raise ValueError('Invalid data-only height quadratic')
    H = (H+H.T)*.5
    units = np.diag([p.METRIC_POSITION_STD_M]*2+[np.deg2rad(p.METRIC_YAW_STD_DEG)])
    A = units @ H[:3, :3] @ units
    eig, basis = np.linalg.eigh((A+A.T)*.5)
    tolerance = max(1., float(np.linalg.norm(H, ord=2)))*1e-8
    if np.linalg.eigvalsh(H).min() < -tolerance:
        raise ValueError('Height data information is not positive semidefinite')
    keep = eig > max(float(eig.max())*1e-10, 1e-12)
    inverse = units @ ((basis[:, keep]/eig[keep]) @ basis[:, keep].T) @ units
    information = float(H[3, 3]-H[3, :3] @ inverse @ H[:3, 3])
    gradient = float(g[3]-H[3, :3] @ inverse @ g[:3])
    # A scale/translation gauge has no height observation, never zero variance.
    threshold = max(1e-10, abs(float(H[3, 3]))*1e-10)
    diag = dict(information_logscale=max(0., information), marginal_gradient_logscale=gradient,
                nuisance_pose_rank=int(keep.sum()), candidate_height_m=h,
                model='local DATA-only height likelihood; planar XY/yaw marginalized; no sensor/height prior')
    if information < -tolerance:
        raise ValueError('Negative marginalized height information')
    if information <= threshold:
        diag.update(observable=False, reason='height_scale_pose_gauge')
        return None, None, diag
    delta_log = -gradient/information
    # Linearize in METRES: dh=h*dlogscale. The exact nonlinear center is
    # independently checked on the original final planes by the caller.
    measurement = h*(1+delta_log)
    variance = h*h/information+p.HEIGHT_OBSERVATION_FLOOR_M**2
    median = float(quality['median_plane_residual_m'])
    if not np.isfinite(median) or median < 0:
        raise ValueError('Invalid height registration quality')
    inflation = max(1., (median/p.FEATURE_STD_M)**2)
    variance *= inflation
    pose_step = -inverse @ (g[:3]+H[:3, 3]*delta_log)
    diag.update(observable=True, measurement_height_m=float(measurement), variance_m2=float(variance),
                scale_log_std=float(1/np.sqrt(information)), quality_inflation=inflation,
                profiled_pose_step=pose_step.tolist())
    return float(measurement), float(variance), diag
