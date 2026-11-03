"""Current-frame CUDA ground support and sensor-normal consistency checks.

Only a 3x3 scatter matrix leaves CUDA. Thickness/count alone cannot identify
a narrow curb plane; in-plane spread and normal innovation are checked too.
These are explicit engineering uncertainty budgets, not a calibrated sensor.
"""
import cupy as cp
import numpy as np
from track3.initialization import GroundInitializationError, plane_basis
from . import parameters as p


class GroundSupportError(GroundInitializationError):
    def __init__(self, reason, diagnostic):
        super().__init__(reason)
        self.diagnostic = dict(diagnostic, accepted=False, reason=reason)


def checked_ground(raw, normal, offset, plane_diagnostic, map_normal, prior_rotation, scale):
    mask = cp.abs(raw @ cp.asarray(normal)+offset) < plane_diagnostic['threshold']
    points = raw[mask]
    if len(points) < p.MIN_SOURCE_GROUND_POINTS:
        raise GroundInitializationError('Ground spatial support too small')
    scatter = cp.asnumpy(cp.cov(points.T))*scale**2
    values = np.linalg.eigvalsh(scatter)
    ratio = float(values[1]/max(values[2], 1e-30))
    diagnostic = dict(inliers=len(points), metric_scatter_eigenvalues=values.tolist(),
                      inplane_ratio=ratio, min_inplane_ratio=p.GROUND_MIN_INPLANE_RATIO,
                      model='current plane inliers; CUDA scatter; angular-cell effective support; uncalibrated')
    if ratio < p.GROUND_MIN_INPLANE_RATIO:
        raise GroundSupportError('Ground in-plane support is narrow/line-like', diagnostic)
    expected = np.asarray(prior_rotation).T @ map_normal
    tangent = plane_basis(expected)[:, :2]
    # Pixel points in the same angular cell are correlated, as in feature H/g.
    rays = points/cp.maximum(cp.linalg.norm(points, axis=1, keepdims=True), 1e-12)
    az = cp.arctan2(rays[:, 1], rays[:, 0])
    el = cp.arctan2(rays[:, 2], cp.linalg.norm(rays[:, :2], axis=1))
    bins = cp.floor(cp.stack([az, el], axis=1)/p.ANGULAR_CELL_RAD).astype(cp.int64)
    keys = (bins[:, 0]+(1 << 20))*(1 << 21)+bins[:, 1]+(1 << 20)
    effective = len(cp.unique(keys))
    spread = tangent.T @ scatter @ tangent
    model_variance = max(p.GROUND_NORMAL_MODEL_STD_M, 1.4826*plane_diagnostic['mad']*scale)**2
    measured_cov = model_variance/max(effective, 1)*np.linalg.inv(spread)
    measured_cov += np.eye(2)*np.deg2rad(p.GROUND_NORMAL_FLOOR_DEG)**2
    # No surveyed camera/IMU extrinsic exists. Keep the existing unknown IMU
    # and extrinsic roll/pitch error budgets; no reference CSV enters here.
    orientation_var = np.deg2rad(p.IMU_ORIENTATION_UNKNOWN_STD_DEG)**2
    orientation_var += np.deg2rad(p.EXTRINSIC_ROTATION_STD_DEG)**2
    prior_cov = np.eye(2)*float(max(orientation_var[:2]))
    residual = tangent.T @ (normal-expected)
    total = prior_cov+measured_cov
    nis = float(residual @ np.linalg.solve(total, residual))
    diagnostic.update(angular_cells=effective, normal_innovation_nis=nis,
        normal_measurement_covariance=measured_cov.tolist(), normal_prior_covariance=prior_cov.tolist(),
        normal_nis_reject=p.GROUND_NORMAL_NIS_REJECT,
        source_normal_measured=np.asarray(normal).tolist(), source_offset_measured=float(offset))
    if nis > p.GROUND_NORMAL_NIS_REJECT or np.dot(normal, expected) <= 0:
        raise GroundSupportError('Ground normal inconsistent with current IMU uncertainty', diagnostic)
    # Fuse normal locally in its two-dimensional tangent plane. Refit ONLY
    # offset on the same inlier set; changing tilt without changing ds would
    # create an artificial scale/XY displacement.
    K = np.linalg.solve(total, prior_cov).T
    fused = expected+tangent @ (K @ residual)
    fused /= np.linalg.norm(fused)
    ds = float(cp.median(-points @ cp.asarray(fused)))
    mad = float(cp.median(cp.abs(points @ cp.asarray(fused)+ds)))
    if ds <= .02 or mad*scale > p.MAX_SOURCE_GROUND_MAD_M:
        raise GroundSupportError('Sensor-supported ground offset is unreliable', diagnostic)
    diagnostic.update(normal_gain=K.tolist(), source_normal_fused=fused.tolist(),
                      source_offset_fused=ds, fused_ground_mad_raw=mad, accepted=True)
    return fused, ds, mad, diagnostic
