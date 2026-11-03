"""One causal height update from an independent current-frame map likelihood."""
import numpy as np
from . import parameters as p


def fuse_height(mean, variance, measurement, measurement_variance):
    values = np.asarray([mean, variance, measurement, measurement_variance], dtype=float)
    if not np.isfinite(values).all() or min(mean, variance, measurement, measurement_variance) <= 0:
        raise ValueError('Invalid height mean/covariance')
    innovation = measurement-mean
    nis = innovation*innovation/(variance+measurement_variance)
    reason = None
    if not p.CAMERA_HEIGHT_BOUNDS_M[0] < measurement < p.CAMERA_HEIGHT_BOUNDS_M[1]:
        reason = 'height_outside_physical_bounds'
    elif abs(innovation) > p.METRIC_MAX_HEIGHT_CHANGE_M:
        reason = 'height_outside_existing_search'
    elif nis > p.CORRECTION_NIS_REJECT_BY_DOF[0]:
        reason = 'height_statistical_innovation_rejected'
    inflation = max(1., nis/p.CORRECTION_NIS_SOFT_BY_DOF[0])
    R = measurement_variance*inflation
    gain = 0. if reason else variance/(variance+R)
    posterior = mean+gain*innovation
    covariance = (1-gain)**2*variance+gain*gain*R
    return float(posterior), float(covariance), dict(
        accepted=reason is None, reason=reason or ('height_downweighted' if inflation > 1 else 'height_measurement_update'),
        prior_height_m=float(mean), prior_variance_m2=float(variance), measurement_height_m=float(measurement),
        measurement_variance_m2=float(measurement_variance), innovation_m=float(innovation),
        innovation_nis=float(nis), variance_inflation=float(inflation), gain=float(gain),
        posterior_height_m=float(posterior), posterior_variance_m2=float(covariance),
        model='independent local map height likelihood fused once; engineering uncertainty not calibrated')
