"""GNSS/IMU heading blocks with a common, time-aligned curved-motion model.

This module only constructs bias observations.  It never changes position,
integrates accelerometers, reads a map, or treats GNSS as ground truth.

Within one original GNSS block we use

    p(t) = c + R(b) integral[(v + a * (s - t_center)) * h(psi_imu(s)) ds].

The IMU direction is integrated at its original timestamps, and one b is fitted
to the whole block.  This avoids subtracting an unweighted mean IMU yaw from a
GNSS regression slope: those have different time weights during acceleration
and turns.  v/a/c are nuisance parameters and are marginalized in the reported
local model uncertainty.  A common lever-arm uncertainty allowance inflates
turning-block uncertainty; it is an explicit experiment assumption, not an
extrinsic calibration or an estimate of absolute heading accuracy.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import least_squares


@dataclass(frozen=True)
class TurnHeadingConfig:
    # A bound on unknown GNSS/reference lever arm, expressed as an experimental
    # one-sigma allowance.  Both tracks use the same value; no calibrated claim.
    lever_arm_std_m: float = 2.0
    max_turn_rate_degps: float = 10.0
    max_heading_span_deg: float = 20.0
    max_normalized_residual: float = 3.0

    def validate(self):
        values = asdict(self)
        if not all(np.isfinite(v) and v > 0 for v in values.values()):
            raise ValueError('Turn-heading configuration must be finite and positive')


def _validate_inputs(gt, xy, it, imu_yaw):
    if (gt.ndim != 1 or it.ndim != 1 or xy.shape != (len(gt), 2)
            or imu_yaw.shape != it.shape or len(gt) < 2 or len(it) < 2):
        raise ValueError('Turn-heading timestamp/position/yaw shape mismatch')
    if (not all(np.all(np.isfinite(x)) for x in (gt, xy, it, imu_yaw))
            or np.any(np.diff(gt) <= 0) or np.any(np.diff(it) <= 0)):
        raise ValueError('Turn-heading inputs require finite, increasing timestamps')


def integrated_heading_basis(t, it, imu_yaw):
    """Return unit-speed and unit-acceleration paths at raw GNSS timestamps.

    The basis origin is t[0], with acceleration expressed at mean(t).  Shifting
    integration origin is absorbed by c.  This function expects unwrapped yaw;
    subtracting a local reference avoids loss of precision at wrap boundaries.
    """
    center = float(np.mean(t))
    yaw_reference = float(np.interp(center, it, imu_yaw))
    interior = it[(it > t[0]) & (it < t[-1])]
    grid = np.unique(np.concatenate((t, interior)))
    direction = np.column_stack((np.cos(np.interp(grid, it, imu_yaw) - yaw_reference),
                                 np.sin(np.interp(grid, it, imu_yaw) - yaw_reference)))
    q0 = cumulative_trapezoid(direction, grid, axis=0, initial=0.)
    q1 = cumulative_trapezoid(direction * (grid - center)[:, None], grid, axis=0, initial=0.)
    at = np.searchsorted(grid, t)
    return q0[at], q1[at], yaw_reference


def _rotate(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    return points @ np.array([[c, s], [-s, c]])


def _fit_motion(t, points, q0, q1, initial_bias, initial_speed, config):
    weights = np.ones(len(t))
    limit = np.deg2rad(config.max_observation_deg)
    state = np.r_[points[0], max(initial_speed, .1), 0., np.clip(initial_bias, -.99*limit, .99*limit)]

    def residual(parameters):
        c, speed, acceleration, bias = parameters[:2], *parameters[2:]
        return c + _rotate(speed*q0 + acceleration*q1, bias) - points

    def jacobian(parameters):
        _, _, speed, acceleration, bias = parameters
        path = _rotate(speed*q0 + acceleration*q1, bias)
        jac = np.zeros((len(t), 2, 5))
        jac[:, :2, :2] = np.eye(2)
        jac[:, :, 2] = _rotate(q0, bias)
        jac[:, :, 3] = _rotate(q1, bias)
        jac[:, :, 4] = np.column_stack((-path[:, 1], path[:, 0]))
        return jac

    fit = None
    for _ in range(4):
        root_weight = np.sqrt(weights)[:, None]
        fit = least_squares(lambda p: (residual(p)*root_weight).ravel(), state,
            jac=lambda p: (jacobian(p)*root_weight[:, :, None]).reshape(-1, 5),
            bounds=([-np.inf, -np.inf, .01, -np.inf, -limit],
                    [np.inf, np.inf, np.inf, np.inf, limit]),
            x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=100)
        state = fit.x
        error = residual(state)
        weights = np.minimum(1., config.robust_gate_sigma*config.gnss_std_m /
                             np.maximum(np.linalg.norm(error, axis=1), 1e-12))
    # Refit with the final robust weights so the state and reported Hessian use
    # the same objective. No GNSS samples are duplicated or interpolated updates.
    root_weight = np.sqrt(weights)[:, None]
    fit = least_squares(lambda p: (residual(p)*root_weight).ravel(), state,
        jac=lambda p: (jacobian(p)*root_weight[:, :, None]).reshape(-1, 5),
        bounds=([-np.inf, -np.inf, .01, -np.inf, -limit],
                [np.inf, np.inf, np.inf, np.inf, limit]),
        x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=100)
    return fit, residual(fit.x), weights, jacobian(fit.x).reshape(-1, 5)


def turn_heading_observations(gt, xy, it, imu_yaw, config, turn_config=None):
    """Construct one bias factor per existing raw GNSS block.

    ``config`` is the existing FusionConfig. Rows retain heading_observations'
    required schema and add fit diagnostics. Use the accepted rows as a
    *replacement* for existing course rows, never as independent additional
    information: both are derived from the same GNSS/IMU samples.
    """
    config.validate()
    turn_config = turn_config or TurnHeadingConfig()
    turn_config.validate()
    gt, xy, it, imu_yaw = (np.asarray(x, dtype=float) for x in (gt, xy, it, imu_yaw))
    _validate_inputs(gt, xy, it, imu_yaw)
    rows = []
    bins = np.floor((gt - gt[0])/config.course_window_s).astype(int)
    for block in np.unique(bins):
        indices = np.flatnonzero(bins == block)
        t = gt[indices]
        row = {'time_s': float(np.mean(t)), 'samples': int(len(t)), 'accepted': False,
               'reason': '', 'speed_mps': None, 'offset_deg': None, 'std_deg': None,
               'source': 'turn_integrated_course', 'acceleration_mps2': None,
               'min_speed_mps': None, 'turn_rate_rms_degps': None,
               'turn_rate_max_degps': None, 'residual_rms_m': None,
               'marginal_std_deg': None, 'lever_arm_std_deg': None}
        if len(t) < 5 or t[-1]-t[0] < 1.0 or t[0] < it[0] or t[-1] > it[-1]:
            row['reason'] = 'insufficient_samples_or_imu_coverage'
            rows.append(row)
            continue
        heading = np.interp(t, it, imu_yaw)
        inside = (it >= t[0]) & (it <= t[-1])
        dense_t = np.r_[t[0], it[inside], t[-1]]
        dense_t = np.unique(dense_t)
        dense_heading = np.interp(dense_t, it, imu_yaw)
        rate = np.gradient(dense_heading, dense_t)
        max_rate = float(np.max(np.abs(rate)))
        rms_rate = float(np.sqrt(trapezoid(rate**2, dense_t)/(t[-1]-t[0])))
        row.update(turn_rate_rms_degps=float(np.rad2deg(rms_rate)),
                   turn_rate_max_degps=float(np.rad2deg(max_rate)))
        if (max_rate >= np.deg2rad(turn_config.max_turn_rate_degps)
                or np.ptp(dense_heading) >= np.deg2rad(turn_config.max_heading_span_deg)):
            row['reason'] = 'turn_exceeds_local_model'
            rows.append(row)
            continue
        q0, q1, reference = integrated_heading_basis(t, it, imu_yaw)
        points = xy[indices]-xy[indices[0]]
        design = np.column_stack((np.ones(len(t)), t-np.mean(t)))
        velocity = np.linalg.lstsq(design, points, rcond=None)[0][1]
        basis_slope = np.linalg.lstsq(design, q0, rcond=None)[0][1]
        initial_bias = np.arctan2(velocity[1], velocity[0])-np.arctan2(basis_slope[1], basis_slope[0])-reference
        initial_bias = float(np.arctan2(np.sin(initial_bias), np.cos(initial_bias)))
        initial_speed = float(np.linalg.norm(velocity))
        if initial_speed <= config.min_course_speed_mps:
            row.update(reason='low_speed', speed_mps=initial_speed)
            rows.append(row)
            continue
        # Basis above uses local yaw=0. Rotate it back to the IMU's map heading;
        # fitted b is therefore the actual map-Z correction, including wraps.
        q0, q1 = _rotate(q0, reference), _rotate(q1, reference)
        fit, error, weights, jac = _fit_motion(t, points, q0, q1, initial_bias, initial_speed, config)
        speed, acceleration, bias = fit.x[2:]
        min_speed = float(np.min(speed+acceleration*(t[[0, -1]]-np.mean(t))))
        dof = max(1, 2*len(t)-5)
        scatter = float(np.sqrt(np.sum(weights[:, None]*error**2)/dof))
        row.update(speed_mps=float(speed), offset_deg=float(np.rad2deg(bias)),
                   acceleration_mps2=float(acceleration), min_speed_mps=min_speed,
                   residual_rms_m=float(np.sqrt(np.mean(np.sum(error**2, axis=1)))))
        weighted_jac = jac*np.repeat(np.sqrt(weights), 2)[:, None]
        singular = np.linalg.svd(weighted_jac, compute_uv=False)
        if not fit.success or not np.all(np.isfinite(fit.x)):
            row['reason'] = 'numerical_fit_failure'
        elif min_speed <= config.min_course_speed_mps:
            row['reason'] = 'low_speed_or_reverse_within_block'
        elif abs(bias) >= np.deg2rad(config.max_observation_deg)*.999:
            row['reason'] = 'heading_disagreement_or_bound'
        elif np.count_nonzero(weights < .5) > 1:
            row['reason'] = 'inconsistent_gnss_block'
        elif scatter > turn_config.max_normalized_residual*config.gnss_std_m:
            row['reason'] = 'curved_motion_model_inconsistent'
        elif singular[-1] <= singular[0]*1e-8:
            row['reason'] = 'heading_information_degenerate'
        else:
            covariance = np.linalg.inv(weighted_jac.T@weighted_jac)
            marginal = max(config.gnss_std_m, scatter)*np.sqrt(max(covariance[4, 4], 0.))
            lever = turn_config.lever_arm_std_m*rms_rate/min_speed
            std = max(np.deg2rad(config.course_std_floor_deg), np.hypot(marginal, lever))
            row.update(accepted=True, reason='accepted', std_deg=float(np.rad2deg(std)),
                       marginal_std_deg=float(np.rad2deg(marginal)),
                       lever_arm_std_deg=float(np.rad2deg(lever)))
        rows.append(row)
    return rows
