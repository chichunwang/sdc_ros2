"""Fuse one current image observation into the existing XY/CV state.

Only speed magnitude (or image-supported zero velocity) is observed. Images
do not specify absolute map position, global yaw, sideways-zero motion, or
heading bias. Correlations with past map height are conservatively floored,
not a claim of a calibrated joint estimator.
"""
import numpy as np
from . import parameters as p


def fuse_visual_speed(state, prior, diagnostic, measurement, gyro_norm):
    result = dict(measurement, filter_accepted=False)
    if not measurement.get('accepted'):
        return prior, diagnostic, result
    if int(measurement['timestamp_ns']) != state.time:
        raise ValueError('Visual observation must be at the current state time')
    variance = float(measurement['variance_m2ps2'])
    if not np.isfinite(variance) or variance <= 0 or not np.isfinite(gyro_norm):
        raise ValueError('Invalid visual velocity uncertainty')
    before = state.cov.copy(); velocity = state.position[2:].copy()
    if measurement['kind'] == 'stationary':
        H = np.c_[np.zeros((2, 2)), np.eye(2)]
        innovation = -velocity; R = np.eye(2)*variance; dof = 2
    elif measurement['kind'] == 'speed':
        speed = float(np.linalg.norm(velocity))
        if speed < .1:
            return prior, diagnostic, dict(result, filter_reason='no_sensor_velocity_direction')
        value = float(measurement['speed_mps'])
        if not np.isfinite(value) or value < 0:
            raise ValueError('Invalid image speed')
        H = np.r_[0., 0., velocity/speed].reshape(1, 4)
        innovation = np.array([value-speed]); dof = 1
        # Image displacement is camera motion; unknown antenna/camera lever
        # induces angular-rate-dependent speed error, never an assumed offset.
        lever_variance = float(np.sum(np.asarray(p.LEVER_ARM_STD_BODY_M)**2))
        R = np.array([[variance+gyro_norm**2*lever_variance]])
    else:
        raise ValueError('Unknown visual velocity observation')
    S = H @ before @ H.T+R
    nis = float(innovation @ np.linalg.solve(S, innovation))
    result.update(innovation=innovation.tolist(), innovation_nis=nis, observation_matrix=H.tolist())
    if nis > p.CORRECTION_NIS_REJECT_BY_DOF[dof-1]:
        return prior, diagnostic, dict(result, filter_reason='visual_velocity_statistical_gate')
    inflation = max(1., nis/p.CORRECTION_NIS_SOFT_BY_DOF[dof-1]); effective = R*inflation
    K = np.linalg.solve(H @ before @ H.T+effective, H @ before).T
    increment = K @ innovation
    covariance = (np.eye(4)-K @ H) @ before @ (np.eye(4)-K @ H).T+K @ effective @ K.T
    covariance = (covariance+covariance.T)*.5
    state.position += increment; state.cov = covariance
    output = np.asarray(prior).copy(); output[:2, 3] += increment[:2]
    C = np.asarray(diagnostic['pose_covariance_xy_yaw']).copy()
    C[:2, :2] += covariance[:2, :2]-before[:2, :2]
    if np.linalg.eigvalsh(C).min() <= 0:
        raise ValueError('Visual update produced non-positive pose covariance')
    diag = dict(diagnostic, velocity_xy_mps=state.position[2:].tolist(),
        position_covariance_xy=state.cov[:2, :2].tolist(), pose_covariance_xy_yaw=C.tolist())
    diag['uncertainty_model'] = dict(diagnostic['uncertainty_model'],
        image_speed='past-height/image interval average; model/height/acceleration/lever floors; cross-frame correlations not fully estimated')
    result.update(filter_accepted=True, filter_reason='current_image_velocity_update',
        effective_measurement_covariance=effective.tolist(), kalman_gain=K.tolist(),
        state_increment=increment.tolist(), covariance_before=before.tolist(), covariance_after=state.cov.tolist(),
        covariance_inflation=inflation, state_reference_policy='kinematic innovation; not calibration rebase')
    if measurement['kind'] == 'stationary':
        result['stationary_course_exclusion'] = state.exclude_image_stationary_course(
            int(measurement['keyframe_timestamp_ns']), int(measurement['timestamp_ns']))
    return output, diag, result
