"""當幀輸出前融合 map-relative correction；沒有回寫過去 pose 的介面。

狀態是地圖座標的 [dx_m, dy_m, dyaw_rad]，不是地面平面座標，也不是
車輛 pose。車輛仍隨每幀 sensor prior 移動；濾波只估計其緩變偏差。
此處的小矩陣在 CPU；點雲 solver 與對應仍在 CUDA。
"""
import numpy as np
from . import parameters as p
from .observability import checked_covariance


def wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def yaw(T):
    return float(np.arctan2(T[1, 0], T[0, 0]))


def pose_correction(candidate, prior):
    return np.r_[candidate[:2, 3]-prior[:2, 3], wrap(yaw(candidate)-yaw(prior))]


def quality_inflation(quality):
    """Coverage 與殘差的保守膨脹；不是經真值校準的 likelihood。"""
    coverage = float(quality['coverage'])
    median = float(quality['median_capped_distance_m'])
    if not np.isfinite([coverage, median]).all() or not 0 < coverage <= 1 or median < 0:
        raise ValueError('Invalid registration quality')
    return max(1., (median/p.FEATURE_STD_M)**2)/coverage**2


def correction_covariance(data_information, jacobian, quality):
    """Data-only GN covariance 投影；不把 GNSS soft prior 再算成獨立量測。

保留四維耦合（包括尺度），弱方向給大 variance，不能用 pinv 把零資訊
誤當成零 variance。角度／公尺先無因次化後反解，並加上模型誤差下限。
"""
    H = np.asarray(data_information, dtype=float)
    J = np.asarray(jacobian, dtype=float)
    if H.shape != (4, 4) or J.shape != (3, 4) or not np.isfinite(H).all() or not np.isfinite(J).all():
        raise ValueError('Invalid correction information/Jacobian')
    units = np.diag([p.METRIC_POSITION_STD_M]*2+
                    [np.deg2rad(p.METRIC_YAW_STD_DEG), p.MAX_SCALE_LOG_STD])
    normalized = units @ ((H+H.T)*.5) @ units
    eig, vectors = np.linalg.eigh(normalized)
    floor = max(1e-8, float(np.max(eig))*1e-10)
    covariance = units @ ((vectors/np.maximum(eig, floor)) @ vectors.T) @ units
    result = J @ covariance @ J.T
    result += np.diag([p.CORRECTION_MEAS_POSITION_FLOOR_M**2]*2+
                      [np.deg2rad(p.CORRECTION_MEAS_YAW_FLOOR_DEG)**2])
    return (result+result.T)*.5*quality_inflation(quality)


class CausalCorrectionFilter:
    def __init__(self):
        self.mean = np.zeros(3)
        self.anchor_covariance = np.diag([p.METRIC_POSITION_STD_M**2]*2+
                                        [np.deg2rad(p.METRIC_YAW_STD_DEG)**2])
        self.covariance = self.anchor_covariance.copy()
        self.last_stamp = None
        self.last_prior = None
        self.last_update = None
        self.last_measurement = None
        self.last_candidate_sensitivity = np.eye(3)
        self.distance_since_update = 0.
        self.turn_since_update = 0.
        self.pending = False
        self.last_observation_projector = np.eye(3)
        # Sensitivity to the sensor calibration reference, accumulated with
        # the ACTUAL gains. No map information => zero compensation.
        self.reference_sensitivity = np.zeros((3, 3))
        self.motion_units = np.diag([1., 1., np.deg2rad(3.)])

    def predict(self, prior, timestamp_ns, speed_mps=None, *, prior_covariance=None, rotation_delta_rad=None, reference_delta=None):
        if self.pending:
            raise RuntimeError('Previous correction prediction needs one update')
        if self.last_stamp is not None and timestamp_ns <= self.last_stamp:
            raise ValueError('Correction timestamps must strictly increase')
        self.reference_delta = np.zeros(3) if reference_delta is None else np.asarray(reference_delta, dtype=float)
        if self.reference_delta.shape != (3,) or not np.isfinite(self.reference_delta).all():
            raise ValueError('Invalid causal calibration reference change')
        self.reference_rebase = self.reference_sensitivity @ self.reference_delta
        if np.any(self.reference_delta):
            self.mean -= self.reference_rebase; self.mean[2] = wrap(self.mean[2])
            if self.last_measurement is not None:
                self.last_measurement -= self.last_candidate_sensitivity @ self.reference_delta
                self.last_measurement[2] = wrap(self.last_measurement[2])
        if prior_covariance is not None:
            self.anchor_covariance = checked_covariance(prior_covariance)
            if self.last_stamp is None:
                self.covariance = self.anchor_covariance.copy()
                self.motion_units = np.linalg.cholesky(self.anchor_covariance)
        dt = 0. if self.last_stamp is None else (timestamp_ns-self.last_stamp)*1e-9
        distance = turn = 0.
        if self.last_prior is not None:
            distance = float(np.linalg.norm(prior[:2, 3]-self.last_prior[:2, 3]))
            relative = prior[:3, :3] @ self.last_prior[:3, :3].T
            turn = float(np.arccos(np.clip((np.trace(relative)-1)*.5, -1., 1.)))
        if speed_mps is not None:
            if not np.isfinite(speed_mps) or speed_mps < 0:
                raise ValueError('Invalid causal sensor speed')
            # A GNSS update may move the prior; that jump must not manufacture
            # a larger vehicle-motion allowance for a map outlier.
            distance = float(speed_mps)*dt
        if rotation_delta_rad is not None:
            if not np.isfinite(rotation_delta_rad) or rotation_delta_rad < 0:
                raise ValueError('Invalid causal gyro rotation')
            # GNSS heading-bias alignment is a calibration update, not a car turn.
            turn = float(rotation_delta_rad)
        qxy = (p.CORRECTION_POSITION_RW_M_SQRT_S**2*dt
               +p.CORRECTION_POSITION_RW_M_SQRT_M**2*distance
               +(p.CORRECTION_TURN_LEVER_M*turn)**2)
        qyaw = (np.deg2rad(p.CORRECTION_YAW_RW_DEG_SQRT_S)**2*dt
                +np.deg2rad(p.CORRECTION_YAW_RW_DEG_SQRT_M)**2*distance)
        self.covariance += np.diag([qxy, qxy, qyaw])
        self.distance_since_update += distance
        self.turn_since_update += turn
        self.previous_stamp = self.last_stamp
        self.last_stamp = int(timestamp_ns)
        self.last_prior = np.asarray(prior).copy()
        self.motion = dict(dt_seconds=dt, speed_mps=None if speed_mps is None else float(speed_mps),
                           distance_m=distance, prior_rotation_rad=turn,
                           rotation_source='prior_difference' if rotation_delta_rad is None else 'arrived_gyro_integral')
        self.predicted_mean = self.mean.copy()
        self.predicted_covariance = self.covariance.copy()
        self.pending = True
        return self.mean.copy()

    def update(self, candidate=None, measurement_covariance=None, *, observation_matrix=None,
               missing_reason='geometry_rejected', candidate_sensitivity=None):
        if not self.pending:
            raise RuntimeError('Correction update requires one prediction')
        self.pending = False
        innovation = R = motion_innovation = None
        motion_reference = None if self.last_measurement is None else self.last_measurement.copy()
        nis = None; weight = 0.; decay = 1.; accepted = False
        motion_ratio = 0.; gain = None; gain_scale_xy = gain_scale_yaw = gain_support_scale = 1.
        H = None; nis_soft = nis_reject = None; innovation_world = None
        step_xy_limit = (p.CORRECTION_POSITION_RATE_MPS*self.motion['dt_seconds']
                         +p.CORRECTION_SPEED_FRACTION*self.motion['distance_m']
                         +p.CORRECTION_TURN_LEVER_M*self.motion['prior_rotation_rad'])
        step_yaw_limit = (np.deg2rad(p.CORRECTION_YAW_RATE_DEGPS)*self.motion['dt_seconds']
                          +p.CORRECTION_TURN_FRACTION*self.motion['prior_rotation_rad'])
        age = (0. if self.last_update is None else (self.last_stamp-self.last_update)*1e-9)
        pos_limit = (p.CORRECTION_POSITION_GATE_BASE_M+p.CORRECTION_POSITION_RATE_MPS*age
                     +p.CORRECTION_SPEED_FRACTION*self.distance_since_update
                     +p.CORRECTION_TURN_LEVER_M*self.turn_since_update)
        yaw_limit = (np.deg2rad(p.CORRECTION_YAW_GATE_BASE_DEG+p.CORRECTION_YAW_RATE_DEGPS*age)
                     +p.CORRECTION_TURN_FRACTION*self.turn_since_update)
        reason = missing_reason
        if candidate is not None:
            candidate = np.asarray(candidate, dtype=float).copy()
            H = np.eye(3) if observation_matrix is None else np.asarray(observation_matrix, dtype=float)
            R = np.asarray(measurement_covariance, dtype=float)
            dof = H.shape[0] if H.ndim == 2 else 0
            if (candidate.shape != (3,) or not 1 <= dof <= 3 or H.shape != (dof, 3)
                    or R.shape != (dof, dof) or not np.isfinite(candidate).all()
                    or not np.isfinite(H).all() or np.linalg.matrix_rank(H) != dof
                    or not np.isfinite(R).all() or not np.allclose(R, R.T, rtol=1e-8, atol=1e-10)
                    or np.linalg.eigvalsh(R).min() <= 0):
                raise ValueError('Invalid correction observation/covariance')
            candidate_sensitivity = np.eye(3) if candidate_sensitivity is None else np.asarray(candidate_sensitivity, dtype=float)
            if candidate_sensitivity.shape != (3, 3) or not np.isfinite(candidate_sensitivity).all():
                raise ValueError('Invalid map candidate calibration sensitivity')
            candidate[2] = wrap(candidate[2])
            innovation_world = candidate-self.mean; innovation_world[2] = wrap(innovation_world[2])
            innovation = H @ innovation_world
            S = H @ self.covariance @ H.T+R
            nis = float(innovation @ np.linalg.solve(S, innovation))
            nis_soft = p.CORRECTION_NIS_SOFT if observation_matrix is None else p.CORRECTION_NIS_SOFT_BY_DOF[dof-1]
            nis_reject = p.CORRECTION_NIS_REJECT if observation_matrix is None else p.CORRECTION_NIS_REJECT_BY_DOF[dof-1]
            # Physical comparison only uses directions BOTH observations support.
            # A newly visible/null candidate component is not a vehicle jump.
            _, _, vt = np.linalg.svd(H @ self.motion_units, full_matrices=False)
            supported = vt.T @ vt
            L = np.linalg.cholesky(self.anchor_covariance)
            _, _, svt = np.linalg.svd(H @ L, full_matrices=False)
            state_projector = L @ svt.T @ svt @ np.linalg.inv(L)
            if motion_reference is not None:
                delta = candidate-motion_reference; delta[2] = wrap(delta[2])
                common = supported @ self.last_observation_projector @ supported
                motion_innovation = self.motion_units @ common @ np.linalg.solve(self.motion_units, delta)
                motion_ratio = float(max(np.linalg.norm(motion_innovation[:2])/pos_limit,
                                         abs(motion_innovation[2])/yaw_limit))
            if motion_ratio > p.CORRECTION_MOTION_REJECT_MULTIPLIER:
                reason = 'motion_innovation_rejected'
            elif nis > nis_reject:
                reason = 'statistical_innovation_rejected'
            else:
                # The engineering motion envelope is a soft allowance. A
                # moderate excess reduces influence rather than starving XY
                # updates whenever yaw disagrees slightly. Extreme motion
                # excess and statistically incompatible candidates reject.
                inflation = max(1., nis/nis_soft, motion_ratio**2)
                effective_R = R*inflation
                K = np.linalg.solve(H @ self.covariance @ H.T+effective_R, H @ self.covariance).T
                # Exclude data-null directions even when old cross covariance
                # would otherwise couple them into this current-frame update.
                K = state_projector @ K
                if self.last_update is not None:
                    # Reacquisition has a wider *candidate* gate after an
                    # outage, but must not merely postpone the jump to the
                    # next accepted frame. Limit influence using THIS frame's
                    # motion/time budget. Use the same scaled gain in Joseph
                    # covariance; never clip only the mean and claim full info.
                    increment = K @ innovation
                    gain_scale_xy = min(1., step_xy_limit/max(np.linalg.norm(increment[:2]), 1e-30))
                    gain_scale_yaw = min(1., step_yaw_limit/max(abs(increment[2]), 1e-30))
                    K = state_projector @ np.diag([gain_scale_xy, gain_scale_xy, gain_scale_yaw]) @ K
                    # Projection can couple axes if a single geometric mode
                    # physically mixes translation and yaw. Only that case
                    # needs a final common bound; full rank remains decoupled.
                    if dof < 3:
                        projected_increment = K @ innovation
                        gain_support_scale = min(1., step_xy_limit/max(np.linalg.norm(projected_increment[:2]), 1e-30),
                                                 step_yaw_limit/max(abs(projected_increment[2]), 1e-30))
                        K *= gain_support_scale
                self.mean += K @ innovation; self.mean[2] = wrap(self.mean[2])
                A = np.eye(3)-K @ H
                self.reference_sensitivity = A @ self.reference_sensitivity+K @ H @ candidate_sensitivity
                self.covariance = A @ self.covariance @ A.T+K @ effective_R @ K.T
                self.covariance = (self.covariance+self.covariance.T)*.5
                gain = K.tolist()
                weight = 1/inflation
                reason = 'downweighted_update' if inflation > 1 or min(gain_scale_xy, gain_scale_yaw, gain_support_scale) < 1 else 'measurement_update'
                accepted = True
                self.last_update = self.last_stamp
                self.last_measurement = candidate.copy()
                self.last_candidate_sensitivity = candidate_sensitivity.copy()
                self.last_observation_projector = supported
                self.distance_since_update = self.turn_since_update = 0.
        if not accepted and self.last_update is not None:
            # Hold the mean during short outages, while its variance grows.
            # After the grace period, relax the old bias toward sensor prior.
            before_age = (self.previous_stamp-self.last_update)*1e-9
            decay_seconds = max(0., age-p.CORRECTION_HOLD_SECONDS)-max(0., before_age-p.CORRECTION_HOLD_SECONDS)
            decay = float(np.exp(-decay_seconds/p.CORRECTION_DECAY_SECONDS))
            self.mean *= decay
            self.reference_sensitivity *= decay
            self.covariance = (decay**2*self.covariance+(1-decay**2)*self.anchor_covariance)
        diag = dict(accepted=accepted, reason=reason, frame='map XY metres, world yaw radians',
            prediction=self.predicted_mean.tolist(), correction=self.mean.tolist(),
            candidate=None if candidate is None else candidate.tolist(),
            innovation=None if innovation is None else innovation.tolist(), innovation_nis=nis,
            motion_innovation=None if motion_innovation is None else motion_innovation.tolist(),
            motion_reference=None if motion_reference is None else motion_reference.tolist(),
            measurement_weight=weight, kalman_gain=gain, gain_scale=float(min(gain_scale_xy, gain_scale_yaw, gain_support_scale)),
            gain_scale_xy=float(gain_scale_xy), gain_scale_yaw=float(gain_scale_yaw),
            gain_scale_observable=float(gain_support_scale),
            observation_matrix=None if H is None else H.tolist(),
            observation_dof=0 if H is None else H.shape[0], nis_soft_threshold=nis_soft, nis_reject_threshold=nis_reject,
            innovation_world=None if innovation_world is None else innovation_world.tolist(),
            innovation_frame='observation_matrix @ map [dx_m,dy_m,dyaw_rad]',
            calibration_reference_delta=self.reference_delta.tolist(),
            calibration_rebase=self.reference_rebase.tolist(),
            calibration_sensitivity=self.reference_sensitivity.tolist(),
            candidate_calibration_sensitivity=None if candidate_sensitivity is None else candidate_sensitivity.tolist(),
            position_update_limit_m=step_xy_limit, yaw_update_limit_rad=step_yaw_limit,
            motion_innovation_ratio=motion_ratio, decay_factor=decay,
            covariance_predicted=self.predicted_covariance.tolist(), covariance=self.covariance.tolist(),
            measurement_covariance=None if R is None else R.tolist(),
            seconds_since_update=None if self.last_update is None else (self.last_stamp-self.last_update)*1e-9,
            position_innovation_limit_m=pos_limit, yaw_innovation_limit_rad=yaw_limit,
            motion=self.motion)
        return self.mean.copy(), diag
