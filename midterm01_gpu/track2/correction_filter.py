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


def bounded_reference_sensitivity(sensitivity, reference_covariance):
    """Conservative information-compatible calibration response.

    In prior-whitened coordinates an independent PSD map information term has
    response W=(I+Lambda)^-1 Lambda: symmetric, 0 <= W <= I. Accumulating
    changing, oblique KH projectors does not preserve that property. Project
    its response onto this set (nearest symmetric contraction in Frobenius
    norm), rather than letting its unbounded XY/yaw cross term move the pose.
    This is a conservative approximation, NOT a full joint sensor/map EKF.
    """
    C = checked_covariance(reference_covariance)
    L = np.linalg.cholesky(C)
    S = np.asarray(sensitivity, dtype=float)
    if S.shape != (3, 3) or not np.isfinite(S).all():
        raise ValueError('Invalid reference sensitivity')
    normalized = np.linalg.solve(L, S @ L)
    symmetric = (normalized+normalized.T)*.5
    values, vectors = np.linalg.eigh(symmetric)
    bounded = (vectors*np.clip(values, 0., 1.)) @ vectors.T
    result = L @ bounded @ np.linalg.inv(L)
    return result, dict(raw_normalized_eigenvalues=values.tolist(),
        bounded_normalized_eigenvalues=np.clip(values, 0., 1.).tolist(),
        normalized_projection_norm=float(np.linalg.norm(normalized-bounded)),
        model='prior-whitened PSD information response contraction; conservative approximation, not joint EKF')


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
        self.last_measurement_is_linear = False
        self.distance_since_update = 0.
        self.turn_since_update = 0.
        self.pending = False
        self.last_observation_projector = np.eye(3)
        # Sensitivity to the sensor calibration reference, accumulated with
        # the ACTUAL gains. No map information => zero compensation.
        self.reference_sensitivity = np.zeros((3, 3))
        self.reference_response_diagnostic = None
        self.motion_units = np.diag([1., 1., np.deg2rad(3.)])
        # A PSD age tensor in FIXED initial-prior-whitened coordinates. Each
        # accepted observation resets only its row space, never every axis.
        # Keep the metric fixed: changing sensor covariance is not new map data.
        self.memory_age = np.zeros((3, 3))

    def _age_memory(self, dt):
        self.memory_age += dt*np.eye(3)
        age, basis = np.linalg.eigh((self.memory_age+self.memory_age.T)*.5)
        elapsed = np.minimum(dt, np.maximum(age-p.CORRECTION_HOLD_SECONDS, 0.))
        if self.last_update is None:
            elapsed[:] = 0.  # No learned map memory exists yet.
        factors = np.exp(-elapsed/p.CORRECTION_DECAY_SECONDS)
        self.memory_decay = np.eye(3)
        if self.last_update is not None and np.any(elapsed > 0):
            L = self.motion_units
            inverse = np.linalg.inv(L)
            D = (basis*factors) @ basis.T
            self.memory_decay = L @ D @ inverse
            self.mean = self.memory_decay @ self.mean
            self.mean[2] = wrap(self.mean[2])
            self.reference_sensitivity = self.memory_decay @ self.reference_sensitivity
            # Independent prior reset in expired modes. Both terms are PSD;
            # for equal factors this is the original d² P +(1-d²) C_anchor.
            reset = L @ (basis*np.sqrt(np.maximum(0., 1-factors**2))) @ basis.T @ inverse
            self.covariance = (self.memory_decay @ self.covariance @ self.memory_decay.T
                               +reset @ self.anchor_covariance @ reset.T)
            self.covariance = (self.covariance+self.covariance.T)*.5
        self.memory_decay_factors = factors

    def predict(self, prior, timestamp_ns, speed_mps=None, *, prior_covariance=None, rotation_delta_rad=None, reference_delta=None):
        if self.pending:
            raise RuntimeError('Previous correction prediction needs one update')
        if self.last_stamp is not None and timestamp_ns <= self.last_stamp:
            raise ValueError('Correction timestamps must strictly increase')
        self.reference_delta = np.zeros(3) if reference_delta is None else np.asarray(reference_delta, dtype=float)
        if self.reference_delta.shape != (3,) or not np.isfinite(self.reference_delta).all():
            raise ValueError('Invalid causal calibration reference change')
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
        # Expire old information BEFORE using its calibration sensitivity and
        # BEFORE adding this frame's map measurement. New data never gets faded
        # merely because another direction has been unseen for a long time.
        self._age_memory(dt)
        if not p.uses_v52_fusion() or (p.v5121_enabled() and p.MAP_BOUNDED_REFERENCE_RESPONSE):
            self.reference_sensitivity, self.reference_response_diagnostic = bounded_reference_sensitivity(
                self.reference_sensitivity, self.anchor_covariance)
        self.reference_rebase_sensitivity = self.reference_sensitivity.copy()
        self.reference_rebase_diagnostic = (None if self.reference_response_diagnostic is None
                                            else self.reference_response_diagnostic.copy())
        self.reference_rebase = self.reference_sensitivity @ self.reference_delta
        if np.any(self.reference_delta):
            self.mean -= self.reference_rebase; self.mean[2] = wrap(self.mean[2])
            if self.last_measurement is not None:
                self.last_measurement -= self.last_candidate_sensitivity @ self.reference_delta
                if not self.last_measurement_is_linear:
                    self.last_measurement[2] = wrap(self.last_measurement[2])
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
               missing_reason='geometry_rejected', candidate_sensitivity=None,
               memory_observation_matrix=None, proposal_validator=None, local_linear_likelihood=False):
        if not self.pending:
            raise RuntimeError('Correction update requires one prediction')
        self.pending = False
        innovation = R = motion_innovation = None
        motion_reference = None if self.last_measurement is None else self.last_measurement.copy()
        nis = None; weight = 0.; accepted = False
        decay = float(np.min(self.memory_decay_factors))
        motion_ratio = 0.; gain = None; gain_scale_xy = gain_scale_yaw = gain_support_scale = 1.
        H = None; nis_soft = nis_reject = None; innovation_world = None
        proposal_validation = None
        nonlinear_gain_fraction = 1.
        proposal_backtracking = None
        step_xy_limit = (p.CORRECTION_POSITION_RATE_MPS*self.motion['dt_seconds']
                         +p.CORRECTION_SPEED_FRACTION*self.motion['distance_m']
                         +p.CORRECTION_TURN_LEVER_M*self.motion['prior_rotation_rad'])
        step_yaw_limit = (np.deg2rad(p.CORRECTION_YAW_RATE_DEGPS)*self.motion['dt_seconds']
                          +p.CORRECTION_TURN_FRACTION*self.motion['prior_rotation_rad'])
        step_lateral_limit = (p.CORRECTION_LATERAL_RATE_MPS*self.motion['dt_seconds']
                              +p.CORRECTION_TURN_LEVER_M*self.motion['prior_rotation_rad'])
        angle = yaw(self.last_prior)
        body = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        gain_scale_longitudinal = gain_scale_lateral = 1.
        delayed_first_anchor = bool(((p.V511_ENABLED and p.STARTUP_BOUND_AFTER_FIRST_FRAME) or p.v5121_enabled())
            and p.uses_v52_fusion() and self.last_update is None and self.previous_stamp is not None)
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
            if not local_linear_likelihood:
                candidate[2] = wrap(candidate[2])
            innovation_world = candidate-self.mean
            if not local_linear_likelihood:
                innovation_world[2] = wrap(innovation_world[2])
            innovation = H @ innovation_world
            S = H @ self.covariance @ H.T+R
            nis = float(innovation @ np.linalg.solve(S, innovation))
            nis_soft = p.CORRECTION_NIS_SOFT if observation_matrix is None else p.CORRECTION_NIS_SOFT_BY_DOF[dof-1]
            nis_reject = p.CORRECTION_NIS_REJECT if observation_matrix is None else p.CORRECTION_NIS_REJECT_BY_DOF[dof-1]
            # Physical comparison only uses directions BOTH observations support.
            # A newly visible/null candidate component is not a vehicle jump.
            memory_H = H if memory_observation_matrix is None else np.asarray(memory_observation_matrix, dtype=float).reshape(-1, 3)
            if not np.isfinite(memory_H).all() or len(memory_H) > dof:
                raise ValueError('Invalid strong memory observation matrix')
            if len(memory_H):
                if (np.linalg.matrix_rank(memory_H) != len(memory_H)
                        or not np.allclose(memory_H, memory_H @ np.linalg.pinv(H) @ H, rtol=1e-8, atol=1e-10)):
                    raise ValueError('Memory refresh must lie in measured row space')
                _, _, vt = np.linalg.svd(memory_H @ self.motion_units, full_matrices=False)
                supported = vt.T @ vt
            else:
                supported = np.zeros((3, 3))
            L = np.linalg.cholesky(self.anchor_covariance)
            _, _, svt = np.linalg.svd(H @ L, full_matrices=False)
            state_projector = L @ svt.T @ svt @ np.linalg.inv(L)
            if motion_reference is not None:
                delta = candidate-motion_reference
                if not local_linear_likelihood:
                    delta[2] = wrap(delta[2])
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
                if self.last_update is not None or delayed_first_anchor:
                    # Reacquisition has a wider *candidate* gate after an
                    # outage, but must not merely postpone the jump to the
                    # next accepted frame. Limit influence using THIS frame's
                    # motion/time budget. Use the same scaled gain in Joseph
                    # covariance; never clip only the mean and claim full info.
                    # Only frame zero can establish a free initial anchor.
                    # A first map update AFTER a prior-only output is already
                    # tracking: use this frame's budget, not the outage length.
                    increment = K @ innovation
                    if p.uses_v52_fusion() and not (p.v5121_enabled() and p.MAP_BODY_LATERAL_BUDGET):
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
                    else:
                        body_increment = body.T @ increment
                        gain_scale_longitudinal = min(1., step_xy_limit/max(abs(body_increment[0]), 1e-30))
                        gain_scale_lateral = min(1., step_lateral_limit/max(abs(body_increment[1]), 1e-30))
                        gain_scale_xy = min(gain_scale_longitudinal, gain_scale_lateral)
                        gain_scale_yaw = min(1., step_yaw_limit/max(abs(increment[2]), 1e-30))
                        K = state_projector @ body @ np.diag([
                            gain_scale_longitudinal, gain_scale_lateral, gain_scale_yaw]) @ body.T @ K
                        if dof == 3:
                            radial = min(1., step_xy_limit/max(np.linalg.norm((K @ innovation)[:2]),1e-30))
                            K = np.diag([radial,radial,1.]) @ K
                            gain_scale_xy *= radial
                        # Projection can couple axes if a single geometric mode
                        # physically mixes translation and yaw. Only that case
                        # needs a final common bound; full rank remains decoupled.
                        if dof < 3:
                            projected_increment = K @ innovation
                            projected_body = body.T @ projected_increment
                            gain_support_scale = min(1., step_xy_limit/max(np.linalg.norm(projected_increment[:2]),1e-30),
                                                     step_xy_limit/max(abs(projected_body[0]), 1e-30),
                                                     step_lateral_limit/max(abs(projected_body[1]), 1e-30),
                                                     step_yaw_limit/max(abs(projected_increment[2]), 1e-30))
                            K *= gain_support_scale
                proposal = self.mean+K @ innovation; proposal[2] = wrap(proposal[2])
                if proposal_validator is not None:
                    proposal_validation = proposal_validator(proposal.copy())
                    if p.v515_enabled():
                        # The local likelihood may be imperfect after partial
                        # fusion. Backtrack the ACTUAL gain on the same current
                        # sample; covariance and sensitivity use that gain too.
                        original_gain = K.copy()
                        attempts = [dict(gain_fraction=1., **proposal_validation)]
                        for half in range(1, 15):
                            if proposal_validation['accepted']:
                                break
                            fraction = 2.**-half
                            trial_gain = original_gain*fraction
                            trial = self.mean+trial_gain @ innovation
                            trial[2] = wrap(trial[2])
                            decision = proposal_validator(trial.copy())
                            attempts.append(dict(gain_fraction=fraction, **decision))
                            if decision['accepted']:
                                K, proposal, proposal_validation = trial_gain, trial, decision
                                nonlinear_gain_fraction = fraction
                        proposal_backtracking = dict(attempts=attempts,
                            accepted=bool(proposal_validation['accepted']),
                            objective='actual emitted pose/scale; fixed planes and rematched same sample')
                    if not proposal_validation['accepted']:
                        reason = proposal_validation['reason']
                if proposal_validation is not None and not proposal_validation['accepted']:
                    # No covariance, sensitivity, age or last-candidate commit.
                    K = None
                else:
                    self.mean = proposal
                if K is not None:
                    self._commit(candidate, candidate_sensitivity, H, K, effective_R, supported)
                    self.last_measurement_is_linear = bool(local_linear_likelihood)
                    gain = K.tolist()
                    weight = 1/inflation
                    reason = 'downweighted_update' if inflation > 1 or min(gain_scale_xy, gain_scale_yaw, gain_support_scale, nonlinear_gain_fraction) < 1 else 'measurement_update'
                    accepted = True
        diag = dict(accepted=accepted, reason=reason, frame='map XY metres, world yaw radians',
            delayed_first_anchor_budget_required=delayed_first_anchor,
            delayed_first_anchor_budget_applied=bool(delayed_first_anchor and accepted),
            proposal_validation=proposal_validation,
            proposal_backtracking=proposal_backtracking,
            nonlinear_gain_fraction=float(nonlinear_gain_fraction),
            local_linear_likelihood=bool(local_linear_likelihood),
            reference_response_constraint=self.reference_response_diagnostic,
            calibration_rebase_sensitivity=self.reference_rebase_sensitivity.tolist(),
            calibration_rebase_constraint=self.reference_rebase_diagnostic,
            prediction=self.predicted_mean.tolist(), correction=self.mean.tolist(),
            candidate=None if candidate is None else candidate.tolist(),
            innovation=None if innovation is None else innovation.tolist(), innovation_nis=nis,
            motion_innovation=None if motion_innovation is None else motion_innovation.tolist(),
            motion_reference=None if motion_reference is None else motion_reference.tolist(),
            measurement_weight=weight, kalman_gain=gain, gain_scale=float(min(gain_scale_xy, gain_scale_yaw, gain_support_scale)*nonlinear_gain_fraction),
            gain_scale_xy=float(gain_scale_xy), gain_scale_yaw=float(gain_scale_yaw),
            gain_scale_longitudinal=float(gain_scale_longitudinal), gain_scale_lateral=float(gain_scale_lateral),
            lateral_update_limit_m=step_lateral_limit, longitudinal_update_limit_m=step_xy_limit,
            correction_budget_frame=('v5.2 world XY norm/yaw' if p.uses_v52_fusion() and not (p.v5121_enabled() and p.MAP_BODY_LATERAL_BUDGET) else 'current sensor-prior forward/left/yaw; forward travel does not enlarge lateral budget'),
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
            memory_age_matrix_whitened=self.memory_age.tolist(),
            memory_age_eigenvalues_seconds=np.maximum(0., np.linalg.eigvalsh(self.memory_age)).tolist(),
            memory_decay_factors=self.memory_decay_factors.tolist(),
            memory_decay_matrix_world=self.memory_decay.tolist(),
            memory_metric=self.motion_units.tolist(),
            memory_age_policy=('fixed initial-prior whitening; only accepted strong row space resets; weak information does not refresh memory; fade before current measurement'
                if p.v515_enabled() else 'fixed initial-prior whitening; only accepted observable row space resets; fade before current measurement'),
            covariance_predicted=self.predicted_covariance.tolist(), covariance=self.covariance.tolist(),
            measurement_covariance=None if R is None else R.tolist(),
            seconds_since_update=None if self.last_update is None else (self.last_stamp-self.last_update)*1e-9,
            position_innovation_limit_m=pos_limit, yaw_innovation_limit_rad=yaw_limit,
            motion=self.motion)
        return self.mean.copy(), diag

    def _commit(self, candidate, candidate_sensitivity, H, K, effective_R, supported):
        A = np.eye(3)-K @ H
        self.reference_sensitivity = A @ self.reference_sensitivity+K @ H @ candidate_sensitivity
        if not p.uses_v52_fusion() or (p.v5121_enabled() and p.MAP_BOUNDED_REFERENCE_RESPONSE):
            self.reference_sensitivity, self.reference_response_diagnostic = bounded_reference_sensitivity(
                self.reference_sensitivity, self.anchor_covariance)
        self.covariance = A @ self.covariance @ A.T+K @ effective_R @ K.T
        self.covariance = (self.covariance+self.covariance.T)*.5
        self.last_update = self.last_stamp
        self.last_measurement = candidate.copy()
        self.last_candidate_sensitivity = candidate_sensitivity.copy()
        self.last_observation_projector = supported
        unseen = np.eye(3)-supported
        self.memory_age = unseen @ self.memory_age @ unseen
        self.memory_age = (self.memory_age+self.memory_age.T)*.5
        self.distance_since_update = self.turn_since_update = 0.
