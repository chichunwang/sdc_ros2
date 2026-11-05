"""因果 GNSS/IMU prior；只更新交付當下，不回寫已輸出的 pose。

GNSS course 使用尚未採用的過去樣本，窗口長度由方向不確定度決定。
輸出 covariance 的順序是 map [x_m, y_m, yaw_rad]，含零均值外參假設
的誤差預算；bag 全零 covariance 絕不解讀為零誤差。
"""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from . import parameters as p


def wrap(x):
    return float(np.arctan2(np.sin(x), np.cos(x)))


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def imu_covariance(values, fallback_std):
    """有效的非零 3×3 small-angle covariance；其餘明記 unknown fallback。"""
    a = np.asarray([] if values is None else values, dtype=float)
    if a.size == 9:
        a = a.reshape(3, 3)
        if (np.isfinite(a).all() and np.allclose(a, a.T, atol=1e-12)
                and np.linalg.eigvalsh(a).min() > 0):
            return a.copy(), 'message_covariance_body_small_angle_assumption'
    return np.diag(np.asarray(fallback_std, dtype=float)**2), 'unknown_engineering_budget'


class CausalSensorPrior:
    def __init__(self, initial_pose, timestamp_ns):
        self.time = int(timestamp_ns)
        self.position = np.r_[initial_pose[:2, 3], 0., 0.]
        self.cov = np.diag([100., 100., 100., 100.])
        self.height = float(initial_pose[2, 3])
        self.orientation = Rotation.from_matrix(initial_pose[:3, :3])
        self.orientation_stamp = None
        self.gyro = np.zeros(3)
        self.gyro_rotation_since_frame = 0.
        self.gyro_matrix_since_frame=np.eye(3)
        self.gyro_std_bound_since_frame=0.
        self.visual_heading_anchor=None
        self.gyro_motion_bias=None
        if p.v5162_enabled():
            from .coordinated_motion import StationaryGyroBias
            self.gyro_motion_bias=StationaryGyroBias(np.deg2rad(p.IMU_GYRO_UNKNOWN_STD_DEGPS),
                np.deg2rad(p.TURN_GYRO_REST_STD_DEGPS),np.deg2rad(p.TURN_GYRO_BIAS_RW_DEGPS_SQRT_S),
                p.IMAGE_SPEED_MAX_INTERVAL_S+p.IMU_MAX_AGE_S,p.CORRECTION_NIS_SOFT,p.CORRECTION_NIS_REJECT)
        self.orientation_cov, self.orientation_cov_source = imu_covariance(
            None, np.deg2rad(p.IMU_ORIENTATION_UNKNOWN_STD_DEG))
        self.gyro_cov, self.gyro_cov_source = imu_covariance(
            None, np.deg2rad(p.IMU_GYRO_UNKNOWN_STD_DEGPS))
        self.imu_history = deque()
        self.block = []
        self.course_pending = False
        self.last_course_decision = None
        self.heading_bias = np.zeros(2)  # [IMU->map yaw bias, GNSS->map yaw bias]
        self.heading_covariance = np.diag(np.deg2rad([
            p.HEADING_INITIAL_BIAS_STD_DEG, p.GNSS_HEADING_BIAS_STD_DEG])**2)
        self.gnss_heading_pending = []
        self.gnss_heading_updates = 0
        self.gnss_heading_decision = None
        self.last_gnss_heading_yaw = None
        self.last_gnss_heading_change_ns = None
        self.previous_gnss_heading_change_ns = None
        self.last_gnss_heading_value = None
        self.last_gnss_position_ns = None
        self.last_gnss_arrival_ns = None
        # Position/velocity changes caused by GNSS updates SINCE last emission.
        # Propagate to camera time before re-expressing map-relative memory.
        self.gnss_reference_shift = np.zeros(4)
        self.bias = 0.
        self.last_frame_bias = 0.
        self.bias_var = np.deg2rad(p.HEADING_INITIAL_BIAS_STD_DEG)**2
        self.last_public_bias_time = self.time
        self.gnss_updates = 0
        self.accepted_blocks = 0
        self.block_records = []
        self.lateral_constraint_gnss_count = 0
        self.lateral_constraint_diagnostic = None
        self.emitted_frames = 0
        self.pending_heading_information = []
        self.heading_information_updates = []
        self.heading_factor_sequence = 0
        self.heading_update_budget_used = 0.
        self.pending_position_information = []
        self.position_factor_sequence = 0
        self.position_information_updates = []
        self.last_position_frame_ns = None
        self.position_information_initialized = False

    @property
    def bias(self):
        return float(self.heading_bias[0])

    @bias.setter
    def bias(self, value):
        self.heading_bias[0] = value

    @property
    def bias_var(self):
        return float(self.heading_covariance[0, 0])

    @bias_var.setter
    def bias_var(self, value):
        self.heading_covariance[0, 0] = value

    def _heading_update_v52(self, observation, H, variance, *, course_rate_limit=False):
        """Current-time 2-state update; course anchors common bias, pair only its difference."""
        H = np.asarray(H, dtype=float)
        innovation = wrap(observation-float(H @ self.heading_bias))
        before = self.heading_covariance.copy()
        nis = innovation**2/float(H @ before @ H+variance)
        inflation = max(1., nis/p.HEADING_ROBUST_GATE_SIGMA**2)
        R = float(variance*inflation)
        K = before @ H/float(H @ before @ H+R)
        actual = K.copy()
        proposed = np.clip(self.bias+K[0]*innovation,
            -np.deg2rad(p.HEADING_MAX_BIAS_DEG), np.deg2rad(p.HEADING_MAX_BIAS_DEG))
        if course_rate_limit and self.accepted_blocks:
            allowance = np.deg2rad(p.HEADING_MAX_BIAS_RATE_DEGPS)*(self.time-self.last_public_bias_time)*1e-9
            proposed = np.clip(proposed,self.bias-allowance,self.bias+allowance)
        if abs(innovation) > 1e-15:
            actual[0] = (proposed-self.bias)/innovation
        self.heading_bias += actual*innovation
        self.heading_bias[:] = [wrap(v) for v in self.heading_bias]
        A = np.eye(2)-np.outer(actual,H)
        self.heading_covariance = A @ before @ A.T+np.outer(actual,actual)*R
        self.heading_covariance = (self.heading_covariance+self.heading_covariance.T)*.5
        self._condition_visual_anchor(A,float(actual[0]*innovation))
        return dict(innovation_rad=innovation,innovation_nis=float(nis),inflation=float(inflation),
            bias_gain=float(actual[0]),unconstrained_bias_gain=float(K[0]),
            bias_variance_before=float(before[0,0]),measurement_variance=R,
            bias_variance_after=self.bias_var,heading_gain=actual.tolist(),
            observation_matrix=H.tolist(),covariance_before=before.tolist(),
            covariance_after=self.heading_covariance.tolist())

    def _heading_factor_step(self, factor):
        """Consume at most the UNUSED fraction of one delivered likelihood.

        Static linear case: multiplying likelihood powers f_k (sum f_k=1)
        is exactly one update with the original R. No extra observations are
        manufactured. For a delayed remainder, bias random-walk uncertainty
        only enlarges R. The approximation never revises a past state.
        """
        H=np.asarray(factor['H']);before=self.heading_covariance.copy()
        innovation=wrap(factor['observation']-float(H@self.heading_bias))
        process=np.diag(np.deg2rad([p.HEADING_BIAS_RW_DEG_SQRT_S,p.GNSS_HEADING_BIAS_RW_DEG_SQRT_S])**2)
        age=max(0.,(self.time-factor['timestamp_ns'])*1e-9)
        base_R=factor['variance']+float(H@process@H)*age
        PH=before@H;s=float(H@PH)
        available=float(factor['remaining'])
        fraction=available
        full=PH/(s+base_R/max(available,1e-30))
        if self.emitted_frames:
            allowance=max(0.,np.deg2rad(p.HEADING_MAX_UPDATE_DEG)-self.heading_update_budget_used)
            proposed=abs(float(full[0]*innovation))
            if proposed>allowance:
                fraction=(0. if allowance<=1e-15 else
                    min(available,base_R/max(abs(float(PH[0]*innovation))/allowance-s,1e-30)))
        R=base_R/max(fraction,1e-30)
        K=np.zeros(2) if fraction==0 else PH/(s+R)
        actual=K.copy()
        value=np.clip(self.bias+K[0]*innovation,-np.deg2rad(p.HEADING_MAX_BIAS_DEG),np.deg2rad(p.HEADING_MAX_BIAS_DEG))
        if abs(innovation)>1e-15:actual[0]=(value-self.bias)/innovation
        self.heading_update_budget_used+=abs(value-self.bias)
        self.heading_bias+=actual*innovation
        self.heading_bias[:]=[wrap(v) for v in self.heading_bias]
        A=np.eye(2)-np.outer(actual,H)
        self.heading_covariance=A@before@A.T+np.outer(actual,actual)*R
        self.heading_covariance=(self.heading_covariance+self.heading_covariance.T)*.5
        self._condition_visual_anchor(A,float(actual[0]*innovation))
        factor['remaining']=max(0.,available-fraction)
        return dict(innovation_rad=innovation,innovation_nis=float(innovation**2/(s+base_R)),
            inflation=factor['inflation'],bias_gain=float(actual[0]),unconstrained_bias_gain=float(full[0]),
            bias_variance_before=float(before[0,0]),measurement_variance=float(R),
            bias_variance_after=self.bias_var,heading_gain=actual.tolist(),observation_matrix=H.tolist(),
            covariance_before=before.tolist(),covariance_after=self.heading_covariance.tolist(),
            heading_factor_id=factor['id'],information_fraction=float(fraction),
            information_fraction_remaining=factor['remaining'],base_measurement_variance=float(base_R),
            factor_age_seconds=age,model='one delivered likelihood split into powers; total fraction <= 1')

    def _heading_update(self, observation, H, variance, *, course_rate_limit=False):
        if p.uses_v52_fusion() and not p.v510_heading_enabled():
            return self._heading_update_v52(observation, H, variance, course_rate_limit=course_rate_limit)
        H=np.asarray(H,dtype=float)
        innovation=wrap(observation-float(H@self.heading_bias))
        nis=innovation**2/float(H@self.heading_covariance@H+variance)
        inflation=max(1.,nis/p.HEADING_ROBUST_GATE_SIGMA**2)
        self.heading_factor_sequence+=1
        factor=dict(id=self.heading_factor_sequence,observation=float(observation),H=H.tolist(),
            variance=float(variance*inflation),inflation=float(inflation),timestamp_ns=self.time,remaining=1.)
        update=self._heading_factor_step(factor)
        self.heading_information_updates.append(update)
        if factor['remaining']>1e-12:self.pending_heading_information.append(factor)
        return update

    def _consume_heading_information(self):
        remaining=[]
        for factor in self.pending_heading_information:
            if (self.time-factor['timestamp_ns'])*1e-9>p.HEADING_MAX_WINDOW_S:
                self.heading_information_updates.append(dict(heading_factor_id=factor['id'],
                    expired_information_fraction=factor['remaining'],reason='unused_heading_information_expired'))
                continue
            update=self._heading_factor_step(factor)
            self.heading_information_updates.append(update)
            if factor['remaining']>1e-12:remaining.append(factor)
        self.pending_heading_information=remaining

    def _lateral_velocity_constraint(self, orientation):
        """One soft vehicle constraint per NEW GNSS delivery, never per pixel/frame.

        State remains antenna [x,y,vx,vy]. Unknown body alignment, heading and
        angular-rate*lever uncertainty enter R; heading is NOT calibrated again
        from these same GNSS data. No saved pose is revisited.
        """
        if self.gnss_updates == self.lateral_constraint_gnss_count:
            self.lateral_constraint_diagnostic = dict(accepted=False, reason='no_new_gnss')
            return
        self.lateral_constraint_gnss_count = self.gnss_updates
        if p.uses_v52_fusion() or not p.LATERAL_VELOCITY_CONSTRAINT_ENABLED or self.orientation_stamp is None:
            self.lateral_constraint_diagnostic = dict(accepted=False, reason='disabled_or_no_imu')
            return
        heading = float(orientation.as_euler('xyz')[2])+self.bias
        normal = np.array([-np.sin(heading), np.cos(heading)])
        H = np.r_[0., 0., normal]
        speed = float(np.linalg.norm(self.position[2:]))
        yaw_var = self.bias_var+float(self.orientation_cov[2,2])+np.deg2rad(p.EXTRINSIC_ROTATION_STD_DEG[2])**2
        lever_var = float(np.sum(np.asarray(p.LEVER_ARM_STD_BODY_M)[:2]**2))
        # Nonzero known mean lever is removed using the current gyro; the
        # velocity constraint is on vehicle motion, not a fixed camera origin.
        Rmap = (Rotation.from_euler('z',self.bias)*orientation).as_matrix()
        lever_velocity = Rmap @ np.cross(self.gyro, np.asarray(p.GNSS_TO_PLY_LEVER_BODY_M))
        innovation = -float(normal @ (self.position[2:]+lever_velocity[:2]))
        variance = (p.LATERAL_VELOCITY_STD_MPS**2+speed**2*yaw_var
                    +float(np.linalg.norm(self.gyro))**2*lever_var)
        S = float(H @ self.cov @ H+variance)
        nis = innovation**2/S
        accepted = nis <= p.CORRECTION_NIS_REJECT_BY_DOF[0]
        inflation = max(1.,nis/p.CORRECTION_NIS_SOFT_BY_DOF[0])
        effective = variance*inflation
        K = self.cov @ H/float(H @ self.cov @ H+effective) if accepted else np.zeros(4)
        delta = K*innovation
        self.position += delta
        A = np.eye(4)-np.outer(K,H)
        self.cov = A @ self.cov @ A.T+np.outer(K,K)*effective
        self.cov = (self.cov+self.cov.T)*.5
        # This is a MODEL constraint, not another GNSS measurement/reference.
        self.lateral_constraint_diagnostic = dict(accepted=bool(accepted),
            reason='soft_body_lateral_velocity' if accepted else 'lateral_velocity_incompatible',
            gnss_update_count=self.gnss_updates,innovation_mps=innovation,variance_m2ps2=variance,
            innovation_nis=nis,gain=K.tolist(),state_delta=delta.tolist(),
            yaw_variance_rad2=yaw_var,lever_velocity_map_mps=lever_velocity.tolist(),
            model='unknown-slip/heading/extrinsic/lever budget; current-time vehicle velocity constraint; no heading update')

    def _gnss_heading(self):
        if not p.GNSS_HEADING_ENABLED:
            self.gnss_heading_pending.clear();return
        remaining=[]
        for row in self.gnss_heading_pending:
            stamp=int(row['timestamp_ns']);q=row.get('heading_quaternion')
            decision=dict(timestamp_ns=stamp,delivered_timestamp_ns=self.time,accepted=False)
            self.gnss_heading_decision=decision
            if q is None or np.shape(q) != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1.) > .02:
                decision['reason']='invalid_or_missing_gnss_orientation';continue
            history=list(self.imu_history)
            if not history or stamp < history[0][0]:
                decision['reason']='no_past_imu_coverage';continue
            if stamp > history[-1][0]:
                age=(stamp-history[-1][0])*1e-9
                if age > p.IMU_MAX_AGE_S:
                    decision['reason']='waiting_for_available_imu';remaining.append(row);continue
                raw=self.orientation*Rotation.from_rotvec(self.gyro*age)
                imu_yaw=float(raw.as_euler('xyz')[2])
            else:
                times=(np.array([x[0] for x in history],np.int64)-stamp)*1e-9
                imu_yaw=float(np.interp(0.,times,np.unwrap([x[1] for x in history])))
            gnss_yaw=float(Rotation.from_quat(q).as_euler('xyz')[2])
            if self.last_gnss_heading_yaw is not None and abs(wrap(gnss_yaw-self.last_gnss_heading_yaw)) <= 1e-6:
                decision['reason']='held_gnss_orientation_already_used';continue
            if self.last_gnss_heading_value is None:
                self.last_gnss_heading_change_ns = stamp
                self.last_gnss_heading_value = gnss_yaw
            elif abs(wrap(gnss_yaw-self.last_gnss_heading_value)) > 1e-6:
                self.previous_gnss_heading_change_ns = self.last_gnss_heading_change_ns
                self.last_gnss_heading_change_ns = stamp
                self.last_gnss_heading_value = gnss_yaw
            interval_start = self.previous_gnss_heading_change_ns
            observation=wrap(gnss_yaw-imu_yaw)
            decision.update(gnss_yaw_rad=gnss_yaw,imu_yaw_rad=wrap(imu_yaw),pair_difference_rad=observation)
            if np.linalg.norm(self.gyro) >= np.deg2rad(p.HEADING_MAX_TURN_RATE_DEGPS):
                decision['reason']='turning_or_held_gnss_heading';continue
            if abs(observation) > np.deg2rad(p.GNSS_HEADING_MAX_PAIR_DIFFERENCE_DEG):
                decision['reason']='incompatible_heading_frames';continue
            # H=[1,-1] observes DIFFERENCE only. Repeated held GNSS headings
            # cannot falsely remove the unobservable common absolute bias.
            process=np.deg2rad(p.HEADING_BIAS_RW_DEG_SQRT_S)**2+np.deg2rad(p.GNSS_HEADING_BIAS_RW_DEG_SQRT_S)**2
            variance=(np.deg2rad(p.GNSS_HEADING_OBSERVATION_STD_DEG)**2+
                      float(self.orientation_cov[2,2])+process*max(0.,(self.time-stamp)*1e-9))
            # PoseStamped has ONE header for XYZ and a frequently held heading.
            # No evidence licenses a fixed time shift. Model the ambiguity of
            # the heading epoch with rotation observed between value changes.
            freshness_variance = 0.
            if interval_start is not None and not p.uses_v52_fusion():
                available = [v for t,v in history if interval_start <= t <= stamp]
                if available:
                    spread = float(np.ptp(np.unwrap(available)))
                    freshness_variance = spread**2
                if interval_start < history[0][0]:
                    freshness_variance += np.deg2rad(p.GNSS_HEADING_BIAS_STD_DEG)**2
            variance += freshness_variance
            update=self._heading_update(observation,[1.,-1.],variance)
            self.gnss_heading_updates+=1
            self.last_gnss_heading_yaw=gnss_yaw
            decision.update(accepted=True,reason='causal_gnss_imu_relative_heading',
                orientation_previous_change_ns=interval_start,
                heading_epoch_uncertainty_variance=freshness_variance,
                physical_heading_delay_s=None, fixed_time_shift_applied=False, **update)
        self.gnss_heading_pending=remaining

    def advance(self, timestamp_ns):
        if timestamp_ns < self.time:
            raise ValueError('Causal prior cannot move backward')
        dt = (int(timestamp_ns)-self.time)*1e-9
        if self.gyro_motion_bias is None:
            F = np.eye(4); F[:2, 2:] = np.eye(2)*dt
            Q = p.GNSS_XY_ACCEL_PSD*np.block([[np.eye(2)*dt**3/3, np.eye(2)*dt**2/2],
                                            [np.eye(2)*dt**2/2, np.eye(2)*dt]])
        else:
            from .coordinated_motion import transition,yaw_rate,rate_sensitivity
            bias=self.gyro_motion_bias;bias.predict(dt)
            rate,variance=yaw_rate(self.orientation,self.gyro-bias.mean,self.gyro_cov+bias.cov,self.orientation_cov)
            F,Q=transition(dt,rate,p.GNSS_XY_ACCEL_PSD)
            sensitivity=rate_sensitivity(self.position[2:],dt,rate)
            Q+=np.outer(sensitivity,sensitivity)*variance
            bias.integrated_turn+=rate*dt;bias.frame_turn+=rate*dt
        self.gyro_rotation_since_frame += float(np.linalg.norm(self.gyro))*dt
        if p.v513_enabled():
            self.gyro_matrix_since_frame=self.gyro_matrix_since_frame@Rotation.from_rotvec(self.gyro*dt).as_matrix()
            self.gyro_std_bound_since_frame+=np.sqrt(max(0.,float(np.linalg.eigvalsh(self.gyro_cov).max())))*dt
        self.position = F @ self.position
        self.gnss_reference_shift = F @ self.gnss_reference_shift
        self.cov = F @ self.cov @ F.T+Q
        # Predict at EVERY delivery time, including intervals with no course.
        self.heading_covariance += np.diag(np.deg2rad([
            p.HEADING_BIAS_RW_DEG_SQRT_S,p.GNSS_HEADING_BIAS_RW_DEG_SQRT_S])**2)*dt
        self.time = int(timestamp_ns)

    def _try_course(self):
        if not self.block or not self.course_pending:
            return
        self.course_pending = False
        ts = np.array([r['timestamp_ns'] for r in self.block], dtype=np.int64)
        t = (ts-ts[0])*1e-9
        row = dict(start_timestamp_ns=int(ts[0]), end_timestamp_ns=int(ts[-1]),
                   delivered_timestamp_ns=self.time, samples=len(ts), accepted=False,
                   span_seconds=float(t[-1]))
        self.last_course_decision = row
        if len(ts) < p.HEADING_MIN_SAMPLES or t[-1] < p.HEADING_MIN_WINDOW_S:
            row['reason'] = 'insufficient_samples'; return
        imu = list(self.imu_history)
        if not imu or imu[0][0] > ts[0] or imu[-1][0] < ts[-1]:
            row['reason'] = 'insufficient_past_imu'
            self.course_pending = True  # later frame may have arrived IMU coverage
            return
        # All interpolation endpoints have ALREADY ARRIVED by self.time.
        it = (np.array([r[0] for r in imu], dtype=np.int64)-ts[0])*1e-9
        heading = np.interp(t, it, np.unwrap([r[1] for r in imu]))
        tau = t-t.mean(); A = np.column_stack([np.ones(len(t)), tau])
        xy = np.array([r['xy'] for r in self.block], dtype=float); xy -= xy[0]
        weights = np.ones(len(t))
        for _ in range(4):
            root = np.sqrt(weights[:, None])
            fit = np.linalg.lstsq(A*root, xy*root, rcond=None)[0]
            residual = xy-A @ fit
            weights = np.minimum(1., p.HEADING_ROBUST_GATE_SIGMA*p.GNSS_STD_M/
                                 np.maximum(np.linalg.norm(residual, axis=1), 1e-12))
        speed = float(np.linalg.norm(fit[1]))
        normal = np.array([-fit[1, 1], fit[1, 0]])/max(speed, 1e-12)
        scatter = max(p.GNSS_STD_M, float(np.sqrt(np.average((residual @ normal)**2, weights=weights))))
        sxx = float(np.sum(weights*(tau-np.average(tau, weights=weights))**2))
        std = max(np.deg2rad(p.HEADING_STD_FLOOR_DEG), scatter/max(np.sqrt(sxx)*speed, 1e-12))
        offset = wrap(np.arctan2(fit[1, 1], fit[1, 0])-float(np.average(heading, weights=weights)))
        turn = np.gradient(heading, t)
        if p.v510_heading_enabled() and len(t) >= 2*p.HEADING_MIN_SAMPLES:
            # An initial stop+acceleration window is not one constant course.
            # Require both disjoint halves to carry observable agreeing directions.
            estimates = []
            for ids in np.array_split(np.arange(len(t)), 2):
                tt = t[ids]-np.mean(t[ids]); XX = np.c_[np.ones(len(ids)), tt]
                slope = np.linalg.lstsq(XX, xy[ids], rcond=None)[0][1]
                ss = float(np.linalg.norm(slope)); sxx_half = float(tt @ tt)
                se = max(np.deg2rad(p.HEADING_STD_FLOOR_DEG), p.GNSS_STD_M/max(ss*np.sqrt(sxx_half), 1e-12))
                estimates.append((float(np.arctan2(slope[1], slope[0])), se, ss))
            agreement = wrap(estimates[0][0]-estimates[1][0])**2/(estimates[0][1]**2+estimates[1][1]**2)
            row['disjoint_half_course'] = [dict(yaw_rad=a, std_rad=b, speed_mps=c) for a,b,c in estimates]
            row['half_course_nis'] = agreement
            if max(e[1] for e in estimates) > np.deg2rad(p.HEADING_MAX_COURSE_STD_DEG) or agreement > p.CORRECTION_NIS_REJECT_BY_DOF[0]:
                row.update(reason='nonuniform_or_unobservable_course_window', speed_mps=speed, std_deg=float(np.degrees(std)))
                return
        row.update(speed_mps=speed, displacement_fit_m=speed*float(t[-1]),
                   observation_deg=float(np.degrees(offset)), std_deg=float(np.degrees(std)))
        if (np.max(np.abs(turn)) >= np.deg2rad(p.HEADING_MAX_TURN_RATE_DEGPS)
                or np.ptp(heading) >= np.deg2rad(p.HEADING_MAX_SPAN_DEG)):
            row['reason'] = 'turning'
            self.block_records.append(row.copy()); self.block = []
            return
        if np.count_nonzero(weights < .5) > 1:
            row['reason'] = 'inconsistent_gnss_block'
            self.block_records.append(row.copy()); self.block = []
            return
        if std > np.deg2rad(p.HEADING_MAX_COURSE_STD_DEG):
            row['reason'] = 'course_direction_unobservable'
            return  # grow/slide the UNUSED past window; stationary samples never create yaw
        if abs(offset) >= np.deg2rad(p.HEADING_MAX_OBSERVATION_DEG):
            row['reason'] = 'heading_disagreement_or_reverse'
            self.block_records.append(row.copy()); self.block = []
            return
        process = np.deg2rad(p.HEADING_BIAS_RW_DEG_SQRT_S)**2
        delay = (self.time-int(ts[0]))*1e-9-float(np.average(t, weights=weights))
        R = std**2+process*max(delay, 0.)
        update=self._heading_update(offset,[1.,0.],R,course_rate_limit=True)
        self.last_public_bias_time = self.time
        self.accepted_blocks += 1
        row.update(accepted=True, reason='forward_observable_course_update',
                   initial_alignment=self.accepted_blocks == 1, std_deg=float(np.degrees(std)),
                   delay_seconds=float(delay),**update)
        self.block_records.append(row.copy())
        self.block = []  # accepted course samples are NEVER reused as new independent data

    def exclude_image_stationary_course(self, start_ns, end_ns):
        """Past image/GNSS-confirmed rest is not a heading displacement baseline.

        Keep ALL GNSS fixes in the position filter. Only remove unused course
        samples inside this observed interval; never revise an accepted bias
        or an already-emitted pose. New moving samples build a fresh window.
        """
        if (type(start_ns) is not int or type(end_ns) is not int
                or start_ns >= end_ns or end_ns != self.time):
            raise ValueError('Stationary course interval must end at current state time')
        before = len(self.block)
        self.block = [r for r in self.block if not start_ns <= int(r['timestamp_ns']) <= end_ns]
        self.course_pending = bool(self.block)
        return dict(start_timestamp_ns=start_ns, end_timestamp_ns=end_ns,
            removed_unused_samples=before-len(self.block), remaining_unused_samples=len(self.block),
            policy='image AND GNSS accepted zero velocity; exclude only unused course samples; GNSS XY unchanged')

    def _condition_visual_anchor(self,A,calibration_step=0.):
        if self.visual_heading_anchor is not None:
            anchor=self.visual_heading_anchor
            anchor['cross']=A@anchor['cross']
            # GNSS/course change the common calibration reference, not the
            # camera's physical turn. Re-express the current measurement's
            # reference only; no old output pose is touched.
            anchor['bias']=wrap(anchor['bias']+calibration_step)
            anchor['camera_rotation']=Rotation.from_euler('z',calibration_step).as_matrix()@anchor['camera_rotation']
            anchor['calibration_rebase_rad']+=calibration_step

    def capture_visual_anchor(self,timestamp_ns,camera_rotation):
        if int(timestamp_ns)!=self.time:raise ValueError('Visual anchor must be current')
        E=Rotation.from_euler('xyz',p.IMU_TO_PLY_RPY_DEG,degrees=True).as_matrix()
        self.visual_heading_anchor=dict(timestamp_ns=int(timestamp_ns),bias=self.bias,
            variance=float(self.heading_covariance[0,0]),cross=self.heading_covariance[:,0].copy(),
            camera_rotation=np.asarray(camera_rotation).copy(),orientation_variance=float(self.orientation_cov[2,2]),
            orientation_covariance_camera=E.T@self.orientation_cov@E,used=False,calibration_rebase_rad=0.)

    def update_relative_heading(self,delta_bias,variance,keyframe_timestamp_ns):
        """Schmidt anchor clone: observes b_now-b_anchor, not absolute yaw.

        The anchor is a Schmidt nuisance during this measurement. Common
        calibration reference changes re-express it separately, never a saved
        pose. Current/anchor cross covariance follows every heading gain.
        """
        anchor=self.visual_heading_anchor
        if anchor is None or anchor['timestamp_ns']!=keyframe_timestamp_ns or keyframe_timestamp_ns>=self.time:
            raise ValueError('Relative yaw requires the matching past heading clone')
        if anchor['used']:raise ValueError('Relative image interval already consumed')
        if not np.isfinite([delta_bias,variance]).all() or variance<=0:raise ValueError('Invalid relative yaw likelihood')
        anchor['used']=True
        P=np.block([[self.heading_covariance,anchor['cross'][:,None]],
                    [anchor['cross'][None,:],np.array([[anchor['variance']]])]])
        H=np.array([1.,0.,-1.]);innovation=wrap(delta_bias-(self.bias-anchor['bias']))
        S=float(H@P@H+variance);nis=innovation**2/S
        allowance=max(0.,np.deg2rad(p.HEADING_MAX_UPDATE_DEG)-self.heading_update_budget_used)
        d=dict(accepted=False,reason='visual_relative_heading_statistical_gate',innovation_rad=innovation,
            innovation_nis=float(nis),anchor_timestamp_ns=keyframe_timestamp_ns,anchor_bias_rad=anchor['bias'],
            joint_covariance_before=P.tolist(),observation_matrix=H.tolist(),absolute_yaw_observed=False)
        if nis>p.CORRECTION_NIS_REJECT_BY_DOF[0]:return d
        if allowance<=1e-15:return dict(d,reason='visual_relative_heading_budget_exhausted')
        R=variance*max(1.,nis/p.CORRECTION_NIS_SOFT_BY_DOF[0]);S=float(H@P@H+R)
        K=(P@H)[:2]/S;step=K*innovation
        K*=min(1.,allowance/max(abs(step[0]),1e-30))
        new_bias=np.clip(self.bias+K[0]*innovation,-np.deg2rad(p.HEADING_MAX_BIAS_DEG),np.deg2rad(p.HEADING_MAX_BIAS_DEG))
        if abs(innovation)>1e-15:K[0]=(new_bias-self.bias)/innovation
        gain=np.r_[K,0.];A=np.eye(3)-np.outer(gain,H)
        after=A@P@A.T+np.outer(gain,gain)*R;after=(after+after.T)*.5
        self.heading_bias+=K*innovation;self.heading_bias[:]=[wrap(x) for x in self.heading_bias]
        self.heading_covariance=after[:2,:2];anchor['cross']=after[:2,2]
        self.heading_update_budget_used+=abs(K[0]*innovation)
        return dict(d,accepted=True,reason='visual_relative_heading_schmidt_update',
            kalman_gain=gain.tolist(),effective_measurement_variance=float(R),joint_covariance_after=after.tolist(),
            bias_step_rad=float(K[0]*innovation),model='current bias difference with frozen causal anchor; common absolute yaw gauge retained')

    def visual_pose_covariance(self,rotation,orientation_age):
        extrinsic=Rotation.from_euler('xyz',p.IMU_TO_PLY_RPY_DEG,degrees=True).as_matrix()
        body=rotation@extrinsic.T;lever=body@np.asarray(p.GNSS_TO_PLY_LEVER_BODY_M)
        Ctheta=body@(self.orientation_cov+orientation_age**2*self.gyro_cov)@body.T;Ctheta[2,2]+=self.bias_var
        f=rotation[:,0];horizontal=float(f[0]**2+f[1]**2)
        if horizontal<1e-8:raise ValueError('Camera yaw undefined')
        Jyaw=np.array([-f[2]*f[0]/horizontal,-f[2]*f[1]/horizontal,1.]);Jlever=-skew(lever)[:2]
        Cextrinsic=body@np.diag(np.deg2rad(p.EXTRINSIC_ROTATION_STD_DEG)**2)@body.T
        Clever=body@np.diag(np.asarray(p.LEVER_ARM_STD_BODY_M)**2)@body.T
        C=np.zeros((3,3));C[:2,:2]=self.cov[:2,:2]+Clever[:2,:2]+Jlever@Ctheta@Jlever.T
        C[:2,2]=Jlever@Ctheta@Jyaw;C[2,:2]=C[:2,2];C[2,2]=Jyaw@(Ctheta+Cextrinsic)@Jyaw
        return C

    def update_gyro_motion_bias(self,start_ns,end_ns):
        if self.gyro_motion_bias is None:return dict(accepted=False,reason='disabled')
        return self.gyro_motion_bias.update(start_ns,end_ns,self.time)

    def event(self, row, *, preframe=False):
        stamp = int(row['timestamp_ns'])
        if not preframe:
            self.advance(stamp)
        if row['kind'] == 'imu':
            self.orientation = Rotation.from_quat(np.asarray(row['quaternion'], dtype=float))
            self.orientation_stamp = stamp
            self.gyro = np.asarray(row['gyro'], dtype=float)
            self.orientation_cov, self.orientation_cov_source = imu_covariance(
                row.get('orientation_covariance'), np.deg2rad(p.IMU_ORIENTATION_UNKNOWN_STD_DEG))
            self.gyro_cov, self.gyro_cov_source = imu_covariance(
                row.get('gyro_covariance'), np.deg2rad(p.IMU_GYRO_UNKNOWN_STD_DEGPS))
            yaw = float(self.orientation.as_euler('xyz')[2])
            self.imu_history.append((stamp, yaw))
            if self.gyro_motion_bias is not None:
                self.gyro_motion_bias.observe(stamp,self.gyro,self.time)
            keep_ns = int((p.HEADING_MAX_WINDOW_S+p.IMU_MAX_AGE_S+1.)*1e9)
            while len(self.imu_history) > 1 and self.imu_history[1][0] < stamp-keep_ns:
                self.imu_history.popleft()
            return None
        if preframe:
            self.gnss_reference_shift[:2] += np.asarray(row['xy'])-self.position[:2]
            self.position[:2] = row['xy']; self.height = float(row['z'])
            self.cov[:2, :2] = np.eye(2)*p.GNSS_STD_M**2
            self.cov[:2, 2:] = self.cov[2:, :2] = 0.
            self.position_information_initialized = True
            decision = dict(accepted=True, reason='available_initial_gnss')
        elif p.v5121_enabled() and p.GNSS_POSITION_INFORMATION_BUDGET:
            # Freeze one original likelihood; no position jump at delivery.
            H = np.c_[np.eye(2),np.zeros((2,2))]
            innovation = np.asarray(row['xy'])-H@self.position
            md = float(innovation@np.linalg.solve(H@self.cov@H.T+np.eye(2)*p.GNSS_STD_M**2,innovation))
            inflation = max(1.,md/p.HEADING_ROBUST_GATE_SIGMA**2)
            angle=float(self.orientation.as_euler('xyz')[2])+self.bias
            axes=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
            # Isotropic XY R factors EXACTLY into two orthogonal 1D factors.
            # Separating them keeps lateral tempering from starving forward
            # velocity/acceleration information from the same delivered fix.
            ids=[]
            # During the initial handoff, one COMMON XY power preserves the
            # original innovation direction instead of drawing forward-then-
            # sideways steps. Tracking can use independent orthogonal powers.
            directions=enumerate(axes) if self.position_information_initialized else [(None,None)]
            for axis,direction in directions:
                self.position_factor_sequence += 1
                factor = dict(id=self.position_factor_sequence,axis=axis,
                    direction=None if direction is None else direction.tolist(),timestamp_ns=stamp,
                    xy=list(row['xy']),variance=p.GNSS_STD_M**2*inflation,remaining=1.)
                self.pending_position_information.append(factor);ids.append(factor['id'])
            decision = dict(accepted=True,reason='arrived_gnss_position_likelihood_queued',
                position_factor_ids=ids,mahalanobis=md,inflation=inflation,
                position_velocity_reference_shift=[0.,0.,0.,0.])
        else:
            H = np.column_stack([np.eye(2), np.zeros((2, 2))])
            R = np.eye(2)*p.GNSS_STD_M**2
            r = np.asarray(row['xy'])-H @ self.position
            S = H @ self.cov @ H.T+R
            md = float(r @ np.linalg.solve(S, r))
            inflation = max(1., md/p.HEADING_ROBUST_GATE_SIGMA**2)
            R *= inflation
            K = np.linalg.solve(H @ self.cov @ H.T+R, H @ self.cov).T
            increment = K @ r
            self.position += increment
            self.gnss_reference_shift += increment
            B = np.eye(4)-K @ H
            self.cov = B @ self.cov @ B.T+K @ R @ K.T
            self.height = float(row['z'])
            decision = dict(accepted=True, reason='forward_gnss_kf', mahalanobis=md, inflation=inflation,
                            position_velocity_reference_shift=increment.tolist())
        self.last_gnss_position_ns = stamp
        self.last_gnss_arrival_ns = row.get('arrival_ns')
        self.gnss_updates += 1
        self.block.append(row)
        max_ns = int(p.HEADING_MAX_WINDOW_S*1e9)
        while len(self.block) > 1 and stamp-self.block[0]['timestamp_ns'] > max_ns:
            self.block.pop(0)
        self.course_pending = True
        if p.GNSS_HEADING_ENABLED and 'heading_quaternion' in row:
            self.gnss_heading_pending.append(row)
        return decision

    def _consume_position_information(self, orientation):
        from .gnss_budget import fractional_position_update
        self.position_information_updates = []
        if not (p.v5121_enabled() and p.GNSS_POSITION_INFORMATION_BUDGET):
            return
        dt=0. if self.last_position_frame_ns is None else (self.time-self.last_position_frame_ns)*1e-9
        self.last_position_frame_ns=self.time
        angle=float(orientation.as_euler('xyz')[2])+self.bias
        body=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        radial=(p.CORRECTION_POSITION_RATE_MPS*dt+p.CORRECTION_SPEED_FRACTION
                *float(np.linalg.norm(self.position[2:]))*dt
                +p.CORRECTION_TURN_LEVER_M*self.gyro_rotation_since_frame)
        # This is a GNSS state innovation, which includes real acceleration.
        # Heading uncertainty turns forward travel into uncertain body-left
        # travel. Account for it here; the map-bias budget stays independent
        # of forward speed and no lateral-zero pseudo-observation is added.
        yaw_std=np.sqrt(self.bias_var+float(self.orientation_cov[2,2])
                        +np.deg2rad(p.EXTRINSIC_ROTATION_STD_DEG[2])**2)
        lateral=((p.CORRECTION_LATERAL_RATE_MPS+float(np.linalg.norm(self.position[2:]))*yaw_std)*dt
                 +p.CORRECTION_TURN_LEVER_M*self.gyro_rotation_since_frame)
        used_radial=used_lateral=0.;remaining=[]
        for factor in self.pending_position_information:
            age=(self.time-factor['timestamp_ns'])*1e-9
            if age<0:raise ValueError('Future GNSS position factor')
            # Past CV process uncertainty has exceeded one original GNSS R:
            # expire unused information instead of accumulating stale factors.
            if p.GNSS_XY_ACCEL_PSD*age**3/3.>p.GNSS_STD_M**2:
                self.position_information_updates.append(dict(position_factor_id=factor['id'],
                    expired_information_fraction=factor['remaining'],reason='unused_gnss_information_expired_cv_uncertainty'))
                continue
            before=self.position.copy()
            self.position,self.cov,d=fractional_position_update(self.position,self.cov,factor['xy'],
                variance=factor['variance'],age_seconds=age,remaining=factor['remaining'],
                acceleration_psd=p.GNSS_XY_ACCEL_PSD,body_rotation=body,
                direction=factor['direction'],
                radial_limit=None if not self.emitted_frames or self.position_information_initialized else max(0.,radial-used_radial),
                lateral_limit=None if not self.emitted_frames else max(0.,lateral-used_lateral))
            increment=self.position-before
            used_radial+=float(np.linalg.norm(increment[:2]));used_lateral+=abs(float((body.T@increment[:2])[1]))
            self.gnss_reference_shift+=increment
            factor['remaining']=d['information_fraction_remaining']
            d['position_factor_id']=factor['id']
            d['coupled_startup_xy']=factor['direction'] is None
            self.position_information_updates.append(d)
            if factor['remaining']>1e-12:remaining.append(factor)
            elif factor['direction'] is None:self.position_information_initialized=True
        self.pending_position_information=remaining

    def for_frame(self, camera_ns, previous_height=None):
        self.advance(camera_ns)
        self.heading_update_budget_used = 0.
        self.heading_information_updates = []
        if not p.uses_v52_fusion() or p.v510_heading_enabled():
            self._consume_heading_information()
        self._gnss_heading()
        self._try_course()
        orientation = self.orientation
        dt = 0.
        if self.orientation_stamp is not None:
            dt = (camera_ns-self.orientation_stamp)*1e-9
            if dt < 0 or dt > p.IMU_MAX_AGE_S:
                raise ValueError('IMU orientation is stale/future; refusing invented sensor prior')
            orientation = orientation*Rotation.from_rotvec(self.gyro*dt)
        self._consume_position_information(orientation)
        self._lateral_velocity_constraint(orientation)
        body = (Rotation.from_euler('z', self.bias)*orientation).as_matrix()
        extrinsic = Rotation.from_euler('xyz', p.IMU_TO_PLY_RPY_DEG, degrees=True).as_matrix()
        T = np.eye(4); T[:3, :3] = body @ extrinsic
        lever = body @ np.asarray(p.GNSS_TO_PLY_LEVER_BODY_M, dtype=float)
        T[:3, 3] = np.r_[self.position[:2], self.height]+lever
        if previous_height is not None:
            T[2, 3] = previous_height  # already camera world height; never add lever twice
        # Left world small-angle covariance and the exact first-order yaw Jacobian.
        Ctheta = body @ (self.orientation_cov+dt**2*self.gyro_cov) @ body.T
        Ctheta[2, 2] += self.bias_var
        Jlever = -skew(lever)[:2]
        f = T[:3, 0]; horizontal = float(f[0]**2+f[1]**2)
        if horizontal < 1e-8:
            raise ValueError('Camera yaw is undefined near vertical forward direction')
        Jyaw = np.array([-f[2]*f[0]/horizontal, -f[2]*f[1]/horizontal, 1.])
        Cextrinsic = body @ np.diag(np.deg2rad(p.EXTRINSIC_ROTATION_STD_DEG)**2) @ body.T
        Clever = body @ np.diag(np.asarray(p.LEVER_ARM_STD_BODY_M)**2) @ body.T
        C = np.zeros((3, 3))
        C[:2, :2] = self.cov[:2, :2]+Clever[:2, :2]+Jlever @ Ctheta @ Jlever.T
        C[:2, 2] = Jlever @ Ctheta @ Jyaw
        C[2, :2] = C[:2, 2]
        C[2, 2] = Jyaw @ (Ctheta+Cextrinsic) @ Jyaw
        bias_delta = wrap(self.bias-self.last_frame_bias)
        old_lever_same_orientation = Rotation.from_euler('z', -bias_delta).as_matrix() @ lever
        position_reference_shift = self.gnss_reference_shift[:2].copy()
        # A GNSS state innovation corrects CV prediction error (including real
        # acceleration); it is NOT a change of extrinsic/calibration reference.
        # Only the heading calibration and its lever-arm displacement rebase
        # learned map memory. Keep the propagated XY innovation for diagnostics.
        calibration_delta = np.r_[(lever-old_lever_same_orientation)[:2], bias_delta]
        self.gnss_reference_shift[:] = 0.
        self.last_frame_bias = self.bias
        physical_rotation = self.gyro_rotation_since_frame
        self.gyro_rotation_since_frame = 0.
        E=Rotation.from_euler('xyz',p.IMU_TO_PLY_RPY_DEG,degrees=True).as_matrix()
        gyro_matrix=E.T@self.gyro_matrix_since_frame@E
        gyro_std_bound=self.gyro_std_bound_since_frame;self.gyro_std_bound_since_frame=0.
        self.gyro_matrix_since_frame=np.eye(3)
        self.emitted_frames += 1
        return T, dict(heading_information_policy=('fractional_likelihood_budget_v510' if p.v510_heading_enabled() else 'baseline_or_historical'),
            position_information_policy=('fractional_likelihood_budget_v5121' if p.v5121_enabled() and p.GNSS_POSITION_INFORMATION_BUDGET else 'v510_immediate_gnss_kf'),
            position_information_updates=self.position_information_updates.copy(),
            pending_position_information=len(self.pending_position_information),
            course_guard_enabled=bool(p.v510_heading_enabled()),
            heading_update_budget_limit_deg=float(p.HEADING_MAX_UPDATE_DEG),
            gyro_rotation_since_frame_rad=physical_rotation,
            gyro_rotation_matrix_since_frame=gyro_matrix.tolist(),imu_orientation_extrapolation_s=float(dt),
            gyro_rotation_std_bound_rad=float(gyro_std_bound),
            lateral_velocity_constraint=self.lateral_constraint_diagnostic,
            heading_update_budget_used_deg=float(np.degrees(self.heading_update_budget_used)),
            heading_information_updates=self.heading_information_updates.copy(),
            pending_heading_information=len(self.pending_heading_information),
            calibration_delta_xy_yaw=calibration_delta.tolist(),
            gnss_position_reference_shift_xy=position_reference_shift.tolist(),
            gnss_state_innovation_xy=position_reference_shift.tolist(),
            calibration_delta_policy='heading calibration and induced lever only; GNSS XY state innovation passes through',
            pose_timestamp_ns=int(camera_ns),
            latest_gnss_position_timestamp_ns=self.last_gnss_position_ns,
            latest_gnss_arrival_ns=self.last_gnss_arrival_ns,
            gnss_position_age_s=None if self.last_gnss_position_ns is None else (int(camera_ns)-self.last_gnss_position_ns)*1e-9,
            gnss_position_prediction=('GNSS update at measurement header, arrived-gyro coordinated turn to camera time' if self.gyro_motion_bias is not None else 'GNSS update at measurement header, CV propagation to camera acquisition time'),
            heading_bias_rad=self.bias, heading_bias_std_rad=float(np.sqrt(self.bias_var)),
            heading_initialized=bool(self.accepted_blocks or self.gnss_heading_updates),
            gnss_heading_updates=self.gnss_heading_updates,
            gnss_heading_bias_rad=float(self.heading_bias[1]),
            heading_bias_covariance=self.heading_covariance.tolist(),
            gnss_heading_decision=self.gnss_heading_decision,
            accepted_heading_blocks=self.accepted_blocks, gnss_updates=self.gnss_updates,
            velocity_xy_mps=self.position[2:].tolist(), imu_orientation_timestamp_ns=self.orientation_stamp,
            pose_covariance_xy_yaw=C.tolist(), pose_covariance_frame='map [x_m,y_m,yaw_rad]',
            position_covariance_xy=self.cov[:2, :2].tolist(),
            uncertainty_model=dict(orientation=self.orientation_cov_source, gyro=self.gyro_cov_source,
                imu_covariance_axes='body small-angle assumption; not calibrated',
                lever_body_m=list(p.GNSS_TO_PLY_LEVER_BODY_M), lever_std_body_m=list(p.LEVER_ARM_STD_BODY_M),
                extrinsic_rpy_deg=list(p.IMU_TO_PLY_RPY_DEG),
                extrinsic_rotation_std_deg=list(p.EXTRINSIC_ROTATION_STD_DEG),
                gnss_course_position_correlation='not explicitly estimated; conservative error floors',
                gnss_heading_epoch=('unknown physical epoch; v5.2 observation variance, no fixed shift' if p.uses_v52_fusion() else 'unknown physical epoch; variance includes past rotation between value changes; no estimated fixed shift'),
                heading_bias_model='independent initial IMU/GNSS biases; shared covariance after pair update; uncalibrated'),
            course_decision=self.last_course_decision, block_records=self.block_records.copy(),
            **({'motion_prediction':self.gyro_motion_bias.frame_diagnostic()} if self.gyro_motion_bias is not None else {}))
