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
        self.bias = 0.
        self.last_frame_bias = 0.
        self.bias_var = np.deg2rad(p.HEADING_INITIAL_BIAS_STD_DEG)**2
        self.last_public_bias_time = self.time
        self.gnss_updates = 0
        self.accepted_blocks = 0
        self.block_records = []

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

    def _heading_update(self, observation, H, variance, *, course_rate_limit=False):
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
        return dict(innovation_rad=innovation,innovation_nis=float(nis),inflation=float(inflation),
            bias_gain=float(actual[0]),unconstrained_bias_gain=float(K[0]),
            bias_variance_before=float(before[0,0]),measurement_variance=R,
            bias_variance_after=self.bias_var,heading_gain=actual.tolist(),
            observation_matrix=H.tolist(),covariance_before=before.tolist(),
            covariance_after=self.heading_covariance.tolist())

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
            update=self._heading_update(observation,[1.,-1.],variance)
            self.gnss_heading_updates+=1
            self.last_gnss_heading_yaw=gnss_yaw
            decision.update(accepted=True,reason='causal_gnss_imu_relative_heading',**update)
        self.gnss_heading_pending=remaining

    def advance(self, timestamp_ns):
        if timestamp_ns < self.time:
            raise ValueError('Causal prior cannot move backward')
        dt = (int(timestamp_ns)-self.time)*1e-9
        F = np.eye(4); F[:2, 2:] = np.eye(2)*dt
        Q = p.GNSS_XY_ACCEL_PSD*np.block([[np.eye(2)*dt**3/3, np.eye(2)*dt**2/2],
                                        [np.eye(2)*dt**2/2, np.eye(2)*dt]])
        self.gyro_rotation_since_frame += float(np.linalg.norm(self.gyro))*dt
        self.position = F @ self.position
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
            keep_ns = int((p.HEADING_MAX_WINDOW_S+p.IMU_MAX_AGE_S+1.)*1e9)
            while len(self.imu_history) > 1 and self.imu_history[1][0] < stamp-keep_ns:
                self.imu_history.popleft()
            return None
        if preframe:
            self.position[:2] = row['xy']; self.height = float(row['z'])
            self.cov[:2, :2] = np.eye(2)*p.GNSS_STD_M**2
            self.cov[:2, 2:] = self.cov[2:, :2] = 0.
            decision = dict(accepted=True, reason='available_initial_gnss')
        else:
            H = np.column_stack([np.eye(2), np.zeros((2, 2))])
            R = np.eye(2)*p.GNSS_STD_M**2
            r = np.asarray(row['xy'])-H @ self.position
            S = H @ self.cov @ H.T+R
            md = float(r @ np.linalg.solve(S, r))
            inflation = max(1., md/p.HEADING_ROBUST_GATE_SIGMA**2)
            R *= inflation
            K = np.linalg.solve(H @ self.cov @ H.T+R, H @ self.cov).T
            self.position += K @ r
            B = np.eye(4)-K @ H
            self.cov = B @ self.cov @ B.T+K @ R @ K.T
            self.height = float(row['z'])
            decision = dict(accepted=True, reason='forward_gnss_kf', mahalanobis=md, inflation=inflation)
        self.gnss_updates += 1
        self.block.append(row)
        max_ns = int(p.HEADING_MAX_WINDOW_S*1e9)
        while len(self.block) > 1 and stamp-self.block[0]['timestamp_ns'] > max_ns:
            self.block.pop(0)
        self.course_pending = True
        if p.GNSS_HEADING_ENABLED and 'heading_quaternion' in row:
            self.gnss_heading_pending.append(row)
        return decision

    def for_frame(self, camera_ns, previous_height=None):
        self.advance(camera_ns)
        self._gnss_heading()
        self._try_course()
        orientation = self.orientation
        dt = 0.
        if self.orientation_stamp is not None:
            dt = (camera_ns-self.orientation_stamp)*1e-9
            if dt < 0 or dt > p.IMU_MAX_AGE_S:
                raise ValueError('IMU orientation is stale/future; refusing invented sensor prior')
            orientation = orientation*Rotation.from_rotvec(self.gyro*dt)
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
        calibration_delta = np.r_[(lever-old_lever_same_orientation)[:2], bias_delta]
        self.last_frame_bias = self.bias
        physical_rotation = self.gyro_rotation_since_frame
        self.gyro_rotation_since_frame = 0.
        return T, dict(gyro_rotation_since_frame_rad=physical_rotation,
            calibration_delta_xy_yaw=calibration_delta.tolist(),
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
                heading_bias_model='independent initial IMU/GNSS biases; shared covariance after pair update; uncalibrated'),
            course_decision=self.last_course_decision, block_records=self.block_records.copy())
