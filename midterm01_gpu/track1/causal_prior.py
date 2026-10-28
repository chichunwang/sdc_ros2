"""5839746c sensor guidance 的因果替代；不讀 map posterior 或未來資料。

保留原 GNSS course / IMU orientation 的 heading-bias 模型與門檻。
改動：前向 position KF、完成後才交付 course block、前向 bias KF、
只用最新已抵達 IMU orientation。沒有 RTS、置中濾波或整段初始校正。
"""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from . import parameters as p


def wrap(x):
    return float(np.arctan2(np.sin(x), np.cos(x)))


class CausalSensorPrior:
    def __init__(self, initial_pose, timestamp_ns):
        self.time = int(timestamp_ns)
        self.position = np.r_[initial_pose[:2, 3], 0., 0.]
        self.cov = np.diag([100., 100., 100., 100.])
        self.height = float(initial_pose[2, 3])
        self.orientation = Rotation.from_matrix(initial_pose[:3, :3])
        self.orientation_stamp = None
        self.gyro = np.zeros(3)
        self.imu_history = deque()
        self.block = []
        self.block_origin = None
        self.block_number = 0
        self.bias = 0.
        self.bias_var = np.deg2rad(10.)**2
        self.bias_time = self.time
        self.last_public_bias_time = self.time
        self.gnss_updates = 0
        self.accepted_blocks = 0
        self.block_records = []

    def advance(self, timestamp_ns):
        if timestamp_ns < self.time:
            raise ValueError('Causal prior cannot move backward')
        dt = (int(timestamp_ns)-self.time)*1e-9
        F = np.eye(4); F[:2, 2:] = np.eye(2)*dt
        Q = p.GNSS_XY_ACCEL_PSD*np.block([[np.eye(2)*dt**3/3, np.eye(2)*dt**2/2],
                                        [np.eye(2)*dt**2/2, np.eye(2)*dt]])
        self.position = F @ self.position
        self.cov = F @ self.cov @ F.T+Q
        self.time = int(timestamp_ns)

    def _close_block(self, cutoff):
        if not self.block:
            return
        rows = self.block
        # The measurement is delivered at block END, never at its midpoint.
        row = dict(end_timestamp_ns=int(cutoff), samples=len(rows), accepted=False)
        self.block = []
        ts = np.array([r['timestamp_ns'] for r in rows], dtype=np.int64)
        t = (ts-ts[0])*1e-9
        if len(rows) < 5 or t[-1] < 1.:
            row['reason'] = 'insufficient_samples'; self.block_records.append(row); return
        imu = list(self.imu_history)
        if not imu or imu[0][0] > ts[0] or imu[-1][0] < ts[-1]:
            row['reason'] = 'insufficient_past_imu'; self.block_records.append(row); return
        # Interpolation here is wholly inside a completed PAST block; no frame
        # prior is interpolated toward a sensor sample beyond its cutoff.
        it = (np.array([r[0] for r in imu], dtype=np.int64)-ts[0])*1e-9
        yaw = np.unwrap([r[1] for r in imu])
        heading = np.interp(t, it, yaw)
        tau = t-t.mean(); A = np.column_stack([np.ones(len(t)), tau])
        xy = np.array([r['xy'] for r in rows]); xy -= xy[0]
        weights = np.ones(len(t))
        for _ in range(4):
            fit = np.linalg.lstsq(A*np.sqrt(weights[:, None]), xy*np.sqrt(weights[:, None]), rcond=None)[0]
            residual = xy-A @ fit
            weights = np.minimum(1., p.HEADING_ROBUST_GATE_SIGMA*p.GNSS_STD_M/
                                 np.maximum(np.linalg.norm(residual, axis=1), 1e-12))
        speed = float(np.linalg.norm(fit[1]))
        offset = wrap(np.arctan2(fit[1, 1], fit[1, 0])-heading.mean())
        # Block-local gradient uses only observations delivered by cutoff.
        turn = np.gradient(heading, t)
        row.update(speed_mps=speed, observation_deg=float(np.degrees(offset)))
        if speed <= p.HEADING_MIN_SPEED_MPS:
            row['reason'] = 'low_speed'
        elif np.max(np.abs(turn)) >= np.deg2rad(p.HEADING_MAX_TURN_RATE_DEGPS) or np.ptp(heading) >= np.deg2rad(p.HEADING_MAX_SPAN_DEG):
            row['reason'] = 'turning'
        elif abs(offset) >= np.deg2rad(p.HEADING_MAX_OBSERVATION_DEG):
            row['reason'] = 'heading_disagreement_or_reverse'
        elif np.count_nonzero(weights < .5) > 1:
            row['reason'] = 'inconsistent_gnss_block'
        else:
            normal = np.array([-fit[1, 1], fit[1, 0]])/speed
            scatter = max(p.GNSS_STD_M, float(np.sqrt(np.average((residual @ normal)**2, weights=weights))))
            sxx = float(np.sum(weights*(tau-np.average(tau, weights=weights))**2))
            std = max(np.deg2rad(p.HEADING_STD_FLOOR_DEG), scatter/(np.sqrt(sxx)*speed))
            process = np.deg2rad(p.HEADING_BIAS_RW_DEG_SQRT_S)**2
            self.bias_var += process*max(0., (cutoff-self.bias_time)*1e-9)
            self.bias_time = int(cutoff)
            # Past block observes its mean heading. Uncertainty grows from its
            # mean time to delivery; no retroactive state update is performed.
            delay = (cutoff-int(ts[0]))*1e-9-float(t.mean())
            R = std**2+process*max(delay, 0.)
            innovation = wrap(offset-self.bias)
            inflation = max(1., innovation**2/(self.bias_var+R)/p.HEADING_ROBUST_GATE_SIGMA**2)
            R *= inflation
            K = self.bias_var/(self.bias_var+R)
            candidate = np.clip(self.bias+K*innovation, -np.deg2rad(p.HEADING_MAX_BIAS_DEG), np.deg2rad(p.HEADING_MAX_BIAS_DEG))
            allowance = np.deg2rad(p.HEADING_MAX_BIAS_RATE_DEGPS)*max(0., (cutoff-self.last_public_bias_time)*1e-9)
            self.bias = float(np.clip(candidate, self.bias-allowance, self.bias+allowance))
            self.last_public_bias_time = int(cutoff)
            self.bias_var = (1-K)**2*self.bias_var+K*K*R
            self.accepted_blocks += 1
            row.update(accepted=True, reason='forward_bias_update', std_deg=float(np.degrees(std)), inflation=float(inflation))
        self.block_records.append(row)

    def event(self, row, *, preframe=False):
        stamp = row['timestamp_ns']
        if not preframe:
            self.advance(stamp)
        if row['kind'] == 'imu':
            q = np.asarray(row['quaternion'], dtype=float)
            self.orientation = Rotation.from_quat(q)
            self.orientation_stamp = stamp
            self.gyro = np.asarray(row['gyro'], dtype=float)
            yaw = float(self.orientation.as_euler('xyz')[2])
            self.imu_history.append((stamp, yaw))
            while self.imu_history and self.imu_history[0][0] < stamp-6_000_000_000:
                self.imu_history.popleft()
            return None
        # Pre-first-image samples can initialize at that frame but never create
        # velocity from a later GNSS point.
        if preframe:
            self.position[:2] = row['xy']; self.height = row['z']
            return dict(accepted=True, reason='available_initial_gnss')
        H = np.column_stack([np.eye(2), np.zeros((2, 2))])
        R = np.eye(2)*p.GNSS_STD_M**2
        r = np.asarray(row['xy'])-H @ self.position
        S = H @ self.cov @ H.T+R
        md = float(r @ np.linalg.solve(S, r))
        inflation = max(1., md/p.HEADING_ROBUST_GATE_SIGMA**2)
        R *= inflation
        K = np.linalg.solve(H @ self.cov @ H.T+R, H @ self.cov).T
        self.position += K @ r
        A = np.eye(4)-K @ H
        self.cov = A @ self.cov @ A.T+K @ R @ K.T
        self.height = float(row['z']); self.gnss_updates += 1
        width = int(round(p.HEADING_COURSE_WINDOW_S*1e9))
        if self.block_origin is None:
            self.block_origin = stamp
        number = (stamp-self.block_origin)//width
        if number > self.block_number:
            self._close_block(self.block_origin+(self.block_number+1)*width)
            self.block_number = int(number)
        self.block.append(row)
        return dict(accepted=True, reason='forward_gnss_kf', mahalanobis=md, inflation=inflation)

    def for_frame(self, camera_ns, previous_height=None):
        self.advance(camera_ns)
        width = int(round(p.HEADING_COURSE_WINDOW_S*1e9))
        if self.block_origin is not None and camera_ns >= self.block_origin+(self.block_number+1)*width:
            self._close_block(self.block_origin+(self.block_number+1)*width)
            self.block_number = (camera_ns-self.block_origin)//width
        orientation = self.orientation
        if self.orientation_stamp is not None:
            dt = (camera_ns-self.orientation_stamp)*1e-9
            if dt > p.IMU_MAX_AGE_S:
                raise ValueError('IMU orientation is stale; refusing invented sensor prior')
            orientation = orientation*Rotation.from_rotvec(self.gyro*max(0., dt))
        T = np.eye(4)
        T[:3, :3] = (Rotation.from_euler('z', self.bias)*orientation).as_matrix()
        T[:3, 3] = [*self.position[:2], self.height if previous_height is None else previous_height]
        return T, dict(heading_bias_rad=self.bias, heading_bias_std_rad=float(np.sqrt(self.bias_var)),
                       accepted_heading_blocks=self.accepted_blocks, gnss_updates=self.gnss_updates,
                       velocity_xy_mps=self.position[2:].tolist(), imu_orientation_timestamp_ns=self.orientation_stamp,
                       block_records=self.block_records.copy())
