"""Forward error-state Kalman filter; no future IMU, GNSS or pose rewriting.

State errors: [map position(3), velocity(3), left rotation(3), vertical gyro
bias(1), log depth scale(1)]. The reduced IMU model projects gyro onto the
orientation world's +Z, assumed parallel to map up. It needs no invented
camera/IMU extrinsic, but is a planar approximation, not full inertial navigation.
Noise defaults are conservative engineering assumptions, not calibrated sensor
covariances. Acceleration is recorded for auditing, never double integrated.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import chi2


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def proper_pose(pose):
    p = np.asarray(pose, dtype=np.float64)
    if (p.shape != (4, 4) or not np.isfinite(p).all()
            or not np.allclose(p[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or abs(np.linalg.det(p[:3, :3])-1) > 1e-6):
        raise ValueError('Expected finite proper T_map_source, not its inverse')
    return p.copy()


def read_initial_pose(path):
    """Same source->map convention as legacy, with strict SO(3) validation."""
    text = path.read_text().replace(',', ' ')
    values = np.fromstring(text, sep=' ')
    if values.size != 16:
        raise ValueError('init_guess must contain exactly 16 numeric values')
    return proper_pose(values.reshape(4, 4))


@dataclass(frozen=True)
class FilterConfig:
    initial_position_std_m: float = 2.
    initial_velocity_std_mps: float = 5.
    initial_rotation_std_deg: float = 8.
    initial_gyro_bias_std_degps: float = .5
    initial_log_scale_std: float = 1.
    acceleration_psd: float = 4.  # m^2/s^3, CV process noise, not accelerometer data
    gyro_noise_deg_sqrt_s: float = .5
    tilt_noise_deg_sqrt_s: float = .5
    gyro_bias_noise_degps_sqrt_s: float = .03
    log_scale_noise_sqrt_s: float = .1
    innovation_probability: float = .999
    max_speed_mps: float = 40.
    max_rotation_rate_degps: float = 90.
    position_jump_slack_m: float = .5
    rotation_jump_slack_deg: float = 3.
    max_map_correction_m: float = 3.
    initialization_correction_m: float = 5.
    max_log_scale_step: float = .15
    min_scale: float = .5
    max_scale: float = 16.
    max_gyro_bias_degps: float = 5.
    max_imu_age_s: float = .08
    tracking_lost_after_s: float = 2.

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('Filter parameters must be finite positive numbers')
        if not .5 < self.innovation_probability < 1:
            raise ValueError('innovation_probability must lie in (.5,1)')
        if self.min_scale >= self.max_scale:
            raise ValueError('Invalid global scale domain')


class ImuCursor:
    """ZOH availability-time stream. No endpoint interpolation or future samples."""
    def __init__(self, arrays):
        self.ns = np.asarray(arrays['availability_ns'])
        self.header_ns = np.asarray(arrays['measurement_ns'])
        self.quat = np.asarray(arrays['quaternion_xyzw'], dtype=float)
        self.gyro = np.asarray(arrays['gyro_rad_s'], dtype=float)
        n = len(self.ns)
        if (not n or self.ns.dtype != np.int64 or self.header_ns.dtype != np.int64
                or self.header_ns.shape != (n,) or np.any(self.ns <= 0)
                or np.any(np.diff(self.ns) <= 0) or np.any(self.header_ns <= 0)
                or np.any(self.header_ns > self.ns) or self.quat.shape != (n, 4)
                or self.gyro.shape != (n, 3) or not np.isfinite(self.quat).all()
                or not np.isfinite(self.gyro).all()
                or not np.allclose(np.linalg.norm(self.quat, axis=1), 1., atol=1e-3, rtol=0)):
            raise ValueError('Invalid availability-gated IMU arrays')
        self.index = 0
        self.last = None

    def take_until(self, ns):
        while self.index < len(self.ns) and int(self.ns[self.index]) <= ns:
            i = self.index
            q = self.quat[i] / np.linalg.norm(self.quat[i])
            # q maps IMU coordinates to its orientation world. Only its +Z axis
            # is used: absolute yaw cancels, and no camera/body axis is assumed.
            vertical_in_imu = Rotation.from_quat(q).as_matrix()[2]
            value = (int(self.ns[i]), int(self.header_ns[i]),
                     float(vertical_in_imu @ self.gyro[i]))
            self.last = value
            self.index += 1
            yield value


class CausalKalman:
    def __init__(self, initial_pose, timestamp_ns, initial_scale, config=None):
        self.config = config or FilterConfig()
        self.pose = proper_pose(initial_pose)
        if (not isinstance(timestamp_ns, (int, np.integer)) or timestamp_ns <= 0
                or not np.isfinite(initial_scale)
                or not self.config.min_scale <= initial_scale <= self.config.max_scale):
            raise ValueError('Invalid initial timestamp/scale')
        self.timestamp_ns = int(timestamp_ns)
        self.initial_ns = int(timestamp_ns)
        self.velocity = np.zeros(3)
        self.bias = 0.
        self.log_scale = float(np.log(initial_scale))
        c = self.config
        std = ([c.initial_position_std_m]*3 + [c.initial_velocity_std_mps]*3
               + [np.deg2rad(c.initial_rotation_std_deg)]*3
               + [np.deg2rad(c.initial_gyro_bias_std_degps), c.initial_log_scale_std])
        self.P = np.diag(np.square(std))
        self.held_imu = None
        self.last_map_ns = None
        self.published_pose = None
        self.published_ns = None

    def _advance(self, end_ns):
        if end_ns <= self.timestamp_ns:
            return
        if self.held_imu is None:
            raise ValueError('No available IMU before the current interval')
        c = self.config
        if (end_ns-self.held_imu[1])/1e9 > c.max_imu_age_s:
            raise ValueError('IMU is stale; prediction stopped, no future extrapolation')
        dt = (end_ns-self.timestamp_ns)/1e9
        angle = (self.held_imu[2]-self.bias)*dt
        rz = Rotation.from_rotvec([0., 0., angle]).as_matrix()
        f = np.eye(11)
        f[:3, 3:6] = np.eye(3)*dt
        f[6:9, 6:9] = rz
        f[8, 9] = -dt
        q = np.zeros((11, 11))
        for k in range(3):
            q[k, k] = c.acceleration_psd*dt**3/3
            q[k, k+3] = q[k+3, k] = c.acceleration_psd*dt**2/2
            q[k+3, k+3] = c.acceleration_psd*dt
        q[6, 6] = q[7, 7] = np.deg2rad(c.tilt_noise_deg_sqrt_s)**2*dt
        q[8, 8] = np.deg2rad(c.gyro_noise_deg_sqrt_s)**2*dt
        q[9, 9] = np.deg2rad(c.gyro_bias_noise_degps_sqrt_s)**2*dt
        q[10, 10] = c.log_scale_noise_sqrt_s**2*dt
        self.pose[:3, 3] += self.velocity*dt
        self.pose[:3, :3] = rz @ self.pose[:3, :3]
        self.P = f @ self.P @ f.T + q
        self.P = (self.P+self.P.T)/2
        self.timestamp_ns = int(end_ns)

    def predict_to(self, timestamp_ns, imu):
        ns = int(timestamp_ns)
        if ns < self.timestamp_ns:
            raise ValueError('Cannot revisit an already emitted timestamp')
        used = 0
        for record in imu.take_until(ns):
            if record[0] > self.timestamp_ns:
                self._advance(record[0])
            self.held_imu = record
            used += 1
        self._advance(ns)
        if self.held_imu is None or (ns-self.held_imu[1])/1e9 > self.config.max_imu_age_s:
            raise ValueError('No fresh past IMU for this frame')
        return {'imu_samples_consumed': used,
                'max_imu_availability_ns': self.held_imu[0],
                'last_imu_measurement_ns': self.held_imu[1],
                'imu_measurement_age_ms': (ns-self.held_imu[1])/1e6,
                'vertical_gyro_rad_s': self.held_imu[2],
                'gyro_bias_rad_s': self.bias,
                'prediction_covariance_diagonal': self.P.diagonal().tolist()}

    def update_map(self, observation, *, estimate_scale=True):
        c = self.config
        report = {'accepted': False, 'reasons': list(observation.get('reasons', [])),
                  'nis': None, 'nis_limit': None}
        if not observation.get('accepted'):
            return report
        try:
            measured = proper_pose(observation['pose'])
            scale = float(observation['scale'])
            covariance = np.asarray(observation['measurement_covariance'], float)
            if (not np.isfinite(scale) or scale <= 0 or covariance.shape != (7, 7)
                    or not np.isfinite(covariance).all()
                    or not np.allclose(covariance, covariance.T, atol=1e-9, rtol=0)
                    or np.linalg.eigvalsh(covariance)[0] <= 0):
                raise ValueError('Invalid map covariance/scale')
        except (KeyError, ValueError):
            report['reasons'].append('invalid_map_observation')
            return report
        bootstrap = self.last_map_ns is None
        rotation_error = Rotation.from_matrix(measured[:3, :3] @ self.pose[:3, :3].T).as_rotvec()
        residual = np.r_[measured[:3, 3]-self.pose[:3, 3], rotation_error,
                         np.log(scale)-self.log_scale]
        correction_limit = c.initialization_correction_m if bootstrap else c.max_map_correction_m
        if np.linalg.norm(residual[:3]) > correction_limit:
            report['reasons'].append('translation_innovation_bound')
        # Before the first map anchor, this corrects uncertain initialization,
        # not motion since an unlocalized published pose. Bound it by the
        # engineering 3-sigma initial rotation domain (24 degrees by default).
        initialization_rotation_limit = np.deg2rad(3*c.initial_rotation_std_deg)
        if bootstrap and np.linalg.norm(rotation_error) > initialization_rotation_limit:
            report['reasons'].append('initialization_rotation_bound')
        if not c.min_scale <= scale <= c.max_scale:
            report['reasons'].append('scale_domain')
        if not bootstrap and estimate_scale and abs(residual[6]) > c.max_log_scale_step:
            report['reasons'].append('scale_innovation_bound')
        if not bootstrap and self.published_pose is not None:
            dt = (self.timestamp_ns-self.published_ns)/1e9
            if np.linalg.norm(measured[:3, 3]-self.published_pose[:3, 3]) > c.max_speed_mps*dt+c.position_jump_slack_m:
                report['reasons'].append('physical_translation_jump')
            rotation_jump = Rotation.from_matrix(measured[:3, :3] @ self.published_pose[:3, :3].T).magnitude()
            if rotation_jump > np.deg2rad(c.max_rotation_rate_degps*dt+c.rotation_jump_slack_deg):
                report['reasons'].append('physical_rotation_jump')
        indices = [0, 1, 2, 6, 7, 8] + ([10] if estimate_scale else [])
        dim = len(indices)
        h = np.eye(11)[indices]
        residual = residual[:dim]
        covariance = covariance[:dim, :dim]
        innovation_covariance = h @ self.P @ h.T + covariance
        nis = float(residual @ np.linalg.solve(innovation_covariance, residual))
        limit = float(chi2.ppf(c.innovation_probability, dim))
        report.update(nis=nis, nis_limit=limit)
        if nis > limit:
            report['reasons'].append('kalman_innovation')
        if report['reasons']:
            return report
        gain = np.linalg.solve(innovation_covariance, h @ self.P).T
        delta = gain @ residual
        if np.linalg.norm(self.velocity+delta[3:6]) > c.max_speed_mps:
            report['reasons'].append('posterior_speed_bound')
            return report
        pose = self.pose.copy()
        pose[:3, 3] += delta[:3]
        pose[:3, :3] = Rotation.from_rotvec(delta[6:9]).as_matrix() @ pose[:3, :3]
        # Correlated Kalman gains may overshoot the raw observation. Apply the
        # physical/domain guards to the posterior too, before mutating state.
        posterior_log_scale = self.log_scale+float(delta[10]) if estimate_scale else self.log_scale
        if not np.log(c.min_scale) <= posterior_log_scale <= np.log(c.max_scale):
            report['reasons'].append('posterior_scale_domain')
        if not bootstrap and estimate_scale and abs(delta[10]) > c.max_log_scale_step:
            report['reasons'].append('posterior_scale_step')
        if abs(self.bias+delta[9]) > np.deg2rad(c.max_gyro_bias_degps):
            report['reasons'].append('posterior_gyro_bias_bound')
        if bootstrap and np.linalg.norm(delta[6:9]) > initialization_rotation_limit:
            report['reasons'].append('posterior_initialization_rotation_bound')
        if np.linalg.norm(delta[:3]) > correction_limit:
            report['reasons'].append('posterior_translation_bound')
        if not bootstrap and self.published_pose is not None:
            if np.linalg.norm(pose[:3, 3]-self.published_pose[:3, 3]) > c.max_speed_mps*dt+c.position_jump_slack_m:
                report['reasons'].append('posterior_physical_translation')
            rotation_jump = Rotation.from_matrix(pose[:3, :3] @ self.published_pose[:3, :3].T).magnitude()
            if rotation_jump > np.deg2rad(c.max_rotation_rate_degps*dt+c.rotation_jump_slack_deg):
                report['reasons'].append('posterior_physical_rotation')
        if report['reasons']:
            return report
        a = np.eye(11)-gain @ h
        posterior = a @ self.P @ a.T + gain @ covariance @ gain.T  # Joseph form
        reset = np.eye(11)
        reset[6:9, 6:9] -= .5*skew(delta[6:9])
        posterior = reset @ posterior @ reset.T
        self.P = (posterior+posterior.T)/2
        self.pose = proper_pose(pose)
        self.velocity += delta[3:6]
        self.bias += float(delta[9])
        if estimate_scale:
            self.log_scale = posterior_log_scale
        self.last_map_ns = self.timestamp_ns
        report.update(accepted=True, correction=delta.tolist())
        return report

    def publish(self):
        if self.published_ns == self.timestamp_ns:
            raise ValueError('A frame may only be published once')
        self.published_pose = self.pose.copy()
        self.published_ns = self.timestamp_ns
        supported_since = self.last_map_ns if self.last_map_ns is not None else self.initial_ns
        unsupported = (self.timestamp_ns-supported_since)/1e9
        return self.pose.copy(), {
            'velocity_mps': self.velocity.tolist(), 'gyro_bias_rad_s': self.bias,
            'scale': float(np.exp(self.log_scale)), 'unsupported_seconds': unsupported,
            'tracking_lost': unsupported > self.config.tracking_lost_after_s,
            'covariance_diagonal': self.P.diagonal().tolist()}
