"""Experimental GNSS/gyro planar batch model; not a calibrated inertial navigator.

States: local x,y,vx,vy,unwrapped yaw,effective yaw-rate bias at GNSS timestamps.
The soft nonholonomic constraint is a vehicle-model assumption, not a sensor.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class KinematicConfig:
    gnss_std_m: float = .3
    accel_psd: float = 4.
    gyro_density_deg_sqrt_s: float = .3
    bias_rw_degps_sqrt_s: float = .02
    lateral_velocity_density_mps_sqrt_s: float = .5
    initial_yaw_std_deg: float = 30.
    initial_rate_bias_std_degps: float = 1.
    robust_gate_sigma: float = 3.

    def validate(self):
        if not all(np.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError('Kinematic model settings must be finite and positive')


class KinematicProblem:
    def __init__(self, gt, xy, it, quat, gyro, training=None, config=None):
        self.config = config or KinematicConfig(); self.config.validate()
        self.gt, self.it = np.asarray(gt, float), np.asarray(it, float)
        self.xy = np.asarray(xy, float)
        self.n = len(self.gt)
        self.training = np.ones(self.n, bool) if training is None else np.asarray(training, bool)
        quat, gyro = np.asarray(quat, float), np.asarray(gyro, float)
        if (self.n < 5 or self.xy.shape != (self.n, 2) or self.training.shape != (self.n,)
                or self.training.sum() < 5 or quat.shape != (len(it), 4) or gyro.shape != (len(it), 3)
                or np.any(np.diff(self.gt) <= 0) or np.any(np.diff(self.it) <= 0)
                or self.gt[0] < self.it[0] or self.gt[-1] > self.it[-1]
                or not all(np.isfinite(x).all() for x in (self.gt, self.it, self.xy, quat, gyro))
                or np.any(np.abs(np.linalg.norm(quat, axis=1)-1) > .02)):
            raise ValueError('Invalid GNSS/IMU/holdout shape, values or coverage')
        self.measurement_indices = np.flatnonzero(self.training)
        # Only retained position values affect origin or initialization.
        self.origin = self.xy[self.measurement_indices[0]].copy()
        self.measured = self.xy[self.training]-self.origin
        rpy = Rotation.from_quat(quat).as_euler('xyz')
        if np.min(np.abs(np.cos(rpy[:, 1]))) < .2:
            raise ValueError('Euler yaw-rate conversion too close to pitch singularity')
        # Correct Euler yaw derivative, rather than assuming raw gyro Z == yaw rate.
        # Requires the gyro and quaternion to describe the same IMU axes.
        self.yaw_rate = (np.sin(rpy[:, 0])*gyro[:, 1] + np.cos(rpy[:, 0])*gyro[:, 2])/np.cos(rpy[:, 1])
        self.integral = cumulative_trapezoid(self.yaw_rate, self.it, initial=0.)
        self.gyro_angle = np.diff(np.interp(self.gt, self.it, self.integral))
        self.imu_yaw = np.interp(self.gt, self.it, np.unwrap(rpy[:, 2]))
        self.dt = np.diff(self.gt)
        self.whitening = []
        for dt in self.dt:
            q = self.config.accel_psd*np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
            self.whitening.append(np.linalg.inv(np.linalg.cholesky(q)))
        self.whitening = np.array(self.whitening)
        self.gyro_std = np.deg2rad(self.config.gyro_density_deg_sqrt_s)*np.sqrt(self.dt)
        self.bias_std = np.deg2rad(self.config.bias_rw_degps_sqrt_s)*np.sqrt(self.dt)
        duration_weights = .5*np.r_[self.dt[0], self.dt[:-1]+self.dt[1:], self.dt[-1]]
        self.lateral_weight = np.sqrt(duration_weights)/self.config.lateral_velocity_density_mps_sqrt_s
        self.initial = np.zeros((self.n, 6))
        for axis in range(2):
            self.initial[:, axis] = np.interp(self.gt, self.gt[self.training], self.measured[:, axis])
        self.initial[:, 2:4] = np.gradient(self.initial[:, :2], self.gt, axis=0)
        self.initial[:, 4] = self.imu_yaw
        # This is a seed, never another measured heading/bias factor.
        self.initial[:, 5] = np.median(self.gyro_angle/self.dt-np.diff(self.imu_yaw)/self.dt)

    def evaluate(self, vector, jacobian=False):
        state = np.asarray(vector).reshape(self.n, 6)
        residuals = []; row_indices = []; col_indices = []; values = []; offset = 0

        def add_jac(rows, cols, data):
            row_indices.extend(np.asarray(rows).ravel()); col_indices.extend(np.asarray(cols).ravel())
            values.extend(np.asarray(data).ravel())

        difference = (state[self.training, :2]-self.measured)/self.config.gnss_std_m
        norms = np.linalg.norm(difference, axis=1)
        gate = self.config.robust_gate_sigma
        scale = np.ones(len(norms)); outside = norms > gate
        scale[outside] = np.sqrt((2*gate*norms[outside]-gate**2)/norms[outside]**2)
        residuals.append((difference*scale[:, None]).ravel())
        if jacobian:
            derivative = np.zeros(len(norms))
            derivative[outside] = (-2*gate/norms[outside]**2+2*gate**2/norms[outside]**3)/(2*scale[outside]*norms[outside])
            blocks = scale[:, None, None]*np.eye(2)+derivative[:, None, None]*difference[:, :, None]*difference[:, None, :]
            for a in range(2):
                for b in range(2):
                    add_jac(2*np.arange(len(norms))+a, 6*self.measurement_indices+b, blocks[:, a, b]/self.config.gnss_std_m)
        offset += 2*len(norms)
        error = np.stack((state[1:, :2]-state[:-1, :2]-state[:-1, 2:4]*self.dt[:, None],
                          state[1:, 2:4]-state[:-1, 2:4]), axis=-1)
        residuals.append(np.einsum('nij,naj->nai', self.whitening, error).ravel())
        edges = np.arange(self.n-1)
        if jacobian:
            for axis in range(2):
                for dimension in range(2):
                    row = offset+4*edges+2*axis+dimension
                    a, b = self.whitening[:, dimension, 0], self.whitening[:, dimension, 1]
                    add_jac(row, 6*edges+axis, -a)
                    add_jac(row, 6*edges+axis+2, -a*self.dt-b)
                    add_jac(row, 6*(edges+1)+axis, a)
                    add_jac(row, 6*(edges+1)+axis+2, b)
        offset += 4*len(edges)
        residuals.append((np.diff(state[:, 4])+.5*(state[:-1, 5]+state[1:, 5])*self.dt-self.gyro_angle)/self.gyro_std)
        if jacobian:
            add_jac(offset+edges, 6*edges+4, -1/self.gyro_std)
            add_jac(offset+edges, 6*(edges+1)+4, 1/self.gyro_std)
            for side in (0, 1): add_jac(offset+edges, 6*(edges+side)+5, .5*self.dt/self.gyro_std)
        offset += len(edges)
        residuals.append(np.diff(state[:, 5])/self.bias_std)
        if jacobian:
            add_jac(offset+edges, 6*edges+5, -1/self.bias_std)
            add_jac(offset+edges, 6*(edges+1)+5, 1/self.bias_std)
        offset += len(edges)
        sine, cosine = np.sin(state[:, 4]), np.cos(state[:, 4])
        residuals.append((-sine*state[:, 2]+cosine*state[:, 3])*self.lateral_weight)
        if jacobian:
            nodes = np.arange(self.n)
            add_jac(offset+nodes, 6*nodes+2, -sine*self.lateral_weight)
            add_jac(offset+nodes, 6*nodes+3, cosine*self.lateral_weight)
            add_jac(offset+nodes, 6*nodes+4, (-cosine*state[:, 2]-sine*state[:, 3])*self.lateral_weight)
        offset += self.n
        yaw_std, bias_std = np.deg2rad(self.config.initial_yaw_std_deg), np.deg2rad(self.config.initial_rate_bias_std_degps)
        residuals.append(np.array([(state[0, 4]-self.imu_yaw[0])/yaw_std, state[0, 5]/bias_std]))
        if jacobian:
            add_jac([offset, offset+1], [4, 5], [1/yaw_std, 1/bias_std])
            return coo_matrix((values, (row_indices, col_indices)), shape=(offset+2, 6*self.n)).tocsr()
        return np.concatenate(residuals)

    def solve(self):
        result = least_squares(self.evaluate, self.initial.ravel(), jac=lambda x: self.evaluate(x, True),
            x_scale='jac', max_nfev=60, ftol=1e-6, xtol=1e-7, gtol=1e-6,
            tr_options={'maxiter': 250, 'atol': 1e-6, 'btol': 1e-6})
        state = result.x.reshape(self.n, 6).copy(); state[:, :2] += self.origin
        return state, {'success': bool(result.success), 'status': int(result.status), 'nfev': result.nfev,
            'cost': float(result.cost), 'optimality': float(result.optimality), 'message': result.message,
            'config': asdict(self.config), 'gnss_measurements_used': int(self.training.sum()),
            'heading_course_factors': 0, 'gyro_bias_range_degps': np.degrees([state[:, 5].min(), state[:, 5].max()]).tolist(),
            'limits': ['Model noise values are shared experimental assumptions, not calibrated covariance.',
                       'Soft nonholonomic motion can be biased by sideslip, unknown lever arm and forward-axis extrinsics.',
                       'Quaternion supplies roll/pitch and a weak initial yaw only, not a repeated independent heading.',
                       'No accelerometer integration or verified full 3D inertial navigation; absolute GNSS bias remains unobservable without anchors.']}
