"""8-state causal planar-heading KF: [p_xyz, v_xyz, yaw, vertical_gyro_bias].

Roll/pitch are carried from accepted 3D registration; no acceleration double
integration and no absolute IMU heading/GNSS injection.
"""
import numpy as np
from .geometry import rz, yaw, wrap
from .motion import turn_motion


class Fusion:
    def __init__(self, T, stamp, cfg):
        self.cfg = cfg; self.T = T.copy(); self.stamp = int(stamp)
        self.x = np.r_[T[:3, 3], np.zeros(3), yaw(T[:3,:3]), 0.]
        self.P = np.diag([cfg['initial_position_sigma_m']**2]*3+[25.]*3+
                         [cfg['initial_yaw_sigma_rad']**2, 0.01**2])
        self.rate = 0.; self.rate_stamp = None; self.used_imu = 0; self.stale_steps = 0

    def step(self, stamp):
        dt = (int(stamp)-self.stamp)*1e-9
        if dt < 0: raise ValueError('Filter time cannot go backwards')
        if dt == 0: return
        cfg = self.cfg
        expiry = None if self.rate_stamp is None else self.rate_stamp+round(cfg['max_imu_age_s']*1e9)
        # Do not hold an old gyro measurement beyond its configured age limit.
        if expiry is not None and self.stamp < expiry < int(stamp):
            self.step(expiry); self.step(stamp); return
        fresh = expiry is not None and self.stamp < expiry
        self.x,F,G = turn_motion(self.x,self.rate,dt,gyro_available=fresh)
        if not fresh: self.stale_steps += 1
        Q = np.zeros((8,8)); q = cfg['acceleration_noise_mps2']**2
        # Continuous white acceleration covariance; p/v cross terms are necessary.
        Q[:3,:3] = np.eye(3)*q*dt**3/3
        Q[:3,3:6] = Q[3:6,:3] = np.eye(3)*q*dt**2/2
        Q[3:6,3:6] = np.eye(3)*q*dt
        Q += np.outer(G,G)*cfg['gyro_noise_radps']**2*dt*(1 if fresh else 100)
        Q[7,7] = cfg['bias_random_walk_radps']**2*dt
        self.P = F@self.P@F.T+Q; self.stamp = int(stamp)

    def predict(self, stamp, samples):
        for sensor_ns, bag_ns, rate in samples:
            # A late measurement cannot be applied retroactively to an emitted pose.
            available_ns = max(sensor_ns, bag_ns)
            if available_ns < self.stamp: continue
            if available_ns > stamp or sensor_ns > stamp: raise ValueError('Future IMU measurement')
            self.step(available_ns); self.rate = rate; self.rate_stamp = sensor_ns; self.used_imu += 1
        self.step(stamp)
        out = self.T.copy(); out[:3, 3] = self.x[:3]
        out[:3,:3] = rz(wrap(self.x[6]-yaw(out[:3,:3]))) @ out[:3,:3]
        self.T = out
        return out.copy()

    def update(self, measurement, initialize=False):
        cfg = self.cfg
        H = np.zeros((4,8)); H[:3,:3] = np.eye(3); H[3,6] = 1
        R = np.diag([cfg['map_position_sigma_m']**2]*3+[cfg['map_yaw_sigma_rad']**2])
        innovation = np.r_[measurement[:3,3]-self.x[:3], wrap(yaw(measurement[:3,:3])-self.x[6])]
        S = H@self.P@H.T+R; nis = float(innovation@np.linalg.solve(S, innovation))
        if not np.isfinite(nis) or (not initialize and nis > cfg['nis_limit_4d']): return False, nis
        K = np.linalg.solve(S, H@self.P).T
        self.x += K@innovation; self.x[6] = wrap(self.x[6])
        A = np.eye(8)-K@H
        self.P = A@self.P@A.T+K@R@K.T  # Joseph form preserves PSD.
        self.P = (self.P+self.P.T)/2
        self.T = measurement.copy(); self.T[:3,3] = self.x[:3]
        self.T[:3,:3] = rz(wrap(self.x[6]-yaw(measurement[:3,:3])))@measurement[:3,:3]
        return True, nis
