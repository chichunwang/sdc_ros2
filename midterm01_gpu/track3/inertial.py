"""v5 global/left error-state EKF, with an IMU-origin nominal state.

Error order: [dp_map,dv_map,dtheta_map,dba_imu,dbg_imu].
T_ply_imu maps metric IMU coordinates into metric camera-FLU coordinates.
Scale belongs to the point cloud registration and is never applied to the IMU.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import chi2
from .geometry import valid_pose
from .motion_parameters import VISUAL_VELOCITY_STD_MPS, STATIONARY_VELOCITY_STD_MPS


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def left_jacobian(v):
    a = np.linalg.norm(v)
    W = skew(v)
    if a < 1e-6:
        return np.eye(3) + .5*W + W@W/6
    return np.eye(3) + (1-np.cos(a))/a**2*W + (a-np.sin(a))/a**3*(W@W)


class InertialState:
    def __init__(self, initial_pose, initial_scale, cfg, scale_bounds=(.2, 30.)):
        self.extrinsic = np.asarray(cfg['T_ply_imu'], dtype=float)
        if not valid_pose(np.asarray(initial_pose)) or not valid_pose(self.extrinsic):
            raise ValueError('Invalid initial pose or T_ply_imu')
        if not cfg.get('extrinsic_status'):
            raise ValueError('Record calibrated or assumed extrinsic_status explicitly')
        self.inverse_extrinsic = np.linalg.inv(self.extrinsic)
        self.gravity = np.asarray(cfg['gravity_map_mps2'], dtype=float)
        self.v = np.asarray(cfg['initial_velocity_map_mps'], dtype=float).copy()
        self.ba = np.asarray(cfg['initial_accel_bias_mps2'], dtype=float).copy()
        self.bg = np.asarray(cfg['initial_gyro_bias_radps'], dtype=float).copy()
        for value in (self.gravity, self.v, self.ba, self.bg):
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError('Inertial vectors must be finite 3-vectors')
        self.X = initial_pose @ self.extrinsic
        self.bounds = tuple(scale_bounds)
        self.scale = float(initial_scale)
        if not self.bounds[0] < self.scale < self.bounds[1]:
            raise ValueError('Initial scale outside registration bounds')
        sigmas = np.asarray(cfg['initial_std'], dtype=float)
        if sigmas.shape != (15,) or not np.isfinite(sigmas).all() or np.any(sigmas <= 0):
            raise ValueError('initial_std requires 15 positive standard deviations')
        self.P = np.diag(sigmas**2)
        self.initial_P = self.P.copy()
        self.initial_v = self.v.copy()
        self.initial_ba = self.ba.copy()
        self.initial_bg = self.bg.copy()
        self.measurement_std = np.asarray(cfg['registration_std'], dtype=float)
        if self.measurement_std.shape != (6,) or not np.isfinite(self.measurement_std).all() or np.any(self.measurement_std <= 0):
            raise ValueError('registration_std requires 6 positive standard deviations')
        noise = np.asarray([cfg[k] for k in ('accel_noise_density', 'gyro_noise_density',
                                           'accel_bias_random_walk', 'gyro_bias_random_walk')])
        if not np.isfinite(noise).all() or np.any(noise <= 0):
            raise ValueError('IMU continuous noise densities must be positive')
        self.Qc = np.diag(np.repeat(noise**2, 3))
        self.max_step = float(cfg['max_integration_step_s'])
        self.max_gap = float(cfg['max_sample_age_s'])
        probability = float(cfg['nis_probability'])
        if not (0 < self.max_step <= self.max_gap and np.isfinite(self.max_gap) and 0 < probability < 1):
            raise ValueError('Invalid integration step, sample age or NIS probability')
        self.nis_limit = float(chi2.ppf(probability, 6))
        self.stamp = None
        self.held = None
        self.pending = None
        self.initialized = False
        self.prediction = 'imu_eskf'
        self.prediction_diagnostics = None
        self.pending_stationary = False

    def _stationary_bias_update(self,samples,dt):
        """Schmidt update: retain uncertain anchor pose, learn sensor biases.

        Stationarity is independently supported by image tracks AND raw gyro.
        Held pose does not become ground truth; its covariance remains nonzero.
        """
        samples=samples or [self.held]
        acceleration=np.mean([s.linear_acceleration for s in samples],axis=0)
        angular=np.mean([s.angular_velocity for s in samples],axis=0)
        R=self.X[:3,:3]
        r=np.r_[acceleration+R.T@self.gravity-self.ba,angular-self.bg]
        H=np.zeros((6,15));H[:3,6:9]=-R.T@skew(self.gravity)
        H[:3,9:12]=H[3:,12:15]=np.eye(3)
        V=np.diag(np.r_[np.diag(self.Qc)[:3],np.diag(self.Qc)[3:6]])/max(dt,1e-6)
        S=H@self.P@H.T+V
        K=np.linalg.solve(S,H@self.P).T
        K[:9,:]=0.  # Pose/velocity constrained by zero motion, not bias measurement.
        dx=K@r;self.ba+=dx[9:12];self.bg+=dx[12:15]
        A=np.eye(15)-K@H
        self.P=A@self.P@A.T+K@V@K.T
        # Zero-velocity projection with finite uncertainty, not an arbitrary speed.
        self.v[:]=0.;self.P[3:6,:]=0.;self.P[:,3:6]=0.
        self.P[3:6,3:6]=np.eye(3)*STATIONARY_VELOCITY_STD_MPS**2
        self.P=(self.P+self.P.T)*.5
        return dict(residual=r.tolist(),velocity_std_mps=STATIONARY_VELOCITY_STD_MPS,
                    pose_policy='retain_previous_pose_and_nonzero_anchor_uncertainty')

    def _visual_velocity_update(self,motion,previous_X,dt):
        """Short-interval average velocity, using past metric depths/current image.

        1 m/s is an explicit engineering std (not sensor calibration). This loose
        constraint reuses previous orientation/scale and is not an independent GT.
        """
        if not valid_pose(np.asarray(motion)):raise ValueError('Invalid visual motion')
        # A relative visual displacement is a BODY-frame observation. Rotating it
        # with yesterday's estimated heading and treating that world vector as an
        # independent observation would feed the filter's own heading back to it.
        Rrel=motion[:3,:3];lever=self.extrinsic[:3,3]
        measured=self.extrinsic[:3,:3].T@Rrel.T@(motion[:3,3]+(Rrel-np.eye(3))@lever)/dt
        R=self.X[:3,:3]
        predicted=R.T@self.v
        H=np.zeros((3,15));H[:,3:6]=R.T;H[:,6:9]=R.T@skew(self.v)
        V=np.eye(3)*VISUAL_VELOCITY_STD_MPS**2
        r=measured-predicted;S=H@self.P@H.T+V
        nis=float(r@np.linalg.solve(S,r));limit=float(chi2.ppf(.999,3))
        accepted=bool(np.isfinite(nis) and nis<=limit)
        if accepted:
            K=np.linalg.solve(S,H@self.P).T;dx=K@r
            self.X[:3,3]+=dx[:3];self.v+=dx[3:6]
            self.X[:3,:3]=Rotation.from_rotvec(dx[6:9]).as_matrix()@self.X[:3,:3]
            self.ba+=dx[9:12];self.bg+=dx[12:15]
            A=np.eye(15)-K@H;self.P=A@self.P@A.T+K@V@K.T
            reset=np.eye(15);reset[6:9,6:9]=left_jacobian(dx[6:9])
            self.P=reset@self.P@reset.T;self.P=(self.P+self.P.T)*.5
        return dict(accepted=accepted,nis=nis,nis_limit=limit,velocity_map_mps=(R@measured).tolist(),
                    velocity_body_mps=measured.tolist(),
                    measurement_std_mps=VISUAL_VELOCITY_STD_MPS,model='current-IMU-frame interval-average velocity; endpoint approximation')

    @property
    def T(self):
        return self.X @ self.inverse_extrinsic

    def _step(self, sample, dt):
        a = sample.linear_acceleration - self.ba
        w = sample.angular_velocity - self.bg
        R = self.X[:3, :3]
        mid = R @ Rotation.from_rotvec(w*(.5*dt)).as_matrix()
        force = mid @ a
        acceleration = force + self.gravity
        self.X[:3, 3] += self.v*dt + .5*acceleration*dt**2
        self.v += acceleration*dt
        self.X[:3, :3] = R @ Rotation.from_rotvec(w*dt).as_matrix()
        F = np.zeros((15, 15))
        F[:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -skew(force)
        F[3:6, 9:12] = -mid
        F[6:9, 12:15] = -mid
        G = np.zeros((15, 12))
        G[3:6, :3] = mid
        G[6:9, 3:6] = mid
        G[9:12, 6:9] = G[12:15, 9:12] = np.eye(3)
        def transition(t):
            return np.eye(15) + F*t + .5*(F@F)*t*t
        Phi = transition(dt)
        # Positive Gauss quadrature of Phi(s) G Qc G' Phi(s)' includes p-v noise.
        drive = G @ self.Qc @ G.T
        Q = np.zeros((15, 15))
        for point, weight in ((.5-np.sqrt(15)/10, 5/18), (.5, 4/9), (.5+np.sqrt(15)/10, 5/18)):
            A = transition(point*dt)
            Q += weight*dt*(A @ drive @ A.T)
        self.P = Phi @ self.P @ Phi.T + Q
        self.P = (self.P+self.P.T)*.5

    def _advance(self, target):
        if target == self.stamp:
            return
        if self.held is None or (target-self.held.timestamp_ns)*1e-9 > self.max_gap:
            raise ValueError('IMU missing or stale; prediction stopped (no silent fallback)')
        dt = (target-self.stamp)*1e-9
        steps = max(1, int(np.ceil(dt/self.max_step)))
        for _ in range(steps):
            self._step(self.held, dt/steps)
        self.stamp = target

    def predict(self, timestamp_ns, samples, *, input_scale=None, stationary=False, visual_motion=None):
        if not isinstance(timestamp_ns, (int, np.integer)):
            raise ValueError('Frame timestamp must be integer nanoseconds')
        timestamp_ns = int(timestamp_ns)
        if input_scale is not None and (not np.isfinite(input_scale) or not self.bounds[0]<input_scale<self.bounds[1]):
            raise ValueError('Invalid causal input scale')
        if stationary and not self.initialized:
            raise ValueError('Stationary constraint requires an accepted map anchor')
        if self.pending is not None:
            raise RuntimeError('Commit the preceding frame before predicting again')
        if self.stamp is not None and timestamp_ns <= self.stamp:
            raise ValueError('Frame timestamps must strictly increase')
        # Validate the entire batch before changing the state.
        previous = self.stamp if self.stamp is not None else -1
        for s in samples:
            if (not isinstance(s.timestamp_ns, (int, np.integer)) or
                    not previous <= s.timestamp_ns <= timestamp_ns):
                raise ValueError('IMU samples must be arrival-ordered, past-only and new')
            if self.stamp is not None and s.timestamp_ns == self.stamp:
                raise ValueError('IMU sample already consumed at preceding frame')
            for v in (s.angular_velocity, s.linear_acceleration):
                if np.shape(v) != (3,) or not np.isfinite(v).all():
                    raise ValueError('Invalid IMU vector')
            previous = s.timestamp_ns
        first = self.stamp is None
        previous_X=self.X.copy();previous_stamp=self.stamp
        if first:
            if not samples or (timestamp_ns-samples[-1].timestamp_ns)*1e-9 > self.max_gap:
                raise ValueError('A recent causal IMU sample is required at the first frame')
            # Initial guess is an anchor at the first PLY time, not at bag start.
            self.held = samples[-1]
            self.stamp = timestamp_ns
        elif stationary:
            dt=(timestamp_ns-self.stamp)*1e-9
            for s in samples:
                if self.held is None or (s.timestamp_ns-self.held.timestamp_ns)*1e-9>self.max_gap:
                    raise ValueError('IMU missing or stale during stationary hold')
                self.held=s
            if self.held is None or (timestamp_ns-self.held.timestamp_ns)*1e-9>self.max_gap:
                raise ValueError('IMU missing or stale during stationary hold')
            self.P[9:12,9:12]+=self.Qc[6:9,6:9]*dt
            self.P[12:15,12:15]+=self.Qc[9:12,9:12]*dt
            zero_motion=self._stationary_bias_update(samples,dt)
            self.stamp=timestamp_ns
        else:
            for s in samples:
                self._advance(s.timestamp_ns)
                self.held = s
            self._advance(timestamp_ns)
        visual_velocity=None
        if visual_motion is not None and not first and not stationary:
            visual_velocity=self._visual_velocity_update(visual_motion,previous_X,(timestamp_ns-previous_stamp)*1e-9)
        prior = self.T.copy()
        if not valid_pose(prior) or not np.isfinite(self.P).all():
            raise ValueError('Invalid inertial prediction')
        if input_scale is not None:self.scale=float(input_scale)
        self.pending_stationary=bool(stationary)
        self.pending = (timestamp_ns, prior.copy(), self.scale)
        self.prediction_diagnostics = dict(
            mode='initial_guess' if first else 'stationary_visual_imu' if stationary else 'imu_eskf', samples=len(samples),
            stationary=bool(stationary),zero_motion_update=zero_motion if stationary else None,
            visual_velocity_update=visual_velocity,
            input_scale_applied=input_scale is not None,
            integration_clock='imu_header_ns_zero_order_hold' if self.held.arrival_timestamp_ns is not None else 'bag_arrival_ns_zero_order_hold',
            latest_bag_timestamp_ns=int(self.held.arrival_timestamp_ns if self.held.arrival_timestamp_ns is not None else self.held.timestamp_ns),
            latest_header_timestamp_ns=int(self.held.header_timestamp_ns),
            sample_age_seconds=(timestamp_ns-self.held.timestamp_ns)*1e-9,
            first_message_id=None if not samples else samples[0].message_id,
            last_message_id=None if not samples else samples[-1].message_id,
            predicted_velocity_map_mps=self.v.tolist())
        return prior

    def _measurement(self, T):
        z = T @ self.extrinsic
        r = np.r_[z[:3, 3]-self.X[:3, 3],
                  Rotation.from_matrix(z[:3, :3] @ self.X[:3, :3].T).as_rotvec()]
        H = np.zeros((6, 15))
        H[:3, :3] = H[3:, 6:9] = np.eye(3)
        # Registration uncertainty is specified at the PLY origin in map axes.
        J = np.eye(6)
        J[:3, 3:] = -skew(T[:3, :3] @ self.extrinsic[:3, 3])
        V = J @ np.diag(self.measurement_std**2) @ J.T
        return z, r, H, V

    def commit(self, estimate):
        if self.pending is None:
            raise RuntimeError('Predict before committing a measurement')
        stamp, prior, prior_scale = self.pending
        candidate = np.asarray(estimate['T'], dtype=float)
        scale = float(estimate['scale'])
        accepted = bool(estimate['accepted'])
        reason = estimate['reason']
        if self.pending_stationary:
            accepted=False;reason='stationary_visual_imu_hold'
        valid = valid_pose(candidate) and np.isfinite(scale) and self.bounds[0] < scale < self.bounds[1]
        if accepted and not valid:
            accepted, reason = False, 'invalid_candidate_pose_or_scale'
        delta = None
        if valid:
            delta = dict(translation_m=(candidate[:3, 3]-prior[:3, 3]).tolist(),
                         rotation_rad=Rotation.from_matrix(candidate[:3, :3] @ prior[:3, :3].T).as_rotvec().tolist(),
                         log_scale=float(np.log(scale/prior_scale)))
        nis = None
        innovation = None
        bootstrap = False
        if valid:
            z, r, H, V = self._measurement(candidate)
            innovation = r.tolist()
        if accepted:
            if not self.initialized:
                # The noisy initial guess must not veto the first geometric anchor.
                # Reset pose covariance/cross terms; do not infer velocity from a jump.
                self.X = z.copy()
                self.v = self.initial_v.copy()
                self.ba = self.initial_ba.copy()
                self.bg = self.initial_bg.copy()
                self.P = self.initial_P.copy()
                pose_ids = [0, 1, 2, 6, 7, 8]
                self.P[pose_ids, :] = 0.
                self.P[:, pose_ids] = 0.
                self.P[np.ix_(pose_ids, pose_ids)] = V
                bootstrap = True
            else:
                S = H @ self.P @ H.T + V
                nis = float(r @ np.linalg.solve(S, r))
                if not np.isfinite(nis) or nis > self.nis_limit:
                    accepted, reason = False, 'kalman_nis_gate'
                else:
                    K = np.linalg.solve(S, H @ self.P).T
                    dx = K @ r
                    self.X[:3, 3] += dx[:3]
                    self.v += dx[3:6]
                    self.X[:3, :3] = Rotation.from_rotvec(dx[6:9]).as_matrix() @ self.X[:3, :3]
                    self.ba += dx[9:12]
                    self.bg += dx[12:15]
                    A = np.eye(15)-K@H
                    self.P = A @ self.P @ A.T + K @ V @ K.T  # Joseph form
                    reset = np.eye(15)
                    reset[6:9, 6:9] = left_jacobian(dx[6:9])
                    self.P = reset @ self.P @ reset.T
                    self.P = (self.P+self.P.T)*.5
            if accepted:
                self.scale = scale
                self.initialized = True
        self.pending = None
        diagnostics = dict(nis=nis, nis_limit=self.nis_limit, innovation=innovation,
                           bootstrap=bootstrap, measurement_accepted=accepted,
                           velocity_map_mps=self.v.tolist(), accel_bias_mps2=self.ba.tolist(),
                           gyro_bias_radps=self.bg.tolist(), covariance_diagonal=np.diag(self.P).tolist(),
                           measurement_std=self.measurement_std.tolist())
        return dict(accepted=accepted, reason=reason, prediction=self.prediction_diagnostics['mode'],
                    decision=('initialized_from_registration' if bootstrap else 'kalman_measurement_update') if accepted else
                    'stationary_pose_hold' if self.pending_stationary else 'retained_imu_prediction',
                    delta=delta, inertial=dict(prediction=self.prediction_diagnostics, update=diagnostics))
