"""因果 XY coordinated turn（協調轉彎）：只用相對轉速，不強迫速度朝向。

4×4 狀態為 map [x,y,vx,vy]。F 與等向白加速度的 Q 由連續模型積分，
小角度用 Taylor 展開避免 cancellation；點雲及配準仍在 CUDA。
"""
import numpy as np
J=np.array([[0.,-1.],[1.,0.]])


def transition(dt,rate,accel_psd):
    """Exact constant-rate F/Q; zero rate reproduces the original CV model."""
    if not np.isfinite([dt,rate,accel_psd]).all() or dt<0 or accel_psd<0:raise ValueError('Invalid motion interval/noise')
    t=rate*dt;I=np.eye(2);F=np.eye(4)
    if rate==0.:
        F[:2,2:]=I*dt
        Q=accel_psd*np.block([[I*dt**3/3,I*dt**2/2],[I*dt**2/2,I*dt]])
        return F,Q
    R=np.cos(t)*I+np.sin(t)*J
    if abs(t)<1e-3:
        a=1-t*t/6+t**4/120-t**6/5040;b=t/2-t**3/24+t**5/720
        xx=1/3-t*t/60+t**4/2520;cross_i=.5-t*t/24+t**4/720;cross_j=t/6-t**3/120+t**5/5040
    else:
        a=np.sin(t)/t;b=(1-np.cos(t))/t;xx=2*(t-np.sin(t))/t**3;cross_i=(1-np.cos(t))/t**2;cross_j=(t-np.sin(t))/t**2
    F[:2,2:]=dt*(a*I+b*J);F[2:,2:]=R
    X=dt**2*(cross_i*I-cross_j*J);Q=accel_psd*np.block([[I*dt**3*xx,X],[X.T,I*dt]])
    return F,(Q+Q.T)*.5


def rate_sensitivity(velocity,dt,rate):
    """Derivative of propagated [position,velocity] with respect to yaw rate."""
    t=dt*rate;I=np.eye(2)
    if abs(t)<1e-3:
        da=-t/3+t**3/30-t**5/840;db=.5-t*t/8+t**4/144
    else:da=(t*np.cos(t)-np.sin(t))/t**2;db=(t*np.sin(t)-(1-np.cos(t)))/t**2
    R=np.cos(t)*I+np.sin(t)*J
    return np.r_[dt**2*(da*I+db*J)@velocity,dt*J@R@velocity]


def yaw_rate(orientation,gyro,gyro_covariance,orientation_covariance):
    """Body gyro -> Euler yaw rate; absolute yaw cancels from this mapping.

    Attitude covariance uses the existing body small-angle assumption. Rate
    uncertainty propagation is local, not a calibrated joint IMU estimator.
    """
    roll,pitch,_=orientation.as_euler('xyz');c=np.cos(pitch)
    if abs(c)<1e-3:raise ValueError('Coordinated-turn yaw rate near Euler singularity')
    s=np.sin(roll);r=np.cos(roll);g=np.array([0.,s/c,r/c]);dg=np.array([[0.,r/c,-s/c],[0.,s*np.sin(pitch)/c**2,r*np.sin(pitch)/c**2]])@gyro
    variance=float(g@gyro_covariance@g+dg@orientation_covariance[:2,:2]@dg)
    return float(g@gyro),max(0.,variance)


class StationaryGyroBias:
    """Only disjoint image/GNSS-confirmed rest intervals teach gyro zero bias.

    This calibration is used by XY motion prediction. It never rotates an
    emitted pose or modifies the existing absolute-heading calibration.
    """
    def __init__(self,prior_std,rest_std,rw_std,retention_seconds,nis_soft,nis_reject):
        prior_std=np.asarray(prior_std,dtype=float)
        if prior_std.shape!=(3,) or not np.isfinite(prior_std).all() or np.any(prior_std<0):
            raise ValueError('Invalid gyro bias prior')
        if not np.isfinite([rest_std,rw_std,retention_seconds,nis_soft,nis_reject]).all() or min(rest_std,retention_seconds,nis_soft)<=0 or rw_std<0 or nis_reject<nis_soft:
            raise ValueError('Invalid gyro bias model')
        from collections import deque
        self.mean=np.zeros(3);self.cov=np.diag(prior_std**2)
        self.rest_std=rest_std;self.rw_std=rw_std
        self.retention_ns=int(retention_seconds*1e9)
        self.nis_soft=nis_soft;self.nis_reject=nis_reject
        self.buffer=deque();self.last_sample=None;self.last_end=None
        self.updates=0;self.integrated_turn=0.;self.frame_turn=0.

    def predict(self,dt):
        if not np.isfinite(dt) or dt<0:raise ValueError('Invalid gyro bias interval')
        self.cov+=np.eye(3)*self.rw_std**2*dt

    def observe(self,stamp,gyro,current_time):
        if type(stamp) is not int or stamp>current_time or (self.last_sample is not None and stamp<=self.last_sample):
            raise ValueError('Gyro sample must be new and no later than current time')
        value=np.asarray(gyro,dtype=float)
        if value.shape!=(3,) or not np.isfinite(value).all():raise ValueError('Invalid gyro sample')
        self.last_sample=stamp;self.buffer.append((stamp,value.copy()))
        while self.buffer and self.buffer[0][0]<stamp-self.retention_ns:self.buffer.popleft()

    def update(self,start,end,current_time):
        if type(start) is not int or type(end) is not int or end!=current_time or start>=end:
            raise ValueError('Stationary interval must end at current time')
        if self.last_end is not None and start<self.last_end:
            raise ValueError('Stationary gyro intervals overlap')
        samples=np.array([g for stamp,g in self.buffer if start<stamp<=end])
        if len(samples)<2:return dict(accepted=False,reason='insufficient_arrived_gyro_samples')
        z=samples.mean(0);R=np.cov(samples.T)/len(samples)+np.eye(3)*self.rest_std**2
        C=self.cov;residual=z-self.mean;nis=float(residual@np.linalg.solve(C+R,residual))
        self.last_end=end
        if nis>self.nis_reject:return dict(accepted=False,reason='gyro_rest_innovation_gate',nis=nis)
        R*=max(1.,nis/self.nis_soft);K=np.linalg.solve(C+R,C).T
        self.mean+=K@residual;A=np.eye(3)-K
        self.cov=A@C@A.T+K@R@K.T;self.cov=(self.cov+self.cov.T)*.5
        self.updates+=1
        return dict(accepted=True,samples=len(samples),start_timestamp_ns=start,end_timestamp_ns=end,
            nis=nis,mean_radps=self.mean.tolist(),covariance_rad2ps2=self.cov.tolist(),
            scope='rest bias for subsequent XY motion prediction only; no past pose or absolute heading correction')

    def frame_diagnostic(self):
        result=dict(model='arrived_gyro_coordinated_turn',gyro_bias_mean_radps=self.mean.tolist(),
            gyro_bias_covariance_rad2ps2=self.cov.tolist(),rest_bias_updates=self.updates,
            signed_turn_since_frame_rad=float(self.frame_turn),integrated_turn_rad=float(self.integrated_turn),
            uncertainty='exact white-acceleration integral; local per-interval gyro uncertainty; engineering budgets, not calibrated truth')
        self.frame_turn=0.
        return result
