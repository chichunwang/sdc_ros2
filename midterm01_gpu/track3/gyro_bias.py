"""Causal three-axis gyro-bias Kalman filter; every reduction stays on CUDA.

Only a visually confirmed stationary interval supplies the observation
omega_measured = bias + noise. Moving gyro rates are never low-pass filtered.
This filter does not observe map registration, scale, acceleration or pose.
"""
import numpy as np
import cupy as cp
from scipy.stats import chi2


class StationaryGyroBias:
    def __init__(self, imu_cfg):
        cfg=imu_cfg.get('gyro_bias_filter',dict(enabled=False))
        self.enabled=cfg['enabled']
        if not isinstance(self.enabled,bool):raise ValueError('gyro_bias_filter.enabled must be bool')
        self.bias=cp.asarray(imu_cfg['initial_gyro_bias_radps'],dtype=cp.float64).copy()
        std=cp.asarray(imu_cfg['initial_std'][12:15],dtype=cp.float64)
        if self.bias.shape!=(3,) or std.shape!=(3,) or not bool(cp.all(cp.isfinite(self.bias)&cp.isfinite(std)&(std>0))):
            raise ValueError('Invalid initial gyro bias/std')
        self.variance=std**2
        self.noise=float(imu_cfg['gyro_noise_density'])
        self.walk=float(imu_cfg['gyro_bias_random_walk'])
        self.max_rate=float(cfg.get('max_raw_gyro_radps',.02))
        probability=float(imu_cfg['nis_probability'])
        if not all(np.isfinite(v) and v>0 for v in (self.noise,self.walk,self.max_rate)) or not 0<probability<1:
            raise ValueError('Invalid gyro-bias noise/gate')
        self.threshold=float(chi2.ppf(probability,3))
        self.stamp=None;self.last_sample=None;self.updates=0

    def step(self,timestamp_ns,samples,stationary):
        if not isinstance(timestamp_ns,(int,np.integer)):
            raise ValueError('Gyro-bias timestamp must be integer nanoseconds')
        timestamp_ns=int(timestamp_ns)
        if self.stamp is not None and timestamp_ns<=self.stamp:
            raise ValueError('Gyro-bias clock must strictly increase')
        previous_sample=self.last_sample;durations=[];rates=[]
        # Validate the entire batch before touching bias, covariance or clocks.
        for sample in samples:
            t=sample.timestamp_ns
            if not isinstance(t,(int,np.integer)) or t>timestamp_ns or (previous_sample is not None and t<=previous_sample):
                raise ValueError('Gyro-bias samples must be unique, ordered and causal')
            if self.stamp is not None and t<=self.stamp:
                raise ValueError('Gyro-bias sample already consumed')
            v=np.asarray(sample.angular_velocity)
            if v.shape!=(3,) or not np.isfinite(v).all():raise ValueError('Invalid gyro-bias observation')
            if self.stamp is not None and previous_sample is not None:
                # Restrict evidence to this image interval. A held sample is
                # never re-counted when the following frame has no new IMU.
                durations.append((t-max(previous_sample,self.stamp))*1e-9)
                rates.append(v)
            previous_sample=int(t)
        first=self.stamp is None
        dt=0. if first else (timestamp_ns-self.stamp)*1e-9
        if self.enabled:self.variance+=self.walk**2*dt
        self.stamp=timestamp_ns;self.last_sample=previous_sample
        reason='disabled' if not self.enabled else 'initial_frame' if first else 'not_stationary' if not stationary else 'no_new_imu'
        diagnostic=dict(enabled=self.enabled,updated=False,reason=reason,samples=len(samples),
            scope='gyro_bias_only',state_frame='imu_link',device='cuda',map_measurement_used=False,
            nis=None,nis_threshold=self.threshold,support_seconds=0.)
        if self.enabled and stationary and durations:
            rates=cp.asarray(np.stack(rates),dtype=cp.float64)
            weights=cp.asarray(durations,dtype=cp.float64)
            support=cp.sum(weights)
            diagnostic['support_seconds']=float(support)
            if bool(cp.max(cp.linalg.norm(rates,axis=1))>=self.max_rate):
                diagnostic['reason']='raw_gyro_motion_gate'
            else:
                mean=cp.sum(weights[:,None]*rates,axis=0)/support
                # Continuous white-noise density -> variance of an interval
                # mean. Clipped partial sampling intervals make this conservative.
                observation_variance=self.noise**2/support
                innovation=mean-self.bias
                S=self.variance+observation_variance
                nis=cp.sum(innovation**2/S)
                diagnostic.update(nis=float(nis),mean_gyro_radps=cp.asnumpy(mean).tolist(),
                                  observation_variance_radps2=float(observation_variance))
                if bool(nis>self.threshold):diagnostic['reason']='innovation_gate'
                else:
                    K=self.variance/S
                    self.bias+=K*innovation
                    # Joseph form for three independent scalar filters.
                    self.variance=(1-K)**2*self.variance+K**2*observation_variance
                    self.updates+=1
                    diagnostic.update(updated=True,reason='stationary_zero_rate',gain=cp.asnumpy(K).tolist())
        if not bool(cp.all(cp.isfinite(self.bias)&cp.isfinite(self.variance)&(self.variance>0))):
            raise FloatingPointError('Nonfinite gyro-bias state/covariance')
        diagnostic.update(bias_radps=cp.asnumpy(self.bias).tolist(),
                          bias_std_radps=cp.asnumpy(cp.sqrt(self.variance)).tolist(),updates=self.updates)
        return diagnostic
