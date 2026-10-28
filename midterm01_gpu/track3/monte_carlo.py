"""Causal CUDA local Monte Carlo search, anchored to the emitted pose.

v8 left a recursively weighted population behind the refined output. v8.1 draws
fresh current-frame candidates around deterministic gyro/PnP prediction from
that output. Scores rank local hypotheses; they are NOT a calibrated Bayesian
posterior. Registration is used once to commit the current geometric estimate,
not as an additional independent sensor measurement. An optional stationary
gyro-bias Kalman filter corrects the IMU rates, never the committed map pose.
"""
import numpy as np
import cupy as cp
from .geometry import valid_pose
from .gpu import voxel_mean
from .nearest import GpuNearest
from .gyro_bias import StationaryGyroBias


from .cuda_math import rotation_exp,create_rng


def normalize_log_weights(logw):
    """Log-sum-exp and ESS reductions on device; no uniform failure fallback."""
    if bool(cp.any(cp.isnan(logw)|cp.isposinf(logw))):
        raise ValueError('Nonfinite particle likelihood')
    if not bool(cp.any(cp.isfinite(logw))):
        raise ValueError('All particle weights are invalid')
    shifted=logw-cp.max(logw)
    normalized=shifted-cp.log(cp.sum(cp.exp(shifted)))
    weights=cp.exp(normalized)
    return normalized,weights,1/cp.sum(weights*weights)


def systematic_indices(weights,rng):
    """One random offset and a GPU prefix sum; no host-side particle loop."""
    cdf=cp.cumsum(weights)
    cdf[-1]=1.0  # Roundoff cannot create an out-of-range parent.
    positions=(cp.arange(len(weights),dtype=cp.float64)+rng.uniform())/len(weights)
    return cp.searchsorted(cdf,positions,side='right')


class MonteCarloState:
    def __init__(self,initial_pose,initial_scale,imu_cfg,cfg,scale_bounds=(.2,30.),motion_limits=None):
        self.cfg=dict(cfg);self.imu_cfg=dict(imu_cfg);self.n=int(cfg['particles'])
        self.ext=cp.asarray(imu_cfg['T_ply_imu'],dtype=cp.float64)
        if not valid_pose(np.asarray(initial_pose)) or not valid_pose(np.asarray(imu_cfg['T_ply_imu'])):
            raise ValueError('Invalid initial pose or T_ply_imu')
        self.bounds=tuple(scale_bounds)
        if len(self.bounds)!=2 or not np.isfinite(self.bounds).all() or not 0<self.bounds[0]<self.bounds[1]:
            raise ValueError('Scale bounds must be finite, positive and ordered')
        if not self.bounds[0]<initial_scale<self.bounds[1]:raise ValueError('Invalid initial scale')
        self.log_bounds=cp.log(cp.asarray(self.bounds,dtype=cp.float64))
        self.rng=create_rng(cfg['seed'])
        self.R=cp.broadcast_to(cp.asarray(initial_pose[:3,:3],dtype=cp.float64),(self.n,3,3)).copy()
        self.p=cp.broadcast_to(cp.asarray(initial_pose[:3,3],dtype=cp.float64),(self.n,3)).copy()
        self.v=cp.broadcast_to(cp.asarray(imu_cfg['initial_velocity_map_mps'],dtype=cp.float64),(self.n,3)).copy()
        self.bg=cp.broadcast_to(cp.asarray(imu_cfg['initial_gyro_bias_radps'],dtype=cp.float64),(self.n,3)).copy()
        self.logs=cp.full(self.n,cp.log(initial_scale),dtype=cp.float64)
        self.logw=cp.full(self.n,-cp.log(float(self.n)),dtype=cp.float64)
        self.T=np.asarray(initial_pose,dtype=float).copy();self.scale=float(initial_scale)
        self.stamp=None;self.held=None;self.pending=None;self.initialized=False
        self.prediction='gpu_monte_carlo';self.nn=None;self.center=None
        self.measurement=None;self.prediction_diagnostics=None;self.representative=0
        self.anchor_R=cp.asarray(initial_pose[:3,:3],dtype=cp.float64).copy()
        self.anchor_p=cp.asarray(initial_pose[:3,3],dtype=cp.float64).copy()
        self.anchor_v=cp.asarray(imu_cfg['initial_velocity_map_mps'],dtype=cp.float64).copy()
        self.bias_filter=StationaryGyroBias(imu_cfg)
        self.nominal_bias=self.bias_filter.bias
        self.registration_scale=float(initial_scale)
        self.motion_limits=dict(motion_limits or dict(max_speed_mps=50.,max_rotation_rate_radps=2.))
        for name in ('max_speed_mps','max_rotation_rate_radps'):
            if not np.isfinite(self.motion_limits[name]) or self.motion_limits[name]<=0:raise ValueError(f'Invalid {name}')
        self.max_gap=float(imu_cfg['max_sample_age_s'])
        for name in ('gyro_noise_density','gyro_bias_random_walk','max_sample_age_s'):
            if not np.isfinite(imu_cfg[name]) or imu_cfg[name]<=0:raise ValueError(f'Invalid {name}')
        for name in ('initial_velocity_map_mps','initial_gyro_bias_radps'):
            if np.shape(imu_cfg[name])!=(3,) or not np.isfinite(imu_cfg[name]).all():raise ValueError(f'Invalid {name}')

    def _noise(self,shape,sigma):
        return self.rng.normal(0.,sigma,size=shape,dtype=cp.float64)

    def _bound_scales(self):
        # Reflected log-scale diffusion, rather than clipping probability mass
        # onto an arbitrary boundary or silently dropping all escaped samples.
        lo,hi=self.log_bounds;span=hi-lo
        x=cp.mod(self.logs-lo,2*span)
        self.logs=lo+cp.where(x<=span,x,2*span-x)
        self.logs=cp.clip(self.logs,lo+1e-12,hi-1e-12)

    def _best(self,weighted=True):
        i=cp.argmax(self.logw) if weighted else self.representative
        T=cp.eye(4,dtype=cp.float64);T[:3,:3]=self.R[i];T[:3,3]=self.p[i]
        return cp.asnumpy(T),float(cp.exp(self.logs[i]))

    def _advance_rotation(self,target):
        if target==self.stamp:return
        if self.held is None or (target-self.held.timestamp_ns)*1e-9>self.max_gap:
            raise ValueError('IMU missing or stale; no silent fallback')
        dt=(target-self.stamp)*1e-9
        w=cp.asarray(self.held.angular_velocity)-self.nominal_bias
        self.anchor_R=(self.anchor_R@self.ext[:3,:3])@rotation_exp(w*dt)@self.ext[:3,:3].T
        self.stamp=target

    def predict(self,timestamp_ns,samples,*,input_scale=None,stationary=False,visual_motion=None):
        if not isinstance(timestamp_ns,(int,np.integer)):raise ValueError('Frame timestamp must be integer nanoseconds')
        timestamp_ns=int(timestamp_ns)
        if self.pending is not None:raise RuntimeError('Commit preceding frame first')
        if self.stamp is not None and timestamp_ns<=self.stamp:raise ValueError('Frame timestamps must strictly increase')
        if stationary and not self.initialized:raise ValueError('Stationary hold requires a map anchor')
        if input_scale is not None and (not np.isfinite(input_scale) or not self.bounds[0]<input_scale<self.bounds[1]):
            raise ValueError('Invalid causal input scale')
        if visual_motion is not None and not valid_pose(np.asarray(visual_motion)):raise ValueError('Invalid visual motion')
        previous=-1 if self.stamp is None else self.stamp
        latest=self.held
        # Validate time, vectors, freshness over the entire batch before mutation.
        for s in samples:
            if not isinstance(s.timestamp_ns,(int,np.integer)) or not previous<s.timestamp_ns<=timestamp_ns:
                raise ValueError('IMU must be ordered and past-only')
            if self.stamp is not None and s.timestamp_ns==self.stamp:raise ValueError('Already-consumed IMU sample')
            for v in (s.angular_velocity,s.linear_acceleration):
                if np.shape(v)!=(3,) or not np.isfinite(v).all():raise ValueError('Invalid IMU vector')
            if latest is not None and (s.timestamp_ns-latest.timestamp_ns)*1e-9>self.max_gap:
                raise ValueError('Stale IMU interval')
            latest=s;previous=s.timestamp_ns
        if latest is None or (timestamp_ns-latest.timestamp_ns)*1e-9>self.max_gap:
            raise ValueError('A recent causal IMU sample is required')
        first=self.stamp is None;old_stamp=self.stamp;old_R=self.anchor_R.copy()
        previous_output=self.T.copy();prior_scale=self.scale
        bias_variance=self.bias_filter.variance.copy()
        dt=0. if first else (timestamp_ns-old_stamp)*1e-9
        if first:
            self.held=samples[-1];self.stamp=timestamp_ns
        elif stationary:
            if samples:self.held=samples[-1]
            self.stamp=timestamp_ns;self.anchor_v.fill(0.)
        else:
            for sample in samples:
                self._advance_rotation(sample.timestamp_ns);self.held=sample
            self._advance_rotation(timestamp_ns)
            if visual_motion is not None:
                displacement=old_R@cp.asarray(visual_motion[:3,3])
                self.anchor_v=(displacement+(self.anchor_R-old_R)@self.ext[:3,3])/dt
            else:
                displacement=self.anchor_v*dt-(self.anchor_R-old_R)@self.ext[:3,3]
            self.anchor_p+=displacement
        bias_diagnostics=self.bias_filter.step(timestamp_ns,samples,stationary)
        if input_scale is not None:prior_scale=float(input_scale)
        prior_gpu=cp.eye(4);prior_gpu[:3,:3]=self.anchor_R;prior_gpu[:3,3]=self.anchor_p
        prior=cp.asnumpy(prior_gpu)
        if stationary:prior=previous_output.copy()
        self.T=prior.copy();self.scale=prior_scale;self.registration_scale=prior_scale
        self.R=cp.broadcast_to(self.anchor_R,(self.n,3,3)).copy()
        self.p=cp.broadcast_to(self.anchor_p,(self.n,3)).copy()
        self.logs=cp.full(self.n,cp.log(prior_scale),dtype=cp.float64)
        self.v=cp.broadcast_to(self.anchor_v,(self.n,3)).copy()
        self.bg=cp.broadcast_to(self.nominal_bias,(self.n,3)).copy()
        if not first and not stationary and self.initialized:
            # Noise explores hypotheses only. It cannot move the emitted prior.
            position_var=self.cfg['position_diffusion_m_sqrt_s']**2*dt
            position_var+=(self.cfg['visual_velocity_std_mps']**2*dt**2 if visual_motion is not None else
                           self.cfg['velocity_diffusion_mps_sqrt_s']**2*dt**3/3.)
            self.p+=self._noise((self.n,3),float(cp.sqrt(cp.asarray(position_var))))
            angle_var=self.imu_cfg['gyro_noise_density']**2*dt+bias_variance*dt**2+self.imu_cfg['gyro_bias_random_walk']**2*dt**3/3.
            dtheta=self._noise((self.n,3),1.)*cp.sqrt(angle_var)
            self.R=(self.R@self.ext[:3,:3])@rotation_exp(dtheta)@self.ext[:3,:3].T
            self.logs+=self._noise(self.n,self.cfg['log_scale_diffusion_sqrt_s']*dt**.5)
            self._bound_scales()
        # Include the undisturbed causal prediction as a candidate every frame.
        self.R[0]=self.anchor_R;self.p[0]=self.anchor_p;self.logs[0]=cp.log(prior_scale)
        self.logw.fill(-cp.log(float(self.n)));self.representative=0
        self.pending=dict(prior=prior.copy(),scale=prior_scale,stationary=bool(stationary),
                          previous_T=previous_output,dt=dt,logw=self.logw.copy())
        self.measurement=dict(applied=False,reason='not_evaluated')
        self.prediction_diagnostics=dict(mode='initial_guess' if first else 'stationary_visual_imu' if stationary else 'gpu_monte_carlo',
            samples=len(samples),stationary=bool(stationary),translation_model='causal_visual_displacement' if visual_motion is not None else 'last_metric_velocity',
            search_mode='fresh_local_candidates_from_committed_output',scores_are_probabilities=False,
            accelerometer_translation_used=False,kalman_used=self.bias_filter.enabled,
            gyro_bias_filter=bias_diagnostics,input_scale_applied=input_scale is not None,
            latest_bag_timestamp_ns=int(self.held.arrival_timestamp_ns if self.held.arrival_timestamp_ns is not None else self.held.timestamp_ns),
            latest_header_timestamp_ns=int(self.held.header_timestamp_ns),sample_age_seconds=(timestamp_ns-self.held.timestamp_ns)*1e-9)
        return prior.copy()

    def observe_cloud(self,points,registration):
        """One common sample/map/score for ALL hypotheses; fixed memory batches."""
        if self.pending is None:raise RuntimeError('Predict before observing cloud')
        if self.measurement['reason']!='not_evaluated':raise RuntimeError('Cloud already observed this frame')
        if not self.initialized or self.pending['stationary']:
            self.measurement=dict(applied=False,reason='bootstrap' if not self.initialized else 'stationary_hold',input_points=len(points))
            return self.T.copy()
        c=self.cfg;p=cp.asarray(points,dtype=cp.float64)
        p=p[cp.isfinite(p).all(axis=1)]
        if len(p)<30:
            self.measurement=dict(applied=False,reason='insufficient_source',input_points=len(points),points=len(p),sample_scale=self.scale);return self.T.copy()
        p,_=voxel_mean(p*self.scale,c['source_leaf_m']);p/=self.scale
        if len(p)>c['max_points']:p=p[cp.linspace(0,len(p)-1,c['max_points']).astype(cp.int64)]
        if len(p)<30:
            self.measurement=dict(applied=False,reason='insufficient_voxel_support',input_points=len(points),points=len(p),sample_scale=self.scale);return self.T.copy()
        center=self.pending['prior'][:3,3]
        if self.center is None or bool(cp.max(cp.abs(cp.asarray(center)-self.center))>registration.cfg['submap_rebuild_distance_m']):
            local=registration.map[cp.all(cp.abs(registration.map-cp.asarray(center))<cp.asarray(registration.cfg['submap_half_extent_m']),axis=1)]
            if len(local)<100:
                self.measurement=dict(applied=False,reason='empty_submap');return self.T.copy()
            self.nn=GpuNearest(local,cell=c['radius_m']);self.center=cp.asarray(center).copy()
        # Equal contribution per occupied azimuth/elevation bin limits the
        # influence of a single dense wall. All candidates share these weights.
        az=cp.floor((cp.arctan2(p[:,1],p[:,0])+cp.pi)/(2*cp.pi)*16).astype(cp.int64)
        el=cp.floor((cp.arctan2(p[:,2],cp.linalg.norm(p[:,:2],axis=1))+cp.pi/2)/cp.pi*8).astype(cp.int64)
        bins=cp.clip(az,0,15)*8+cp.clip(el,0,7)
        count=cp.bincount(bins,minlength=128);weights=1/count[bins];weights/=cp.sum(weights)
        costs=cp.empty(self.n,dtype=cp.float64);coverage=cp.empty_like(costs)
        for start in range(0,self.n,c['batch_particles']):
            end=min(start+c['batch_particles'],self.n)
            q=cp.einsum('bij,kj->bki',self.R[start:end],p)*cp.exp(self.logs[start:end,None,None])+self.p[start:end,None,:]
            ids,d2=self.nn.query(q.reshape(-1,3),c['radius_m'])
            d2=d2.reshape(end-start,len(p));good=(ids>=0).reshape(end-start,len(p))
            costs[start:end]=cp.sum(c['robust_m']**2*cp.log1p(d2/c['robust_m']**2)*weights,axis=1)
            coverage[start:end]=cp.sum(good*weights,axis=1)
        valid=coverage>=c['min_coverage']
        if not bool(cp.any(valid)):
            self.measurement=dict(applied=False,reason='all_particles_low_coverage',max_coverage=float(cp.max(coverage)),points=len(p))
            return self.T.copy()
        # Same fixed causal scale reference as the registration regularizer.
        # Randomly selected scales must not redefine the prior being optimized.
        e=self.logs-cp.log(self.pending['scale'])
        objective=costs+e*e/registration.cfg.get('log_scale_sigma',.15)**2
        score=-c['effective_points']*objective/(2*c['robust_m']**2)
        self.logw,w,ess=normalize_log_weights(cp.where(valid,score,-cp.inf))
        selected=cp.argmax(self.logw);candidate,self.registration_scale=self._best()
        self.measurement=dict(applied=True,reason='common_cloud_robust_likelihood',points=len(p),ess=float(ess),
            selected_particle=int(selected),selected_cost=float(costs[selected]),selected_objective=float(objective[selected]),selected_coverage=float(coverage[selected]),
            input_points=len(points),sample_scale=self.scale,scale_reference=self.pending['scale'],scores_are_probabilities=False,
            valid_particles=int(cp.sum(valid)),max_weight=float(cp.max(w)),effective_points=c['effective_points'],
            reductions_device='cuda',refinement_scored_as_independent_measurement=False)
        return candidate

    def _bootstrap(self,estimate):
        # Startup candidates have already been independently refined by v7's
        # geometric initializer. Record their count; no posterior is carried.
        init=estimate.get('initialization') or {}
        anchors=[row for row in init.get('refinements',[]) if row.get('eligible')]
        self.R=cp.broadcast_to(cp.asarray(estimate['T'][:3,:3]),(self.n,3,3)).copy()
        self.p=cp.broadcast_to(cp.asarray(estimate['T'][:3,3]),(self.n,3)).copy()
        self.logs=cp.full(self.n,cp.log(estimate['scale']),dtype=cp.float64)
        self.logw.fill(-cp.log(float(self.n)));self.representative=0
        return len(anchors) or 1

    def commit(self,estimate):
        if self.pending is None:raise RuntimeError('Predict before committing')
        pending=self.pending;candidate=np.asarray(estimate['T'],dtype=float);scale=float(estimate['scale'])
        valid=valid_pose(candidate) and np.isfinite(scale) and self.bounds[0]<scale<self.bounds[1]
        accepted=bool(estimate['accepted']) and valid and not pending['stationary']
        reason='stationary_visual_imu_hold' if pending['stationary'] else 'invalid_candidate_pose_or_scale' if estimate['accepted'] and not valid else estimate['reason']
        bootstrap=accepted and not self.initialized;anchor_count=None
        motion_gate=dict(checked=False,reason='bootstrap_or_hold_or_rejected')
        if accepted and not bootstrap:
            dt=pending['dt'];old=cp.asarray(pending['previous_T'])
            distance=cp.linalg.norm(cp.asarray(candidate[:3,3])-old[:3,3])
            relative=cp.asarray(candidate[:3,:3])@old[:3,:3].T
            angle=cp.arccos(cp.clip((cp.trace(relative)-1)*.5,-1.,1.))
            speed=float(distance/dt);turn=float(angle/dt)
            gate=bool(speed<=self.motion_limits['max_speed_mps'] and turn<=self.motion_limits['max_rotation_rate_radps'])
            motion_gate=dict(checked=True,accepted=gate,translation_speed_mps=speed,rotation_rate_radps=turn,
                             max_speed_mps=self.motion_limits['max_speed_mps'],max_rotation_rate_radps=self.motion_limits['max_rotation_rate_radps'],
                             source='same_physical_limits_as_visual_motion_gate')
            if not gate:accepted=False;reason='map_motion_limit'
        if accepted:
            if bootstrap:
                anchor_count=self._bootstrap(estimate)
                self.anchor_v=cp.asarray(self.imu_cfg['initial_velocity_map_mps'],dtype=cp.float64).copy()
            self.T=candidate.copy();self.scale=scale;self.initialized=True
        else:
            self.logw=pending['logw'];self.T=pending['prior'].copy();self.scale=pending['scale']
        # Commit one coherent state, including after rejection and stationary hold.
        self.anchor_R=cp.asarray(self.T[:3,:3]).copy();self.anchor_p=cp.asarray(self.T[:3,3]).copy()
        self.registration_scale=self.scale
        d=None
        if valid:
            # Diagnostics use device reductions too; no CPU rotation-vector sum.
            relative=cp.asarray(candidate[:3,:3])@cp.asarray(pending['prior'][:3,:3]).T
            angle=cp.arccos(cp.clip((cp.trace(relative)-1)*.5,-1.,1.))
            d=dict(translation_m=cp.asnumpy(cp.asarray(candidate[:3,3])-cp.asarray(pending['prior'][:3,3])).tolist(),
                   rotation_angle_rad=float(angle),log_scale=float(cp.log(scale/pending['scale'])))
        cloud=dict(self.measurement,committed=bool(accepted and self.measurement['applied']),
                   rolled_back=bool(not accepted and self.measurement['applied']))
        update=dict(bootstrap=bool(bootstrap),measurement_accepted=bool(accepted),kalman_used=False,
            particles=self.n,cloud=cloud,resampled=False,unique_ancestors=None,
            gyro_bias_kalman_used=self.bias_filter.enabled,pose_kalman_used=False,
            search_mode='causal_local_monte_carlo_search',scores_are_probabilities=False,
            next_prediction_matches_output=True,motion_gate=motion_gate,
            bootstrap_anchors=anchor_count,posterior_recentered_by_refinement=False)
        self.pending=None
        return dict(accepted=bool(accepted),reason=reason,prediction=self.prediction_diagnostics['mode'],
            decision='initialized_from_registration' if bootstrap else 'monte_carlo_refined_registration' if accepted else 'stationary_pose_hold' if pending['stationary'] else 'retained_causal_prediction',
            delta=d,inertial=dict(prediction=self.prediction_diagnostics,update=update),monte_carlo=update)
