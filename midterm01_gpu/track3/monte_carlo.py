"""Causal FP64 CUDA particle localization, without a Kalman filter.

Particle state: camera-FLU pose, metric map velocity, gyro bias and log scale.
Gyro drives orientation; causal PnP displacement drives translation when valid.
Without PnP, velocity follows a random walk. Accelerometer translation is
explicitly unused: uncalibrated gravity/extrinsics would accumulate drift.

Cloud likelihood is a tempered robust geometric score, not calibrated sensor
probability. Startup uses a data-derived mixture of eligible v7 anchors. Local
NDT/ICP refinement is an output point estimate: it is NEVER scored again as an
independent observation, nor used to collapse/recenter the particle posterior.
"""
import numpy as np
import cupy as cp
from .geometry import valid_pose
from .gpu import voxel_mean
from .nearest import GpuNearest


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
    def __init__(self,initial_pose,initial_scale,imu_cfg,cfg,scale_bounds=(.2,30.)):
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
        self.R=cp.broadcast_to(cp.asarray(initial_pose[:3,:3]),(self.n,3,3)).copy()
        self.p=cp.broadcast_to(cp.asarray(initial_pose[:3,3]),(self.n,3)).copy()
        self.v=cp.broadcast_to(cp.asarray(imu_cfg['initial_velocity_map_mps'],dtype=cp.float64),(self.n,3)).copy()
        self.bg=cp.broadcast_to(cp.asarray(imu_cfg['initial_gyro_bias_radps'],dtype=cp.float64),(self.n,3)).copy()
        self.logs=cp.full(self.n,cp.log(initial_scale),dtype=cp.float64)
        self.logw=cp.full(self.n,-cp.log(float(self.n)),dtype=cp.float64)
        self.T=np.asarray(initial_pose,dtype=float).copy();self.scale=float(initial_scale)
        self.stamp=None;self.held=None;self.pending=None;self.initialized=False
        self.prediction='gpu_monte_carlo';self.nn=None;self.center=None
        self.measurement=None;self.prediction_diagnostics=None;self.representative=0
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
        w=cp.asarray(self.held.angular_velocity)-self.bg
        omega=w*dt+self._noise((self.n,3),self.imu_cfg['gyro_noise_density']*dt**.5)
        self.R=(self.R@self.ext[:3,:3])@rotation_exp(omega)@self.ext[:3,:3].T
        self.bg+=self._noise((self.n,3),self.imu_cfg['gyro_bias_random_walk']*dt**.5)
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
            if not isinstance(s.timestamp_ns,(int,np.integer)) or not previous<=s.timestamp_ns<=timestamp_ns:
                raise ValueError('IMU must be ordered and past-only')
            if self.stamp is not None and s.timestamp_ns==self.stamp:raise ValueError('Already-consumed IMU sample')
            for v in (s.angular_velocity,s.linear_acceleration):
                if np.shape(v)!=(3,) or not np.isfinite(v).all():raise ValueError('Invalid IMU vector')
            if latest is not None and (s.timestamp_ns-latest.timestamp_ns)*1e-9>self.max_gap:
                raise ValueError('Stale IMU interval')
            latest=s;previous=s.timestamp_ns
        if latest is None or (timestamp_ns-latest.timestamp_ns)*1e-9>self.max_gap:
            raise ValueError('A recent causal IMU sample is required')
        first=self.stamp is None;old_stamp=self.stamp;old_R=self.R.copy()
        previous_output=self.T.copy()
        if first:
            self.held=samples[-1];self.stamp=timestamp_ns
        elif stationary:
            # Independent image+gyro rest detector owns this decision. No bias
            # correction pretending the uncertain anchor orientation is truth.
            if samples:self.held=samples[-1]
            self.stamp=timestamp_ns;self.v.fill(0.)
        else:
            for s in samples:
                self._advance_rotation(s.timestamp_ns);self.held=s
            self._advance_rotation(timestamp_ns)
            dt=(timestamp_ns-old_stamp)*1e-9
            if visual_motion is not None:
                displacement=old_R@cp.asarray(visual_motion[:3,3])
                # PnP translation already describes the PLY origin; do not add
                # a lever-arm correction to the same displacement twice.
                displacement+=self._noise((self.n,3),self.cfg['visual_velocity_std_mps']*dt)
                # Keep velocity at the IMU origin, including when the next
                # frame loses PnP and switches to the lever-arm motion model.
                self.v=(displacement+(self.R-old_R)@self.ext[:3,3])/dt
            else:
                self.v+=self._noise((self.n,3),self.cfg['velocity_diffusion_mps_sqrt_s']*dt**.5)
                displacement=self.v*dt-(self.R-old_R)@self.ext[:3,3]
            self.p+=displacement+self._noise((self.n,3),self.cfg['position_diffusion_m_sqrt_s']*dt**.5)
            self.logs+=self._noise(self.n,self.cfg['log_scale_diffusion_sqrt_s']*dt**.5)
            self._bound_scales()
        if input_scale is not None:
            self.logs+=cp.log(input_scale/self.scale);self._bound_scales()
        prior,prior_scale=self._best(weighted=False)
        if stationary:prior=previous_output;prior_scale=self.scale if input_scale is None else float(input_scale)
        self.T=prior.copy();self.scale=prior_scale
        self.pending=dict(prior=prior.copy(),scale=prior_scale,stationary=bool(stationary),logw=self.logw.copy())
        self.measurement=dict(applied=False,reason='not_evaluated')
        self.prediction_diagnostics=dict(mode='initial_guess' if first else 'stationary_visual_imu' if stationary else 'gpu_monte_carlo',
            samples=len(samples),stationary=bool(stationary),translation_model='causal_visual_displacement' if visual_motion is not None else 'metric_velocity_random_walk',
            accelerometer_translation_used=False,kalman_used=False,input_scale_applied=input_scale is not None,
            latest_bag_timestamp_ns=int(self.held.arrival_timestamp_ns if self.held.arrival_timestamp_ns is not None else self.held.timestamp_ns),
            latest_header_timestamp_ns=int(self.held.header_timestamp_ns),sample_age_seconds=(timestamp_ns-self.held.timestamp_ns)*1e-9)
        return prior.copy()

    def observe_cloud(self,points,registration):
        """One common sample/map/score for ALL hypotheses; fixed memory batches."""
        if self.pending is None:raise RuntimeError('Predict before observing cloud')
        if self.measurement['reason']!='not_evaluated':raise RuntimeError('Cloud already observed this frame')
        if not self.initialized or self.pending['stationary']:
            self.measurement=dict(applied=False,reason='bootstrap' if not self.initialized else 'stationary_hold')
            return self.T.copy()
        c=self.cfg;p=cp.asarray(points,dtype=cp.float64)
        p=p[cp.isfinite(p).all(axis=1)]
        if len(p)<30:
            self.measurement=dict(applied=False,reason='insufficient_source');return self.T.copy()
        p,_=voxel_mean(p*self.scale,c['source_leaf_m']);p/=self.scale
        if len(p)>c['max_points']:p=p[cp.linspace(0,len(p)-1,c['max_points']).astype(cp.int64)]
        if len(p)<30:
            self.measurement=dict(applied=False,reason='insufficient_voxel_support');return self.T.copy()
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
        likelihood=-c['effective_points']*costs/(2*c['robust_m']**2)
        self.logw,w,ess=normalize_log_weights(cp.where(valid,self.logw+likelihood,-cp.inf))
        selected=cp.argmax(self.logw);self.T,self.scale=self._best()
        self.measurement=dict(applied=True,reason='common_cloud_robust_likelihood',points=len(p),ess=float(ess),
            selected_particle=int(selected),selected_cost=float(costs[selected]),selected_coverage=float(coverage[selected]),
            valid_particles=int(cp.sum(valid)),max_weight=float(cp.max(w)),effective_points=c['effective_points'],
            reductions_device='cuda',refinement_scored_as_independent_measurement=False)
        return self.T.copy()

    def _bootstrap(self,estimate):
        init=estimate.get('initialization') or {}
        anchors=[r for r in init.get('refinements',[]) if r.get('eligible')]
        if not anchors:anchors=[dict(T=estimate['T'],scale=estimate['scale'])]
        R=cp.asarray([np.asarray(a['T'])[:3,:3] for a in anchors]);p=cp.asarray([np.asarray(a['T'])[:3,3] for a in anchors])
        scales=cp.asarray([a['scale'] for a in anchors]);ids=cp.arange(self.n)%len(anchors)
        self.R=rotation_exp(self._noise((self.n,3),self.cfg['bootstrap_rotation_std_rad']))@R[ids]
        self.p=p[ids]+self._noise((self.n,3),self.cfg['bootstrap_position_std_m'])
        self.logs=cp.log(scales[ids])+self._noise(self.n,self.cfg['bootstrap_log_scale_std']);self._bound_scales()
        self.logw.fill(-cp.log(float(self.n)))
        self.R[0]=cp.asarray(estimate['T'][:3,:3]);self.p[0]=cp.asarray(estimate['T'][:3,3]);self.logs[0]=cp.log(estimate['scale'])
        self.representative=0
        self.v[:]=cp.asarray(self.imu_cfg['initial_velocity_map_mps'])
        self.bg=cp.asarray(self.imu_cfg['initial_gyro_bias_radps'])+self._noise((self.n,3),1.)*cp.asarray(self.imu_cfg['initial_std'][12:15])
        return len(anchors)

    def commit(self,estimate):
        if self.pending is None:raise RuntimeError('Predict before committing')
        pending=self.pending;candidate=np.asarray(estimate['T'],dtype=float);scale=float(estimate['scale'])
        valid=valid_pose(candidate) and np.isfinite(scale) and self.bounds[0]<scale<self.bounds[1]
        accepted=bool(estimate['accepted']) and valid and not pending['stationary']
        reason='stationary_visual_imu_hold' if pending['stationary'] else 'invalid_candidate_pose_or_scale' if estimate['accepted'] and not valid else estimate['reason']
        bootstrap=accepted and not self.initialized;resampled=False;ancestors=None;anchor_count=None
        if accepted:
            if bootstrap:anchor_count=self._bootstrap(estimate)
            elif self.measurement['applied']:
                self.logw,w,ess=normalize_log_weights(self.logw)
                self.representative=int(cp.argmax(self.logw))
                if bool(ess<self.cfg['ess_fraction']*self.n):
                    ids=systematic_indices(w,self.rng);ancestors=int(cp.unique(ids).size)
                    self.representative=int(cp.flatnonzero(ids==self.representative)[0])
                    self.R=self.R[ids].copy();self.p=self.p[ids].copy();self.v=self.v[ids].copy()
                    self.bg=self.bg[ids].copy();self.logs=self.logs[ids].copy();self.logw.fill(-cp.log(float(self.n)))
                    resampled=True
            self.T=candidate.copy();self.scale=scale;self.initialized=True
        else:
            # A failed refinement must not commit an unsupported cloud update.
            self.logw=pending['logw'];self.T=pending['prior'].copy();self.scale=pending['scale']
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
            particles=self.n,cloud=cloud,resampled=resampled,unique_ancestors=ancestors,
            bootstrap_anchors=anchor_count,posterior_recentered_by_refinement=False)
        self.pending=None
        return dict(accepted=bool(accepted),reason=reason,prediction=self.prediction_diagnostics['mode'],
            decision='initialized_from_registration' if bootstrap else 'monte_carlo_refined_registration' if accepted else 'stationary_pose_hold' if pending['stationary'] else 'retained_particle_prediction',
            delta=d,inertial=dict(prediction=self.prediction_diagnostics,update=update),monte_carlo=update)
