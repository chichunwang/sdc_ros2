"""Causal motion evidence and raw-reconstruction scale, never a map pose solve.

Image tracking is CPU OpenCV; pixel/cloud association and 3D distance reductions
stay on CUDA. Pair distances cancel camera R,t, so they compare reconstruction
units instead of treating an apparent depth change as vehicle translation.
"""
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
from .io import frames, sha256
from .motion_parameters import PARAMETERS


def track_images(before, after, mask, cfg):
    p0=cv2.goodFeaturesToTrack(before,maxCorners=cfg['max_features'],qualityLevel=.01,minDistance=12,mask=mask)
    if p0 is None:return None
    lk=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    p1,s1,_=cv2.calcOpticalFlowPyrLK(before,after,p0,None,**lk)
    if p1 is None:return None
    back,s0,_=cv2.calcOpticalFlowPyrLK(after,before,p1,None,**lk)
    if back is None:return None
    good=(s1.ravel()!=0)&(s0.ravel()!=0)&(np.linalg.norm(back-p0,axis=2).ravel()<cfg['forward_backward_px'])
    a=p0.reshape(-1,2)[good];b=p1.reshape(-1,2)[good]
    h,w=mask.shape
    inside=(b[:,0]>=0)&(b[:,0]<w)&(b[:,1]>=0)&(b[:,1]<h)
    a=a[inside];b=b[inside]
    if len(a)<cfg['min_tracks']:return None
    flow=np.linalg.norm(b-a,axis=1)
    cells=np.unique((np.clip(a[:,1]*6/h,0,5).astype(int)*8)+np.clip(a[:,0]*8/w,0,7).astype(int))
    if len(cells)<cfg['min_grid_cells']:return None
    return a,b,dict(tracks=len(a),grid_cells=len(cells),median_px=float(np.median(flow)),p90_px=float(np.percentile(flow,90)))


class RestDetector:
    """Only current/past evidence; quiet IMU alone never declares rest."""
    def __init__(self,cfg):
        self.cfg=cfg;self.since=None

    def update(self,stamp,previous_stamp,flow,baseline_flow,gyro_max,enabled):
        c=self.cfg
        quiet=(enabled and flow is not None and baseline_flow is not None and
               gyro_max is not None and gyro_max<c['max_raw_gyro_radps'] and
               flow['median_px']<c['quiet_median_px'] and flow['p90_px']<c['quiet_p90_px'] and
               baseline_flow['median_px']<c['anchor_median_px'] and baseline_flow['p90_px']<c['anchor_p90_px'])
        if not quiet:self.since=None;return False
        if self.since is None:self.since=previous_stamp
        return (stamp-self.since)*1e-9>=c['quiet_duration_s']


def scale_from_pairs(previous,current,cfg):
    """Robust invariant lengths; no alignment R/t and no CPU cloud fallback."""
    import cupy as cp
    if len(previous)<cfg['scale_min_points']:return None,dict(reason='insufficient_3d_tracks',points=len(previous))
    ids=cp.linspace(0,len(previous)-1,min(len(previous),cfg['scale_max_points'])).astype(cp.int64)
    a=cp.asarray(previous,dtype=cp.float64)[ids];b=cp.asarray(current,dtype=cp.float64)[ids]
    i,j=cp.triu_indices(len(a),1)
    da=cp.linalg.norm(a[i]-a[j],axis=1);db=cp.linalg.norm(b[i]-b[j],axis=1)
    # Exclude very short baselines where depth noise dominates the length ratio.
    good=(da>.05*cp.median(cp.linalg.norm(a,axis=1)))&(db>.05*cp.median(cp.linalg.norm(b,axis=1)))
    if int(good.sum())<100:return None,dict(reason='insufficient_3d_baselines')
    logs=cp.log(da[good]/db[good]);center=cp.median(logs)
    mad=float(cp.median(cp.abs(logs-center)))
    keep=cp.abs(logs-center)<=max(3*1.4826*mad,.02)
    fraction=float(keep.mean())
    if mad>cfg['scale_max_log_mad'] or fraction<cfg['scale_min_pair_fraction']:
        return None,dict(reason='nonuniform_reconstruction_or_dynamic_tracks',log_mad=mad,inlier_fraction=fraction)
    ratio=float(cp.exp(cp.median(logs[keep])))
    if not np.isfinite(ratio) or not .5<ratio<2.:
        return None,dict(reason='implausible_relative_scale',ratio=ratio)
    return ratio,dict(reason='causal_3d_pair_lengths',log_mad=mad,inlier_fraction=fraction,
                      points=len(a),pairs=int(good.sum()),ratio=ratio)


def associated_points(points,pixels,K,cfg):
    import cupy as cp
    from .search import closest_pixels
    p=cp.asarray(points,dtype=cp.float64);p=p[cp.isfinite(p).all(axis=1)&(p[:,0]>0)]
    if not len(p):return None,None
    uv=cp.stack([K[0,2]-K[0,0]*p[:,1]/p[:,0],K[1,2]-K[1,1]*p[:,2]/p[:,0]],axis=1)
    ids,d2=closest_pixels(pixels,uv)
    valid=d2<=cfg['depth_projection_px']**2
    return p[ids],valid


class MotionConstraints:
    def __init__(self,data,camera,expected_frames):
        self.cfg=deepcopy(PARAMETERS);self.camera=camera
        self.images={r['timestamp_ns']:r['path'] for r in frames(Path(data)/'Track3_images/Track3_f_images',suffix='.jpg',expected=expected_frames)}
        self.previous=None;self.current=None;self.quiet_reference=None;self.scale_reference=None
        self.rest=RestDetector(self.cfg);self.was_stationary=False

    def observe(self,stamp,points,current_scale,samples,enabled,measurement_timestamp_ns=None):
        measurement_stamp=stamp if measurement_timestamp_ns is None else measurement_timestamp_ns
        path=self.images[stamp];gray=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if gray is None:raise ValueError(f'Unreadable motion evidence image: {path}')
        gray=cv2.remap(gray,*self.camera.maps,cv2.INTER_LINEAR)
        self.current=dict(stamp=stamp,measurement_stamp=measurement_stamp,gray=gray,points=np.asarray(points).copy(),scale=float(current_scale))
        diag=dict(stationary=False,scale_valid=False,image=str(path),image_sha256=sha256(path),
                  source='current_and_past_images_plus_raw_gyro',parameters=self.cfg)
        self.visual_motion=None
        if self.previous is None or not enabled:
            self.quiet_reference=None;self.rest.since=None
            return None,False,diag
        flow=track_images(self.previous['gray'],gray,self.camera.mask,self.cfg)
        baseline=self.quiet_reference or self.previous
        baseline_flow=flow if baseline is self.previous else track_images(baseline['gray'],gray,self.camera.mask,self.cfg)
        gyro=max((float(np.linalg.norm(s.angular_velocity)) for s in samples),default=None)
        stationary=self.rest.update(measurement_stamp,self.previous.get('measurement_stamp',self.previous['stamp']),None if flow is None else flow[2],
                                    None if baseline_flow is None else baseline_flow[2],gyro,enabled)
        if self.rest.since is None:self.quiet_reference=None
        elif self.quiet_reference is None:self.quiet_reference=self.previous
        if stationary and not self.was_stationary:
            # Anchor at the last committed frame, never move old poses to fit it.
            self.scale_reference=self.previous
        reference=self.scale_reference or self.previous
        matched=flow if reference is self.previous else track_images(reference['gray'],gray,self.camera.mask,self.cfg)
        if not stationary and (matched is None or matched[2]['median_px']>self.cfg['keyframe_median_px'] or
                               (measurement_stamp-reference.get('measurement_stamp',reference['stamp']))*1e-9>self.cfg['keyframe_max_age_s']):
            reference=self.scale_reference=self.previous;matched=flow
        scale=None;metric_depth=None;scale_diag=dict(reason='insufficient_image_tracks')
        if matched is not None:
            import cupy as cp
            a,va=associated_points(reference['points'],matched[0],self.camera.K,self.cfg)
            b,vb=associated_points(points,matched[1],self.camera.K,self.cfg)
            if a is not None and b is not None:
                valid=va&vb
                if int(valid.sum()):metric_depth=float(cp.median(cp.linalg.norm(a[valid],axis=1)))*reference['scale']
                ratio,scale_diag=scale_from_pairs(a[valid],b[valid],self.cfg)
                if ratio is not None:
                    scale=reference['scale']*ratio
                    if not np.isfinite(scale) or not .2<scale<30.:
                        scale=None;scale_diag['reason']='absolute_scale_bounds'
        if stationary and (scale is None or metric_depth is None or metric_depth>self.cfg['stationary_max_depth_m']):
            stationary=False
            diag['stationary_rejection']='insufficient_reliable_nearby_3d_support'
        pnp_diag=dict(accepted=False,reason='stationary' if stationary else 'insufficient_image_tracks')
        if not stationary and flow is not None:
            import cupy as cp
            from .visual_prior import relative_pose_from_pnp
            objects,valid=associated_points(self.previous['points'],flow[0],self.camera.K,self.cfg)
            if objects is not None:
                good=cp.asnumpy(valid)
                objects=cp.asnumpy(objects[valid])*self.previous['scale'];pixels=flow[1][good]
                if len(objects)>=self.cfg['pnp_min_inliers']:
                    cv2.setRNGSeed(246)
                    ok,rvec,tvec,inliers=cv2.solvePnPRansac(objects,pixels,self.camera.K,None,iterationsCount=200,
                        reprojectionError=self.cfg['pnp_reprojection_px'],confidence=.999,flags=cv2.SOLVEPNP_EPNP)
                    count=0 if inliers is None else len(inliers)
                    pnp_diag.update(inliers=count,correspondences=len(objects))
                    if ok and count>=self.cfg['pnp_min_inliers'] and count/len(objects)>=self.cfg['pnp_min_fraction']:
                        ids=inliers.ravel()
                        ok,rvec,tvec=cv2.solvePnP(objects[ids],pixels[ids],self.camera.K,None,rvec,tvec,True,flags=cv2.SOLVEPNP_ITERATIVE)
                        motion=relative_pose_from_pnp(rvec,tvec)
                        projected,_=cv2.projectPoints(objects[ids],rvec,tvec,self.camera.K,None)
                        reprojection=float(np.median(np.linalg.norm(projected.reshape(-1,2)-pixels[ids],axis=1)))
                        dt=(measurement_stamp-self.previous.get('measurement_stamp',self.previous['stamp']))*1e-9
                        from scipy.spatial.transform import Rotation
                        speed=float(np.linalg.norm(motion[:3,3])/dt)
                        turn=float(Rotation.from_matrix(motion[:3,:3]).magnitude()/dt)
                        accepted=bool(ok and reprojection<=self.cfg['pnp_reprojection_px'] and
                                      speed<=self.cfg['max_speed_mps'] and turn<=self.cfg['max_rotation_rate_radps'])
                        pnp_diag.update(accepted=accepted,reason='causal_visual_velocity' if accepted else 'visual_motion_gate',
                                        speed_mps=speed,rotation_rate_radps=turn,reprojection_median_px=reprojection)
                        if accepted:self.visual_motion=motion
                    else:pnp_diag['reason']='pnp_outliers'
        diag.update(stationary=stationary,scale_valid=scale is not None,scale=scale,metric_feature_depth_m=metric_depth,
                    visual_velocity=pnp_diag,
                    reference_timestamp_ns=reference['stamp'],reference_scale=reference['scale'],
                    image_flow=None if flow is None else flow[2],baseline_flow=None if baseline_flow is None else baseline_flow[2],
                    raw_gyro_max_radps=gyro,quiet_seconds=0. if self.rest.since is None else (measurement_stamp-self.rest.since)*1e-9,
                    measurement_timestamp_ns=measurement_stamp,
                    quiet_reference_timestamp_ns=None if self.quiet_reference is None else self.quiet_reference['stamp'],
                    quiet_since_measurement_ns=self.rest.since,
                    scale_diagnostics=scale_diag)
        self.was_stationary=stationary
        return scale,stationary,diag

    def store(self,scale,*,anchored=True,bootstrap=False):
        self.current['scale']=float(scale)
        self.previous=self.current
        if self.scale_reference is None or not anchored or bootstrap:
            self.scale_reference=self.previous
        if bootstrap:
            self.quiet_reference=None;self.rest.since=None;self.was_stationary=False
