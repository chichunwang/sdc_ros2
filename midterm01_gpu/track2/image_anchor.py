"""Causal metric depth reference, separate from current map-pose acceptance."""
import numpy as np
from . import parameters as p


def metric_reference(registration):
    planes=registration.get('planes') or {};ground=planes.get('source_ground')
    failure=dict(accepted=False,reason='missing_current_ground_or_height_memory')
    if ground is None:return failure
    h=registration.get('height_anchor_after_m');var=registration.get('height_variance_after_m2')
    ds=ground.get('offset');mad=ground.get('fused_mad_raw',ground.get('mad'))
    n=np.asarray(ground.get('normal',[]),dtype=float)
    if h is None or var is None or ds is None or mad is None:return failure
    if (n.shape!=(3,) or not np.isfinite(np.r_[h,var,ds,mad,n]).all()
            or var<0 or ds<=0 or mad<0 or abs(np.linalg.norm(n)-1.)>.02):
        return dict(failure,reason='invalid_ground_height_uncertainty')
    scale=h/ds;hvar=var+p.HEIGHT_OBSERVATION_FLOOR_M**2
    logvar=hvar/h**2+(mad/ds)**2 if h>0 else np.inf
    angle=registration.get('ground_normal_difference_before_deg')
    sigma=np.sqrt(max(np.asarray(p.IMU_ORIENTATION_UNKNOWN_STD_DEG)[:2]**2
                     +np.asarray(p.EXTRINSIC_ROTATION_STD_DEG)[:2]**2))
    valid=(p.CAMERA_HEIGHT_BOUNDS_M[0]<h<p.CAMERA_HEIGHT_BOUNDS_M[1]
        and p.SCALE_BOUNDS[0]<scale<p.SCALE_BOUNDS[1] and n[2]>.85
        and ground.get('inliers',0)>=p.MIN_SOURCE_GROUND_POINTS
        and mad*scale<=p.MAX_SOURCE_GROUND_MAD_M and logvar<=p.MAX_SCALE_LOG_STD**2
        and angle is not None and np.isfinite(angle) and (angle/sigma)**2<=p.GROUND_NORMAL_NIS_REJECT)
    if not valid:return dict(failure,reason='ground_height_memory_untrusted',height_m=float(h),
                             scale_log_variance=float(logvar) if np.isfinite(logvar) else None)
    return dict(accepted=True,reason='current_ground_with_causal_height_memory',
        height_m=float(h),height_variance_m2=float(hvar),scale=float(scale),scale_log_variance=float(logvar),
        normal=n.tolist(),offset_raw=float(ds),threshold_raw=float(ground['threshold']),
        ground_mad_raw=float(mad),ground_inliers=int(ground['inliers']),
        ground_normal_model_variance_rad2=float(np.deg2rad(p.GROUND_NORMAL_FLOOR_DEG)**2),
        height_memory_variance_m2=float(var),height_model_floor_m2=float(p.HEIGHT_OBSERVATION_FLOOR_M**2),
        map_pose_accepted=bool(registration.get('accepted')),current_scale_validated=bool(registration.get('scale_validated')),
        uncertainty_scope='past/current causal height plus original floor and correlated plane MAD; not independent map information')
