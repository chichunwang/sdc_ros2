"""Causal image speed / relative motion updates into XY and heading states.

v5.13 uses disjoint image intervals for vector velocity or relative yaw. A
relative yaw observes bias change, never constant absolute map-heading bias.
Height, depth, gyro and lever uncertainty are propagated with model floors;
shared cross-sensor correlations are not a calibrated joint estimator.
"""
import numpy as np
from . import parameters as p


def fuse_visual_speed(state, prior, diagnostic, measurement, gyro_norm):
    result = dict(measurement, filter_accepted=False)
    if not measurement.get('accepted'):
        return prior, diagnostic, result
    if measurement.get('kind') in ('vector_velocity','relative_yaw'):
        return fuse_relative_motion(state,prior,diagnostic,measurement,gyro_norm)
    if int(measurement['timestamp_ns']) != state.time:
        raise ValueError('Visual observation must be at the current state time')
    if p.v514_enabled() and measurement.get('image_interval_model'):
        anchor=state.visual_heading_anchor
        if anchor is None or anchor['timestamp_ns']!=int(measurement['keyframe_timestamp_ns']):
            raise ValueError('Image/heading anchors differ')
        if anchor['used']:raise ValueError('Relative image interval already consumed')
        anchor['used']=True
    variance = float(measurement['variance_m2ps2'])
    if not np.isfinite(variance) or variance <= 0 or not np.isfinite(gyro_norm):
        raise ValueError('Invalid visual velocity uncertainty')
    before = state.cov.copy(); velocity = state.position[2:].copy()
    if measurement['kind'] == 'stationary':
        H = np.c_[np.zeros((2, 2)), np.eye(2)]
        innovation = -velocity; R = np.eye(2)*variance; dof = 2
    elif measurement['kind'] == 'speed':
        speed = float(np.linalg.norm(velocity))
        if speed < .1:
            return prior, diagnostic, dict(result, filter_reason='no_sensor_velocity_direction')
        value = float(measurement['speed_mps'])
        if not np.isfinite(value) or value < 0:
            raise ValueError('Invalid image speed')
        H = np.r_[0., 0., velocity/speed].reshape(1, 4)
        innovation = np.array([value-speed]); dof = 1
        # Image displacement is camera motion; unknown antenna/camera lever
        # induces angular-rate-dependent speed error, never an assumed offset.
        lever_variance = float(np.sum(np.asarray(p.LEVER_ARM_STD_BODY_M)**2))
        R = np.array([[variance+gyro_norm**2*lever_variance]])
    else:
        raise ValueError('Unknown visual velocity observation')
    S = H @ before @ H.T+R
    nis = float(innovation @ np.linalg.solve(S, innovation))
    result.update(innovation=innovation.tolist(), innovation_nis=nis, observation_matrix=H.tolist())
    if nis > p.CORRECTION_NIS_REJECT_BY_DOF[dof-1]:
        return prior, diagnostic, dict(result, filter_reason='visual_velocity_statistical_gate')
    inflation = max(1., nis/p.CORRECTION_NIS_SOFT_BY_DOF[dof-1]); effective = R*inflation
    K = np.linalg.solve(H @ before @ H.T+effective, H @ before).T
    increment = K @ innovation
    covariance = (np.eye(4)-K @ H) @ before @ (np.eye(4)-K @ H).T+K @ effective @ K.T
    covariance = (covariance+covariance.T)*.5
    state.position += increment; state.cov = covariance
    output = np.asarray(prior).copy(); output[:2, 3] += increment[:2]
    C = np.asarray(diagnostic['pose_covariance_xy_yaw']).copy()
    C[:2, :2] += covariance[:2, :2]-before[:2, :2]
    if np.linalg.eigvalsh(C).min() <= 0:
        raise ValueError('Visual update produced non-positive pose covariance')
    diag = dict(diagnostic, velocity_xy_mps=state.position[2:].tolist(),
        position_covariance_xy=state.cov[:2, :2].tolist(), pose_covariance_xy_yaw=C.tolist())
    diag['uncertainty_model'] = dict(diagnostic['uncertainty_model'],
        image_speed='past-height/image interval average; model/height/acceleration/lever floors; cross-frame correlations not fully estimated')
    result.update(filter_accepted=True, filter_reason='current_image_velocity_update',
        effective_measurement_covariance=effective.tolist(), kalman_gain=K.tolist(),
        state_increment=increment.tolist(), covariance_before=before.tolist(), covariance_after=state.cov.tolist(),
        covariance_inflation=inflation, state_reference_policy='kinematic innovation; not calibration rebase')
    if measurement['kind'] == 'stationary':
        result['stationary_course_exclusion'] = state.exclude_image_stationary_course(
            int(measurement['keyframe_timestamp_ns']), int(measurement['timestamp_ns']))
        if getattr(state,'gyro_motion_bias',None) is not None:
            result['gyro_motion_bias_update']=state.update_gyro_motion_bias(
                int(measurement['keyframe_timestamp_ns']),int(measurement['timestamp_ns']))
    return output, diag, result


def fuse_relative_motion(state,prior,diagnostic,measurement,gyro_norm):
    """One relative-motion channel per disjoint interval; never absolute pose."""
    from scipy.spatial.transform import Rotation
    from .causal_prior import wrap,skew
    if not p.v513_enabled():raise ValueError('Relative image model disabled')
    if int(measurement['timestamp_ns'])!=state.time:raise ValueError('Visual measurement must be current')
    dt=float(measurement['interval_s']);stamp=int(measurement['keyframe_timestamp_ns'])
    if dt<=0 or abs((state.time-stamp)*1e-9-dt)>1e-9:raise ValueError('Invalid relative image interval')
    anchor=state.visual_heading_anchor
    if anchor is None or anchor['timestamp_ns']!=stamp:raise ValueError('Image/heading anchors differ')
    if anchor['used']:raise ValueError('Relative image interval already consumed')
    T=np.asarray(measurement['relative_previous_current']);C=np.asarray(measurement['pose_covariance'])
    Ranchor=np.asarray(anchor['camera_rotation'])
    if T.shape!=(4,4) or C.shape!=(6,6) or not np.isfinite(T).all() or not np.isfinite(C).all() or np.linalg.eigvalsh(C).min()< -1e-10:
        raise ValueError('Invalid relative image pose/covariance')
    if measurement['kind']=='vector_velocity' and p.v514_enabled():
        # The map heading used to rotate this vector is also used by the
        # sensor state. Its common error is not an independent direction
        # observation. The speed norm avoids feeding that heading back as
        # new lateral information; existing scalar-speed fusion is reused.
        model=measurement.get('anchor_model') or {}
        normal=np.asarray(model.get('normal',[0.,0.,1.]),dtype=float)
        if normal.shape!=(3,) or not np.isfinite(normal).all() or abs(np.linalg.norm(normal)-1.)>.02:
            raise ValueError('Invalid relative speed ground normal')
        normal=normal/np.linalg.norm(normal);planar=np.eye(3)-np.outer(normal,normal)
        velocity=planar@T[:3,3]/dt;covariance=planar@C[:3,:3]@planar.T/dt**2
        variance=max(0.,float(np.linalg.eigvalsh(covariance).max()))
        variance+=p.IMAGE_SPEED_STD_MPS**2+p.GNSS_XY_ACCEL_PSD*dt/4
        variance+=float(T[:3,3]@T[:3,3])/dt**2*float(model.get('ground_normal_model_variance_rad2',np.deg2rad(p.GROUND_NORMAL_FLOOR_DEG)**2))
        scalar=dict(measurement,kind='speed',relative_channel='vector_velocity',
            image_interval_model='relative_translation_speed_norm',speed_mps=float(np.linalg.norm(velocity)),
            variance_m2ps2=variance,relative_planar_velocity_covariance=covariance.tolist(),
            fusion_scope='scalar speed; common absolute heading is not an independent velocity direction')
        return fuse_visual_speed(state,prior,diagnostic,scalar,gyro_norm)
    result=dict(measurement,filter_accepted=False,anchor_reference_rotation_applied=Ranchor.tolist(),
                anchor_calibration_rebase_rad=anchor['calibration_rebase_rad'])
    output=np.asarray(prior).copy();diag=dict(diagnostic)
    if measurement['kind']=='vector_velocity':
        anchor['used']=True
        velocity=Ranchor@T[:3,3]/dt
        E=Rotation.from_euler('xyz',p.IMU_TO_PLY_RPY_DEG,degrees=True).as_matrix()
        body=prior[:3,:3]@E.T
        lever_velocity=body@np.cross(state.gyro,np.asarray(p.GNSS_TO_PLY_LEVER_BODY_M))
        value=velocity[:2]-lever_velocity[:2]
        Cangle=Ranchor@anchor['orientation_covariance_camera']@Ranchor.T
        Cangle[2,2]+=anchor['variance']
        anchor_body=Ranchor@E.T
        Cangle+=anchor_body@np.diag(np.deg2rad(p.EXTRINSIC_ROTATION_STD_DEG)**2)@anchor_body.T
        R=Ranchor@C[:3,:3]@Ranchor.T/dt**2+skew(velocity)@Cangle@skew(velocity).T
        R=R[:2,:2]+np.eye(2)*(p.IMAGE_SPEED_STD_MPS**2+p.GNSS_XY_ACCEL_PSD*dt/4
            +gyro_norm**2*float(np.sum(np.asarray(p.LEVER_ARM_STD_BODY_M)**2)))
        H=np.c_[np.zeros((2,2)),np.eye(2)];before=state.cov.copy()
        innovation=value-state.position[2:];S=H@before@H.T+R;nis=float(innovation@np.linalg.solve(S,innovation))
        result.update(observation_matrix=H.tolist(),innovation=innovation.tolist(),innovation_nis=nis,
                      velocity_measurement_map_mps=value.tolist())
        if nis>p.CORRECTION_NIS_REJECT_BY_DOF[1]:return prior,diagnostic,dict(result,filter_reason='visual_vector_velocity_statistical_gate')
        effective=R*max(1.,nis/p.CORRECTION_NIS_SOFT_BY_DOF[1]);K=np.linalg.solve(H@before@H.T+effective,H@before).T
        increment=K@innovation;A=np.eye(4)-K@H
        state.position+=increment;state.cov=A@before@A.T+K@effective@K.T;state.cov=(state.cov+state.cov.T)*.5
        output[:2,3]+=increment[:2]
        result.update(filter_accepted=True,filter_reason='current_image_vector_velocity_update',state_increment=increment.tolist(),
            kalman_gain=K.tolist(),effective_measurement_covariance=effective.tolist(),covariance_before=before.tolist(),covariance_after=state.cov.tolist())
    else:
        def yaw(R):return float(np.arctan2(R[1,0],R[0,0]))
        visual_delta=wrap(yaw(Ranchor@T[:3,:3])-yaw(Ranchor))
        sensor_delta=wrap(yaw(prior[:3,:3])-state.bias-(yaw(Ranchor)-anchor['bias']))
        rv=Rotation.from_matrix(T[:3,:3]).as_rotvec()
        gradient=np.array([wrap(yaw(Ranchor@Rotation.from_rotvec(rv+np.eye(3)[i]*1e-6).as_matrix())
                              -yaw(Ranchor@Rotation.from_rotvec(rv-np.eye(3)[i]*1e-6).as_matrix()))/2e-6 for i in range(3)])
        variance=float(gradient@C[3:,3:]@gradient+anchor['orientation_variance']+state.orientation_cov[2,2])
        variance+=np.deg2rad(p.EXTRINSIC_ROTATION_STD_DEG[2])**2*float(rv@rv)
        before_bias=state.bias
        decision=state.update_relative_heading(wrap(visual_delta-sensor_delta),variance,stamp)
        result.update(relative_heading_update=decision,filter_accepted=decision['accepted'],filter_reason=decision['reason'])
        if not decision['accepted']:return prior,diagnostic,result
        change=wrap(state.bias-before_bias);turn=Rotation.from_euler('z',change).as_matrix()
        E=Rotation.from_euler('xyz',p.IMU_TO_PLY_RPY_DEG,degrees=True).as_matrix()
        lever=prior[:3,:3]@E.T@np.asarray(p.GNSS_TO_PLY_LEVER_BODY_M)
        lever_change=(turn@lever-lever)[:2];output[:2,3]+=lever_change;output[:3,:3]=turn@output[:3,:3]
        diag['calibration_delta_xy_yaw']=(np.asarray(diag['calibration_delta_xy_yaw'])+np.r_[lever_change,change]).tolist()
        state.last_frame_bias=state.bias  # This calibration rebase is delivered NOW, never again next frame.
        diag.update(heading_bias_rad=state.bias,heading_bias_std_rad=float(np.sqrt(state.bias_var)),
            gnss_heading_bias_rad=float(state.heading_bias[1]),heading_bias_covariance=state.heading_covariance.tolist(),
            heading_update_budget_used_deg=float(np.degrees(state.heading_update_budget_used)))
    poseC=state.visual_pose_covariance(output[:3,:3],diagnostic.get('imu_orientation_extrapolation_s',0.))
    if np.linalg.eigvalsh(poseC).min()<=0:raise ValueError('Relative visual fusion produced invalid pose covariance')
    diag.update(velocity_xy_mps=state.position[2:].tolist(),position_covariance_xy=state.cov[:2,:2].tolist(),pose_covariance_xy_yaw=poseC.tolist())
    diag['uncertainty_model']=dict(diag['uncertainty_model'],image_relative='disjoint velocity/yaw intervals; Schmidt heading clone; height/gyro/depth/lever floors; cross-sensor correlations not fully calibrated')
    return output,diag,result
