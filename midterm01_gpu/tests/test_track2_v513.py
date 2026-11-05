"""Generated sparse geometry and causal fusion; never reads actual Track data."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from track2 import parameters as p
from track2.image_anchor import metric_reference
from track2.relative_geometry import compare_relative,OPTICAL_TO_FLU,project
from track2.causal_prior import CausalSensorPrior
from track2.visual_update import fuse_visual_speed


@pytest.fixture(autouse=True)
def production(monkeypatch):
    monkeypatch.setattr(p,'V514_ENABLED',False)
    monkeypatch.setattr(p,'V510_ENABLED',True);monkeypatch.setattr(p,'V513_ENABLED',True)
    monkeypatch.setattr(p,'V5121_ENABLED',False)


def registration(**kw):
    d=dict(accepted=False,scale_validated=False,height_anchor_after_m=1.6,height_variance_after_m2=.05**2,
        ground_normal_difference_before_deg=1.,planes=dict(source_ground=dict(offset=.2,normal=[0.,0.,1.],mad=.002,inliers=200,threshold=.015)))
    d.update(kw);return d


def test_metric_anchor_does_not_require_current_map_pose_or_candidate_scale():
    d=metric_reference(registration(scale=100.));assert d['accepted'] and d['scale']==pytest.approx(8.)
    assert not d['map_pose_accepted'] and not d['current_scale_validated']
    assert d['height_variance_m2']>.05**2 and d['scale_log_variance']>0


@pytest.mark.parametrize('variance',[1.,-1.,np.nan])
def test_untrusted_height_memory_does_not_create_an_image_depth_anchor(variance):
    assert not metric_reference(registration(height_variance_after_m2=variance))['accepted']


def scene(yaw=.025,translation=(1.,.15,.01)):
    rng=np.random.default_rng(513);objects=rng.uniform([5.,-8.,-2.],[30.,8.,6.],(700,3))
    K=np.array([[400.,0.,320.],[0.,400.,240.],[0.,0.,1.]])
    R=Rotation.from_euler('xyz',[.003,-.002,yaw]).as_matrix();t=np.asarray(translation)
    before,_=project(objects,OPTICAL_TO_FLU.T,np.zeros(3),K)
    pixels,depth=project(objects,OPTICAL_TO_FLU.T@R.T,-OPTICAL_TO_FLU.T@R.T@t,K)
    keep=(before[:,0]>4)&(before[:,0]<636)&(before[:,1]>4)&(before[:,1]<476)&(pixels[:,0]>4)&(pixels[:,0]<636)&(pixels[:,1]>4)&(pixels[:,1]<476)&(depth>0)
    objects,pixels=objects[keep],pixels[keep];pixels+=rng.normal(0.,.15,pixels.shape);pixels[::10]+=[22.,-18.]
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=t
    return objects,pixels,K,T


@pytest.mark.parametrize('yaw',[0.,.025,-.04])
def test_full_background_pnp_and_gyro_translation_recover_same_flu_motion(yaw):
    objects,pixels,K,T=scene(yaw)
    results=compare_relative(objects,pixels,K,.25,(480,640),T[:3,:3],np.deg2rad(.2)**2,.001)
    for name,d in results.items():
        assert d['accepted'],(name,d)
        recovered=np.asarray(d['relative_previous_current'])
        np.testing.assert_allclose(recovered[:3,3],T[:3,3],atol=.025)
        assert Rotation.from_matrix(recovered[:3,:3].T@T[:3,:3]).magnitude()<.003
        C=np.asarray(d['pose_covariance']);assert np.linalg.eigvalsh(C).min()>0
        assert d['grid_cells']>=8 and d['common_capped_pixel_cost']>0


def test_incompatible_gyro_does_not_pass_as_a_static_background_motion():
    objects,pixels,K,T=scene();wrong=T[:3,:3]@Rotation.from_euler('z',.3).as_matrix()
    d=compare_relative(objects,pixels,K,.25,(480,640),wrong,1e-7,.001)
    assert not d['pnp']['accepted'] and not d['gyro_translation']['accepted']


def sensor_pair():
    s=CausalSensorPrior(np.eye(4),1_000_000_000);s.position[2:]=[10.,0.]
    T,d=s.for_frame(s.time);s.capture_visual_anchor(s.time,T[:3,:3]);T,d=s.for_frame(1_250_000_000)
    return s,T,d


def measurement(kind):
    T=np.eye(4);T[:3,3]=[2.5,.25,0.]
    return dict(accepted=True,kind=kind,timestamp_ns=1_250_000_000,keyframe_timestamp_ns=1_000_000_000,
        interval_s=.25,relative_previous_current=T.tolist(),pose_covariance=np.diag([.01]*3+[.001]*3).tolist(),anchor_reference_rotation=np.eye(3).tolist())


def test_vector_velocity_observes_sideways_motion_and_updates_actual_joseph():
    s,T,d=sensor_pair();out,diag,u=fuse_visual_speed(s,T,d,measurement('vector_velocity'),0.)
    assert u['filter_accepted'] and s.position[3]>.5 and out[1,3]>0
    H=np.asarray(u['observation_matrix']);K=np.asarray(u['kalman_gain']);P=np.asarray(u['covariance_before']);A=np.eye(4)-K@H
    np.testing.assert_allclose(s.cov,A@P@A.T+K@np.asarray(u['effective_measurement_covariance'])@K.T,atol=1e-12)
    np.testing.assert_array_equal(out[:3,:3],T[:3,:3])
    with pytest.raises(ValueError,match='already consumed'):fuse_visual_speed(s,T,d,measurement('vector_velocity'),0.)


def test_relative_yaw_uses_anchor_cross_covariance_and_rebases_only_once():
    s,T,d=sensor_pair();s.orientation=Rotation.from_euler('z',-.02)
    T[:3,:3]=s.orientation.as_matrix();m=measurement('relative_yaw');m['relative_previous_current']=np.eye(4).tolist()
    out,diag,u=fuse_visual_speed(s,T,d,m,0.)
    decision=u['relative_heading_update'];assert decision['accepted'] and s.bias>0
    P=np.asarray(decision['joint_covariance_before']);H=np.asarray(decision['observation_matrix']);K=np.asarray(decision['kalman_gain']);A=np.eye(3)-np.outer(K,H)
    np.testing.assert_allclose(np.asarray(decision['joint_covariance_after']),A@P@A.T+np.outer(K,K)*decision['effective_measurement_variance'],atol=1e-12)
    assert not decision['absolute_yaw_observed']
    assert diag['calibration_delta_xy_yaw'][2]==pytest.approx(decision['bias_step_rad'])
    _,nextdiag=s.for_frame(1_300_000_000)
    assert nextdiag['calibration_delta_xy_yaw'][2]==pytest.approx(0.)


def test_common_heading_calibration_change_does_not_become_false_visual_turn():
    s,T,d=sensor_pair();s._heading_update(.03,[1.,0.],.001)
    T[:3,:3]=Rotation.from_euler('z',s.bias).as_matrix()
    m=measurement('relative_yaw');m['relative_previous_current']=np.eye(4).tolist()
    _,_,u=fuse_visual_speed(s,T,d,m,0.)
    assert u['relative_heading_update']['innovation_rad']==pytest.approx(0.,abs=1e-14)


def test_relative_yaw_cannot_observe_the_common_absolute_heading_gauge(monkeypatch):
    monkeypatch.setattr(p,'HEADING_BIAS_RW_DEG_SQRT_S',0.);monkeypatch.setattr(p,'GNSS_HEADING_BIAS_RW_DEG_SQRT_S',0.)
    s,T,d=sensor_pair();before=s.heading_covariance.copy()
    u=s.update_relative_heading(.001,.001,1_000_000_000)
    np.testing.assert_array_equal(u['kalman_gain'],np.zeros(3))
    np.testing.assert_array_equal(s.heading_covariance,before)


def test_gyro_rotation_matrix_is_causal_and_noncommuting():
    s=CausalSensorPrior(np.eye(4),1_000_000_000)
    s.event(dict(kind='imu',timestamp_ns=s.time,quaternion=[0.,0.,0.,1.],gyro=[.2,0.,0.]))
    s.for_frame(s.time)
    s.event(dict(kind='imu',timestamp_ns=1_100_000_000,quaternion=[0.,0.,0.,1.],gyro=[0.,.3,0.]))
    _,d=s.for_frame(1_200_000_000)
    expected=Rotation.from_rotvec([.02,0.,0.]).as_matrix()@Rotation.from_rotvec([0.,.03,0.]).as_matrix()
    np.testing.assert_allclose(d['gyro_rotation_matrix_since_frame'],expected,atol=1e-14)
    assert d['gyro_rotation_std_bound_rad']>0.


def test_relative_velocity_uses_camera_rotation_and_known_lever(monkeypatch):
    monkeypatch.setattr(p,'GNSS_TO_PLY_LEVER_BODY_M',(1.,0.,0.))
    s,T,d=sensor_pair();s.gyro=np.array([0.,0.,.2])
    m=measurement('vector_velocity');m['relative_previous_current'][1][3]=.05
    _,_,u=fuse_visual_speed(s,T,d,m,.2)
    # Camera lateral speed .2 = antenna lateral speed 0 + omega cross lever.
    np.testing.assert_allclose(u['velocity_measurement_map_mps'],[10.,0.],atol=1e-12)


def test_relative_fusion_prefix_is_unchanged_by_later_images():
    def sequence(count):
        s=CausalSensorPrior(np.eye(4),1_000_000_000);s.position[2:]=[10.,0.]
        pose,d=s.for_frame(s.time);s.capture_visual_anchor(s.time,pose[:3,:3]);rows=[]
        for k in range(1,count+1):
            stamp=1_000_000_000+k*250_000_000
            s.event(dict(kind='imu',timestamp_ns=stamp,quaternion=[0.,0.,0.,1.],gyro=[0.,0.,0.]))
            pose,d=s.for_frame(stamp);m=measurement('vector_velocity' if k%2 else 'relative_yaw')
            m.update(timestamp_ns=stamp,keyframe_timestamp_ns=stamp-250_000_000)
            pose,d,u=fuse_visual_speed(s,pose,d,m,0.)
            rows.append(dict(pose=pose.tolist(),covariance=s.cov.tolist(),heading=s.heading_covariance.tolist(),decision=u))
            s.capture_visual_anchor(stamp,pose[:3,:3])
        return rows
    assert sequence(11)==sequence(29)[:11]
