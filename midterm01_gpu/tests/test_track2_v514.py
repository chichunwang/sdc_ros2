"""Generated physical contracts for v5.14; no actual Track localization."""
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from track2 import parameters as p
from track2.relative_geometry import (depth_on_feature_rays,project,OPTICAL_TO_FLU,
    refine_pixel_translation,gyro_translation_fit)
from track2.visual_update import fuse_visual_speed
from test_track2_v513 import sensor_pair,measurement,scene


@pytest.fixture(autouse=True)
def production(monkeypatch):
    monkeypatch.setattr(p,'V513_ENABLED',True)
    monkeypatch.setattr(p,'V514_ENABLED',True)
    monkeypatch.setattr(p,'V510_ENABLED',True)


def test_neighbour_depth_does_not_move_a_stationary_keypoint_off_its_ray():
    objects,_,K,_=scene(0.,(0.,0.,0.))
    pixels,_=project(objects,OPTICAL_TO_FLU.T,np.zeros(3),K)
    features=pixels+[1.2,-.7]  # allowed neighbour association, not camera motion
    snapped=depth_on_feature_rays(objects,features,K)
    recovered,_=project(snapped,OPTICAL_TO_FLU.T,np.zeros(3),K)
    np.testing.assert_allclose(recovered,features,atol=1e-12)
    np.testing.assert_array_equal(snapped[:,0],objects[:,0])
    fit=gyro_translation_fit(snapped,features,K,.25,(480,640),np.eye(3))
    assert fit['accepted']
    np.testing.assert_allclose(np.asarray(fit['relative_previous_current'])[:3,3],0.,atol=1e-10)


def test_pixel_refinement_removes_depth_weighting_and_decreases_robust_cost():
    rng=np.random.default_rng(514);K=np.array([[400.,0.,320.],[0.,400.,240.],[0.,0.,1.]])
    xyz=rng.uniform([5.,-2.,-1.],[35.,2.,1.],(300,3));R=OPTICAL_TO_FLU.T
    true_t=np.array([.1,-.03,-1.5]);uv,_=project(xyz,R,true_t,K)
    uv+=rng.normal(0.,.2,uv.shape);uv[xyz[:,0]>25.]+=[.8,-.6]
    normalized=(uv-K[:2,2])/[K[0,0],K[1,1]]
    A=np.zeros((len(xyz),2,3));A[:,0,0]=1.;A[:,1,1]=1.;A[:,:,2]=-normalized
    q=xyz@R.T;b=normalized*q[:,2,None]-q[:,:2]
    algebraic=np.linalg.lstsq(A.reshape(-1,3),b.ravel(),rcond=None)[0]
    fitted,trace=refine_pixel_translation(xyz,uv,K,R,algebraic)
    assert trace and all(s['pixel_cost_after']<=s['pixel_cost_before'] for s in trace)
    before=np.linalg.norm(project(xyz,R,algebraic,K)[0]-uv,axis=1)
    after=np.linalg.norm(project(xyz,R,fitted,K)[0]-uv,axis=1)
    assert np.mean(after**2)<np.mean(before**2)
    assert np.linalg.norm(fitted-true_t)<np.linalg.norm(algebraic-true_t)


def test_final_gyro_consensus_is_recomputed_with_original_quality_threshold():
    objects,pixels,K,T=scene(.025,(2.,.15,.01))
    result=gyro_translation_fit(objects,pixels,K,.25,(480,640),T[:3,:3],gyro_variance=.00001)
    assert result['accepted']
    recovered=np.asarray(result['relative_previous_current']);R=OPTICAL_TO_FLU.T@recovered[:3,:3].T
    uv,depth=project(objects,R,-R@recovered[:3,3],K)
    count=np.count_nonzero((depth>0)&(np.linalg.norm(uv-pixels,axis=1)<=p.IMAGE_SPEED_REPROJECTION_PX))
    assert result['inliers']==count and count>=.6*len(objects)
    np.testing.assert_allclose(recovered[:3,3],T[:3,3],atol=.025)


def test_relative_speed_does_not_feed_shared_heading_as_sideways_velocity():
    state,T,d=sensor_pair();state.cov=np.diag([1.,1.,2.,2.])
    before=state.position.copy();heading=state.heading_covariance.copy()
    m=measurement('vector_velocity')
    out,diag,u=fuse_visual_speed(state,T,d,m,0.)
    assert u['filter_accepted'] and u['kind']=='speed' and u['relative_channel']=='vector_velocity'
    assert u['speed_mps']==pytest.approx(np.hypot(2.5,.25)/.25)
    assert state.position[3]==before[3] and out[1,3]==T[1,3]
    np.testing.assert_array_equal(state.heading_covariance,heading)
    P=np.asarray(u['covariance_before']);K=np.asarray(u['kalman_gain']);H=np.asarray(u['observation_matrix'])
    A=np.eye(4)-K@H
    np.testing.assert_allclose(state.cov,A@P@A.T+K@np.asarray(u['effective_measurement_covariance'])@K.T)
    with pytest.raises(ValueError,match='already consumed'):fuse_visual_speed(state,T,d,m,0.)


def test_speed_norm_is_invariant_to_common_absolute_heading_reference():
    values=[]
    for angle in [0.,.8,-2.7]:
        s,T,d=sensor_pair();s.visual_heading_anchor['camera_rotation']=Rotation.from_euler('z',angle).as_matrix()
        _,_,u=fuse_visual_speed(s,T,d,measurement('vector_velocity'),0.)
        values.append((u['speed_mps'],u['variance_m2ps2'],u['innovation']))
    assert values[0]==values[1]==values[2]


def test_rejected_speed_interval_is_not_reused_or_silently_accepted():
    s,T,d=sensor_pair();s.cov=np.eye(4)*.01;before=s.position.copy()
    m=measurement('vector_velocity');m['relative_previous_current'][0][3]=100.
    # Small uncertainty isolates the unchanged statistical gate.
    m['pose_covariance']=(np.eye(6)*1e-8).tolist()
    out,_,u=fuse_visual_speed(s,T,d,m,0.)
    assert not u['filter_accepted'] and u['filter_reason']=='visual_velocity_statistical_gate'
    np.testing.assert_array_equal(s.position,before);np.testing.assert_array_equal(out,T)
    assert s.visual_heading_anchor['used']


def test_road_fallback_is_one_alternative_observation_after_relative_rejection(tmp_path,monkeypatch):
    from track2.image_speed import CausalGroundSpeed
    from test_track2_image_speed import camera,project as ground_project
    import track2.image_speed as module
    monkeypatch.setattr(p,'IMAGE_SPEED_DOWNSAMPLE',1)
    obj=CausalGroundSpeed(tmp_path,camera());obj.folder.mkdir(parents=True)
    x,y=np.meshgrid(np.linspace(5.,30.,19),np.linspace(-8.,8.,25))
    xyz=np.c_[x.ravel(),y.ravel(),np.full(x.size,-1.6)]
    a,b=ground_project(xyz),ground_project(xyz-[2.,.1,0.])
    good=(a[:,0]>3)&(a[:,0]<637)&(b[:,0]>3)&(b[:,0]<637)&(a[:,1]<477)&(b[:,1]<477)
    a,b=a[good],b[good]
    monkeypatch.setattr(module,'track_images',lambda *args,**kw:(a,b))
    obj.anchor=dict(stamp=1_000_000_000,path='generated',sha256='generated',gray=np.zeros((480,640),np.uint8),
        ground=dict(mask=np.full((480,640),255,np.uint8),normal=np.array([0.,0.,1.]),height=1.6,height_variance=.0001))
    path=obj.folder/'generated_ts_1250000000.jpg';cv2.imwrite(str(path),np.zeros((480,640),np.uint8))
    monkeypatch.setattr(obj,'_relative_observation',lambda *args:dict(accepted=False,reason='pnp_outliers',kind='relative_yaw'))
    d=obj.observe(1_250_000_000,1_250_000_000,0.,rotation_previous_current=np.eye(3))
    assert d['accepted'] and d['kind']=='speed' and d['relative_channel']=='ground_speed_fallback'
    assert d['speed_mps']==pytest.approx(np.hypot(2.,.1)/.25,abs=.02)
    assert d['relative_attempt']['reason']=='pnp_outliers' and obj.interval_produced


def test_qualified_relative_candidate_does_not_also_run_road_fallback(tmp_path,monkeypatch):
    from track2.image_speed import CausalGroundSpeed
    from test_track2_image_speed import camera
    import track2.image_speed as module
    monkeypatch.setattr(p,'IMAGE_SPEED_DOWNSAMPLE',1)
    obj=CausalGroundSpeed(tmp_path,camera());obj.folder.mkdir(parents=True)
    obj.anchor=dict(stamp=1_000_000_000,path='generated',sha256='generated',gray=np.zeros((480,640),np.uint8),ground={})
    path=obj.folder/'generated_ts_1250000000.jpg';cv2.imwrite(str(path),np.zeros((480,640),np.uint8))
    a=np.random.default_rng(5).uniform([5.,5.],[630.,470.],(100,2))
    monkeypatch.setattr(module,'track_images',lambda *args,**kw:(a,a+5.))
    monkeypatch.setattr(obj,'_relative_observation',lambda *args:dict(accepted=True,reason='relative_geometry',kind='vector_velocity'))
    monkeypatch.setattr(obj,'_ground_observation',lambda *args:pytest.fail('interval used twice'))
    assert obj.observe(1_250_000_000,1_250_000_000,0.,rotation_previous_current=np.eye(3))['accepted']


def test_cuda_depth_tracking_to_scalar_fusion_preserves_feature_rays(tmp_path,monkeypatch):
    import cupy as cp
    import track2.image_speed as module
    from test_track2_v513_cuda import observer
    from test_track2_v513 import registration
    monkeypatch.setattr(p,'IMAGE_SPEED_DOWNSAMPLE',1)
    obj=observer(tmp_path)
    objects,_,K,T=scene(.025,(2.5,.15,.01))
    before,_=project(objects,OPTICAL_TO_FLU.T,np.zeros(3),K)
    before=before+[.8,-.5]  # keypoints and nearest PLY rays differ
    exact=depth_on_feature_rays(objects,before,K)
    R=OPTICAL_TO_FLU.T@T[:3,:3].T
    pixels,_=project(exact,R,-R@T[:3,3],K)
    obj.store(cp.asarray(objects/8.),8.,registration(),reference_rotation=np.eye(3))
    path=obj.folder/'generated_ts_1250000000.jpg';cv2.imwrite(str(path),np.zeros((480,640),np.uint8))
    monkeypatch.setattr(module,'track_images',lambda *args,**kw:(before,pixels))
    m=obj.observe(1_250_000_000,1_250_000_000,.026,rotation_previous_current=T[:3,:3])
    assert m['accepted'] and m['kind']=='vector_velocity'
    assert m['depth_association_rms_px']>0.
    np.testing.assert_allclose(np.asarray(m['relative_previous_current'])[:3,3],T[:3,3],atol=.025)
    state,prior,d=sensor_pair();_,_,fused=fuse_visual_speed(state,prior,d,m,0.)
    assert fused['filter_accepted'] and fused['kind']=='speed'
    assert state.visual_heading_anchor['used']
