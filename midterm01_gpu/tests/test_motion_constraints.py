"""Generated-only invariants: rest/release, scale units, rigid solve and causality."""
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from track3.motion_constraints import RestDetector, scale_from_pairs, track_images
from track3.motion_parameters import PARAMETERS
from track3.inertial import InertialState
from track3.imu import ImuSample

STAMP=1727239790258245070
CFG=json.loads((Path(__file__).resolve().parents[1]/'config/track3.json').read_text())['imu']
# Synthetic measurements below use identity IMU axes, independent of user RPY.
CFG.update(extrinsic_source='config', T_ply_imu=np.eye(4).tolist(),
           extrinsic_status='synthetic_identity_fixture')


def imu(t,w=(0,0,0),a=(0,0,9.80665)):
    return ImuSample(t,t-100,np.asarray(w,dtype=float),np.asarray(a,dtype=float),'imu_link')


def state():
    s=InertialState(np.eye(4),7.5,CFG)
    s.predict(STAMP,[imu(STAMP)])
    s.commit(dict(T=np.eye(4),scale=7.5,accepted=True,reason='accepted'))
    return s


def test_rest_needs_images_duration_and_gyro_and_releases_on_turn_or_translation():
    d=RestDetector(PARAMETERS);quiet=dict(median_px=.1,p90_px=.2)
    assert not d.update(STAMP+100_000_000,STAMP,quiet,quiet,.001,True)
    assert d.update(STAMP+400_000_000,STAMP+300_000_000,quiet,quiet,.001,True)
    assert not d.update(STAMP+500_000_000,STAMP+400_000_000,quiet,quiet,.2,True)
    moving=dict(median_px=2.,p90_px=4.)
    assert not d.update(STAMP+600_000_000,STAMP+500_000_000,moving,moving,.001,True)
    assert not d.update(STAMP+700_000_000,STAMP+600_000_000,None,None,.001,True)
    assert not d.update(STAMP+800_000_000,STAMP+700_000_000,quiet,quiet,.001,False)


def test_rest_cannot_silently_follow_slow_accumulated_motion():
    d=RestDetector(PARAMETERS);q=dict(median_px=.1,p90_px=.2)
    assert not d.update(STAMP+100_000_000,STAMP,q,q,.001,True)
    # Even tiny frame-to-frame flow is rejected once anchor disparity grows.
    assert not d.update(STAMP+400_000_000,STAMP+300_000_000,q,dict(median_px=1.,p90_px=2.),.001,True)


def test_pair_scale_is_inverse_of_raw_shrink_and_invariant_to_rotation_translation():
    rng=np.random.default_rng(55);p=rng.uniform([-2,-4,-1],[6,4,2],(220,3))
    R=Rotation.from_euler('xyz',[.2,-.1,.3]).as_matrix()
    current=.8*(p@R.T+np.array([1.,2.,-.4]))
    ratio,diag=scale_from_pairs(p,current,PARAMETERS)
    assert ratio==pytest.approx(1.25,abs=1e-12)
    assert diag['reason']=='causal_3d_pair_lengths'
    # The product, not the numerical scale value alone, preserves metric size.
    assert 7.5*ratio*.8==pytest.approx(7.5)


def test_nonuniform_reconstruction_does_not_silently_change_scale():
    r=np.random.default_rng(51);p=r.normal(size=(150,3));q=p*r.uniform(.4,2.,size=(150,1))
    ratio,diag=scale_from_pairs(p,q,PARAMETERS)
    assert ratio is None and diag['reason']=='nonuniform_reconstruction_or_dynamic_tracks'


def test_static_constraint_holds_pose_and_zero_velocity_but_tracks_input_units():
    s=state();anchor=s.T.copy()
    for i in range(1,20):
        t=STAMP+i*50_000_000
        prior=s.predict(t,[imu(t,a=(.15,0,9.80665))],input_scale=7.5/(1-.01*i),stationary=True)
        candidate=np.eye(4);candidate[:3,3]=[10,5,2]
        d=s.commit(dict(T=candidate,scale=3.,accepted=True,reason='accepted'))
        np.testing.assert_array_equal(prior,anchor);np.testing.assert_array_equal(s.T,anchor)
        np.testing.assert_array_equal(s.v,np.zeros(3))
        assert s.scale==pytest.approx(7.5/(1-.01*i))
        assert not d['accepted'] and d['decision']=='stationary_pose_hold'
        assert np.linalg.eigvalsh(s.P).min()>-1e-10
    t=STAMP+1_000_000_000
    s.predict(t,[imu(t,w=(0,0,.2))],stationary=False)
    s.commit(dict(T=s.T.copy(),scale=s.scale,accepted=False,reason='no_map'))
    # With a held zero-velocity sample at the previous timestamp, angular rate
    # starts at the newly arrived sample and acts on the next interval.
    s.predict(t+50_000_000,[imu(t+50_000_000,w=(0,0,.2))])
    assert Rotation.from_matrix(s.T[:3,:3]).magnitude()>0.005


def test_visual_translation_moves_prior_during_turn_even_if_map_is_rejected():
    s=state();relative=np.eye(4)
    relative[:3,:3]=Rotation.from_rotvec([0,0,.01]).as_matrix();relative[0,3]=.1
    for i in range(1,15):
        t=STAMP+i*50_000_000
        prior=s.predict(t,[imu(t,w=(0,0,.2))],visual_motion=relative)
        d=s.commit(dict(T=np.eye(4),scale=7.5,accepted=False,reason='no_map'))
        assert d['inertial']['prediction']['visual_velocity_update']['accepted']
    assert np.linalg.norm(s.T[:3,3])>1.0
    assert Rotation.from_matrix(s.T[:3,:3]).magnitude()>.05
    assert np.linalg.norm(s.v)>1.0


def test_stationary_never_consumes_future_imu_and_input_scale_is_not_map_candidate():
    s=state();before=s.T.copy()
    with pytest.raises(ValueError):s.predict(STAMP+10,[imu(STAMP+11)],stationary=True)
    np.testing.assert_array_equal(before,s.T)
    t=STAMP+50_000_000
    s.predict(t,[imu(t)],input_scale=8.)
    s.commit(dict(T=np.eye(4),scale=30.,accepted=False,reason='invalid'))
    assert s.scale==8.


def test_rigid_registration_does_not_optimize_scale():
    from test_core import CFG,cloud
    from track3.gpu import Registration
    cfg=json.loads(json.dumps(CFG['registration']))
    cfg['stages']=[dict(mode='ndt',voxel_m=2.,iterations=30),dict(mode='icp',voxel_m=1.,iterations=40)]
    T=np.eye(4);T[:3,3]=[.1,-.1,.05]
    m=cloud();result=Registration(m,cfg).align(m/7.5,T,7.5,fixed_scale=True)
    assert result['accepted'] and result['scale']==7.5 and not result['scale_optimized']
    assert np.linalg.norm(result['T'][:3,3])<.05


def test_generated_images_depth_scale_and_turn_release(tmp_path):
    import cv2
    from types import SimpleNamespace
    from track3.motion_constraints import MotionConstraints
    images=tmp_path/'Track3_images/Track3_f_images';images.mkdir(parents=True)
    h,w=480,640;rng=np.random.default_rng(46)
    gray=rng.integers(0,256,size=(h,w),dtype=np.uint8)
    gray=cv2.GaussianBlur(gray,(5,5),.8)
    for i in range(7):
        current=gray if i<6 else cv2.warpAffine(gray,np.float32([[1,0,8],[0,1,0]]),(w,h))
        cv2.imwrite(str(images/f'{i:06d}_ts_{STAMP+i*100000000}.jpg'),current)
    u,v=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
    camera=SimpleNamespace(K=np.array([[400.,0,320],[0,400.,240],[0,0,1.]]),
                           maps=(u,v),mask=np.full((h,w),255,dtype='uint8'))
    u,v=np.meshgrid(np.arange(0,w,4),np.arange(0,h,4))
    depth=10+.001*u+.002*v
    points=np.column_stack([depth.ravel(),((320-u)/400*depth).ravel(),((240-v)/400*depth).ravel()])/7.5
    monitor=MotionConstraints(tmp_path,camera,7);scale=7.5
    stationary=[];saved_scales=[]
    for i in range(7):
        factor=1-.04*i
        proposed,rest,diag=monitor.observe(STAMP+i*100_000_000,points*factor,scale,
                                           [imu(STAMP+i*100_000_000,w=(0,0,.2 if i==6 else 0))],i>0)
        if proposed is not None:scale=proposed
        monitor.store(scale,anchored=True,bootstrap=i==0)
        stationary.append(rest)
        saved_scales.append(scale)
        if 0<i<6:assert scale==pytest.approx(7.5/factor,rel=.02)
    assert any(stationary[3:6])
    assert stationary[6] is False  # Both visual motion and turn gyro release rest.
    # Change a future image, then recompute an earlier prefix. No earlier
    # decision or scale may depend on that image's contents.
    cv2.imwrite(str(images/f'{6:06d}_ts_{STAMP+600000000}.jpg'),np.zeros((h,w),dtype='uint8'))
    prefix=MotionConstraints(tmp_path,camera,7);prefix_scale=7.5
    for i in range(5):
        value,rest,_=prefix.observe(STAMP+i*100_000_000,points*(1-.04*i),prefix_scale,
                                    [imu(STAMP+i*100_000_000)],i>0)
        if value is not None:prefix_scale=value
        prefix.store(prefix_scale,anchored=True,bootstrap=i==0)
        assert prefix_scale==pytest.approx(saved_scales[i],abs=1e-12)
        assert rest==stationary[i]


def test_visual_velocity_uses_map_axes_and_metric_imu_lever_arm():
    cfg=json.loads(json.dumps(CFG));E=np.eye(4);E[0,3]=1.;cfg['T_ply_imu']=E.tolist()
    T=np.eye(4);T[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix()
    s=InertialState(T,7.5,cfg);old=s.X.copy()
    motion=np.eye(4);motion[:3,:3]=Rotation.from_euler('z',.02).as_matrix();motion[0,3]=.2
    s.X[:3,:3]=T[:3,:3]@motion[:3,:3]@E[:3,:3]
    result=s._visual_velocity_update(motion,old,.1)
    expected=T[:3,:3]@(np.array([.2,0,0])+(motion[:3,:3]-np.eye(3))@np.array([1.,0,0]))/.1
    np.testing.assert_allclose(result['velocity_map_mps'],expected,atol=1e-13)


def test_body_velocity_global_left_orientation_jacobian():
    from track3.inertial import skew
    R=Rotation.from_euler('xyz',[.2,-.3,.6]).as_matrix();v=np.array([3.,-1.,.4])
    H=R.T@skew(v);eps=1e-6
    numerical=np.empty((3,3))
    for j in range(3):
        d=np.eye(3)[j]*eps
        plus=(Rotation.from_rotvec(d).as_matrix()@R).T@v
        minus=(Rotation.from_rotvec(-d).as_matrix()@R).T@v
        numerical[:,j]=(plus-minus)/(2*eps)
    np.testing.assert_allclose(H,numerical,atol=1e-9)
