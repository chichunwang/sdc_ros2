"""Generated counterexamples for v8's pose/scale and stop/release failures."""
import json
from pathlib import Path
import numpy as np
import cupy as cp
import pytest
from scipy.spatial.transform import Rotation
from test_monte_carlo import STAMP,state,anchored,sample,estimate,registration_map


def test_rejected_candidate_scale_never_changes_public_scale_or_next_search_center():
    s=anchored(state(16));m,reg=registration_map()
    for i in range(1,21):
        stamp=STAMP+i*36_000_000
        prior=s.predict(stamp,[sample(stamp)])
        s.logs[:]=cp.log(1.5);s.p[:]=0.;s.R[:]=cp.eye(3)
        s.observe_cloud(m/3.,reg)
        assert s.scale==3.  # v8 overwrote this with its selected random scale.
        d=s.commit(estimate(accepted=False,scale=1.5))
        assert not d['accepted'] and s.scale==3.
        np.testing.assert_array_equal(s.T,prior)
        assert float(cp.linalg.norm(s.anchor_p))==0.
    s.predict(STAMP+21*36_000_000,[sample(STAMP+21*36_000_000)])
    assert float(cp.exp(s.logs[0]))==pytest.approx(3.)


def test_long_straight_stop_resume_uses_committed_anchor_and_never_old_particles(tmp_path):
    s=anchored(state(32));records=[];truth=np.eye(4)
    for i in range(1,121):
        stamp=STAMP+i*36_000_000;rest=31<=i<=90
        motion=None
        if not rest:motion=np.eye(4);motion[0,3]=.12
        prior=s.predict(stamp,[sample(stamp)],stationary=rest,visual_motion=motion)
        expected=truth.copy()
        if not rest:expected[0,3]+=.12
        np.testing.assert_allclose(prior,expected,atol=1e-12)
        # A small geometric correction must be the next frame's anchor.
        refined=expected.copy()
        if not rest:refined[1,3]+=.003
        d=s.commit(estimate(refined,scale=3.))
        truth=expected if rest else refined
        np.testing.assert_allclose(s.T,truth,atol=1e-12)
        np.testing.assert_allclose(cp.asnumpy(s.anchor_p),s.T[:3,3],atol=0,rtol=0)
        assert s.scale==3. and d['monte_carlo']['next_prediction_matches_output']
        records.append(dict(idx=i,stationary=rest,expected=truth[:3,3].tolist(),emitted=s.T[:3,3].tolist(),scale=s.scale))
    (tmp_path/'stop_resume_evidence.json').write_text(json.dumps(dict(kind='synthetic_state_mechanism_with_injected_geometric_measurements_not_real_localization',frames=records),indent=2)+'\n')


@pytest.mark.parametrize('kind',['translation','rotation'])
def test_implausible_registration_cannot_be_committed(kind):
    s=anchored(state());t=STAMP+36_000_000;prior=s.predict(t,[sample(t)])
    T=np.eye(4)
    if kind=='translation':T[0,3]=10.56443
    else:T[:3,:3]=Rotation.from_rotvec([0,0,1.6]).as_matrix()
    d=s.commit(estimate(T,scale=1.))
    assert not d['accepted'] and d['reason']=='map_motion_limit'
    assert d['monte_carlo']['motion_gate']['checked']
    np.testing.assert_array_equal(s.T,prior);assert s.scale==3.


def test_invalid_scale_never_leaks_through_accepted_registration():
    s=anchored(state());t=STAMP+36_000_000;prior=s.predict(t,[sample(t)])
    d=s.commit(estimate(scale=float('nan')))
    assert not d['accepted'];assert s.scale==3.;np.testing.assert_array_equal(s.T,prior)


def test_registration_sampling_uses_fixed_reference_not_selected_scale():
    from test_core import CFG,cloud
    from track3.gpu import Registration
    cfg=json.loads(json.dumps(CFG['registration']));cfg['stages']=[dict(mode='icp',voxel_m=1.,iterations=1)]
    m=cloud();reg=Registration(m,cfg)
    a=reg.align(m/3.,np.eye(4),3.,scale_reference=3.)
    b=reg.align(m/3.,np.eye(4),1.5,scale_reference=3.)
    assert a['stages'][0]['source_points']==b['stages'][0]['source_points']


def test_scale_regularizer_is_anchored_to_causal_scale_not_random_seed():
    from test_core import CFG,cloud
    from track3.gpu import Registration
    cfg=json.loads(json.dumps(CFG['registration']))
    cfg['stages']=[dict(mode='ndt',voxel_m=2.,iterations=40),dict(mode='icp',voxel_m=1.,iterations=40)]
    m=cloud();result=Registration(m,cfg).align(m/3.,np.eye(4),2.85,scale_reference=3.)
    assert result['accepted']
    assert abs(result['scale']-3.)<.04


def test_pnp_is_evaluated_when_preliminary_rest_is_rejected_by_depth(tmp_path):
    import cv2
    from types import SimpleNamespace
    from track3.motion_constraints import MotionConstraints
    from test_motion_constraints import imu
    images=tmp_path/'Track3_images/Track3_f_images';images.mkdir(parents=True)
    h,w=480,640;rng=np.random.default_rng(46)
    gray=cv2.GaussianBlur(rng.integers(0,256,size=(h,w),dtype=np.uint8),(5,5),.8)
    for i in range(2):cv2.imwrite(str(images/f'{i:06d}_ts_{STAMP+i*100000000}.jpg'),gray)
    u,v=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
    camera=SimpleNamespace(K=np.array([[400.,0,320],[0,400.,240],[0,0,1.]]),maps=(u,v),mask=np.full((h,w),255,dtype='uint8'))
    u,v=np.meshgrid(np.arange(0,w,4),np.arange(0,h,4));depth=10+.001*u+.002*v
    points=np.column_stack([depth.ravel(),((320-u)/400*depth).ravel(),((240-v)/400*depth).ravel()])/7.5
    mon=MotionConstraints(tmp_path,camera,2)
    mon.observe(STAMP,points,7.5,[imu(STAMP)],False);mon.store(7.5,bootstrap=True)
    mon.rest.since=STAMP-1_000_000_000
    inconsistent=points*rng.uniform(.3,2.,size=(len(points),1))
    _,rest,diag=mon.observe(STAMP+100_000_000,inconsistent,7.5,[imu(STAMP+100_000_000)],True)
    assert not rest and diag['stationary_rejection']=='insufficient_reliable_nearby_3d_support'
    assert diag['visual_velocity']['reason']!='stationary'
    assert diag['visual_velocity']['accepted'] and mon.visual_motion is not None


def test_generated_geometry_straight_stop_scale_change_and_turn_resume(tmp_path):
    from test_core import CFG,cloud
    from track3.gpu import Registration
    s=state(128);cfg=json.loads(json.dumps(CFG['registration']))
    cfg['stages']=[dict(mode='ndt',voxel_m=2.,iterations=40),dict(mode='icp',voxel_m=1.,iterations=40)]
    m=cloud();reg=Registration(m,cfg);truth=np.eye(4);rows=[]
    for i in range(25):
        rest=9<=i<=16;moving=i>0 and not rest;turn=i>=17
        relative=np.eye(4)
        if moving:relative[0,3]=.08
        if turn:relative[:3,:3]=Rotation.from_rotvec([0,0,.015]).as_matrix()
        truth=truth@relative
        true_scale=3.*(1+.02*min(max(i-8,0),8))
        stamp=STAMP+i*100_000_000
        prior=s.predict(stamp,[sample(stamp,(0.,0.,.15 if turn else 0.))],
                        stationary=rest,input_scale=true_scale if rest else None,
                        visual_motion=relative if moving else None)
        source=(m-truth[:3,3])@truth[:3,:3]/true_scale
        seed=s.observe_cloud(source,reg)
        result=reg.align(source,seed,s.registration_scale,initialize=not s.initialized,
                         fixed_scale=rest,scale_reference=s.scale)
        d=s.commit(result)
        error=float(np.linalg.norm(s.T[:3,3]-truth[:3,3]))
        angle=float(Rotation.from_matrix(s.T[:3,:3]@truth[:3,:3].T).magnitude())
        assert error<.2 and angle<.03
        if moving or i==0:assert d['accepted'],(i,d['reason'])
        if rest:assert d['decision']=='stationary_pose_hold'
        rows.append(dict(idx=i,timestamp_ns=stamp,stationary=rest,true_T=truth.tolist(),predicted_T=prior.tolist(),final_T=s.T.tolist(),true_scale=true_scale,scale=s.scale,position_error_m=error,rotation_error_rad=angle,map_accepted=d['accepted'],cloud=d['monte_carlo']['cloud']))
    (tmp_path/'geometry_stop_resume_evidence.json').write_text(json.dumps(dict(kind='generated_map_registration_not_real_Track3',frames=rows),indent=2)+'\n')
