"""Generated causal motion and symmetric line/plane geometry, no real data."""
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from track2 import parameters as p
from track2.causal_prior import CausalSensorPrior
from track2.correction_filter import CausalCorrectionFilter

C=np.diag([.36,.36,np.deg2rad(10.)**2])

def test_post_emission_heading_gain_cap_preserves_actual_joint_joseph():
    s=CausalSensorPrior(np.eye(4),1_000_000_000)
    s.event(dict(kind='imu',timestamp_ns=s.time,quaternion=[0.,0.,0.,1.],gyro=[0.,0.,0.]))
    s.for_frame(s.time)
    P=s.heading_covariance.copy();before=s.heading_bias.copy()
    d=s._heading_update(np.deg2rad(8.),[1.,0.],np.deg2rad(1.)**2,course_rate_limit=True)
    assert abs(s.bias-before[0])==pytest.approx(np.deg2rad(p.HEADING_MAX_UPDATE_DEG))
    H=np.array(d['observation_matrix']);K=np.array(d['heading_gain']);A=np.eye(2)-np.outer(K,H)
    np.testing.assert_allclose(s.heading_covariance,A@P@A.T+np.outer(K,K)*d['measurement_variance'],atol=1e-15)
    assert d['bias_gain']<d['unconstrained_bias_gain']
    frozen=s.bias
    s._heading_update(np.deg2rad(-8.),[1.,-1.],np.deg2rad(1.)**2)
    assert s.bias==frozen  # Pair and course cannot each spend the frame budget.


def test_split_heading_likelihood_conserves_total_information_and_matches_single_static_update(monkeypatch):
    monkeypatch.setattr(p,'HEADING_BIAS_RW_DEG_SQRT_S',0.)
    monkeypatch.setattr(p,'GNSS_HEADING_BIAS_RW_DEG_SQRT_S',0.)
    s=CausalSensorPrior(np.eye(4),1_000_000_000)
    s.orientation_stamp=s.time;s.for_frame(s.time)
    P=s.heading_covariance.copy();H=np.array([1.,0.]);z=np.deg2rad(4.);R=np.deg2rad(2.)**2
    d=s._heading_update(z,H,R)
    fractions=[d['information_fraction']];saved=[]
    for k in range(1,30):
        s.event(dict(kind='imu',timestamp_ns=1_000_000_000+k*50_000_000,quaternion=[0.,0.,0.,1.],gyro=[0.,0.,0.]))
        T,diag=s.for_frame(s.time);saved.append(T.copy())
        fractions.extend(v.get('information_fraction',0.) for v in diag['heading_information_updates'])
        assert diag['heading_update_budget_used_deg']<=p.HEADING_MAX_UPDATE_DEG+1e-12
    assert sum(fractions)==pytest.approx(1.,abs=1e-12) and not s.pending_heading_information
    K=P@H/(H@P@H+R);A=np.eye(2)-np.outer(K,H)
    np.testing.assert_allclose(s.heading_bias,K*z,atol=1e-12)
    np.testing.assert_allclose(s.heading_covariance,A@P@A.T+np.outer(K,K)*R,atol=1e-13)
    frozen=saved[0].copy();s._heading_update(-z,H,R)
    np.testing.assert_array_equal(saved[0],frozen)


def test_bounded_course_initialization_eventually_converges_with_fresh_past_windows():
    s=CausalSensorPrior(np.eye(4),1_000_000_000);angle=np.deg2rad(3.);previous=0.
    for k in range(601):
        stamp=1_000_000_000+k*50_000_000
        s.event(dict(kind='imu',timestamp_ns=stamp,quaternion=[0.,0.,0.,1.],gyro=[0.,0.,0.]))
        if k%5==0:
            s.event(dict(kind='gnss',timestamp_ns=stamp,xy=(k*.05*2*np.array([np.cos(angle),np.sin(angle)])).tolist(),z=1.6))
        _,d=s.for_frame(stamp)
        assert abs(s.bias-previous)<=np.deg2rad(p.HEADING_MAX_UPDATE_DEG)+1e-14
        previous=s.bias
    assert s.bias==pytest.approx(angle,abs=np.deg2rad(.2))
    assert s.accepted_blocks>1


def test_velocity_constraint_preserves_prefix_and_improves_generated_noisy_straight_motion(monkeypatch):
    def simulate(n,enabled):
        monkeypatch.setattr(p,'LATERAL_VELOCITY_CONSTRAINT_ENABLED',enabled)
        noise=np.random.default_rng(912).normal(0,p.GNSS_STD_M,(300,2))
        s=CausalSensorPrior(np.eye(4),1_000_000_000);out=[]
        for k in range(n):
            stamp=1_000_000_000+k*50_000_000
            s.event(dict(kind='imu',timestamp_ns=stamp,quaternion=[0.,0.,0.,1.],gyro=[0.,0.,0.]))
            if k%5==0:
                xy=np.array([8*k*.05,0.])+noise[k//5]
                s.event(dict(kind='gnss',timestamp_ns=stamp,xy=xy.tolist(),z=1.6))
            T,d=s.for_frame(stamp);out.append((T.copy(),json.dumps(d,sort_keys=True,allow_nan=False)))
        return out
    full=simulate(500,True);short=simulate(180,True);old=simulate(500,False)
    for a,b in zip(short,full):np.testing.assert_array_equal(a[0],b[0]);assert a[1]==b[1]
    new_error=np.sqrt(np.mean([T[1,3]**2 for T,_ in full[100:]]))
    old_error=np.sqrt(np.mean([T[1,3]**2 for T,_ in old[100:]]))
    assert new_error<old_error


def constrained(gyro=0.,bias_std=1.,enabled=True):
    s=CausalSensorPrior(np.eye(4),1_000_000_000)
    s.orientation_stamp=s.time;s.gyro=np.array([0.,0.,gyro]);s.bias_var=np.deg2rad(bias_std)**2
    s.position=np.array([0.,0.,8.,1.]);s.cov=np.diag([.2,.2,.16,.16]);s.gnss_updates=1
    before=s.cov.copy();s._lateral_velocity_constraint(Rotation.identity())
    return s,before


def test_lateral_velocity_model_uses_uncertainty_and_once_per_new_gnss():
    s,P=constrained();d=s.lateral_constraint_diagnostic
    assert d['accepted'] and 0<s.position[3]<1.
    K=np.array(d['gain']);H=np.array([0.,0.,0.,1.]);A=np.eye(4)-np.outer(K,H)
    np.testing.assert_allclose(s.cov,A@P@A.T+np.outer(K,K)*d['variance_m2ps2'],atol=1e-14)
    unchanged=s.position.copy();s._lateral_velocity_constraint(Rotation.identity())
    np.testing.assert_array_equal(s.position,unchanged)
    assert s.lateral_constraint_diagnostic['reason']=='no_new_gnss'
    turn,_=constrained(gyro=.8);uncertain,_=constrained(bias_std=10.)
    assert turn.lateral_constraint_diagnostic['variance_m2ps2']>d['variance_m2ps2']
    assert turn.position[3]>s.position[3] and uncertain.position[3]>s.position[3]


def test_high_speed_does_not_open_lateral_correction_budget():
    T=np.eye(4);angle=.7;T[:3,:3]=Rotation.from_euler('z',angle).as_matrix()
    body=T[:3,:3];R=np.diag([.02,.02,.001]);result=[]
    for speed in [0.,15.]:
        f=CausalCorrectionFilter();f.predict(T,1_000_000_000,prior_covariance=C);f.update([0.,0.,0.],R)
        f.predict(T,1_050_000_000,speed,prior_covariance=C,rotation_delta_rad=0.)
        value,d=f.update(body@[.1,.1,0.],R)
        local=body.T@value
        assert abs(local[1])<=d['lateral_update_limit_m']+1e-12
        K=np.array(d['kalman_gain']);P=np.array(d['covariance_predicted']);A=np.eye(3)-K
        np.testing.assert_allclose(f.covariance,A@P@A.T+K@R@K.T,atol=1e-13)
        result.append((local,d))
    assert result[0][1]['lateral_update_limit_m']==result[1][1]['lateral_update_limit_m']
    assert result[1][1]['longitudinal_update_limit_m']>result[0][1]['longitudinal_update_limit_m']


def line_scene(horizontal=False):
    x,y=np.meshgrid(np.arange(2.,22.,.3),np.arange(-8.,8.,.3))
    ground=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    poles=[]
    for a,b in [(5.,-3.),(9.,4.),(14.,-2.),(18.,5.)]:
        z=np.arange(.3,4.6,.1)
        points=np.c_[np.full(len(z),a),np.full(len(z),b),z]
        if horizontal:points=np.c_[a+z,np.full(len(z),b),np.full(len(z),1.5)]
        poles.append(points)
    return np.concatenate([ground]+poles)


def test_cuda_vertical_lines_add_longitudinal_information_and_recover_generated_pose(monkeypatch):
    import cupy as cp
    from track2.metric_registration import MetricRegistration
    from test_track2_metric_scale import truth,raw_cloud
    xyz=line_scene();T=truth();prior=T.copy();prior[:2,3]+=[.25,-.18]
    reg=MetricRegistration(cp.asarray(xyz),T)
    raw=cp.asarray(raw_cloud(xyz,T,8.))
    pose,scale,d=reg.refine(raw,prior,1_000_000_000,prior_covariance=C)
    assert d['planes']['source_line_features']>0 and d['line_matches']>=20,d
    assert d['accepted'],d
    assert np.linalg.norm(pose[:2,3]-T[:2,3])<np.linalg.norm(prior[:2,3]-T[:2,3])*.5
    assert np.linalg.eigvalsh(np.array(d['observability']['pose_information_normalized'])).min()>0
    monkeypatch.setattr(p,'LINE_FEATURES_ENABLED',False)
    baseline=MetricRegistration(cp.asarray(xyz),T)
    _,_,old=baseline.refine(raw,prior,1_000_000_000,prior_covariance=C)
    assert old['line_matches']==0 and not old['accepted']


def test_cuda_line_loss_gradient_and_unobserved_line_axis():
    import cupy as cp
    from track2.metric_registration import MetricRegistration
    from test_track2_metric_scale import truth,raw_cloud
    xyz=line_scene();T=truth();reg=MetricRegistration(cp.asarray(xyz),T)
    reg.prepare(cp.asarray(raw_cloud(xyz,T,8.)),T,C)
    x=reg.seed+np.array([.02,-.01,.001,.002]);e=reg.evaluate(x,.75,.3,derivatives=True,include_prior=False)
    assert e['line_matches']>0
    gradient=[]
    for i in range(4):
        delta=np.eye(4)[i]*1e-6
        a=reg.evaluate(x+delta,.75,.3,include_prior=False,association=e['association'])['cost']
        b=reg.evaluate(x-delta,.75,.3,include_prior=False,association=e['association'])['cost']
        gradient.append((a-b)/2e-6)
    np.testing.assert_allclose(e['g'],gradient,rtol=2e-5,atol=2e-5)
    a=e['association'];mask=a['line_matches'];P=a['projector'][mask]
    directions=reg.geometry.line_direction[cp.maximum(a['ids'][mask],0)]
    np.testing.assert_allclose(cp.asnumpy(cp.einsum('nij,nj->ni',P,directions)),0.,atol=1e-12)
    reg.source_line_direction[:]=cp.array([1.,0.,0.])
    assert reg.evaluate(x,.75,.3)['line_matches']==0


def test_cuda_horizontal_rails_do_not_become_vertical_landmarks():
    import cupy as cp
    from track2.metric_registration import MetricRegistration
    from test_track2_metric_scale import truth,raw_cloud
    xyz=line_scene(horizontal=True);T=truth();reg=MetricRegistration(cp.asarray(xyz),T)
    reg.prepare(cp.asarray(raw_cloud(xyz,T,8.)),T,C)
    assert reg.evaluate(reg.seed,.75,.3)['line_matches']==0

@pytest.fixture(autouse=True)
def historical_model(monkeypatch):
    """Isolate retained v5.3-v5.5 experiments from production v5.6."""
    monkeypatch.setattr(p, 'FUSION_PROFILE', 'v55_experimental')
