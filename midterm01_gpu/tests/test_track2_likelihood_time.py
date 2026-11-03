"""生成資料：不重複 prior、GNSS 狀態重表達與時間契約。"""
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from track2.observability import likelihood_observation
from track2.causal_prior import CausalSensorPrior
from track2.correction_filter import CausalCorrectionFilter

QUALITY=dict(coverage=.8,median_plane_residual_m=.05)


def test_likelihood_recovers_same_data_measurement_from_different_regularized_candidates():
    # Analytic linear generative experiment, including pose/scale correlation.
    A=np.array([[5.,1.,0.,2.],[0.,4.,1.,-1.],[1.,0.,8.,1.],[0.,1.,0.,6.]])
    D=A.T@A;truth=np.array([.2,-.1,.015,.03]);C=np.diag([.3,.4,.02])
    O=R=measured=None
    for strength in (.1,1.,20.):
        regularizer=np.diag([1.,2.,20.,4.])*strength
        x=np.linalg.solve(D+regularizer,D@truth)
        # Even a MAP estimate far from the data optimum must yield the SAME
        # independent local measurement. The old scheme returned x[:3].
        measured,O,R,d=likelihood_observation(D,D@(x-truth),np.eye(3,4),C,QUALITY,x[:3])
        assert len(O)==3
        np.testing.assert_allclose(measured,truth[:3],atol=2e-15)
        assert x[3]+d['likelihood_nuisance_logscale_step'] == pytest.approx(truth[3], abs=2e-15)
        assert np.linalg.norm(x[:3]-truth[:3])>.0001
    f=CausalCorrectionFilter();f.predict(np.eye(4),1_000_000_000,prior_covariance=C)
    fused,diag=f.update(measured,R,observation_matrix=O)
    exact=np.linalg.solve(np.linalg.inv(C)+O.T@np.linalg.solve(R,O),O.T@np.linalg.solve(R,O@measured))
    np.testing.assert_allclose(fused,exact,atol=1e-13)


def test_likelihood_marginalizes_scale_and_preserves_null_and_units():
    D=np.array([[100.,0.,0.,100.],[0.,50.,0.,0.],[0.,0.,2000.,0.],[100.,0.,0.,100.]])
    x=np.array([.2,-.1,.01,.3]);z=np.array([.3,.15,.03,.2]);C=np.diag([.3,.3,.03])
    m,O,R,d=likelihood_observation(D,D@(x-z),np.eye(3,4),C,QUALITY,x[:3])
    assert len(O)==2
    np.testing.assert_allclose(m[1:],z[1:3],atol=1e-13)
    assert m[0]==pytest.approx(x[0]) # scale gauge: no fake x information
    B=np.array([[1.,.1,0.,.2],[0.,1.,0.,0.],[0.,0.,2.,0.],[0.,0.,0.,1.]])
    other,_,_,_=likelihood_observation(B.T@D@B,B.T@D@(x-z),np.eye(3,4)@B,C,QUALITY,x[:3])
    np.testing.assert_allclose(other,m,atol=1e-13)
    empty,OO,RR,dd=likelihood_observation(np.zeros((4,4)),np.zeros(4),np.eye(3,4),C,QUALITY,x[:3])
    assert len(OO)==0;np.testing.assert_array_equal(empty,x[:3])


def imu(stamp,yaw=0.):
    return dict(kind='imu',timestamp_ns=stamp,quaternion=Rotation.from_euler('z',yaw).as_quat().tolist(),gyro=[0.,0.,0.])


def test_gnss_state_innovation_propagates_once_but_is_not_calibration():
    s=CausalSensorPrior(np.eye(4),1_000_000_000)
    s.event(imu(1_000_000_000));s.for_frame(1_000_000_000)
    before=s.position.copy();s.event(imu(1_050_000_000))
    s.event(dict(kind='gnss',timestamp_ns=1_050_000_000,arrival_ns=1_052_000_000,xy=[2.,-.4],z=1.6))
    delta=s.gnss_reference_shift.copy()
    s.event(imu(1_100_000_000));_,d=s.for_frame(1_100_000_000)
    np.testing.assert_allclose(d['gnss_position_reference_shift_xy'],delta[:2]+.05*delta[2:],atol=1e-14)
    assert d['pose_timestamp_ns']==1_100_000_000 and d['gnss_position_age_s']==pytest.approx(.05)
    np.testing.assert_array_equal(d['calibration_delta_xy_yaw'],[0.,0.,0.])
    np.testing.assert_array_equal(d['gnss_state_innovation_xy'],d['gnss_position_reference_shift_xy'])
    _,n=s.for_frame(1_120_000_000)
    np.testing.assert_array_equal(n['gnss_position_reference_shift_xy'],[0.,0.])


def test_true_calibration_reference_reexpresses_world_anchored_map_memory():
    f=CausalCorrectionFilter();T=np.eye(4);C=np.diag([1.,1.,.02])
    f.predict(T,1_000_000_000,prior_covariance=C)
    old,d=f.update([.1,0.,0.],np.diag([.01,.01,.01]))
    sensitivity=f.reference_sensitivity.copy();jump=np.array([.3,-.2,0.])
    shifted=T.copy();shifted[:2,3]+=jump[:2]
    f.predict(shifted,1_050_000_000,prior_covariance=C,reference_delta=jump,speed_mps=0.,rotation_delta_rad=0.)
    new,d=f.update(None)
    np.testing.assert_allclose(shifted[:2,3]+new[:2],old[:2]+((np.eye(3)-sensitivity)@jump)[:2],atol=1e-14)
    assert d['motion']['distance_m']==0.


def test_heading_epoch_variance_uses_only_past_rotation_between_value_changes():
    def run(n):
        s=CausalSensorPrior(np.eye(4),1_000_000_000);out=[]
        for k in range(n):
            t=1_000_000_000+k*50_000_000;s.event(imu(t,.01*k))
            if k%5==0:
                angle=0. if k<20 else .2
                s.event(dict(kind='gnss',timestamp_ns=t,xy=[0.,0.],z=1.6,heading_quaternion=Rotation.from_euler('z',angle).as_quat().tolist()))
            T,d=s.for_frame(t);out.append((T,json.dumps(d,sort_keys=True)))
        return s,out
    s,short=run(24);_,long=run(40)
    for a,b in zip(short,long):np.testing.assert_array_equal(a[0],b[0]);assert a[1]==b[1]
    d=s.gnss_heading_decision
    assert s.gnss_heading_updates==2
    assert d['heading_epoch_uncertainty_variance']>0
    assert d['physical_heading_delay_s'] is None and not d['fixed_time_shift_applied']


def test_cuda_saved_data_gradient_excludes_sensor_and_height_priors(monkeypatch):
    import cupy as cp
    from track2.metric_registration import MetricRegistration
    from test_track2_metric_scale import scene,truth,raw_cloud,TEST_COV
    T=truth();reg=MetricRegistration(cp.asarray(scene()),T)
    reg.prepare(cp.asarray(raw_cloud(scene(),T,8.)),T,TEST_COV)
    x=reg.seed+np.array([.1,-.06,.001,.003])
    whole=reg.evaluate(x,.75,.3,derivatives=True)
    data=reg.evaluate(x,.75,.3,derivatives=True,include_prior=False,association=whole['association'])
    np.testing.assert_array_equal(whole['data_g'],data['g'])
    np.testing.assert_array_equal(whole['data_H'],data['H'])
    assert np.linalg.norm(whole['g']-whole['data_g'])>0
    # An incompatible strong center must not escape into the emitted pose;
    # v5.4 checks statistical compatibility then the actual finite proposal.
    import track2.metric_registration as module
    original=module.likelihood_observation
    def outside(*args,**kwargs):
        m,O,R,d=original(*args,**kwargs);m[0]+=100.
        return m,O,R,d
    monkeypatch.setattr(module,'likelihood_observation',outside)
    out,scale,diag=reg.refine(cp.asarray(raw_cloud(scene(),T,8.)),T,1_000_000_000,prior_covariance=TEST_COV)
    assert diag['geometry_accepted'] and not diag['accepted']
    assert diag['correction_filter']['reason']=='statistical_innovation_rejected'
    assert not diag['likelihood_observation']['within_existing_search_bounds']
    np.testing.assert_allclose(out[:2,3],T[:2,3],atol=1e-12)


def test_cuda_rejects_likelihood_center_inside_bounds_when_actual_map_loss_increases(monkeypatch):
    import cupy as cp
    import track2.metric_registration as module
    from test_track2_metric_scale import scene,truth,raw_cloud,TEST_COV
    T=truth();reg=module.MetricRegistration(cp.asarray(scene()),T)
    original=module.likelihood_observation
    def nonlinear_bad_center(*args,**kwargs):
        m,O,R,d=original(*args,**kwargs)
        m[0]+=.3  # inside the original 3m search bound, away from actual planes
        return m,O,R,d
    monkeypatch.setattr(module,'likelihood_observation',nonlinear_bad_center)
    out,scale,diag=reg.refine(cp.asarray(raw_cloud(scene(),T,8.)),T,1_000_000_000,prior_covariance=TEST_COV)
    assert diag['geometry_accepted'] and not diag['accepted']
    likelihood=diag['likelihood_observation']
    assert likelihood['within_existing_search_bounds']
    assert not likelihood['fixed_association_loss_valid']
    assert likelihood['fixed_association_data_cost'] > likelihood['candidate_data_cost']
    assert diag['correction_filter']['reason']=='posterior_map_loss_increased'
    assert not diag['correction_filter']['proposal_validation']['accepted']
    np.testing.assert_allclose(out[:2,3],T[:2,3],atol=1e-12)

from track2 import parameters as p

@pytest.fixture(autouse=True)
def historical_model(monkeypatch):
    """Isolate retained v5.3-v5.5 experiments from production v5.6."""
    monkeypatch.setattr(p, 'FUSION_PROFILE', 'v55_experimental')
