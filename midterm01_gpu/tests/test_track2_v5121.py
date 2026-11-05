"""Generated CV/likelihood and lateral handoff checks; no real localization."""
import numpy as np
import pytest
from track2 import parameters as p
from track2.causal_prior import CausalSensorPrior
from track2.correction_filter import CausalCorrectionFilter
from track2.gnss_budget import fractional_position_update


@pytest.fixture(autouse=True)
def current_model(monkeypatch):
    monkeypatch.setattr(p,'V510_ENABLED',True)
    monkeypatch.setattr(p,'V511_ENABLED',False)
    monkeypatch.setattr(p,'V512_ENABLED',False)
    monkeypatch.setattr(p,'V5121_ENABLED',True)
    monkeypatch.setattr(p,'V513_ENABLED',False)


def position_step(x,P,z,*,fraction=1.,age=0.,radial=None,lateral=None):
    return fractional_position_update(x,P,z,variance=.09,age_seconds=age,
        remaining=fraction,acceleration_psd=0.,body_rotation=np.eye(2),
        radial_limit=radial,lateral_limit=lateral)


def test_fractional_likelihood_powers_equal_one_observation_not_repeated_data():
    x=np.zeros(4);P=np.diag([2.,3.,4.,5.]);z=[.7,-.4]
    expected,expected_cov,_=position_step(x,P,z)
    used=0.
    for _ in range(20):
        x,P,d=position_step(x,P,z,fraction=.05);used+=d['information_fraction']
    assert used==pytest.approx(1.)
    np.testing.assert_allclose(x,expected,atol=1e-14)
    np.testing.assert_allclose(P,expected_cov,atol=1e-14)


def test_deferred_row_observes_past_position_not_a_stale_current_target():
    x=np.array([10.,0.,10.,0.]);P=np.eye(4)
    after,cov,d=position_step(x,P,[0.,0.],age=1.)
    np.testing.assert_array_equal(after,x)
    np.testing.assert_array_equal(d['observation_matrix'],np.c_[np.eye(2),-np.eye(2)])
    assert np.linalg.eigvalsh(cov).min()>0


def test_bounded_factor_uses_its_actual_information_gain_in_joseph_covariance():
    x=np.zeros(4);P=np.array([[2.,.1,.2,0.],[.1,3.,0.,.3],[.2,0.,1.,0.],[0.,.3,0.,1.]])
    after,cov,d=position_step(x,P,[.6,-.45],radial=.02,lateral=.008)
    assert 0<d['information_fraction']<1
    assert np.linalg.norm(after[:2])<=.02+1e-14 and abs(after[1])<=.008+1e-14
    K=np.asarray(d['kalman_gain']);H=np.asarray(d['observation_matrix'])
    A=np.eye(4)-K@H;R=np.asarray(d['effective_measurement_covariance'])
    np.testing.assert_allclose(after,x+K@np.asarray(d['innovation']),atol=1e-14)
    np.testing.assert_allclose(cov,A@P@A.T+K@R@K.T,atol=1e-14)
    assert np.linalg.eigvalsh(cov).min()>0


def gnss(s,stamp,xy):
    return s.event(dict(kind='gnss',timestamp_ns=stamp,xy=xy,z=1.6))


def test_delayed_first_gnss_cannot_bypass_frame_budget_or_claim_full_information():
    s=CausalSensorPrior(np.eye(4),1_000_000_000);s.for_frame(s.time)
    decision=gnss(s,1_020_000_000,[.5,-.45])
    np.testing.assert_array_equal(s.position,np.zeros(4))
    assert decision['reason']=='arrived_gnss_position_likelihood_queued'
    out,d=s.for_frame(1_040_000_000)
    assert np.linalg.norm(out[:2,3])<=.02+1e-12 and abs(out[1,3])<=.008+1e-12
    update=d['position_information_updates'][0]
    assert 0<update['information_fraction']<1 and s.pending_position_information
    np.testing.assert_allclose(d['gnss_state_innovation_xy'],out[:2,3])


def test_initial_common_xy_power_preserves_innovation_direction_without_an_axis_corner():
    s=CausalSensorPrior(np.eye(4),1_000_000_000);s.for_frame(s.time)
    target=np.array([.5,-.45])
    gnss(s,1_020_000_000,target)
    for i in range(1,8):
        out,d=s.for_frame(1_000_000_000+i*40_000_000)
        assert abs(out[0,3]*target[1]-out[1,3]*target[0])<1e-12
        assert all(u['coupled_startup_xy'] for u in d['position_information_updates'])


def test_multiple_position_factors_share_one_camera_frame_budget():
    s=CausalSensorPrior(np.eye(4),1_000_000_000);s.for_frame(s.time)
    gnss(s,1_010_000_000,[.5,.4]);gnss(s,1_020_000_000,[.6,.5])
    out,d=s.for_frame(1_040_000_000)
    changes=[np.asarray(x['state_increment']) for x in d['position_information_updates']]
    assert sum(np.linalg.norm(x[:2]) for x in changes)<=.02+1e-12
    assert sum(abs(x[1]) for x in changes)<=.008+1e-12


def test_true_initial_frame_has_no_artificial_zero_time_gnss_limit():
    s=CausalSensorPrior(np.eye(4),1_000_000_000)
    gnss(s,s.time,[.5,-.45]);out,d=s.for_frame(s.time)
    assert np.linalg.norm(out[:2,3])>.6
    assert d['position_information_updates'][0]['bootstrap_unbounded']
    assert d['position_information_updates'][0]['information_fraction']==1.


def test_old_unconsumed_factor_expires_without_claiming_its_covariance_information():
    s=CausalSensorPrior(np.eye(4),1_000_000_000);s.for_frame(s.time)
    gnss(s,1_010_000_000,[.5,.4]);s.for_frame(1_040_000_000)
    before=s.position.copy();s.advance(2_000_000_000);prediction=s.position.copy();P=s.cov.copy()
    _,d=s.for_frame(s.time)
    np.testing.assert_array_equal(s.position,prediction);np.testing.assert_array_equal(s.cov,P)
    assert not s.pending_position_information
    assert d['position_information_updates'][0]['expired_information_fraction']>0


def test_position_factor_consumption_never_exceeds_one_and_outputs_are_causal():
    def run(count):
        s=CausalSensorPrior(np.eye(4),1_000_000_000);out=[];consumed={}
        for i in range(count):
            stamp=1_000_000_000+i*40_000_000
            if i%6==2:gnss(s,stamp,[.4+i*.01,-.3])
            T,d=s.for_frame(stamp);out.append(T.copy())
            for u in d['position_information_updates']:
                if 'information_fraction' in u:
                    key=u['position_factor_id'];consumed[key]=consumed.get(key,0.)+u['information_fraction']
        assert all(0<=v<=1+1e-12 for v in consumed.values())
        return out
    short=run(20);long=run(40)
    for a,b in zip(short,long):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('H',[np.eye(3),np.array([[0.,1.,1.]])])
def test_production_map_gain_respects_body_lateral_budget_and_data_row_space(H):
    f=CausalCorrectionFilter();C=np.diag([1.,1.,.01]);prior=np.eye(4)
    f.predict(prior,1_000_000_000,15.,prior_covariance=C,rotation_delta_rad=0.)
    f.update(np.zeros(3),np.eye(len(H))*.01,observation_matrix=H)
    f.predict(prior,1_040_000_000,15.,prior_covariance=C,rotation_delta_rad=0.)
    out,d=f.update(np.array([.1,.1,.01]),np.eye(len(H))*.01,observation_matrix=H)
    assert d['accepted']
    assert abs(out[1])<=d['lateral_update_limit_m']+1e-12
    assert np.linalg.norm(out[:2])<=d['position_update_limit_m']+1e-12
    K=np.asarray(d['kalman_gain']);P=np.asarray(d['covariance_predicted']);A=np.eye(3)-K@H
    R=np.asarray(d['measurement_covariance'])/d['measurement_weight']
    np.testing.assert_allclose(f.covariance,A@P@A.T+K@R@K.T,atol=1e-12)
    if len(H)<3:
        L=np.linalg.cholesky(C);_,_,Vt=np.linalg.svd(H@L,full_matrices=False)
        projector=L@Vt.T@Vt@np.linalg.inv(L)
        np.testing.assert_allclose(projector@out,out,atol=1e-12)


def test_late_map_anchor_is_bounded_even_when_v511_and_v512_are_disabled():
    f=CausalCorrectionFilter();C=np.diag([100.,100.,.03])
    f.predict(np.eye(4),1_000_000_000,0.,prior_covariance=C,rotation_delta_rad=0.);f.update()
    f.predict(np.eye(4),1_040_000_000,0.,prior_covariance=C,rotation_delta_rad=0.)
    out,d=f.update(np.array([-.5,.4,.01]),np.diag([.01,.01,.001]))
    assert d['accepted'] and d['delayed_first_anchor_budget_applied']
    assert abs(out[1])<=.008+1e-12
    assert np.linalg.norm(out[:2])<=.02+1e-12


def test_reference_yaw_cross_response_cannot_be_an_unbounded_oblique_projector():
    f=CausalCorrectionFilter();C=np.diag([.4,.4,.03])
    f.predict(np.eye(4),1_000_000_000,0.,prior_covariance=C,rotation_delta_rad=0.);f.update()
    f.reference_sensitivity=np.array([[1.,0.,-12.],[0.,1.,15.],[0.,0.,0.]])
    f.predict(np.eye(4),1_040_000_000,0.,prior_covariance=C,rotation_delta_rad=0.,reference_delta=[0.,0.,.005])
    L=np.linalg.cholesky(C);S=np.linalg.solve(L,f.reference_rebase_sensitivity@L)
    np.testing.assert_allclose(S,S.T,atol=1e-12)
    assert np.linalg.eigvalsh(S).min()>=-1e-12 and np.linalg.eigvalsh(S).max()<=1+1e-12
    f.update()


def test_production_parent_is_v510_not_v511_or_v512():
    assert p.uses_v52_fusion() and p.v510_heading_enabled() and p.v5121_enabled()
    assert not p.V511_ENABLED and not p.v512_enabled()


def test_orthogonal_axis_factors_are_exactly_one_original_isotropic_xy_likelihood():
    x=np.array([.2,-.1,.3,-.2]);G=np.arange(16,dtype=float).reshape(4,4)/20
    P=G@G.T+np.eye(4);z=[.8,-.5]
    expected,expected_cov,_=position_step(x,P,z)
    angle=.7;axes=[[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]]
    for axis in axes:
        x,P,_=fractional_position_update(x,P,z,variance=.09,age_seconds=0.,
            remaining=1.,acceleration_psd=0.,body_rotation=np.eye(2),direction=axis)
    np.testing.assert_allclose(x,expected,atol=1e-14)
    np.testing.assert_allclose(P,expected_cov,atol=1e-14)


@pytest.mark.parametrize('trajectory',['accelerating_with_heading_error','constant_speed_turn'])
def test_gnss_budget_preserves_kinematic_information_under_acceleration_and_turns(trajectory):
    from scipy.spatial.transform import Rotation
    s=CausalSensorPrior(np.eye(4),1_000_000_000);errors=[]
    for i in range(201):
        t=i*.05;stamp=1_000_000_000+i*50_000_000
        if trajectory=='constant_speed_turn':
            omega=.15;radius=100.
            xy=np.array([radius*np.sin(omega*t),radius*(1-np.cos(omega*t))])
            heading=omega*t
        else:
            omega=0.;xy=np.array([max(0.,t-1.)**2,0.]);heading=np.deg2rad(15.)
        s.event(dict(kind='imu',timestamp_ns=stamp,quaternion=Rotation.from_euler('z',heading).as_quat().tolist(),gyro=[0.,0.,omega]))
        if i%5==0:gnss(s,stamp,xy)
        T,d=s.for_frame(stamp)
        if i>=20:errors.append(np.linalg.norm(T[:2,3]-xy))
        assert np.linalg.eigvalsh(s.cov).min()>0
    assert np.sqrt(np.mean(np.square(errors)))<p.GNSS_STD_M
