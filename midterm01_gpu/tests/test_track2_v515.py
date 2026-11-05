"""v5.10-based map likelihood, weak modes and actual-gain validation.

Generated matrices/scenes only; no Track inputs or leaderboard labels.
"""
import numpy as np
import pytest
from track2 import parameters as p
from track2.correction_filter import CausalCorrectionFilter
from track2.observability import geometry_observation,likelihood_observation
from track2.posterior_validation import make_proposal_validator

QUALITY=dict(coverage=.2,median_plane_residual_m=.05)


@pytest.fixture(autouse=True)
def production(monkeypatch):
    monkeypatch.setattr(p,'FUSION_PROFILE','v52_directional_memory')
    monkeypatch.setattr(p,'V510_ENABLED',True)
    for name in ['V511_ENABLED','V512_ENABLED','V5121_ENABLED','V513_ENABLED','V514_ENABLED']:
        monkeypatch.setattr(p,name,False)
    monkeypatch.setattr(p,'V515_ENABLED',True)


def filter_at_start(C=None):
    f=CausalCorrectionFilter();f.predict(np.eye(4),1_000_000_000,0.,prior_covariance=np.eye(3) if C is None else C)
    return f


def test_positive_weak_data_is_weighted_and_exact_null_is_unobserved():
    C=np.diag([.36,.36,np.deg2rad(10.)**2]);H=np.diag([.2,0.,1000.,100.])
    target=np.array([.4,99.,.01,0.])
    m,O,R,d=likelihood_observation(H,-H@target,np.eye(3,4),C,QUALITY,np.zeros(3))
    assert d['rank']==2 and d['strong_rank']==1
    np.testing.assert_allclose(O[:,1],0.,atol=1e-14)
    f=filter_at_start(C)
    out,u=f.update(m,R,observation_matrix=O,memory_observation_matrix=d['strong_observation_matrix'],local_linear_likelihood=True)
    info=O.T@np.linalg.solve(R,O)
    expected=np.linalg.solve(np.linalg.inv(C)+info,info@m)
    np.testing.assert_allclose(out,expected,atol=1e-13)
    assert u['accepted'] and 0.<out[0]<.1 and out[1]==0.


@pytest.mark.parametrize('variance',[1.,4.,9.])
def test_regularized_search_candidate_does_not_square_the_prior_gain(monkeypatch,variance):
    monkeypatch.setattr(p,'CORRECTION_MEAS_POSITION_FLOOR_M',0.)
    monkeypatch.setattr(p,'CORRECTION_MEAS_YAW_FLOOR_DEG',0.)
    C=np.eye(3)*variance;data_H=np.eye(4);truth=np.array([.8,-.4,.02])
    gain=variance/(variance+1.);regularized=gain*truth
    m,O,R,d=likelihood_observation(data_H,np.r_[regularized-truth,0.],np.eye(3,4),C,
        dict(coverage=1.,median_plane_residual_m=0.),regularized)
    f=filter_at_start(C);out,u=f.update(m,R,observation_matrix=O,local_linear_likelihood=True)
    np.testing.assert_allclose(out,gain*truth,atol=1e-12)
    assert np.linalg.norm(out-truth)<np.linalg.norm(gain**2*truth-truth)


def test_weak_only_information_does_not_refresh_strong_memory_age():
    f=CausalCorrectionFilter();f.predict(np.eye(4),1_000_000_000,0.,prior_covariance=np.eye(3));f.update()
    f.predict(np.eye(4),1_100_000_000,0.,prior_covariance=np.eye(3));age=f.memory_age.copy()
    H=np.diag([.2,.1,.05,100.]);m,O,R,d=likelihood_observation(H,-H@np.array([.1,-.1,.01,0.]),np.eye(3,4),np.eye(3),QUALITY,np.zeros(3))
    assert d['rank']==3 and d['strong_rank']==0
    _,u=f.update(m,R,observation_matrix=O,memory_observation_matrix=d['strong_observation_matrix'],local_linear_likelihood=True)
    assert u['accepted'];np.testing.assert_allclose(f.memory_age,age,atol=1e-14)
    np.testing.assert_array_equal(f.last_observation_projector,np.zeros((3,3)))


def test_scale_pose_gauge_does_not_become_an_exact_translation_measurement():
    row=np.array([1.,0.,0.,2.]);H=np.outer(row,row)
    O,R,d=geometry_observation(H,np.eye(3,4),np.eye(3),QUALITY)
    assert O.shape==(0,3) and R.shape==(0,0) and d['rank']==0


def test_local_weak_yaw_center_is_not_wrapped_before_fusion():
    H=np.diag([100.,100.,.01,100.]);target=np.array([0.,0.,4.,0.])
    m,O,R,d=likelihood_observation(H,-H@target,np.eye(3,4),np.eye(3),QUALITY,np.zeros(3))
    assert m[2]==pytest.approx(4.)
    f=filter_at_start();out,u=f.update(m,R,observation_matrix=O,local_linear_likelihood=True)
    assert u['accepted'] and 0.<out[2]<.05


def validator(cost,common=None):
    return make_proposal_validator(state_for_proposal=lambda x:x,
        valid=lambda x:np.isfinite(x).all() and np.linalg.norm(x[:2])<3.,
        fixed_cost=cost,baseline_fixed_cost=cost(np.zeros(3)),
        common_cost=common or cost,baseline_common_cost=(common or cost)(np.zeros(3)))


def test_backtracked_actual_gain_matches_joseph_covariance_and_sensitivity():
    f=filter_at_start();before=f.covariance.copy();cost=lambda x:(float(x[0])-.1)**2
    out,u=f.update([2.,0.,0.],np.eye(3)*.01,proposal_validator=validator(cost),
        candidate_sensitivity=np.eye(3),local_linear_likelihood=True)
    assert u['accepted'] and u['nonlinear_gain_fraction']==pytest.approx(1/16)
    assert cost(out)<=cost(np.zeros(3))
    K=np.asarray(u['kalman_gain']);A=np.eye(3)-K
    np.testing.assert_allclose(out,K@np.array([2.,0.,0.]),atol=1e-14)
    np.testing.assert_allclose(f.covariance,A@before@A.T+K@np.eye(3)*.01@K.T,atol=1e-14)
    np.testing.assert_allclose(f.reference_sensitivity,K,atol=1e-14)
    assert len(u['proposal_backtracking']['attempts'])==5


def test_all_failed_posterior_trials_leave_filter_commit_atomic():
    f=filter_at_start();before=[x.copy() for x in [f.mean,f.covariance,f.memory_age,f.reference_sensitivity]]
    _,u=f.update([2.,0.,0.],np.eye(3)*.01,proposal_validator=validator(lambda x:(float(x[0])+.2)**2))
    assert not u['accepted'] and not u['proposal_backtracking']['accepted']
    for actual,expected in zip([f.mean,f.covariance,f.memory_age,f.reference_sensitivity],before):
        np.testing.assert_array_equal(actual,expected)
    assert f.last_update is None and f.last_measurement is None


def test_parallel_surface_alias_is_not_rescued_by_a_smaller_gain():
    f=filter_at_start();fixed=lambda x:(float(x[0])-.5)**2
    common=lambda x:min(float(x[0])**2,(float(x[0])-.5)**2)
    _,u=f.update([.5,0.,0.],np.eye(3)*.01,proposal_validator=validator(fixed,common))
    assert not u['accepted']
    assert all(not trial['rematched_loss_valid'] for trial in u['proposal_backtracking']['attempts'])


def test_history_cannot_change_an_already_emitted_prefix():
    def sequence(count):
        f=CausalCorrectionFilter();rows=[]
        for k in range(count):
            f.predict(np.eye(4),1_000_000_000+k*50_000_000,0.,prior_covariance=np.eye(3))
            cost=lambda x:(float(x[0])-.1)**2
            out,u=f.update([.8,0.,0.],np.eye(3)*.1,proposal_validator=validator(cost),local_linear_likelihood=True)
            rows.append((out.tolist(),f.covariance.tolist(),u))
        return rows
    assert sequence(9)==sequence(21)[:9]


def test_cuda_production_map_likelihood_validates_actual_emitted_pose():
    import cupy as cp
    from track2.metric_registration import MetricRegistration
    from test_track2_metric_scale import scene,truth,raw_cloud,TEST_COV
    T=truth();prior=T.copy();prior[:2,3]+=[.2,-.15];reg=MetricRegistration(cp.asarray(scene()),T)
    out,scale,d=reg.refine(cp.asarray(raw_cloud(scene(),T,8.)),prior,1_000_000_000,prior_covariance=TEST_COV)
    assert d['accepted'] and d['geometry_accepted']
    assert d['map_measurement_model']=='current_data_likelihood_prior_used_once'
    u=d['correction_filter'];v=u['proposal_validation']
    assert u['local_linear_likelihood'] and u['proposal_backtracking']['accepted']
    assert v['proposal_data_cost']<=v['baseline_data_cost']+1e-12
    assert v['proposal_common_feature_cost']<=v['baseline_common_feature_cost']+1e-12
    assert d['observability']['rank']>=d['observability']['strong_rank']
    assert np.isfinite(out).all() and np.isfinite(scale)


def test_generated_image_runner_preserves_csv_txt_and_causal_prefix(tmp_path,monkeypatch):
    from test_track2_image_speed import test_image_enabled_runner_prefix_and_measurement_output
    test_image_enabled_runner_prefix_and_measurement_output(tmp_path,monkeypatch)
