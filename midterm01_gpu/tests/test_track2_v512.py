"""Generated Gaussian and surface-alias checks; CPU small matrices only."""
import numpy as np
import pytest
from track2 import parameters as p
from track2.observability import likelihood_observation
from track2.correction_filter import CausalCorrectionFilter
from track2.posterior_validation import make_proposal_validator


@pytest.mark.parametrize('variance', [1., 4., 9.])
def test_gaussian_map_prior_is_used_once_instead_of_squaring_its_gain(monkeypatch, variance):
    monkeypatch.setattr(p, 'CORRECTION_MEAS_POSITION_FLOOR_M', 0.)
    monkeypatch.setattr(p, 'CORRECTION_MEAS_YAW_FLOOR_DEG', 0.)
    C = np.eye(3)*variance; data_H = np.eye(4)
    truth = np.array([.8, -.4, .02]); gain = variance/(variance+1.)
    regularized = gain*truth
    gradient = np.r_[regularized-truth, 0.]
    J = np.c_[np.eye(3), np.zeros(3)]
    measurement, O, R, d = likelihood_observation(data_H, gradient, J, C,
        dict(coverage=1., median_plane_residual_m=0.), regularized)
    np.testing.assert_allclose(measurement, truth, atol=1e-12)
    new = CausalCorrectionFilter(); old = CausalCorrectionFilter()
    for f in [new, old]:f.predict(np.eye(4), 1_000_000_000, 0., prior_covariance=C)
    new_mean, _ = new.update(measurement, R, observation_matrix=O,
        candidate_sensitivity=np.eye(3), local_linear_likelihood=True)
    old_mean, _ = old.update(regularized, R, observation_matrix=O)
    np.testing.assert_allclose(new_mean, gain*truth, atol=1e-12)
    np.testing.assert_allclose(old_mean, gain**2*truth, atol=1e-12)
    assert np.linalg.norm(new_mean-truth) < np.linalg.norm(old_mean-truth)


def test_scale_nuisance_and_pose_cross_terms_are_profiled_from_data_gradient():
    G = np.array([[4.,1.,2.,1.], [0.,3.,1.,-2.], [0.,0.,4.,1.], [0.,0.,0.,3.]])
    H = G.T @ G; target = np.array([.7,-.2,.03,.04])
    regularizer = np.diag([1.,2.,3.,8.])
    candidate = np.linalg.solve(H+regularizer, H @ target)
    measurement, O, _, diag = likelihood_observation(H, H @ (candidate-target),
        np.c_[np.eye(3), np.zeros(3)], np.eye(3),
        dict(coverage=.3, median_plane_residual_m=.1), candidate[:3])
    np.testing.assert_allclose(O @ measurement, O @ target[:3], atol=1e-12)
    assert candidate[3]+diag['likelihood_nuisance_logscale_step'] == pytest.approx(target[3])


def test_rank_one_data_does_not_invent_yaw_or_unobserved_position():
    H = np.diag([4.,0.,0.,2.]); c = np.array([.4,0.,0.])
    measurement, O, R, d = likelihood_observation(H, np.array([-1.6,0.,0.,0.]),
        np.c_[np.eye(3), np.zeros(3)], np.eye(3),
        dict(coverage=.1, median_plane_residual_m=.1), c)
    assert d['rank'] == 1 and O.shape == (1,3)
    f = CausalCorrectionFilter(); f.predict(np.eye(4),1_000_000_000,0.,prior_covariance=np.eye(3))
    out, diag = f.update(measurement,R,observation_matrix=O,
        candidate_sensitivity=np.eye(3),local_linear_likelihood=True)
    assert diag['accepted'] and out[0] > 0.
    np.testing.assert_array_equal(out[1:],np.zeros(2))


def validator(*, rematch):
    # Two parallel surfaces: the prediction is on A=0. The candidate's final
    # association points to B=.5. A partial update towards B reduces B's fixed
    # residual while making the actual nearest-surface residual worse.
    fixed = lambda x:(float(x[0])-.5)**2
    common = lambda x:min(float(x[0])**2,(float(x[0])-.5)**2)
    return make_proposal_validator(state_for_proposal=lambda x:np.asarray(x),
        valid=lambda x:np.isfinite(x).all() and abs(x[0])<3.,
        fixed_cost=fixed,baseline_fixed_cost=fixed([0.]),
        common_cost=common if rematch else None,
        baseline_common_cost=common([0.]) if rematch else None)


def test_parallel_surface_alias_cannot_hide_behind_fixed_plane_improvement():
    assert validator(rematch=False)([.1])['accepted']
    checked = validator(rematch=True)([.1])
    assert checked['fixed_plane_loss_valid']
    assert not checked['accepted'] and not checked['rematched_loss_valid']
    assert checked['reason']=='posterior_rematched_feature_loss_increased'


def test_consistent_actual_posterior_improves_both_checks():
    cost=lambda x:(float(x[0])-.5)**2
    v=make_proposal_validator(state_for_proposal=lambda x:x,valid=lambda x:True,
        fixed_cost=cost,baseline_fixed_cost=cost([0.]),
        common_cost=cost,baseline_common_cost=cost([0.]))
    d=v([.1]);assert d['accepted'] and d['fixed_plane_loss_valid'] and d['rematched_loss_valid']


def test_out_of_search_posterior_never_evaluates_point_cost_callbacks():
    def forbidden(x):raise AssertionError('Cost called outside allowed search')
    v=make_proposal_validator(state_for_proposal=lambda x:x,valid=lambda x:False,
        fixed_cost=forbidden,baseline_fixed_cost=1.,common_cost=forbidden,baseline_common_cost=1.)
    d=v([99.]);assert not d['accepted'] and d['reason']=='posterior_outside_search'


def test_child_model_is_disabled_with_parent_v511(monkeypatch):
    monkeypatch.setattr(p,'V511_ENABLED',False)
    assert not p.v512_enabled()


def test_nonfinite_cost_is_rejected_without_breaking_finite_json_contract():
    import json
    v=make_proposal_validator(state_for_proposal=lambda x:x,valid=lambda x:True,
        fixed_cost=lambda x:np.nan,baseline_fixed_cost=1.)
    d=v([0.]);assert not d['accepted'] and d['proposal_data_cost'] is None
    assert d['reason']=='posterior_nonfinite_fixed_plane_cost'
    json.dumps(d,allow_nan=False)


@pytest.fixture(autouse=True)
def historical_model_profile(monkeypatch):
    monkeypatch.setattr(p, 'V511_ENABLED', True)
    monkeypatch.setattr(p, 'V5121_ENABLED', False)
    monkeypatch.setattr(p, 'V513_ENABLED', False)
    monkeypatch.setattr(p, 'V512_ENABLED', True)
