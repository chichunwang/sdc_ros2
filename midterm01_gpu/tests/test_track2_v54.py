"""Generated v5.4 mechanisms; no real Track files, reference CSV or labels."""
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from track2.correction_filter import CausalCorrectionFilter, bounded_reference_sensitivity
from track2.observability import geometry_observation, likelihood_observation
from track2 import parameters as p


C = np.diag([.36, .36, np.deg2rad(10.)**2])
QUALITY = dict(coverage=.1, median_plane_residual_m=.05)


def test_reference_constraint_is_covariance_aware_and_not_an_arbitrary_xy_clip():
    S = np.array([[.4, .2, 12.], [-.1, .3, 5.], [.03, -.1, .8]])
    B, d = bounded_reference_sensitivity(S, C)
    L = np.linalg.cholesky(C)
    W = np.linalg.solve(L, B @ L)
    np.testing.assert_allclose(W, W.T, atol=1e-14)
    assert np.linalg.eigvalsh(W).min() > -1e-14
    assert np.linalg.eigvalsh(W).max() < 1+1e-14
    # A joint normalized covariance with this cross response is PSD.
    joint = np.block([[np.eye(3), W], [W.T, np.eye(3)]])
    assert np.linalg.eigvalsh(joint).min() > -1e-14
    units = np.diag([100., 100., 1.])
    other, _ = bounded_reference_sensitivity(units @ S @ np.linalg.inv(units), units @ C @ units.T)
    np.testing.assert_allclose(other, units @ B @ np.linalg.inv(units), atol=1e-11)
    assert d['normalized_projection_norm'] > 0
    # With no learned map response, a calibration change must pass through.
    zero, _ = bounded_reference_sensitivity(np.zeros((3, 3)), C)
    np.testing.assert_array_equal(zero, np.zeros((3, 3)))


def test_rotating_modes_and_changing_prior_never_accumulate_unbounded_rebase():
    f = CausalCorrectionFilter()
    maximum = 0.
    for k in range(180):
        covariance = C.copy(); covariance[2, 2] *= .3+np.cos(k*.03)**2
        f.predict(np.eye(4), 1_000_000_000+k*50_000_000, prior_covariance=covariance, rotation_delta_rad=0.)
        O = np.array([[np.cos(k*.11), np.sin(k*.11), 3*np.cos(k*.17)]])
        _, d = f.update([.1, -.05, .01], np.array([[.04]]), observation_matrix=O)
        L = np.linalg.cholesky(covariance)
        W = np.linalg.solve(L, f.reference_sensitivity @ L)
        maximum = max(maximum, np.linalg.norm(W, 2))
        assert np.linalg.eigvalsh(f.covariance).min() > 0
        assert np.linalg.norm(W-W.T) < 1e-12
    assert maximum <= 1+1e-12


def test_weak_positive_information_is_weighted_and_exact_null_stays_unsupported():
    D = np.diag([.2, 0., 1000., 100.])
    center = np.array([.4, 99., .01, 0.])
    m, O, R, d = likelihood_observation(D, -D @ center, np.eye(3, 4), C, QUALITY, np.zeros(3))
    assert d['rank'] == 2 and d['strong_rank'] == 1
    np.testing.assert_allclose(O[:, 1], 0., atol=1e-14)
    f = CausalCorrectionFilter(); f.predict(np.eye(4), 1_000_000_000, prior_covariance=C)
    out, diag = f.update(m, R, observation_matrix=O,
        memory_observation_matrix=d['strong_observation_matrix'], local_linear_likelihood=True)
    assert diag['accepted'] and 0 < out[0] < .1 and out[1] == 0.
    information = O.T @ np.linalg.solve(R, O)
    oracle = np.linalg.solve(np.linalg.inv(C)+information, information @ m)
    np.testing.assert_allclose(out, oracle, atol=1e-13)


def test_proposal_rejection_is_atomic_including_covariance_and_memory():
    f = CausalCorrectionFilter(); f.predict(np.eye(4), 1_000_000_000, prior_covariance=C)
    before = (f.mean.copy(), f.covariance.copy(), f.memory_age.copy(), f.reference_sensitivity.copy())
    out, d = f.update([.2, .1, .01], np.diag([.03, .03, .002]),
                     proposal_validator=lambda c: dict(accepted=False, reason='synthetic_nonlinear_loss'))
    assert not d['accepted'] and d['reason'] == 'synthetic_nonlinear_loss'
    for a, b in zip(before, [f.mean, f.covariance, f.memory_age, f.reference_sensitivity]):
        np.testing.assert_array_equal(a, b)
    assert f.last_update is None and f.last_measurement is None


def test_weak_yaw_quadratic_center_is_not_wrapped_as_a_physical_pose():
    D = np.diag([100., 100., .01, 100.])
    center = np.array([0., 0., 4., 0.])
    m, O, R, d = likelihood_observation(D, -D @ center, np.eye(3, 4), C, QUALITY, np.zeros(3))
    assert m[2] == pytest.approx(4.)
    f = CausalCorrectionFilter(); f.predict(np.eye(4), 1_000_000_000, prior_covariance=C)
    out, diag = f.update(m, R, observation_matrix=O,
        memory_observation_matrix=d['strong_observation_matrix'], local_linear_likelihood=True)
    assert diag['accepted'] and 0 < out[2] < .01
    json.dumps(diag, allow_nan=False)


@pytest.mark.parametrize('width,tilt,should_pass', [(6., 0., True), (.15, 0., False), (6., 15., False)])
def test_cuda_ground_shape_and_normal_gates(width, tilt, should_pass):
    import cupy as cp
    from track2.ground_support import checked_ground, GroundSupportError
    x, y = np.meshgrid(np.linspace(2., 10., 60), np.linspace(-width/2, width/2, 40))
    world = np.c_[x.ravel(), y.ravel(), np.full(x.size, -1.6)]
    R = Rotation.from_euler('x', np.deg2rad(tilt)).as_matrix()
    raw = world @ R.T/8.
    normal = R @ [0., 0., 1.]
    sd = dict(threshold=.015, mad=.001, inliers=len(raw))
    call = lambda: checked_ground(cp.asarray(raw), normal, .2, sd, np.array([0., 0., 1.]), np.eye(3), 8.)
    if should_pass:
        n, ds, mad, d = call()
        assert ds == pytest.approx(.2) and d['accepted']
        np.testing.assert_allclose(n, [0., 0., 1.], atol=1e-13)
    else:
        with pytest.raises(GroundSupportError) as error:
            call()
        assert not error.value.diagnostic['accepted']


def test_cuda_weak_ground_fallback_does_not_emit_rejected_tilt_or_new_scale():
    import cupy as cp
    from track2.metric_registration import MetricRegistration
    from test_track2_metric_scale import scene, truth, raw_cloud
    T = truth(); reg = MetricRegistration(cp.asarray(scene()), T)
    _, scale, good = reg.refine(cp.asarray(raw_cloud(scene(), T, 8.)), T, 1_000_000_000,
                               prior_covariance=C)
    assert good['accepted']
    # Narrow ground has many points but only one substantial in-plane axis.
    x, y = np.meshgrid(np.linspace(2., 10., 60), np.linspace(-.1, .1, 50))
    raw = np.c_[x.ravel(), y.ravel(), np.full(x.size, -1.6)]/8.
    out, after, bad = reg.refine(cp.asarray(raw), T, 1_050_000_000, prior_covariance=C)
    assert not bad['accepted'] and bad['reason'] == 'ground_support_failed'
    assert after == scale and not bad['scale_validated']
    assert bad['ground_support']['inplane_ratio'] < p.GROUND_MIN_INPLANE_RATIO
    # Only prior rotation plus learned world-yaw correction can remain.
    expected = Rotation.from_euler('z', reg.correction_filter.mean[2]).as_matrix() @ T[:3, :3]
    np.testing.assert_allclose(out[:3, :3], expected, atol=1e-13)


def test_cuda_weak_center_outside_search_can_have_a_valid_small_posterior(monkeypatch):
    import cupy as cp
    import track2.metric_registration as module
    from test_track2_metric_scale import scene, truth, raw_cloud
    T = truth(); prior = T.copy(); prior[0, 3] += .2
    original = module.likelihood_observation
    def weak_far_center(*args, **kwargs):
        m, O, R, d = original(*args, **kwargs)
        m[0] = -5.
        return m, O, R*1000., d
    monkeypatch.setattr(module, 'likelihood_observation', weak_far_center)
    reg = module.MetricRegistration(cp.asarray(scene()), T)
    out, scale, diag = reg.refine(cp.asarray(raw_cloud(scene(), T, 8.)), prior, 1_000_000_000,
                                 prior_covariance=C)
    assert diag['geometry_accepted'] and diag['accepted'], diag
    assert not diag['likelihood_observation']['within_existing_search_bounds']
    check = diag['correction_filter']['proposal_validation']
    assert check['accepted'] and check['proposal_data_cost'] <= check['baseline_data_cost']
    assert abs(out[0, 3]-T[0, 3]) < abs(prior[0, 3]-T[0, 3])

@pytest.fixture(autouse=True)
def historical_model(monkeypatch):
    """Isolate retained v5.3-v5.5 experiments from production v5.6."""
    monkeypatch.setattr(p, 'FUSION_PROFILE', 'v55_experimental')
