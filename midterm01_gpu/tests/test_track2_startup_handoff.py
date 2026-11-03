"""Generated delayed map anchors only; no bag, PLY or saved-pose localization."""
import numpy as np
import pytest
from track2 import parameters as p
from track2.correction_filter import CausalCorrectionFilter


C = np.diag([100., 100., np.deg2rad(10.)**2])
R = np.diag([.1**2, .1**2, np.deg2rad(.5)**2])


def predict(f, k, *, shift=None, reference_delta=None):
    T = np.eye(4)
    if shift is not None:
        T[:2, 3] = shift
    f.predict(T, 1_000_000_000+k*40_000_000, 0., prior_covariance=C,
              rotation_delta_rad=0., reference_delta=reference_delta)


def delayed():
    f = CausalCorrectionFilter()
    predict(f, 0)
    f.update(missing_reason='generated_no_geometry')
    return f


def test_delayed_first_anchor_obeys_same_frame_xy_and_yaw_budget():
    f = delayed(); predict(f, 1)
    result, d = f.update(np.array([-1., .9, .02]), R)
    assert d['accepted'] and d['delayed_first_anchor_budget_applied']
    assert np.linalg.norm(result[:2]) <= d['position_update_limit_m']+1e-12
    assert abs(result[2]) <= d['yaw_update_limit_rad']+1e-12
    assert d['position_update_limit_m'] == pytest.approx(.02)
    assert f.last_update == 1_040_000_000


def test_true_first_frame_can_initialize_without_a_zero_time_budget():
    f = CausalCorrectionFilter(); predict(f, 0)
    result, d = f.update(np.array([-1., .9, .02]), R)
    assert d['accepted'] and not d['delayed_first_anchor_budget_applied']
    assert np.linalg.norm(result[:2]) > 1.
    assert d['position_update_limit_m'] == 0.


def test_partial_geometry_uses_limited_gain_for_mean_covariance_and_sensitivity():
    f = delayed(); predict(f, 1)
    H = np.array([[1., 0., 2.], [0., 1., -3.]])
    S = np.array([[.9, .02, -.4], [.01, .85, .7], [0., 0., .5]])
    result, d = f.update(np.array([-1., .9, .02]), np.diag([.01, .02]),
                         observation_matrix=H, candidate_sensitivity=S)
    assert d['accepted'] and d['delayed_first_anchor_budget_applied']
    assert np.linalg.norm(result[:2]) <= d['position_update_limit_m']+1e-12
    assert abs(result[2]) <= d['yaw_update_limit_rad']+1e-12
    K = np.array(d['kalman_gain']); A = np.eye(3)-K @ H
    P = np.array(d['covariance_predicted'])
    effective_R = np.array(d['measurement_covariance'])/d['measurement_weight']
    np.testing.assert_allclose(f.covariance, A @ P @ A.T+K @ effective_R @ K.T, atol=1e-12)
    np.testing.assert_allclose(f.reference_sensitivity, K @ H @ S, atol=1e-12)
    assert np.linalg.eigvalsh(f.covariance).min() > 0.


def test_rejected_first_proposal_stays_unanchored_and_next_frame_is_still_bounded():
    f = delayed(); predict(f, 1)
    result, d = f.update(np.array([-1., .9, .02]), R,
        proposal_validator=lambda proposal:dict(accepted=False, reason='generated_rejection'))
    assert not d['accepted'] and f.last_update is None
    np.testing.assert_array_equal(result, np.zeros(3))
    np.testing.assert_array_equal(f.reference_sensitivity, np.zeros((3, 3)))
    predict(f, 2)
    result, d = f.update(np.array([-1., .9, .02]), R)
    assert d['accepted'] and d['delayed_first_anchor_budget_applied']
    assert np.linalg.norm(result[:2]) <= .02+1e-12  # Not two accumulated frame budgets.


@pytest.mark.parametrize('disable', ['V511_ENABLED', 'STARTUP_BOUND_AFTER_FIRST_FRAME'])
def test_disabling_fix_preserves_legacy_delayed_first_anchor(monkeypatch, disable):
    monkeypatch.setattr(p, disable, False)
    f = delayed(); predict(f, 1)
    result, d = f.update(np.array([-1., .9, .02]), R)
    assert d['accepted'] and not d['delayed_first_anchor_budget_applied']
    assert np.linalg.norm(result[:2]) > 1.


def test_sensor_position_jump_remains_one_prior_update_not_extra_map_motion_budget():
    f = delayed(); predict(f, 1)
    before, _ = f.update(np.array([-1., .9, .02]), R)
    predict(f, 2, shift=[.5, -.45])
    after, d = f.update()
    np.testing.assert_array_equal(after, before)
    assert d['motion']['distance_m'] == 0.
    # GNSS changes the prior; this map-state fix does not cancel that information.
    np.testing.assert_allclose(f.last_prior[:2, 3]+after[:2]-before[:2], [.5, -.45])
