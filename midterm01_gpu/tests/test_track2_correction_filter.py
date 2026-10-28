"""Generated correction states; no real bag/PLY localization."""
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from track2 import parameters as p
from track2.correction_filter import (
    CausalCorrectionFilter, correction_covariance, pose_correction, wrap)


def prior(t=0., angle=0.):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('z', angle).as_matrix()
    T[:3, 3] = [8*t, 0., 1.6]
    return T


def measurement_covariance():
    return np.diag([.15**2, .15**2, np.deg2rad(.5)**2])


def step(f, k, candidate, covariance=None, *, dt=.05, speed=8., angle=0.):
    f.predict(prior(k*dt, angle), 1_000_000_000+round(k*dt*1e9), speed)
    return f.update(candidate, covariance if covariance is not None else measurement_covariance())


def test_short_failure_preserves_bias_but_vehicle_keeps_moving():
    f = CausalCorrectionFilter()
    c, first = step(f, 0, [.4, -.6, .02])
    P = f.covariance.copy()
    for k in range(1, 7):
        held, d = step(f, k, None)
        np.testing.assert_array_equal(held, c)
        assert not d['accepted'] and d['decay_factor'] == 1.
        assert np.all(np.diag(f.covariance) > np.diag(P))
        # Holding the correction must not hold the pose at the old position.
        assert prior(k*.05)[0, 3]+held[0] == pytest.approx(8*k*.05+c[0])


def test_long_failure_decays_by_seconds_and_can_reacquire():
    def lost(dt):
        f = CausalCorrectionFilter(); c, _ = step(f, 0, [.4, -.6, .02], dt=dt)
        for k in range(1, round(12/dt)+1): out, diag = step(f, k, None, dt=dt)
        return f, c, out, diag
    fine, c, out, d = lost(.05)
    coarse, _, out_coarse, _ = lost(.2)
    expected = c*np.exp(-(12-p.CORRECTION_HOLD_SECONDS)/p.CORRECTION_DECAY_SECONDS)
    np.testing.assert_allclose(out, expected, rtol=1e-12)
    np.testing.assert_allclose(out_coarse, expected, rtol=1e-12)
    assert np.linalg.norm(out) < .03*np.linalg.norm(c)
    assert np.diag(fine.covariance).min() > 0.
    _, d = step(fine, 241, [.4, -.6, .02])
    assert d['accepted'], d  # Failure duration enlarges the reacquisition allowance.


def test_motion_outlier_rejected_and_next_normal_candidate_recovers():
    f = CausalCorrectionFilter()
    for k in range(30): good, _ = step(f, k, [.12, 1.29, .01])
    out, d = step(f, 30, [.57, 2.34, -.023])
    assert not d['accepted'] and d['reason'] in ('motion_innovation_rejected', 'statistical_innovation_rejected')
    np.testing.assert_array_equal(out, good)
    out, d = step(f, 31, [.19, 1.38, .008])
    assert d['accepted']
    assert np.linalg.norm(out[:2]-good[:2]) < .08
    assert d['motion']['distance_m'] == pytest.approx(.4)


def test_moderate_motion_excess_is_downweighted_extreme_excess_rejects():
    f = CausalCorrectionFilter(); step(f, 0, [0., 0., 0.])
    # Deliberately uncertain measurement: NIS alone should not hard-reject it.
    R = np.diag([1., 1., np.deg2rad(3.)**2])
    c, d = step(f, 1, [.6, 0., 0.], R)
    assert d['accepted'] and d['reason'] == 'downweighted_update'
    assert 1 < d['motion_innovation_ratio'] < p.CORRECTION_MOTION_REJECT_MULTIPLIER
    assert 0 < d['measurement_weight'] < 1.
    assert d['kalman_gain'] is not None
    assert 0 < c[0] < .3
    _, d = step(f, 2, [2., 0., 0.], R)
    assert not d['accepted'] and d['reason'] == 'motion_innovation_rejected'


def test_reacquisition_does_not_postpone_a_jump_and_covariance_matches_partial_gain():
    f = CausalCorrectionFilter(); step(f, 0, [.1, .1, 0.])
    for k in range(1, 61): before, _ = step(f, k, None)
    P = f.covariance.copy()
    after, d = step(f, 61, [1., 1., .03])
    assert d['accepted'] and d['gain_scale'] < 1.
    delta = after-before
    assert np.linalg.norm(delta[:2]) <= d['position_update_limit_m']+1e-12
    assert abs(delta[2]) <= d['yaw_update_limit_rad']+1e-12
    K = np.array(d['kalman_gain']); R = np.array(d['measurement_covariance'])/d['measurement_weight']
    A = np.eye(3)-K; Pminus = np.array(d['covariance_predicted'])
    expected = A @ Pminus @ A.T+K @ R @ K.T
    np.testing.assert_allclose(f.covariance, expected, atol=1e-12)
    assert np.linalg.eigvalsh(f.covariance).min() > 0.
    for k in range(62, 72):
        previous = f.mean.copy()
        out, d = step(f, k, [1., 1., .03])
        assert d['accepted'], d
        assert np.linalg.norm(out[:2]-previous[:2]) <= d['position_update_limit_m']+1e-12
        assert abs(wrap(out[2]-previous[2])) <= d['yaw_update_limit_rad']+1e-12


def test_nis_soft_weight_hard_rejection_and_quality_weight():
    R = np.diag([.005, .005, np.deg2rad(.2)**2])
    f = CausalCorrectionFilter()
    for k in range(100): base, _ = step(f, k, [.1, -.2, .01], R)
    # Between the soft/hard statistical bounds and inside the motion envelope.
    f.predict(prior(5.), 6_000_000_000, 8.)
    delta = np.sqrt(10*(f.covariance[0, 0]+R[0, 0]))
    out, d = f.update(base+[delta, 0., 0.], R)
    assert d['accepted'] and d['reason'] == 'downweighted_update'
    assert d['innovation_nis'] == pytest.approx(10.)
    assert 0 < d['measurement_weight'] < 1.
    assert 0 < out[0]-base[0] < delta/2
    # A yaw innovation can fail the statistical gate while inside motion bounds.
    f.predict(prior(5.05), 6_050_000_000, 8.)
    precise = R.copy(); precise[2, 2] = np.deg2rad(.05)**2
    c = f.mean.copy(); c[2] += np.sqrt(17*(f.covariance[2, 2]+precise[2, 2]))
    _, d = f.update(c, precise)
    assert not d['accepted'] and d['reason'] == 'statistical_innovation_rejected'
    f1 = CausalCorrectionFilter(); f2 = CausalCorrectionFilter()
    a, _ = step(f1, 0, [.3, .2, .01], R)
    b, _ = step(f2, 0, [.3, .2, .01], R*20)
    assert np.linalg.norm(b) < np.linalg.norm(a)


def test_motion_allowance_uses_time_speed_and_rotation():
    def allowance(dt, speed, angle):
        f = CausalCorrectionFilter(); step(f, 0, [0., 0., 0.])
        _, d = step(f, 1, None, dt=dt, speed=speed, angle=angle)
        return d
    base = allowance(.05, 0., 0.)
    faster = allowance(.05, 10., 0.)
    longer = allowance(.2, 10., 0.)
    turning = allowance(.05, 10., .03)
    assert base['position_innovation_limit_m'] < faster['position_innovation_limit_m'] < longer['position_innovation_limit_m']
    assert turning['position_innovation_limit_m'] > faster['position_innovation_limit_m']
    assert turning['yaw_innovation_limit_rad'] > faster['yaw_innovation_limit_rad']


def test_information_weak_direction_quality_and_covariance_are_conservative():
    J = np.eye(3, 4)
    H = np.diag([0., 100., 100., 100.])
    good = dict(coverage=.95, median_capped_distance_m=.15)
    bad = dict(coverage=.65, median_capped_distance_m=.45)
    R = correction_covariance(H, J, good)
    degraded = correction_covariance(H, J, bad)
    assert R[0, 0] > 1e6  # pinv must not invent certainty in the null space.
    assert np.linalg.eigvalsh(R).min() > 0
    assert np.all(np.diag(degraded) > np.diag(R))
    f = CausalCorrectionFilter(); out, _ = step(f, 0, [.5, .2, .01], R)
    assert abs(out[0]) < 1e-6 and out[1] > .1


def test_wrapped_yaw_prefix_and_invalid_clock():
    f = CausalCorrectionFilter(); step(f, 0, [0., 0., 0.])
    f.mean[2] = np.pi-.001
    f.last_measurement[2] = np.pi-.001
    out, d = step(f, 1, [0., 0., -np.pi+.001])
    assert d['accepted'] and d['innovation'][2] == pytest.approx(.002)
    assert abs(wrap(out[2]-np.pi)) < .001
    with pytest.raises(ValueError, match='strictly increase'):
        f.predict(prior(), 1_050_000_000)
    def run(n):
        g = CausalCorrectionFilter(); records = []
        for k in range(n):
            _, d = step(g, k, None if k % 13 == 0 else [.1, -.2, .001*k], angle=.001*k)
            records.append(json.dumps(d, allow_nan=False, sort_keys=True))
        return records
    assert run(35) == run(100)[:35]
