"""v5.7 baseline plus causal course and heading information contracts."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from track2 import parameters as p
from track2.causal_prior import CausalSensorPrior


def state():
    s = CausalSensorPrior(np.eye(4), 1_000_000_000)
    s.orientation_stamp = s.time
    s.for_frame(s.time)
    return s


@pytest.mark.parametrize('H', [[1., 0.], [1., -1.]])
def test_production_heading_spends_one_likelihood_with_joint_joseph(monkeypatch, H):
    monkeypatch.setattr(p, 'HEADING_BIAS_RW_DEG_SQRT_S', 0.)
    monkeypatch.setattr(p, 'GNSS_HEADING_BIAS_RW_DEG_SQRT_S', 0.)
    s = state(); P = s.heading_covariance.copy()
    H = np.asarray(H); z = np.deg2rad(8.); R = np.deg2rad(2.)**2
    d = s._heading_update(z, H, R)
    spent = d['information_fraction']
    for k in range(1, 100):
        s.event(dict(kind='imu', timestamp_ns=1_000_000_000+k*50_000_000,
                     quaternion=[0., 0., 0., 1.], gyro=[0., 0., 0.]))
        _, diagnostic = s.for_frame(s.time)
        assert diagnostic['heading_update_budget_used_deg'] <= p.HEADING_MAX_UPDATE_DEG+1e-12
        for update in diagnostic['heading_information_updates']:
            f = update.get('information_fraction', 0.)
            assert 0. <= f <= 1.
            spent += f
    assert spent == pytest.approx(1., abs=1e-12)
    assert not s.pending_heading_information
    K = P @ H / (H @ P @ H+R); A = np.eye(2)-np.outer(K, H)
    np.testing.assert_allclose(s.heading_bias, K*z, atol=1e-12)
    np.testing.assert_allclose(s.heading_covariance, A @ P @ A.T+np.outer(K, K)*R, atol=1e-13)


def test_pair_and_course_share_the_absolute_frame_budget():
    s = state(); total = 0.
    for H, z in (([1., -1.], 8.), ([1., 0.], -8.), ([1., 0.], 8.)):
        P = s.heading_covariance.copy(); before = s.bias
        d = s._heading_update(np.deg2rad(z), H, np.deg2rad(2.)**2)
        total += abs(s.bias-before)
        K = np.asarray(d['heading_gain']); A = np.eye(2)-np.outer(K, H)
        np.testing.assert_allclose(s.heading_covariance,
            A @ P @ A.T+np.outer(K, K)*d['measurement_variance'], atol=1e-14)
    assert total <= np.deg2rad(p.HEADING_MAX_UPDATE_DEG)+1e-14
    assert len({v['heading_factor_id'] for v in s.heading_information_updates}) == 3


def test_expired_unused_information_is_discarded_without_reobservation():
    s = state(); d = s._heading_update(np.deg2rad(8.), [1., 0.], np.deg2rad(2.)**2)
    assert d['information_fraction_remaining'] > 0.
    s.advance(s.time+int((p.HEADING_MAX_WINDOW_S+1.)*1e9))
    bias = s.heading_bias.copy(); s.heading_information_updates = []
    s._consume_heading_information()
    np.testing.assert_array_equal(s.heading_bias, bias)
    assert not s.pending_heading_information
    assert s.heading_information_updates[0]['reason'] == 'unused_heading_information_expired'


def course(points):
    s = CausalSensorPrior(np.eye(4), 1_000_000_000)
    for k, xy in enumerate(points):
        stamp = 1_000_000_000+k*250_000_000
        s.event(dict(kind='imu', timestamp_ns=stamp, quaternion=[0., 0., 0., 1.], gyro=[0., 0., 0.]))
        s.event(dict(kind='gnss', timestamp_ns=stamp, xy=list(xy), z=1.6))
    s.for_frame(s.time)
    return s


def test_stop_then_acceleration_does_not_form_one_straight_course(monkeypatch):
    points = [[max(0., k-20)**2*.02, 0.] for k in range(33)]
    guarded = course(points)
    assert guarded.accepted_blocks == 0
    assert guarded.last_course_decision['reason'] == 'nonuniform_or_unobservable_course_window'
    assert len(guarded.last_course_decision['disjoint_half_course']) == 2
    assert guarded.block  # still-unused samples may slide forward, no pose rewrite


def test_consistent_observable_halves_allow_a_course():
    s = course([[k, .03*k] for k in range(33)])
    assert s.accepted_blocks == 1
    assert s.last_course_decision['accepted']
    assert s.last_course_decision['half_course_nis'] < 1e-12
    assert not s.block


def test_incompatible_observable_halves_reject_course():
    points = [[min(k, 16), max(k-16, 0)*.6] for k in range(33)]
    s = course(points)
    assert s.accepted_blocks == 0
    d = s.last_course_decision
    assert d['reason'] == 'nonuniform_or_unobservable_course_window'
    assert d['half_course_nis'] > p.CORRECTION_NIS_REJECT_BY_DOF[0]


def test_new_heading_path_is_exactly_causal_in_stream_prefix():
    def simulate(n):
        s = CausalSensorPrior(np.eye(4), 1_000_000_000); rows = []
        for k in range(n):
            stamp = 1_000_000_000+k*50_000_000
            s.event(dict(kind='imu', timestamp_ns=stamp, quaternion=[0., 0., 0., 1.], gyro=[0., 0., 0.]))
            if k % 5 == 0:
                s.event(dict(kind='gnss', timestamp_ns=stamp, xy=[k*.3, k*.01], z=1.6))
            T, d = s.for_frame(stamp)
            rows.append((T.tolist(), d))
        return rows
    assert simulate(75) == simulate(220)[:75]


def test_legacy_switch_and_unverified_output_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(p, 'V510_ENABLED', False)
    s = state(); d = s._heading_update(.1, [1., 0.], .01)
    assert 'information_fraction' not in d and not s.pending_heading_information
    script = Path(__file__).resolve().parents[1]/'scripts/run_track2.py'
    spec = importlib.util.spec_from_file_location('v510_runner_guard', script)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'verify_snapshot', lambda _: None)
    monkeypatch.setattr(p, 'OUTPUT_REFERENCE', 'body')
    with pytest.raises(ValueError, match='camera_ply'):
        runner.execute(tmp_path)
    assert not list(tmp_path.iterdir())


def test_course_guard_blocks_a_stop_start_window_accepted_by_v57(monkeypatch):
    # 4 seconds stopped, then 1 m/s for 4 seconds. Whole-window fit looks
    # observable (~0.5 m/s, 2.53 deg) although its first half has no course.
    points = [[max(0., k-16)*.25, 0.] for k in range(33)]
    monkeypatch.setattr(p, 'V510_ENABLED', False)
    legacy = course(points)
    assert legacy.accepted_blocks == 1
    monkeypatch.setattr(p, 'V510_ENABLED', True)
    guarded = course(points)
    assert guarded.accepted_blocks == 0
    d = guarded.last_course_decision
    assert d['reason'] == 'nonuniform_or_unobservable_course_window'
    assert d['disjoint_half_course'][0]['speed_mps'] == 0.
    assert guarded.bias_var > legacy.bias_var


@pytest.fixture(autouse=True)
def isolate_pre_v513_model(monkeypatch):
    from track2 import parameters as model_parameters
    monkeypatch.setattr(model_parameters, 'V513_ENABLED', False)
