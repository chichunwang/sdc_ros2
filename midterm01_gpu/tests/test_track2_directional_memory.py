"""Generated motion/geometry regressions; no real Track frames or truth files."""
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from track2 import parameters as p
from track2.causal_prior import CausalSensorPrior
from track2.correction_filter import CausalCorrectionFilter


COV = np.diag([.36, .36, np.deg2rad(10.)**2])
R = np.diag([.15**2, .15**2, np.deg2rad(.5)**2])
YZ = np.array([[0., 1., 0.], [0., 0., 1.]])


def partial_motion(n=201, accelerate=True):
    prior = CausalSensorPrior(np.eye(4), 1_000_000_000)
    correction = CausalCorrectionFilter()
    result = []
    for k in range(n):
        t = k*.05
        stamp = 1_000_000_000+k*50_000_000
        truth = max(0., t-1.)**2 if accelerate else 0.
        prior.event(dict(kind='imu', timestamp_ns=stamp,
                         quaternion=[0., 0., 0., 1.], gyro=[0., 0., 0.]))
        if k % 5 == 0:
            prior.event(dict(kind='gnss', timestamp_ns=stamp, xy=[truth, 0.], z=1.6))
        T, sensor = prior.for_frame(stamp)
        H = np.eye(3) if k < 10 else YZ
        covariance = R if k < 10 else R[1:, 1:]
        correction.predict(T, stamp, np.linalg.norm(prior.position[2:]),
                           prior_covariance=sensor['pose_covariance_xy_yaw'],
                           rotation_delta_rad=0., reference_delta=sensor['calibration_delta_xy_yaw'])
        mean, diag = correction.update([truth-T[0, 3], -T[1, 3], 0.], covariance,
                                       observation_matrix=H)
        assert diag['accepted']
        result.append(dict(t=t, truth=truth, prior_x=float(T[0, 3]),
                           output_x=float(T[0, 3]+mean[0]), sensor=sensor, update=diag))
    return correction, result


def test_acceleration_gnss_innovation_is_not_cancelled_by_unobserved_map_x():
    f, rows = partial_motion()
    truth = np.array([r['truth'] for r in rows])
    sensor = np.array([r['prior_x'] for r in rows])
    output = np.array([r['output_x'] for r in rows])
    # X has no map observation after .45s: effective GNSS corrections must pass.
    np.testing.assert_allclose(output, sensor, atol=1e-12)
    assert np.sqrt(np.mean((output[20:]-truth[20:])**2)) < .2
    assert rows[-1]['update']['seconds_since_update'] == 0.
    assert max(rows[-1]['update']['memory_age_eigenvalues_seconds']) == pytest.approx(9.55)
    assert f.reference_sensitivity[0, 0] < .06
    assert max(np.linalg.norm(r['sensor']['gnss_state_innovation_xy']) for r in rows) > .1
    assert all(r['sensor']['calibration_delta_xy_yaw'] == [0., 0., 0.] for r in rows)


@pytest.mark.parametrize('dt', [.05, .2])
def test_partial_success_retains_short_gap_and_expires_only_old_direction(dt):
    f = CausalCorrectionFilter()
    f.predict(np.eye(4), 1_000_000_000, prior_covariance=COV)
    initial, _ = f.update([.4, 0., 0.], R)
    for k in range(1, round(4/dt)+1):
        t = k*dt
        f.predict(np.eye(4), 1_000_000_000+round(t*1e9), 0., prior_covariance=COV)
        out, diag = f.update([99., 0., 0.], R[1:, 1:], observation_matrix=YZ)
        assert diag['accepted'] and diag['seconds_since_update'] == 0.
        expected = initial[0]*np.exp(-max(0., t-p.CORRECTION_HOLD_SECONDS)/p.CORRECTION_DECAY_SECONDS)
        assert out[0] == pytest.approx(expected, rel=1e-12)
        assert out[1] == 0. and out[2] == 0.
        assert np.linalg.eigvalsh(f.covariance).min() > 0.
    assert f.reference_sensitivity[0, 0] < .4
    assert f.reference_sensitivity[1, 1] > .99
    assert f.reference_sensitivity[2, 2] > .99


def test_rejected_partial_measurement_does_not_refresh_its_direction():
    f = CausalCorrectionFilter()
    f.predict(np.eye(4), 1_000_000_000, prior_covariance=COV)
    f.update([.2, .2, 0.], R)
    for k in range(1, 31):
        f.predict(np.eye(4), 1_000_000_000+k*50_000_000, 0., prior_covariance=COV)
        _, d = f.update([0., 100., 0.], R[1:, 1:], observation_matrix=YZ)
        assert not d['accepted']
    np.testing.assert_allclose(d['memory_age_eigenvalues_seconds'], [1.5]*3, atol=1e-12)


def test_rotated_partial_modes_expire_in_geometry_coordinates_and_covariance_stays_psd():
    def simulate(angle):
        rotate = Rotation.from_euler('z', angle).as_matrix()
        H = YZ @ rotate.T
        f = CausalCorrectionFilter()
        f.predict(np.eye(4), 1_000_000_000, prior_covariance=COV)
        f.update(rotate @ [.4, .2, .01], R)
        for k in range(1, 81):
            # Sensor covariance changing does not change the memory clock metric.
            C = COV.copy(); C[0, 0] *= 1+.3*np.sin(k*.2)
            C = rotate @ C @ rotate.T
            f.predict(np.eye(4), 1_000_000_000+k*50_000_000, 0., prior_covariance=C)
            mean, d = f.update(rotate @ [.4, .2, .01], R[1:, 1:], observation_matrix=H)
            assert d['accepted']
            assert np.linalg.eigvalsh(f.covariance).min() > 0.
        return rotate.T @ mean, np.linalg.eigvalsh(f.memory_age)
    base, age = simulate(0.)
    turned, other = simulate(.63)
    np.testing.assert_allclose(base, turned, atol=1e-12)
    np.testing.assert_allclose(age, other, atol=1e-12)
    assert age[-1] == pytest.approx(4.)


def test_direction_reacquisition_is_still_limited_after_long_partial_success():
    f, rows = partial_motion(accelerate=False)
    old = f.mean.copy()
    f.predict(np.eye(4), 11_050_000_000, 0., prior_covariance=COV)
    mean, d = f.update([.25, 0., 0.], R)
    assert d['accepted'] and d['gain_scale_xy'] < 1.
    assert np.linalg.norm((mean-old)[:2]) <= d['position_update_limit_m']+1e-12
    np.testing.assert_allclose(f.memory_age, 0., atol=1e-12)


def test_genuine_heading_calibration_still_rebases_its_lever_displacement(monkeypatch):
    monkeypatch.setattr(p, 'GNSS_TO_PLY_LEVER_BODY_M', (1.2, .4, 0.))
    s = CausalSensorPrior(np.eye(4), 1_000_000_000)
    s.event(dict(kind='imu', timestamp_ns=1_000_000_000,
                 quaternion=[0., 0., 0., 1.], gyro=[0., 0., 0.]))
    old, _ = s.for_frame(1_000_000_000)
    s.bias = .1
    new, d = s.for_frame(1_050_000_000)
    np.testing.assert_allclose(d['calibration_delta_xy_yaw'][:2], new[:2, 3]-old[:2, 3], atol=1e-14)
    assert d['calibration_delta_xy_yaw'][2] == pytest.approx(.1)


def test_directional_memory_is_prefix_causal_and_diagnostics_are_finite():
    _, short = partial_motion(45)
    _, long = partial_motion(95)
    assert json.dumps(short, sort_keys=True, allow_nan=False) == json.dumps(long[:45], sort_keys=True, allow_nan=False)
