"""Production v5.6: v5.2 measurements, independent directional memory expiry."""
import json
import numpy as np
import pytest
from track2 import parameters as p
from track2.causal_prior import CausalSensorPrior
from track2.correction_filter import CausalCorrectionFilter
from track2.observability import geometry_observation

C = np.diag([.36, .36, np.deg2rad(10.)**2])
R = np.diag([.15**2, .15**2, np.deg2rad(.5)**2])
YZ = np.array([[0., 1., 0.], [0., 0., 1.]])


@pytest.fixture(autouse=True)
def retained_v57_heading_contract(monkeypatch):
    monkeypatch.setattr(p, 'V510_ENABLED', False)
    monkeypatch.setattr(p, 'V511_ENABLED', False)


def test_production_profile_and_unknown_profile_fails(monkeypatch):
    assert p.VERSION == 'track2_v510_turn_patch_v5_16_4_perf2'
    assert not p.V515_ENABLED
    assert not p.V5165_ENABLED
    assert p.uses_v52_fusion()
    monkeypatch.setattr(p, 'FUSION_PROFILE', 'typo')
    with pytest.raises(ValueError, match='Unknown Track2 fusion profile'):
        p.uses_v52_fusion()


def test_runner_rejects_historical_profile_before_writing_output(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[1]/'scripts/run_track2.py'
    spec = importlib.util.spec_from_file_location('v56_runner_guard', script)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'verify_snapshot', lambda _: None)
    monkeypatch.setattr(p, 'FUSION_PROFILE', 'v55_experimental')
    with pytest.raises(ValueError, match='restricted to component regressions'):
        runner.execute(tmp_path)
    assert not list(tmp_path.iterdir())


def test_weak_positive_modes_remain_omitted_like_v52():
    O, covariance, d = geometry_observation(np.diag([.2, 0., 1000., 100.]),
        np.eye(3, 4), C, dict(coverage=.1, median_plane_residual_m=.05))
    assert d['rank'] == d['strong_rank'] == 1
    np.testing.assert_allclose(O[:, :2], 0., atol=1e-14)
    assert covariance.shape == (1, 1)


def test_heading_pair_keeps_v52_single_update_and_actual_joseph():
    s = CausalSensorPrior(np.eye(4), 1_000_000_000)
    s.orientation_stamp = s.time
    s.for_frame(s.time)
    before = s.heading_covariance.copy()
    H = np.array([1., -1.]); z = np.deg2rad(8.); variance = np.deg2rad(3.)**2
    K = before @ H / (H @ before @ H + variance)
    d = s._heading_update(z, H, variance)
    np.testing.assert_allclose(s.heading_bias, K*z, atol=1e-15)
    A = np.eye(2)-np.outer(K, H)
    np.testing.assert_allclose(s.heading_covariance, A @ before @ A.T+np.outer(K, K)*variance, atol=1e-15)
    assert not s.pending_heading_information and 'information_fraction' not in d


def test_lateral_pseudo_measurement_is_inactive_in_production():
    from scipy.spatial.transform import Rotation
    s = CausalSensorPrior(np.eye(4), 1_000_000_000)
    s.orientation_stamp = s.time; s.gnss_updates = 1
    s.position = np.array([0., 0., 8., 1.]); before = s.position.copy(); covariance = s.cov.copy()
    s._lateral_velocity_constraint(Rotation.identity())
    np.testing.assert_array_equal(s.position, before)
    np.testing.assert_array_equal(s.cov, covariance)
    assert not s.lateral_constraint_diagnostic['accepted']


def simulate(n):
    f = CausalCorrectionFilter(); rows = []
    for k in range(n):
        t = k*.05
        f.predict(np.eye(4), 1_000_000_000+k*50_000_000, 0., prior_covariance=C, rotation_delta_rad=0.)
        H, covariance = (np.eye(3), R) if k == 0 else (YZ, R[1:, 1:])
        value, d = f.update([.4, .1, .01], covariance, observation_matrix=H)
        assert d['accepted']
        rows.append(dict(t=t, value=value.tolist(), diagnostic=d))
    return rows


def test_unobserved_mode_expires_while_other_modes_keep_updating():
    rows = simulate(121); initial = rows[0]['value'][0]
    for row in rows:
        t = row['t']
        expected = initial*np.exp(-max(0., t-p.CORRECTION_HOLD_SECONDS)/p.CORRECTION_DECAY_SECONDS)
        assert row['value'][0] == pytest.approx(expected, abs=2e-14)
        assert np.linalg.eigvalsh(row['diagnostic']['covariance']).min() > 0
    assert rows[-1]['diagnostic']['seconds_since_update'] == 0.
    assert max(rows[-1]['diagnostic']['memory_age_eigenvalues_seconds']) == pytest.approx(6.)
    assert abs(rows[-1]['value'][1]-.1) < .005


def test_production_directional_memory_prefix_is_unchanged():
    short, long = simulate(25), simulate(121)
    assert json.dumps(short, sort_keys=True, allow_nan=False) == json.dumps(long[:25], sort_keys=True, allow_nan=False)


def test_cuda_production_uses_plane_candidate_and_original_height_contract():
    import cupy as cp
    from track2.metric_registration import MetricRegistration
    from test_track2_metric_scale import scene, truth, raw_cloud
    T = truth(); prior = T.copy(); prior[0, 3] += .2
    reg = MetricRegistration(cp.asarray(scene()), T)
    pose, scale, d = reg.refine(cp.asarray(raw_cloud(scene(), T, 8.)), prior,
                               1_000_000_000, prior_covariance=C)
    assert d['accepted'] and d['fusion_profile'] == 'v52_directional_memory'
    assert d['line_matches'] == 0 and d['planes']['source_line_features'] == 0
    assert d['planes']['ground_support'] is None
    assert 'likelihood_observation' not in d and 'height_likelihood' not in d
    assert not d['correction_filter']['local_linear_likelihood']
    assert scale == pytest.approx(d['candidate_scale'])
    assert d['scale_source'] == 'current_features_and_ground'
    assert np.isfinite(pose).all() and np.linalg.eigvalsh(reg.correction_filter.covariance).min() > 0
