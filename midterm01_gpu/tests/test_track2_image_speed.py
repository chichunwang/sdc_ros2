"""Generated camera geometry/causality; no reference CSV or real localization."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from track2 import parameters as p
from track2.image_speed import CausalGroundSpeed, ground_points, ground_speed
from track2.visual_update import fuse_visual_speed
from track2.causal_prior import CausalSensorPrior
from track3.visual_prior import RDF_TO_FLU

K = np.array([[400., 0., 320.], [0., 400., 240.], [0., 0., 1.]])


def camera():
    y, x = np.mgrid[:480, :640]
    return SimpleNamespace(K=K, maps=(x.astype('f4'), y.astype('f4')),
                           mask=np.full((480, 640), 255, np.uint8))


def project(points):
    optical = np.asarray(points) @ RDF_TO_FLU
    return optical[:, :2]/optical[:, 2, None]*[K[0, 0], K[1, 1]]+[K[0, 2], K[1, 2]]


def test_ground_rays_recover_metres_and_reject_horizon():
    xyz = np.array([[8., 2., -1.6], [15., -3., -1.6]])
    recovered, good = ground_points(project(xyz), K, [0., 0., 1.], 1.6)
    assert good.all(); np.testing.assert_allclose(recovered, xyz, atol=1e-13)
    scaled, _ = ground_points(project(xyz), K, [0., 0., 1.], 3.2)
    np.testing.assert_allclose(scaled, xyz*2, atol=1e-13)
    _, valid = ground_points([[320., 240.], [320., 100.]], K, [0., 0., 1.], 1.6)
    assert not valid.any()


@pytest.mark.parametrize('yaw', [0., .02, -.025])
def test_generated_planar_pnp_recovers_speed_with_outliers(yaw):
    x, y = np.meshgrid(np.linspace(5., 30., 19), np.linspace(-8., 8., 25))
    xyz = np.c_[x.ravel(), y.ravel(), np.full(x.size, -1.6)]
    translation = np.array([2., .1, 0.]); R = Rotation.from_euler('z', yaw).as_matrix()
    current = (xyz-translation) @ R
    a, b = project(xyz), project(current)
    good = ((a[:, 0] > 3) & (a[:, 0] < 637) & (b[:, 0] > 3) & (b[:, 0] < 637)
            & (a[:, 1] < 477) & (b[:, 1] < 477))
    a, b = a[good], b[good]
    b = b.copy(); b[::7] += [25., -17.]
    d = ground_speed(a, b, K, [0., 0., 1.], 1.6, .01**2, .25, (480, 640))
    assert d['accepted'], d
    assert d['speed_mps'] == pytest.approx(np.linalg.norm(translation)/.25, abs=.02)
    assert d['variance_m2ps2'] >= p.IMAGE_SPEED_STD_MPS**2
    assert abs(d['vertical_motion_m']) < .01


def test_clustered_ground_tracks_do_not_manufacture_velocity():
    uv = np.random.default_rng(5).uniform([300., 290.], [310., 300.], (100, 2))
    d = ground_speed(uv, uv, K, [0., 0., 1.], 1.6, .01, .25, (480, 640))
    # Exact null geometry may reject. A small but full-rank cluster must
    # instead carry its large uncertainty, including at zero speed where a
    # naive derivative of ||v|| would incorrectly become zero.
    if d['accepted']:
        wide = np.random.default_rng(5).uniform([150., 280.], [490., 450.], (100, 2))
        other = ground_speed(wide, wide, K, [0., 0., 1.], 1.6, .01, .25, (480, 640))
        assert other['accepted']
        assert d['image_speed_variance_m2ps2'] > 100*other['image_speed_variance_m2ps2']
        assert d['variance_m2ps2'] > other['variance_m2ps2']
        assert d['speed_mps'] < .01


def sensor():
    obj = CausalSensorPrior(np.eye(4), 1_000_000_000)
    obj.orientation_stamp = obj.time
    T, d = obj.for_frame(obj.time)
    obj.position[2:] = [8., 1.]
    return obj, T, d


def test_scalar_visual_speed_updates_actual_joint_joseph_without_heading_change():
    obj, T, d = sensor(); before = obj.cov.copy(); bias = obj.heading_bias.copy()
    state_before = obj.position.copy(); old_T = T.copy()
    m = dict(accepted=True, kind='speed', timestamp_ns=obj.time, speed_mps=8.5, variance_m2ps2=1.)
    output, new, result = fuse_visual_speed(obj, T, d, m, .01)
    assert result['filter_accepted']
    H = np.array(result['observation_matrix']); gain = np.array(result['kalman_gain'])
    A = np.eye(4)-gain @ H; R = np.array(result['effective_measurement_covariance'])
    np.testing.assert_allclose(obj.cov, A @ before @ A.T+gain @ R @ gain.T, atol=1e-13)
    np.testing.assert_array_equal(obj.heading_bias, bias); np.testing.assert_array_equal(T, old_T)
    np.testing.assert_allclose(output[:2, 3], T[:2, 3]+obj.position[:2]-state_before[:2])
    assert new['calibration_delta_xy_yaw'] == d['calibration_delta_xy_yaw']
    assert np.linalg.eigvalsh(obj.cov).min() > 0


def test_stationary_image_observation_must_agree_with_sensor_velocity():
    obj, T, d = sensor(); obj.cov = np.eye(4)*.1
    before = obj.position.copy(); covariance = obj.cov.copy()
    m = dict(accepted=True, kind='stationary', timestamp_ns=obj.time,
             keyframe_timestamp_ns=obj.time-250_000_000, variance_m2ps2=.09)
    out, new, result = fuse_visual_speed(obj, T, d, m, 0.)
    assert not result['filter_accepted']
    np.testing.assert_array_equal(obj.position, before); np.testing.assert_array_equal(obj.cov, covariance)
    np.testing.assert_array_equal(out, T)
    obj.position[2:] = [.08, -.03]
    _, _, result = fuse_visual_speed(obj, T, d, m, 0.)
    assert result['filter_accepted'] and np.linalg.norm(obj.position[2:]) < .05


def test_stationary_images_remove_only_unused_course_samples_not_gnss_state():
    obj, T, d = sensor(); stamp = obj.time
    obj.block = [dict(timestamp_ns=stamp-300_000_000, xy=[0., 0.]),
                 dict(timestamp_ns=stamp-200_000_000, xy=[.1, 0.]),
                 dict(timestamp_ns=stamp-10_000_000, xy=[.1, .1])]
    before = obj.position.copy(); covariance = obj.cov.copy(); bias = obj.heading_bias.copy()
    result = obj.exclude_image_stationary_course(stamp-250_000_000, stamp)
    assert result['removed_unused_samples'] == 2 and len(obj.block) == 1
    assert obj.block[0]['timestamp_ns'] == stamp-300_000_000
    np.testing.assert_array_equal(obj.position, before); np.testing.assert_array_equal(obj.cov, covariance)
    np.testing.assert_array_equal(obj.heading_bias, bias)
    with pytest.raises(ValueError):obj.exclude_image_stationary_course(stamp, stamp+1)


def images(folder, n):
    path = folder/'Track2_images/Track2_f_images'; path.mkdir(parents=True)
    texture = np.random.default_rng(57).integers(0, 256, (480, 640), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (5, 5), .7)
    for k in range(n):
        cv2.imwrite(str(path/f'{k:06d}_ts_{1_000_000_000+k*50_000_000}.jpg'), texture)


def test_disjoint_intervals_are_consumed_once_and_prefix_is_identical(tmp_path):
    images(tmp_path, 16)
    def simulate(n):
        model = CausalGroundSpeed(tmp_path, camera()); result = []
        for k in range(n):
            stamp = 1_000_000_000+k*50_000_000
            d = model.observe(stamp, stamp, 0.); result.append(d)
            model.store(None, 1., dict(scale_validated=False))
        return model, result
    _, short = simulate(8); model, long = simulate(16)
    assert short == long[:8]
    accepted = [k for k, d in enumerate(long) if d['accepted']]
    assert accepted == [5, 10, 15]
    intervals = [(d['keyframe_timestamp_ns'], d['timestamp_ns']) for d in long if d['accepted']]
    assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
    with pytest.raises(ValueError, match='clocks'):
        model.observe(2_000_000_000, 2_100_000_000, 0.)


def test_cuda_ground_support_uses_nearest_visible_surface(tmp_path):
    import cupy as cp
    model = CausalGroundSpeed(tmp_path, camera())
    # Near non-ground object occludes a farther road point on the same ray.
    raw = cp.asarray([[1., 0., -.2], [.5, 0., -.1], [2., .2, -.2]])
    mask = model._visible_ground_mask(raw, [0., 0., 1.], .2, .01)
    assert mask[160, 160] == 0
    assert mask[140, 140] > 0


def test_image_enabled_runner_prefix_and_measurement_output(tmp_path, monkeypatch):
    import cupy as cp
    from plyfile import PlyData, PlyElement
    import track2.sensors as sm
    import track2.camera as cm
    import track3.time_alignment as clock
    from test_track2_metric_scale import scene, truth, raw_cloud
    from test_track2_restored import imu
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('image_enabled_runner', root/'scripts/run_track2.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'verify_snapshot', lambda _: None)
    data = tmp_path/'data'; (data/'raw_ply').mkdir(parents=True); images(data, 11)
    xyz = scene(); T = truth(); dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    a = np.empty(len(xyz), dtype=dtype)
    for k, key in enumerate(('x', 'y', 'z')):a[key] = xyz[:, k]
    (data/'map.pcd').write_bytes(f'FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nPOINTS {len(a)}\nDATA binary\n'.encode()+a.tobytes())
    np.savetxt(data/'init_guess.txt', T, delimiter=','); (data/'Track2.db3').write_bytes(b'generated sensor mocks')
    raw = raw_cloud(xyz, T, 8.)
    for k, key in enumerate(('x', 'y', 'z')):a[key] = raw[:, k]
    for i in range(11):PlyData([PlyElement.describe(a, 'vertex')]).write(str(data/'raw_ply'/f'{i:06d}_ts_{1_000_000_000+i*50_000_000}.ply'))
    class Sensors:
        def __init__(self, *_):pass
        def take(self, arrival, stamp):return [imu(stamp, np.arctan2(T[1, 0], T[0, 0]))]
        def summary(self):return {'generated': True}
        def close(self):pass
    class Clock:
        def __init__(self, *_):pass
        def for_frame(self, stamp):return stamp
        def close(self):pass
    class Camera:
        def __init__(self, *_):
            c = camera(); self.K = c.K; self.maps = c.maps; self.mask = c.mask; self.sources = {'generated': True}
        def apply(self, raw):return raw, {'generated': True}
    monkeypatch.setattr(sm, 'SensorStream', Sensors); monkeypatch.setattr(clock, 'CameraClock', Clock)
    monkeypatch.setattr(cm, 'FrontMask', Camera); monkeypatch.setattr(p, 'DATA_DIR', str(data))
    monkeypatch.setattr(p, 'EXPECTED_FRAMES', 11); monkeypatch.setattr(p, 'IMAGE_GROUND_SPEED_ENABLED', True)
    short, full = tmp_path/'short', tmp_path/'full'; short.mkdir(); full.mkdir()
    runner.execute(short, max_frames=7); runner.execute(full, max_frames=11)
    for name, lines in (('result_2.csv', 8), ('poses.txt', 7), ('pose_trace.jsonl', 7)):
        assert (short/name).read_bytes() == b''.join((full/name).read_bytes().splitlines(keepends=True)[:lines])
    records = [json.loads(line) for line in (full/'observations.jsonl').read_text().splitlines()]
    assert records[5]['image_ground_speed']['filter_accepted']
    assert records[10]['image_ground_speed']['filter_accepted']
    assert sum(bool(row['image_ground_speed']['filter_accepted']) for row in records) == 2
    assert all(type(row['image_ground_speed'].get('timestamp_ns', row['camera_timestamp_ns'])) is int for row in records)


@pytest.fixture(autouse=True)
def isolate_pre_v513_model(monkeypatch):
    from track2 import parameters as model_parameters
    monkeypatch.setattr(model_parameters, 'V513_ENABLED', False)
