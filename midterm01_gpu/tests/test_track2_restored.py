"""Generated causal/prefix + 5839746c C++ numerical oracle checks."""
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from track2.causal_prior import CausalSensorPrior
from track2 import parameters as p


def initial_pose():
    T = np.eye(4); T[:3, 3] = [3272., -984., 17.]
    return T


def imu(t, yaw=.2):
    return dict(kind='imu', timestamp_ns=int(t), arrival_ns=int(t), gyro=[0., 0., 0.],
                accel=[0., 0., 9.8], quaternion=Rotation.from_euler('z', yaw).as_quat().tolist())


def synthetic_priors(n):
    origin = 1_000_000_000
    s = CausalSensorPrior(initial_pose(), origin)
    result = []; diagnostic = []
    for k in range(n):
        t = origin+k*50_000_000
        s.event(imu(t))
        if k % 5 == 0:
            dt = k*.05
            s.event(dict(kind='gnss', timestamp_ns=t, arrival_ns=t,
                         xy=[3272.+8*np.cos(.25)*dt, -984.+8*np.sin(.25)*dt], z=17.))
        T, d = s.for_frame(t)
        result.append(T); diagnostic.append(d)
    return np.stack(result), diagnostic


def test_causal_prior_prefix_and_heading_blocks():
    short, short_d = synthetic_priors(65)
    full, full_d = synthetic_priors(180)
    np.testing.assert_array_equal(short, full[:65])
    assert short_d == full_d[:65]
    assert full_d[-1]['accepted_heading_blocks'] >= 2
    assert full_d[-1]['heading_bias_rad'] > 0
    for k, d in enumerate(full_d):
        cutoff = 1_000_000_000+k*50_000_000
        assert all(r['end_timestamp_ns'] <= cutoff for r in d['block_records'])
        assert d['imu_orientation_timestamp_ns'] <= cutoff
    # Initial speed is zero, never seeded from a later GNSS measurement.
    np.testing.assert_array_equal(full_d[0]['velocity_xy_mps'], [0., 0.])


def test_no_heading_before_block_complete_and_stale_imu_rejected():
    poses, diag = synthetic_priors(20)
    assert all(d['heading_bias_rad'] == 0. for d in diag)
    s = CausalSensorPrior(initial_pose(), 1_000_000_000)
    s.event(imu(1_000_000_000))
    with pytest.raises(ValueError, match='stale'):
        s.for_frame(1_200_000_000)
    with pytest.raises(ValueError, match='backward'):
        s.advance(1)


def test_future_data_cannot_rewrite_saved_prefix():
    s = CausalSensorPrior(initial_pose(), 1_000_000_000)
    s.event(imu(1_000_000_000))
    first, d = s.for_frame(1_000_000_000)
    frozen = first.copy()
    s.event(imu(2_000_000_000, 1.))
    s.event(dict(kind='gnss', timestamp_ns=2_000_000_000, xy=[9999., 9999.], z=999.))
    s.for_frame(2_000_000_000)
    np.testing.assert_array_equal(first, frozen)


def test_stream_availability_future_and_late(tmp_path):
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import Imu
    from geometry_msgs.msg import PoseStamped
    from track2.sensors import SensorStream
    path = tmp_path/'bag.db3'; db = sqlite3.connect(path)
    db.execute('CREATE TABLE topics(id INTEGER,name TEXT,type TEXT,serialization_format TEXT)')
    db.execute('CREATE TABLE messages(id INTEGER,timestamp INTEGER,topic_id INTEGER,data BLOB)')
    db.executemany('INSERT INTO topics VALUES(?,?,?,?)', [(1, '/imu/data', 'sensor_msgs/msg/Imu', 'cdr'), (2, '/localizer_node/gnss_pose', 'geometry_msgs/msg/PoseStamped', 'cdr')])
    for i, (kind, arrival, stamp) in enumerate([(1, 110, 100), (2, 115, 112), (1, 130, 90)]):
        m = Imu() if kind == 1 else PoseStamped()
        m.header.stamp.sec, m.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
        if kind == 1: m.header.frame_id = 'imu_link'; m.orientation.w = 1.
        db.execute('INSERT INTO messages VALUES(?,?,?,?)', (i, arrival, kind, serialize_message(m)))
    db.commit(); db.close()
    s = SensorStream(path)
    assert [r['timestamp_ns'] for r in s.take(120, 105)] == [100]
    assert [r['timestamp_ns'] for r in s.take(150, 120)] == [112]
    assert s.summary()['late'] == 1
    s.close()


def cloud():
    x, y = np.meshgrid(np.arange(2., 14., .5), np.arange(-5., 5., .5))
    ground = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, -1.5)])
    x, z = np.meshgrid(np.arange(2., 14., .5), np.arange(-1., 3., .5))
    side = np.column_stack([x.ravel(), np.full(x.size, 4.), z.ravel()])
    y, z = np.meshgrid(np.arange(-4., 4., .5), np.arange(-1., 3., .5))
    front = np.column_stack([np.full(y.size, 8.), y.ravel(), z.ravel()])
    pts = np.concatenate([ground, side, front])
    return (pts+np.random.default_rng(21).normal(0, .002, pts.shape)).astype(np.float32)


@pytest.fixture
def cuda():
    import cupy as cp
    assert cp.cuda.runtime.getDeviceCount() > 0
    return cp


def oracle(src, target, prior, previous, alpha):
    path = os.environ['TRACK2_REFERENCE_LIBRARY']
    lib = ctypes.CDLL(path)
    f = lib.original_guided
    fp = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    dp = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    f.argtypes = [fp, ctypes.c_int, fp, ctypes.c_int, fp, dp, ctypes.c_double, dp]
    f.restype = ctypes.c_int
    result = np.zeros(28)
    accepted = f(np.ascontiguousarray(src, dtype=np.float32), len(src),
                 np.ascontiguousarray(target, dtype=np.float32), len(target),
                 np.ascontiguousarray(prior, dtype=np.float32),
                 np.ascontiguousarray(previous), alpha, result)
    assert accepted >= 0
    return bool(accepted), result


def test_geometry_cuda_pca_matches_original_definition(cuda):
    from track2.guided import Geometry
    from track3.nearest import GpuNearest
    from scipy.spatial import cKDTree
    pts = cloud(); nn = GpuNearest(cuda.asarray(pts), cell=2.5); geometry = Geometry(nn)
    ids = cuda.arange(len(nn.points)); geometry.ensure(ids)
    sorted_pts = cuda.asnumpy(nn.points)
    d, index = cKDTree(sorted_pts).query(sorted_pts, k=16)
    covariances = []
    normals = []; confidence = []; reliable = []
    for i in range(len(pts)):
        points = sorted_pts[index[i][d[i] <= 2.5]]-sorted_pts[i]
        cov = (points-points.mean(axis=0)).T @ (points-points.mean(axis=0))/len(points)
        val, vec = np.linalg.eigh(cov); val = np.maximum(val, 0.)
        planarity = (val[1]-val[0])/max(val[2], 1e-30)
        reliable.append(len(points) >= 6 and val[1] >= .04 and val[2] >= 1e-8 and val[0] <= .25*val[1] and planarity >= .2)
        normals.append(np.outer(vec[:, 0], vec[:, 0])); confidence.append(np.clip(planarity, 0., 1.))
    np.testing.assert_array_equal(cuda.asnumpy(geometry.reliable), reliable)
    np.testing.assert_allclose(cuda.asnumpy(geometry.confidence), confidence, atol=1e-10)
    n = cuda.asnumpy(geometry.normal)
    np.testing.assert_allclose(n[:, :, None]*n[:, None, :], normals, atol=1e-10)


@pytest.mark.parametrize('alpha', [.25, 1.])
def test_cpp_5839746c_solver_numerical_match(cuda, alpha, monkeypatch, tmp_path):
    from track2.guided import GuidedRegistration
    src = cloud()
    truth = initial_pose(); truth[:3, :3] = Rotation.from_euler('xyz', [.03, -.02, .2]).as_matrix()
    target = (src @ truth[:3, :3].T+truth[:3, 3]).astype(np.float32)
    prior = truth.copy(); prior[:3, 3] += [.3, -.2, .1]
    prior[:3, :3] = Rotation.from_euler('z', .008).as_matrix() @ prior[:3, :3]
    prior = prior.astype(np.float32).astype(float)
    previous = np.array([.01, -.005, .003, .0001])
    monkeypatch.setattr(p, 'SENSOR_SMOOTHING_ALPHA', alpha)
    ref_accepted, reference = oracle(src, target, prior, previous, alpha)
    reg = GuidedRegistration(cuda.asarray(target)); reg.previous_correction = previous.copy()
    out, diagnostic = reg.refine(cuda.asarray(src), prior)
    assert diagnostic['accepted'] == ref_accepted
    np.testing.assert_allclose(out, reference[:16].reshape(4, 4), atol=3e-4, rtol=0)
    np.testing.assert_allclose(diagnostic['correction'], reference[16:20], atol=3e-4, rtol=0)
    assert diagnostic['cost'] == pytest.approx(reference[21], abs=3e-4)
    assert diagnostic['prior_cost'] == pytest.approx(reference[20], abs=3e-4)
    (tmp_path/'comparison.json').write_text(json.dumps(dict(alpha=alpha,
        reference_pose=reference[:16].reshape(4,4).tolist(), cuda_pose=out.tolist(),
        max_pose_abs_difference=float(np.max(np.abs(out-reference[:16].reshape(4,4)))),
        max_correction_abs_difference=float(np.max(np.abs(np.asarray(diagnostic['correction'])-reference[16:20]))),
        cost_abs_difference=abs(diagnostic['cost']-float(reference[21])),
        reference_cost=float(reference[21]), cuda_cost=diagnostic['cost'],
        scope='generated data; not true Track2 accuracy'), indent=2)+'\n')
    if diagnostic['accepted']:
        assert diagnostic['cost'] <= diagnostic['prior_cost']+1e-9


def test_no_overlap_resets_correction_to_current_sensor_prior(cuda):
    from track2.guided import GuidedRegistration
    reg = GuidedRegistration(cuda.asarray(cloud()+500.))
    reg.previous_correction = np.ones(4)
    prior = np.eye(4)
    out, diagnostic = reg.refine(cuda.asarray(cloud()), prior)
    assert not diagnostic['accepted']
    np.testing.assert_array_equal(out, prior)
    np.testing.assert_array_equal(reg.previous_correction, np.zeros(4))


@pytest.mark.parametrize('turn_enabled',[False,True])
@pytest.mark.parametrize('surface_enabled',[False,True])
@pytest.mark.parametrize('height_enabled',[False,True])
def test_generated_output_runner(tmp_path, monkeypatch, cuda, turn_enabled, surface_enabled, height_enabled):
    monkeypatch.setattr(p, 'V5162_ENABLED', turn_enabled)
    monkeypatch.setattr(p, 'V5161_ENABLED', surface_enabled)
    monkeypatch.setattr(p, 'V5165_ENABLED', height_enabled)
    # This legacy fixture contains no images; the image-enabled runner has a
    # separate generated end-to-end fixture.
    monkeypatch.setattr(p, 'IMAGE_GROUND_SPEED_ENABLED', False)
    from plyfile import PlyData, PlyElement
    import track2.sensors as sm
    import track2.camera as camera_module
    import track3.time_alignment as cm
    from track2.analysis import validate
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('restored_runner', root/'scripts/run_track2.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'verify_snapshot', lambda _: None)
    data = tmp_path/'data'; data.mkdir(); (data/'raw_ply').mkdir()
    T = initial_pose(); T[:3, :3] = Rotation.from_euler('z', .2).as_matrix()
    xyz = cloud(); target = xyz @ T[:3, :3].T+T[:3, 3]
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    a = np.zeros(len(xyz), dtype=dtype)
    for i, c in enumerate(('x', 'y', 'z')): a[c] = target[:, i]
    (data/'map.pcd').write_bytes(f'FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nPOINTS {len(a)}\nDATA binary\n'.encode()+a.tobytes())
    np.savetxt(data/'init_guess.txt', T, delimiter=','); (data/'Track2.db3').write_bytes(b'generated mock')
    for k in range(5):
        for i, c in enumerate(('x', 'y', 'z')): a[c] = xyz[:, i]/3.
        PlyData([PlyElement.describe(a, 'vertex')]).write(str(data/'raw_ply'/f'{k:06d}_ts_{1_000_000_000+k*50_000_000}.ply'))
    class Sensors:
        def __init__(self, *_): pass
        def take(self, arrival, camera): return [imu(camera)]
        def summary(self): return {'generated': True}
        def close(self): pass
    class Clock:
        def __init__(self, *_): pass
        def for_frame(self, t): return t-7_000_000
        def close(self): pass
    class Camera:
        def __init__(self, *_): self.sources={'generated':'identity_for_synthetic_cloud'}
        def apply(self, raw): return raw, {'generated':True}
    monkeypatch.setattr(sm, 'SensorStream', Sensors); monkeypatch.setattr(cm, 'CameraClock', Clock)
    monkeypatch.setattr(camera_module, 'FrontMask', Camera)
    import track2.visualization as visualization
    display_stamps=[]
    class Display:
        def publish_map(self, *_): pass
        def publish_submap(self, *_): pass
        def publish_frame(self, raw, pose, scale, stamp, **kwargs): display_stamps.append(stamp)
        def close(self): pass
    monkeypatch.setattr(visualization, 'Track2RvizPublisher', Display)
    monkeypatch.setattr(p, 'DATA_DIR', str(data)); monkeypatch.setattr(p, 'EXPECTED_FRAMES', 5)
    run = tmp_path/'generated_run'; run.mkdir()
    runner.execute(run, max_frames=3, publish_rviz=True)
    validate(run, expected=3)
    info = json.loads((run/'test_info.json').read_text())
    assert info['status'] == 'completed' and info['frame_count'] == 3
    assert info['base_commit'] == p.BASE_COMMIT
    assert not (run/'analysis').exists()
    scope = json.loads((run/'algorithm_scope.json').read_text())
    assert scope['baseline_run'] == 'track02_full_2611031941'
    assert scope['heading_enabled'] and scope['v5162_enabled'] == turn_enabled
    assert scope['v5163_enabled'] == turn_enabled
    assert scope['v5161_enabled'] == surface_enabled
    data_height = height_enabled and turn_enabled and not surface_enabled
    assert scope['v5165_enabled'] == data_height
    if data_height:
        assert scope['height_filter_model'] == 'data_only_local_likelihood_once'
        assert 'original v5.10 map association' in scope['numerical_changes'][0]
    assert scope['excluded_v59'] == ['adaptive feature ranges']+([] if data_height else ['data-only height update'])+([] if turn_enabled else ['stationary gyro bias learning'])
    obs = [json.loads(line) for line in (run/'observations.jsonl').read_text().splitlines()]
    traces = [json.loads(line) for line in (run/'pose_trace.jsonl').read_text().splitlines()]
    assert display_stamps == [o['camera_timestamp_ns'] for o in obs]
    assert all(('motion_prediction' in o['sensor_prior']) == turn_enabled for o in obs)
    from track2.correction_filter import pose_correction
    for o, trace in zip(obs, traces):
        assert trace['camera_timestamp_ns'] == o['camera_timestamp_ns'] == o['timestamp_ns']-7_000_000
        d = o['registration']; memory = d['correction_filter']
        assert d['accepted'] == memory['accepted'] == trace['map_accepted']
        np.testing.assert_allclose(memory['correction'],
            pose_correction(np.array(trace['final']), np.array(trace['predicted'])), atol=1e-10)
        assert trace['registered'] == (d['candidate_pose'] if d['geometry_accepted'] else None)
        assert memory['motion']['speed_mps'] == pytest.approx(o['state'][4])

    longer = tmp_path/'generated_longer_run'; longer.mkdir()
    runner.execute(longer, max_frames=5)
    validate(longer, expected=5)
    for name, lines in [('poses.txt', 3), ('result_2.csv', 4), ('pose_trace.jsonl', 3)]:
        assert (run/name).read_bytes().splitlines() == (longer/name).read_bytes().splitlines()[:lines]


def test_full_analysis_read_only_and_contract_failures(tmp_path):
    from track2.analysis import analyze, validate
    from track3.io import sha256
    run = tmp_path/'full_analysis'; run.mkdir()
    n = 3299
    with (run/'result_2.csv').open('w') as f:
        f.write('idx,x,y,yaw\n')
        for i in range(n): f.write(f'{i},{i*.01:.9f},0.0,0.0\n')
    T = np.eye(4); T[2, 3] = 1.573; T = T[:3].ravel()
    poses = np.tile(T, (n, 1)); poses[:, 3] = np.arange(n)*.01
    np.savetxt(run/'poses.txt', poses, fmt='%.12f')
    with (run/'timings.csv').open('w') as f:
        f.write('idx,original_idx,timestamp_ns,backend,load_ms,registration_ms,total_ms,map_accepted,scale\n')
        for i in range(n): f.write(f'{i},{i},{1_000_000_000+i*50_000_000},cuda_plane,1,2,4,true,8\n')
    with (run/'frame_manifest.csv').open('w') as f:
        f.write('idx,original_idx,timestamp_ns,camera_timestamp_ns,path,sha256\n')
        for i in range(n):
            stamp = 1_000_000_000+i*50_000_000
            f.write(f'{i},{i},{stamp},{stamp-100},generated,generated\n')
    (run/'test_info.json').write_text(json.dumps(dict(status='completed', mode='full', frame_count=n, generated=True)))
    with (run/'observations.jsonl').open('w') as f:
        for i in range(n):
            memory = dict(gain_scale_xy=.9, gain_scale_yaw=.5, nis_soft_threshold=5.991, nis_reject_threshold=13.816, correction=[.1, .2, .01], covariance=np.diag([.01, .01, .0001]).tolist(),
                          innovation_nis=1., accepted=True, seconds_since_update=0.,
                          proposal_validation=dict(accepted=True,baseline_data_cost=1.,proposal_data_cost=.9,
                              baseline_common_feature_cost=.2,proposal_common_feature_cost=.19))
            f.write(json.dumps(dict(state=[i*.01, 0., 0., .001, .2], scale=8.,
                                   sensor_prior=dict(pose_covariance_xy_yaw=np.diag([.36,.36,.03]).tolist(),
                                       motion_prediction=dict(model='arrived_gyro_coordinated_turn',
                                           gyro_bias_mean_radps=[0., 0., -.005], gyro_bias_covariance_rad2ps2=(np.eye(3)*1e-6).tolist(),
                                           rest_bias_updates=i//10, signed_turn_since_frame_rad=.0001, integrated_turn_rad=i*.0001),
                                       position_information_policy='fractional_likelihood_budget_v5121',
                                       gnss_state_innovation_xy=[.01,0.] if i%10==0 else [0.,0.],
                                       position_information_updates=[dict(position_factor_id=i//10,information_fraction=1.)] if i%10==0 else [],
                                       heading_information_policy='fractional_likelihood_budget_v510',
                                       heading_update_budget_used_deg=.2 if i % 10 == 0 else 0.,
                                       heading_update_budget_limit_deg=.5,
                                       heading_information_updates=[dict(heading_factor_id=i//10, information_fraction=1.)] if i % 10 == 0 else []),
                                   image_ground_speed=dict(kind='stationary' if i % 10 == 0 else 'relative_yaw' if i % 20 == 15 else 'vector_velocity',
                                       speed_mps=.2, filter_accepted=i % 5 == 0, innovation_nis=.5,
                                       reason='generated_consensus',
                                       solver_comparison=dict(
                                           pnp=dict(accepted=True, inliers=30, common_capped_pixel_cost=.8),
                                           gyro_translation=dict(accepted=i % 2 == 0, inliers=25, common_capped_pixel_cost=None if i % 2 else .9)),
                                       anchor_store=dict(accepted=i % 2 == 0, reason='generated_anchor',
                                           anchor_retained=i % 2 == 1, height_variance_m2=None if i % 2 else .01),
                                       relative_heading_update=dict(accepted=i % 20 == 15, innovation_rad=.01,
                                           innovation_nis=.5, bias_step_rad=.001, absolute_yaw_observed=False)),
                                   registration=dict(correction_filter=memory, observability=dict(rank=2,normalized_eigenvalues=[.1,2.,5.]),
                                       surface_profile=dict(normal_effective_directions=1.5, dominant_normal_bin_fraction=.8,
                                           normal_gram_eigenvalues=[.1,.2,1.], range_bands=[dict(min_m=0.,max_m=10.,matches=20,
                                               signed_median_m=None if i % 2 else .03, abs_p90_m=.1)]),
                                       stages=[dict(reason='generated_converged', iteration_trace=[dict(cost_before=1.,cost_after_fixed=.9,
                                           data_cost_before=1.,data_cost_after_fixed=.9,association_change_fraction=None if i % 2 else .1,
                                           step_xy_m=.01,step_yaw_rad=.001,step_logscale=.001,line_search_halvings=0,step_accepted=True)])],
                                       map_likelihood=dict(likelihood_step_world=[.05,.02,.001]),
                                       candidate_selection=dict(startup=i<4,fully_refined=2 if i<4 else 1))))+'\n')
    paths = [x for x in run.iterdir() if x.is_file()]
    before = {f.name: sha256(f) for f in paths}
    analyze(run)
    assert before == {f.name: sha256(f) for f in paths}
    for name in ('01_trajectory', '02_pose_changes', '03_registration_timing', '04_fusion_and_scale', '05_map_correction_memory', '06_observability_uncertainty', '07_image_ground_speed', '08_heading_information', '09_candidate_validation', '10_map_likelihood', '11_position_information', '12_image_relative', '13_geometry_diagnostics', '14_coordinated_turn'):
        for ext in ('png', 'pdf'): assert (run/'analysis'/f'{name}.{ext}').stat().st_size > 1000
    turn_summary = json.loads((run/'analysis/coordinated_turn_summary.json').read_text())
    assert turn_summary['diagnostic_frames'] == list(range(n))
    assert turn_summary['final']['rest_bias_updates'] == (n-1)//10
    image_summary = json.loads((run/'analysis/image_speed_summary.json').read_text())
    assert image_summary['accepted_frames'] == list(range(0, n, 5))
    assert image_summary['accepted_moving_intervals'] == len(range(5, n, 20))  # Relative yaw never counts as speed.
    assert 15 not in image_summary['unused_estimate_frames']
    relative_summary = json.loads((run/'analysis/image_relative_summary.json').read_text())
    assert relative_summary['channels']['relative_yaw']['accepted_frames'] == list(range(15, n, 20))
    assert relative_summary['channels']['vector_velocity']['reasons']['generated_consensus'] > 0
    assert relative_summary['solvers']['pnp']['qualified_count'] == n
    assert relative_summary['solvers']['gyro_translation']['qualified_count'] == len(range(0, n, 2))
    assert relative_summary['anchor_retained_frames'] == list(range(1, n, 2))
    assert not relative_summary['absolute_yaw_observed_frames']
    geometry_summary = json.loads((run/'analysis/geometry_diagnostics_summary.json').read_text())
    assert geometry_summary['surface_profile_frames'] == list(range(n))
    assert geometry_summary['recorded_iterations'] == n
    assert geometry_summary['stage_reasons'] == {'generated_converged': n}
    heading_summary = json.loads((run/'analysis/heading_information_summary.json').read_text())
    assert heading_summary['post_bootstrap_budget_exceedances'] == 0
    assert heading_summary['maximum_factor_fraction'] == 1.
    selection_summary=json.loads((run/'analysis/candidate_validation_summary.json').read_text())
    assert selection_summary['maximum_complete_solves']==2
    assert selection_summary['startup_frames']==[0,1,2,3]
    assert not selection_summary['proposal_reject_frames']
    likelihood_summary=json.loads((run/'analysis/map_likelihood_summary.json').read_text())
    assert likelihood_summary['data_likelihood_frames']==list(range(n))
    assert not likelihood_summary['rematched_cost_reject_frames']
    position_summary=json.loads((run/'analysis/position_information_summary.json').read_text())
    assert position_summary['maximum_axis_factor_fraction']==1.
    assert position_summary['expired_unused_axis_fraction']==0.
    # A partial same-source render can be retried without touching any pose.
    (run/'analysis'/'03_registration_timing.pdf').unlink()
    (run/'analysis'/'12_image_relative.pdf').unlink()
    (run/'analysis'/'13_geometry_diagnostics.png').unlink()
    analyze(run)
    assert (run/'analysis'/'03_registration_timing.pdf').is_file()
    assert (run/'analysis'/'12_image_relative.pdf').stat().st_size > 1000
    assert (run/'analysis'/'13_geometry_diagnostics.png').stat().st_size > 1000
    assert before == {f.name: sha256(f) for f in paths}
    (run/'analysis'/'user_note.txt').write_text('preserve user artifact')
    with pytest.raises(FileExistsError): analyze(run)
    (run/'analysis'/'user_note.txt').unlink()
    (run/'test_info.json').write_text(json.dumps(dict(status='completed', mode='full', frame_count=n, generated='changed source')))
    with pytest.raises(FileExistsError): analyze(run)
    values = (run/'result_2.csv').read_text().replace('0,0.000000000,0.0,0.0', '0,0.000000000,0.0,1.0', 1)
    (run/'result_2.csv').write_text(values)
    with pytest.raises(ValueError, match='yaw mismatch'): validate(run)


@pytest.fixture(autouse=True)
def isolate_pre_v513_model(monkeypatch):
    from track2 import parameters as model_parameters
    monkeypatch.setattr(model_parameters, 'V513_ENABLED', False)
