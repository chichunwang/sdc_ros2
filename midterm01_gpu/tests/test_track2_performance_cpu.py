"""CPU algebra checks only: AST-load numerical methods without importing CuPy.

These checks do not execute CUDA or establish GPU speed/pose equivalence.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from track2 import parameters as p

ROOT = Path(__file__).resolve().parents[1]


class NumpyAlgebra:
    # Deliberately not a production backend or fallback.
    def __getattr__(self, name):
        if name == 'asnumpy':
            return np.asarray
        return getattr(np, name)


def load_methods(root=ROOT):
    scope = dict(np=np, cp=NumpyAlgebra(), Rotation=Rotation, p=p,
        wrap=lambda a: float(np.arctan2(np.sin(a), np.cos(a))))
    from track2.correction_filter import pose_correction
    scope['pose_correction'] = pose_correction
    for path, wanted in [('track3/initialization.py', 'transform4'),
                         ('track2/metric_registration.py', 'MetricRegistration')]:
        tree = ast.parse((root/path).read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == wanted]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root/path), 'exec'), scope)
    return scope


def fixture_registration(scope, count=128, matches=90):
    rng = np.random.default_rng(321)
    reg = scope['MetricRegistration'].__new__(scope['MetricRegistration'])
    reg.source_features = rng.normal(size=(count, 3))
    reg.weight = rng.uniform(.01, 1., count)
    reg.Bm = Rotation.from_euler('xyz', [.03, -.02, .1]).as_matrix()
    reg.Bs = Rotation.from_euler('xyz', [.01, -.03, -.1]).as_matrix()
    reg.nm = reg.Bm[:, 2]
    reg.origin = np.array([10., 20., 1.]); reg.ds = .2; reg.height = 1.6
    reg.height_variance = .15**2
    reg.prior_pose = np.eye(4); reg.prior_pose[:3, 3] = [10., 20., 2.6]
    C = np.array([[.36, .04, .002], [.04, .49, -.001], [.002, -.001, .03]])
    reg.prior_precision = np.linalg.inv(C)
    target = rng.normal(size=(count, 3))
    ids = np.arange(count); ids[matches:] = -1
    reg.feature_nn = SimpleNamespace(points=target,
        query=lambda q, radius: (ids.copy(), np.where(ids >= 0,
            np.minimum(((q-target)**2).sum(axis=1), radius**2), radius**2)))
    normal = rng.normal(size=(count, 3)); normal /= np.linalg.norm(normal, axis=1)[:, None]
    reg.geometry = SimpleNamespace(ensure=lambda _: None, reliable=np.ones(count, dtype=bool),
        normal=normal, center=target+.01, confidence=rng.uniform(.2, 1., count))
    return reg


@pytest.mark.parametrize('matches', [0, 12, 90, 128])
@pytest.mark.parametrize('include_prior', [False, True])
def test_fast_cost_and_derivatives_equal_full_diagnostics(matches, include_prior):
    scope = load_methods(); reg = fixture_registration(scope, matches=matches)
    x = np.array([.1, -.2, .03, np.log(8.)])
    full = reg.evaluate(x, 1.5, .5, derivatives=True, include_prior=include_prior)
    # Check rematched and fixed-plane paths, including unmatched points and the
    # insufficient-match derivative branch. Diagnostic removal cannot change H/g.
    for association in (None, full['association']):
        detail = reg.evaluate(x, 1.5, .5, derivatives=True, include_prior=include_prior, association=association)
        fast = reg.evaluate(x, 1.5, .5, derivatives=True, include_prior=include_prior,
                            association=association, diagnostics=False)
        for name in ('cost', 'data_cost', 'prior_cost', 'matches', 'H', 'g', 'data_H'):
            np.testing.assert_array_equal(fast[name], detail[name])
        # A trial must not sort medians; avoid even allocating its unused Jacobian.
        original_transform = scope['transform4']
        def no_jacobian(u, xx, *, jacobian=True):
            assert not jacobian
            return original_transform(u, xx, jacobian=jacobian)
        scope['transform4'] = no_jacobian
        scope['cp'].median = lambda *_: (_ for _ in ()).throw(AssertionError('diagnostic sort in trial'))
        trial = reg.evaluate(x, 1.5, .5, include_prior=include_prior,
                             association=association, diagnostics=False)
        np.testing.assert_array_equal(trial['cost'], detail['cost'])
        scope['transform4'] = original_transform
        del scope['cp'].median


def test_jacobian_optional_keeps_identical_transformed_points():
    scope = load_methods(); rng = np.random.default_rng(19)
    for n in (0, 1, 6000):
        u = rng.normal(size=(n, 3)); x = np.array([.12, -.3, .07, np.log(8.)])
        q, J = scope['transform4'](u, x)
        fast, unused = scope['transform4'](u, x, jacobian=False)
        np.testing.assert_array_equal(q, fast)
        assert unused is None and J.shape == (n, 3, 4)


def test_device_mask_algebra_preserves_edge_cases_and_counts():
    tree = ast.parse((ROOT/'track2/camera.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'device_mask')
    scope = dict(cp=NumpyAlgebra())
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'device_mask', 'exec'), scope)
    K = np.array([[2., 0, 2.], [0, 2., 2.], [0, 0, 1.]])
    mask = np.ones((5, 5), dtype=np.uint8); mask[2, 2] = 0
    raw = np.array([[1., 0, 0], [1., .5, 0], [1., -.5, -.5],
        [0., 1, 1], [-1., 0, 0], [1., 100, 0], [np.nan, 0, 0]])
    with np.errstate(invalid='ignore'):
        out, d = scope['device_mask'](raw, K, mask)
    np.testing.assert_array_equal(out, raw[[1, 2]])
    assert (d['input_points'], d['kept_points'], d['removed_points']) == (7, 2, 5)
    empty, d = scope['device_mask'](np.empty((0, 3)), K, mask)
    assert empty.shape == (0, 3) and d['removed_points'] == 0


def test_world_display_input_is_not_transformed_twice():
    from track2.visualization import Track2RvizPublisher
    v = Track2RvizPublisher(); clouds = []; poses = []
    v.current_cloud_pub = SimpleNamespace(publish=clouds.append)
    v.pose_pub = SimpleNamespace(publish=poses.append)
    v.registration_status_pub = v.status_pub = SimpleNamespace(publish=lambda _: None)
    v.tf = SimpleNamespace(sendTransform=lambda _: None)
    T = np.eye(4); T[:3, 3] = [10., 20., 3.]
    T[:3, :3] = Rotation.from_euler('z', .3).as_matrix()
    raw = np.array([[1., 0., 0.], [2., 1., -.2]])
    world = (raw @ T[:3, :3].T*8.+T[:3, 3]).astype(np.float32)
    before = world.copy()
    try:
        v.publish_frame(world, T, 8., 1_000_000_123, accepted=True,
                        reason='feature_metric_update', cloud_in_map=True)
        actual = np.frombuffer(clouds[0].data, dtype='<f4').reshape(-1, 4)[:, :3]
        np.testing.assert_array_equal(actual, world)
        np.testing.assert_array_equal(world, before)
        assert poses[0].pose.position.x == T[0, 3]
        assert poses[0].header.stamp.nanosec == 123
    finally:
        v.close()
