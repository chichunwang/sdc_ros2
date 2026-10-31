"""Generated CUDA equivalence; execute only while holding the shared GPU lock."""
import numpy as np
import pytest
from test_track2_metric_scale import scene, truth, raw_cloud, TEST_COV


@pytest.fixture
def cp():
    import cupy
    assert cupy.cuda.runtime.getDeviceCount() > 0
    return cupy


@pytest.mark.parametrize('derivatives', [False, True])
def test_fast_cuda_objective_matches_complete_diagnostics(cp, derivatives):
    from track2.metric_registration import MetricRegistration
    T = truth(); reg = MetricRegistration(cp.asarray(scene()), T)
    reg.prepare(cp.asarray(raw_cloud(scene(), T, 8.)), T, TEST_COV)
    x = reg.seed+np.array([.12, -.08, .002, .004])
    complete = reg.evaluate(x, .75, .3, derivatives=derivatives)
    for fixed in (None, complete['association']):
        full = reg.evaluate(x, .75, .3, derivatives=derivatives, association=fixed)
        fast = reg.evaluate(x, .75, .3, derivatives=derivatives, association=fixed, diagnostics=False)
        for key in ('cost', 'data_cost', 'prior_cost', 'H', 'g', 'data_H'):
            np.testing.assert_array_equal(fast[key], full[key])
        if derivatives:
            assert fast['matches'] == full['matches']


def test_cuda_mask_stays_on_device_and_matches_previous_host_result(cp):
    from track2.camera import device_mask
    from track3.visual_prior import filter_by_camera_mask
    rng = np.random.default_rng(19)
    raw = rng.normal(size=(6000, 3)); raw[:, 0] += 2.
    K = np.array([[12., 0, 20.], [0, 12., 10.], [0, 0, 1.]])
    mask = rng.integers(0, 2, (21, 41), dtype=np.uint8)*255
    expected, previous = filter_by_camera_mask(raw, K, mask)
    actual, diagnostic = device_mask(raw, K, cp.asarray(mask))
    assert isinstance(actual, cp.ndarray)
    np.testing.assert_array_equal(cp.asnumpy(actual), expected)
    assert diagnostic == previous
    empty, d = device_mask(np.empty((0, 3)), K, cp.asarray(mask))
    assert isinstance(empty, cp.ndarray) and empty.shape == (0, 3)
    assert d['input_points'] == d['kept_points'] == d['removed_points'] == 0
