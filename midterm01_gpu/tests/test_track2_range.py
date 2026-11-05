"""Generated range/scale contract: a shorter physical radius removes far structure."""
import numpy as np
import pytest
from track2 import parameters as p
from test_track2_metric_scale import scene, truth, raw_cloud, TEST_COV


@pytest.mark.parametrize('raw_scale', [8., 10.])
def test_metric_range_keeps_near_structure_and_excludes_far_structure(monkeypatch, raw_scale):
    import cupy as cp
    from track2.metric_registration import MetricRegistration
    monkeypatch.setattr(p, 'FUSION_PROFILE', 'v52_directional_memory')
    y, z = np.meshgrid(np.arange(-4., 4., .3), np.arange(.6, 3., .3))
    far = np.column_stack([np.full(y.size, 34.), y.ravel(), z.ravel()])
    xyz = np.concatenate([scene(), far]); T = truth()
    counts = {}; maxima = {}
    for radius in [40., 30.]:
        monkeypatch.setattr(p, 'FEATURE_MAX_RANGE_M', radius)
        reg = MetricRegistration(cp.asarray(xyz), T)
        d = reg.prepare(cp.asarray(raw_cloud(xyz, T, raw_scale)), T, TEST_COV)
        distance = cp.linalg.norm(reg.raw_features, axis=1)*d['scale_seed']
        assert bool(cp.all(distance <= radius+1e-10))
        counts[radius] = len(distance)
        maxima[radius] = float(distance.max())
    assert maxima[40.] > 32. and maxima[30.] < 30.
    assert counts[40.] > counts[30.] > 100
