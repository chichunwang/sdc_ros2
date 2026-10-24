import json
from pathlib import Path
import numpy as np
import pytest
from track3.gpu import Registration, VoxelModel, SparseVoxelModelError
from test_core import cloud


def config():
    cfg = json.loads((Path(__file__).resolve().parents[1]/'config/track3.json').read_text())['registration']
    cfg['stages'] = [dict(mode='icp', voxel_m=1., iterations=30)]
    cfg['submap_half_extent_m'] = [20, 20, 10]
    return cfg


def test_sparse_map_is_rejection_not_fatal():
    p = np.indices((6, 6, 4)).reshape(3, -1).T.astype(float)*1.1
    result = Registration(p, config()).align(p/3, np.eye(4), 3)
    assert not result['accepted']
    assert result['reason'] == 'insufficient_populated_map_voxels'
    assert result['model_diagnostics']['populated_voxels'] == 0
    assert result['model_diagnostics']['point_count'] == len(p)
    np.testing.assert_array_equal(result['T'], np.eye(4))
    with pytest.raises(SparseVoxelModelError):
        VoxelModel(np.empty((0, 3)), 1.)


def test_sparse_rebuild_preserves_cache_and_allows_reentry():
    dense = cloud()
    sparse = np.indices((6,6,4)).reshape(3,-1).T.astype(float)*1.1+[150,0,0]
    reg = Registration(np.r_[dense, sparse], config())
    assert reg.align(dense/3, np.eye(4), 3)['accepted']
    old_models = reg.models
    old_center = reg.center.copy()
    away = np.eye(4); away[0,3] = 150
    rejected = reg.align(dense/3, away, 3)
    assert rejected['reason'] == 'insufficient_populated_map_voxels'
    assert reg.models is old_models
    np.testing.assert_array_equal(reg.center, old_center)
    assert reg.align(dense/3, np.eye(4), 3)['accepted']


def test_unrelated_map_index_error_is_not_suppressed():
    huge = np.tile([1e10, 1e10, 1e10], (150,1))
    prior = np.eye(4); prior[:3,3] = huge[0]
    with pytest.raises(ValueError, match='21-bit'):
        Registration(huge,config()).align(np.ones((100,3)),prior,3)
