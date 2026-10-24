"""CUDA equivalence against the previous vectorized distance definitions."""
import numpy as np
import pytest
from track3.gpu import cp, VoxelModel, keys, NEIGHBORS
from track3.search import closest_pixels


def reference(model,q,mode):
    cells=cp.floor(q/model.leaf).astype(cp.int64)
    lookup=keys(cells[:,None,:]+NEIGHBORS[None,:,:])
    ids=cp.minimum(cp.searchsorted(model.keys,lookup),len(model.keys)-1)
    present=model.keys[ids]==lookup
    residual=model.means[ids]-q[:,None,:]
    d2=cp.sum(residual**2,axis=2)
    present&=d2<(1.5*model.leaf)**2
    metric=cp.einsum('nci,ncij,ncj->nc',residual,model.precision[ids],residual) if mode=='ndt' else d2
    metric=cp.where(present,metric,cp.inf)
    chosen=cp.argmin(metric,axis=1);row=cp.arange(len(q))
    return ids[row,chosen],cp.isfinite(metric[row,chosen])


@pytest.mark.parametrize('mode',['ndt','icp'])
def test_gaussian_search_matches_previous_definition(mode):
    rng=np.random.default_rng(31)
    center=np.indices((12,8,4)).reshape(3,-1).T*1.2+[1000,-200,10]
    points=np.repeat(center,12,axis=0)+rng.normal(0,.05,(len(center)*12,3))
    model=VoxelModel(points,1.)
    q=cp.asarray(rng.uniform([999,-201,9],[1016,-189,17],(513,3)))
    expected,valid=reference(model,q,mode);actual,good=model.correspond(q,mode)
    np.testing.assert_array_equal(cp.asnumpy(good),cp.asnumpy(valid))
    np.testing.assert_array_equal(cp.asnumpy(actual[good]),cp.asnumpy(expected[valid]))


def test_cell_ties_empty_queries_and_signed_range_are_preserved():
    model=VoxelModel.__new__(VoxelModel);model.leaf=1.
    model.means=cp.asarray([[.25,.25,.25],[1.25,.25,.25]],dtype=cp.float64)
    model.keys=keys(cp.floor(model.means))
    model.precision=cp.tile(cp.eye(3,dtype=cp.float64),(2,1,1))
    query=cp.asarray([[.75,.25,.25]])
    for mode in ['ndt','icp']:
        ids,good=model.correspond(query,mode)
        assert bool(good[0]) and int(ids[0])==0
    assert len(model.correspond(cp.empty((0,3)))[0])==0
    with pytest.raises(ValueError,match='21-bit'):
        model.correspond(cp.asarray([[2**20-1.,0,0]]))


@pytest.mark.parametrize('count',[1,3,129,2003])
def test_pixel_search_exact_indices_and_distances(count):
    rng=np.random.default_rng(56)
    source=rng.normal(0,500,(23,4));target=rng.normal(0,500,(count,4))
    q=cp.asarray(source)[:,::2];p=cp.asarray(target)[:,::2]
    distance=cp.sum((q[:,None,:]-p[None,:,:])**2,axis=2)
    expected=cp.argmin(distance,axis=1)
    ids,d2=closest_pixels(q,p)
    np.testing.assert_array_equal(cp.asnumpy(ids),cp.asnumpy(expected))
    np.testing.assert_allclose(cp.asnumpy(d2),cp.asnumpy(distance[cp.arange(len(q)),expected]),rtol=1e-14,atol=1e-10)


def test_pixel_ties_choose_first_original_index():
    q=cp.asarray([[0.,0.],[1.,0.]])
    p=cp.asarray([[1.,0.],[-1.,0.],[1.,0.]])
    ids,d2=closest_pixels(q,p)
    np.testing.assert_array_equal(cp.asnumpy(ids),[0,0])
    np.testing.assert_array_equal(cp.asnumpy(d2),[1.,0.])
    assert len(closest_pixels(cp.empty((0,2)),p)[0])==0
    with pytest.raises(ValueError):closest_pixels(q,cp.empty((0,2)))
