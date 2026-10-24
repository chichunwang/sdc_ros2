import numpy as np
from track3.preprocessing import mean_knn_distance,filter_camera_cloud
from track3.gpu import cp


def test_cuda_knn_matches_exact_distances_without_self():
    p=np.random.default_rng(23).normal(0,.01,(65,3))+[1000.,10.,2.]
    distance=np.linalg.norm(p[:,None]-p[None,:],axis=2)
    np.fill_diagonal(distance,np.inf)
    expected=np.sort(distance,axis=1)[:,:8].mean(axis=1)
    np.testing.assert_allclose(cp.asnumpy(mean_knn_distance(p,k=8)),expected,atol=2e-8)


def test_depth_aware_filter_keeps_distant_surface_and_removes_isolated_point():
    rng=np.random.default_rng(91)
    near=rng.normal(0,.02,(180,3))+[2.,0.,0.]
    far=near*10
    p=np.r_[near,far,[[2.,8.,7.]]];original=p.copy()
    filtered,info,mask=filter_camera_cloud(p,dict(k=8,sigma_multiplier=3.))
    keep=cp.asnumpy(mask)
    assert not keep[-1]
    assert keep[:180].mean()>.9 and keep[180:360].mean()>.9
    np.testing.assert_array_equal(p,original)
    assert info['kept_points']==len(filtered) and info['removed_points']>=1
