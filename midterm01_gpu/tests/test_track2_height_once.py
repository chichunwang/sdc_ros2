"""An independent physical-height Gaussian model must not reuse its prior."""
import numpy as np
import pytest
from track2 import parameters as p
from track2.observability import height_observation
from track2.height_filter import fuse_height


@pytest.mark.parametrize('prior_height',[1.50,1.75])
@pytest.mark.parametrize('prior_std',[.02,.1,.4])
def test_data_height_invariant_to_prior_and_once_fusion_matches_bayes(prior_height,prior_std):
    rng=np.random.default_rng(5165)
    A=rng.normal(size=(12,4))@np.diag([2.,2.,50.,2.])
    H=A.T@A
    truth=np.array([.03,-.02,.002,1.63])
    y=A@truth+np.linspace(-.01,.01,len(A))
    b=A.T@y
    prior=np.array([-.1,.1,-.01,prior_height])
    variance=np.diag([.3**2,.3**2,.03**2,prior_std**2])
    map_state=np.linalg.solve(H+np.linalg.inv(variance),b+np.linalg.solve(variance,prior))
    height=map_state[3]
    J=np.diag([1.,1.,1.,height])  # physical-height derivative w.r.t. logscale
    data_H=J.T@H@J
    data_g=J.T@(H@map_state-b)
    observed,R,_=height_observation(data_H,data_g,height,dict(coverage=.5,median_plane_residual_m=.1))
    nuisance=np.linalg.solve(H[:3,:3],H[:3,3])
    information=H[3,3]-H[3,:3]@nuisance
    independent_height=(b[3]-H[3,:3]@np.linalg.solve(H[:3,:3],b[:3]))/information
    independent_variance=1/information+p.HEIGHT_OBSERVATION_FLOOR_M**2
    np.testing.assert_allclose(observed,independent_height,rtol=0,atol=1e-12)
    np.testing.assert_allclose(R,independent_variance,rtol=0,atol=1e-12)
    mean,cov,decision=fuse_height(prior_height,prior_std**2,observed,R)
    assert decision['accepted']
    K=prior_std**2/(prior_std**2+independent_variance)
    np.testing.assert_allclose(mean,prior_height+K*(independent_height-prior_height),atol=1e-12)
    np.testing.assert_allclose(cov,1/(1/prior_std**2+1/independent_variance),atol=1e-12)

