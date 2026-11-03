"""Generated physical-height likelihoods; no real frames or hidden truth."""
import numpy as np
import pytest
from track2 import parameters as p
from track2.observability import height_observation
from track2.height_filter import fuse_height

QUALITY=dict(median_plane_residual_m=.05)


@pytest.mark.parametrize('regularizer_strength',[.05,1.,50.])
def test_height_data_measurement_recovers_truth_independent_of_solver_prior(regularizer_strength):
    # Generative linear observations in [XY,yaw,height], with pose/height coupling.
    A=np.array([[4.,1.,0.,2.],[0.,3.,1.,-1.],[1.,0.,30.,.5],[0.,0.,0.,10.]])
    D=A.T@A;truth=np.array([.2,-.1,.015,1.7]);prior=np.array([0.,0.,0.,1.57])
    regularizer=np.diag([1.,2.,20.,4.])*regularizer_strength
    candidate=np.linalg.solve(D+regularizer,D@truth+regularizer@prior)
    J=np.diag([1.,1.,1.,candidate[3]]) # dh/dlogscale=h
    m,R,d=height_observation(J.T@D@J,J.T@D@(candidate-truth),candidate[3],QUALITY)
    assert m == pytest.approx(truth[3],abs=1e-13)
    assert R == pytest.approx(np.linalg.inv(D)[3,3]+p.HEIGHT_OBSERVATION_FLOOR_M**2,rel=1e-12)
    assert abs(candidate[3]-truth[3])>1e-6
    mu,P,diag=fuse_height(prior[3],.15**2,m,R)
    oracle=(prior[3]/.15**2+truth[3]/R)/(1/.15**2+1/R)
    assert diag['accepted'] and diag['variance_inflation']==1.
    assert mu == pytest.approx(oracle,abs=1e-14)
    assert P == pytest.approx(1/(1/.15**2+1/R),abs=1e-14)


def test_reusing_height_map_candidate_under_corrects_but_claims_same_variance():
    initial,variance,truth,R=1.57,.15**2,1.7,.1**2
    once,P,good=fuse_height(initial,variance,truth,R)
    old,Pold,_=fuse_height(initial,variance,once,R)
    K=variance/(variance+R)
    assert once == pytest.approx(initial+K*(truth-initial))
    assert old == pytest.approx(initial+K*K*(truth-initial))
    assert abs(old-truth)>abs(once-truth)
    assert Pold == pytest.approx(P) # confidence claimed despite smaller correction


def test_height_translation_gauge_has_no_fake_height_observation():
    row=np.array([1.,0.,0.,1.]);D=100*np.outer(row,row)+np.diag([0.,100.,1000.,0.])
    m,R,d=height_observation(D,np.zeros(4),1.6,QUALITY)
    assert m is None and R is None and not d['observable']


def test_height_nuisance_reparametrization_preserves_measurement_and_noise():
    D=np.array([[8.,1.,0.,2.],[1.,5.,0.,1.],[0.,0.,30.,.3],[2.,1.,.3,10.]])
    g=np.array([.1,-.1,.01,-.2])
    a=height_observation(D,g,1.6,QUALITY)
    B=np.eye(4);B[:3,:3]=np.array([[100.,3.,0.],[0.,.1,0.],[0.,0.,2.]])
    b=height_observation(B.T@D@B,B.T@g,1.6,QUALITY)
    np.testing.assert_allclose(a[:2],b[:2],rtol=1e-12,atol=1e-12)


def test_height_large_innovation_rejects_and_moderate_is_downweighted():
    m,P,d=fuse_height(1.6,.01,2.8,.01)
    assert not d['accepted'] and m==1.6 and P==.01
    m,P,d=fuse_height(1.6,.001,1.76,.004)
    assert d['accepted'] and d['variance_inflation']>1
    K=d['gain'];R=.004*d['variance_inflation']
    assert P == pytest.approx((1-K)**2*.001+K*K*R)


@pytest.mark.parametrize('height',[0.,np.nan])
def test_invalid_height_is_not_silently_used(height):
    with pytest.raises(ValueError):height_observation(np.eye(4),np.zeros(4),height,QUALITY)


def test_cuda_emitted_scale_and_position_use_same_current_height_posterior():
    import cupy as cp
    from track2.metric_registration import MetricRegistration
    from test_track2_metric_scale import scene,truth,raw_cloud,TEST_COV
    T=truth();reg=MetricRegistration(cp.asarray(scene()),T)
    pose,scale,d=reg.refine(cp.asarray(raw_cloud(scene(),T,8.)),T,1_000_000_000,prior_covariance=TEST_COV)
    assert d['accepted'] and d['height_update']['accepted']
    assert scale*reg.ds == pytest.approx(reg.height,abs=1e-12)
    assert reg.nm@pose[:3,3]+reg.dm == pytest.approx(reg.height,abs=1e-12)
    assert d['height_likelihood']['fixed_association_loss_valid']
    assert d['scale_source']=='current_data_height_posterior_and_ground'


def test_cuda_bad_height_center_does_not_poison_memory(monkeypatch):
    import cupy as cp
    import track2.metric_registration as module
    from test_track2_metric_scale import scene,truth,raw_cloud,TEST_COV
    original=module.height_observation
    def bad(*args,**kwargs):
        m,R,d=original(*args,**kwargs)
        return m+.15,R,d
    monkeypatch.setattr(module,'height_observation',bad)
    T=truth();reg=module.MetricRegistration(cp.asarray(scene()),T)
    pose,scale,d=reg.refine(cp.asarray(raw_cloud(scene(),T,8.)),T,1_000_000_000,prior_covariance=TEST_COV)
    # v5.4 validates the joint emitted pose/height; bad height must never
    # commit, and the coupled pose proposal may also be rejected safely.
    assert not d['height_update']['accepted']
    assert d['height_update']['reason']=='height_likelihood_loss_increased'
    assert d['height_anchor_after_m']==pytest.approx(d['planes']['height_prior_m'])
    assert scale*reg.ds==pytest.approx(d['planes']['height_prior_m'])

@pytest.fixture(autouse=True)
def historical_model(monkeypatch):
    """Isolate retained v5.3-v5.5 experiments from production v5.6."""
    monkeypatch.setattr(p, 'FUSION_PROFILE', 'v55_experimental')
