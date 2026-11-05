"""Generated CUDA scene integration; no real Track2 frames."""
import numpy as np
import cupy as cp
import pytest
from track2 import parameters as p
from track2.metric_registration import MetricRegistration
from test_track2_metric_scale import scene,truth,raw_cloud,TEST_COV


def test_generated_cuda_data_likelihood_uses_single_commit_and_both_posterior_checks():
    T=truth();prior=T.copy();prior[:2,3]+=[.2,-.15]
    reg=MetricRegistration(cp.asarray(scene()),T)
    out,scale,d=reg.refine(cp.asarray(raw_cloud(scene(),T,8.)),prior,1_000_000_000,prior_covariance=TEST_COV)
    assert d['geometry_accepted'] and d['accepted'],d
    assert d['map_measurement_model']=='current_data_likelihood_prior_used_once'
    assert d['map_likelihood']['likelihood_model'].startswith('current fixed-association data-only')
    assert d['correction_filter']['local_linear_likelihood']
    v=d['correction_filter']['proposal_validation']
    assert v['accepted'] and v['fixed_plane_loss_valid'] and v['rematched_loss_valid']
    assert v['proposal_data_cost']<=v['baseline_data_cost']+1e-12
    assert v['proposal_common_feature_cost']<=v['baseline_common_feature_cost']+1e-12
    assert reg.correction_filter.last_update==1_000_000_000
    assert len(d['data_gradient'])==4 and np.isfinite(out).all() and np.isfinite(scale)


@pytest.mark.parametrize('rank,reason,initial_ok,expected_extra',[
    (1,'iteration_limit',False,2),
    (1,'line_search_failed',False,2),
    (0,'iteration_limit',False,0),
    (1,'stationary',False,0),
    (1,'iteration_limit',True,0),
])
def test_tracking_yaw_sigma_recovery_only_for_informative_nonconvergence(monkeypatch,rank,reason,initial_ok,expected_extra):
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from track2 import parameters as p
    obj=MetricRegistration.__new__(MetricRegistration)
    obj.seed=np.array([0.,0.,0.,np.log(8.)]);obj.prior_covariance=np.diag([1.,1.,.01])
    obj.correction_filter=SimpleNamespace(last_update=1);obj.yaw_search_bound=.4
    obj.valid=lambda x:abs(x[2])<=.4
    obj.state_for_correction=lambda prior,c,scale:np.r_[c,np.log(scale)]
    obj.evaluate=lambda x,*a,**k:dict(cost=float(x[2]**2))
    def pose(x):
        T=np.eye(4);T[:3,:3]=Rotation.from_euler('z',x[2]).as_matrix();T[:2,3]=x[:2]
        return T,float(np.exp(x[3]))
    obj.pose=pose;calls=[]
    def solve(x):
        calls.append(x.copy())
        return dict(x=x.copy(),geometry_accepted=bool(initial_ok or abs(x[2])>=.05),
            observable=dict(rank=rank),stages=[dict(reason=reason)],
            final=dict(matches=100,angular_support=20.),support_required=4.)
    obj._solve_candidate=solve
    obj.score_pose=lambda T,s:dict(feature_cost=abs(np.arctan2(T[1,0],T[0,0])-.1))
    result=obj._select_current_candidates(np.eye(4),obj.seed.copy())
    assert result['selection']['nonconvergence_recovery_solves']==expected_extra
    assert len(calls)==1+expected_extra
    if expected_extra:
        assert result['geometry_accepted'] and result['x'][2]==pytest.approx(.1)


@pytest.fixture(autouse=True)
def historical_model_profile(monkeypatch):
    monkeypatch.setattr(p, 'V511_ENABLED', True)
    monkeypatch.setattr(p, 'V5121_ENABLED', False)
    monkeypatch.setattr(p, 'V513_ENABLED', False)
    monkeypatch.setattr(p, 'V512_ENABLED', True)
