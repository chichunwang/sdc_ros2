"""Generated hypothesis selection and causal posterior validation only."""
import numpy as np
import pytest
from track2 import parameters as p


@pytest.fixture(autouse=True)
def historical_v511_mechanisms(monkeypatch):
    monkeypatch.setattr(p, 'V512_ENABLED', False)
    monkeypatch.setattr(p, 'V511_ENABLED', True)
    monkeypatch.setattr(p, 'V5121_ENABLED', False)
    monkeypatch.setattr(p, 'V513_ENABLED', False)


def selector(monkeypatch, *, tracking=False, first_ok=False, all_fail=False):
    from track2.metric_registration import MetricRegistration
    from types import SimpleNamespace
    monkeypatch.setattr(p, 'STARTUP_INCLUDE_GIVEN_YAW', False)
    monkeypatch.setattr(p, 'STARTUP_INCLUDE_MAP_AXIS', False)
    monkeypatch.setattr(p, 'STARTUP_HEIGHT_SEED_SIGMAS', (0.,))
    obj = MetricRegistration.__new__(MetricRegistration)
    obj.seed = np.array([0.,0.,0.,np.log(8.)]); obj.height=1.6
    obj.ds=.2; obj.height_variance=.0225; obj.prior_covariance=np.diag([1.,1.,.01])
    obj.yaw_search_bound=.4
    obj.correction_filter=SimpleNamespace(last_update=1 if tracking else None)
    obj.state_for_correction=lambda prior,c,scale:np.r_[c,np.log(scale)]
    obj.valid=lambda x:True
    obj.evaluate=lambda x,*a,**k:dict(cost=float(x[2]**2))
    solved=[]
    def solve(x):
        solved.append(x.copy())
        accepted = (first_ok or x[2]!=0.) and not all_fail
        return dict(x=x.copy(),geometry_accepted=accepted,observable=dict(rank=1))
    obj._solve_candidate=solve
    obj.pose=lambda x:(x,8.)
    obj.score_pose=lambda x,s:dict(feature_cost=.001 if x[2]==0 else .02 if x[2]>0 else .03)
    return obj,solved


def test_startup_refines_every_seed_before_quality_first_common_selection(monkeypatch):
    obj,calls=selector(monkeypatch)
    result=obj._select_current_candidates(np.eye(4),obj.seed.copy())
    assert len(calls)==3
    assert result['geometry_accepted'] and result['x'][2] > 0
    assert result['selection']['fully_refined']==3
    # The cheaper invalid basin must not win over a qualified result.
    assert result['common_score'] > .001


def test_startup_candidate_order_does_not_change_selected_basin(monkeypatch):
    obj,_=selector(monkeypatch);a=obj._select_current_candidates(np.eye(4),obj.seed.copy())
    monkeypatch.setattr(p,'MAP_STARTUP_YAW_SEED_SIGMAS',tuple(reversed(p.MAP_STARTUP_YAW_SEED_SIGMAS)))
    obj,_=selector(monkeypatch);b=obj._select_current_candidates(np.eye(4),obj.seed.copy())
    np.testing.assert_array_equal(a['x'],b['x'])


def test_all_failed_candidates_do_not_claim_a_map_anchor(monkeypatch):
    obj,_=selector(monkeypatch,all_fail=True)
    result=obj._select_current_candidates(np.eye(4),obj.seed.copy())
    assert not result['geometry_accepted']
    assert not any(c['geometry_accepted'] for c in result['selection']['candidates'])


def test_good_tracking_keeps_one_solve_and_failure_retries_current_carry(monkeypatch):
    obj,calls=selector(monkeypatch,tracking=True,first_ok=True)
    carry=obj.seed.copy();carry[2]=.1
    result=obj._select_current_candidates(np.eye(4),carry)
    assert len(calls)==1 and not result['selection']['startup']
    obj,calls=selector(monkeypatch,tracking=True)
    result=obj._select_current_candidates(np.eye(4),carry)
    assert len(calls)==2 and result['geometry_accepted']


def test_disabled_selection_retains_one_legacy_seed_solve(monkeypatch):
    obj,calls=selector(monkeypatch)
    monkeypatch.setattr(p,'V511_ENABLED',False)
    result=obj._select_current_candidates(np.eye(4),obj.seed.copy())
    assert len(calls)==1 and not result['geometry_accepted']
    assert 'selection' not in result


def test_cuda_generated_startup_commits_once_and_validates_actual_posterior():
    import cupy as cp
    from track2.metric_registration import MetricRegistration
    from test_track2_metric_scale import scene,truth,raw_cloud,TEST_COV
    T=truth();prior=T.copy();prior[:2,3]+=[.2,-.15]
    reg=MetricRegistration(cp.asarray(scene()),T)
    out,scale,d=reg.refine(cp.asarray(raw_cloud(scene(),T,8.)),prior,1_000_000_000,prior_covariance=TEST_COV)
    assert d['accepted'],d
    selection=d['candidate_selection']
    assert selection['startup'] and selection['fully_refined']>=3
    assert sum(t['selected'] for t in selection['candidates'])==1
    v=d['correction_filter']['proposal_validation']
    assert v['accepted'] and v['proposal_data_cost'] <= v['baseline_data_cost']+1e-12
    assert v['reason']=='posterior_fixed_plane_loss_not_increased'
    assert reg.correction_filter.last_stamp==1_000_000_000
    assert np.isfinite(out).all() and np.isfinite(scale)
