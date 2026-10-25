"""Startup basin selection on generated geometry, never Track3 observations."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from track3.gpu import Registration
from track3.initialization import GroundInitializer


def scene():
    rng=np.random.default_rng(94)
    ground=rng.uniform([-12,-8,0],[14,8,0],(7000,3))
    wall=rng.uniform([9,-6,.2],[9,6,5],(3500,3))
    side=rng.uniform([2,5,.2],[10,5,3],(2500,3))
    m=np.r_[ground,wall,side]
    truth=np.eye(4);truth[:3,3]=[.5,-.3,2.]
    truth[:3,:3]=Rotation.from_euler('xyz',[.04,-.03,.2]).as_matrix()
    source=(m-truth[:3,3])@truth[:3,:3]/7.
    source=source[(source[:,0]>.2)&(np.abs(source[:,1]/source[:,0])<1.3)]
    prior=truth.copy();prior[:3,3]+=[.3,-.2,.3]
    prior[:3,:3]=Rotation.from_euler('xyz',[.2,-.18,.12]).as_matrix()
    cfg=json.loads((Path(__file__).resolve().parents[1]/'config/track3.json').read_text())['registration']
    cfg['preprocessing']['enabled']=False
    # One fine stage isolates basin selection from the coarse NDT heuristic.
    cfg['stages']=[dict(mode='icp',voxel_m=1.,iterations=60)]
    return m,source,truth,prior,cfg


def test_common_feature_score_penalizes_missing_support_and_uses_raw_scale():
    m,source,truth,prior,cfg=scene()
    init=GroundInitializer(m,cfg['initializer'])
    init.align(source,prior,3.)
    good=init.score_pose(truth,7.)
    tilted=truth.copy();tilted[:3,:3]=Rotation.from_rotvec([0,0,.18]).as_matrix()@truth[:3,:3]
    tilted[:3,3]+=[1.,-2.,0.]
    assert good['cost']<init.score_pose(tilted,7.)['cost']
    assert good['cost']<init.score_pose(truth,3.)['cost']
    away=truth.copy();away[:3,3]+=[100.,100.,100.]
    bad=init.score_pose(away,7.)
    expected=bad['robust_m']**2*np.log1p((bad['radius_m']/bad['robust_m'])**2)
    assert bad['source_features']==good['source_features']
    assert bad['coverage']==0 and bad['median_capped_distance_m']==bad['radius_m']
    assert bad['cost']==pytest.approx(expected)
    assert good['coverage']>.95


def test_alternate_seed_can_recover_after_initial_winner_fails(tmp_path,monkeypatch):
    m,source,truth,prior,cfg=scene()
    # A wrong local basin can have the best grounded proxy cost. A previously
    # unconverged alternate must still get full refinement before selection.
    wrong=truth.copy();wrong[:3,3]+=[24.,0.,0.]
    near=truth.copy();near[:3,3]+=[.15,-.12,.04]
    align=GroundInitializer.align
    context={}
    def proposals(self,*args):
        if not context:
            result=align(self,*args)  # Genuine fixed CUDA feature sample and map.
            context.update(result=result,features={k:v for k,v in vars(self).items() if k.startswith('feature_')})
        else:
            vars(self).update(context['features'])
            result=dict(context['result'])
        result.update(T=wrong,scale=7.,accepted=True,candidates=[
            dict(T=wrong.tolist(),scale=7.,cost=0.,converged=True),
            dict(T=near.tolist(),scale=7.,cost=.1,converged=False)])
        return result
    monkeypatch.setattr(GroundInitializer,'align',proposals)
    reg=Registration(m,cfg)
    single=reg._align_seed(source,wrong,7.,True,False)
    assert not single['accepted']
    result=reg.align(source,prior,3.,initialize=True)
    assert result['accepted']
    assert np.linalg.norm(result['T'][:3,3]-truth[:3,3])<.2
    assert Rotation.from_matrix(result['T'][:3,:3]@truth[:3,:3].T).magnitude()<.03
    assert abs(result['scale']-7.)<.1
    selection=result['initialization']['selection']
    assert selection['selected_candidate_index']==1 and selection['evaluated_count']==2
    assert not result['initialization']['candidates'][1]['converged']
    assert not result['initialization']['refinements'][0]['eligible']
    np.testing.assert_array_equal(reg.center,prior[:3,3])
    # Reversing the input hypotheses cannot change the geometric answer or
    # the shared submap. Timing fields and candidate indices may differ.
    def reversed_proposals(self,*args):
        result=proposals(self,*args)
        result['candidates']=result['candidates'][::-1]
        return result
    monkeypatch.setattr(GroundInitializer,'align',reversed_proposals)
    reverse=Registration(m,cfg).align(source,prior,3.,initialize=True)
    np.testing.assert_allclose(reverse['T'],result['T'],atol=1e-12,rtol=0)
    assert reverse['scale']==pytest.approx(result['scale'],abs=1e-12)
    (tmp_path/'synthetic_recovery.json').write_text(json.dumps(dict(
        fixture='generated_two_basin_proposals_not_real_localization',
        truth_T=truth.tolist(),truth_scale=7.,single_seed_T=single['T'].tolist(),
        single_seed_accepted=single['accepted'],selected_T=result['T'].tolist(),
        selected_scale=result['scale'],initialization=result['initialization']),indent=2))


def test_generated_default_multihypothesis_startup(tmp_path):
    m,source,truth,prior,cfg=scene()
    result=Registration(m,cfg).align(source,prior,3.,initialize=True)
    assert result['accepted']
    assert np.linalg.norm(result['T'][:3,3]-truth[:3,3])<.2
    assert Rotation.from_matrix(result['T'][:3,:3]@truth[:3,:3].T).magnitude()<.03
    assert abs(result['scale']-7.)<.1
    records=result['initialization']['refinements']
    assert len(records)==len(result['initialization']['candidates'])
    chosen=result['initialization']['selection']['selected_candidate_index']
    assert records[chosen]['eligible']
    assert records[chosen]['feature_score']['cost']==min(
        r['feature_score']['cost'] for r in records if r['eligible'])
    # This artifact is synthetic, including its truth; no competition poses
    # or external predictions appear in candidate selection or evaluation.
    (tmp_path/'synthetic_default.json').write_text(json.dumps(dict(
        fixture='generated_default_startup_not_real_localization',
        truth_T=truth.tolist(),truth_scale=7.,selected_T=result['T'].tolist(),
        selected_scale=result['scale'],initialization=result['initialization']),indent=2))


@pytest.mark.parametrize('all_rejected',[False,True])
def test_low_score_cannot_override_registration_gate(monkeypatch,all_rejected):
    m,source,truth,prior,cfg=scene()
    align=GroundInitializer.align
    def proposals(self,*args):
        result=align(self,*args)
        result['candidates']=[dict(T=truth.tolist(),scale=7.,cost=0.),
                              dict(T=truth.tolist(),scale=7.,cost=1.)]
        return result
    monkeypatch.setattr(GroundInitializer,'align',proposals)
    reg=Registration(m,cfg)
    calls=[]
    def refine(*args,**kwargs):
        T=truth.copy();index=len(calls);calls.append(index)
        if index:T[0,3]+=.1
        accepted=index==1 and not all_rejected
        return dict(T=T,scale=7.,accepted=accepted,reason='accepted' if accepted else 'degenerate_geometry')
    monkeypatch.setattr(reg,'_align_seed',refine)
    result=reg.align(source,prior,3.,initialize=True)
    records=result['initialization']['refinements']
    assert records[0]['feature_score']['cost']<records[1]['feature_score']['cost']
    assert not records[0]['eligible']
    assert result['accepted']==(not all_rejected)
    if all_rejected:
        assert result['reason']=='startup_no_valid_candidate'
        from track3.localization import PoseState
        state=PoseState(prior,3.)
        prediction=state.predict(1727239790258245070)
        decision=state.commit(result)
        assert not decision['accepted'] and not state.initialized and state.scale==3.
        np.testing.assert_array_equal(state.T,prediction)
    else:
        assert result['initialization']['selection']['selected_candidate_index']==1
