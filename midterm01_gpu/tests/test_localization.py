import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from track3.localization import PoseState


def measurement(x=1., angle=.2, scale=3.2, accepted=True):
    T=np.eye(4);T[:3,3]=[x,2.,3.]
    T[:3,:3]=Rotation.from_rotvec([.03,-.05,angle]).as_matrix()
    return dict(T=T,scale=scale,accepted=accepted,reason='accepted' if accepted else 'low_coverage')


def test_direct_commit_and_complete_rejection():
    state=PoseState(np.eye(4),3.)
    stamp=1727239790258245070
    np.testing.assert_array_equal(state.predict(stamp),np.eye(4))
    obs=measurement();decision=state.commit(obs)
    assert decision['prediction']=='initial_guess'
    np.testing.assert_array_equal(state.T,obs['T'])
    assert state.scale==3.2 and state.initialized
    prior=state.predict(stamp+1)
    rejected=measurement(x=100.,angle=1.2,scale=8.,accepted=False)
    decision=state.commit(rejected)
    assert decision['decision']=='retained_prior'
    np.testing.assert_array_equal(state.T,prior)
    assert state.scale==3.2
    np.testing.assert_array_equal(state.predict(stamp+2),prior)
    assert not state.accepted_history


def test_time_scaled_motion_uses_only_consecutive_accepted_poses():
    state=PoseState(np.eye(4),3.,'constant_velocity')
    stamp=1727239790258245070
    for i in range(2):
        state.predict(stamp+i*100_000_000)
        obs=measurement(x=i,scale=3.)
        obs['T'][:3,:3]=Rotation.from_rotvec([0,0,i*.1]).as_matrix()
        state.commit(obs)
    predicted=state.predict(stamp+300_000_000)
    assert predicted[0,3]==pytest.approx(3.)
    np.testing.assert_allclose(predicted[:3,:3],Rotation.from_rotvec([0,0,.3]).as_matrix(),atol=1e-14)
    state.commit(measurement(x=200,accepted=False))
    # Rejected candidate cannot train or sustain the velocity extrapolation.
    np.testing.assert_array_equal(state.predict(stamp+400_000_000),predicted)
    state.commit(measurement(x=4))
    np.testing.assert_array_equal(state.predict(stamp+500_000_000),state.T)


@pytest.mark.parametrize('bad', ['rotation','translation','scale'])
def test_invalid_accepted_candidate_retains_prior(bad):
    state=PoseState(np.eye(4),3.)
    prior=state.predict(123)
    obs=measurement()
    if bad=='rotation':obs['T'][0,0]=2.
    if bad=='translation':obs['T'][0,3]=np.nan
    if bad=='scale':obs['scale']=float('inf')
    decision=state.commit(obs)
    assert not decision['accepted'] and decision['reason']=='invalid_candidate_pose_or_scale'
    np.testing.assert_array_equal(state.T,prior)
    assert state.scale==3. and not state.initialized


def test_prefix_causality_and_integer_clock():
    def sequence(n):
        state=PoseState(np.eye(4),3.,'constant_velocity');out=[]
        for i in range(n):
            state.predict(1727239790258245070+i*100_000_000)
            state.commit(measurement(x=i,accepted=i!=2));out.append(state.T.copy())
        return out,state
    short,_=sequence(3);long,state=sequence(5)
    np.testing.assert_array_equal(short,long[:3])
    with pytest.raises(ValueError):state.predict(float(state.stamp+1))
    with pytest.raises(ValueError):state.predict(state.stamp)
    state.predict(state.stamp+1)
    with pytest.raises(RuntimeError):state.predict(state.stamp+2)
