"""Generated yaw geometry, robustness and causal ESKF integration checks."""
from copy import deepcopy
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from track3.yaw_constraints import YawConstraint, yaw_jacobian
from test_inertial import CFG, STAMP, initialized, observation, sample


def rot(yaw, pitch=0., roll=0.):
    return Rotation.from_euler('xyz',[roll,pitch,yaw],degrees=True).as_matrix()


def constraint(enabled=True):
    return YawConstraint(dict(enabled=enabled,huber_sigma=3.,min_horizontal_norm=.1))


def covariances(std=1.):
    P=np.eye(3)*np.deg2rad(std)**2
    C=np.diag([.25,.25,.25,*([np.deg2rad(std)**2]*3)])
    return P,C


@pytest.mark.parametrize('angles',[(20,30,170),(-15,-45,-120),(0,0,0)])
def test_yaw_projection_against_left_rotation_finite_difference(angles):
    R=Rotation.from_euler('xyz',angles,degrees=True).as_matrix()
    numerical=[]
    for axis in np.eye(3):
        plus=Rotation.from_rotvec(axis*1e-6).as_matrix()@R
        minus=Rotation.from_rotvec(-axis*1e-6).as_matrix()@R
        difference=np.arctan2(plus[1,0],plus[0,0])-np.arctan2(minus[1,0],minus[0,0])
        numerical.append(np.arctan2(np.sin(difference),np.cos(difference))/2e-6)
    np.testing.assert_allclose(yaw_jacobian(R),numerical,atol=2e-10)


def test_wrap_boundary_and_real_turn_matching_prediction_are_not_suppressed():
    P,C=covariances(5.)
    result,diag=constraint().covariance(rot(179),rot(-179),P,C)
    assert diag['innovation_deg']==pytest.approx(2.) and diag['weight']==1.
    np.testing.assert_array_equal(result,C)
    # A large absolute turn is fine when gyro prediction follows the same turn.
    result,diag=constraint().covariance(rot(92),rot(92),P,C)
    np.testing.assert_array_equal(result,C)
    assert diag['innovation_deg']==0 and not diag['covariance_inflated']


def test_only_yaw_projection_variance_is_inflated_and_inputs_are_preserved():
    P,C=covariances();before=C.copy();R=rot(30,pitch=20,roll=-10)
    result,diag=constraint().covariance(rot(0,pitch=20,roll=-10),R,P,C)
    assert 0<diag['weight']<1 and diag['covariance_inflated']
    extra=result-C
    assert np.linalg.eigvalsh(extra).min()>-1e-15
    assert np.linalg.eigvalsh(result).min()>0
    np.testing.assert_array_equal(result[:3,:3],C[:3,:3])
    np.testing.assert_array_equal(result[:3,3:],C[:3,3:])
    j=yaw_jacobian(R)
    assert j@result[3:,3:]@j==pytest.approx((j@C[3:,3:]@j)/diag['weight'])
    np.testing.assert_array_equal(C,before)
    json.dumps(diag,allow_nan=False)


def test_uncertain_prediction_does_not_automatically_overrule_map_heading():
    P,C=covariances()
    _,tight=constraint().covariance(rot(0),rot(20),P,C)
    result,uncertain=constraint().covariance(rot(0),rot(20),P*10000,C)
    assert tight['weight']<1 and uncertain['weight']==1
    np.testing.assert_array_equal(result,C)


def test_disabled_and_near_vertical_heading_keep_base_covariance():
    P,C=covariances()
    result,diag=constraint(False).covariance(rot(0),rot(50),P,C)
    np.testing.assert_array_equal(result,C)
    assert diag['reason']=='disabled'
    result,diag=constraint().covariance(rot(0,pitch=89.9),rot(50,pitch=89.9),P,C)
    np.testing.assert_array_equal(result,C)
    assert diag['reason']=='near_vertical_forward_axis'
    json.dumps(diag,allow_nan=False)


@pytest.mark.parametrize('config',[
    {'enabled':1}, {'enabled':True,'huber_sigma':0},
    {'enabled':True,'huber_sigma':float('nan')},
    {'enabled':True,'min_horizontal_norm':0},
])
def test_invalid_settings_fail_before_filter_state_changes(config):
    with pytest.raises(ValueError):YawConstraint(config)


def filter_config(enabled=True,map_yaw_std=1.):
    cfg=deepcopy(CFG)
    cfg['registration_std']=[.5,.5,.5,*np.deg2rad([5.,5.,map_yaw_std])]
    cfg['yaw_constraint']=dict(enabled=enabled,huber_sigma=3.,min_horizontal_norm=.1)
    return cfg


def test_same_accepted_outlier_has_less_yaw_influence_in_single_eskf_update():
    outcomes=[]
    for enabled in (False,True):
        s=initialized(filter_config(enabled))
        s.predict(STAMP+10_000_000,[sample(STAMP+10_000_000)])
        z=np.eye(4);z[:3,:3]=rot(5)
        d=s.commit(observation(z))
        assert d['accepted']
        outcomes.append((abs(np.arctan2(s.T[1,0],s.T[0,0])),d))
        assert np.linalg.eigvalsh(s.P).min()>0
    assert 0<outcomes[1][0]<outcomes[0][0]
    assert outcomes[1][1]['inertial']['update']['yaw_constraint']['weight']<1


def test_bootstrap_not_penalized_for_noisy_initial_yaw():
    from track3.inertial import InertialState
    s=InertialState(np.eye(4),3.,filter_config())
    s.predict(STAMP,[sample(STAMP)])
    z=np.eye(4);z[:3,:3]=rot(120)
    d=s.commit(observation(z))
    np.testing.assert_allclose(s.T,z,atol=1e-14)
    assert d['accepted'] and d['inertial']['update']['yaw_constraint']['reason']=='initial_bootstrap_not_downweighted'


def test_prefix_results_and_diagnostics_do_not_depend_on_future_measurements():
    def run(n):
        s=initialized(filter_config());poses=[];diagnostics=[]
        for i in range(1,n+1):
            stamp=STAMP+i*10_000_000
            s.predict(stamp,[sample(stamp,w=(0,0,.1))])
            z=np.eye(4);z[:3,:3]=rot(5 if i==2 else i*.1)
            d=s.commit(observation(z))
            poses.append(s.T.copy());diagnostics.append(d)
        return poses,diagnostics
    short,short_diag=run(3);long,long_diag=run(6)
    np.testing.assert_array_equal(short,long[:3])
    assert short_diag==long_diag[:3]


def test_rejected_map_candidate_cannot_influence_robust_filter():
    s=initialized(filter_config());prior=s.predict(STAMP+10_000_000,[sample(STAMP+10_000_000)])
    before=s.P.copy()
    z=np.eye(4);z[:3,:3]=rot(70)
    d=s.commit(observation(z,accepted=False))
    np.testing.assert_array_equal(s.T,prior)
    np.testing.assert_array_equal(s.P,before)
    assert d['inertial']['update']['yaw_constraint']['reason']=='registration_not_accepted'


def test_lower_map_yaw_std_increases_map_correction_for_same_prior_uncertainty():
    outputs=[]
    for std in (1.,5.):
        s=initialized(filter_config(False,map_yaw_std=std))
        # Hold prior uncertainty equal; otherwise bootstrap also changes it.
        s.P[6:9,6:9]=np.eye(3)*np.deg2rad(2.)**2
        s.predict(STAMP+10_000_000,[sample(STAMP+10_000_000)])
        z=np.eye(4);z[:3,:3]=rot(2)
        assert s.commit(observation(z))['accepted']
        outputs.append(float(np.arctan2(s.T[1,0],s.T[0,0])))
    assert outputs[0]>outputs[1]>0
