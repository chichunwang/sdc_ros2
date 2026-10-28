"""Known synthetic bias, real-turn response and causal CUDA numerical checks."""
from copy import deepcopy
import json
import numpy as np
import cupy as cp
import pytest
from scipy.spatial.transform import Rotation
from track3.gyro_bias import StationaryGyroBias
from track3.imu_parameters import resolve_motion_config
from test_monte_carlo import STAMP,config,state,sample,estimate


def active_config():return resolve_motion_config(config())


def test_cuda_kalman_matches_independent_scalar_joseph_reference(monkeypatch):
    cfg=active_config();f=StationaryGyroBias(cfg)
    f.step(STAMP,[sample(STAMP)],False)
    z=np.array([.003,-.004,.006]);h=.01
    P=np.array(cfg['initial_std'][12:15])**2+cfg['gyro_bias_random_walk']**2*h
    R=cfg['gyro_noise_density']**2/h;K=P/(P+R)
    # Runtime arithmetic must not silently call a host solver or host reduction.
    def forbidden(*a,**kw):raise AssertionError('CPU numerical solve/reduction')
    monkeypatch.setattr(np.linalg,'solve',forbidden);monkeypatch.setattr(np,'sum',forbidden)
    diag=f.step(STAMP+10_000_000,[sample(STAMP+10_000_000,z)],True)
    np.testing.assert_allclose(cp.asnumpy(f.bias),K*z,rtol=1e-14)
    np.testing.assert_allclose(cp.asnumpy(f.variance),(1-K)**2*P+K**2*R,rtol=1e-14)
    assert diag['updated'] and diag['device']=='cuda' and not diag['map_measurement_used']


def test_no_new_samples_cannot_reuse_held_observation_and_motion_does_not_train():
    cfg=active_config();f=StationaryGyroBias(cfg)
    f.step(STAMP,[sample(STAMP)],False)
    a=f.step(STAMP+10_000_000,[sample(STAMP+10_000_000,(0,0,.006))],True)
    old=f.bias.copy();P=f.variance.copy()
    b=f.step(STAMP+20_000_000,[],True)
    assert a['updated'] and not b['updated'] and b['reason']=='no_new_imu'
    np.testing.assert_array_equal(cp.asnumpy(f.bias),cp.asnumpy(old))
    assert bool(cp.all(f.variance>P))
    c=f.step(STAMP+30_000_000,[sample(STAMP+30_000_000,(0,0,.3))],False)
    assert c['reason']=='not_stationary' and not c['updated']
    np.testing.assert_array_equal(cp.asnumpy(f.bias),cp.asnumpy(old))


@pytest.mark.parametrize('bad',['future','duplicate','past','nan','float_time'])
def test_bad_batch_cannot_mutate_bias_covariance_or_clock(bad):
    f=StationaryGyroBias(active_config());f.step(STAMP,[sample(STAMP)],False)
    t=STAMP+10_000_000;samples=[sample(t)]
    if bad=='future':samples=[sample(t+1)]
    if bad=='duplicate':samples=[sample(t),sample(t)]
    if bad=='past':samples=[sample(STAMP)]
    if bad=='nan':samples[0].angular_velocity[0]=np.nan
    if bad=='float_time':t=float(t)
    before=(f.bias.copy(),f.variance.copy(),f.stamp,f.last_sample)
    with pytest.raises(ValueError):f.step(t,samples,True)
    np.testing.assert_array_equal(cp.asnumpy(f.bias),cp.asnumpy(before[0]))
    np.testing.assert_array_equal(cp.asnumpy(f.variance),cp.asnumpy(before[1]))
    assert (f.stamp,f.last_sample)==before[2:]


def test_raw_motion_and_nis_gates_protect_bias():
    cfg=active_config();cfg['gyro_noise_density']=1e-6;cfg['initial_std'][12:15]=[1e-5]*3
    f=StationaryGyroBias(cfg);f.step(STAMP,[sample(STAMP)],False)
    assert f.step(STAMP+10_000_000,[sample(STAMP+10_000_000,(0,0,.3))],True)['reason']=='raw_gyro_motion_gate'
    assert f.step(STAMP+20_000_000,[sample(STAMP+20_000_000,(0,0,.01))],True)['reason']=='innovation_gate'
    np.testing.assert_array_equal(cp.asnumpy(f.bias),np.zeros(3))


def test_constant_observation_has_same_information_at_different_sample_rates():
    results=[]
    for step_ns in (10_000_000,5_000_000):
        f=StationaryGyroBias(active_config());f.step(STAMP,[sample(STAMP)],False)
        for i in range(1,11):
            t=STAMP+i*100_000_000
            batch=[sample(s,(.003,-.004,.006)) for s in range(t-100_000_000+step_ns,t+1,step_ns)]
            f.step(t,batch,True)
        results.append((cp.asnumpy(f.bias),cp.asnumpy(f.variance)))
    np.testing.assert_allclose(results[0],results[1],rtol=2e-14,atol=1e-16)


def test_no_map_or_acceleration_bias_feedback_and_first_anchor_is_identical():
    cfg=active_config();s=state(16,cfg)
    cfg_off=deepcopy(cfg);cfg_off['gyro_bias_filter']['enabled']=False;off=state(16,cfg_off)
    initial=np.eye(4);initial[:3,3]=[15.,-3.,2.]
    for obj in (s,off):
        obj.predict(STAMP,[sample(STAMP,(0,0,.006))]);obj.commit(estimate(initial,scale=6.4))
    np.testing.assert_array_equal(s.T,off.T)
    assert s.scale==off.scale and s.bias_filter.updates==0
    for i in range(1,51):
        t=STAMP+i*100_000_000
        samples=[sample(t-100_000_000+j*10_000_000,(0,0,.006)) for j in range(1,11)]
        s.predict(t,samples,stationary=True);s.commit(estimate())
    before=s.nominal_bias.copy();P=s.bias_filter.variance.copy();old=s.T.copy()
    t=STAMP+5_100_000_000
    samples=[sample(t-100_000_000+j*10_000_000,(0,0,.306)) for j in range(1,11)]
    for x in samples:x.linear_acceleration=np.array([1e6,-1e6,1e6])
    prior=s.predict(t,samples)
    expected=old[:3,:3]@Rotation.from_rotvec([0,0,.006*.01+.306*.09-float(before[2])*.1]).as_matrix()
    np.testing.assert_allclose(prior[:3,:3],expected,atol=2e-14)
    impossible=prior.copy();impossible[0,3]+=30.
    assert not s.commit(estimate(impossible))['accepted']
    np.testing.assert_array_equal(cp.asnumpy(s.nominal_bias),cp.asnumpy(before))
    assert bool(cp.all(s.bias_filter.variance>P))


def test_synthetic_straight_stop_resume_turn_and_prefix_invariance(tmp_path):
    def run(frames):
        cfg=active_config();filtered=state(16,cfg)
        cfg_off=deepcopy(cfg);cfg_off['gyro_bias_filter']['enabled']=False;unfiltered=state(16,cfg_off)
        rng=np.random.default_rng(411);bias=np.array([.003,-.004,.006]);truth=np.eye(3);rows=[]
        for i in range(frames):
            t=STAMP+i*100_000_000;rest=10<=i<60;turn=i>=60
            omega=np.array([0.,0.,.3 if turn else 0.])
            if i:
                # Existing IMU ZOH uses the last stop sample for the first .01 s.
                angle=.027 if i==60 else (.03 if turn else 0.)
                truth=truth@Rotation.from_rotvec([0,0,angle]).as_matrix()
            samples=[sample(t-100_000_000+j*10_000_000,omega+bias+rng.normal(0,.0005,3)) for j in range(1,11)]
            errors=[]
            for obj in (filtered,unfiltered):
                obj.predict(t,samples,stationary=rest)
                obj.commit(estimate(obj.T.copy(),accepted=i==0))
                errors.append(float(Rotation.from_matrix(obj.T[:3,:3]@truth.T).magnitude()))
            rows.append(dict(idx=i,stationary=rest,turn=turn,bias_true_radps=bias.tolist(),
                bias_estimated_radps=cp.asnumpy(filtered.nominal_bias).tolist(),
                filtered_angle_error_rad=errors[0],unfiltered_angle_error_rad=errors[1],
                filter=filtered.prediction_diagnostics['gyro_bias_filter']))
        return rows
    rows=run(90);prefix=run(65)
    assert rows[:65]==prefix  # Future turning data cannot revise the stop estimate.
    assert rows[-1]['filtered_angle_error_rad']<rows[-1]['unfiltered_angle_error_rad']*.35
    # First moving prefix precedes any stationary evidence and remains identical.
    assert all(r['filtered_angle_error_rad']==r['unfiltered_angle_error_rad'] for r in rows[:10])
    (tmp_path/'gyro_bias_synthetic_evidence.json').write_text(json.dumps(dict(
        kind='synthetic_known_bias_not_Track3_accuracy',frames=rows),indent=2)+'\n')


def test_filter_can_be_disabled_in_python_parameters(monkeypatch):
    import track3.imu_parameters as p
    monkeypatch.setattr(p,'GYRO_BIAS_KALMAN_ENABLED',False)
    assert not p.resolve_motion_config(config())['gyro_bias_filter']['enabled']
    monkeypatch.setattr(p,'GYRO_BIAS_KALMAN_ENABLED',1)
    with pytest.raises(ValueError):p.resolve_motion_config(config())
