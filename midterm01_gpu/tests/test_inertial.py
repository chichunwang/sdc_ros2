"""Generated mechanics, causality, uncertainty and rejection checks (no real pose)."""
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
from track3.imu import ImuSample
from track3.inertial import InertialState, skew, left_jacobian

CFG = json.loads((Path(__file__).resolve().parents[1]/'config/track3.json').read_text())['imu']
# Synthetic measurements below use identity IMU axes, independent of user RPY.
CFG.update(extrinsic_source='config', T_ply_imu=np.eye(4).tolist(),
           extrinsic_status='synthetic_identity_fixture')
STAMP = 1727239790258245070


def sample(stamp, a=(0, 0, 9.80665), w=(0, 0, 0)):
    return ImuSample(stamp, stamp-100, np.array(w, dtype=float), np.array(a, dtype=float), 'imu_link')


def observation(T=None, accepted=True, scale=3.):
    return dict(T=np.eye(4) if T is None else T, accepted=accepted, scale=scale,
                reason='accepted' if accepted else 'degenerate_geometry')


def initialized(cfg=CFG, T=None):
    T = np.eye(4) if T is None else T
    state = InertialState(T, 3., cfg)
    state.predict(STAMP, [sample(STAMP)])
    assert state.commit(observation(T))['inertial']['update']['bootstrap']
    return state


def propagate(state, n=100, a=(0, 0, 9.80665), w=(0, 0, 0)):
    for i in range(n):
        stamp=STAMP+(i+1)*10_000_000
        state.predict(stamp, [sample(stamp, a, w)])
        state.commit(observation(accepted=False, scale=18.))


def test_stationary_specific_force_cancels_gravity_and_preserves_scale():
    s=initialized()
    propagate(s)
    np.testing.assert_allclose(s.T, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(s.v, 0, atol=1e-12)
    assert s.scale == 3
    np.testing.assert_allclose(s.P, s.P.T, atol=1e-14)
    assert np.linalg.eigvalsh(s.P).min() > 0


def test_constant_acceleration_and_rotation_mechanics():
    s=initialized()
    # ZOH: put the acceleration at t0, before integrating, not at t1.
    s.held=sample(STAMP, a=(2, 0, 9.80665))
    propagate(s, a=(2, 0, 9.80665))
    np.testing.assert_allclose(s.T[:3, 3], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(s.v, [2, 0, 0], atol=1e-12)
    turn=initialized();turn.held=sample(STAMP, w=(0, 0, .3))
    propagate(turn, w=(0, 0, .3))
    np.testing.assert_allclose(turn.T[:3, :3], Rotation.from_rotvec([0, 0, .3]).as_matrix(), atol=1e-12)


def test_rotated_imu_and_metric_lever_arm():
    cfg=json.loads(json.dumps(CFG))
    E=np.eye(4);E[:3,:3]=Rotation.from_euler('xyz', [.2,-.3,.4]).as_matrix();E[:3,3]=[1.,.3,.2]
    cfg['T_ply_imu']=E.tolist()
    s=initialized(cfg)
    accel=E[:3,:3].T@np.array([0,0,9.80665])
    s.held=sample(STAMP, a=accel)
    propagate(s, a=accel)
    np.testing.assert_allclose(s.T, np.eye(4), atol=1e-12)
    # At constant angular rate about IMU origin, camera origin follows the lever arm.
    s=initialized(cfg);w=E[:3,:3].T@np.array([0,0,.2]);s.held=sample(STAMP, a=accel, w=w)
    # Remove gravity/force for this isolated rotational geometry test.
    s.gravity[:]=0;s.held.linear_acceleration[:]=0
    propagate(s, a=(0,0,0), w=w)
    R=Rotation.from_rotvec([0,0,.2]).as_matrix()
    np.testing.assert_allclose(s.T[:3,:3], R, atol=1e-12)
    np.testing.assert_allclose(s.T[:3,3], E[:3,3]-R@E[:3,3], atol=1e-12)
    z,r,H,V=s._measurement(np.eye(4))
    assert abs(V[0,5]) > 0   # lever-arm-induced position/orientation correlation


def test_first_anchor_not_vetoed_and_does_not_create_velocity_from_jump():
    s=InertialState(np.eye(4), 3., CFG)
    s.predict(STAMP,[sample(STAMP)])
    T=np.eye(4);T[:3,3]=[20,4,2];T[:3,:3]=Rotation.from_euler('z',120,degrees=True).as_matrix()
    d=s.commit(observation(T,scale=7.))
    assert d['accepted'] and d['inertial']['update']['nis'] is None
    np.testing.assert_allclose(s.T, T, atol=1e-14)
    np.testing.assert_array_equal(s.v, np.zeros(3))
    assert s.scale == 7


def test_kalman_fuses_pose_and_learns_velocity_then_rejects_bad_measurement():
    s=initialized()
    for i in range(1,31):
        stamp=STAMP+i*100_000_000
        s.predict(stamp,[sample(stamp-50_000_000),sample(stamp)])
        z=np.eye(4);z[0,3]=.1*i
        d=s.commit(observation(z))
        assert d['accepted']
    assert .8 < s.v[0] < 1.2
    prior=s.predict(STAMP+3_100_000_000,[sample(STAMP+3_050_000_000),sample(STAMP+3_100_000_000)])
    P=s.P.copy();v=s.v.copy();ba=s.ba.copy();bg=s.bg.copy()
    z=np.eye(4);z[0,3]=10000
    d=s.commit(observation(z,scale=9))
    assert not d['accepted'] and d['reason']=='kalman_nis_gate'
    np.testing.assert_array_equal(s.T,prior)
    for before,after in ((P,s.P),(v,s.v),(ba,s.ba),(bg,s.bg)):
        np.testing.assert_array_equal(before,after)
    assert s.scale==3.


def test_registration_rejection_cannot_leak_nan_pose_or_scale():
    s=initialized();prior=s.predict(STAMP+10_000_000,[sample(STAMP+10_000_000)])
    P=s.P.copy();T=np.full((4,4),np.nan)
    d=s.commit(observation(T, accepted=False, scale=float('inf')))
    np.testing.assert_array_equal(s.T,prior)
    np.testing.assert_array_equal(s.P,P)
    assert not d['accepted'] and s.scale==3
    json.dumps(d,allow_nan=False)


def test_prefix_causality_and_no_future_sample_integration():
    def run(n):
        s=initialized();out=[]
        for i in range(1,n+1):
            stamp=STAMP+i*10_000_000
            prior=s.predict(stamp,[sample(stamp,w=(0,0,i*.1))])
            s.commit(observation(accepted=False));out.append(prior)
        return out
    np.testing.assert_array_equal(run(3),run(6)[:3])
    s=initialized();X=s.X.copy()
    with pytest.raises(ValueError):s.predict(STAMP+10,[sample(STAMP+11)])
    np.testing.assert_array_equal(X,s.X)
    with pytest.raises(ValueError):s.predict(float(STAMP+10),[])
    with pytest.raises(ValueError):s.predict(STAMP,[])
    with pytest.raises(ValueError,match='stale'):s.predict(STAMP+200_000_000,[])


def test_covariance_discretization_against_van_loan_stationary():
    s=initialized();P=s.P.copy();dt=.01
    F=np.zeros((15,15));F[:3,3:6]=np.eye(3);F[3:6,6:9]=-skew([0,0,9.80665]);F[3:6,9:12]=-np.eye(3);F[6:9,12:15]=-np.eye(3)
    G=np.zeros((15,12));G[3:6,:3]=G[6:9,3:6]=G[9:12,6:9]=G[12:15,9:12]=np.eye(3)
    A=np.zeros((30,30));A[:15,:15]=F;A[:15,15:]=G@s.Qc@G.T;A[15:,15:]=-F.T
    E=expm(A*dt);Phi=E[:15,:15];Q=E[:15,15:]@Phi.T
    s._step(sample(STAMP),dt)
    np.testing.assert_allclose(s.P, Phi@P@Phi.T+Q, atol=5e-8,rtol=1e-5)
    assert s.P[0,3]>0 and s.P[6,12]<0  # velocity and gyro bias become observable


def test_reset_jacobian_matches_rotation_finite_difference():
    v=np.array([.2,-.15,.3]);base=Rotation.from_rotvec(v).as_matrix()
    numerical=np.empty((3,3))
    for i in range(3):
        eps=np.eye(3)[i]*1e-6
        plus=Rotation.from_matrix(Rotation.from_rotvec(v+eps).as_matrix()@base.T).as_rotvec()
        minus=Rotation.from_matrix(Rotation.from_rotvec(v-eps).as_matrix()@base.T).as_rotvec()
        numerical[:,i]=(plus-minus)/2e-6
    np.testing.assert_allclose(left_jacobian(v),numerical,atol=1e-9)


def test_global_error_transition_against_nonlinear_finite_difference():
    cfg=json.loads(json.dumps(CFG));dt=.0001
    T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',[.2,-.3,.7]).as_matrix()
    a=np.array([1.,-.5,9.7]);w=np.array([.1,-.2,.3]);m=sample(STAMP,a,w)
    baseline=initialized(cfg,T);baseline.v[:]=[3.,2.,1.]
    start_R=baseline.X[:3,:3].copy()
    baseline._step(m,dt)
    numerical=np.zeros((15,15));eps=1e-6
    for j in range(15):
        perturb=initialized(cfg,T);perturb.v[:]=[3.,2.,1.]
        if j<3:perturb.X[j,3]+=eps
        elif j<6:perturb.v[j-3]+=eps
        elif j<9:perturb.X[:3,:3]=Rotation.from_rotvec(np.eye(3)[j-6]*eps).as_matrix()@start_R
        elif j<12:perturb.ba[j-9]+=eps
        else:perturb.bg[j-12]+=eps
        perturb._step(m,dt)
        numerical[:,j]=np.r_[perturb.X[:3,3]-baseline.X[:3,3],perturb.v-baseline.v,
            Rotation.from_matrix(perturb.X[:3,:3]@baseline.X[:3,:3].T).as_rotvec(),
            perturb.ba-baseline.ba,perturb.bg-baseline.bg]/eps
    Rmid=start_R@Rotation.from_rotvec(w*dt/2).as_matrix()
    F=np.zeros((15,15));F[:3,3:6]=np.eye(3);F[3:6,6:9]=-skew(Rmid@a)
    F[3:6,9:12]=-Rmid;F[6:9,12:15]=-Rmid
    np.testing.assert_allclose(numerical,np.eye(15)+F*dt+.5*(F@F)*dt**2,atol=2e-8)


def test_stationary_map_observations_make_bias_observable():
    s=initialized();s.held=sample(STAMP,a=(0,0,9.90665),w=(0,0,.02))
    for i in range(1,201):
        stamp=STAMP+i*100_000_000
        s.predict(stamp,[sample(stamp-50_000_000,a=(0,0,9.90665),w=(0,0,.02)),
                         sample(stamp,a=(0,0,9.90665),w=(0,0,.02))])
        assert s.commit(observation())['accepted']
    assert s.bg[2]==pytest.approx(.02,abs=.004)
    assert s.ba[2]==pytest.approx(.1,abs=.02)
