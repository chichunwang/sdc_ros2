"""Independent motion ODE checks, rest calibration and causal sensor contracts."""
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
from track2 import parameters as p
from track2.coordinated_motion import transition,rate_sensitivity,StationaryGyroBias
from track2.causal_prior import CausalSensorPrior


@pytest.fixture(autouse=True)
def profile(monkeypatch):
    monkeypatch.setattr(p,'FUSION_PROFILE','v52_directional_memory')
    monkeypatch.setattr(p,'V510_ENABLED',True)
    monkeypatch.setattr(p,'V5162_ENABLED',True)
    for name in ['V511_ENABLED','V512_ENABLED','V5121_ENABLED','V513_ENABLED','V514_ENABLED','V515_ENABLED']:
        monkeypatch.setattr(p,name,False)


def test_transition_and_noise_against_independent_continuous_ode():
    nodes,weights=np.polynomial.legendre.leggauss(64)
    for dt in [.01,.25,1.]:
        for rate in [-2.,-.1,-1e-8,0.,1e-8,.1,2.]:
            F,Q=transition(dt,rate,4.)
            A=np.zeros((4,4));A[:2,2:]=np.eye(2);A[2:,2:]=[[0.,-rate],[rate,0.]]
            G=np.r_[np.zeros((2,2)),np.eye(2)];integral=np.zeros((4,4))
            for u,w in zip((nodes+1)*dt/2,weights*dt/2):
                E=expm(A*u);integral+=w*4.*(E@G@G.T@E.T)
            np.testing.assert_allclose(F,expm(A*dt),atol=1e-12,rtol=0)
            np.testing.assert_allclose(Q,integral,atol=1e-10,rtol=0)
            assert np.linalg.eigvalsh(Q).min()>0
            f,q=transition(dt/2,rate,4.)
            np.testing.assert_allclose(F,f@f,atol=1e-12,rtol=0)
            np.testing.assert_allclose(Q,f@q@f.T+q,atol=1e-10,rtol=0)
            x=np.array([4.,-3.,12.,2.]);eps=1e-5
            finite=(transition(dt,rate+eps,4.)[0]@x-transition(dt,rate-eps,4.)[0]@x)/(2*eps)
            np.testing.assert_allclose(rate_sensitivity(x[2:],dt,rate),finite,atol=1e-7,rtol=0)


@pytest.mark.parametrize('rate,expected',[(1.,[2.,2.,0.,2.]),(-1.,[2.,-2.,0.,-2.])])
def test_signed_quarter_circle(rate,expected):
    np.testing.assert_allclose(transition(np.pi/2,rate,0.)[0]@np.array([0.,0.,2.,0.]),expected,atol=1e-12)


def test_motion_does_not_force_absolute_heading_or_lateral_velocity():
    outputs=[]
    for heading in [0.,1.7]:
        s=CausalSensorPrior(np.eye(4),1_000_000_000)
        s.orientation=Rotation.from_euler('z',heading);s.gyro=np.array([0.,0.,.4])
        s.position=np.array([3.,-2.,5.,2.]);s.cov=np.eye(4)*2.
        s.advance(1_250_000_000);outputs.append((s.position,s.cov))
    np.testing.assert_array_equal(outputs[0][0],outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1],outputs[1][1])
    assert abs(outputs[0][0][3])>1.  # The measured world velocity is not forced onto body forward.


def test_disabled_model_preserves_exact_cv_arithmetic(monkeypatch):
    monkeypatch.setattr(p,'V5162_ENABLED',False)
    s=CausalSensorPrior(np.eye(4),1_000_000_000);s.gyro=np.array([0.,0.,.4])
    x=np.array([3.,-2.,5.,2.]);C=np.eye(4)*2.;s.position=x.copy();s.cov=C.copy();dt=.25
    F=np.eye(4);F[:2,2:]=np.eye(2)*dt
    Q=p.GNSS_XY_ACCEL_PSD*np.block([[np.eye(2)*dt**3/3,np.eye(2)*dt**2/2],[np.eye(2)*dt**2/2,np.eye(2)*dt]])
    s.advance(1_250_000_000)
    assert s.gyro_motion_bias is None
    np.testing.assert_array_equal(s.position,F@x);np.testing.assert_array_equal(s.cov,F@C@F.T+Q)


def test_rest_bias_uses_only_arrived_disjoint_samples():
    b=StationaryGyroBias(np.ones(3)*.01,.001,.0003,1.,7.815,16.266)
    for i in range(1,11):b.observe(1_000_000_000+i*10_000_000,np.array([.004,.002,-.003]),1_100_000_000)
    np.testing.assert_array_equal(b.mean,np.zeros(3))
    with pytest.raises(ValueError,match='current time'):b.update(1_000_000_000,1_200_000_000,1_100_000_000)
    d=b.update(1_000_000_000,1_100_000_000,1_100_000_000)
    assert d['accepted'] and d['samples']==10 and np.linalg.eigvalsh(b.cov).min()>0
    np.testing.assert_allclose(b.mean,[.004,.002,-.003],atol=5e-5)
    with pytest.raises(ValueError,match='overlap'):b.update(1_000_000_000,1_100_000_000,1_100_000_000)
    with pytest.raises(ValueError,match='current time'):b.observe(1_200_000_000,np.zeros(3),1_100_000_000)


def rest_state():
    s=CausalSensorPrior(np.eye(4),1_000_000_000)
    for i in range(1,26):
        s.event(dict(kind='imu',timestamp_ns=1_000_000_000+i*10_000_000,quaternion=[0.,0.,0.,1.],gyro=[0.,0.,.003]))
    return s


def test_only_filter_accepted_image_rest_calibrates_future_motion():
    from track2.visual_update import fuse_visual_speed
    measurement=dict(accepted=True,kind='stationary',timestamp_ns=1_250_000_000,
        keyframe_timestamp_ns=1_000_000_000,variance_m2ps2=.09)
    s=rest_state();T,d=s.for_frame(s.time);saved=T.copy()
    result,_,image=fuse_visual_speed(s,T,d,measurement,float(np.linalg.norm(s.gyro)))
    assert image['filter_accepted'] and image['gyro_motion_bias_update']['accepted']
    assert s.gyro_motion_bias.updates==1 and s.gyro_motion_bias.mean[2]>0
    np.testing.assert_array_equal(T,saved);np.testing.assert_array_equal(result,saved)
    rejected=rest_state();rejected.position[2:]=[20.,0.];rejected.cov=np.eye(4)*.01
    T,d=rejected.for_frame(rejected.time)
    _,_,image=fuse_visual_speed(rejected,T,d,measurement,float(np.linalg.norm(rejected.gyro)))
    assert not image['filter_accepted'] and rejected.gyro_motion_bias.updates==0


def test_current_sensor_prefix_independent_of_future_deliveries():
    def run(n):
        s=CausalSensorPrior(np.eye(4),1_000_000_000);out=[]
        for i in range(n):
            stamp=1_000_000_000+i*40_000_000;angle=.04*i*.04
            s.event(dict(kind='imu',timestamp_ns=stamp,quaternion=Rotation.from_euler('z',angle).as_quat().tolist(),gyro=[0.,0.,.04]))
            if i%5==0:s.event(dict(kind='gnss',timestamp_ns=stamp,xy=[i*.04,.1*np.sin(angle)],z=1.,heading_quaternion=None))
            T,_=s.for_frame(stamp);out.append(np.r_[T[:2,3],s.position[2:],s.cov.ravel()])
        return np.array(out)
    np.testing.assert_array_equal(run(20),run(35)[:20])
