import json
from pathlib import Path
import numpy as np
from track3.motion import turn_motion
from track3.filter import Fusion


def test_quarter_turn_position_and_velocity():
    x=np.zeros(8);x[3]=2.
    y,F,G=turn_motion(x,np.pi/2,1.)
    np.testing.assert_allclose(y[:2],[4/np.pi,4/np.pi],atol=1e-12)
    np.testing.assert_allclose(y[3:5],[0,2],atol=1e-12)
    assert abs(y[6]-np.pi/2)<1e-12
    straight,_,_=turn_motion(x,0.,1.)
    np.testing.assert_allclose(straight[:6],[2,0,0,2,0,0])
    assert np.isfinite(F).all() and np.isfinite(G).all()


def test_turn_state_jacobian_at_zero_and_finite_turn():
    x=np.array([1.,2.,3.,4.,-2.,.1,.2,.03])
    for rate in [x[7],.4]:
        y,F,G=turn_motion(x,rate,.1)
        for j in range(8):
            d=np.zeros(8);d[j]=1e-6
            plus,_,_=turn_motion(x+d,rate,.1);minus,_,_=turn_motion(x-d,rate,.1)
            np.testing.assert_allclose(F[:,j],(plus-minus)/2e-6,atol=1e-8)


def test_expired_gyro_does_not_turn_forever():
    cfg=json.loads((Path(__file__).resolve().parents[1]/'config/track3.json').read_text())['filter']
    k=Fusion(np.eye(4),0,cfg);k.x[3]=2.;k.rate=1.;k.rate_stamp=0
    k.step(1_000_000_000)
    assert abs(k.x[6]-.1)<1e-12
    np.testing.assert_allclose(k.x[3:5],[2*np.cos(.1),2*np.sin(.1)],atol=1e-12)
    assert np.linalg.eigvalsh(k.P).min()>-1e-9
