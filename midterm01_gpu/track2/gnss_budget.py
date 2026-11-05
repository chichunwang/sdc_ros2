"""One delivered GNSS likelihood, consumed causally in bounded powers.

This updates the CV estimator before registration. It never adjusts a saved
pose. A deferred position factor uses H=[I,-age I], not a stale position as
the current target. Process noise enlarges its uncertainty; state/process
cross-correlation is not fully estimated by this conservative approximation.
"""
import numpy as np


def fractional_position_update(mean, covariance, xy, *, variance, age_seconds,
                               remaining, acceleration_psd, body_rotation,
                               radial_limit=None, lateral_limit=None, direction=None):
    x=np.asarray(mean,dtype=float);P=np.asarray(covariance,dtype=float)
    z=np.asarray(xy,dtype=float);B=np.asarray(body_rotation,dtype=float)
    if (x.shape!=(4,) or P.shape!=(4,4) or z.shape!=(2,) or B.shape!=(2,2)
            or not np.isfinite(np.r_[x,P.ravel(),z,B.ravel(),variance,age_seconds,
                                       remaining,acceleration_psd]).all()
            or variance<=0 or age_seconds<0 or not 0<remaining<=1
            or acceleration_psd<0 or np.linalg.eigvalsh(P).min()<=0
            or not np.allclose(B.T@B,np.eye(2),atol=1e-12)):
        raise ValueError('Invalid causal GNSS factor')
    bounded=radial_limit is not None or lateral_limit is not None
    limits=[v for v in [radial_limit,lateral_limit] if v is not None]
    if any(v<0 or not np.isfinite(v) for v in limits):
        raise ValueError('Invalid GNSS influence budget')
    H=np.c_[np.eye(2),-age_seconds*np.eye(2)]
    if direction is not None:
        direction=np.asarray(direction,dtype=float)
        if direction.shape!=(2,) or not np.isfinite(direction).all() or not np.isclose(direction@direction,1.):
            raise ValueError('Invalid GNSS factor direction')
        H=direction.reshape(1,2)@H;z=np.array([float(direction@z)])
    innovation=z-H@x
    # H F(-age) Q(age) F(-age).T H.T = q age^3/3 I for CV.
    base_R=np.eye(len(H))*(variance+acceleration_psd*age_seconds**3/3.)
    PH=P@H.T;S=H@PH
    def trial(fraction):
        if fraction==0:return np.zeros((4,len(H))),np.zeros(4),None
        R=base_R/fraction
        K=np.linalg.solve(S+R,PH.T).T
        return K,K@innovation,R
    def inside(increment):
        return ((radial_limit is None or np.linalg.norm(increment[:2])<=radial_limit)
            and (lateral_limit is None or abs(float((B.T@increment[:2])[1]))<=lateral_limit))
    fraction=float(remaining);K,increment,R=trial(fraction)
    if not inside(increment):
        low=0.;high=fraction
        for _ in range(48):
            middle=(low+high)*.5
            if inside(trial(middle)[1]):low=middle
            else:high=middle
        fraction=low;K,increment,R=trial(fraction)
    assert inside(increment)
    if fraction>0:
        A=np.eye(4)-K@H
        after=A@P@A.T+K@R@K.T
        after=(after+after.T)*.5
    else:after=P.copy()
    return x+increment,after,dict(information_fraction=fraction,
        information_fraction_remaining=max(0.,float(remaining)-fraction),
        observation_matrix=H.tolist(),innovation=innovation.tolist(),
        state_increment=increment.tolist(),kalman_gain=K.tolist(),
        covariance_before=P.tolist(),covariance_after=after.tolist(),
        effective_measurement_covariance=None if R is None else R.tolist(),
        base_measurement_covariance=base_R.tolist(),age_seconds=float(age_seconds),
        radial_limit_m=radial_limit,lateral_limit_m=lateral_limit,
        bootstrap_unbounded=not bounded,
        model='one arrived position likelihood; total consumed power <= 1; deferred CV row and process inflation')
