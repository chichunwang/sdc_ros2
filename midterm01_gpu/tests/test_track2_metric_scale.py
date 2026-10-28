"""Generated metric geometry, changing raw units and causal state contracts."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from track2 import parameters as p


@pytest.fixture
def cp():
    import cupy
    assert cupy.cuda.runtime.getDeviceCount() > 0
    return cupy


def scene():
    x,y = np.meshgrid(np.arange(2.,22.,.3),np.arange(-8.,8.,.3))
    ground=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
    y,z=np.meshgrid(np.arange(-7.,7.,.3),np.arange(.3,4.,.3))
    front=np.column_stack([np.full(y.size,12.),y.ravel(),z.ravel()])
    x,z=np.meshgrid(np.arange(2.,22.,.3),np.arange(.3,4.,.3))
    side=np.column_stack([x.ravel(),np.full(x.size,6.),z.ravel()])
    # Second depth prevents false scale/translation ambiguity.
    y,z=np.meshgrid(np.arange(-3.,3.,.3),np.arange(.3,3.,.3))
    pillar=np.column_stack([np.full(y.size,5.),y.ravel(),z.ravel()])
    return np.concatenate([ground,front,side,pillar])


def truth():
    T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',[.02,-.03,.2]).as_matrix()
    T[:3,3]=[0.,0.,1.6]
    return T


def raw_cloud(xyz,T,scale):
    return ((xyz-T[:3,3]) @ T[:3,:3])/scale


def test_grounded_recovers_scale_and_tilt_instead_of_fixed_three(cp):
    from track2.metric_registration import MetricRegistration
    xyz=scene();T=truth();prior=T.copy()
    prior[:3,:3]=Rotation.from_euler('xyz',[.06,-.08,.215]).as_matrix()
    prior[:2,3]+=[.2,-.15]
    reg=MetricRegistration(cp.asarray(xyz),T)
    out,scale,diag=reg.refine(cp.asarray(raw_cloud(xyz,T,8.)),prior,1_000_000_000)
    assert diag['accepted'],diag
    assert scale == pytest.approx(8.,abs=.12)
    assert diag['scale_validated']
    np.testing.assert_allclose(out[:3,3],T[:3,3],atol=.05)
    assert Rotation.from_matrix(out[:3,:3] @ T[:3,:3].T).magnitude() < .01
    assert diag['ground_normal_difference_before_deg'] > 2.
    assert diag['feature_cost'] < diag['seed_feature_cost']


def test_scale_changes_with_raw_units_and_pose_stays_metric(cp):
    from track2.metric_registration import MetricRegistration
    xyz=scene();T=truth();reg=MetricRegistration(cp.asarray(xyz),T)
    results=[]
    for k,s in enumerate([7.,8.,9.]):
        out,estimated,diag=reg.refine(cp.asarray(raw_cloud(xyz,T,s)),T,1_000_000_000+k*50_000_000)
        assert diag['accepted'],diag
        assert estimated == pytest.approx(s,abs=.10)
        np.testing.assert_allclose(out[:3,3],T[:3,3],atol=.03)
        results.append(estimated)
    assert results[-1]-results[0] > 1.8
    with pytest.raises(ValueError,match='strictly increase'):
        reg.refine(cp.asarray(raw_cloud(xyz,T,9.)),T,1_000_000_000)


def test_metric_objective_gradient_includes_height_prior(cp):
    from track2.metric_registration import MetricRegistration
    xyz=scene();T=truth();reg=MetricRegistration(cp.asarray(xyz),T)
    reg.prepare(cp.asarray(raw_cloud(xyz,T,8.)),T)
    x=reg.seed.copy();x += [.012,-.016,.001,.002]
    e=reg.evaluate(x,.75,.3,derivatives=True)
    numeric=[]
    for i in range(4):
        delta=np.zeros(4);delta[i]=1e-6
        a=reg.evaluate(x+delta,.75,.3)['cost'];b=reg.evaluate(x-delta,.75,.3)['cost']
        numeric.append((a-b)/2e-6)
    np.testing.assert_allclose(2*e['g'],numeric,rtol=2e-5,atol=2e-5)
    assert np.linalg.eigvalsh(e['H']).min() > 0


def test_scale_prefix_exact_and_failed_support_does_not_calibrate(cp):
    from track2.metric_registration import MetricRegistration
    xyz=scene();T=truth()
    def run(n):
        reg=MetricRegistration(cp.asarray(xyz),T);out=[]
        for k in range(n):
            a,s,d=reg.refine(cp.asarray(raw_cloud(xyz,T,8.+k*.1)),T,1_000_000_000+k*50_000_000)
            out.append((a,s,reg.height,reg.height_variance))
        return out
    short,long=run(2),run(3)
    for a,b in zip(short,long):
        np.testing.assert_array_equal(a[0],b[0]); assert a[1:]==b[1:]
    reg=MetricRegistration(cp.asarray(xyz),T)
    out,s,d=reg.refine(cp.empty((0,3)),T,1_000_000_000)
    assert not d['accepted'] and not d['scale_validated']
    assert reg.height is None
    np.testing.assert_array_equal(out,T)


def test_world_correction_mapping_jacobian_and_ground_normal(cp):
    from track2.metric_registration import MetricRegistration
    from track2.correction_filter import pose_correction
    # Sloped map plane and nontrivial source tilt exercise all coordinate axes.
    R=Rotation.from_euler('xyz',[.035,-.04,.15]).as_matrix()
    xyz=scene() @ R.T; T=truth();T[:3,:3]=R @ T[:3,:3];T[:3,3]=R @ T[:3,3]
    reg=MetricRegistration(cp.asarray(xyz),T)
    reg.prepare(cp.asarray(raw_cloud(xyz,T,8.)),T)
    c=np.array([.3,-.2,.02]);x=reg.state_for_correction(T,c,8.)
    out,_=reg.pose(x)
    np.testing.assert_allclose(pose_correction(out,T),c,atol=1e-11)
    np.testing.assert_allclose(out[:3,:3] @ reg.ns,reg.nm,atol=1e-12)
    assert reg.nm @ out[:3,3]+reg.dm == pytest.approx(8*reg.ds,abs=1e-12)
    numeric=np.zeros((3,4))
    for i in range(4):
        d=np.zeros(4);d[i]=1e-6
        a,_=reg.pose(x+d);b,_=reg.pose(x-d)
        numeric[:,i]=(pose_correction(a,T)-pose_correction(b,T))/2e-6
    np.testing.assert_allclose(reg.correction_jacobian(x),numeric,atol=1e-8)


def test_failed_ground_or_features_keep_world_correction_before_emission(cp,monkeypatch):
    from track2.metric_registration import MetricRegistration
    from track2.correction_filter import pose_correction
    xyz=scene();T=truth();prior=T.copy();prior[:2,3]+=[.2,-.15]
    raw=cp.asarray(raw_cloud(xyz,T,8.));reg=MetricRegistration(cp.asarray(xyz),T)
    first,s,d=reg.refine(raw,prior,1_000_000_000,speed_mps=8.)
    assert d['accepted'];c=pose_correction(first,prior);height=reg.height
    assert np.linalg.norm(c[:2]) > .15
    moved=prior.copy();moved[0,3]+=.4
    out,_,d=reg.refine(cp.empty((0,3)),moved,1_050_000_000,speed_mps=8.)
    assert not d['accepted'] and not d['scale_validated']
    np.testing.assert_allclose(pose_correction(out,moved),c,atol=1e-12)
    assert reg.height == height
    monkeypatch.setattr(p,'MIN_METRIC_MATCHES',10**9)
    out,_,d=reg.refine(raw,prior,1_100_000_000,speed_mps=8.)
    assert not d['geometry_accepted'] and not d['accepted']
    np.testing.assert_allclose(pose_correction(out,prior),c,atol=1e-11)
    assert reg.height == height and d['height_update_gain'] == 0.
