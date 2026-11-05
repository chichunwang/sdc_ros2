"""Bounded bilateral surface matching: guards, derivatives and causal prefix."""
from types import SimpleNamespace
import numpy as np
import pytest
from track2 import parameters as p


@pytest.fixture(autouse=True)
def candidate_profile(monkeypatch):
    monkeypatch.setattr(p, 'FUSION_PROFILE', 'v52_directional_memory')
    monkeypatch.setattr(p, 'V510_ENABLED', True)
    monkeypatch.setattr(p, 'V5161_ENABLED', True)
    for name in ['V511_ENABLED', 'V512_ENABLED', 'V5121_ENABLED', 'V513_ENABLED', 'V514_ENABLED', 'V515_ENABLED']:
        monkeypatch.setattr(p, name, False)


def fallback_case(*, alternative=.25, normal=(0., 1., 0.), source_good=True,
                  nearest_reliable=False, radius=.75):
    import cupy as cp
    from track3.nearest import GpuNearest
    from track2.surface_correspondence import bounded_surface_fallback
    nn = GpuNearest(cp.array([[.1, 0., 0.], [alternative, 0., 0.]]), cell=1.5)
    # GpuNearest sorts by spatial key; both points are in the same cell here.
    query = cp.zeros((1, 3)); ids, d2 = nn.query(query, radius)
    geometry = SimpleNamespace(reliable=cp.array([nearest_reliable, True]),
        normal=cp.array([[0., 1., 0.], normal]), ensure=lambda ids: None)
    result, rescued = bounded_surface_fallback(nn, geometry, query, ids, d2,
        cp.array([source_good]), cp.array([[0., 1., 0.]]), np.eye(3), radius)
    return int(result[0]), bool(rescued[0])


def test_unreliable_nearest_can_use_close_source_compatible_surface():
    assert fallback_case() == (1, True)
    assert fallback_case(normal=(0., -1., 0.)) == (1, True)


@pytest.mark.parametrize('options', [dict(nearest_reliable=True), dict(source_good=False),
    dict(normal=(1., 0., 0.)), dict(alternative=.5), dict(radius=.2)])
def test_original_matches_and_radius_normal_voxel_guards(options):
    assert fallback_case(**options) == (0, False)


def test_source_normal_guard_and_tracking_gate_preserve_startup(monkeypatch):
    import cupy as cp
    from track2.metric_registration import MetricRegistration
    from test_track2_metric_scale import scene, truth, raw_cloud
    xyz=scene();T=truth();C=np.diag([.36,.36,np.deg2rad(2.)**2])
    reg=MetricRegistration(cp.asarray(xyz),T)
    d=reg.prepare(cp.asarray(raw_cloud(xyz,T,8.)),T,C)
    assert not d['surface_fallback_tracking'] and reg.source_plane_good is None
    reg.correction_filter.last_update=1_000_000_000
    d=reg.prepare(cp.asarray(raw_cloud(xyz,T,8.)),T,C)
    assert d['surface_fallback_tracking'] and d['source_plane_features']>0
    C[2,2]=np.deg2rad(4.)**2
    d=reg.prepare(cp.asarray(raw_cloud(xyz,T,8.)),T,C)
    assert not d['surface_fallback_tracking'] and reg.source_plane_good is None
    monkeypatch.setattr(p,'V515_ENABLED',True)
    assert not p.v5161_enabled()


def test_rescued_fixed_association_gradient_and_confidence_are_consistent():
    import cupy as cp
    from track2.metric_registration import MetricRegistration
    from test_track2_metric_scale import scene, truth, raw_cloud
    T=truth();reg=MetricRegistration(cp.asarray(scene()),T)
    reg.correction_filter.last_update=1_000_000_000
    reg.prepare(cp.asarray(raw_cloud(scene(),T,8.)),T,np.diag([.36,.36,np.deg2rad(2.)**2]))
    x=reg.seed+np.array([.12,.07,.002,.0005])
    before=reg.evaluate(x,.75,.3,derivatives=True)
    a=before['association'];rows=cp.flatnonzero(a['good']&reg.source_plane_good)[:8]
    reg.geometry.reliable[a['ids'][rows]]=False
    d=reg.evaluate(x,.75,.3,derivatives=True,include_prior=False)
    assert d['surface_fallback_matches']>0
    a=d['association'];mask=a['surface_fallback_matches']
    assert bool(cp.all(a['confidence'][mask]<=reg.source_plane_confidence[mask]+1e-14))
    gradient=[]
    for k in range(4):
        e=np.eye(4)[k]*1e-6
        hi=reg.evaluate(x+e,.75,.3,include_prior=False,association=a)['cost']
        lo=reg.evaluate(x-e,.75,.3,include_prior=False,association=a)['cost']
        gradient.append((hi-lo)/2e-6)
    np.testing.assert_allclose(d['data_g'],gradient,rtol=2e-5,atol=2e-5)


def test_generated_sequence_keeps_already_emitted_prefix():
    import cupy as cp
    from track2.metric_registration import MetricRegistration
    from test_track2_metric_scale import scene, truth, raw_cloud
    xyz=scene();T0=truth();C=np.diag([.36,.36,np.deg2rad(2.)**2])
    def run(count):
        reg=MetricRegistration(cp.asarray(xyz),T0);rows=[]
        for k in range(count):
            T=T0.copy();T[0,3]+=.08*k;prior=T.copy();prior[:2,3]+=[.15,-.1]
            out,scale,d=reg.refine(cp.asarray(raw_cloud(xyz,T,8.)),prior,
                1_000_000_000+k*100_000_000,prior_covariance=C,speed_mps=.8)
            rows.append((out.tolist(),scale,d['accepted'],reg.correction_filter.covariance.tolist()))
        return rows
    assert run(3)==run(6)[:3]
