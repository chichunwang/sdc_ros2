"""Generated CUDA point/depth and metric integration; no actual Track frames."""
from types import SimpleNamespace
import cv2
import numpy as np
import cupy as cp
import pytest
from track2 import parameters as p
from track2.image_speed import CausalGroundSpeed
from track2.geometry_diagnostics import angular_stratified_cap,surface_profile
from track2.metric_registration import MetricRegistration
from test_track2_v513 import scene,registration
from test_track2_metric_scale import scene as map_scene,truth,raw_cloud,TEST_COV


@pytest.fixture(autouse=True)
def model(monkeypatch):
    monkeypatch.setattr(p,'V514_ENABLED',False)
    monkeypatch.setattr(p,'V513_ENABLED',True);monkeypatch.setattr(p,'V510_ENABLED',True)
    monkeypatch.setattr(p,'V511_ENABLED',False);monkeypatch.setattr(p,'V512_ENABLED',False)
    monkeypatch.setattr(p,'V5121_ENABLED',False);monkeypatch.setattr(p,'IMAGE_SPEED_DOWNSAMPLE',1)


def camera():
    y,x=np.mgrid[:480,:640]
    return SimpleNamespace(K=np.array([[400.,0.,320.],[0.,400.,240.],[0.,0.,1.]]),
        maps=(x.astype('f4'),y.astype('f4')),mask=np.full((480,640),255,np.uint8))


def observer(tmp_path):
    obj=CausalGroundSpeed(tmp_path,camera());obj.folder.mkdir(parents=True)
    obj.current=dict(stamp=1_000_000_000,arrival_ns=1_000_000_000,gray=np.zeros((480,640),np.uint8),path='generated',sha256='generated')
    obj.refresh=True
    return obj


def test_map_rejected_anchor_stores_current_image_and_cuda_depth_using_height_memory(tmp_path):
    obj=observer(tmp_path);objects,_,_,_=scene();raw=cp.asarray(objects/8.)
    obj.store(raw,100.,registration(),reference_rotation=np.eye(3))
    assert obj.anchor_diagnostic['accepted'] and not obj.anchor_diagnostic['map_pose_accepted']
    assert isinstance(obj.anchor['depth']['points'],cp.ndarray)
    assert len(obj.anchor['depth']['points'])>40
    assert obj.anchor['gray'] is obj.current['gray'] and obj.anchor['stamp']==obj.current['stamp']


def test_failed_unused_interval_can_retain_matching_old_image_but_used_interval_cannot(tmp_path):
    obj=observer(tmp_path);objects,_,_,_=scene();obj.store(cp.asarray(objects/8.),8.,registration(),reference_rotation=np.eye(3))
    old=obj.anchor;obj.current=dict(obj.current,stamp=1_300_000_000,gray=np.ones((480,640),np.uint8));obj.gyro_R=np.diag([-1.,-1.,1.])
    obj.store(None,8.,registration(height_variance_after_m2=10.),reference_rotation=np.eye(3))
    assert obj.anchor is old and obj.anchor_diagnostic['anchor_retained']
    assert obj.anchor_diagnostic['anchor_available'] and obj.anchor_diagnostic['accepted']
    assert not obj.anchor_diagnostic['current_reference']['accepted']
    assert obj.anchor_diagnostic['height_variance_m2']==old['metric_reference']['height_variance_m2']
    assert not np.array_equal(obj.gyro_R,np.eye(3))
    obj.interval_produced=True
    obj.store(None,8.,registration(height_variance_after_m2=10.),reference_rotation=np.eye(3))
    assert obj.anchor is not old and obj.anchor['depth'] is None
    assert obj.anchor['gray'] is obj.current['gray'] and obj.anchor['stamp']==1_300_000_000


def test_background_depth_tracking_runs_both_real_solvers_and_selects_vector_channel(tmp_path,monkeypatch):
    import track2.image_speed as module
    from track2.relative_geometry import project,OPTICAL_TO_FLU
    obj=observer(tmp_path);objects,pixels,K,T=scene();raw=cp.asarray(objects/8.)
    before,_=project(objects,OPTICAL_TO_FLU.T,np.zeros(3),K)
    obj.store(raw,8.,registration(),reference_rotation=np.eye(3))
    path=obj.folder/'generated_ts_1250000000.jpg';cv2.imwrite(str(path),np.zeros((480,640),np.uint8))
    monkeypatch.setattr(module,'track_images',lambda *a,**kw:(before,pixels))
    d=obj.observe(1_250_000_000,1_250_000_000,.026,rotation_previous_current=T[:3,:3])
    assert d['accepted'],d
    assert d['kind']=='vector_velocity' and d['solver']=='gyro_translation'
    assert d['solver_comparison']['pnp']['accepted'] and d['solver_comparison']['gyro_translation']['accepted']
    np.testing.assert_allclose(np.asarray(d['relative_previous_current'])[:3,3],T[:3,3],atol=.03)
    obj.store(raw,8.,registration(),reference_rotation=np.eye(3))
    path=obj.folder/'generated_ts_1500000000.jpg';cv2.imwrite(str(path),np.zeros((480,640),np.uint8))
    second=obj.observe(1_500_000_000,1_500_000_000,.026,rotation_previous_current=T[:3,:3])
    assert second['accepted'] and second['kind']=='relative_yaw' and second['solver']=='pnp'
    assert second['keyframe_timestamp_ns']==d['timestamp_ns']


def test_angular_cap_preserves_rare_view_cells_without_increasing_point_budget():
    rng=np.random.default_rng(513)
    common=np.c_[rng.uniform(1.,4.,11000),rng.uniform(-.03,.03,11000),rng.uniform(-.15,.15,11000)]
    rare=np.c_[rng.uniform(1.,4.,1000),rng.choice([-1.,1.],1000)*rng.uniform(.4,1.,1000),rng.uniform(-.15,.15,1000)]
    sample=cp.asarray(np.r_[common,rare]);selected,d=angular_stratified_cap(sample,.2,1000)
    again,_=angular_stratified_cap(sample,.2,1000)
    assert len(selected)==1000 and d['angular_cells_after']==d['angular_cells_before']
    assert bool(cp.any(selected[:,1]>.4)) and bool(cp.any(selected[:,1]<-.4))
    cp.testing.assert_array_equal(selected,again)


def test_normal_profile_distinguishes_many_parallel_points_from_diverse_surfaces():
    def profile(normals):
        n=cp.asarray(normals);count=len(n)
        return surface_profile(dict(good=cp.ones(count,dtype=bool),normals=n,confidence=cp.ones(count)),
            cp.ones(count)*.1,cp.linspace(1.,39.,count),cp.ones(count))
    one=profile(np.tile([0.,1.,0.],(300,1)));three=profile(np.repeat(np.eye(3),100,axis=0))
    assert one['normal_effective_directions']==pytest.approx(1.)
    assert three['normal_effective_directions']==pytest.approx(3.)
    assert one['dominant_normal_bin_fraction']==pytest.approx(1.)
    assert sum(b['matches'] for b in three['range_bands'])==300


def test_line_constraints_do_not_invent_plane_normal_diversity_or_signed_residuals():
    n=20;good=cp.ones(n,dtype=bool)
    d=surface_profile(dict(good=good,normals=cp.zeros((n,3)),confidence=cp.ones(n),line_matches=good),
        cp.ones(n)*.2,cp.ones(n)*5.,cp.ones(n))
    assert d['line_matches']==20 and d['plane_matches']==0 and d['normal_effective_directions']==0
    assert d['dominant_normal_bin_fraction'] is None and d['range_bands'][0]['signed_median_m'] is None


def test_v513_image_enabled_runner_preserves_output_contract_and_prefix(tmp_path,monkeypatch):
    from test_track2_image_speed import test_image_enabled_runner_prefix_and_measurement_output
    test_image_enabled_runner_prefix_and_measurement_output(tmp_path,monkeypatch)


def test_metric_solver_diagnostics_do_not_relax_gates_and_log_monotone_fixed_losses():
    T=truth();prior=T.copy();prior[:2,3]+=[.2,-.15]
    reg=MetricRegistration(cp.asarray(map_scene()),T)
    out,scale,d=reg.refine(cp.asarray(raw_cloud(map_scene(),T,8.)),prior,1_000_000_000,prior_covariance=TEST_COV)
    assert d['accepted'] and d['geometry_accepted']
    assert 'surface_profile' in d and d['surface_profile']['normal_effective_directions']>=1.
    for stage in d['stages']:
        assert stage['iteration_trace']
        for step in stage['iteration_trace']:
            if step.get('step_accepted'):assert step['cost_after_fixed']<=step['cost_before']+1e-12
