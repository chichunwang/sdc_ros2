"""Generated evidence only: CPU references verify CUDA arithmetic/contracts."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cupy as cp
import pytest
from scipy.spatial.transform import Rotation
from track3.cuda_math import rotation_exp,create_rng
from track3.monte_carlo import MonteCarloState,normalize_log_weights,systematic_indices
from track3.monte_carlo_parameters import resolve_config
from track3.imu_parameters import resolve_imu_config

STAMP=1727239790226594548


def config():
    c=json.loads((Path(__file__).resolve().parents[1]/'config/track3.json').read_text())['imu']
    c=resolve_imu_config(c)
    c['T_ply_imu']=np.eye(4).tolist();c['extrinsic_status']='synthetic_identity'
    return c


def state(n=64,imu=None):
    c=resolve_config();c.update(particles=n,max_points=256,batch_particles=16)
    return MonteCarloState(np.eye(4),3.,config() if imu is None else imu,c)


def sample(t,w=(0.,0.,0.)):
    return SimpleNamespace(timestamp_ns=int(t),header_timestamp_ns=int(t),arrival_timestamp_ns=None,
        angular_velocity=np.array(w),linear_acceleration=np.array([0.,0.,9.80665]))


def estimate(T=None,accepted=True,scale=3.):
    return dict(T=np.eye(4) if T is None else T,scale=scale,accepted=accepted,reason='accepted' if accepted else 'low_coverage')


def anchored(s):
    s.predict(STAMP,[sample(STAMP)])
    d=s.commit(estimate());assert d['inertial']['update']['bootstrap']
    return s


def registration_map():
    rng=np.random.default_rng(1986)
    m=rng.uniform([-3,-4,-2],[8,5,3],(3000,3))
    return m,SimpleNamespace(map=cp.asarray(m),cfg=dict(submap_rebuild_distance_m=15.,submap_half_extent_m=[90.,90.,30.]))


@pytest.mark.parametrize('magnitude',[0.,1e-12,.3,3.1415926535])
def test_cuda_rotation_exponential_matches_reference(magnitude):
    v=np.array([[magnitude,0.,0.],[0.,-magnitude/2,magnitude/2]])
    got=cp.asnumpy(rotation_exp(v))
    np.testing.assert_allclose(got,Rotation.from_rotvec(v).as_matrix(),rtol=1e-13,atol=1e-14)


def test_log_weights_and_ess_survive_extreme_offsets():
    logw,w,ess=normalize_log_weights(cp.array([-1e8,-1e8-1,-cp.inf]))
    expected=np.array([1.,np.exp(-1),0.]);expected/=expected.sum()
    np.testing.assert_allclose(cp.asnumpy(w),expected,rtol=1e-14)
    assert float(ess)==pytest.approx(1/(expected@expected))
    assert not bool(cp.any(cp.isnan(logw)))
    logs,_,_=normalize_log_weights(cp.array([-10000.,-11000.]))
    assert bool(cp.isfinite(logs).all()) and float(logs[1])==pytest.approx(-1000.)
    with pytest.raises(ValueError):normalize_log_weights(cp.full(4,-cp.inf))


def test_systematic_resampling_uniform_and_concentrated():
    rng=create_rng(42)
    np.testing.assert_array_equal(cp.asnumpy(systematic_indices(cp.full(64,1/64),rng)),np.arange(64))
    w=cp.zeros(64);w[17]=1.
    assert bool(cp.all(systematic_indices(w,rng)==17))


def test_generated_map_prefers_matching_mode_without_mode_averaging(tmp_path):
    s=anchored(state());m,reg=registration_map()
    s.predict(STAMP+36_000_000,[sample(STAMP+30_000_000)])
    # The map spans 11 m in X. A 24 m shift guarantees no overlap even
    # after the 1.5 m search radius; partial-overlap modes are not "invalid".
    s.R[:]=cp.eye(3);s.p[:]=0.;s.p[16:,0]=24.;s.logs[:]=cp.log(3.);s.logw[:]=-cp.log(float(s.n))
    before=cp.asnumpy(s.p).copy()
    prior=s.observe_cloud(m/3.,reg)
    assert abs(prior[0,3])<1e-12
    assert s.measurement['valid_particles']<s.n
    assert float(cp.sum(cp.exp(s.logw[:16])))>.99
    np.savez(tmp_path/'particle_mode_evidence.npz',positions=before,weights=cp.asnumpy(cp.exp(s.logw)),
             source=m/3.,map_points=m,selected_T=prior,truth_T=np.eye(4))
    (tmp_path/'particle_mode_metrics.json').write_text(json.dumps(s.measurement,indent=2)+'\n')
    T=np.eye(4);T[0,3]=.05
    d=s.commit(estimate(T));assert d['accepted'] and not d['monte_carlo']['resampled']
    assert s.representative>=0 and s.representative<s.n
    # Current-frame candidates are not a posterior to be recentered. The next
    # prediction starts from the committed output, not from these old candidates.
    np.testing.assert_array_equal(cp.asnumpy(s.p),before)
    assert s.T[0,3]==.05 and not d['monte_carlo']['posterior_recentered_by_refinement']


def test_all_bad_particles_preserve_weights_and_rejected_pose():
    s=anchored(state());m,reg=registration_map()
    prior=s.predict(STAMP+36_000_000,[sample(STAMP+30_000_000)])
    s.p[:]=1000.;before=s.logw.copy()
    np.testing.assert_array_equal(s.observe_cloud(m/3.,reg),prior)
    assert not s.measurement['applied']
    T=np.eye(4);T[0,3]=999.
    d=s.commit(estimate(T,accepted=False))
    assert not d['accepted'];np.testing.assert_array_equal(cp.asnumpy(s.logw),cp.asnumpy(before))
    np.testing.assert_array_equal(s.T,prior)


def test_failed_refinement_rolls_back_likelihood_without_resampling():
    s=anchored(state());m,reg=registration_map()
    s.predict(STAMP+36_000_000,[sample(STAMP+30_000_000)])
    before=s.logw.copy();s.observe_cloud(m/3.,reg)
    assert s.measurement['applied']
    d=s.commit(estimate(accepted=False))
    np.testing.assert_array_equal(cp.asnumpy(s.logw),cp.asnumpy(before))
    assert not d['monte_carlo']['resampled']


def test_startup_records_eligible_modes_but_commits_the_selected_anchor():
    s=state(128);s.predict(STAMP,[sample(STAMP)])
    other=np.eye(4);other[0,3]=10.
    e=estimate();e['initialization']=dict(refinements=[dict(eligible=True,T=np.eye(4).tolist(),scale=3.),dict(eligible=True,T=other.tolist(),scale=4.)])
    d=s.commit(e)
    assert d['monte_carlo']['bootstrap_anchors']==2
    assert bool(cp.all(s.p[:,0]==0.))
    assert d['monte_carlo']['next_prediction_matches_output']
    assert s.scale==3. and s.T[0,3]==0.


def test_stationary_retains_exact_output_and_scale_evidence():
    s=anchored(state());T=s.T.copy();R=s.R.copy();p=s.p.copy()
    prior=s.predict(STAMP+36_000_000,[sample(STAMP+30_000_000)],stationary=True,input_scale=3.1)
    m,reg=registration_map();s.observe_cloud(m/3.,reg)
    candidate=np.eye(4);candidate[0,3]=10.
    d=s.commit(estimate(candidate,scale=4.))
    np.testing.assert_array_equal(s.T,T);np.testing.assert_array_equal(prior,T)
    np.testing.assert_allclose(cp.asnumpy(s.R),np.tile(T[:3,:3],(s.n,1,1)),atol=1e-14)
    np.testing.assert_allclose(cp.asnumpy(s.p),np.tile(T[:3,3],(s.n,1)),atol=1e-14)
    assert s.scale==3.1 and d['decision']=='stationary_pose_hold'


@pytest.mark.parametrize('bad',['float_frame','future','old','nan','stale'])
def test_bad_time_or_imu_is_rejected_before_state_mutation(bad):
    s=anchored(state());p=s.p.copy();R=s.R.copy();stamp=s.stamp
    t=STAMP+36_000_000;batch=[sample(STAMP+30_000_000)]
    if bad=='float_frame':t=float(t)
    if bad=='future':batch=[sample(t+1)]
    if bad=='old':batch=[sample(STAMP)]
    if bad=='nan':batch[0].angular_velocity[0]=np.nan
    if bad=='stale':t=STAMP+200_000_000;batch=[]
    with pytest.raises(ValueError):s.predict(t,batch)
    assert s.stamp==stamp and s.pending is None
    np.testing.assert_array_equal(cp.asnumpy(p),cp.asnumpy(s.p));np.testing.assert_array_equal(cp.asnumpy(R),cp.asnumpy(s.R))


def test_causal_visual_displacement_uses_header_dt_and_prior_body_frame():
    s=anchored(state());s.R[:]=cp.eye(3);s.p[:]=0.;s.bg[:]=0.
    s.cfg['position_diffusion_m_sqrt_s']=0.;s.cfg['visual_velocity_std_mps']=0.
    M=np.eye(4);M[0,3]=.3
    s.predict(STAMP+36_000_000,[sample(STAMP+30_000_000)],visual_motion=M)
    np.testing.assert_allclose(cp.asnumpy(s.p),np.tile([.3,0,0],(s.n,1)),atol=1e-14)
    np.testing.assert_allclose(cp.asnumpy(s.v[:,0]),.3/.036,atol=1e-14)
    assert not s.prediction_diagnostics['kalman_used']


def test_gyro_extrinsic_and_lever_arm_rotation_contract():
    c=config();E=np.eye(4);E[:3,:3]=Rotation.from_euler('x',.4).as_matrix();E[:3,3]=[.5,.2,.1]
    c['T_ply_imu']=E.tolist();c['gyro_noise_density']=1e-20;c['gyro_bias_random_walk']=1e-20
    s=state(16,c);s.predict(STAMP,[sample(STAMP,(0.,0.,1.))]);s.commit(estimate())
    s.R[:]=cp.eye(3);s.p[:]=0.;s.v[:]=0.;s.bg[:]=0.
    s.cfg['position_diffusion_m_sqrt_s']=0.;s.cfg['velocity_diffusion_mps_sqrt_s']=0.
    s.predict(STAMP+36_000_000,[sample(STAMP+30_000_000,(0.,0.,1.))])
    truth=E[:3,:3]@Rotation.from_rotvec([0,0,.036]).as_matrix()@E[:3,:3].T
    np.testing.assert_allclose(s.T[:3,:3],truth,atol=2e-14)
    np.testing.assert_allclose(cp.asnumpy(s.R[0]),truth,atol=2e-14)
    expected=-(truth-np.eye(3))@E[:3,3]
    np.testing.assert_allclose(cp.asnumpy(s.p),np.tile(expected,(s.n,1)),atol=2e-14)


def test_visual_to_velocity_fallback_keeps_imu_origin_velocity():
    c=config();E=np.eye(4);E[:3,3]=[.5,.2,.1];c['T_ply_imu']=E.tolist()
    c['gyro_noise_density']=c['gyro_bias_random_walk']=1e-20
    s=state(16,c);s.predict(STAMP,[sample(STAMP,(0.,0.,1.))]);s.commit(estimate())
    s.R[:]=cp.eye(3);s.p[:]=0.;s.v[:]=0.;s.bg[:]=0.
    for name in ('position_diffusion_m_sqrt_s','visual_velocity_std_mps','velocity_diffusion_mps_sqrt_s'):s.cfg[name]=0.
    M=np.eye(4);M[0,3]=.3;dt=.036
    s.predict(STAMP+36_000_000,[sample(STAMP+30_000_000,(0.,0.,1.))],visual_motion=M)
    R1=Rotation.from_rotvec([0,0,dt]).as_matrix();v=(M[:3,3]+(R1-np.eye(3))@E[:3,3])/dt
    np.testing.assert_allclose(cp.asnumpy(s.v),np.tile(v,(s.n,1)),atol=2e-14)
    s.commit(estimate(accepted=False));s.predict(STAMP+72_000_000,[sample(STAMP+60_000_000,(0.,0.,1.))])
    R2=Rotation.from_rotvec([0,0,2*dt]).as_matrix();expected=M[:3,3]+v*dt-(R2-R1)@E[:3,3]
    np.testing.assert_allclose(cp.asnumpy(s.p),np.tile(expected,(s.n,1)),atol=2e-14)


def test_prefix_reproducibility_uses_local_rng_and_never_future_data():
    def run(n):
        s=state(16);out=[]
        for i in range(n):
            s.predict(STAMP+i*36_000_000,[sample(STAMP+i*36_000_000)])
            s.commit(estimate(accepted=i==0));out.append(s.T.copy())
        return out
    short=run(3);long=run(5)
    np.testing.assert_array_equal(short,long[:3])


def test_camera_header_clock_keeps_particle_motion_causal(tmp_path):
    from test_sensor_clock import database
    from track3.imu import ImuReader
    from track3.time_alignment import CameraClock,AcquisitionImuReader
    path=database(tmp_path);cam=CameraClock(path);imu=AcquisitionImuReader(ImuReader(path))
    s=state(16)
    try:
        t0=cam.for_frame(STAMP+32_000_000);batch=imu.take_for_frame(STAMP+32_000_000,t0)
        s.predict(t0,batch);s.commit(estimate())
        s.R[:]=cp.eye(3);s.p[:]=0.;s.bg[:]=0.
        s.cfg['visual_velocity_std_mps']=0.;s.cfg['position_diffusion_m_sqrt_s']=0.
        t1=cam.for_frame(STAMP+44_000_000);batch=imu.take_for_frame(STAMP+44_000_000,t1)
        motion=np.eye(4);motion[0,3]=.3
        s.predict(t1,batch,visual_motion=motion)
        assert t1-t0==36_000_000
        np.testing.assert_allclose(cp.asnumpy(s.v[:,0]),.3/.036,atol=1e-14)
        assert s.prediction_diagnostics['latest_bag_timestamp_ns']<=STAMP+44_000_000
        assert s.prediction_diagnostics['latest_header_timestamp_ns']<=t1
        assert all(sample.timestamp_ns<=t1 for sample in batch)
    finally:imu.close();cam.close()


def test_registration_solver_never_calls_numpy_solve(monkeypatch):
    from test_core import CFG,cloud
    from track3.gpu import Registration,linearize,VoxelModel
    cfg=deepcopy(CFG['registration']);cfg['solver_device']='cuda'
    def forbidden(*a,**k):raise AssertionError('CPU linear solve called')
    monkeypatch.setattr(np.linalg,'solve',forbidden)
    m=cloud();p=m/3.
    H,g,*_=linearize(cp.asarray(p),np.eye(4),3.,VoxelModel(m,2.),'ndt',.3,device=True)
    assert isinstance(H,cp.ndarray) and isinstance(g,cp.ndarray)
    result=Registration(m,cfg).align(p,np.eye(4),3.)
    assert result['accepted'] and isinstance(result['hessian'],cp.ndarray)


def test_ground_initializer_solver_never_calls_numpy_solve(monkeypatch):
    from test_startup_selection import scene
    from track3.initialization import GroundInitializer
    m,source,truth,prior,cfg=scene()
    cfg['initializer']['solver_device']='cuda'
    def forbidden(*a,**k):raise AssertionError('CPU ground solve called')
    monkeypatch.setattr(np.linalg,'solve',forbidden)
    result=GroundInitializer(m,cfg['initializer']).align(source,prior,3.)
    assert abs(result['scale']-7.)<.2
    assert np.linalg.norm(result['T'][:3,3]-truth[:3,3])<.2
