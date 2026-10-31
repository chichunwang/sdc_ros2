"""生成資料的 covariance／可觀測性／因果航向檢查；不需要 CUDA 或真實資料。"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from track2 import parameters as p
from track2.causal_prior import CausalSensorPrior
from track2.correction_filter import CausalCorrectionFilter, wrap
from track2.observability import checked_covariance, geometry_observation


def imu(t, yaw=0., cov=None):
    return dict(kind='imu', timestamp_ns=int(t), gyro=[0., 0., 0.],
                quaternion=Rotation.from_euler('z', yaw).as_quat().tolist(),
                orientation_covariance=[0.]*9 if cov is None else np.asarray(cov).ravel().tolist(),
                gyro_covariance=[0.]*9)


def generated_prior(seconds, speed=2., course_deg=3., stationary=False):
    s = CausalSensorPrior(np.eye(4), 1_000_000_000)
    output=[]
    for k in range(round(seconds*20)+1):
        t = 1_000_000_000+k*50_000_000
        s.event(imu(t))
        if k % 5 == 0:
            angle = np.deg2rad(course_deg if k < 161 else course_deg+4.)
            # Continuous straight segments; second segment tests later rate cap.
            d = k*.05
            if stationary:
                xy = [.005*np.sin(d), .005*np.cos(d)]
            else:
                a = min(d, 8.)*speed*np.array([np.cos(np.deg2rad(course_deg)), np.sin(np.deg2rad(course_deg))])
                b = max(d-8., 0.)*speed*np.array([np.cos(angle), np.sin(angle)])
                xy = (a+b).tolist()
            s.event(dict(kind='gnss', timestamp_ns=t, xy=xy, z=1.6))
        T, diag = s.for_frame(t)
        output.append((T, diag))
    return s, output


def test_current_covariance_contains_heading_lever_and_predicts_between_observations():
    s = CausalSensorPrior(np.eye(4), 1_000_000_000)
    s.event(imu(1_000_000_000))
    s.event(dict(kind='gnss', timestamp_ns=1_000_000_000, xy=[0., 0.], z=1.6))
    _, a = s.for_frame(1_000_000_000)
    C = checked_covariance(a['pose_covariance_xy_yaw'])
    assert C[2, 2] > np.deg2rad(10.)**2
    assert C[0, 0] >= s.cov[0, 0]+p.LEVER_ARM_STD_BODY_M[0]**2-1e-12
    assert a['uncertainty_model']['orientation'] == 'unknown_engineering_budget'
    s.event(imu(2_000_000_000)); _, b = s.for_frame(2_000_000_000)
    D = checked_covariance(b['pose_covariance_xy_yaw'])
    assert D[0, 0] > C[0, 0] and D[2, 2] > C[2, 2]
    with pytest.raises(ValueError): checked_covariance(np.diag([1., 1., 0.]))


def test_lever_cross_covariance_matches_numerical_world_jacobian(monkeypatch):
    monkeypatch.setattr(p, 'GNSS_TO_PLY_LEVER_BODY_M', (1.2, -.4, .3))
    monkeypatch.setattr(p, 'LEVER_ARM_STD_BODY_M', (.01, .02, .03))
    monkeypatch.setattr(p, 'EXTRINSIC_ROTATION_STD_DEG', (0., 0., 0.))
    T0 = np.eye(4); R = Rotation.from_euler('xyz', [.1, -.07, .3]).as_matrix()
    s = CausalSensorPrior(T0, 1_000_000_000)
    raw_C = np.diag([.002, .003, .004])
    msg = imu(1_000_000_000, cov=raw_C); msg['quaternion'] = Rotation.from_matrix(R).as_quat().tolist()
    s.event(msg); s.bias_var = .005
    T, d = s.for_frame(1_000_000_000)
    lever = np.asarray(p.GNSS_TO_PLY_LEVER_BODY_M); offset = R @ lever
    numerical = np.zeros((3, 3)); epsilon = 1e-6
    for k in range(3):
        e = np.eye(3)[k]*epsilon
        a = Rotation.from_rotvec(e).as_matrix() @ R
        b = Rotation.from_rotvec(-e).as_matrix() @ R
        numerical[:2, k] = ((a @ lever)[:2]-(b @ lever)[:2])/(2*epsilon)
        numerical[2, k] = wrap(np.arctan2(a[1, 0], a[0, 0])-np.arctan2(b[1, 0], b[0, 0]))/(2*epsilon)
    world_C = R @ raw_C @ R.T; world_C[2, 2] += .005
    expected = numerical @ world_C @ numerical.T
    expected[:2, :2] += s.cov[:2, :2]+(R @ np.diag(np.array(p.LEVER_ARM_STD_BODY_M)**2) @ R.T)[:2, :2]
    np.testing.assert_allclose(T[:3, 3], offset, atol=1e-12)
    np.testing.assert_allclose(d['pose_covariance_xy_yaw'], expected, atol=1e-9)
    assert abs(expected[0, 2])+abs(expected[1, 2]) > 1e-3


def test_past_displacement_observability_initializes_below_old_five_mps_gate():
    s, output = generated_prior(8., speed=2.)
    accepted = [r for r in s.block_records if r['accepted']]
    assert accepted and accepted[0]['speed_mps'] < p.HEADING_MIN_SPEED_MPS
    assert accepted[0]['delivered_timestamp_ns'] < 1_000_000_000+8_000_000_000
    assert accepted[0]['std_deg'] <= p.HEADING_MAX_COURSE_STD_DEG
    assert output[-1][1]['heading_bias_rad'] > np.deg2rad(2.)
    for before, after in zip(accepted, accepted[1:]):
        assert before['end_timestamp_ns'] < after['start_timestamp_ns']
    for k, (_, d) in enumerate(output):
        assert all(r['end_timestamp_ns'] <= r['delivered_timestamp_ns'] <= 1_000_000_000+k*50_000_000 for r in d['block_records'])


def test_stationary_cannot_invent_course_and_future_prefix_is_identical():
    s, _ = generated_prior(20., stationary=True)
    assert s.accepted_blocks == 0 and s.bias == 0.
    assert len(s.block) <= round(p.HEADING_MAX_WINDOW_S*4)+1
    _, short = generated_prior(5.)
    _, long = generated_prior(12.)
    for (T, a), (U, b) in zip(short, long):
        np.testing.assert_array_equal(T, U); assert a == b


def test_heading_covariance_uses_effective_rate_limited_gain():
    s, _ = generated_prior(20., speed=4.)
    accepted = [r for r in s.block_records if r['accepted']]
    limited = [r for r in accepted if r['bias_gain'] < r['unconstrained_bias_gain']-1e-8]
    assert limited
    for r in accepted:
        K = r['bias_gain']
        expected = (1-K)**2*r['bias_variance_before']+K*K*r['measurement_variance']
        assert r['bias_variance_after'] == pytest.approx(expected, abs=1e-15)


def geometry(H, J=None, C=None):
    return geometry_observation(H, np.eye(3, 4) if J is None else J,
        np.diag([.5**2, .5**2, np.deg2rad(10.)**2]) if C is None else C,
        dict(coverage=.9, median_plane_residual_m=.05))


def test_null_geometry_has_no_measurement_row_and_cannot_correct_tangent():
    H, R, d = geometry(np.diag([100., 0., 1000., 100.]))
    assert d['rank'] == 2
    np.testing.assert_allclose(H[:, 1], 0., atol=1e-14)
    f = CausalCorrectionFilter(); f.predict(np.eye(4), 1_000_000_000)
    mean, update = f.update([.2, 99., .01], R, observation_matrix=H)
    assert update['accepted'] and mean[1] == 0.
    assert f.covariance[1, 1] == pytest.approx(1.)
    K = np.asarray(update['kalman_gain']); A = np.eye(3)-K @ H
    expected = A @ np.array(update['covariance_predicted']) @ A.T+K @ R @ K.T
    np.testing.assert_allclose(f.covariance, expected, atol=1e-12)
    empty, covariance, d = geometry(np.zeros((4, 4)))
    assert empty.shape == (0, 3) and covariance.shape == (0, 0) and d['rank'] == 0


def test_scale_translation_gauge_is_not_false_pose_information():
    v = np.array([1., 0., 0., 10.])
    information = 100*np.outer(v, v)+np.diag([0., 100., 1000., 0.])
    H, _, d = geometry(information)
    assert d['rank'] == 2
    np.testing.assert_allclose(H[:, 0], 0., atol=1e-10)
    assert d['scale_information'] < 1e-8


def test_geometry_analysis_respects_covariance_and_units_and_slope_scale_coupling():
    information = np.diag([100., 50., 2000., 500.])
    C = np.array([[.25, .03, .002], [.03, .16, -.001], [.002, -.001, .03]])
    J = np.array([[.99, -.1, 0., .06], [.1, .99, 0., -.04], [0., 0., .98, 0.]])
    H, R, d = geometry(information, J, C)
    # Coordinate reparametrization x=B*x' must not create map information.
    B = np.array([[1., .2, 0., .3], [0., 1., 0., -.2], [0., 0., 2., .01], [0., 0., 0., 1.]])
    _, _, e = geometry(B.T @ information @ B, J @ B, C)
    np.testing.assert_allclose(d['normalized_eigenvalues'], e['normalized_eigenvalues'], rtol=1e-10)
    assert d['scale_information'] == pytest.approx(e['scale_information'], rel=1e-10)
    centimetres = np.diag([100., 100., 1.])
    _, _, c = geometry(information, centimetres @ J, centimetres @ C @ centimetres.T)
    np.testing.assert_allclose(d['normalized_eigenvalues'], c['normalized_eigenvalues'], rtol=1e-10)
    _, _, weak = geometry(np.diag([1., 1., 1., 100.]), C=np.diag([.01, .01, .01]))
    assert weak['rank'] == 0  # data adds less information than the current sensor model
    assert np.linalg.eigvalsh(R).min() > 0 and H.shape == (3, 3)


def test_yaw_gain_limit_does_not_reduce_safe_xy_and_joseph_uses_actual_gain():
    f = CausalCorrectionFilter(); T = np.eye(4)
    R = np.diag([.15**2, .15**2, np.deg2rad(.5)**2])
    f.predict(T, 1_000_000_000); f.update([0., 0., 0.], R)
    f.predict(T, 1_100_000_000, 0.)
    before = f.mean.copy()
    out, d = f.update([.01, 0., np.deg2rad(.5)], R)
    assert d['accepted'] and d['gain_scale_xy'] == 1. and 0 < d['gain_scale_yaw'] < 1.
    P = np.array(d['covariance_predicted']); K0 = np.linalg.solve(P+R, P).T
    assert out[0]-before[0] == pytest.approx((K0 @ np.array(d['innovation']))[0])
    K = np.array(d['kalman_gain']); A = np.eye(3)-K
    np.testing.assert_allclose(f.covariance, A @ P @ A.T+K @ R @ K.T, atol=1e-12)


def test_newly_visible_direction_uses_nis_not_a_false_raw_motion_jump():
    f = CausalCorrectionFilter(); T = np.eye(4)
    f.predict(T, 1_000_000_000)
    f.update([0., 99., .5], np.array([[.04]]), observation_matrix=np.array([[1., 0., 0.]]))
    f.predict(T, 1_100_000_000)
    _, d = f.update([0., .3, 0.], np.diag([.04, .04, .01]), observation_matrix=np.eye(3))
    assert d['accepted'], d
    np.testing.assert_allclose(d['motion_innovation'], [0., 0., 0.], atol=1e-12)
    assert d['observation_dof'] == 3


def test_heading_alignment_is_not_physical_vehicle_rotation():
    s, output = generated_prior(8., speed=2.)
    assert s.accepted_blocks > 0
    assert all(d['gyro_rotation_since_frame_rad'] == 0. for _, d in output)
    f = CausalCorrectionFilter(); f.predict(np.eye(4), 1_000_000_000); f.update()
    T = np.eye(4); T[:3, :3] = Rotation.from_euler('z', .1).as_matrix()
    f.predict(T, 1_100_000_000, rotation_delta_rad=0.)
    _, d = f.update()
    assert d['motion']['prior_rotation_rad'] == 0.
    assert d['motion']['rotation_source'] == 'arrived_gyro_integral'


def test_gain_limits_cannot_create_motion_in_a_geometric_null_direction():
    f = CausalCorrectionFilter(); T = np.eye(4)
    unit = np.diag([1., 1., np.deg2rad(3.)])
    H = np.array([[1., 0., 1/np.deg2rad(3.)]])/np.sqrt(2)
    f.predict(T, 1_000_000_000); f.update([0., 0., 0.], np.array([[.04]]), observation_matrix=H)
    f.predict(T, 1_100_000_000, 0.)
    before = f.mean.copy()
    out, d = f.update([.05, 99., np.deg2rad(1.)], np.array([[.04]]), observation_matrix=H)
    assert d['accepted'], d
    null = np.array([-1., 0., 1.])/np.sqrt(2)
    np.testing.assert_allclose(null @ np.linalg.solve(unit, out-before), 0., atol=1e-14)
    assert out[1] == 0.
    assert np.linalg.norm((out-before)[:2]) <= d['position_update_limit_m']+1e-12
    assert abs(out[2]-before[2]) <= d['yaw_update_limit_rad']+1e-12
    K = np.array(d['kalman_gain']); R = np.array(d['measurement_covariance'])/d['measurement_weight']
    A = np.eye(3)-K @ H
    np.testing.assert_allclose(f.covariance, A @ np.array(d['covariance_predicted']) @ A.T+K @ R @ K.T, atol=1e-12)


def test_calibration_change_reexpresses_only_learned_map_memory_without_double_heading():
    T = np.eye(4); C = np.diag([.36, .36, np.deg2rad(10.)**2])
    R = np.diag([.15**2, .15**2, np.deg2rad(.5)**2])
    f = CausalCorrectionFilter()
    f.predict(T, 1_000_000_000, prior_covariance=C)
    old, _ = f.update([.2, -.1, np.deg2rad(3.)], R)
    sensitivity = f.reference_sensitivity.copy()
    change = np.array([0., 0., np.deg2rad(3.)])
    T[:3,:3] = Rotation.from_euler('z', change[2]).as_matrix()
    f.predict(T, 1_050_000_000, prior_covariance=C, rotation_delta_rad=0., reference_delta=change)
    current, d = f.update(None)
    np.testing.assert_allclose(current, old-sensitivity @ change, atol=1e-14)
    # Strong map heading + new GNSS calibration must NOT apply the old 3° twice.
    assert abs((change+current)[2]-old[2]) < np.deg2rad(.01)
    np.testing.assert_allclose(f.last_measurement, [.2, -.1, 0.], atol=1e-14)
    # No map, or map only supporting XY, must allow the new heading through.
    for H in (None, np.array([[1.,0.,0.]])):
        g = CausalCorrectionFilter(); g.predict(np.eye(4), 1_000_000_000, prior_covariance=C)
        if H is None: g.update()
        else: g.update([.2, 99., .5], np.array([[.04]]), observation_matrix=H)
        g.predict(T, 1_050_000_000, prior_covariance=C, rotation_delta_rad=0., reference_delta=change)
        out, _ = g.update(None)
        assert out[2] == 0.


def test_reference_sensitivity_uses_limited_gain_and_fades_with_stale_memory():
    f = CausalCorrectionFilter(); T = np.eye(4); R = np.diag([.02,.02,.001])
    f.predict(T,1_000_000_000);f.update([.1,.1,.01],R)
    before = f.reference_sensitivity.copy()
    f.predict(T,1_100_000_000)
    _,d=f.update([.3,.2,.03],R)
    K=np.asarray(d['kalman_gain']); H=np.eye(3)
    np.testing.assert_allclose(f.reference_sensitivity,(np.eye(3)-K@H)@before+K@H,atol=1e-14)
    before=f.reference_sensitivity.copy()
    f.predict(T,3_100_000_000);_,d=f.update(None)
    np.testing.assert_allclose(f.reference_sensitivity,before*d['decay_factor'],atol=1e-14)


def test_causal_heading_and_map_memory_do_not_apply_the_same_alignment_twice():
    def simulate(seconds):
        _, priors = generated_prior(seconds, speed=2.)
        f = CausalCorrectionFilter(); result=[]; physical=[]
        for k, (T, d) in enumerate(priors):
            C=np.asarray(d['pose_covariance_xy_yaw'])
            f.predict(T,1_000_000_000+k*50_000_000,np.linalg.norm(d['velocity_xy_mps']),
                prior_covariance=C, rotation_delta_rad=d['gyro_rotation_since_frame_rad'],
                reference_delta=d['calibration_delta_xy_yaw'])
            heading=np.arctan2(T[1,0],T[0,0])
            _, update=f.update([0.,0.,wrap(np.deg2rad(3.)-heading)],
                np.diag([.15**2,.15**2,np.deg2rad(.5)**2]))
            physical.append(heading+f.mean[2]);result.append(update)
        return np.asarray(physical), result, priors
    short,a,_=simulate(4.)
    long,b,priors=simulate(6.)
    np.testing.assert_array_equal(short,long[:len(short)]);assert a==b[:len(a)]
    transitions=[k for k,(_,d) in enumerate(priors) if abs(d['calibration_delta_xy_yaw'][2])>1e-6]
    assert transitions
    for k in transitions:
        assert abs(np.degrees(long[k]-long[k-1]))<.02
        assert b[k]['accepted']


def test_reference_response_accounts_for_sensor_regularization_of_the_candidate():
    f = CausalCorrectionFilter();T=np.eye(4);R=np.diag([.02,.02,.001])
    # A quadratic data term as strong as its prior has candidate response 1/2.
    B=np.eye(3)*.5
    for k in range(80):
        f.predict(T,1_000_000_000+k*50_000_000)
        f.update([.1, .05, .02],R,candidate_sensitivity=B)
    before=f.mean.copy();sensitivity=f.reference_sensitivity.copy()
    delta=np.array([.2,.1,.04])
    f.predict(T,5_000_000_000,reference_delta=delta,rotation_delta_rad=0.)
    predicted=f.mean.copy()
    np.testing.assert_allclose(predicted,before-sensitivity@delta,atol=1e-14)
    np.testing.assert_allclose(f.last_measurement,0.,atol=1e-14)
    assert sensitivity[2,2]<.51  # do not rebase as if candidate ignored its prior
    _,d=f.update([0.,0.,0.],R,candidate_sensitivity=B)
    assert d['accepted'],d


def test_reliable_subset_information_is_not_penalized_twice_by_global_plane_fraction():
    H4=np.diag([100.,100.,1000.,100.]);J=np.eye(3,4);C=np.diag([.36,.36,.03])
    a=geometry_observation(H4,J,C,dict(coverage=.9,median_plane_residual_m=.1))
    b=geometry_observation(H4,J,C,dict(coverage=.08,median_plane_residual_m=.1))
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
    assert a[2]['rank']==b[2]['rank']==3
    # Losing reliable samples reduces data information itself, not an unrelated
    # whole-image percentage applied again to an identical measured subset.
    c=geometry_observation(H4*.1,J,C,dict(coverage=.08,median_plane_residual_m=.1))
    assert np.trace(c[1])>np.trace(b[1])
