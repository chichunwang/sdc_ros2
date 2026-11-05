#!/usr/bin/env python3
"""每幀線上完成 pose 並立即寫出；禁止重新寫入已輸出 pose。"""
import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import signal
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from track3.io import frames, sha256, write_json
from track2.ply_io import load_ply
from track3.geometry import valid_pose
from track2 import parameters as p

TIMINGS = 'idx,original_idx,timestamp_ns,backend,load_ms,registration_ms,total_ms,map_accepted,scale'.split(',')
OUTPUTS = ('result_2.csv', 'poses.txt', 'timings.csv', 'frame_manifest.csv', 'observations.jsonl', 'pose_trace.jsonl')


def verify_snapshot(run):
    root = Path(__file__).resolve().parents[1]
    if root != run/'source':
        raise ValueError('Must execute this run source snapshot')
    manifest = json.loads((run/'source_manifest.json').read_text())
    for name, digest in manifest['sources'].items():
        if sha256(root/name) != digest:
            raise ValueError(f'Source snapshot SHA mismatch: {name}')


def execute(run, max_frames=None, publish_rviz=False):
    verify_snapshot(run)
    if not p.uses_v52_fusion():
        raise ValueError('Historical fusion profile is restricted to component regressions')
    if p.OUTPUT_REFERENCE != 'camera_ply':
        raise ValueError('Only camera_ply output is implemented; unverified body relabeling is prohibited')
    if (run/'test_info.json').exists():
        raise FileExistsError('Run already used')
    start = time.perf_counter()
    info = dict(status='running', version=p.VERSION, track=2, frame_count=0,
                mode='full' if max_frames is None else 'test', expected_frames=p.EXPECTED_FRAMES,
                score=None, score_source=None, graph_seconds=0.,
                causality=('arrived-gyro coordinated-turn XY prediction + disjoint image/GNSS-confirmed rest gyro calibration + forward GNSS heading + current-frame metric registration and causal correction filter' if p.v5162_enabled() else 'forward GNSS heading prior + disjoint past/current image speed + current-frame metric scale/ground + causal correction-state filter'),
                base_commit=p.BASE_COMMIT,
                timing_clock='camera header; availability cutoff=PLY bag timestamp')
    write_json(run/'test_info.json', info)
    write_json(run/'parameters.json', {k: v for k, v in vars(p).items() if k.isupper()})
    sensors = clock = visualizer = image_speed = None
    tlocal = None
    registration_seconds = 0.
    accepted_count = 0
    def stopped(signum, _frame):
        raise InterruptedError(f'Received signal {signum}')
    old_handlers = {s: signal.signal(s, stopped) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        import cupy as cp
        from track2.map_io import load_gpu_map
        from track2.metric_registration import MetricRegistration
        from track2.camera import FrontMask
        from track2.causal_prior import CausalSensorPrior
        from track2.sensors import SensorStream
        from track3.time_alignment import CameraClock
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('CUDA required; CPU fallback is prohibited')
        # Limit CuPy pool without altering Track3 or global driver state.
        cp.get_default_memory_pool().set_limit(size=4*1024**3)
        data = Path(p.DATA_DIR)
        rows = frames(data/'raw_ply', expected=p.EXPECTED_FRAMES)
        if max_frames is not None:
            rows = rows[:max_frames]
        initial = np.loadtxt(data/'init_guess.txt', delimiter=',')
        if not valid_pose(initial):
            raise ValueError('Invalid initial pose')
        inputs = {name: dict(path=str(data/name), sha256=sha256(data/name))
                  for name in ('map.pcd', 'init_guess.txt', 'Track2.db3')}
        write_json(run/'input_sources.json', dict(sources=inputs, map_fields_used=['x', 'y', 'z'],
            pose_convention='T_map_ply, camera FLU, metres; yaw atan2(R10,R00) radians',
            gnss_policy='forward XY CV Kalman and current quaternion with unknown GNSS heading bias; unknown GNSS frame assumed map; lever mean and uncertainty from Python parameters; altitude not a height calibration',
            imu_policy='arrived orientation + causal gyro; joint IMU/GNSS heading biases; past observable course anchors common bias; body small-angle covariance and IMU-to-PLY extrinsic assumptions logged per frame'))
        warm = time.perf_counter()
        (cp.eye(3) @ cp.eye(3)).sum().item()
        info['warmup_seconds'] = time.perf_counter()-warm
        map_start = time.perf_counter()
        xyz, count = load_gpu_map(data/'map.pcd', p.METRIC_MAP_LEAF_M)
        info.update(map_load_seconds=time.perf_counter()-map_start, map_raw_points=count,
                    map_voxel_points=len(xyz), gpu_device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
        registration = MetricRegistration(xyz, initial)
        camera = FrontMask(data)
        if p.IMAGE_GROUND_SPEED_ENABLED:
            from track2.image_speed import CausalGroundSpeed
            image_speed = CausalGroundSpeed(data, camera)
        info['image_ground_speed_enabled'] = bool(p.IMAGE_GROUND_SPEED_ENABLED)
        write_json(run/'camera_sources.json', camera.sources)
        write_json(run/'reference_contract.json', dict(output=p.OUTPUT_REFERENCE,
            pose='T_map_ply, FLU, metres, yaw atan2(R10,R00) radians',
            status=p.REFERENCE_CONTRACT_STATUS, calibration_status=p.EXTRINSIC_CALIBRATION_STATUS,
            gnss_to_ply_lever_body_m=p.GNSS_TO_PLY_LEVER_BODY_M,
            imu_to_ply_rpy_deg=p.IMU_TO_PLY_RPY_DEG, author_track1_offsets_applied=False,
            limitation='official scoring frame/lever and physical sensor latency are not established by timestamp pairing'))
        write_json(run/'algorithm_scope.json', dict(version=p.VERSION, baseline_run='track02_full_2611031941',
            baseline_score=0.42699, baseline_score_source='user_report',
            numerical_changes=(['v5.16.4 original v5.10 map association, causal turn and finest road patches retained; fixed40m',
                'height anchor uses data-only local Schur likelihood once; unchanged fixed-plane nonlinear validation and Joseph height filter; no current pose/scale policy change'] if p.v5165_enabled() else [('v5.16.2 causal turn and v5.16.1 map association retained; fixed40m'
                if p.v5161_enabled() else 'v5.10 original map association restored; v5.16.2 causal turn retained; fixed40m'),
                'road-plane speed features require the complete finest LK window inside the past validated road mask; original40-track gate unchanged'] if p.v5163_enabled() else
                ['v5.16.1 map association and v5.10 fusion; 40m and original quality gates retained',
                'XY velocity propagated with arrived IMU yaw rate; exact constant-turn transition and isotropic acceleration covariance',
                'disjoint image/GNSS-confirmed stationary intervals learn motion-only gyro bias; no past pose or absolute-heading correction'] if p.v5162_enabled() else
                ['v5.16/v5.10 numerical fusion baseline; 40m and original quality gates retained',
                'bounded source/map normal-compatible surface fallback for unreliable nearest map points',
                'fallback only after past map anchor and yaw std <=3deg; original reliable matches preserved',
                'current correspondence radius and one-map-voxel distance margin; confidence=min(source,target)'] if p.v5161_enabled() else
                ['v5.10 frozen numerical baseline; historical v5.11-v5.14 branches disabled',
                'data-only current-map likelihood; sensor prior used once',
                'positive weak directions retain their uncertainty; exact numerical null omitted; only strong directions refresh memory',
                'actual posterior gain backtracking on fixed planes and rematched same sample; Joseph covariance uses actual gain'] if p.v515_enabled() else
                ['v5.10 production numerical baseline; v5.11/v5.12/v5.12.1 disabled', 'causal height/ground anchors independent of current map-pose acceptance',
                'exact feature rays; pixel-refined gyro translation; scalar relative speed or one road fallback; disjoint intervals' if p.v514_enabled() else 'sparse background PnP compared with gyro-fixed translation; disjoint vector-velocity and relative-yaw intervals',
                'angular-stratified feature cap and normal/solver diagnostics; original quality gates retained'] if p.v513_enabled() else
                ['v5.10 numerical baseline restored after scored v5.15 regression',
                 f'fixed source feature range {p.FEATURE_MAX_RANGE_M:g} m; original sampling, registration and fusion']),
            feature_max_range_m=p.FEATURE_MAX_RANGE_M,
            v5165_enabled=bool(p.v5165_enabled()),
            height_filter_model='data_only_local_likelihood_once' if p.v5165_enabled() else 'legacy_MAP_height_center',
            v5163_enabled=bool(p.v5163_enabled()),
            road_patch_support_policy='finest LK window inside past road mask' if p.v5163_enabled() else None,
            v5162_enabled=bool(p.v5162_enabled()),
            turn_gyro_rest_std_degps=p.TURN_GYRO_REST_STD_DEGPS,
            turn_gyro_bias_rw_degps_sqrt_s=p.TURN_GYRO_BIAS_RW_DEGPS_SQRT_S,
            v5161_enabled=bool(p.v5161_enabled()),
            surface_fallback_normal_max_deg=p.SURFACE_FALLBACK_NORMAL_MAX_DEG,
            surface_fallback_yaw_std_deg=p.SURFACE_FALLBACK_YAW_STD_DEG,
            metric_submap_half_extent_m=p.METRIC_SUBMAP_HALF_EXTENT_M,
            metric_correspondence_stages=p.METRIC_STAGES,
            v515_enabled=bool(p.v515_enabled()),
            v513_enabled=bool(p.v513_enabled()),
            v514_enabled=bool(p.v514_enabled()),
            image_geometry_contract=('v5.10 validated-ground scalar speed; no relative-image fallback' if not p.v513_enabled() else
                'exact feature ray / pixel reprojection / scalar relative speed / one road fallback' if p.v514_enabled() else 'v5.13 nearest PLY point / algebraic gyro translation / map vector'),
            image_anchor_memory_enabled=bool(p.v513_enabled() and p.IMAGE_ANCHOR_MEMORY_ENABLED),
            relative_translation_enabled=bool(p.v513_enabled() and p.IMAGE_RELATIVE_TRANSLATION_ENABLED),
            relative_yaw_enabled=bool(p.v513_enabled() and p.IMAGE_RELATIVE_YAW_ENABLED),
            relative_translation_solver=p.IMAGE_RELATIVE_SOLVER,
            angular_stratified_cap=bool(p.v513_enabled() and p.MAP_ANGULAR_STRATIFIED_CAP),
            geometry_diagnostics=bool(p.v513_enabled() and p.MAP_GEOMETRY_DIAGNOSTICS),
            retained_v57=['EPNP image speed', f'{p.FEATURE_MAX_RANGE_M:g}m feature range', 'MAP height fusion', 'directional map correction memory'],
            excluded_v59=['adaptive feature ranges']+([] if p.v5165_enabled() else ['data-only height update'])+([] if p.v5162_enabled() else ['stationary gyro bias learning']),
            heading_enabled=p.v510_heading_enabled(),
            modules={name:sha256(Path(__file__).resolve().parents[1]/name) for name in
                ('track2/image_speed.py','track2/metric_registration.py','track2/visual_update.py','track2/causal_prior.py','track2/parameters.py','track2/coordinated_motion.py')}))
        if publish_rviz:
            from track2.visualization import Track2RvizPublisher
            visualizer = Track2RvizPublisher()
            visualizer.publish_map(cp.asnumpy(xyz[::max(1, (len(xyz)+299999)//300000)]), rows[0]['timestamp_ns'])
        info['rviz_enabled'] = bool(publish_rviz)
        sensors = SensorStream(data/'Track2.db3', p.GNSS_EXPECTED_FRAME)
        clock = CameraClock(data/'Track2.db3')
        state = None
        previous_height = None
        info['preparation_seconds'] = time.perf_counter()-start
        tlocal = time.perf_counter()
        with ExitStack() as stack:
            files = {name: stack.enter_context((run/name).open('x', newline='')) for name in OUTPUTS}
            result = csv.writer(files['result_2.csv']); result.writerow(['idx', 'x', 'y', 'yaw'])
            timings = csv.writer(files['timings.csv']); timings.writerow(TIMINGS)
            manifest = csv.writer(files['frame_manifest.csv'])
            manifest.writerow(['idx', 'original_idx', 'timestamp_ns', 'camera_timestamp_ns', 'path', 'sha256'])
            for idx, row in enumerate(rows):
                tick = time.perf_counter()
                stamp = row['timestamp_ns']
                camera_ns = clock.for_frame(stamp)
                events = sensors.take(stamp, camera_ns)
                if state is None:
                    state = CausalSensorPrior(initial, camera_ns)
                sensor_updates = []
                for event in events:
                    decision = state.event(event, preframe=event['timestamp_ns'] < state.time)
                    if event['kind'] == 'gnss':
                        sensor_updates.append(dict(event, update=decision))
                prior, prior_diag = state.for_frame(camera_ns, previous_height)
                sensor_done = time.perf_counter()
                cloud = load_ply(row['path'])
                ply_done = time.perf_counter()
                masked, camera_diag = camera.apply(cloud)
                raw = cp.asarray(masked)
                cp.cuda.get_current_stream().synchronize()
                mask_done = time.perf_counter()
                image_observation = dict(accepted=False, filter_accepted=False, reason='disabled')
                image_started = time.perf_counter()
                if image_speed is not None:
                    from track2.visual_update import fuse_visual_speed
                    measured = image_speed.observe(stamp, camera_ns, prior_diag['gyro_rotation_since_frame_rad'],
                        **({'rotation_previous_current':prior_diag['gyro_rotation_matrix_since_frame'], 'gyro_std_bound_rad':prior_diag['gyro_rotation_std_bound_rad']} if p.v513_enabled() else {}))
                    prior, prior_diag, image_observation = fuse_visual_speed(
                        state, prior, prior_diag, measured, float(np.linalg.norm(state.gyro)))
                image_done = time.perf_counter()
                image_ms = (image_done-image_started)*1000
                source_sha = sha256(row['path'])
                sha_done = time.perf_counter()
                load_ms = (time.perf_counter()-tick)*1000
                reg_tick = time.perf_counter()
                final, scale, reg_diag = registration.refine(raw, prior, camera_ns,
                    speed_mps=float(np.linalg.norm(state.position[2:])),
                    prior_covariance=np.asarray(prior_diag['pose_covariance_xy_yaw']),
                    rotation_delta_rad=prior_diag['gyro_rotation_since_frame_rad'],
                    reference_delta=prior_diag['calibration_delta_xy_yaw'])
                cp.cuda.get_current_stream().synchronize()
                reg_seconds = time.perf_counter()-reg_tick
                registration_seconds += reg_seconds
                map_decision = dict(accepted=reg_diag['accepted'], reason=reg_diag['reason'])
                store_started = time.perf_counter()
                if image_speed is not None:
                    image_speed.store(raw, scale, reg_diag, **({'reference_rotation':prior[:3,:3]} if p.v513_enabled() else {}))
                    if p.v513_enabled() and image_speed.anchor is not None:
                        anchor_stamp=image_speed.anchor['stamp']
                        if state.visual_heading_anchor is None or state.visual_heading_anchor['timestamp_ns']!=anchor_stamp:
                            if anchor_stamp!=camera_ns:raise ValueError('Cannot invent a past heading anchor')
                            state.capture_visual_anchor(camera_ns,prior[:3,:3])
                    image_observation['anchor_store']=image_speed.anchor_diagnostic if p.v513_enabled() else None
                image_store_ms = (time.perf_counter()-store_started)*1000
                accepted_count += int(map_decision['accepted'])
                previous_height = float(final[2, 3])
                if not valid_pose(final) or not np.isfinite(state.position).all() or not np.isfinite(state.cov).all():
                    raise ValueError('Nonfinite or invalid emitted pose/filter')
                display_tick = time.perf_counter()
                if visualizer is not None:
                    if registration.nn is not None:
                        visualizer.publish_submap(cp.asnumpy(registration.nn.points), camera_ns)
                    # Keep the full point transform on CUDA. ROS still needs
                    # host float32 message bytes; this is the only display copy.
                    display_cloud = raw @ cp.asarray(final[:3, :3]).T*scale+cp.asarray(final[:3, 3])
                    visualizer.publish_frame(cp.asnumpy(display_cloud.astype(cp.float32)), final, scale, camera_ns,
                        accepted=map_decision['accepted'], reason=map_decision['reason'],
                        scale_validated=reg_diag['scale_validated'],
                        speed_mps=float(np.linalg.norm(state.position[2:])), cloud_in_map=True)
                display_ms = (time.perf_counter()-display_tick)*1000
                total_ms = (time.perf_counter()-tick)*1000
                yaw = float(np.arctan2(final[1, 0], final[0, 0]))
                result.writerow([idx, f'{final[0,3]:.9f}', f'{final[1,3]:.9f}', f'{yaw:.11f}'])
                files['poses.txt'].write(' '.join(f'{v:.12f}' for v in final[:3].ravel())+'\n')
                timings.writerow([idx, row['original_idx'], stamp, 'cuda_observable_plane', f'{load_ms:.6f}',
                                  f'{reg_seconds*1000:.6f}', f'{total_ms:.6f}', str(map_decision['accepted']).lower(), f'{scale:.12f}'])
                manifest.writerow([idx, row['original_idx'], stamp, camera_ns, str(row['path']), source_sha])
                observation = dict(idx=idx, original_idx=row['original_idx'], timestamp_ns=stamp,
                    camera_timestamp_ns=camera_ns, elapsed_seconds=time.perf_counter()-tlocal,
                    registration=reg_diag, map_update=map_decision, gnss=sensor_updates,
                    phase_timings_ms=dict(sensor_prior=(sensor_done-tick)*1000,
                        ply_read=(ply_done-sensor_done)*1000, camera_mask_cuda=(mask_done-ply_done)*1000,
                        image_ground_speed=image_ms,
                        image_ground_model_store=image_store_ms,
                        source_sha256=(sha_done-image_done)*1000,
                        registration=reg_seconds*1000, rviz=display_ms,
                        clock='synchronized wall time; CPU/GPU overlap is not separated'),
                    scale=scale, camera_mask=camera_diag, sensor_prior=prior_diag, image_ground_speed=image_observation,
                    state=[float(final[0,3]), float(final[1,3]), yaw, state.bias, float(np.linalg.norm(state.position[2:]))],
                    position_covariance_diag=np.diag(state.cov).tolist(),
                    sensor_counts=sensors.summary())
                files['observations.jsonl'].write(json.dumps(observation, allow_nan=False)+'\n')
                trace = dict(idx=idx, timestamp_ns=stamp, camera_timestamp_ns=camera_ns, predicted=prior.tolist(),
                    registered=reg_diag['candidate_pose'] if reg_diag['geometry_accepted'] else None,
                    final=final.tolist(), map_accepted=map_decision['accepted'], scale=scale,
                    geometry_accepted=reg_diag['geometry_accepted'],
                    correction_filter=reg_diag['correction_filter'])
                files['pose_trace.jsonl'].write(json.dumps(trace, allow_nan=False)+'\n')
                for f in files.values():
                    f.flush()
                info.update(frame_count=idx+1, map_accepted_frames=accepted_count,
                            registration_seconds=registration_seconds, localization_seconds=time.perf_counter()-tlocal)
                write_json(run/'test_info.json', info)
                print(f"idx={row['original_idx']} map={map_decision['accepted']} scale={scale:.4f} "
                      f"xy=({final[0,3]:.3f},{final[1,3]:.3f}) "
                      f"rank={0 if reg_diag.get('observability') is None else reg_diag['observability']['rank']} "
                      f"scale_valid={reg_diag['scale_validated']} v={np.linalg.norm(state.position[2:]):.3f} bias={np.degrees(state.bias):.4f}deg elapsed={time.perf_counter()-tlocal:.1f}s", flush=True)
        info['status'] = 'completed'
    except BaseException as exc:
        info.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else 'failed',
                    failure_type=type(exc).__name__, failure=str(exc))
        (run/'failure.log').write_text(traceback.format_exc())
        raise
    finally:
        if tlocal is not None:
            info['localization_seconds'] = time.perf_counter()-tlocal
        if sensors is not None:
            info['sensors'] = sensors.summary(); sensors.close()
        if clock is not None:
            clock.close()
        if visualizer is not None:
            visualizer.close()
        info.update(registration_seconds=registration_seconds, total_seconds=time.perf_counter()-start,
                    output_sha256={name: sha256(run/name) for name in OUTPUTS if (run/name).exists()})
        write_json(run/'test_info.json', info)
        for s, handler in old_handlers.items():
            signal.signal(s, handler)
    if max_frames is None and info['frame_count'] == p.EXPECTED_FRAMES:
        plot_start = time.perf_counter()
        try:
            from track2.analysis import analyze
            analyze(run)
            write_json(run/'analysis_status.json', dict(status='completed', elapsed_seconds=time.perf_counter()-plot_start))
        except Exception as exc:
            (run/'analysis_failure.log').write_text(traceback.format_exc())
            write_json(run/'analysis_status.json', dict(status='failed', failure=str(exc), elapsed_seconds=time.perf_counter()-plot_start))
        # Timing belongs to analysis_status.json; preserve test_info.json SHA
        # after analysis takes its source manifest.


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--run-dir', type=Path, required=True)
    a.add_argument('--publish-rviz', action='store_true')
    mode = a.add_mutually_exclusive_group(required=True)
    mode.add_argument('--full', action='store_true')
    mode.add_argument('--max-frames', type=int)
    args = a.parse_args()
    if args.max_frames is not None and not 0 < args.max_frames < p.EXPECTED_FRAMES:
        a.error('Short test requires 1..3298 frames')
    execute(args.run_dir.resolve(), args.max_frames, publish_rviz=args.publish_rviz)
