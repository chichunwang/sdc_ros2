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
from track3.io import frames, load_ply, sha256, write_json
from track3.geometry import valid_pose
from track1 import parameters as p

TIMINGS = 'idx,original_idx,timestamp_ns,backend,load_ms,registration_ms,total_ms,map_accepted,scale'.split(',')
OUTPUTS = ('result_1.csv', 'poses.txt', 'timings.csv', 'frame_manifest.csv', 'observations.jsonl', 'pose_trace.jsonl')


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
    if (run/'test_info.json').exists():
        raise FileExistsError('Run already used')
    start = time.perf_counter()
    info = dict(status='running', version=p.VERSION, track=1, frame_count=0,
                mode='full' if max_frames is None else 'test', expected_frames=p.EXPECTED_FRAMES,
                score=None, score_source=None, graph_seconds=0.,
                causality='forward GNSS heading prior + current-frame metric scale/ground + causal correction-state filter',
                base_commit=p.BASE_COMMIT,
                timing_clock='camera header; availability cutoff=PLY bag timestamp')
    write_json(run/'test_info.json', info)
    write_json(run/'parameters.json', {k: v for k, v in vars(p).items() if k.isupper()})
    sensors = clock = visualizer = None
    tlocal = None
    registration_seconds = 0.
    accepted_count = 0
    def stopped(signum, _frame):
        raise InterruptedError(f'Received signal {signum}')
    old_handlers = {s: signal.signal(s, stopped) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        import cupy as cp
        from track1.map_io import load_gpu_map
        from track1.metric_registration import MetricRegistration
        from track1.camera import FrontMask
        from track1.causal_prior import CausalSensorPrior
        from track1.sensors import SensorStream
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
                  for name in ('map.pcd', 'init_guess.txt', 'Track1.db3')}
        write_json(run/'input_sources.json', dict(sources=inputs, map_fields_used=['x', 'y', 'z'],
            pose_convention='T_map_ply, camera FLU, metres; yaw atan2(R10,R00) radians',
            gnss_policy='forward XY CV Kalman sigma=0.3m; latest altitude only for first-frame height; empty frame assumed map; zero lever uncalibrated',
            imu_policy='original guided model: latest IMU quaternion + causal gyro extrapolation; compatible axes/zero lever assumed'))
        warm = time.perf_counter()
        (cp.eye(3) @ cp.eye(3)).sum().item()
        info['warmup_seconds'] = time.perf_counter()-warm
        map_start = time.perf_counter()
        xyz, count = load_gpu_map(data/'map.pcd', p.METRIC_MAP_LEAF_M)
        info.update(map_load_seconds=time.perf_counter()-map_start, map_raw_points=count,
                    map_voxel_points=len(xyz), gpu_device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
        registration = MetricRegistration(xyz, initial)
        camera = FrontMask(data)
        write_json(run/'camera_sources.json', camera.sources)
        if publish_rviz:
            from track1.visualization import Track1RvizPublisher
            visualizer = Track1RvizPublisher()
            visualizer.publish_map(cp.asnumpy(xyz[::max(1, (len(xyz)+299999)//300000)]), rows[0]['timestamp_ns'])
        info['rviz_enabled'] = bool(publish_rviz)
        sensors = SensorStream(data/'Track1.db3', p.GNSS_EXPECTED_FRAME)
        clock = CameraClock(data/'Track1.db3')
        state = None
        previous_height = None
        info['preparation_seconds'] = time.perf_counter()-start
        tlocal = time.perf_counter()
        with ExitStack() as stack:
            files = {name: stack.enter_context((run/name).open('x', newline='')) for name in OUTPUTS}
            result = csv.writer(files['result_1.csv']); result.writerow(['idx', 'x', 'y', 'yaw'])
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
                masked, camera_diag = camera.apply(load_ply(row['path']))
                raw = cp.asarray(masked)
                source_sha = sha256(row['path'])
                cp.cuda.get_current_stream().synchronize()
                load_ms = (time.perf_counter()-tick)*1000
                reg_tick = time.perf_counter()
                final, scale, reg_diag = registration.refine(raw, prior, camera_ns,
                    speed_mps=float(np.linalg.norm(state.position[2:])))
                cp.cuda.get_current_stream().synchronize()
                reg_seconds = time.perf_counter()-reg_tick
                registration_seconds += reg_seconds
                map_decision = dict(accepted=reg_diag['accepted'], reason=reg_diag['reason'])
                accepted_count += int(map_decision['accepted'])
                previous_height = float(final[2, 3])
                if not valid_pose(final) or not np.isfinite(state.position).all() or not np.isfinite(state.cov).all():
                    raise ValueError('Nonfinite or invalid emitted pose/filter')
                if visualizer is not None:
                    if registration.nn is not None:
                        visualizer.publish_submap(cp.asnumpy(registration.nn.points), stamp)
                    visualizer.publish_frame(cp.asnumpy(raw), final, scale, stamp,
                        accepted=map_decision['accepted'], reason=map_decision['reason'],
                        scale_validated=reg_diag['scale_validated'],
                        speed_mps=float(np.linalg.norm(state.position[2:])))
                total_ms = (time.perf_counter()-tick)*1000
                yaw = float(np.arctan2(final[1, 0], final[0, 0]))
                result.writerow([idx, f'{final[0,3]:.9f}', f'{final[1,3]:.9f}', f'{yaw:.11f}'])
                files['poses.txt'].write(' '.join(f'{v:.12f}' for v in final[:3].ravel())+'\n')
                timings.writerow([idx, row['original_idx'], stamp, 'cuda_metric_scale', f'{load_ms:.6f}',
                                  f'{reg_seconds*1000:.6f}', f'{total_ms:.6f}', str(map_decision['accepted']).lower(), f'{scale:.12f}'])
                manifest.writerow([idx, row['original_idx'], stamp, camera_ns, str(row['path']), source_sha])
                observation = dict(idx=idx, original_idx=row['original_idx'], timestamp_ns=stamp,
                    camera_timestamp_ns=camera_ns, elapsed_seconds=time.perf_counter()-tlocal,
                    registration=reg_diag, map_update=map_decision, gnss=sensor_updates,
                    scale=scale, camera_mask=camera_diag, sensor_prior=prior_diag,
                    state=[float(final[0,3]), float(final[1,3]), yaw, state.bias, float(np.linalg.norm(state.position[2:]))],
                    position_covariance_diag=np.diag(state.cov).tolist(),
                    sensor_counts=sensors.summary())
                files['observations.jsonl'].write(json.dumps(observation, allow_nan=False)+'\n')
                trace = dict(idx=idx, timestamp_ns=stamp, predicted=prior.tolist(),
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
            from track1.analysis import analyze
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
        a.error('Short test requires 1..3010 frames')
    execute(args.run_dir.resolve(), args.max_frames, publish_rviz=args.publish_rviz)
