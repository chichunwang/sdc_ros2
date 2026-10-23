#!/usr/bin/env python3
"""Track3 forward IMU Kalman + motion rejection + bounded current-frame scale."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import shlex
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'track3_causal_vertical_imu_kalman_sim3_v2'
BACKEND = 'cuda_float64_geometry_cpu_small_solver_forward_kalman'
TIMING_HEADER = ['idx', 'original_idx', 'timestamp_ns', 'backend', 'load_ms',
                 'registration_ms', 'total_ms', 'map_accepted', 'scale']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--track', type=int, choices=(3,), default=3)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--full', action='store_true')
    group.add_argument('--frames', type=int)
    p.add_argument('--scale-mode', choices=('fixed', 'estimate'), default='estimate')
    p.add_argument('--initial-scale', type=float, default=3., help='uncalibrated numerical seed, not a calibration')
    p.add_argument('--scale-min', type=float, default=.5)
    p.add_argument('--scale-max', type=float, default=16.)
    p.add_argument('--data-root', type=Path, default=ROOT.parent/'midterm01/data')
    p.add_argument('--container', default='sdc_midterm01_gpu')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--dry-run', action='store_true')
    group.add_argument('--audit-only', action='store_true', help='CPU input/coordinate checks only; no poses/GPU/map fitting')
    args = p.parse_args(argv)
    args.frames = 100 if args.frames is None else args.frames
    import math
    if (args.frames <= 0 or not all(math.isfinite(v) and v > 0 for v in
            (args.initial_scale, args.scale_min, args.scale_max))
            or not args.scale_min < args.initial_scale < args.scale_max):
        p.error('positive frames and 0 < scale-min < initial-scale < scale-max are required')
    if args.audit_only and args.full:
        p.error('--audit-only uses --frames, not --full')
    args.data_root = args.data_root.resolve()
    return args


def select_frames(args):
    frames = []
    for path in (args.data_root/'Track3/raw_ply').iterdir():
        match = re.fullmatch(r'(\d+)_ts_(\d+)\.ply', path.name)
        if path.is_file() and match:
            frames.append((int(match[1]), int(match[2]), path))
        elif path.is_file() and path.suffix.lower() == '.ply':
            raise ValueError('Malformed PLY source name: '+path.name)
    frames.sort(key=lambda row: row[0])
    if (not frames or any(r[1] <= 0 for r in frames)
            or [r[0] for r in frames] != list(range(len(frames)))
            or any(b[1] <= a[1] for a, b in zip(frames, frames[1:]))):
        raise ValueError('Expected complete ordered 0-based PLY names and increasing integer times')
    if args.full and len(frames) != 3142:
        raise ValueError('Track3 full requires exactly 3142 source frames')
    selected = frames if args.full else frames[:args.frames]
    if not selected or (not args.full and len(selected) != args.frames):
        raise ValueError('Requested prefix exceeds available source')
    return selected


def verify_front_timestamps(bag, selected):
    """Only compare the selected prefix with bag metadata; do not decode images."""
    with sqlite3.connect(bag.resolve().as_uri()+'?mode=ro', uri=True) as connection:
        topic = connection.execute('SELECT id,type FROM topics WHERE name=?',
                                   ('/lucid_cameras_x00/gige_100_f_hdr/h265',)).fetchall()
        if len(topic) != 1 or topic[0][1] != 'sensor_msgs/msg/Image':
            raise ValueError('Missing/wrong front camera topic schema')
        stamps = [r[0] for r in connection.execute(
            'SELECT timestamp FROM messages WHERE topic_id=? AND timestamp<=? ORDER BY timestamp,id',
            (topic[0][0], selected[-1][1]))]
    if stamps != [r[1] for r in selected]:
        raise ValueError('PLY timestamps do not match the front bag prefix')


def verify_container(args, run):
    result = subprocess.run(['docker', 'inspect', args.container], capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)[0]
    mounts = {m['Destination']: Path(m['Source']).resolve() for m in info['Mounts']}
    required = {'/root/data': args.data_root, '/root/outputs': (ROOT/'outputs').resolve(),
                '/root/scripts': (ROOT/'scripts').resolve(), '/root/midterm': (ROOT/'workspace').resolve()}
    if not info['State']['Running'] or any(mounts.get(k) != v for k, v in required.items()):
        raise ValueError('Running GPU container data/outputs/scripts/workspace mounts do not match this checkout')
    data_mount = next(m for m in info['Mounts'] if m['Destination'] == '/root/data')
    if data_mount.get('RW') is not False:
        raise ValueError('Original data must be mounted read-only')
    write_json(run/'container_mounts.json', {k: str(v) for k, v in mounts.items()})


def export_imu(args, run, end_ns):
    crun = '/root/outputs/'+run.relative_to(ROOT/'outputs').as_posix()
    command = ['docker', 'exec', args.container, 'bash', '-c',
               'source /opt/ros/humble/setup.bash && PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 exec python3 '+
               shlex.join(['/root/scripts/export_track3_imu.py', '--bag', '/root/data/Track3/Track3.db3',
                           '--output', crun+'/imu.npz', '--end-ns', str(end_ns)])]
    write_json(run/'imu_export_command.json', command)
    with (run/'imu_export.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def coordinate_audit(initial, arrays, first_ns, first_path, np):
    from scipy.spatial.transform import Rotation
    from three_camera_calibration import load_ply
    selected = arrays['availability_ns'] <= first_ns
    if not selected.any():
        raise ValueError('No IMU was available at the first frame')
    q = arrays['quaternion_xyzw'][selected]
    a = arrays['accel_m_s2'][selected]
    world_a = Rotation.from_quat(q).apply(a)
    norm = np.linalg.norm(world_a, axis=1)
    # Sanity check of the DECLARED +Z-up planar approximation, using only data
    # available before initialization. This cannot calibrate a missing extrinsic.
    vertical_cosine = world_a[:, 2]/np.maximum(norm, 1e-12)
    gravity_check = bool(np.median(vertical_cosine) > .8 and 5. < np.median(norm) < 15.)
    if not gravity_check:
        raise ValueError('Initial IMU acceleration is inconsistent with declared +Z-up approximation')
    cloud = load_ply(first_path, max_points=12000, leaf=.03)
    return {'matrix_convention': 'p_map = T_map_source @ [s*p_raw,1]; no inverse or axis transform',
            'initial_pose': initial.tolist(),
            'rotation_det': float(np.linalg.det(initial[:3, :3])),
            'rotation_orthogonality_error': float(np.max(abs(initial[:3, :3].T @ initial[:3, :3]-np.eye(3)))),
            'first_raw_xyz_quantiles': np.quantile(cloud, [0, .5, 1], axis=0).tolist(),
            'first_raw_positive_axis_fractions': (cloud > 0).mean(axis=0).tolist(),
            'first_available_imu_samples': int(selected.sum()),
            'initial_imu_world_acceleration_median': np.median(world_a, axis=0).tolist(),
            'initial_vertical_cosine_median': float(np.median(vertical_cosine)),
            'gravity_sanity_passed': gravity_check,
            'imu_model': 'gyro projected using orientation-world +Z, assumed parallel to map +Z; planar approximation',
            'absolute_imu_yaw_used': False, 'accelerometer_position_integration': False,
            'imu_camera_extrinsic_known': False, 'upstream_raw_ply_axes_certified': False,
            'initial_guess_accuracy_known': False, 'official_submission_sample_verified': False,
            'calibration_status': 'numerical convention checked; physical frame/calibration not certified',
            'noise_status': 'all-zero message covariance means unknown; filter uses explicit engineering noise assumptions'}


def gpu_crop_voxel(points, center, torch, np, *, half_size=100., leaf=.3):
    """Chunked GPU crop and deterministic original-row voxel representative."""
    low = torch.tensor(np.asarray(center)-half_size, device='cuda', dtype=torch.float64)
    high = low+2*half_size
    chunks = []
    for start in range(0, len(points), 262144):
        # Copy avoids writable aliases to the immutable memory-mapped raw cache.
        block = torch.tensor(np.array(points[start:start+262144], dtype=np.float64), device='cuda')
        selected = ((block >= low) & (block <= high)).all(dim=1)
        if bool(selected.any().item()):
            chunks.append(block[selected])
    if not chunks:
        raise ValueError('Empty static map tile around the current prediction')
    local = torch.cat(chunks)
    cells = torch.floor(local/leaf).to(torch.int64)
    minimum = cells.amin(dim=0)
    sizes = (cells.amax(dim=0)-minimum+1).cpu().tolist()
    if int(sizes[0])*int(sizes[1])*int(sizes[2]) >= 2**63:
        raise ValueError('Map voxel key exceeds int64')
    offsets = cells-minimum
    keys = (offsets[:, 0]*sizes[1]+offsets[:, 1])*sizes[2]+offsets[:, 2]
    order = torch.argsort(keys, stable=True)
    ordered = keys[order]
    first = torch.ones(len(order), device='cuda', dtype=torch.bool)
    first[1:] = ordered[1:] != ordered[:-1]
    keep = torch.sort(order[first], stable=True).values
    output = local[keep].cpu().numpy()
    torch.cuda.synchronize()
    return output, {'raw_crop_points': len(local), 'voxel_points': len(output), 'leaf_m': leaf,
                    'bounds': {'min': low.cpu().tolist(), 'max': high.cpu().tolist()},
                    'voxel_method': 'stable first original XYZ row, no centroid/coordinate change'}


def validate_saved(run, expected, np):
    def read_csv(name):
        with (run/name).open() as stream:
            return list(csv.DictReader(stream))
    rows = read_csv('diagnostic.csv')
    frames = read_csv('frames.csv')
    timings = read_csv('timings.csv')
    matrices = np.loadtxt(run/'result.txt', ndmin=2)
    errors = []
    if matrices.shape != (expected, 12) or not np.isfinite(matrices).all():
        return ['invalid pose matrix count/shape/finite values']
    poses = matrices.reshape(-1, 3, 4)
    for label, records in [('CSV', rows), ('frames', frames), ('timings', timings)]:
        if len(records) != expected or [int(r['idx']) for r in records] != list(range(expected)):
            errors.append(label+' count/idx mismatch')
    if not rows or not timings:
        return errors+['empty CSV or timings']
    if list(rows[0]) != ['idx', 'x', 'y', 'yaw'] or list(timings[0]) != TIMING_HEADER:
        return errors+['fixed output header mismatch']
    if len(rows) == expected:
        saved = np.array([[float(r[k]) for k in ('x', 'y', 'yaw')] for r in rows])
        truth = np.column_stack([poses[:, 0, 3], poses[:, 1, 3], np.arctan2(poses[:, 1, 0], poses[:, 0, 0])])
        if not np.isfinite(saved).all() or not np.allclose(saved, truth, atol=1e-8, rtol=0):
            errors.append('CSV and matrices disagree')
    r = poses[:, :, :3]
    if not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=1e-6, rtol=0) or not np.allclose(np.linalg.det(r), 1., atol=1e-6, rtol=0):
        errors.append('improper rotations')
    if [r['timestamp_ns'] for r in timings] != [r['timestamp_ns'] for r in frames]:
        errors.append('timings/frame timestamp mismatch')
    ns = [int(r['timestamp_ns']) for r in frames]
    if any(s <= 0 for s in ns) or any(b <= a for a, b in zip(ns, ns[1:])):
        errors.append('invalid/nonincreasing frame timestamps')
    if [int(r['original_idx']) for r in timings] != list(range(expected)):
        errors.append('original_idx must match the original 0-based prefix')
    values = np.array([[float(r[k]) for k in ('load_ms', 'registration_ms', 'total_ms', 'scale')] for r in timings])
    if not np.isfinite(values).all() or (values < 0).any() or (values[:, 3] <= 0).any():
        errors.append('invalid timing or scale values')
    observation_path = run/'observations.jsonl'
    if observation_path.exists():
        observations = [json.loads(line) for line in observation_path.read_text().splitlines()]
        if len(observations) != expected:
            errors.append('observation count mismatch')
        elif any(r['timestamp_ns'] != stamp or r['imu']['max_imu_availability_ns'] > stamp
                 or r['imu']['last_imu_measurement_ns'] > stamp for r, stamp in zip(observations, ns)):
            errors.append('observation timestamp/future-IMU violation')
    return errors


def main(argv=None):
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    args = arguments(argv)
    selected = select_frames(args)
    from causal_track3_filter import CausalKalman, FilterConfig, ImuCursor, read_initial_pose
    from causal_scale_registration import CausalScaleConfig
    initial = read_initial_pose(args.data_root/'Track3/init_guess.txt')
    filter_config = FilterConfig(min_scale=args.scale_min, max_scale=args.scale_max)
    scale_config = CausalScaleConfig(min_scale=args.scale_min, max_scale=args.scale_max,
                                    startup_rotation_deg=3*filter_config.initial_rotation_std_deg)
    plan = {'method': METHOD, 'track': 3, 'frames': len(selected), 'full': args.full,
            'data_root': str(args.data_root), 'container': args.container,
            'start_frame': 0, 'scale_mode': args.scale_mode, 'initial_scale': args.initial_scale,
            'initial_scale_status': 'uncalibrated numerical seed',
            'filter_model': asdict(filter_config), 'scale_model': asdict(scale_config),
            'initialization': 'current-frame ground-plane training seed; metric PCA per candidate scale; startup rotation bound 3*initial rotation std',
            'backend': BACKEND, 'cuda_fallback': False, 'gnss_used': False,
            'rts_used': False, 'batch_graph_used': False, 'offline_trajectory_correction': False,
            'imu_input': 'only availability_ns <= current frame; ZOH, no future interpolation',
            'imu_model': 'gravity-vertical planar gyro projection, absolute yaw unused; acceleration not integrated',
            'frame_contract': 'T_map_source, raw axes retained; camera/IMU extrinsic and upstream axes not certified',
            'submission_contract': '3142 rows/header known; official sample ID/yaw/frame not verified',
            'map_policy': 'static raw XYZ only; current-prediction tiles, CUDA .3m first-row voxel',
            'cpu_components': ['ASCII/SQLite/disk parsing', 'at most 7-unknown registration solver',
                               '11-state forward Kalman and numerical checks',
                               'exact kNN tie and voxel boundary adjudication when triggered'],
            'preparation_timing': 'source hashing/cache, CUDA warmup and tile preparation recorded separately',
            'timings_header': TIMING_HEADER, 'replay_supported': False}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    from accuracy.benchmark import create_run_directory
    run = create_run_directory(ROOT/'outputs/run', METHOD, track=3, full=args.full)
    print(f'Run: {run.resolve()}', flush=True)
    started = time.perf_counter()
    summary = {'track': 3, 'method': METHOD, 'backend': BACKEND, 'frames': len(selected),
               'expected_frames': len(selected), 'processed_frames': 0, 'returncode': 1,
               'output_complete_and_finite': False, 'submission_frame_set_complete': False,
               'validation_errors': [], 'map_accepted_frames': 0, 'tracking_lost_frames': 0,
               'tracking_established': False, 'first_map_accepted_original_idx': None,
               'max_unsupported_seconds': 0.,
               'graph_elapsed_seconds': 0., 'map_preparation_seconds': 0., 'map_frame_seconds': 0.,
               'cuda_warmup_seconds': 0.,
               'replay_supported': plan['replay_supported'], 'submission_compliance_certified': False,
               'accuracy': 'Not evaluated; no precision improvement or full-run stability claim'}
    try:
        import numpy as np
        write_json(run/'params.json', plan)
        info = json.loads((run/'test_info.json').read_text())
        info.update(algorithm=METHOD, score=None, localization_executed=False)
        write_json(run/'test_info.json', info)
        (run/'command.sh').write_text('#!/bin/sh\n'+shlex.join([sys.executable, str(Path(__file__).resolve()), *actual_argv])+'\n')
        sources = sorted(p for p in (ROOT/'scripts').rglob('*.py') if '__pycache__' not in p.parts)
        hashes = {}
        for source in sources:
            relative = source.relative_to(ROOT)
            content = source.read_bytes()
            copy = run/'pipeline_sources'/relative
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(content)
            hashes[str(relative)] = hashlib.sha256(content).hexdigest()
        write_json(run/'pipeline_source_sha256.json', hashes)
        for name, command in [('git_commit.txt', ['rev-parse', 'HEAD']), ('worktree.patch', ['diff', '--binary', 'HEAD'])]:
            result = subprocess.run(['git', '-C', str(ROOT.parent), *command], capture_output=True, check=True)
            (run/name).write_bytes(result.stdout)
        verify_container(args, run)
        identities = {name: {'path': str(args.data_root/'Track3'/name), 'sha256': digest(args.data_root/'Track3'/name)}
                      for name in ('init_guess.txt', 'map.pcd', 'Track3.db3')}
        write_json(run/'input_sha256.json', identities)
        verify_front_timestamps(args.data_root/'Track3/Track3.db3', selected)
        source_rows = [{'idx': i, 'original_idx': original, 'timestamp_ns': ns,
                        'filename': path.name, 'sha256': digest(path)} for i, (original, ns, path) in enumerate(selected)]
        write_json(run/'source_input.json', {'track': 3, 'frames': source_rows})
        with (run/'frames.csv').open('x', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['idx', 'timestamp_ns', 'filename'])
            writer.writerows((r['idx'], r['timestamp_ns'], r['filename']) for r in source_rows)
        export_imu(args, run, selected[-1][1])
        with np.load(run/'imu.npz', allow_pickle=False) as raw:
            arrays = {key: raw[key] for key in raw.files}
        audit = coordinate_audit(initial, arrays, selected[0][1], selected[0][2], np)
        write_json(run/'coordinate_audit.json', audit)
        print('Coordinates: T_map_source checked; camera/IMU extrinsic unavailable; using gravity-vertical planar IMU approximation.', flush=True)
        if args.audit_only:
            summary.update(audit_only=True, returncode=0, localization_executed=False)
            print(f'Audit: {run}/coordinate_audit.json (no localization/GPU)', flush=True)
            return 0
        import torch
        from scipy.spatial.transform import Rotation
        from causal_map_cache import cache_xyz_map
        from causal_scale_registration import CausalScaleRegistrar
        from cuda_scale_geometry import NORMAL_EIGH_BATCH_SIZE
        from cuda_scale_input import load_ply_gpu
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; CPU fallback disabled')
        preparation = time.perf_counter()
        cache = cache_xyz_map(args.data_root/'Track3/map.pcd', run/'map_cache')
        map_points = np.load(cache, mmap_mode='r', allow_pickle=False)
        summary['map_preparation_seconds'] += time.perf_counter()-preparation
        kalman = CausalKalman(initial, selected[0][1], args.initial_scale, filter_config)
        imu = ImuCursor(arrays)
        registrar, center = None, None
        tile_count = 0
        info.update(localization_executed=True)
        write_json(run/'test_info.json', info)
        write_json(run/'sensor_prior.json', {'source': 'current-frame forward Kalman prediction, past IMU/map only',
                                           'absolute_imu_yaw_used': False, 'gnss_used': False})
        files = {}
        try:
            for name in ('result.txt', 'diagnostic.csv', 'timings.csv', 'observations.jsonl', 'sensor_prior.csv'):
                files[name] = (run/name).open('x', newline='' if name.endswith('.csv') else None)
            csv_writer = csv.writer(files['diagnostic.csv']); csv_writer.writerow(['idx', 'x', 'y', 'yaw'])
            timing_writer = csv.writer(files['timings.csv']); timing_writer.writerow(TIMING_HEADER)
            prior_writer = csv.writer(files['sensor_prior.csv'])
            prior_writer.writerow(['idx', 'timestamp_ns', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'yaw'])
            for i, (original, ns, path) in enumerate(selected):
                summary['failed_original_idx'] = original
                imu_report = kalman.predict_to(ns, imu)
                predicted = kalman.pose.copy()
                if registrar is None or np.max(abs(predicted[:3, 3]-center)) > 40.:
                    preparation = time.perf_counter()
                    registrar = None  # Release previous tile before allocating the next.
                    torch.cuda.empty_cache()
                    center = np.floor(predicted[:3, 3]/40.+.5)*40.
                    tile, tile_report = gpu_crop_voxel(map_points, center, torch, np)
                    tile_dir = run/'map_tiles'/f'{tile_count:04d}'
                    tile_dir.mkdir(parents=True)
                    write_json(tile_dir/'identity.json', dict(tile_report, first_frame_idx=original, center=center.tolist(),
                               xyz_float64_sha256=hashlib.sha256(tile.astype('<f8', copy=False).tobytes()).hexdigest()))
                    registrar = CausalScaleRegistrar(tile, scale_config, cache_dir=tile_dir/'cuda_cache')
                    torch.cuda.synchronize()
                    warm_started = time.perf_counter()
                    registrar.scene.surfaces(np.arange(min(32, len(tile))))
                    warm_pose = np.eye(4); warm_pose[:3, 3] = tile[0]
                    # Empty-quality synthetic query warms pairing kernels. It
                    # estimates no sensor pose and uses no future measurements.
                    registrar.pairs(registrar.scene, np.zeros((1, 3)), np.array([[1., 0., 0.]]),
                                    np.zeros(1, bool), np.zeros(1), warm_pose, 1., 0., 3., registrar.base)
                    torch.cuda.synchronize()
                    warm_elapsed = time.perf_counter()-warm_started
                    summary['cuda_warmup_seconds'] += warm_elapsed
                    elapsed = time.perf_counter()-preparation-warm_elapsed
                    summary['map_preparation_seconds'] += elapsed
                    tile_count += 1
                    print(f'Preparation: map_tile={tile_count}, points={len(tile)}, time={elapsed:.3f} s, '
                          f'warmup={warm_elapsed:.3f} s (excluded from frame timings)', flush=True)
                    if tile_count == 1:
                        write_json(run/'cuda_runtime.json', {
                            'gpu_model': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
                            'cuda_build': torch.version.cuda, 'kernel_identity': registrar.scene.kernels.identity,
                            'normal_eigh_batch_size': NORMAL_EIGH_BATCH_SIZE, 'cuda_fallback': False})
                frame_started = time.perf_counter()
                points = load_ply_gpu(path, max_points=12000, leaf=.03)
                loaded = time.perf_counter()
                torch.cuda.synchronize()
                observation = registrar.register(points, predicted, float(np.exp(kalman.log_scale)),
                                                 bootstrap=kalman.last_map_ns is None,
                                                 estimate_scale=args.scale_mode == 'estimate')
                update = kalman.update_map(observation, estimate_scale=args.scale_mode == 'estimate')
                pose, state = kalman.publish()
                torch.cuda.synchronize()
                finished = time.perf_counter()
                accepted = bool(update['accepted'])
                load_ms, registration_ms = (loaded-frame_started)*1000, (finished-loaded)*1000
                total_ms = (finished-frame_started)*1000
                q = Rotation.from_matrix(predicted[:3, :3]).as_quat()
                prior_writer.writerow([i, ns, *predicted[:3, 3], *q, float(np.arctan2(predicted[1, 0], predicted[0, 0]))])
                files['result.txt'].write(' '.join(format(x, '.12g') for x in pose[:3].ravel())+'\n')
                csv_writer.writerow([i, format(pose[0, 3], '.12g'), format(pose[1, 3], '.12g'),
                                     format(float(np.arctan2(pose[1, 0], pose[0, 0])), '.12g')])
                timing_writer.writerow([i, original, ns, BACKEND, load_ms, registration_ms, total_ms, accepted, state['scale']])
                observation.update(idx=i, original_idx=original, timestamp_ns=ns,
                                   geometry_accepted=observation['accepted'], accepted=accepted,
                                   filter_update=update, imu=imu_report, state=state,
                                   elapsed_seconds=total_ms/1000, load_ms=load_ms,
                                   registration_ms=registration_ms, total_ms=total_ms)
                files['observations.jsonl'].write(json.dumps(observation, allow_nan=False)+'\n')
                for stream in files.values():
                    stream.flush()
                summary['processed_frames'] += 1
                summary['map_accepted_frames'] += int(accepted)
                summary['tracking_lost_frames'] += int(state['tracking_lost'])
                summary['max_unsupported_seconds'] = max(summary['max_unsupported_seconds'], state['unsupported_seconds'])
                if accepted and not summary['tracking_established']:
                    summary['tracking_established'] = True
                    summary['first_map_accepted_original_idx'] = original
                summary['map_frame_seconds'] += total_ms/1000
                print(f'Frame {i+1}/{len(selected)}: idx={original}, map={"accepted" if accepted else "rejected"}, '
                      f'scale={state["scale"]:.6f}, load={load_ms:.2f} ms, '
                      f'registration={registration_ms:.2f} ms, frame={total_ms:.2f} ms', flush=True)
        finally:
            for stream in files.values():
                stream.close()
        print('Graph: executed=false, applied=false, time=0.00 ms', flush=True)
        errors = validate_saved(run, len(selected), np)
        for source, (_, _, path) in zip(source_rows, selected):
            if digest(path) != source['sha256']:
                errors.append('Source cloud changed: '+path.name)
        for name, identity in identities.items():
            if digest(identity['path']) != identity['sha256']:
                errors.append('Source input changed: '+name)
        for relative, value in hashes.items():
            if digest(ROOT/relative) != value:
                errors.append('Program changed: '+relative)
        if errors:
            raise ValueError('; '.join(errors))
        if not summary['tracking_established']:
            raise RuntimeError('No map observation was accepted: translation tracking never initialized. '
                               'Raw outputs retained for diagnosis; this is not a successful localization run.')
        summary.pop('failed_original_idx', None)
        summary.update(returncode=0, output_complete_and_finite=True,
                       submission_frame_set_complete=bool(args.full), localization_executed=True,
                       csv_sha256=digest(run/'diagnostic.csv'), result_sha256=digest(run/'result.txt'),
                       map_tiles=tile_count,
                       causality_evidence='availability-gated inputs and immutable per-frame serialization; synthetic prefix tests; real prefix validation pending')
    except Exception as error:
        summary['validation_errors'].append(f'{type(error).__name__}: {error}')
        print(f'Failed: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        raise
    finally:
        summary['pipeline_elapsed_seconds'] = time.perf_counter()-started
        if 'torch' in locals() and torch.cuda.is_initialized():
            summary['cuda_memory'] = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                      'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        write_json(run/'validation.json', summary)
    from run_analysis import generate_after_full
    generate_after_full(run, summary, full=args.full)
    accepted = summary['map_accepted_frames']
    print(f'Finished: frames={len(selected)}, accepted_map={accepted}, rejected_map={len(selected)-accepted}, '
          f'total={summary["pipeline_elapsed_seconds"]:.3f} s', flush=True)
    print(f'CSV: {run}/diagnostic.csv', flush=True)
    print(f'Validation: {run}/validation.json', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
