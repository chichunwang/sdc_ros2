#!/usr/bin/env python3
"""Track3: original scored four-DOF ICP, fixed scale, past gyro, no GNSS/RTS."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from causal_track3_filter import ImuCursor, read_initial_pose, proper_pose
from old_icp_runtime import (CarryGyro, active_localizations, build_bridge,
                             initial_ground_tilt, parse_result)
from run_track3_causal import (ROOT, TIMING_HEADER, coordinate_audit, digest,
    export_imu, select_frames, validate_saved, verify_container, verify_front_timestamps, write_json)

METHOD = 'track3_frozen_261019_four_dof_icp_no_gnss_v1'
BACKEND = 'historical_cpp_cuda_robust_icp'
CORE = ROOT/'workspace/src/track3_old_icp'


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--full', action='store_true')
    group.add_argument('--frames', type=int, default=100)
    parser.add_argument('--ply-scale', type=float, default=3., help='fixed uncalibrated PLY multiplier; no scale estimation')
    parser.add_argument('--initial-tilt', choices=('ground', 'official'), default='ground',
                        help='ground: one-time source floor to map-up approximation; official: unchanged init rotation')
    parser.add_argument('--geometry-mode', choices=('directional', 'off'), default='directional')
    parser.add_argument('--data-root', type=Path, default=ROOT.parent/'midterm01/data')
    parser.add_argument('--container', default='sdc_midterm01_gpu')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.frames <= 0 or not np.isfinite(args.ply_scale) or args.ply_scale <= 0:
        parser.error('positive finite ply-scale and positive frames required')
    args.data_root = args.data_root.resolve()
    return args


def snapshot_sources(run):
    paths = [ROOT/'scripts'/name for name in ('run_track3_old_icp.py', 'old_icp_runtime.py',
             'old_icp_bridge.cpp', 'run_track3_causal.py', 'causal_track3_filter.py',
             'export_track3_imu.py', 'three_camera_calibration.py', 'run_analysis.py', 'plot_run_analysis.py')]
    paths += [p for p in CORE.rglob('*') if p.is_file()]
    hashes = {}
    for source in paths:
        relative = source.relative_to(ROOT)
        target = run/'pipeline_sources'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_bytes()
        target.write_bytes(content)
        hashes[str(relative)] = hashlib.sha256(content).hexdigest()
    write_json(run/'pipeline_source_sha256.json', hashes)
    provenance = json.loads((CORE/'provenance.json').read_text())
    for name, expected in provenance['source_sha256'].items():
        if digest(CORE/name) != expected:
            raise ValueError('Frozen historical core modified: '+name)
    return hashes


def main(argv=None):
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    args = arguments(argv)
    selected = select_frames(args)
    initial = read_initial_pose(args.data_root/'Track3/init_guess.txt')
    provenance = json.loads((CORE/'provenance.json').read_text())
    plan = {'method': METHOD, 'track': 3, 'frames': len(selected), 'full': args.full,
        'backend': BACKEND, 'core_origin': provenance, 'core_modified': False,
        'data_root': str(args.data_root), 'ply_scale': args.ply_scale, 'scale_mode': 'fixed',
        'initial_tilt': args.initial_tilt, 'geometry_mode': args.geometry_mode,
        'prior': 'last output XYZ and roll/pitch; past availability-gated vertical gyro increments yaw',
        'gnss_used': False, 'rts_used': False, 'batch_graph_used': False, 'kalman_used': False,
        'new_sim3_optimizer_used': False, 'offline_trajectory_correction': False,
        'old_regularizer': 'unchanged XY 0.5m / Z 1m / yaw 3deg around motion prior; not a GNSS observation',
        'old_parameters': {'map_leaf_m': 1., 'input_leaf_m': .3, 'submap_half_xy_m': 100.,
            'submap_half_z_m': 30., 'max_xy_correction_m': 1.5, 'max_yaw_correction_deg': 8.,
            'smoothing_alpha': .25, 'normal_gain': 2., 'tangent_weight': .05},
        'cpu_components': ['PCL parsing/voxel/crop', 'historical map normal PCA',
                           'historical four-unknown solver', 'gyro integration, first-frame plane fit and output'],
        'cuda_components': ['unchanged historical nearest-neighbor kernel'], 'cuda_fallback': False,
        'imu_extrinsic_status': 'unknown; gravity-vertical planar approximation',
        'replay_supported': False, 'timings_header': TIMING_HEADER,
        'accuracy': 'not evaluated; historical Track1/2 scores do not transfer to Track3'}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    active = active_localizations()
    if active:
        raise RuntimeError(f'Another localization is active (PID {active}); let it finish before this run')
    from accuracy.benchmark import create_run_directory
    run = create_run_directory(ROOT/'outputs/run', METHOD, track=3, full=args.full)
    print(f'Run: {run.resolve()}', flush=True)
    started = time.perf_counter()
    summary = {'track': 3, 'method': METHOD, 'backend': BACKEND, 'frames': len(selected),
        'expected_frames': len(selected), 'processed_frames': 0, 'returncode': 1,
        'output_complete_and_finite': False, 'submission_frame_set_complete': False,
        'validation_errors': [], 'map_accepted_frames': 0, 'tracking_lost_frames': 0,
        'max_unsupported_seconds': 0., 'graph_elapsed_seconds': 0.,
        'localization_executed': False, 'replay_supported': False,
        'submission_compliance_certified': False, 'accuracy': plan['accuracy']}
    process = None
    try:
        write_json(run/'params.json', plan)
        info = json.loads((run/'test_info.json').read_text())
        info.update(algorithm=METHOD, score=None, localization_executed=False)
        write_json(run/'test_info.json', info)
        (run/'command.sh').write_text('#!/bin/sh\n'+shlex.join([sys.executable, str(Path(__file__).resolve()), *actual_argv])+'\n')
        hashes = snapshot_sources(run)
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
        with np.load(run/'imu.npz', allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        write_json(run/'coordinate_audit.json', coordinate_audit(initial, arrays, selected[0][1], selected[0][2], np))
        prepare = time.perf_counter()
        if args.initial_tilt == 'ground':
            from three_camera_calibration import load_ply
            raw = load_ply(selected[0][2], max_points=12000, leaf=.03)
            initial, tilt = initial_ground_tilt(raw, initial, args.ply_scale)
        else:
            tilt = {'mode': 'official', 'rotation_changed': False, 'translation_changed': False}
        write_json(run/'initialization.json', dict(tilt, pose=initial.tolist(),
            data_cutoff_ns=selected[0][1], uses_only_first_cloud=True))
        summary['initialization_seconds'] = time.perf_counter()-prepare
        build = time.perf_counter()
        binary = build_bridge(ROOT, run, args.container)
        summary['build_seconds'] = time.perf_counter()-build
        write_json(run/'binary_sha256.json', {'old_icp_bridge': digest(binary),
            'kernel_library': subprocess.check_output(['docker', 'exec', args.container, 'sha256sum',
                '/root/midterm/install/cuda_ndt_backend/lib/libcuda_ndt_backend.so'], text=True).split()[0]})
        binary_inside = '/root/outputs/'+str(binary.relative_to(ROOT/'outputs'))
        command = ['docker', 'exec', '-i', args.container, binary_inside,
            '--map', '/root/data/Track3/map.pcd', '--scale', str(args.ply_scale), '--geometry', args.geometry_mode]
        write_json(run/'bridge_command.json', command)
        with (run/'bridge.log').open('x') as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                       text=True, bufsize=1)
            ready = process.stdout.readline().split()
            if len(ready) != 4 or ready[0] != 'READY':
                raise RuntimeError('Historical core preparation failed; see bridge.log')
            summary.update(map_points=int(ready[1]), map_preparation_seconds=float(ready[2])/1000,
                           cuda_warmup_seconds=float(ready[3])/1000, localization_executed=True)
            info['localization_executed'] = True; write_json(run/'test_info.json', info)
            print(f'Preparation: map_points={ready[1]}, map={float(ready[2])/1000:.3f} s, '
                  f'warmup={float(ready[3])/1000:.3f} s (excluded from frame timings)', flush=True)
            predictor = CarryGyro(initial, selected[0][1]); cursor = ImuCursor(arrays)
            last_accepted_ns = selected[0][1]
            with (run/'result.txt').open('x') as matrices, (run/'diagnostic.csv').open('x', newline='') as csv_file, \
                    (run/'timings.csv').open('x', newline='') as timing_file, (run/'observations.jsonl').open('x') as observations, \
                    (run/'sensor_prior.csv').open('x', newline='') as priors:
                writer = csv.writer(csv_file); writer.writerow(['idx', 'x', 'y', 'yaw'])
                timing = csv.writer(timing_file); timing.writerow(TIMING_HEADER)
                prior_writer = csv.writer(priors)
                prior_writer.writerow(['idx', 'timestamp_ns', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'yaw'])
                for i, (original, ns, path) in enumerate(selected):
                    summary['failed_original_idx'] = original
                    prior, imu_report = predictor.predict_to(ns, cursor)
                    file_inside = '/root/data/Track3/raw_ply/'+path.name
                    request = 'FRAME '+json.dumps(file_inside)+' '+' '.join(format(x, '.17g') for x in prior.ravel())+'\n'
                    process.stdin.write(request); process.stdin.flush()
                    report = parse_result(process.stdout.readline())
                    pose = np.asarray(report['pose'], float)
                    # The historical float matrices are retained; no projection,
                    # Kalman correction, motion gate or trajectory rewrite.
                    predictor.pose = proper_pose(pose)
                    accepted = report['accepted']
                    if accepted:
                        last_accepted_ns = ns
                    unsupported = (ns-last_accepted_ns)/1e9
                    state = {'scale': args.ply_scale, 'unsupported_seconds': unsupported,
                             'tracking_lost': unsupported > 2.}
                    q = Rotation.from_matrix(prior[:3, :3]).as_quat()
                    prior_writer.writerow([i, ns, *prior[:3, 3], *q, float(np.arctan2(prior[1, 0], prior[0, 0]))])
                    matrices.write(' '.join(format(x, '.12g') for x in pose[:3].ravel())+'\n')
                    writer.writerow([i, format(pose[0, 3], '.12g'), format(pose[1, 3], '.12g'),
                        format(float(np.arctan2(pose[1, 0], pose[0, 0])), '.12g')])
                    timing.writerow([i, original, ns, BACKEND, report['load_ms'], report['registration_ms'],
                                     report['total_ms'], accepted, args.ply_scale])
                    report.update(idx=i, original_idx=original, timestamp_ns=ns, scale=args.ply_scale,
                        imu=imu_report, state=state, elapsed_seconds=report['total_ms']/1000,
                        fit_scope='unchanged historical four-DOF core; motion prior instead of GNSS',
                        accuracy='no ground-truth claim')
                    observations.write(json.dumps(report, allow_nan=False)+'\n')
                    for stream in (matrices, csv_file, timing_file, observations, priors):
                        stream.flush()
                    summary['processed_frames'] += 1
                    summary['map_accepted_frames'] += int(accepted)
                    summary['tracking_lost_frames'] += int(state['tracking_lost'])
                    summary['max_unsupported_seconds'] = max(summary['max_unsupported_seconds'], unsupported)
                    print(f'Frame {i+1}/{len(selected)}: idx={original}, map={"accepted" if accepted else "rejected"}, '
                        f'scale={args.ply_scale:.6f}, load={report["load_ms"]:.2f} ms, '
                        f'registration={report["registration_ms"]:.2f} ms, frame={report["total_ms"]:.2f} ms', flush=True)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError('Historical bridge failed; see bridge.log')
        print('Graph: executed=false, applied=false, time=0.00 ms', flush=True)
        errors = validate_saved(run, len(selected), np)
        for relative, expected in hashes.items():
            if digest(ROOT/relative) != expected:
                errors.append('Program changed: '+relative)
        for name, identity in identities.items():
            if digest(identity['path']) != identity['sha256']:
                errors.append('Input changed: '+name)
        for row, (_, _, path) in zip(source_rows, selected):
            if digest(path) != row['sha256']:
                errors.append('Source cloud changed: '+path.name)
        if errors:
            raise ValueError('; '.join(errors))
        if not summary['map_accepted_frames']:
            raise RuntimeError('No historical ICP map update was accepted; raw outputs retained for diagnosis')
        summary.pop('failed_original_idx', None)
        summary.update(returncode=0, output_complete_and_finite=True,
            submission_frame_set_complete=bool(args.full), csv_sha256=digest(run/'diagnostic.csv'),
            result_sha256=digest(run/'result.txt'))
    except Exception as error:
        summary['validation_errors'].append(f'{type(error).__name__}: {error}')
        print(f'Failed: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        raise
    finally:
        if process is not None and process.poll() is None:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            process.wait()
        summary['pipeline_elapsed_seconds'] = time.perf_counter()-started
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
