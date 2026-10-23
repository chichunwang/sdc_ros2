#!/usr/bin/env python3
"""獨立逐幀 map 尺度配準與慢變軌跡修正；兩條 Track 共用模型參數。"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'map_scale_slow_v1'
TIMING_HEADER = ['idx', 'original_idx', 'timestamp_ns', 'backend', 'load_ms',
                 'registration_ms', 'total_ms', 'map_accepted', 'scale']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', type=int, choices=(1, 2), default=2)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--full', action='store_true')
    group.add_argument('--frames', type=int, default=None)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--data-root', type=Path, default=ROOT.parent/'midterm01/data')
    parser.add_argument('--container', default='sdc_midterm01_gpu')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.full and args.frames is not None:
        parser.error('--full 與 --frames 不可同時指定')
    if args.frames is None:
        args.frames = 48
    if args.frames <= 0 or args.start_frame < 0 or (args.full and args.start_frame != 0):
        parser.error('frames 必須為正數，start-frame 非負，full 只能從 0 開始')
    args.data_root = args.data_root.resolve()
    return args


def select_frames(args):
    frames = []
    for path in (args.data_root/f'Track{args.track}/raw_ply').iterdir():
        match = re.search(r'_ts_(\d+)', path.name)
        if path.is_file() and path.suffix.lower() == '.ply' and match and int(match[1]) > 0:
            frames.append((int(match[1]), path))
    frames.sort(key=lambda item: item[0])
    if len({stamp for stamp, _ in frames}) != len(frames):
        raise ValueError('來源 timestamp 重複')
    expected = {1: 3011, 2: 3299}[args.track]
    if args.full and len(frames) != expected:
        raise ValueError(f'full 來源需 {expected} 幀，實際 {len(frames)}')
    selected = frames if args.full else frames[args.start_frame:args.start_frame + args.frames]
    if not selected or (not args.full and len(selected) != args.frames):
        raise ValueError('指定影格範圍超出來源')
    return selected


def run_docker(args, run, name, script_args):
    command = ['docker', 'exec', args.container, 'bash', '-c',
               'source /opt/ros/humble/setup.bash && exec python3 ' + shlex.join(script_args)]
    write_json(run/(name+'_command.json'), command)
    with (run/(name+'.log')).open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def validate(run, stamps, expected, np):
    errors = []
    with (run/'diagnostic.csv').open() as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['idx', 'x', 'y', 'yaw']:
            errors.append('CSV header must be exactly idx,x,y,yaw')
        rows = list(reader)
    matrices = np.loadtxt(run/'result.txt', ndmin=2)
    frames = list(csv.DictReader((run/'frames.csv').open()))
    if matrices.shape != (len(stamps), 12) or not np.isfinite(matrices).all():
        errors.append('result shape/nonfinite')
    else:
        rotations = matrices.reshape(-1, 3, 4)[:, :, :3]
        if (not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-6)
                or not np.allclose(np.linalg.det(rotations), 1., atol=1e-6)):
            errors.append('rotation is not orthonormal/proper')
    if len(rows) != len(stamps) or [int(r['idx']) for r in rows] != list(range(len(stamps))):
        errors.append('CSV count/idx')
    elif not np.isfinite([[float(r[k]) for k in ('x', 'y', 'yaw')] for r in rows]).all():
        errors.append('CSV nonfinite')
    elif matrices.shape == (len(stamps), 12):
        poses = matrices.reshape(-1, 3, 4)
        if not np.allclose([[float(r[k]) for k in ('x', 'y', 'yaw')] for r in rows],
                           np.column_stack((poses[:, 0, 3], poses[:, 1, 3], np.arctan2(poses[:, 1, 0], poses[:, 0, 0]))), rtol=0, atol=1e-8):
            errors.append('CSV/result disagreement')
    if ([int(r['timestamp_ns']) for r in frames] != stamps
            or [int(r['idx']) for r in frames] != list(range(len(stamps)))):
        errors.append('frame manifest timestamp/idx mismatch')
    if expected is not None and len(stamps) != expected:
        errors.append('full count mismatch')
    return errors


def main():
    args = arguments()
    selected = select_frames(args)
    plan = {'method': METHOD, 'backend': 'scipy_cpu_kdtree', 'track': args.track, 'full': args.full, 'frames': len(selected),
            'start_frame': args.start_frame, 'data_root': str(args.data_root), 'container': args.container,
            'prior': 'raw bag motion_rts, independent registration initialization only',
            'map_leaf_m': .3, 'output_parent': str(ROOT/'outputs/run'),
            'scale_mode': 'per_frame_map_observation', 'replay_supported': False,
            'progress_format': 'frame_timing_v1', 'timings_header': TIMING_HEADER,
            'timing_scope': {
                'load_ms': 'load_ply only',
                'registration_ms': 'register_frame only',
                'total_ms': 'load start through registration completion; excludes serialization, graph and preprocessing',
                'graph_elapsed_seconds': 'smooth_trajectory only',
                'map_frame_seconds': 'sum of per-frame total_ms / 1000',
                'pipeline_elapsed_seconds': 'entire run after directory creation through validation'}}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    from accuracy.benchmark import create_run_directory
    run = create_run_directory(ROOT/'outputs/run', METHOD, track=args.track, full=args.full)
    started = time.perf_counter()
    summary = {'track': args.track, 'frames': len(selected), 'returncode': 1,
               'output_complete_and_finite': False, 'submission_frame_set_complete': False,
               'replay_supported': False,
               'validation_errors': [], 'map_accepted_frames': 0, 'graph_report': None,
               'graph_elapsed_seconds': 0., 'map_frame_seconds': 0.}
    print(f'Run: {run.resolve()}', flush=True)
    try:
        import numpy as np
        import scipy
        from scipy.spatial.transform import Rotation
        from accuracy.horizontal_cache import prepare_fine_map
        from three_camera_calibration import load_binary_map, load_ply
        from map_scale_icp import ScaleICPConfig, LayeredScaleMap, register_frame
        from slow_map_graph import SlowMapConfig, smooth_trajectory
        config = ScaleICPConfig()
        graph_config = SlowMapConfig()
        write_json(run/'params.json', dict(plan, model=asdict(config), slow_map_model=asdict(graph_config)))
        info = json.loads((run/'test_info.json').read_text())
        info.update(algorithm=METHOD, score=None)
        write_json(run/'test_info.json', info)
        (run/'command.sh').write_text('#!/bin/sh\n'+shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])+'\n')
        repo = ROOT.parent
        for filename, command in [('git_commit.txt', ['rev-parse', 'HEAD']),
                                  ('git_status.txt', ['status', '--short']), ('worktree.patch', ['diff', '--binary', 'HEAD'])]:
            result = subprocess.run(['git', '-C', str(repo), *command], capture_output=True, check=True)
            (run/filename).write_bytes(result.stdout)
        # Snapshot Python execution dependencies, including newly added/untracked modules.
        sources = sorted(p for p in (ROOT/'scripts').rglob('*.py') if '__pycache__' not in p.parts)
        sources.extend(ROOT/path for path in ('workspace/src/localization/src/map_modified.cpp',
                                               'workspace/src/localization/CMakeLists.txt'))
        required_sources = ('run_map_smoother.py', 'map_scale_icp.py', 'slow_map_graph.py',
                            'kinematic_graph.py', 'horizontal_surfel.py', 'run_analysis.py',
                            'plot_run_analysis.py')
        if any(not (ROOT/'scripts'/name).is_file() for name in required_sources):
            raise ValueError('Required execution source missing from snapshot')
        hashes = {}
        for path in sources:
            relative = path.relative_to(ROOT)
            target = run/'pipeline_sources'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            content = path.read_bytes()
            target.write_bytes(content)
            hashes[str(relative)] = hashlib.sha256(content).hexdigest()
        write_json(run/'pipeline_source_sha256.json', hashes)
        write_json(run/'runtime.json', {'python': sys.version, 'executable': sys.executable,
                                       'numpy': np.__version__, 'scipy': scipy.__version__})
        source_rows = [{'idx': i, 'original_idx': args.start_frame+i, 'timestamp_ns': ns,
                        'filename': path.name, 'sha256': digest(path)} for i, (ns, path) in enumerate(selected)]
        write_json(run/'source_input.json', {'track': args.track, 'frames': source_rows})
        with (run/'frames.csv').open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['idx', 'timestamp_ns', 'filename'])
            writer.writerows((r['idx'], r['timestamp_ns'], r['filename']) for r in source_rows)
        bag = args.data_root/f'Track{args.track}/Track{args.track}.db3'
        raw_map = args.data_root/f'Track{args.track}/map.pcd'
        identities = {'bag': {'path': str(bag), 'sha256': digest(bag)},
                      'map': {'path': str(raw_map), 'sha256': digest(raw_map)}}
        write_json(run/'input_sha256.json', identities)
        crun = '/root/outputs/'+run.relative_to(ROOT/'outputs').as_posix()
        inspected = subprocess.run(['docker', 'inspect', args.container], capture_output=True, text=True, check=True)
        mounts = {m['Destination']: Path(m['Source']).resolve()
                  for m in json.loads(inspected.stdout)[0]['Mounts']}
        required_mounts = {'/root/data': args.data_root, '/root/outputs': (ROOT/'outputs').resolve(),
                           '/root/scripts': (ROOT/'scripts').resolve(),
                           '/root/midterm': (ROOT/'workspace').resolve()}
        if any(mounts.get(target) != source for target, source in required_mounts.items()):
            raise ValueError('容器 data/outputs/scripts 掛載與本輪主機來源不一致')
        write_json(run/'container_mounts.json', {key: str(value) for key, value in mounts.items()})
        map_binary = ROOT/'workspace/build/localization/map_modified'
        binary_command = ['docker', 'exec', args.container, 'sha256sum',
                          '/root/midterm/install/localization/lib/localization/map_modified']
        write_json(run/'map_modifier_identity_command.json', binary_command)
        binary_result = subprocess.run(binary_command, capture_output=True, text=True, check=True)
        container_binary_sha = binary_result.stdout.split()[0]
        if not re.fullmatch(r'[0-9a-f]{64}', container_binary_sha) or digest(map_binary) != container_binary_sha:
            raise ValueError('map_modified container/build binary SHA mismatch')
        write_json(run/'map_modifier_identity.json', {
            'container': args.container, 'container_entrypoint': binary_command[-1],
            'binary_sha256': container_binary_sha, 'host_binary_path': str(map_binary),
            'source_sha256': {name: hashes[name] for name in hashes
                              if name in ('workspace/src/localization/src/map_modified.cpp',
                                          'workspace/src/localization/CMakeLists.txt')},
            'source_snapshot': 'pipeline_sources/workspace/src/localization',
            'command': binary_command})
        cbag = f'/root/data/Track{args.track}/Track{args.track}.db3'
        run_docker(args, run, 'sensor_preparation', ['/root/scripts/prepare_sensor_prior.py', '--bag', cbag,
                   '--frames', crun+'/frames.csv', '--output', crun+'/sensor_prior.csv', '--fusion', 'motion_rts'])
        run_docker(args, run, 'raw_sensor_export', ['/root/scripts/audit_input_heading.py', 'export',
                   '--input', cbag, '--output', crun+'/raw_sensors.npz'])
        prior_rows = list(csv.DictReader((run/'sensor_prior.csv').open()))
        stamps = [ns for ns, _ in selected]
        if [int(r['timestamp_ns']) for r in prior_rows] != stamps:
            raise ValueError('prior timestamp/count mismatch')
        xyz = np.array([[float(r[k]) for k in ('x', 'y', 'z')] for r in prior_rows])
        quat = np.array([[float(r[k]) for k in ('qx', 'qy', 'qz', 'qw')] for r in prior_rows])
        if not np.isfinite(xyz).all() or not np.isfinite(quat).all() or not np.allclose(np.linalg.norm(quat, axis=1), 1., atol=1e-5):
            raise ValueError('prior nonfinite/invalid quaternion')
        priors = np.tile(np.eye(4), (len(stamps), 1, 1))
        priors[:, :3, :3] = Rotation.from_quat(quat).as_matrix()
        priors[:, :3, 3] = xyz
        bounds = {'min': (xyz.min(axis=0)-[65, 65, 25]).tolist(),
                  'max': (xyz.max(axis=0)+[65, 65, 25]).tolist()}
        fine = prepare_fine_map(run/'fine_map', args.data_root, args.track, args.container, bounds, leaf=.3)
        scene = LayeredScaleMap(load_binary_map(fine), config)
        observations = []
        with (run/'observations.jsonl').open('x') as stream, (run/'timings.csv').open('x', newline='') as timing_stream:
            timing_writer = csv.writer(timing_stream)
            timing_writer.writerow(TIMING_HEADER)
            timing_stream.flush()
            for i, (_, path) in enumerate(selected):
                frame_started = time.perf_counter()
                points = load_ply(path, max_points=12000, leaf=.03)
                load_finished = time.perf_counter()
                registration_started = time.perf_counter()
                observation = register_frame(scene, points, priors[i], config)
                frame_finished = time.perf_counter()
                load_ms = (load_finished-frame_started)*1000
                registration_ms = (frame_finished-registration_started)*1000
                total_ms = (frame_finished-frame_started)*1000
                observation.update(idx=i, original_idx=args.start_frame+i, timestamp_ns=selected[i][0],
                                   elapsed_seconds=frame_finished-frame_started,
                                   load_ms=load_ms, registration_ms=registration_ms, total_ms=total_ms)
                observations.append(observation)
                summary['map_frame_seconds'] += total_ms/1000
                timing_writer.writerow([i, args.start_frame+i, selected[i][0], 'scipy_cpu_kdtree',
                                        load_ms, registration_ms, total_ms,
                                        bool(observation['accepted']), observation['scale']])
                timing_stream.flush()
                stream.write(json.dumps(observation, allow_nan=False)+'\n')
                stream.flush()
                status = 'accepted' if observation['accepted'] else 'rejected'
                print(f'Frame {i+1}/{len(stamps)}: idx={args.start_frame+i}, map={status}, '
                      f'scale={observation["scale"]:.6f}, load={load_ms:.2f} ms, '
                      f'registration={registration_ms:.2f} ms, frame={total_ms:.2f} ms', flush=True)
        with np.load(run/'raw_sensors.npz', allow_pickle=False) as raw:
            raw_dict = {key: raw[key] for key in raw.files}
        graph_stamps = np.asarray(stamps, dtype=np.int64)
        graph_started = time.perf_counter()
        poses, report, states = smooth_trajectory(raw_dict, graph_stamps, priors, observations, config=graph_config)
        summary['graph_elapsed_seconds'] = time.perf_counter()-graph_started
        write_json(run/'graph_report.json', report)
        executed = bool(report['joint_estimation_executed'])
        summary['joint_estimation_executed'] = executed
        summary['initialization_source'] = 'current run raw-bag motion_rts sensor_prior.csv'
        print(f'Graph: executed={str(executed).lower()}, applied={str(bool(report["applied"])).lower()}, '
              f'time={summary["graph_elapsed_seconds"]*1000:.2f} ms', flush=True)
        if not executed:
            print('無受支持慢變 map anchors：保留本輪 motion_rts 感測初始化；'
                  '此輸出未與已評分 C++ 定位結果作 bytes 比較。', flush=True)
        print(f'graph 原因: {report.get("reason", "unspecified")}', flush=True)
        np.savez_compressed(run/'graph_states.npz', **states)
        poses = np.asarray(poses, dtype=float)
        if poses.shape != priors.shape or not np.isfinite(poses).all() or not np.allclose(poses[:, 3, :], [0, 0, 0, 1]):
            raise ValueError('graph output shape/nonfinite/homogeneous row')
        np.savetxt(run/'result.txt', poses[:, :3, :].reshape(-1, 12), fmt='%.12g')
        with (run/'diagnostic.csv').open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['idx', 'x', 'y', 'yaw'])
            writer.writerows((i, format(p[0, 3], '.12g'), format(p[1, 3], '.12g'),
                              format(np.arctan2(p[1, 0], p[0, 0]), '.12g')) for i, p in enumerate(poses))
        for key, path in [('prior', run/'sensor_prior.csv'), ('raw_sensors', run/'raw_sensors.npz'), ('fine_map', fine)]:
            identities[key] = {'path': str(path), 'sha256': digest(path)}
        write_json(run/'input_sha256.json', identities)
        errors = validate(run, stamps, {1: 3011, 2: 3299}[args.track] if args.full else None, np)
        for row, (_, path) in zip(source_rows, selected):
            if digest(path) != row['sha256']:
                errors.append(f'frame changed during execution: {path.name}')
        for relative, value in hashes.items():
            if digest(ROOT/relative) != value:
                errors.append(f'program changed during execution: {relative}')
        final_binary = subprocess.run(binary_command, capture_output=True, text=True, check=True)
        if final_binary.stdout.split()[0] != container_binary_sha or digest(map_binary) != container_binary_sha:
            errors.append('map_modified binary changed during execution')
        summary.update(validation_errors=errors, map_accepted_frames=sum(bool(r['accepted']) for r in observations),
                       graph_report=report, csv_sha256=digest(run/'diagnostic.csv'), result_sha256=digest(run/'result.txt'))
        if errors:
            raise ValueError('; '.join(errors))
        summary.update(returncode=0, output_complete_and_finite=True,
                       submission_frame_set_complete=bool(args.full))
    except Exception as error:
        summary['validation_errors'].append(f'{type(error).__name__}: {error}')
        raise
    finally:
        summary['pipeline_elapsed_seconds'] = time.perf_counter()-started
        write_json(run/'validation.json', summary)
    from run_analysis import generate_after_full
    generate_after_full(run, summary, full=args.full)
    accepted = summary['map_accepted_frames']
    print(f'Finished: frames={len(selected)}, accepted_map={accepted}, '
          f'rejected_map={len(selected)-accepted}, total={summary["pipeline_elapsed_seconds"]:.3f} s', flush=True)
    print(f'CSV: {run}/diagnostic.csv', flush=True)
    print(f'Validation: {run}/validation.json', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
