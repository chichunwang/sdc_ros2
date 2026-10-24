#!/usr/bin/env python3
"""獨立 C++/CUDA Track3 launcher；尺度 seed 未標定，精度未知。"""
import argparse
import ast
import csv
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

GPUROOT = Path(__file__).resolve().parents[1]
SOURCE = GPUROOT / 'workspace/src/track3_rewrite'
METHOD = 'track3_cpp_cuda_observable_icp_v1'
TIMINGS = 'idx,original_idx,timestamp_ns,backend,load_ms,registration_ms,total_ms,map_accepted,scale'.split(',')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('須為有限正數')
    return result


def frame_count(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('frames 須為正整數')
    return result


def inventory(data, count, full):
    frames = []
    for p in (data / 'Track3/raw_ply').glob('*.ply'):
        match = re.fullmatch(r'(\d+)_ts_(\d+)\.ply', p.name)
        if not match:
            raise ValueError(f'非法 frame 檔名：{p.name}')
        idx, stamp = map(int, match.groups())
        if stamp > 2**63 - 1:
            raise ValueError('timestamp 超出 int64')
        frames.append((idx, stamp, p.resolve()))
    frames.sort()
    if [f[0] for f in frames] != list(range(len(frames))):
        raise ValueError('frame 索引必須完整且不重複，自 0 開始')
    if not frames or frames[0][1] <= 0 or any(a[1] >= b[1] for a, b in zip(frames, frames[1:])):
        raise ValueError('timestamp 必須嚴格遞增')
    if full and len(frames) != 3142:
        raise ValueError('full 必須有 3142 幀')
    if len(frames) < count:
        raise ValueError('frame 數量不足')
    rows = [line.replace(',', ' ').split() for line in (data / 'Track3/init_guess.txt').read_text().splitlines() if line.strip()]
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError('init_guess 必須為 4x4')
    if not all(math.isfinite(float(x)) for row in rows for x in row):
        raise ValueError('init_guess 含非有限值')
    return frames[:count]


def refuse_running():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = proc.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        names = [Path(x).name for x in argv if x]
        native = names and names[0] in ('localization', 'old_icp_bridge', 'track3_rewrite')
        python = names and ('python' in names[0]) and any(re.fullmatch(r'run_track3.*\.py|run_map_smoother.*\.py|benchmark\.py|test_track3_map_registration\.py', n) for n in names[1:])
        python = python or (names and 'python' in names[0] and any(Path(a).resolve() == Path(__file__).resolve() for a in argv[1:] if a))
        if native or python:
            raise RuntimeError(f'偵測到定位程序 PID {proc.name}；拒絕並行，不停止該程序')


def verify_container(args):
    info = json.loads(subprocess.check_output(['docker', 'inspect', args.container], text=True))[0]
    if not info['State']['Running']:
        raise ValueError('指定容器未 running；請手動啟動')
    mounts = {m['Destination']: m for m in info['Mounts']}
    for dest, host in [('/root/data', args.data_root), ('/root/midterm', GPUROOT / 'workspace'), ('/root/outputs', GPUROOT / 'outputs')]:
        mount = mounts.get(dest, {})
        if mount.get('Type') != 'bind' or Path(mount.get('Source', '/')).resolve() != host.resolve():
            raise ValueError(f'容器 bind mount 不符：{dest}')
        if dest == '/root/data' and mount.get('RW') is not False:
            raise ValueError('/root/data 必須唯讀')


def map_identity(cache, data):
    cache = cache.resolve()
    cache.relative_to((GPUROOT / 'outputs/run').resolve())
    metadata_path = cache.parent / 'map_xyz.json'
    meta = json.loads(metadata_path.read_text())
    if any(meta.get(key) is not False for key in ('voxel_applied', 'crop_applied', 'pose_processing_applied')):
        raise ValueError('map cache 必須無 voxel/crop/pose_processing')
    if meta.get('dtype') != 'float32' or not isinstance(meta.get('shape'), list) or len(meta['shape']) != 2 or meta['shape'][1] != 3 or meta['shape'][0] <= 0 or meta.get('points') != meta['shape'][0]:
        raise ValueError('map metadata 必須為非空 float32 Nx3')
    identities = {str(data / 'Track3/map.pcd'): sha(data / 'Track3/map.pcd'), str(cache): sha(cache), str(metadata_path): sha(metadata_path)}
    if identities[str(data / 'Track3/map.pcd')] != meta.get('source_sha256') or identities[str(cache)] != meta.get('cache_sha256'):
        raise ValueError('map cache 來源或 payload SHA 不符')
    with cache.open('rb') as f:
        if f.read(6) != b'\x93NUMPY':
            raise ValueError('cache 不是 NPY')
        version = f.read(2)
        if version != b'\x01\x00':
            raise ValueError('native map loader supports NPY v1 only')
        size = struct.unpack('<H', f.read(2))[0]
        header = ast.literal_eval(f.read(size).decode('latin1').strip())
        if header['descr'] not in ('<f4', '=f4') or header['fortran_order'] or list(header['shape']) != meta['shape']:
            raise ValueError('NPY header 與 metadata 不符')
        if cache.stat().st_size - f.tell() != meta['shape'][0] * 12:
            raise ValueError('NPY payload 大小不符')
        # Native loader checks finite XYZ while streaming to CUDA. Do not scan
        # 172 million values through a second Python scalar decoder.
    return cache, identities


def extract_map(source, directory):
    """Independently stream original ASCII PCD XYZ to an unprocessed NPY.

    NumPy is used only for bulk ASCII parsing here, never as an old solver or
    tracker dependency. This path makes historical run caches optional.
    """
    import numpy as np
    directory.mkdir()
    before = sha(source)
    partial, output = directory/'map_xyz.partial.npy', directory/'map_xyz.npy'
    with source.open('r', encoding='ascii') as stream:
        header = {}
        for _ in range(64):
            line = stream.readline()
            if not line:
                raise ValueError('Truncated PCD header')
            words = line.split()
            if not words or words[0].startswith('#'):
                continue
            if words[0] in header:
                raise ValueError('Duplicate PCD header field')
            header[words[0]] = words[1:]
            if words[0] == 'DATA':
                break
        if header.get('DATA') != ['ascii']:
            raise ValueError('Original map parser requires ASCII PCD')
        fields = header['FIELDS'];counts = list(map(int, header['COUNT']))
        if len(fields) != len(set(fields)) or any(len(header[k]) != len(fields) for k in ('SIZE', 'TYPE', 'COUNT')) or any(n <= 0 for n in counts):
            raise ValueError('Invalid PCD schema')
        columns = []
        for axis in 'xyz':
            index = fields.index(axis)
            if (header['SIZE'][index], header['TYPE'][index], counts[index]) != ('4', 'F', 1):
                raise ValueError('Expected original float32 XYZ')
            columns.append(sum(counts[:index]))
        count = int(header['POINTS'][0])
        if count <= 0 or count != int(header['WIDTH'][0])*int(header['HEIGHT'][0]):
            raise ValueError('PCD point count differs from dimensions')
        with partial.open('xb'):
            pass
        mapped = np.lib.format.open_memmap(partial, mode='w+', dtype='<f4', shape=(count, 3))
        try:
            for start in range(0, count, 100000):
                n = min(100000, count-start)
                lines = list(itertools.islice(stream, n))
                if len(lines) != n:
                    raise ValueError('Truncated PCD payload')
                rows = np.loadtxt(io.StringIO(''.join(lines)), dtype=np.float64, ndmin=2)
                if rows.shape != (n, sum(counts)):
                    raise ValueError('Invalid PCD payload columns')
                xyz = rows[:, columns].astype('<f4')
                if not np.isfinite(xyz).all():
                    raise ValueError('Nonfinite map XYZ')
                mapped[start:start+n] = xyz
            if any(line.strip() for line in stream):
                raise ValueError('Extra PCD payload')
            mapped.flush()
        finally:
            del mapped
    if sha(source) != before:
        raise ValueError('Original map changed during extraction')
    partial.rename(output)
    write_json(directory/'map_xyz.json', dict(source=str(source), source_sha256=before,
        cache_sha256=sha(output), shape=[count, 3], points=count, dtype='float32',
        voxel_applied=False, crop_applied=False, pose_processing_applied=False))
    return output


def unique_run(full):
    root = GPUROOT / 'outputs/run'
    root.mkdir(parents=True, exist_ok=True)
    name = 'track03_' + ('full_' if full else 'test_') + datetime.now(ZoneInfo('Asia/Taipei')).strftime('%y%m%d%H%M')
    number = 1
    while True:
        path = root / (name if number == 1 else f'{name}_{number:04d}')
        try:
            path.mkdir()
            return path
        except FileExistsError:
            number += 1


def logged(command, logfile, live=False, interrupt_command=None):
    with logfile.open('w') as log:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        previous = signal.getsignal(signal.SIGINT)
        pending_interrupt = False
        def forward_pending():
            nonlocal pending_interrupt
            if not pending_interrupt:
                return
            try:
                if interrupt_command is not None:
                    interrupt_command()
                else:
                    child.send_signal(signal.SIGINT)
                pending_interrupt = False
            except (OSError, ValueError, subprocess.CalledProcessError):
                # Native startup may not have written its PID yet. Retry when
                # the first output arrives, while continuing to drain stdout.
                pass
        def interrupt(signum, frame):
            nonlocal pending_interrupt
            pending_interrupt = True
            forward_pending()
        signal.signal(signal.SIGINT, interrupt)
        try:
            for line in child.stdout:
                forward_pending()
                log.write(line)
                log.flush()
                if live:
                    print(line, end='', flush=True)
            return child.wait()
        except BaseException:
            interrupt(signal.SIGINT, None)
            child.wait()
            raise
        finally:
            signal.signal(signal.SIGINT, previous)
            child.stdout.close()


def analyze_full(run):
    started = time.perf_counter()
    status = dict(status='failed', localization_success=True, trajectory_modified=False,
                  time_scope='analysis only; excluded from localization timings')
    try:
        code = logged([sys.executable, str(GPUROOT/'scripts/plot_run_analysis.py'), '--run', str(run)], run/'analysis.log')
        status['returncode'] = code
        if code or json.loads((run/'analysis/manifest.json').read_text()).get('complete') is not True:
            raise RuntimeError('Plot process or manifest incomplete')
        status['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as error:
        status['error'] = str(error) or 'KeyboardInterrupt during analysis'
    status['elapsed_seconds'] = time.perf_counter()-started
    try:
        write_json(run/'analysis_status.json', status)
    except OSError as error:
        print(f'Analysis status write failed: {error}; localization results preserved.', file=sys.stderr)


def read_csv(path, fields, count):
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fields:
            raise ValueError(f'{path.name} 欄位不符')
        rows = list(reader)
    if len(rows) != count or [int(r['idx']) for r in rows] != list(range(count)):
        raise ValueError(f'{path.name} 筆數或索引不符')
    return rows


def validate(run, frames, identities):
    count = len(frames)
    status = json.loads((run / 'native_status.json').read_text())
    if status['processed_frames'] != count or status['returncode'] != 0:
        raise ValueError('native status 未完整成功')
    accepted = int(status['map_accepted_frames'])
    pair = int(status['pair_accepted_frames'])
    if not 0 <= accepted <= count or not 0 <= pair <= max(0, count - 1):
        raise ValueError('native counters 不合法')
    matrices = [[float(x) for x in line.split()] for line in (run / 'result.txt').read_text().splitlines()]
    if len(matrices) != count or any(len(m) != 12 or not all(map(math.isfinite, m)) for m in matrices):
        raise ValueError('result.txt 筆數、矩陣或有限值不符')
    poses = read_csv(run / 'diagnostic.csv', ['idx', 'x', 'y', 'yaw'], count)
    timing = read_csv(run / 'timings.csv', TIMINGS, count)
    manifest = read_csv(run / 'frames.csv', ['idx', 'timestamp_ns', 'filename'], count)
    observations = [json.loads(line) for line in (run / 'observations.jsonl').read_text().splitlines()]
    source_records = [json.loads(line) for line in (run / 'source_frames.jsonl').read_text().splitlines()]
    def finite_json(value):
        if isinstance(value, float) and not math.isfinite(value):
            return False
        if isinstance(value, dict):
            return all(finite_json(v) for v in value.values())
        if isinstance(value, list):
            return all(finite_json(v) for v in value)
        return True
    if not finite_json(observations):
        raise ValueError('observations 含非有限值')
    if len(observations) != count or len(source_records) != count:
        raise ValueError('observations 筆數不符')
    for i, (m, pose, t, f, original) in enumerate(zip(matrices, poses, timing, manifest, frames)):
        for key, expected in [('x', m[3]), ('y', m[7]), ('yaw', math.atan2(m[4], m[0]))]:
            value = float(pose[key])
            delta = math.atan2(math.sin(value - expected), math.cos(value - expected)) if key == 'yaw' else value - expected
            if not math.isfinite(value) or abs(delta) > 1e-6:
                raise ValueError(f'CSV 與矩陣不符 frame {i}')
        if int(t['original_idx']) != original[0] or int(t['timestamp_ns']) != original[1] or int(f['timestamp_ns']) != original[1] or Path(f['filename']).name != original[2].name:
            raise ValueError(f'frame/time 契約不符 frame {i}')
        for key in ('load_ms', 'registration_ms', 'total_ms', 'scale'):
            value = float(t[key])
            if not math.isfinite(value) or value < 0 or (key == 'scale' and value == 0):
                raise ValueError('timings 非有限或不合法')
        o = observations[i]
        if (o['idx'] != i or o['original_idx'] != original[0] or o['timestamp_ns'] != original[1]
                or o['input_cutoff_ns'] != original[1] or o['imu_used'] is not False or o['gnss_used'] is not False):
            raise ValueError('observation current-frame/causality contract mismatch')
        if not math.isfinite(float(o['elapsed_seconds'])) or float(o['elapsed_seconds']) < 0:
            raise ValueError('observation elapsed_seconds 非有限')
        rotation = [[m[4*a+b] for b in range(3)] for a in range(3)]
        if any(abs(sum(rotation[k][a]*rotation[k][b] for k in range(3))-(1 if a==b else 0)) > 1e-7 for a in range(3) for b in range(3)):
            raise ValueError('matrix rotation is not orthonormal')
        r = rotation
        det = r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])-r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])+r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0])
        if abs(det-1.) > 1e-7:
            raise ValueError('matrix has improper rotation')
        if type(o['accepted']) is not bool or t['map_accepted'].lower() != str(o['accepted']).lower() or abs(float(t['scale'])-o['scale']) > 1e-10:
            raise ValueError('observation/timing mismatch')
        source = source_records[i]
        if (source['idx'] != i or source['timestamp_ns'] != original[1] or source['filename'] != original[2].name
                or sha(original[2]) != source['sha256']):
            raise ValueError('Current-frame source SHA contract mismatch')
    if sum(t['map_accepted'].lower() in ('1', 'true') for t in timing) != accepted:
        raise ValueError('map acceptance counter 不符')
    for filename, digest in identities.items():
        if sha(filename) != digest:
            raise ValueError(f'執行期間來源變更：{filename}')
    if not math.isfinite(float(status['total_seconds'])) or status['total_seconds'] < 0:
        raise ValueError('native total_seconds 不合法')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--full', action='store_true')
    group.add_argument('--frames', type=frame_count, default=32)
    parser.add_argument('--data-root', type=Path, default=GPUROOT.parent / 'midterm01/data')
    parser.add_argument('--map-cache', type=Path)
    parser.add_argument('--initial-scale', type=positive, default=3.0, help='未標定的正尺度 seed')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--build-only', action='store_true')
    mode.add_argument('--check', action='store_true', help='僅 build 與 native 合成 CTest')
    parser.add_argument('--container', default='sdc_midterm01_gpu')
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    count = 3142 if args.full else args.frames
    run = None
    executed = False
    rc = None
    validation = {'track': 3, 'frames': count, 'method': METHOD, 'expected_frames': count, 'processed_frames': 0, 'localization_executed': False, 'output_complete_and_finite': False, 'submission_frame_set_complete': False, 'validation_errors': [], 'returncode': None, 'accuracy': 'unknown', 'imu_used': False, 'gnss_used': False, 'submission_compliance_certified': False, 'replay_supported': False}
    try:
        frames = [] if args.check or args.build_only else inventory(args.data_root, count, args.full)
        if args.dry_run:
            print(json.dumps({'method': METHOD, 'frames': len(frames), 'initial_scale': args.initial_scale, 'initial_scale_calibrated': False, 'map_cache_payload_verified': False, 'cuda_initialized': False, 'accuracy': 'unknown', 'localization_executed': False}, ensure_ascii=False, indent=2))
            return 0
        refuse_running()
        verify_container(args)
        run = unique_run(args.full and not (args.check or args.build_only))
        print(f'Run: {run}', flush=True)
        params = vars(args).copy()
        params = {key: str(value) if isinstance(value, Path) else value for key, value in params.items()}
        params.update(inputs=['frontPLY', 'mapXYZ', 'init_guess'], imu=False, gnss=False, external_predictions=False, future_information=False, initial_scale_calibrated=False)
        params.update(method=METHOD, source_leaf_raw=.1, map_leaf_m=1., map_crop_half_xy_m=100., map_crop_half_z_m=30.,
                      pair_radii_m=[4.,2.,1.], bootstrap_radii_m=[8.,4.,2.,1.], map_radii_m=[4.,2.,1.],
                      pair_sim3=True, map_scale='from pair after bootstrap', map_metric='point_to_plane_observed_directions',
                      point_sim3_scale_domain_factors=[.5, 2.], point_sim3_required_rank=7, plane_map_allows_partial_rank=True,
                      cpu_fallback=False, rts_used=False, offline_trajectory_correction=False, replay_supported=False)
        write_json(run / 'params.json', params)
        info = {'method': METHOD, 'score': None, 'accuracy': 'unknown', 'params': params}
        write_json(run / 'test_info.json', info)
        (run / 'command.sh').write_text('#!/bin/sh\n' + shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]) + '\n')
        (run / 'git_commit.txt').write_text(subprocess.check_output(['git', '-C', str(GPUROOT), 'rev-parse', 'HEAD'], text=True))
        (run / 'worktree.patch').write_bytes(subprocess.check_output(['git', '-C', str(GPUROOT), 'diff', '--binary', 'HEAD', '--', '.']))
        sources = [Path(__file__).resolve(), *sorted(p for p in SOURCE.rglob('*') if p.is_file())]
        identities = {str(p): sha(p) for p in sources}
        for source in sources:
            target = run / 'source_snapshot' / source.relative_to(GPUROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha(target) != identities[str(source)]:
                raise ValueError('來源在快照期間變更')
        write_json(run / 'source_sha256.json', {str(p.relative_to(GPUROOT)): identities[str(p)] for p in sources})
        cpu = GPUROOT.parent / 'midterm01/workspace/src/localization'
        write_json(run / 'cpu_template_source_sha256.json', {str(p.relative_to(GPUROOT.parent)): sha(p) for p in sorted(cpu.rglob('*')) if p.is_file()})
        (run / 'frames.tsv').write_text(''.join(f'{idx}\t{stamp}\t/root/data/Track3/raw_ply/{path.name}\n' for idx, stamp, path in frames))
        cache = None
        if not (args.check or args.build_only):
            requested_cache = args.map_cache or extract_map(args.data_root/'Track3/map.pcd', run/'map_cache')
            cache, cache_ids = map_identity(requested_cache, args.data_root)
            identities.update(cache_ids)
            # Future PLY payloads are never read before their native frame.
            # The native app records the SHA of each current input at use time.
            identities[str(args.data_root / 'Track3/init_guess.txt')] = sha(args.data_root / 'Track3/init_guess.txt')
        write_json(run / 'input_source_sha256.json', identities)
        container_run = '/root/outputs/run/' + run.name
        build = container_run + '/native_build'
        build_command = f'source /opt/ros/humble/setup.bash && cmake -S /root/midterm/src/track3_rewrite -B {shlex.quote(build)} -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build {shlex.quote(build)} -j2'
        rc = logged(['docker', 'exec', args.container, 'bash', '-c', build_command], run / 'build.log')
        if rc:
            raise RuntimeError(f'native build 失敗：{rc}')
        if args.check:
            rc = logged(['docker', 'exec', args.container, 'ctest', '--test-dir', build, '--output-on-failure'], run / 'mechanisms.log')
            if rc:
                raise RuntimeError(f'native 合成 CTest 失敗：{rc}')
        if args.check or args.build_only:
            validation.update(returncode=0, build_success=True, mechanisms_checked=args.check)
            write_json(run / 'validation.json', validation)
            return 0
        refuse_running()
        for filename, digest in identities.items():
            if sha(filename) != digest:
                raise ValueError(f'準備期間來源變更：{filename}')
        container_cache = '/root/outputs/' + str(cache.relative_to(GPUROOT / 'outputs'))
        executed = True
        def stop_this_native():
            pid = int((run / 'native_pid.txt').read_text().strip())
            script = 'import os,signal,sys; p=int(sys.argv[1]); a=open(f"/proc/{p}/cmdline","rb").read().split(b"\\0"); expected=sys.argv[2].encode(); out=sys.argv[3].encode(); assert a[0]==expected and out in a; os.kill(p,signal.SIGINT)'
            subprocess.run(['docker', 'exec', args.container, 'python3', '-c', script, str(pid), build + '/track3_rewrite', container_run], check=True)
        rc = logged(['docker', 'exec', '-i', args.container, build + '/track3_rewrite', '--map', container_cache, '--frames', container_run + '/frames.tsv', '--init', '/root/data/Track3/init_guess.txt', '--out', container_run, '--initial-scale', str(args.initial_scale)], run / 'native.log', live=True, interrupt_command=stop_this_native)
        if rc:
            raise RuntimeError(f'native 定位失敗或中斷：{rc}')
        status = validate(run, frames, identities)
        validation.update(returncode=0, localization_executed=True, processed_frames=count, map_accepted_frames=status['map_accepted_frames'], pair_accepted_frames=status['pair_accepted_frames'], tracking_lost_frames=status.get('tracking_lost_frames'), output_complete_and_finite=True, submission_frame_set_complete=args.full)
        write_json(run / 'validation.json', validation)
        print('Graph: executed=false, applied=false, time=0.00 ms', flush=True)
        print(f"Finished: frames={count}, accepted_map={status['map_accepted_frames']}, rejected_map={count - status['map_accepted_frames']}, total={status['total_seconds']:.3f} s", flush=True)
        print(f'CSV: {run / "diagnostic.csv"}\nValidation: {run / "validation.json"}', flush=True)
        if args.full:
            analyze_full(run)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        validation.update(returncode=rc if rc else 1, localization_executed=executed)
        validation['validation_errors'].append(str(exc) or 'KeyboardInterrupt')
        if run:
            try:
                status = json.loads((run / 'native_status.json').read_text())
                for key in ('processed_frames', 'map_accepted_frames', 'pair_accepted_frames'):
                    if key in status:
                        validation[key] = status[key]
                validation['native_failure'] = {k: status.get(k) for k in ('failure', 'stage', 'failed_original_idx', 'returncode')}
            except (OSError, ValueError):
                pass
            write_json(run / 'validation.json', validation)
        print(f'Error: {exc or "KeyboardInterrupt"}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
