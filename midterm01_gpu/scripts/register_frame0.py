#!/usr/bin/env python3
"""只使用官方 frame 0 PLY 與 XYZ 地圖，保存未融合的直接配準結果。"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback
from types import SimpleNamespace

# 保留 ROS entrypoint 的 PYTHONPATH，並優先使用本輪快照。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TIMING_FIELDS = ['idx', 'original_idx', 'timestamp_ns', 'backend', 'load_ms',
                 'registration_ms', 'total_ms', 'map_accepted', 'scale']


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not (run / 'source_manifest.json').is_file():
        parser.error('run-dir 必須先由 create_run.py 建立來源快照')
    if (run / 'test_info.json').exists():
        parser.error('拒絕覆寫既有 test_info.json，請建立新 run')
    started = time.perf_counter()
    visualizer = None
    info = dict(status='running', frame_count=0, frames=1, imu_used=False,
                localization_mode='camera_only_single_frame', gpu_backend='cupy_cuda',
                created_at=datetime.now(timezone.utc).isoformat(),
                source_manifest_sha256=sha256(run / 'source_manifest.json'),
                score=None, accuracy_verified=False,
                timing_seconds=dict(preparation=0., load=0., registration=0.,
                                    visualization=0., graph=0., plotting=0.),
                failure=None)
    write_json(run / 'test_info.json', info)
    try:
        import numpy as np
        import cupy as cp
        from track3.geometry import valid_pose, yaw
        from track3.io import load_map, load_ply
        from track3.grounded import GroundedRegistration
        from track3.visualization import RvizPublisher

        cfg = json.loads(args.config.read_text())
        info['algorithm_version'] = cfg['version']
        write_json(run / 'parameters.json', cfg)
        data = Path(cfg['data_dir'])
        matches = sorted((data / 'raw_ply').glob('000000_ts_*.ply'))
        if len(matches) != 1:
            raise ValueError(f'frame 0 PLY 必須恰好一個，實際 {len(matches)}')
        ply = matches[0]
        match = re.fullmatch(r'000000_ts_(\d+)\.ply', ply.name)
        if match is None:
            raise ValueError(f'無效 frame 0 名稱：{ply.name}')
        timestamp_ns = int(match[1])
        sources = {'config': args.config.resolve(), 'map': data / 'map.pcd',
                   'initial_pose': data / 'init_guess.txt', 'frame0': ply}
        write_json(run / 'input_sources.json', {
            key: {'path': str(path), 'sha256': sha256(path)}
            for key, path in sources.items()})
        with (run / 'frame_manifest.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['idx', 'original_idx', 'timestamp_ns', 'path', 'sha256'])
            writer.writerow([0, 0, timestamp_ns, str(ply), sha256(ply)])
        info['timing_seconds']['preparation'] = time.perf_counter() - started
        loaded = time.perf_counter()
        T_initial = np.loadtxt(sources['initial_pose'], delimiter=',')
        if not valid_pose(T_initial):
            raise ValueError('init_guess 不是有效 SE(3)')
        points = load_ply(ply)
        map_points, map_count = load_map(sources['map'], cfg['map_leaf_m'])
        cp.cuda.Stream.null.synchronize()
        info['timing_seconds']['load'] = time.perf_counter() - loaded
        info['point_counts'] = {'source': len(points), 'map_raw': int(map_count),
                                'map_downsampled': len(map_points)}
        visualizer = RvizPublisher()
        visualizer.publish_map(cp.asnumpy(map_points), timestamp_ns)
        registration_started = time.perf_counter()
        registration = GroundedRegistration(map_points, cfg['registration'])
        result = registration.align_frame0(points, T_initial, initial_scales=None)
        cp.cuda.Stream.null.synchronize()
        info['timing_seconds']['registration'] = time.perf_counter() - registration_started
        T = np.asarray(result['T'], dtype=np.float64)
        scale = float(result['scale'])
        if not valid_pose(T) or not np.isfinite(scale) or scale <= 0:
            raise ValueError('配準回傳無效 pose／scale')
        accepted = bool(result['accepted'])
        write_json(run / 'registration.json', dict(
            T=T.tolist(), scale=scale, accepted=accepted, reason=result['reason'],
            diagnostics=result['diagnostics'], accuracy_verified=False))
        write_json(run / 'candidates.json', result.get('candidates', []))
        np.savetxt(run / 'poses.txt', T[:3, :4].reshape(1, 12), fmt='%.17g')
        with (run / 'result_3.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['idx', 'x', 'y', 'yaw'])
            writer.writerow([0, float(T[0, 3]), float(T[1, 3]), yaw(T[:3, :3])])
        total_ms = 1000 * (info['timing_seconds']['load'] +
                           info['timing_seconds']['registration'])
        with (run / 'timings.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(TIMING_FIELDS)
            writer.writerow([0, 0, timestamp_ns, 'cupy_cuda',
                             1000 * info['timing_seconds']['load'],
                             1000 * info['timing_seconds']['registration'],
                             total_ms, int(accepted), scale])
        visualization_started = time.perf_counter()
        adapter = registration if (hasattr(registration, 'center') and
                                   bool(getattr(registration, 'models', {}))) else SimpleNamespace(center=None)
        visualizer.publish_frame(points, T, scale, timestamp_ns, adapter)
        deadline = time.perf_counter() + 1.
        while time.perf_counter() < deadline:
            visualizer.spin()
            time.sleep(.01)
        info['timing_seconds']['visualization'] = time.perf_counter() - visualization_started
        info.update(status='completed' if accepted else 'rejected', frame_count=1,
                    map_accepted=accepted, scale=scale, reason=result['reason'])
        info['outputs_sha256'] = {name: sha256(run / name) for name in
                                  ('poses.txt', 'result_3.csv', 'timings.csv',
                                   'registration.json', 'candidates.json')}
        print(f'idx=0 frames=1/1 map={int(accepted)} scale={scale:.6f} '
              f'reg={1000 * info["timing_seconds"]["registration"]:.1f}ms '
              f'reason={result["reason"]}', flush=True)
        return 0 if accepted else 2
    except Exception as exc:
        failure = traceback.format_exc()
        (run / 'failure.log').write_text(failure)
        info.update(status='failed', failure={'type': type(exc).__name__, 'message': str(exc)})
        print(failure, file=sys.stderr)
        return 1
    finally:
        if visualizer is not None:
            try:
                visualizer.close()
            except Exception:
                (run / 'rviz_close_failure.log').write_text(traceback.format_exc())
        info['elapsed_seconds'] = time.perf_counter() - started
        write_json(run / 'test_info.json', info)


if __name__ == '__main__':
    sys.exit(main())
