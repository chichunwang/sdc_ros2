#!/usr/bin/env python3
"""Read-only plots for a successfully completed Track3 full run.

This entry point never estimates or alters poses. Use --run-dir PATH after the
runner has finalized test_info.json. Partial runs are deliberately rejected.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import time
import traceback
from datetime import datetime, timezone

TIMING_FIELDS = [
    'idx', 'original_idx', 'timestamp_ns', 'backend', 'load_ms',
    'registration_ms', 'total_ms', 'map_accepted', 'scale',
]
PLOT_STEMS = ['01_trajectory', '02_pose_changes', '03_registration_timing']
MEMORY_STEM = '04_gpu_memory'
FRAME_COUNT = 3142


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def read_rows(path: Path, fields: list[str] | None = None) -> list[dict]:
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if fields is not None and reader.fieldnames != fields:
            raise ValueError(f'{path.name}: expected columns {fields}, got {reader.fieldnames}')
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError(f'{path.name}: malformed CSV row')
        return rows


def integer(value: str, label: str) -> int:
    # Never route timestamps through float: ROS nanoseconds exceed 2**53.
    if not re.fullmatch(r'[0-9]+', value):
        raise ValueError(f'{label}: expected a nonnegative decimal integer, got {value!r}')
    return int(value)


def validate_inputs(run: Path, np):
    info = json.loads((run / 'test_info.json').read_text())
    if info.get('status') != 'success' or info.get('mode') != 'full':
        raise ValueError('Only status=success, mode=full runs can receive full analysis')
    if info.get('frame_count') != FRAME_COUNT:
        raise ValueError(f'test_info.json: frame_count must equal {FRAME_COUNT}')
    result = read_rows(run / 'result_3.csv', ['idx', 'x', 'y', 'yaw'])
    timing = read_rows(run / 'timings.csv', TIMING_FIELDS)
    if len(result) != FRAME_COUNT or len(timing) != FRAME_COUNT:
        raise ValueError('CSV and timings must each contain exactly 3142 data rows')
    if [integer(r['idx'], 'CSV idx') for r in result] != list(range(FRAME_COUNT)):
        raise ValueError('result_3.csv: idx must be contiguous and 0-based')
    if [integer(r['idx'], 'timing idx') for r in timing] != list(range(FRAME_COUNT)):
        raise ValueError('timings.csv: idx must be contiguous and 0-based')
    original = [integer(r['original_idx'], 'original_idx') for r in timing]
    if original != list(range(FRAME_COUNT)):
        raise ValueError('Full Track3 original_idx must cover 0..3141 in order')
    timestamps = [integer(r['timestamp_ns'], 'timestamp_ns') for r in timing]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError('Frame timestamps must strictly increase')
    if any(not row['backend'].strip() for row in timing):
        raise ValueError('timings.csv: backend must not be empty')
    if any(row['map_accepted'].lower() not in {'0', '1', 'false', 'true'} for row in timing):
        raise ValueError('timings.csv: map_accepted must be 0/1 or true/false')
    values = np.asarray([[float(r[k]) for k in ('x', 'y', 'yaw')] for r in result])
    durations = np.asarray([[float(r[k]) for k in ('load_ms', 'registration_ms', 'total_ms')]
                            for r in timing])
    scales = np.asarray([float(r['scale']) for r in timing])
    if not np.isfinite(values).all() or not np.isfinite(durations).all():
        raise ValueError('CSV and timing numeric values must be finite')
    if (durations < 0).any() or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Timing values must be nonnegative; scale must be finite and positive')
    poses = np.loadtxt(run / 'poses.txt', ndmin=2)
    if poses.shape != (FRAME_COUNT, 12) or not np.isfinite(poses).all():
        raise ValueError('poses.txt must have 3142 rows, each with 12 finite matrix values')
    if not np.allclose(poses[:, [3, 7]], values[:, :2], rtol=0, atol=1e-6):
        raise ValueError('CSV x/y do not agree with pose matrices')
    yaw_error = np.arctan2(np.sin(np.arctan2(poses[:, 4], poses[:, 0]) - values[:, 2]),
                           np.cos(np.arctan2(poses[:, 4], poses[:, 0]) - values[:, 2]))
    if np.max(np.abs(yaw_error)) > 1e-6:
        raise ValueError('CSV yaw does not agree with atan2(R10,R00), in radians')
    manifest = run / 'frame_manifest.csv'
    if manifest.exists():
        frames = read_rows(manifest)
        if len(frames) != FRAME_COUNT or any('timestamp_ns' not in row for row in frames):
            raise ValueError('frame_manifest.csv must contain 3142 timestamp_ns entries')
        if [integer(r['timestamp_ns'], 'manifest timestamp_ns') for r in frames] != timestamps:
            raise ValueError('frame_manifest.csv timestamps disagree with timings.csv')
        for key, expected in [('idx', list(range(FRAME_COUNT))), ('original_idx', original)]:
            if key in frames[0] and [integer(r[key], f'manifest {key}') for r in frames] != expected:
                raise ValueError(f'frame_manifest.csv {key} disagrees with timings.csv')
    return values, poses, durations


def plot_run(run: Path) -> dict:
    started = time.monotonic()
    if not run.is_dir():
        raise ValueError(f'Run directory does not exist: {run}')
    output = run / 'analysis'
    old_manifest = output / 'manifest.json'
    required = [run / name for name in ('test_info.json', 'result_3.csv', 'poses.txt', 'timings.csv')]
    if (run / 'frame_manifest.csv').exists():
        required.append(run / 'frame_manifest.csv')
    memory_path=run/'gpu_memory.jsonl'
    if memory_path.exists():required.append(memory_path)
    sources = {path.name: {'sha256': sha256(path), 'bytes': path.stat().st_size} for path in required}
    if old_manifest.exists():
        old = json.loads(old_manifest.read_text())
        if old.get('status') == 'success':
            raise FileExistsError('Completed analysis already exists; refusing to overwrite it')
        if old.get('sources') != sources:
            raise ValueError('Partial analysis source hashes differ; refusing to replace evidence')
    # Refuse to replace unrecognized files from another analysis workflow.
    if output.exists():
        recognized = {'manifest.json', 'README.md'} | {f'{name}.{ext}' for name in PLOT_STEMS+[MEMORY_STEM] for ext in ('png', 'pdf')}
        if any(path.name not in recognized for path in output.iterdir()):
            raise FileExistsError('Analysis directory contains unrecognized files; preserved')
        if any(output.iterdir()) and not old_manifest.exists():
            raise FileExistsError('Existing analysis has no source manifest; preserved')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    values, poses, durations = validate_inputs(run, np)
    output.mkdir(exist_ok=True)
    metadata = {
        'status': 'in_progress', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'frame_count': FRAME_COUNT, 'sources': sources,
        'script': {'path': str(Path(__file__).resolve()), 'sha256': sha256(Path(__file__))},
        'numpy_version': np.__version__, 'matplotlib_version': matplotlib.__version__,
        'note': 'Read-only diagnostics without ground truth; plots do not establish localization accuracy.',
    }
    write_json(old_manifest, metadata)
    try:
        frame_idx = np.arange(FRAME_COUNT)
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.plot(values[:, 0], values[:, 1], linewidth=1.0, label='Estimated trajectory')
        ax.scatter(*values[0, :2], marker='o', s=45, color='tab:green', label='Start', zorder=3)
        ax.scatter(*values[-1, :2], marker='x', s=55, color='tab:red', label='End', zorder=3)
        ax.set(xlabel='Map X [m]', ylabel='Map Y [m]', title='Track3 estimated trajectory (no ground truth)')
        ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=0.3); ax.legend()
        save_figure(fig, output, PLOT_STEMS[0]); plt.close(fig)

        displacement = np.linalg.norm(np.diff(poses[:, [3, 7, 11]], axis=0), axis=1)
        raw_yaw_delta = np.diff(values[:, 2])
        yaw_delta = np.arctan2(np.sin(raw_yaw_delta), np.cos(raw_yaw_delta))
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        axes[0].plot(frame_idx[1:], displacement, linewidth=0.8)
        axes[0].set(ylabel='3D displacement [m]', title='Changes between successive emitted poses')
        axes[1].plot(frame_idx[1:], yaw_delta, linewidth=0.8, color='tab:orange')
        axes[1].set(xlabel='Frame idx', ylabel='Wrapped yaw change [rad]')
        axes[0].set_ylim(0, max(float(displacement.max())*1.1, 0.001))
        yaw_limit = max(float(np.abs(yaw_delta).max())*1.1, 0.001)
        axes[1].set_ylim(-yaw_limit, yaw_limit)
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.ticklabel_format(axis='y', useOffset=False)
        save_figure(fig, output, PLOT_STEMS[1]); plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        for column, label, ax in zip(range(3), ['Load', 'Registration', 'Total frame'], axes):
            ax.plot(frame_idx, durations[:, column], linewidth=0.8)
            ax.set(ylabel=f'{label} [ms]', ylim=(0, max(float(durations[:, column].max())*1.1, 0.001))); ax.grid(alpha=0.3)
        axes[0].set_title('Recorded per-frame processing times')
        axes[-1].set_xlabel('Frame idx')
        save_figure(fig, output, PLOT_STEMS[2]); plt.close(fig)
        memory_note=''
        if memory_path.exists():
            records=[json.loads(line) for line in memory_path.read_text().splitlines() if line.strip()]
            samples=[row for row in records if row.get('stage')=='frame_end']
            if [row.get('idx') for row in samples]!=list(range(FRAME_COUNT)):
                raise ValueError('GPU memory frame_end samples must cover exactly 0..3141')
            fields=['pool_used_bytes','pool_reserved_bytes','pool_limit_bytes','device_free_bytes','device_total_bytes']
            if any(type(row.get(key)) is not int or row[key]<0 for row in samples for key in fields):
                raise ValueError('GPU memory measurements must be nonnegative integer bytes')
            used,reserved,limit,free,total=np.asarray([[row[key] for key in fields] for row in samples],dtype=np.float64).T/(1024**3)
            fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
            axes[0].plot(frame_idx,used,label='CuPy pool in use')
            axes[0].plot(frame_idx,reserved,label='CuPy pool reserved')
            axes[0].plot(frame_idx,limit,'--',label='Pool limit (excludes CUDA libraries/context)')
            axes[0].set(ylabel='GiB',title='GPU memory: end-of-frame samples, not an absolute peak');axes[0].legend()
            axes[1].plot(frame_idx,free,label='Whole device free',color='tab:green')
            axes[1].plot(frame_idx,total-free,label='Device total - free (driver/RViz/others)',color='tab:orange')
            axes[1].set(xlabel='Frame idx',ylabel='GiB');axes[1].legend()
            for ax in axes:ax.grid(alpha=.3)
            save_figure(fig,output,MEMORY_STEM);plt.close(fig)
            memory_note='- `04_gpu_memory.png/pdf`：逐幀末端的 CuPy in-use／reserved／limit 與整張卡 free／used，包含其他程序；不是絕對峰值。\n'
        # A changing source could indicate a still-running process. Never claim a
        # successful analysis of an input snapshot that changed while plotting.
        if any(sha256(path) != sources[path.name]['sha256'] for path in required):
            raise RuntimeError('Source changed during plotting; analysis is incomplete')
        (output / 'README.md').write_text(
            '# Track3 full run 唯讀圖表\n\n'
            '來源為本 run 的最終逐幀輸出，SHA256 與版本保存於 manifest.json。\n'
            '無 ground truth；軌跡、變化量、耗時與完整輸出不代表定位精度。\n\n'
            '- `01_trajectory.png/pdf`：map XY 軌跡、起點與終點，等比例座標。\n'
            '- `02_pose_changes.png/pdf`：相鄰 pose 的三維位移與 wrap 到 [-π, π] 的 yaw 差（rad）。\n'
            '- `03_registration_timing.png/pdf`：載入、配準與總 frame 耗時，單位 ms。\n\n'
            +memory_note+
            '原始 pose、CSV、參數與定位結果均未修改。準備、暖機與其他階段時間不包含在逐幀耗時圖中。\n'
            '本工具只接受成功的 3142 幀 full run；失敗資訊另存 run/analysis_status.json。\n',
            encoding='utf-8',
        )
        metadata['status'] = 'success'
        metadata['outputs'] = {path.name: {'sha256': sha256(path), 'bytes': path.stat().st_size}
                               for path in sorted(output.iterdir()) if path.name != 'manifest.json'}
        metadata['elapsed_seconds'] = time.monotonic() - started
        write_json(old_manifest, metadata)
        return {'status': 'success', 'elapsed_seconds': time.monotonic() - started,
                'analysis_dir': str(output), 'frame_count': FRAME_COUNT}
    finally:
        plt.close('all')


def save_figure(fig, output: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f'{stem}.png', dpi=160)
    fig.savefig(output / f'{stem}.pdf')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    started = time.monotonic()
    try:
        result = plot_run(run)
    except Exception as error:
        result = {'status': 'failed', 'error': str(error),
                  'elapsed_seconds': time.monotonic() - started}
        if run.is_dir():
            (run / 'analysis.log').open('a').write(traceback.format_exc() + '\n')
            # Preserve an existing successful analysis and its success status.
            prior_status = run / 'analysis_status.json'
            prior = json.loads(prior_status.read_text()) if prior_status.exists() else {}
            if prior.get('status') != 'success':
                write_json(prior_status, result)
        print(json.dumps(result, ensure_ascii=False))
        return 1
    write_json(run / 'analysis_status.json', result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
