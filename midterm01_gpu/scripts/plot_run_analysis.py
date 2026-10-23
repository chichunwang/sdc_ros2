#!/usr/bin/env python3
"""將一個已完成 full run 的保存結果畫到該 run/analysis；不執行定位或修正 pose。"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

from run_analysis import full_run_ready


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def load_run(run):
    """核對實際列數/索引/整數時間；沒有的可選診斷不捏造。"""
    import numpy as np
    validation = json.loads((run/'validation.json').read_text())
    if not full_run_ready(validation, full=True):
        raise ValueError('Analysis requires a complete, successful, validated full Track')
    n = validation['frames']
    consumed = ['validation.json', 'diagnostic.csv', 'result.txt', 'frames.csv', 'timings.csv']
    csv_rows = read_rows(run/'diagnostic.csv')
    frame_rows = read_rows(run/'frames.csv')
    timing_rows = read_rows(run/'timings.csv')
    for label, rows in [('diagnostic', csv_rows), ('frames', frame_rows), ('timings', timing_rows)]:
        if len(rows) != n or [int(row['idx']) for row in rows] != list(range(n)):
            raise ValueError(f'{label}: row count/idx differs from validated full run')
    stamps = [int(row['timestamp_ns']) for row in frame_rows]
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError('Frame timestamps must be strictly increasing integer nanoseconds')
    if [int(row['timestamp_ns']) for row in timing_rows] != stamps:
        raise ValueError('Timing timestamps differ from frame manifest')
    # 先減整數 origin 才轉 float，保留 epoch 奈秒精度。
    t = np.asarray([stamp-stamps[0] for stamp in stamps], dtype=np.float64)*1e-9
    pose = np.loadtxt(run/'result.txt', ndmin=2)
    if pose.shape != (n, 12) or not np.isfinite(pose).all():
        raise ValueError('Expected one finite 3x4 pose per frame')
    csv_pose = np.asarray([[float(row[key]) for key in ('x', 'y', 'yaw')] for row in csv_rows])
    yaw = np.arctan2(pose[:, 4], pose[:, 0])
    dyaw = np.arctan2(np.sin(csv_pose[:, 2]-yaw), np.cos(csv_pose[:, 2]-yaw))
    if (not np.isfinite(csv_pose).all() or not np.allclose(csv_pose[:, :2], pose[:, [3, 7]], atol=1e-5, rtol=0)
            or np.max(np.abs(dyaw)) > 1e-6):
        raise ValueError('Diagnostic CSV disagrees with saved pose matrices')
    data = {'validation': validation, 'n': n, 't': t, 'stamps': stamps,
            'xyz': pose[:, [3, 7, 11]], 'yaw': yaw, 'timings': timing_rows,
            'prior': None, 'guidance': None, 'observations': None, 'params': {}}
    for filename, key in [('sensor_prior.csv', 'prior'), ('guidance.csv', 'guidance')]:
        if (run/filename).exists():
            rows = read_rows(run/filename)
            if len(rows) != n or [int(row['timestamp_ns']) for row in rows] != stamps:
                raise ValueError(f'{filename}: timestamps/count differ from frames')
            data[key] = rows
            consumed.append(filename)
    if (run/'observations.jsonl').exists():
        rows = [json.loads(line) for line in (run/'observations.jsonl').read_text().splitlines() if line.strip()]
        if len(rows) != n or [int(row['timestamp_ns']) for row in rows] != stamps:
            raise ValueError('Map observations timestamps/count differ from frames')
        data['observations'] = rows
        consumed.append('observations.jsonl')
    if (run/'params.json').exists():
        data['params'] = json.loads((run/'params.json').read_text())
        consumed.append('params.json')
    data['inputs'] = {name: digest(run/name) for name in consumed}
    return data


def numeric(rows, name):
    """缺欄/無效數值以 NaN 顯示為空隙，不補成零。"""
    import numpy as np
    values = []
    for row in rows:
        try:
            value = float(row.get(name, 'nan'))
        except (TypeError, ValueError):
            value = float('nan')
        values.append(value if np.isfinite(value) else float('nan'))
    return np.asarray(values)


def flag(rows, name):
    import numpy as np
    return np.asarray([1. if str(row.get(name)).lower() in ('1', 'true') else
                       0. if str(row.get(name)).lower() in ('0', 'false') else float('nan') for row in rows])


def figures(data, title):
    import matplotlib.pyplot as plt
    import numpy as np
    t, xyz, yaw, n = data['t'], data['xyz'], data['yaw'], data['n']
    prior = data['prior']
    prior_xyz = np.column_stack([numeric(prior, key) for key in ('x', 'y', 'z')]) if prior else None
    if prior_xyz is not None and not np.isfinite(prior_xyz).all():
        raise ValueError('Nonfinite saved sensor prior position')
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'legend.fontsize': 8,
                         'lines.linewidth': 1.2})

    def decorate(fig, name):
        fig.suptitle(f'{title}\n{name} | saved diagnostics, no ground truth', fontsize=12)
        for ax in fig.axes:
            ax.grid(alpha=.22)
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.3), layout='constrained')
    origin = xyz[0, :2]
    if prior_xyz is not None:
        axes[0].plot(*(prior_xyz[:, :2]-origin).T, color='.65', label='Saved sensor prior')
    axes[0].plot(*(xyz[:, :2]-origin).T, label='Output')
    axes[0].scatter(0, 0, s=45, marker='o', color='green', label='Start')
    axes[0].scatter(*(xyz[-1, :2]-origin), s=55, marker='x', color='crimson', label='End')
    axes[0].set(xlabel='Map X - output start X (m)', ylabel='Map Y - output start Y (m)',
                title='XY trajectory; equal scale, no alignment fit', aspect='equal')
    axes[0].legend()
    for i, name in enumerate(('X', 'Y', 'Z')):
        axes[1].plot(t, xyz[:, i]-xyz[0, i], label=name)
    axes[1].set(xlabel='Elapsed frame time (s)', ylabel='Position - output start (m)', title='Saved position components')
    axes[1].legend()
    yield '01_trajectory', decorate(fig, 'Trajectory')

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4), layout='constrained')
    axes[0, 0].plot(t, np.degrees(np.unwrap(yaw)))
    axes[0, 0].set(ylabel='Unwrapped output yaw (deg)', title='Yaw for display only')
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    yaw_step = np.degrees(np.arctan2(np.sin(np.diff(yaw)), np.cos(np.diff(yaw))))
    axes[0, 1].plot(t[1:], step)
    axes[0, 1].set(ylabel='Adjacent 3D translation (m)', title='Motion between saved output frames')
    axes[1, 0].plot(t[1:], yaw_step)
    axes[1, 0].set(ylabel='Wrapped adjacent yaw change (deg)', title='Adjacent saved output yaw increments')
    if prior_xyz is not None:
        axes[1, 1].plot(t, np.linalg.norm(xyz[:, :2]-prior_xyz[:, :2], axis=1))
        axes[1, 1].set(ylabel='Output - prior XY norm (m)', title='Difference to saved prior; not ATE')
    else:
        axes[1, 1].plot(t, xyz[:, 2])
        axes[1, 1].set(ylabel='Saved map Z (m)', title='No saved sensor prior available')
    for ax in axes.flat:
        ax.set_xlabel('Elapsed frame time (s)')
        ax.set_xlim(t[0], t[-1])
    yield '02_pose_changes', decorate(fig, 'Pose changes')

    rows = data['timings']
    registration_key = 'registration_ms' if 'registration_ms' in rows[0] else 'ndt_ms'
    accepted_key = 'map_accepted' if 'map_accepted' in rows[0] else 'converged'
    accepted = flag(rows, accepted_key)
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4), layout='constrained')
    for key, label in [('load_ms', 'Load'), (registration_key, 'Registration'), ('total_ms', 'Frame total')]:
        axes[0, 0].plot(t, numeric(rows, key), label=label)
    axes[0, 0].set(ylabel='Time (ms)', title='Per-frame timings; preparation / graph excluded')
    axes[0, 0].legend()
    axes[0, 1].step(t, accepted, where='post', lw=.9)
    finite = np.isfinite(accepted)
    k = int(np.nansum(accepted))
    axes[0, 1].set(ylim=(-.1, 1.1), yticks=[0, 1], ylabel=accepted_key,
                   title=f'Map accepted / converged: {k}/{int(finite.sum())}; not accuracy')
    if data['observations']:
        residual = numeric(data['observations'], 'heldout_median_m')
        residual_label = 'Held-out median (m)'
        residual_title = 'Map candidate residual, including rejected frames'
    elif data['guidance']:
        residual = numeric(data['guidance'], 'final_clipped_rmse')
        residual_label = 'Clipped RMSE (m)'
        residual_title = 'Guided registration diagnostic; not pose error'
    else:
        residual = numeric(rows, 'score')
        residual_label = 'Backend fitness score (backend-defined units)'
        residual_title = 'Fitness; not comparable across different backends'
    axes[1, 0].plot(t, residual)
    axes[1, 0].set(ylabel=residual_label, title=residual_title)
    scale = numeric(rows, 'scale')
    if not np.isfinite(scale).any() and data['observations']:
        scale = numeric(data['observations'], 'scale')
    if np.isfinite(scale).any():
        axes[1, 1].plot(t, scale)
        axes[1, 1].set(ylabel='PLY coordinate multiplier', title='Saved per-frame scale; not calibrated truth')
    elif 'ply_scale' in data['params']:
        axes[1, 1].axhline(float(data['params']['ply_scale']))
        axes[1, 1].set(ylabel='PLY coordinate multiplier', title='Fixed configured scale; not calibrated truth')
    else:
        axes[1, 1].text(.5, .5, 'No saved scale diagnostic', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Unavailable diagnostics are not fabricated')
    for ax in axes.flat:
        ax.set_xlabel('Elapsed frame time (s)')
        ax.set_xlim(t[0], t[-1])
    yield '03_registration_timing', decorate(fig, 'Registration and timing')


def generate(run):
    """成功才發布 analysis/；既有圖完整且來源一致時重用，不覆寫其他證據。"""
    run = Path(run).resolve()
    data = load_run(run)
    output = run/'analysis'
    programs = {name: digest(Path(__file__).with_name(name)) for name in ('plot_run_analysis.py', 'run_analysis.py')}
    if output.exists():
        manifest = json.loads((output/'manifest.json').read_text())
        if (manifest.get('complete') is True and manifest.get('input_sha256') == data['inputs']
                and manifest.get('program_sha256') == programs
                and all(digest(output/item['path']) == item['sha256'] for item in manifest['artifacts'])):
            print(f'Analysis already complete: {output}')
            return output
        raise FileExistsError('Existing analysis differs/incomplete; preserved without overwriting')
    stage = Path(tempfile.mkdtemp(prefix='.analysis_', dir=run))
    try:
        os.environ.setdefault('MPLCONFIGDIR', str(stage/'.matplotlib'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        artifacts = []
        for stem, fig in figures(data, run.name):
            try:
                fig.canvas.draw()
                for extension in ('png', 'pdf'):
                    target = stage/f'{stem}.{extension}'
                    fig.savefig(target, dpi=150)
                    artifacts.append({'path': target.name, 'sha256': digest(target), 'bytes': target.stat().st_size})
            finally:
                plt.close(fig)
        # 再核對讀入資料未被定位/其他程式改寫，繪圖不改原輸入。
        if any(digest(run/name) != value for name, value in data['inputs'].items()):
            raise ValueError('Saved inputs changed during analysis; figures not published')
        step = np.linalg.norm(np.diff(data['xyz'], axis=0), axis=1)
        manifest = {'schema': 'sdc_full_run_analysis_v1', 'complete': True,
                    'created_utc': datetime.now(timezone.utc).isoformat(),
                    'source_run': str(run), 'track': data['validation']['track'], 'frames': data['n'],
                    'timestamp_origin_ns': data['stamps'][0], 'timestamp_end_ns': data['stamps'][-1],
                    'duration_s': float(data['t'][-1]), 'input_sha256': data['inputs'], 'program_sha256': programs,
                    'renderer': {'python': sys.version, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
                    'artifacts': artifacts, 'ground_truth_available': False, 'accuracy_evaluated': False,
                    'localization_rerun': False, 'trajectory_modified': False,
                    'diagnostics_not_accuracy': {'max_adjacent_translation_m': float(step.max()),
                                                'saved_path_length_m': float(step.sum())},
                    'plot_conventions': {'xy': 'map axes minus this output start; equal aspect; no fit',
                        'angles': 'internal yaw radians displayed in degrees; not submission unit certification',
                        'time': 'integer nanoseconds minus origin, then seconds',
                        'missing_values': 'gaps; no zero filling',
                        'timing': 'saved frame timings only; analysis/preparation/graph not apportioned'}}
        (stage/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        (stage/'README.md').write_text(
            '# Full run 分析圖\n\n'
            f'來源：`{run.name}`，Track {manifest["track"]}，{data["n"]} 幀。\n\n'
            '- `01_trajectory.png/pdf`：XY 等比例軌跡及各軸位置變化。\n'
            '- `02_pose_changes.png/pdf`：yaw、相鄰位移與轉角；有 prior 時顯示相對差。\n'
            '- `03_registration_timing.png/pdf`：接受／收斂狀態、殘差、尺度與逐幀耗時。\n\n'
            '各 run 使用相同圖名、單位和定義，座標顯示範圍依資料完整範圍自動決定；比較時請看軸刻度。'
            'XY 只減本輪第一幀原點，沒有額外對齊擬合。曲線、接受率、殘差與 prior 差不是 ATE 或真值。'
            '本階段不修正 pose、不重跑定位，也不認證 base 算法合規。\n\n'
            '`manifest.json` 記錄來源／圖／程式 SHA、整數時間與套件版本。'
            '產圖狀態及時間另外記於 run 根目錄 `analysis_status.json`／`analysis.log`。\n', encoding='utf-8')
        shutil.rmtree(stage/'.matplotlib', ignore_errors=True)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f'Analysis: {output} (3 PNG + 3 PDF)')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    status = {'status': 'failed', 'localization_rerun': False, 'trajectory_modified': False,
              'time_scope': 'analysis only; excluded from localization timings',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'command': [sys.executable, *sys.argv]}
    try:
        output = generate(args.run)
        status.update(status='complete', output=str(output), manifest='analysis/manifest.json', returncode=0)
        return 0
    except Exception as error:
        status.update(error=f'{type(error).__name__}: {error}', returncode=1)
        raise
    finally:
        status['elapsed_seconds'] = time.perf_counter()-started
        # 拒絕短段/失敗資料時不建立任何分析產物；自動 hook 的 log/status 另由父程序處理。
        try:
            saved = json.loads((args.run/'validation.json').read_text())
            if full_run_ready(saved, full=True):
                (args.run/'analysis_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2)+'\n')
        except (OSError, ValueError):
            pass


if __name__ == '__main__':
    raise SystemExit(main())
