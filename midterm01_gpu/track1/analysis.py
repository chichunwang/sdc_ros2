"""唯讀 Track1 完整 full 圖表；與 Track3 analyze_run.py 隔離。"""
import csv
import json
from pathlib import Path
import numpy as np
from track3.io import sha256, write_json


def validate(run, expected=3011):
    run = Path(run)
    with (run/'result_1.csv').open() as f:
        reader = csv.reader(f)
        if next(reader) != ['idx', 'x', 'y', 'yaw']:
            raise ValueError('Unexpected result columns')
        result = np.array([[float(v) for v in row] for row in reader])
    poses = np.loadtxt(run/'poses.txt', ndmin=2)
    with (run/'timings.csv').open() as f:
        timing = list(csv.DictReader(f))
    with (run/'frame_manifest.csv').open() as f:
        manifest = list(csv.DictReader(f))
    if result.shape != (expected, 4) or poses.shape != (expected, 12) or len(timing) != expected or len(manifest) != expected:
        raise ValueError('Incomplete output')
    if not np.isfinite(result).all() or not np.isfinite(poses).all() or not np.array_equal(result[:, 0], np.arange(expected)):
        raise ValueError('Invalid result index/data')
    if not np.allclose(result[:, 1:3], poses[:, [3, 7]], atol=1e-8, rtol=0):
        raise ValueError('CSV/TXT positions disagree')
    delta = result[:, 3]-np.arctan2(poses[:, 4], poses[:, 0])
    if not np.all(np.abs(np.arctan2(np.sin(delta), np.cos(delta))) < 1e-8):
        raise ValueError('CSV/TXT yaw mismatch')
    last_stamp = last_camera = None
    for i, (m, t) in enumerate(zip(manifest, timing)):
        stamp = int(m['timestamp_ns']); camera = int(m['camera_timestamp_ns'])
        if (int(m['idx']) != i or int(m['original_idx']) != i or int(t['idx']) != i
                or int(t['original_idx']) != i or int(t['timestamp_ns']) != stamp
                or not 0 < camera <= stamp or (last_stamp is not None and (stamp <= last_stamp or camera <= last_camera))):
            raise ValueError('Frame/time contract mismatch')
        R = poses[i].reshape(3, 4)[:, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-8) or abs(np.linalg.det(R)-1) > 1e-8:
            raise ValueError('Non-rigid TXT rotation')
        last_stamp, last_camera = stamp, camera
    durations = np.array([[float(t[k]) for k in ('load_ms', 'registration_ms', 'total_ms')] for t in timing])
    if not np.isfinite(durations).all() or np.any(durations < 0):
        raise ValueError('Invalid timing')
    return result, poses, durations


def analyze(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run = Path(run)
    info = json.loads((run/'test_info.json').read_text())
    if info['status'] != 'completed' or info['mode'] != 'full' or info['frame_count'] != 3011:
        raise ValueError('Only a completed 3011-frame full receives standard analysis')
    result, poses, durations = validate(run)
    obs = [json.loads(line) for line in (run/'observations.jsonl').read_text().splitlines()]
    has_memory = (len(obs) == len(result) and
                  all('correction_filter' in o.get('registration', {}) for o in obs))
    names = ['result_1.csv', 'poses.txt', 'timings.csv', 'frame_manifest.csv', 'observations.jsonl', 'test_info.json']
    sources = {n: sha256(run/n) for n in names}
    output = run/'analysis'
    output.mkdir(exist_ok=True)
    manifest_path = output/'manifest.json'
    stems = ('01_trajectory', '02_pose_changes', '03_registration_timing', '04_fusion_and_scale')
    if has_memory:
        stems += ('05_map_correction_memory',)
    recognized = {'manifest.json', 'README.md'} | {f'{s}.{ext}' for s in stems for ext in ('png', 'pdf')}
    if any(f.name not in recognized for f in output.iterdir()):
        raise FileExistsError('Unrecognized analysis artifacts preserved')
    if any(output.iterdir()):
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()).get('sources') != sources:
            raise FileExistsError('Existing analysis source SHA differs or has no manifest; preserved')
    write_json(manifest_path, dict(status='in_progress', sources=sources,
                                  script_sha256=sha256(Path(__file__))))
    def save(fig, stem):
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(output/f'{stem}.{ext}', dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(result[:, 1], result[:, 2], lw=1, label='Causal fused camera pose')
    ax.scatter(*result[0, 1:3], label='Start'); ax.scatter(*result[-1, 1:3], marker='x', label='End')
    ax.set(xlabel='Map X [m]', ylabel='Map Y [m]', title='Track1 estimated trajectory (no ground truth)')
    ax.axis('equal'); ax.grid(alpha=.3); ax.legend(); save(fig, '01_trajectory')
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(np.linalg.norm(np.diff(poses[:, [3, 7, 11]], axis=0), axis=1))
    d = np.diff(result[:, 3]); axes[1].plot(np.arctan2(np.sin(d), np.cos(d)))
    axes[0].set(ylabel='3D displacement [m]', title='Changes between emitted poses')
    axes[1].set(xlabel='Frame idx', ylabel='Yaw change [rad]')
    for ax in axes: ax.grid(alpha=.3)
    save(fig, '02_pose_changes')
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for i, label in enumerate(('Load + sensors', 'Registration', 'Total frame')):
        axes[i].plot(durations[:, i]); axes[i].set(ylabel=label+' [ms]'); axes[i].grid(alpha=.3)
    axes[-1].set_xlabel('Frame idx'); save(fig, '03_registration_timing')
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot([np.degrees(o['state'][3]) for o in obs]); axes[0].set_ylabel('Heading bias [deg]')
    axes[1].plot([o['state'][4] for o in obs]); axes[1].set_ylabel('Sensor speed [m/s]')
    axes[2].plot([o['scale'] for o in obs]); axes[2].set(xlabel='Frame idx', ylabel='PLY scale')
    for ax in axes: ax.grid(alpha=.3)
    save(fig, '04_fusion_and_scale')
    if has_memory:
        memory = [o['registration']['correction_filter'] for o in obs]
        correction = np.array([d['correction'] for d in memory])
        std = np.sqrt(np.array([np.diag(d['covariance']) for d in memory]))
        idx = np.arange(len(memory))
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        for j, name in enumerate(('Map X', 'Map Y')):
            axes[0].plot(idx, correction[:, j], label=name)
            axes[0].fill_between(idx, correction[:, j]-std[:, j], correction[:, j]+std[:, j], alpha=.15)
        axes[0].set(ylabel='Correction [m]', title='Causal map correction state (bands: model 1-sigma, uncalibrated)')
        axes[0].legend()
        axes[1].plot(idx, np.degrees(correction[:, 2]))
        axes[1].fill_between(idx, np.degrees(correction[:, 2]-std[:, 2]), np.degrees(correction[:, 2]+std[:, 2]), alpha=.15)
        axes[1].set_ylabel('Yaw correction [deg]')
        nis = [np.nan if d['innovation_nis'] is None else d['innovation_nis'] for d in memory]
        axes[2].plot(idx, nis, lw=.8)
        from . import parameters as p
        axes[2].axhline(p.CORRECTION_NIS_SOFT, color='tab:orange', ls='--', label='Soft gate')
        axes[2].axhline(p.CORRECTION_NIS_REJECT, color='tab:red', ls='--', label='Hard gate')
        axes[2].set(ylabel='Innovation NIS', yscale='symlog'); axes[2].legend()
        age = np.array([np.nan if d['seconds_since_update'] is None else d['seconds_since_update'] for d in memory])
        axes[3].plot(idx, age, label='Age of last used map update')
        reject = np.array([not d['accepted'] for d in memory])
        axes[3].scatter(idx[reject], age[reject], s=7, color='tab:red', label='No current map update')
        axes[3].set(xlabel='Frame idx', ylabel='Age [s]'); axes[3].legend()
        for ax in axes: ax.grid(alpha=.3)
        save(fig, '05_map_correction_memory')
    # Never change input files while plotting, including info used as evidence.
    if sources != {n: sha256(run/n) for n in names}:
        raise RuntimeError('Input changed during analysis')
    write_json(manifest_path, dict(status='completed', sources=sources,
        script_sha256=sha256(Path(__file__)), ground_truth_available=False,
        caveat='Acceptance and residuals are not localization accuracy'))
    (output/'README.md').write_text('# Track1 因果定位唯讀圖表\n\n軌跡、相鄰 pose 變化、耗時及融合狀態；沒有獨立真值。\n\n'
        +('05 顯示 world XY/yaw 修正、模型不確定度、innovation NIS 及有效配準的失聯時間；不確定度尚未經真值校準。\n\n' if has_memory else '')
        +'GNSS 是融合輸入，不是真值；接受率與殘差不能證明精度。來源 SHA 見 manifest.json。\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--run-dir', type=Path, required=True)
    analyze(parser.parse_args().run_dir)
