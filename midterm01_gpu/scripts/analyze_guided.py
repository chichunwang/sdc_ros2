#!/usr/bin/env python3
"""完整軌跡診斷：感測器輸入不是真值；不計算或推估 Kaggle 成績。"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def load_run(path):
    frames = csv_rows(path / 'frames.csv')
    poses = np.loadtxt(path / 'result.txt').reshape(-1, 3, 4)
    assert len(frames) == len(poses)
    return frames, poses


def matrix_priors(path):
    rows = csv_rows(path / 'sensor_prior.csv')
    positions = np.array([[float(row[k]) for k in ('x', 'y', 'z')] for row in rows])
    quaternions = np.array([[float(row[k]) for k in ('qx', 'qy', 'qz', 'qw')] for row in rows])
    matrices = np.zeros((len(rows), 3, 4))
    matrices[:, :, :3] = Rotation.from_quat(quaternions).as_matrix()
    matrices[:, :, 3] = positions
    return rows, matrices


def trajectory_metrics(poses):
    xyz = poses[:, :, 3]
    angles = Rotation.from_matrix(poses[:, :, :3]).as_euler('xyz', degrees=True)
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    yaw_steps = np.abs((np.diff(angles[:, 2]) + 180) % 360 - 180)
    return {'frames': len(poses), 'start_end_xy_distance_m': float(np.linalg.norm(xyz[-1, :2] - xyz[0, :2])),
            'max_adjacent_translation_m': float(np.max(steps)), 'max_adjacent_yaw_deg': float(np.max(yaw_steps)),
            'roll_range_deg': [float(np.min(angles[:, 0])), float(np.max(angles[:, 0]))],
            'pitch_range_deg': [float(np.min(angles[:, 1])), float(np.max(angles[:, 1]))]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track1-run', type=Path, required=True)
    parser.add_argument('--track2-run', type=Path, required=True)
    parser.add_argument('--reference-track1', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {'scope': 'Sensor inputs and map consistency; no ground-truth or new Kaggle evaluation', 'tracks': {}}
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    fig.suptitle('Full sequences: GNSS/IMU are inputs, not ground truth; no new Kaggle score', fontsize=12)
    plt.rcParams.update({'axes.grid': True, 'grid.alpha': .2})
    for column, (track, directory) in enumerate([(1, args.track1_run), (2, args.track2_run)]):
        frames, poses = load_run(directory)
        prior_rows, priors = matrix_priors(directory)
        stamps = np.array([int(row['timestamp_ns']) for row in frames], dtype=np.int64)
        assert stamps.tolist() == [int(row['timestamp_ns']) for row in prior_rows]
        elapsed = (stamps - stamps[0]) / 1e9
        guidance = csv_rows(directory / 'guidance.csv')
        assert len(guidance) == len(poses)
        assert all(float(row['final_cost']) <= float(row['prior_cost']) + 1e-8 for row in guidance)
        correction = np.linalg.norm(poses[:, :2, 3] - priors[:, :2, 3], axis=1)
        assert np.max(correction) < 1.501
        methods = [('GNSS/IMU input prior', priors, '#888888', '--'), ('Sensor-guided estimate', poses, '#1467ad', '-')]
        info = {'run': str(directory), 'guided': trajectory_metrics(poses),
                'max_xy_offset_to_input_m_not_accuracy': float(np.max(correction)),
                'validation': json.loads((directory / 'validation.json').read_text()),
                'sample_map_evaluation': None}
        if track == 1:
            reference_frames, reference = load_run(args.reference_track1)
            assert frames == reference_frames
            methods.append(('Previous CUDA D2D', reference, '#ba4a2e', '-'))
            info['previous_cuda'] = trajectory_metrics(reference)
        for label, matrix, color, style in methods:
            yaw = np.unwrap(np.arctan2(matrix[:, 1, 0], matrix[:, 0, 0])) * 180 / np.pi
            axes[0, column].plot(matrix[:, 0, 3], matrix[:, 1, 3], style, color=color, label=label, lw=1.3)
            axes[1, column].plot(elapsed, yaw, style, color=color, label=label, lw=1)
            axes[2, column].plot(elapsed[1:], np.linalg.norm(np.diff(matrix[:, :, 3], axis=0), axis=1), style, color=color, label=label, lw=.8)
        axes[0, column].set(title=f'Track{track}: map-frame XY', xlabel='Map X (m)', ylabel='Map Y (m)')
        axes[0, column].set_aspect('equal', adjustable='datalim')
        axes[0, column].ticklabel_format(useOffset=False, style='plain')
        axes[1, column].set(title=f'Track{track}: unwrapped yaw', xlabel='Data time (s)', ylabel='Yaw (deg)')
        axes[2, column].set(title=f'Track{track}: adjacent pose displacement', xlabel='Data time (s)', ylabel='Step (m)')
        for row in range(3):
            axes[row, column].grid(alpha=.2)
            axes[row, column].legend(frameon=False, fontsize=8)
        # Fixed, uniformly spaced frames over the entire sequence, not just the start.
        indices = np.unique(np.linspace(0, len(frames) - 1, 40, dtype=int))
        sample = args.output / f'track{track}_map_sample'
        sample.mkdir(exist_ok=True)
        (sample / 'filenames.txt').write_text(''.join(frames[i]['filename'] + '\n' for i in indices))
        with (sample / 'frames.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['sample_idx', 'original_idx', 'timestamp_ns', 'filename'])
            writer.writerows((j, int(i), frames[i]['timestamp_ns'], frames[i]['filename']) for j, i in enumerate(indices))
        np.savetxt(sample / 'prior_result.txt', priors[indices].reshape(-1, 12), fmt='%.10g')
        np.savetxt(sample / 'guided_result.txt', poses[indices].reshape(-1, 12), fmt='%.10g')
        info['sample_map_evaluation'] = str(sample)
        summary['tracks'][str(track)] = info
    for extension in ('png', 'pdf'):
        fig.savefig(args.output / ('full_trajectory_comparison.' + extension), dpi=170)
    plt.close(fig)
    (args.output / 'trajectory_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({track: {key: info[key] for key in ('guided', 'max_xy_offset_to_input_m_not_accuracy')} for track, info in summary['tracks'].items()}, indent=2))


if __name__ == '__main__':
    main()
