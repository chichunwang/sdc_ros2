#!/usr/bin/env python3
"""Read existing full runs: score-facing XY/yaw changes are not truth errors."""
from output_layout import loads_saved_json, saved_path

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from three_camera_source import sha256


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def statistics(values):
    values = np.asarray(values, dtype=float)
    return {'rms': float(np.sqrt(np.mean(values**2))),
            'median': float(np.median(values)),
            'absolute_p90': float(np.quantile(np.abs(values), .9)),
            'absolute_max': float(np.max(np.abs(values)))}


def yaw(poses):
    return np.arctan2(poses[:, 1, 0], poses[:, 0, 0])


def angle_difference(a, b):
    return np.degrees(np.arctan2(np.sin(a-b), np.cos(a-b)))


def checked_run(run):
    run = run.resolve()
    validation = loads_saved_json((run/'validation.json').read_text())
    params = loads_saved_json((run/'params.json').read_text())
    track = validation['track']
    expected = {1: 3011, 2: 3299}[track]
    if (validation['returncode'] != 0 or not validation['output_complete_and_finite']
            or validation['validation_errors'] or validation.get('submission_frame_set_complete') is False
            or params['max_frames'] != expected):
        raise ValueError(f'Incomplete or failed full run: {run}')
    poses = np.loadtxt(run/'result.txt').reshape(-1, 3, 4)
    frames = read_csv(run/'frames.csv')
    priors = read_csv(run/'sensor_prior.csv')
    submission = read_csv(run/'diagnostic.csv')
    if not len(poses) == len(frames) == len(priors) == len(submission) == expected:
        raise ValueError('Full-run count mismatch')
    if not np.isfinite(poses).all():
        raise ValueError('Nonfinite pose')
    rotation = poses[:, :, :3]
    if (np.max(np.abs(rotation.transpose(0, 2, 1)@rotation-np.eye(3))) > 1e-5
            or np.max(np.abs(np.linalg.det(rotation)-1)) > 1e-5):
        raise ValueError('Invalid rotation')
    if any(list(row) != ['idx', 'x', 'y', 'yaw'] for row in submission):
        raise ValueError('Submission columns mismatch')
    if any(int(row['idx']) != index for index, row in enumerate(submission)):
        raise ValueError('Submission indices mismatch')
    if any(int(row['idx']) != index for index, row in enumerate(frames)):
        raise ValueError('Frame indices mismatch')
    actual = np.array([[float(row[k]) for k in ('x', 'y', 'yaw')] for row in submission])
    projected = np.column_stack((poses[:, 0, 3], poses[:, 1, 3], yaw(poses)))
    if not np.isfinite(actual).all() or np.max(np.abs(actual-projected)) > 1e-10:
        raise ValueError('Submission differs from result matrices')
    stamps = [int(f['timestamp_ns']) for f in frames]
    if (any(b <= a for a, b in zip(stamps, stamps[1:]))
            or stamps != [int(row['timestamp_ns']) for row in priors]):
        raise ValueError('Invalid frame/prior timestamps')
    return run, validation, poses, frames, priors


def run_diagnostics(data):
    run, validation, poses, frames, priors = data
    prior_xy = np.array([[float(row[k]) for k in ('x', 'y')] for row in priors])
    quats = np.array([[float(row[k]) for k in ('qx', 'qy', 'qz', 'qw')] for row in priors])
    x, y, z, w = quats.T
    prior_yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    geometry = read_csv(run/'geometry.csv')
    if len(geometry) != len(poses) or [int(row['timestamp_ns']) for row in geometry] != [int(row['timestamp_ns']) for row in frames]:
        raise ValueError('Geometry manifest mismatch')
    information = np.array([[[float(row[f'scaled_information_r{i}c{j}'])
        for j in range(4)] for i in range(4)] for row in geometry])
    # Nuisance Z is eliminated from the data-only scaled normal information.
    indices = [0, 1, 3]
    cross = information[:, indices, 2]
    schur = (information[:, indices][:, :, indices]
             - np.einsum('ni,nj->nij', cross, cross)
             / np.maximum(information[:, 2:3, 2:3], 1e-12))
    return {'run': str(run), 'track': validation['track'], 'frames': len(poses),
            'sha256': {name: sha256(run/name) for name in
                ('result.txt', 'diagnostic.csv', 'sensor_prior.csv', 'frames.csv', 'validation.json')},
            'z_span_m': float(np.ptp(poses[:, 2, 3])),
            'max_adjacent_z_change_m': float(np.max(np.abs(np.diff(poses[:, 2, 3])))),
            'xy_difference_to_input_prior_m': statistics(np.linalg.norm(poses[:, :2, 3]-prior_xy, axis=1)),
            'yaw_difference_to_input_prior_deg': statistics(angle_difference(yaw(poses), prior_yaw)),
            'normal_information_rank_histogram': {str(k): sum(int(row['observable_rank']) == k for row in geometry) for k in range(5)},
            'schur_normal_information_eigenvalues_median': np.median(np.linalg.eigvalsh(schur), axis=0).tolist(),
            'schur_normal_information_over_prior_diagonal_median': np.median(
                np.diagonal(schur, axis1=1, axis2=2)/[4., 4., 1.], axis=0).tolist(),
            'information_scope': 'Data-only scaled normal information; excludes tangent floor and normal gain. Not actual solver Hessian, calibrated covariance or truth accuracy.',
            'map_accepted_frames': validation['guidance_not_accuracy']['map_accepted_frames']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=saved_path, required=True)
    parser.add_argument('--candidate', type=saved_path, required=True)
    parser.add_argument('--output', type=saved_path, required=True)
    args = parser.parse_args()
    old, new = checked_run(args.baseline), checked_run(args.candidate)
    if old[1]['track'] != new[1]['track'] or old[3] != new[3]:
        raise ValueError('Track/frame order differs')
    if (old[0]/'sensor_prior.csv').read_bytes() != (new[0]/'sensor_prior.csv').read_bytes():
        raise ValueError('Expected identical sensor prior bytes for this height comparison')
    xy = np.linalg.norm(new[2][:, :2, 3]-old[2][:, :2, 3], axis=1)
    dyaw = angle_difference(yaw(new[2]), yaw(old[2]))
    only_z = old[2].copy()
    only_z[:, 2, 3] = new[2][:, 2, 3]
    assert np.array_equal(only_z[:, :2, 3], old[2][:, :2, 3])
    assert np.array_equal(yaw(only_z), yaw(old[2]))
    report = {'baseline': run_diagnostics(old), 'candidate': run_diagnostics(new),
        'same_frame_order': True, 'same_sensor_prior_bytes': True,
        'version_differences_not_truth_errors': {
            'xy_m': statistics(xy), 'yaw_deg': statistics(dyaw),
            'z_m': statistics(new[2][:, 2, 3]-old[2][:, 2, 3]),
            'xy_max_original_idx': int(np.argmax(xy)), 'yaw_max_original_idx': int(np.argmax(np.abs(dyaw)))},
        'only_z_exchange_leaves_all_submission_values_identical': True,
        'uploaded_file_hash_independently_verified': False,
        'limits': ['No ground truth: version changes cannot identify which individual frames are more accurate.',
                   'CSV contains x/y/yaw, no z/roll/pitch. Geometric changes can indirectly change x/y/yaw through registration.',
                   'One aggregate score cannot separate translation from yaw error or prove how much each change contributed.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(args.output)


if __name__ == '__main__':
    main()
