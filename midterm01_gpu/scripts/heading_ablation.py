#!/usr/bin/env python3
"""Prepare frozen-XYZ heading ablation and guarded time-block input validation.

This entry never runs localization. Orientation provenance remains unverified.
"""
from output_layout import loads_saved_json, saved_path, scored_reference_run
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from prepare_sensor_prior import make_priors
from prior_ablation import dump, rows
from sensor_fusion import (FusionConfig, bias_rts, heading_observations,
                           independent_heading_product, pose_heading_observations, wrap)
from three_camera_source import sha256

MODES = ('motion_rts', 'pose_heading_rts', 'independent_heading_rts')


def select_factors(course, pose, mode):
    if mode == 'independent_heading_rts':
        return independent_heading_product(course + pose)
    return [r for r in (course if mode == 'motion_rts' else pose) if r['accepted']]


def guarded_validation(gt, positions, it, imu_yaw, pose_quat, config):
    course = heading_observations(gt, positions[:, :2], it, imu_yaw, config)
    pose = pose_heading_observations(gt, pose_quat, it, imu_yaw, config)
    # Twelve fixed, spread-out blocks; remove the whole target block plus both
    # adjacent 2s blocks from BOTH observation families. No refit noise tuning.
    chosen = np.unique(np.rint(np.linspace(2, len(course)-3, 12)).astype(int))
    result = []
    for number, index in enumerate(chosen):
        train_course = [r for j, r in enumerate(course) if abs(j-index) > 1]
        train_pose = [r for j, r in enumerate(pose) if abs(j-index) > 1]
        item = {'block': int(index), 'time_s': course[index]['time_s'],
                'split': 'design' if number % 2 == 0 else 'holdout',
                'course_reference_accepted': course[index]['accepted'],
                'pose_reference_accepted': pose[index]['accepted'], 'methods': {}}
        for mode in MODES:
            selected = select_factors(train_course, train_pose, mode)
            if len(selected) < config.min_heading_blocks:
                raise ValueError('Insufficient independent training blocks; do not reuse fixed global bias in holdout')
            bias, _, meta = bias_rts(np.array([r['time_s'] for r in selected]),
                np.deg2rad([r['offset_deg'] for r in selected]),
                np.deg2rad([r['std_deg'] for r in selected]), np.array([item['time_s']]), config)
            predicted = float(np.rad2deg(bias[0]))
            method = {'predicted_bias_deg': predicted, 'training_knots': len(selected), **meta}
            for name, reference in (('course', course[index]), ('pose', pose[index])):
                method['heldout_'+name+'_residual_deg'] = (float(np.rad2deg(wrap(np.deg2rad(predicted-reference['offset_deg']))))
                                                          if reference['accepted'] else None)
            item['methods'][mode] = method
        result.append(item)
    summary = []
    for split in ('design', 'holdout'):
        for source in ('course', 'pose'):
            group = [r for r in result if r['split'] == split and r[source+'_reference_accepted']]
            summary.append({'split': split, 'reference': source, 'blocks': len(group),
                'rmse_deg': {mode: float(np.sqrt(np.mean([r['methods'][mode]['heldout_'+source+'_residual_deg']**2 for r in group])))
                             for mode in MODES} if group else {}})
    return {'blocks': result, 'summary': summary,
            'validation_is_ground_truth': False, 'uncertainty_tuned_on_holdout': False,
            'guard': 'target original 2-second block and neighboring blocks excluded from both observation families',
            'limits': ['Input heading and course are noisy references, possibly correlated with training/IMU.',
                       'Better prediction of supplied orientation does not prove better vehicle heading.',
                       'Retained IMU quaternion is an input at test timestamps, never a heading label.']}


def prepare(track, baseline, raw_path, output, windows):
    config = FusionConfig()
    raw_meta = loads_saved_json(raw_path.with_suffix('.json').read_text())
    if raw_meta['raw_values_sha256'] != sha256(raw_path):
        raise ValueError('Raw export hash mismatch')
    with np.load(raw_path) as data:
        d = {key: data[key] for key in data.files}
    original = rows(baseline/'sensor_prior.csv')
    ns = np.array([int(r['timestamp_ns']) for r in original], dtype=np.int64)
    common = d['gnss_bag_ns'], d['gnss_xyz'], d['imu_bag_ns'], d['imu_quat'], ns
    baseline_xyz = np.array([[r[k] for k in ('x', 'y', 'z')] for r in original], float)
    baseline_quat = np.array([[r[k] for k in ('qx', 'qy', 'qz', 'qw')] for r in original], float)
    report = {'track': track, 'baseline_run': str(baseline), 'raw_export': str(raw_path),
              'raw_sha256': sha256(raw_path), 'raw_bag_sha256': raw_meta['bag_sha256'],
              'windows': windows, 'fixed_config': asdict(config), 'methods': {}}
    for mode in MODES:
        details = {}
        xyz, quat, meta = make_priors(*common, fusion=mode, details=details,
                                     gnss_quaternions=d['gnss_quat'])
        if not np.allclose(xyz, baseline_xyz, rtol=0, atol=2e-10):
            raise ValueError('Motion XYZ differs from frozen scored baseline')
        angles = Rotation.from_quat(quat).as_euler('xyz')
        original_angles = Rotation.from_quat(baseline_quat).as_euler('xyz')
        if not np.allclose(angles[:, :2], original_angles[:, :2], rtol=0, atol=1e-10):
            raise ValueError('Heading-only candidate changed roll/pitch')
        if mode == 'motion_rts' and not np.allclose(quat, baseline_quat, rtol=0, atol=1e-12):
            raise ValueError('Baseline heading could not be reproduced')
        candidate = output/f'track{track}_{mode}_prior'; candidate.mkdir()
        import csv
        with (candidate/'sensor_prior.csv').open('x', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(list(original[0]))
            for old, q in zip(original, quat):
                # Freeze original timestamp and XYZ *strings*, not just floats.
                writer.writerow([old[k] for k in ('timestamp_ns', 'x', 'y', 'z')]+
                                ([old[k] for k in ('qx', 'qy', 'qz', 'qw')] if mode == 'motion_rts' else list(q)))
        with (candidate/'sensor_prior.fusion.csv').open('x', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(list(details['diagnostic']))
            writer.writerows(zip(*details['diagnostic'].values()))
        with (candidate/'sensor_prior.heading_blocks.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(details['blocks'][0]))
            writer.writeheader(); writer.writerows(details['blocks'])
        meta['heading_only_ablation'] = {'baseline': str(baseline), 'baseline_prior_sha256': sha256(baseline/'sensor_prior.csv'),
            'frozen': 'original timestamp/XYZ strings and IMU roll/pitch; no map posterior as sensor data',
            'orientation_source_verified': False, 'raw_export_sha256': sha256(raw_path)}
        dump(candidate/'sensor_prior.json', meta)
        dump(candidate/'validation.json', {'track': track, 'profile': 'sensor_guided', 'sensor_fusion': mode,
                                          'prior_only_not_localization': True})
        dyaw = np.degrees(wrap(angles[:, 2]-original_angles[:, 2]))
        report['methods'][mode] = {'prior': str(candidate), 'yaw_change_from_scored_rms_deg': float(np.sqrt(np.mean(dyaw**2))),
            'yaw_change_from_scored_range_deg': [float(dyaw.min()), float(dyaw.max())], **meta['sensor_fusion']}
    origin = int(d['gnss_bag_ns'][0])
    gt, it = (d['gnss_bag_ns']-origin)/1e9, (d['imu_bag_ns']-origin)/1e9
    imu_yaw = np.unwrap(Rotation.from_quat(d['imu_quat']).as_euler('xyz')[:, 2])
    report['guarded_validation'] = guarded_validation(gt, d['gnss_xyz'], it, imu_yaw, d['gnss_quat'], config)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=saved_path, required=True)
    parser.add_argument('--raw-directory', type=saved_path, required=True)
    parser.add_argument('--window-experiment', type=saved_path, required=True)
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    output = args.output.resolve(); output.relative_to((root/'outputs/run').resolve())
    output.mkdir()  # never overwrite an experiment
    previous = loads_saved_json((args.window_experiment/'prior_ablation.json').read_text())
    reports = [prepare(track, scored_reference_run(root,track),
                       args.raw_directory/f'input_track{track}.npz', output,
                       next(r['windows'] for r in previous if r['track'] == track)) for track in (1, 2)]
    dump(output/'heading_ablation.json', reports)
    sources = output/'sources'; sources.mkdir()
    for path in (Path(__file__), root/'scripts/sensor_fusion.py', root/'scripts/prepare_sensor_prior.py',
                 root/'scripts/audit_input_heading.py', root/'scripts/accuracy/benchmark.py'):
        shutil.copy2(path, sources/path.name)
    dump(output/'source_sha256.json', {p.name: sha256(p) for p in sources.iterdir()})
    print(json.dumps([{'track': r['track'], 'methods': {m: r['methods'][m]['yaw_change_from_scored_rms_deg'] for m in MODES},
                      'validation': r['guarded_validation']['summary']} for r in reports], indent=2))


if __name__ == '__main__':
    main()
