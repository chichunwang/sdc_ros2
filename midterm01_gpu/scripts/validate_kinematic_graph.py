#!/usr/bin/env python3
"""Validate GNSS/gyro prototype on fixed guarded GNSS blocks, never localization."""
from output_layout import loads_saved_json, saved_path
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from kinematic_graph import KinematicProblem
from prior_ablation import dump
from three_camera_source import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=saved_path, required=True)
    parser.add_argument('--previous-experiment', type=saved_path, required=True)
    parser.add_argument('--raw-directory', type=saved_path, required=True)
    args = parser.parse_args()
    output = args.experiment/'kinematic_validation.json'
    if output.exists(): raise ValueError('Preserve previous validation')
    original = loads_saved_json((args.previous_experiment/'fixed_time_block_validation.json').read_text())
    reports = []
    for track in (1, 2):
        raw_path = args.raw_directory/f'input_track{track}.npz'
        with np.load(raw_path) as data:
            raw = {key: data[key] for key in data.files}
        with np.load(args.previous_experiment/f'raw_track{track}.npz') as reference:
            for current, previous in (('gnss_bag_ns', 'gnss_ns'), ('gnss_xyz', 'xyz'),
                                      ('imu_bag_ns', 'imu_ns'), ('imu_quat', 'quat')):
                if not np.array_equal(raw[current], reference[previous]):
                    raise ValueError('Validation raw inputs differ from previous motion RTS study')
        gnss_ns, imu_ns = raw['gnss_bag_ns'], raw['imu_bag_ns']; origin = int(gnss_ns[0])
        gt, it = (gnss_ns-origin)/1e9, (imu_ns-origin)/1e9
        problem_args = gt, raw['gnss_xyz'][:, :2], it, raw['imu_quat'], raw['angular_velocity']
        state, fit = KinematicProblem(*problem_args).solve()
        print(f'Track{track} full sensor fit: success={fit["success"]}, nfev={fit["nfev"]}', flush=True)
        with (args.experiment/f'kinematic_track{track}_state.npz').open('xb') as stream:
            np.savez_compressed(stream, timestamp_ns=gnss_ns, state=state)
        dump(args.experiment/f'kinematic_track{track}_fit.json', fit)
        blocks = []
        for block in next(r['blocks'] for r in original if r['track'] == track):
            delta = (gnss_ns-int(block['center_ns']))/1e9
            validation, training = np.abs(delta) <= .375, np.abs(delta) > .625
            if gnss_ns[validation].tolist() != block['validation_timestamps_ns']:
                raise ValueError('Heldout timestamp mismatch')
            predicted, meta = KinematicProblem(*problem_args, training=training).solve()
            error = predicted[validation, :2]-raw['gnss_xyz'][validation, :2]
            item = dict(block, kinematic_prediction_rmse_m=float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
                        optimizer_success=meta['success'], nfev=meta['nfev'], optimality=meta['optimality'])
            blocks.append(item)
            print(f'Track{track} {block["split"]} t={block["center_s"]:.2f}: motion={block["motion_rts_prediction_rmse_m"]:.5f}, '
                  f'joint={item["kinematic_prediction_rmse_m"]:.5f}, success={meta["success"]}', flush=True)
        summary = []
        for split in ('design', 'holdout'):
            chosen = [b for b in blocks if b['split'] == split]
            count = sum(b['validation_rows'] for b in chosen)
            summary.append({'split': split, 'blocks': len(chosen), 'measurements': count,
                'motion_rts_rmse_m': float(np.sqrt(sum(b['validation_rows']*b['motion_rts_prediction_rmse_m']**2 for b in chosen)/count)),
                'kinematic_rmse_m': float(np.sqrt(sum(b['validation_rows']*b['kinematic_prediction_rmse_m']**2 for b in chosen)/count)),
                'kinematic_better_blocks': sum(b['kinematic_prediction_rmse_m'] < b['motion_rts_prediction_rmse_m'] for b in chosen),
                'all_optimizer_success': all(b['optimizer_success'] for b in chosen)})
        reports.append({'track': track, 'raw_export_sha256': sha256(raw_path), 'fit': fit,
                        'blocks': blocks, 'summary': summary})
        dump(args.experiment/'kinematic_validation_progress.json', reports)
    dump(output, {'tracks': reports, 'production_fusion_connected': False, 'map_factors_connected': False,
        'noise_tuned_on_holdout': False, 'full_localization_started': False,
        'limits': ['Heldout GNSS prediction is input consistency, not ground truth accuracy.',
                   'Forward-axis/lever-arm and nonholonomic assumptions are not externally calibrated.',
                   'No candidate becomes production or scored best solely from this diagnostic.']})


if __name__ == '__main__':
    main()
