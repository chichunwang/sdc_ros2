#!/usr/bin/env python3
"""Read-only heading experiment summary; no score or ground truth estimation."""
from output_layout import loads_saved_json, saved_path
import argparse
import json
from pathlib import Path

import numpy as np

from heading_ablation import MODES
from prior_ablation import dump, rows
from sensor_fusion import wrap
from three_camera_source import sha256


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=saved_path)
    parser.add_argument('--previous-experiment', type=saved_path, required=True)
    args = parser.parse_args(); experiment = args.experiment
    plans = loads_saved_json((experiment/'heading_ablation.json').read_text())
    pairs = loads_saved_json((experiment/'ablation_runs.json').read_text())
    previous = loads_saved_json((args.previous_experiment/'ablation_runs.json').read_text())
    if len(pairs) != sum(len(r['windows']) for r in plans):
        raise ValueError('Some planned short windows are incomplete')
    records = []
    for pair in pairs:
        record = {k: pair[k] for k in ('track', 'class', 'split', 'start', 'frames')}
        controls = next(r for r in previous if all(r[k] == pair[k] for k in ('track', 'class', 'split', 'start')))
        baseline = Path(pair['runs']['motion_rts'])
        old_baseline = Path(controls['runs']['motion_rts'])
        same = {name: (baseline/name).read_bytes() == (old_baseline/name).read_bytes()
                for name in ('result.txt', 'diagnostic.csv', 'geometry.csv', 'height.csv')}
        if not all(same.values()):
            raise ValueError('Unchanged motion baseline differs from prior scored-candidate control')
        record['baseline_byte_identical_to_previous_control'] = same
        base = np.loadtxt(baseline/'result.txt').reshape(-1, 3, 4)
        yaw = np.arctan2(base[:, 1, 0], base[:, 0, 0])
        record['methods'] = {}
        for mode in MODES:
            run = Path(pair['runs'][mode])
            check = loads_saved_json((run/'validation.json').read_text())
            if check['returncode'] or check['validation_errors'] or not check['output_complete_and_finite']:
                raise ValueError('Invalid short output')
            pose = np.loadtxt(run/'result.txt').reshape(-1, 3, 4)
            guidance = rows(run/'guidance.csv')
            geometry = rows(run/'geometry.csv')
            record['methods'][mode] = {'run': str(run),
                'output_xy_difference_to_control_rms_m': rms(np.linalg.norm(pose[:, :2, 3]-base[:, :2, 3], axis=1)),
                'output_yaw_difference_to_control_rms_deg': rms(np.degrees(wrap(np.arctan2(pose[:, 1, 0], pose[:, 0, 0])-yaw))),
                'accepted': sum(int(r['map_accepted']) for r in guidance),
                'mean_clipped_map_rmse_m': float(np.mean([float(r['final_clipped_rmse']) for r in guidance])),
                'mean_plane_rmse_m': float(np.mean([float(r['plane_rmse_m']) for r in geometry])),
                'output_sha256': {name: sha256(run/name) for name in ('result.txt', 'diagnostic.csv', 'validation.json')}}
        records.append(record)
    summary = {'guarded_validation': [{'track': r['track'], 'summary': r['guarded_validation']['summary']} for r in plans],
        'short_windows': records, 'short_poses': sum(3*r['frames'] for r in records),
        'all_original_controls_byte_identical': True,
        'decision': 'Mixed heldout input residuals; no promotion or parameter search. Full scoring, if desired, is manual.',
        'new_full_localization_started': False, 'new_kaggle_score': None,
        'limits': ['Pose orientation provenance and independence remain assumed, not verified.',
                   'Both heading references are noisy inputs, not ground truth.',
                   'Map residual, acceptance and version differences do not measure pose accuracy.',
                   'Short carry windows reset Z history; they do not replicate full-Track behavior.']}
    dump(experiment/'summary.json', summary)
    print(json.dumps({'short_poses': summary['short_poses'], 'all_original_controls_byte_identical': True,
                      'decision': summary['decision']}, indent=2))


if __name__ == '__main__':
    main()
