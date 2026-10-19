#!/usr/bin/env python3
"""Bounded, paired directional/GICP experiment with the same frozen sensor priors."""
from output_layout import loads_saved_json, saved_path, scored_reference_run
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from prior_ablation import dump, rows
from sensor_fusion import wrap
from three_camera_source import sha256


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=saved_path, required=True)
    parser.add_argument('--window-experiment', type=saved_path, required=True)
    for track in (1, 2):
        parser.add_argument(f'--map-cache-track{track}', type=saved_path, required=True)
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    experiment = args.experiment.resolve(); experiment.relative_to((root/'outputs/run').resolve())
    output = experiment/'short_runs'; output.mkdir()  # preserve prior attempts
    old = loads_saved_json((args.window_experiment/'ablation_runs.json').read_text())
    if not 1 <= len(old) <= 12 or any(not 1 <= p['frames'] <= 48 for p in old):
        raise ValueError('Only up to twelve preselected 48-frame windows, never full localization')
    snapshot = experiment/'sources'; snapshot.mkdir()
    for file in (Path(__file__), root/'scripts/accuracy/geometry_report.py', root/'scripts/accuracy/benchmark.py'):
        shutil.copy2(file, snapshot/file.name)
    records = []
    for previous in old:
        track, start, count = previous['track'], previous['start'], previous['frames']
        pair = {k: previous[k] for k in ('track', 'class', 'split', 'start', 'frames')}
        pair['runs'] = {}; invariants = None
        baseline_prior = scored_reference_run(root,track)
        for mode in ('directional', 'gicp'):
            name = f'track{track}_{pair["class"]}_{pair["split"]}_{mode}'
            command = [sys.executable, str(root/'scripts/benchmark.py'), '--track', str(track),
                '--start-frame', str(start), '--frames', str(count), '--profile', 'sensor_guided',
                '--sensor-fusion', 'motion_rts', '--geometry-mode', mode, '--height-mode', 'carry',
                '--prior-run', str(baseline_prior), '--map-cache', str(getattr(args, f'map_cache_track{track}').resolve())]
            dump(output/f'{name}.command.json', command)
            print(f'{name}: idx {start} + {count}', flush=True)
            log_path = output/f'{name}.log'
            with log_path.open('x') as log:
                result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            names = [line[5:] for line in log_path.read_text().splitlines() if line.startswith('Run: ')]
            if result.returncode or not names:
                raise RuntimeError(f'Bounded run failed; inspect {log_path}')
            run = Path(names[-1]); check = loads_saved_json((run/'validation.json').read_text())
            if check['returncode'] or check['validation_errors'] or not check['output_complete_and_finite'] or check['submission_frame_set_complete']:
                raise ValueError(f'Invalid short output: {run}')
            params = loads_saved_json((run/'params.json').read_text())
            for key in ('save_path', 'ply_path', 'sensor_prior_file', 'geometry_mode'):
                params.pop(key)
            identities = {'params': params, 'prior': sha256(run/'sensor_prior.csv'),
                'binary': sha256(run/'binary_sha256.txt'), 'source': sha256(run/'source_input.json')}
            if invariants is None: invariants = identities
            elif identities != invariants: raise ValueError('Non-registration ablation conditions changed')
            if mode == 'directional':
                control = Path(previous['runs']['motion_rts'])
                same = {f: (run/f).read_bytes() == (control/f).read_bytes()
                        for f in ('result.txt', 'diagnostic.csv', 'geometry.csv', 'height.csv')}
                pair['scored_method_control_byte_identical'] = same
                if not all(same.values()):
                    raise ValueError(f'Original directional behavior changed: {same}')
            pair['runs'][mode] = str(run)
            (output/name).symlink_to(Path('../../')/run.name, target_is_directory=True)
        a, b = [Path(pair['runs'][mode]) for mode in ('directional', 'gicp')]
        poses_a, poses_b = [np.loadtxt(p/'result.txt').reshape(-1, 3, 4) for p in (a, b)]
        yaw = [np.arctan2(p[:, 1, 0], p[:, 0, 0]) for p in (poses_a, poses_b)]
        pair['version_differences_not_accuracy'] = {
            'xy_rms_m': rms(np.linalg.norm(poses_b[:, :2, 3]-poses_a[:, :2, 3], axis=1)),
            'yaw_rms_deg': rms(np.degrees(wrap(yaw[1]-yaw[0])))}
        pair['diagnostics_not_accuracy'] = {}
        for mode in ('directional', 'gicp'):
            run = Path(pair['runs'][mode]); updates = rows(run/'guidance.csv')
            geometry = rows(run/'geometry.csv'); timing = rows(run/'timings.csv')
            pair['diagnostics_not_accuracy'][mode] = {'accepted': sum(int(r['map_accepted']) for r in updates),
                'mean_clipped_map_rmse_m': float(np.mean([float(r['final_clipped_rmse']) for r in updates])),
                'mean_plane_rmse_m': float(np.mean([float(r['plane_rmse_m']) for r in geometry])),
                'mean_solver_ms': float(np.mean([float(r['ndt_ms']) for r in timing])),
                'median_unique_targets': float(np.median([int(r['unique_targets']) for r in geometry]))}
            if mode == 'gicp':
                source = rows(run/'gicp.csv')
                pair['diagnostics_not_accuracy'][mode]['mean_source_reliable_fraction'] = float(np.mean([
                    int(r['source_reliable_points'])/max(1, int(r['input_points'])) for r in source]))
        records.append(pair); dump(experiment/'ablation_runs.json', records)
    dump(experiment/'summary.json', {'pairs': records, 'total_poses': sum(2*p['frames'] for p in records),
        'full_localization_started': False, 'scored_best_unchanged': True,
        'no_noise_or_gain_search': True, 'same_prior_cloud_binary_non_geometry_parameters': True,
        'limits': ['Map residual and acceptance are not pose accuracy or Kaggle score.',
                   'Surface covariances encode local geometry, not calibrated camera depth noise.',
                   'Short carry windows reset height history; full localization remains manual.']})
    print('Completed matched bounded windows; no full Track launched.', flush=True)


if __name__ == '__main__':
    main()
