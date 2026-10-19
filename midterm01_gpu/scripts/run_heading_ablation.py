#!/usr/bin/env python3
"""Run at most twelve preselected 48-frame windows with frozen XYZ, three headings."""
from output_layout import loads_saved_json, saved_path
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from heading_ablation import MODES
from prior_ablation import dump, rows
from three_camera_source import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=saved_path, required=True)
    for track in (1, 2):
        parser.add_argument(f'--map-cache-track{track}', type=saved_path, required=True)
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    experiment = args.experiment.resolve(); experiment.relative_to((root/'outputs/run').resolve())
    if (experiment/'short_runs').exists():
        raise ValueError('Preserve prior short runs; use a new experiment')
    reports = loads_saved_json((experiment/'heading_ablation.json').read_text())
    windows = [(r, w) for r in reports for w in r['windows']]
    if not 1 <= len(windows) <= 12 or any(not 1 <= w['frames'] <= 48 for _, w in windows):
        raise ValueError('Only bounded, preselected short windows are permitted')
    short = experiment/'short_runs'; short.mkdir()
    records = []
    for report, window in windows:
        track = report['track']; item = {'track': track, **window, 'runs': {}}
        reference = None
        for mode in MODES:
            name = f'track{track}_{window["class"]}_{window["split"]}_{mode}'
            command = [sys.executable, str(root/'scripts/benchmark.py'), '--track', str(track),
                '--frames', str(window['frames']), '--start-frame', str(window['start']),
                '--profile', 'sensor_guided', '--sensor-fusion', mode, '--geometry-mode', 'directional',
                '--height-mode', 'carry', '--prior-run', report['methods'][mode]['prior'],
                '--map-cache', str(getattr(args, f'map_cache_track{track}').resolve())]
            dump(short/f'{name}.command.json', command)
            print(f'{name}: {window["start"]} + {window["frames"]} frames', flush=True)
            log_path = short/f'{name}.log'
            with log_path.open('x') as log:
                result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            matches = [line[5:] for line in log_path.read_text().splitlines() if line.startswith('Run: ')]
            if result.returncode or not matches:
                raise RuntimeError(f'Short run failed: {log_path}')
            run = Path(matches[-1]); check = loads_saved_json((run/'validation.json').read_text())
            if check['returncode'] or check['validation_errors'] or not check['output_complete_and_finite'] or check['submission_frame_set_complete']:
                raise ValueError(f'Invalid bounded output: {run}')
            prior = rows(run/'sensor_prior.csv')
            attitude = Rotation.from_quat([[r[k] for k in ('qx', 'qy', 'qz', 'qw')] for r in prior]).as_euler('xyz')
            params = loads_saved_json((run/'params.json').read_text())
            for key in ('save_path', 'sensor_prior_file', 'ply_path'):
                params.pop(key)
            invariants = {'binary': sha256(run/'binary_sha256.txt'), 'source': sha256(run/'source_input.json'),
                          'params': params, 'frozen_xyz': [[r[k] for k in ('timestamp_ns', 'x', 'y', 'z')] for r in prior]}
            if reference is None:
                reference = invariants, attitude[:, :2]
            elif invariants != reference[0] or not np.allclose(attitude[:, :2], reference[1], atol=1e-10, rtol=0):
                raise ValueError('Heading-only control conditions changed')
            item['runs'][mode] = str(run)
            (short/name).symlink_to(Path('../../')/run.name, target_is_directory=True)
        records.append(item); dump(experiment/'ablation_runs.json', records)
    dump(experiment/'short_verification.json', {'windows': len(records), 'methods': list(MODES),
        'total_poses': sum(3*r['frames'] for r in records), 'all_outputs_complete': True,
        'frozen_xyz_strings_roll_pitch_binary_clouds_solver_params': True, 'full_localization_started': False})


if __name__ == '__main__':
    main()
