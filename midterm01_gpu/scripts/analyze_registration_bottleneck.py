#!/usr/bin/env python3
"""Read existing scored outputs and raw IMU exports; never run localization."""
from output_layout import saved_path, scored_reference_run
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from prior_ablation import rows
from three_camera_source import sha256


def statistics(value):
    value = np.asarray(value, float)
    return {'rms': float(np.sqrt(np.mean(value**2))), 'median': float(np.median(value)),
            'p95': float(np.quantile(value, .95)), 'max': float(value.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=saved_path, required=True)
    parser.add_argument('--raw-directory', type=saved_path, required=True)
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    result = []
    for track in (1, 2):
        run = scored_reference_run(root,track)
        guidance, geometry = rows(run/'guidance.csv'), rows(run/'geometry.csv')
        xy = np.hypot([float(r['dx']) for r in guidance], [float(r['dy']) for r in guidance])
        yaw = np.abs(np.degrees([float(r['dyaw']) for r in guidance]))
        rank_one = [r for r in geometry if r['observable_rank'] == '1']
        dominant_components = np.array([[float(r[f'scaled_eigenvector_r{i}c3'])**2 for i in range(4)]
                                       for r in rank_one])
        raw = args.raw_directory/f'input_track{track}.npz'
        with np.load(raw) as values:
            data = {key: values[key] for key in values.files}
        acc, quat = data['linear_acceleration'], data['imu_quat']
        # This is a convention hypothesis, not an extrinsic/gravity calibration.
        rotated_acc = Rotation.from_quat(quat).apply(acc)
        result.append({'track': track, 'scored_run': str(run), 'frames': len(guidance),
            'input_sha256': {f: sha256(run/f) for f in ('guidance.csv', 'geometry.csv', 'params.json')},
            'xy_correction_m': statistics(xy), 'yaw_correction_abs_deg': statistics(yaw),
            'xy_at_95percent_1_5m_bound': int(sum(xy >= 1.425)),
            'yaw_at_95percent_8deg_bound': int(sum(yaw >= 7.6)),
            'iterations_histogram': dict(Counter(r['iterations'] for r in guidance)),
            'update_status_histogram': dict(Counter(r['update_status'] for r in guidance)),
            'data_normal_rank_histogram': dict(Counter(r['observable_rank'] for r in geometry)),
            'rank_one_fraction': float(np.mean([int(r['observable_rank']) == 1 for r in geometry])),
            'rank_one_dominant_scaled_eigenvector_component_squared_median_xyzyaw': np.median(dominant_components, axis=0).tolist(),
            'rank_one_dominant_vector_z_squared_above_0_8_fraction': float(np.mean(dominant_components[:, 2] > .8)),
            'median_unique_targets': float(np.median([int(r['unique_targets']) for r in geometry])),
            'median_reliable_points': float(np.median([int(r['reliable_points']) for r in geometry])),
            'raw_export_sha256': sha256(raw),
            'imu_angular_velocity_available': bool(np.isfinite(data['angular_velocity']).all()),
            'raw_acceleration_magnitude_mps2': statistics(np.linalg.norm(acc, axis=1)),
            'if_quaternion_body_to_world_mean_rotated_acceleration_mps2': rotated_acc.mean(axis=0).tolist(),
            'all_covariances_zero': all(np.all(data[k] == 0) for k in
                ('orientation_covariance', 'angular_velocity_covariance', 'linear_acceleration_covariance'))})
    report = {'tracks': result, 'new_localization_started': False,
        'limits': ['Correction magnitude is not pose error; total score does not separate translation and rotation.',
                   'Rank uses the existing 5%-of-largest/1e-4 diagnostic threshold, not calibrated observability.',
                   'Geometry information excludes prior, tangential floor and normal gain; it is not posterior covariance.',
                   'Quaternion-rotated acceleration assumes body-to-world convention and is not extrinsic calibration.',
                   'All-zero covariance means unavailable noise values, not zero measurement noise.']}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
