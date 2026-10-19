#!/usr/bin/env python3
"""Evidence audit; never infer an antenna/body transform from topic names or score."""
from output_layout import loads_saved_json, saved_path
import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from prior_ablation import dump
from three_camera_source import sha256


def wrap(value):
    return np.arctan2(np.sin(value), np.cos(value))


def lag_diagnostic(gt, supplied, it, imu):
    """Fit offset+lag on alternating time blocks; report only, no synchronization change."""
    bins = np.floor((gt-gt[0])/8).astype(int)
    interior = (gt > it[0]+.6) & (gt < it[-1]-.6)
    design, holdout = interior & (bins % 2 == 0), interior & (bins % 2 == 1)
    def residual(x, mask):
        return wrap(supplied[mask]-np.interp(gt[mask]+x[0], it, imu)-x[1])
    fit = least_squares(lambda x: residual(x, design), [0., np.median(wrap(supplied[design]-np.interp(gt[design], it, imu)))],
                        bounds=([-.5, -np.pi], [.5, np.pi]), loss='huber', f_scale=np.deg2rad(2))
    offset_only = float(np.median(wrap(supplied[design]-np.interp(gt[design], it, imu))))
    values = np.linalg.svd(fit.jac, compute_uv=False)
    def rms(v): return float(np.degrees(np.sqrt(np.mean(v**2))))
    return {'lag_s': float(fit.x[0]), 'offset_deg': float(np.degrees(fit.x[1])),
        'design_samples': int(design.sum()), 'holdout_samples': int(holdout.sum()),
        'design_rms_deg': rms(residual(fit.x, design)), 'holdout_rms_deg': rms(residual(fit.x, holdout)),
        'holdout_offset_only_rms_deg': rms(residual([0, offset_only], holdout)),
        'jacobian_singular_values': values.tolist(), 'at_boundary': bool(np.any(fit.active_mask)),
        'applied': False, 'source_verified': False,
        'scope': 'Lag/heading-offset fit between noisy possibly correlated orientations; filter delay, noise and convention can mimic hardware lag.'}


def audit(raw_path, data_root, track, official_pdf=None):
    with np.load(raw_path) as data: raw = {key: data[key] for key in data.files}
    origin = int(raw['gnss_bag_ns'][0]); gt, it = [(raw[k]-origin)/1e9 for k in ('gnss_bag_ns', 'imu_bag_ns')]
    bag = data_root/f'Track{track}/Track{track}.db3'
    with sqlite3.connect(f'file:{bag.resolve()}?mode=ro', uri=True) as connection:
        topics = [{'name': name, 'type': kind} for name, kind in connection.execute('SELECT name,type FROM topics')]
    calibration = data_root/f'Track{track}/Track{track}_images/camera_info/lucid_camera_extrinsics.yaml'
    rig = yaml.safe_load(calibration.read_text())['transforms']; checks = {}
    for name, item in rig.items():
        transform = np.array(item['matrix'], float)
        checks[name] = {'determinant': float(np.linalg.det(transform[:3, :3])),
            'orthogonality_error': float(np.linalg.norm(transform[:3, :3].T@transform[:3, :3]-np.eye(3))),
            'homogeneous_bottom_row_valid': bool(np.allclose(transform[3], [0, 0, 0, 1]))}
    closure = {}
    for label in ('front_left', 'front_right'):
        a, b = [np.array(rig[name]['matrix']) for name in (f'T_front_{label}', f'T_{label}_front')]
        closure[label] = float(np.linalg.norm(a@b-np.eye(4)))
    provided = np.unwrap(Rotation.from_quat(raw['gnss_quat']).as_euler('xyz')[:, 2])
    imu = np.unwrap(Rotation.from_quat(raw['imu_quat']).as_euler('xyz')[:, 2])
    time = {}
    for prefix in ('gnss', 'imu'):
        delta = (raw[prefix+'_bag_ns']-raw[prefix+'_header_ns'])/1e6
        time[prefix] = {'bag_minus_header_median_ms': float(np.median(delta)),
            'bag_minus_header_min_max_ms': [float(delta.min()), float(delta.max())],
            'bag_strictly_increasing': bool(np.all(np.diff(raw[prefix+'_bag_ns']) > 0)),
            'header_strictly_increasing': bool(np.all(np.diff(raw[prefix+'_header_ns']) > 0))}
    export = loads_saved_json(raw_path.with_suffix('.json').read_text())
    if export['bag_sha256'] != sha256(bag) or export['raw_values_sha256'] != sha256(raw_path):
        raise ValueError('Raw export does not match current bag/hash')
    return {'track': track, 'source_bag_sha256': export['bag_sha256'], 'raw_sha256': sha256(raw_path),
        'topics': topics, 'frame_ids': export['frames'], 'tf_static_present': any(t['name'] == '/tf_static' for t in topics),
        'camera_camera_transform_checks': checks, 'camera_inverse_closure_errors': closure,
        'camera_calibration_sha256': sha256(calibration),
        'camera_camera_rig_valid': all(abs(c['determinant']-1)<1e-6 and c['orthogonality_error']<1e-6 for c in checks.values()) and max(closure.values())<1e-6,
        'time': time, 'orientation_lag_diagnostic': lag_diagnostic(gt, provided, it, imu),
        'gnss_z_distinct_values': len(np.unique(raw['gnss_xyz'][:, 2])),
        'unresolved': ['GNSS antenna/body/map datum and origin definition', 'camera-source to body/IMU rigid transform',
                       'publisher/provenance and independence of GNSS orientation', 'hardware acquisition/filter latency',
                       'official submission vehicle reference point'],
        'official_pdf_sha256': sha256(official_pdf) if official_pdf else None,
        'official_statement': 'Task provides noisy init guess and camera-camera calibration; neither is a measured body/IMU extrinsic.',
        'decisions': {'extra_extrinsic_applied': False, 'time_shift_applied_s': 0.,
                      'gnss_orientation_used_as_independent_factor': False,
                      'accelerometer_preintegration_enabled': False,
                      'output_frame': 'same source-to-map convention as scored baseline; not newly certified vehicle/body origin',
                      'gyro_model': 'conditional baseline-compatible yaw-rate model; not verified full 3D IMU/body calibration'},
        'limits': ['Header/bag agreement cannot establish sensor internal synchronization.',
                   'Local data cannot recover unknown absolute body/GNSS lever arms without sufficient independent reference/excitation.',
                   'No calibration is inferred from noisy init_guess, map fit or leaderboard score.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-directory', type=saved_path, required=True)
    parser.add_argument('--data-root', type=saved_path, required=True)
    parser.add_argument('--output', type=saved_path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve the previous contract audit')
    pdf = args.data_root.parent/'docs/SDC 2026 Midterm01 Competition -  Localization.pdf'
    reports = [audit(args.raw_directory/f'input_track{track}.npz', args.data_root, track, pdf) for track in (1, 2)]
    dump(args.output, reports)
    print(json.dumps([{'track': r['track'], 'rig_valid': r['camera_camera_rig_valid'],
                      'lag': r['orientation_lag_diagnostic']} for r in reports], indent=2))


if __name__ == '__main__': main()
