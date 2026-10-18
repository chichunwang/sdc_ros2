#!/usr/bin/env python3
"""三相機尺度／時間診斷；使用原始感測器與 map，不計算 ATE 或 Kaggle 分數。"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from prepare_sensor_prior import load_bag
from three_camera_source import load_index, camera_headers, resolve_camera_timestamps, sha256, CAMERAS
from three_camera_calibration import load_ply, load_binary_map, SurfaceMap, fit_scale


def stats(values):
    values = np.asarray(values, dtype=float)
    return {'min': float(np.min(values)), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'max': float(np.max(values))}


def synchronize(data_root, track, rows, output):
    bag = data_root / f'Track{track}/Track{track}.db3'
    headers = camera_headers(bag)
    resolved = {tag: resolve_camera_timestamps(headers[tag], [int(r[f'{tag}_ts']) for r in rows]) for tag in CAMERAS}
    gnss, imu = load_bag(bag)
    origin = gnss[0][0]
    gt = (np.array([r[0] for r in gnss], dtype=np.int64) - origin) / 1e9
    xyz = savgol_filter(np.array([r[1] for r in gnss]), 5, 2, axis=0)
    it = (np.array([r[0] for r in imu], dtype=np.int64) - origin) / 1e9
    yaw = np.unwrap(Rotation.from_quat([r[1] for r in imu]).as_euler('xyz')[:, 2])
    records = []
    for i, r in enumerate(rows):
        record = {'original_idx': int(r['index'])}
        for tag in CAMERAS:
            bag_ns, header_ns, frame, basis = resolved[tag][i]
            record[f'{tag}_index_time_basis'] = basis
            record[f'{tag}_bag_ns'] = bag_ns
            record[f'{tag}_header_ns'] = header_ns
            record[f'{tag}_bag_minus_header_ns'] = bag_ns - header_ns
            record[f'{tag}_frame_id'] = frame
        times = np.array([(record[f'{tag}_header_ns'] - origin) / 1e9 for tag in CAMERAS])
        positions = np.column_stack([np.interp(times, gt, xyz[:, axis]) for axis in range(3)])
        angles = np.interp(times, it, yaw)
        record['sensor_edge_clamp'] = int(np.any((times < gt[0]) | (times > gt[-1]) | (times < it[0]) | (times > it[-1])))
        for i, tag in enumerate(('fl', 'fr'), 1):
            record[f'{tag}_minus_front_bag_ns'] = record[f'{tag}_bag_ns'] - record['f_bag_ns']
            record[f'{tag}_minus_front_header_ns'] = record[f'{tag}_header_ns'] - record['f_header_ns']
            record[f'{tag}_translation_sensitivity_m'] = float(np.linalg.norm(positions[i] - positions[0]))
            record[f'{tag}_yaw_sensitivity_deg'] = float(np.rad2deg(angles[i] - angles[0]))
        records.append(record)
    with (output / 'synchronization.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {'groups': len(records), 'reference_time': 'front_camera_bag_timestamp',
              'index_to_message_exact_match': True, 'per_camera_deskew_applied': False,
              'deskew_status': 'unavailable for already merged XYZRGB: no point camera identity or original per-view pose',
              'header_semantics': 'message capture timestamp candidate, not independently verified exposure/hardware clock',
              'sensor_edge_clamped_groups': sum(r['sensor_edge_clamp'] for r in records),
              'bag_bytes': bag.stat().st_size, 'cameras': {}}
    for tag in CAMERAS:
        d = {'bag_minus_header_ms': stats([r[f'{tag}_bag_minus_header_ns'] / 1e6 for r in records]),
             'frame_ids': sorted({r[f'{tag}_frame_id'] for r in records}),
             'index_time_basis_counts': {basis: sum(r[f'{tag}_index_time_basis'] == basis for r in records)
                                        for basis in ('bag', 'header')}}
        if tag != 'f':
            d.update(header_minus_front_ms=stats([r[f'{tag}_minus_front_header_ns'] / 1e6 for r in records]),
                     bag_minus_front_ms=stats([r[f'{tag}_minus_front_bag_ns'] / 1e6 for r in records]),
                     translation_sensitivity_m=stats([r[f'{tag}_translation_sensitivity_m'] for r in records]),
                     yaw_sensitivity_deg=stats([r[f'{tag}_yaw_sensitivity_deg'] for r in records]))
        report['cameras'][tag] = d
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/root/data'))
    parser.add_argument('--track', type=int, choices=(1, 2), required=True)
    parser.add_argument('--baseline-run', type=Path, required=True, help='Original full heading_rts run, sensor_prior only')
    parser.add_argument('--map-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fit-indices', default='500,1000,1600,2200')
    parser.add_argument('--holdout-indices', default='750,1300,1900,2600')
    parser.add_argument('--reference-scale', type=float, help='Additional held-out evaluation of scale fitted on the other Track')
    args = parser.parse_args()
    args.output.resolve().relative_to(Path('/root/outputs').resolve())
    args.output.mkdir(parents=True)
    metadata = json.loads((args.baseline_run / 'validation.json').read_text())
    if metadata['track'] != args.track or metadata.get('sensor_fusion') != 'heading_rts':
        parser.error('Wrong baseline Track/fusion')
    frames = list(csv.DictReader((args.baseline_run / 'frames.csv').open()))
    priors = list(csv.DictReader((args.baseline_run / 'sensor_prior.csv').open()))
    stamps = [int(r['timestamp_ns']) for r in frames]
    if stamps != [int(r['timestamp_ns']) for r in priors]:
        raise ValueError('Prior/frame time mismatch')
    root, index, contract = load_index(args.data_root, args.track, stamps)
    synchronization = synchronize(args.data_root, args.track, index, args.output)
    (args.output / 'synchronization.json').write_text(json.dumps(synchronization, indent=2) + '\n')
    identity = json.loads(args.map_cache.with_suffix('.pcd.json').read_text())
    actual_map_hash = sha256(args.data_root / f'Track{args.track}/map.pcd')
    if identity['track'] != args.track or identity['map_voxel_leaf_size'] != 1 or identity['map_sha256'] != actual_map_hash:
        raise ValueError('Map cache identity mismatch')
    surface_map = SurfaceMap(load_binary_map(args.map_cache))
    fit_indices = list(map(int, args.fit_indices.split(',')))
    holdout_indices = list(map(int, args.holdout_indices.split(',')))
    if set(fit_indices) & set(holdout_indices) or len(set(fit_indices + holdout_indices)) != len(fit_indices + holdout_indices):
        raise ValueError('Fit/holdout frames must be disjoint unique sets')
    if min(fit_indices + holdout_indices) < 0 or max(fit_indices + holdout_indices) >= len(stamps):
        raise ValueError('Frame index outside source')
    selected, provenance = {}, []
    for i in fit_indices + holdout_indices:
        path = root / index[i]['ply']
        p = priors[i]
        prior = np.eye(4)
        prior[:3, :3] = Rotation.from_quat([float(p[k]) for k in ('qx', 'qy', 'qz', 'qw')]).as_matrix()
        prior[:3, 3] = [float(p[k]) for k in ('x', 'y', 'z')]
        selected[i] = (load_ply(path), prior)
        provenance.append({'original_idx': i, 'timestamp_ns': stamps[i], 'source_sha256': sha256(path),
                           'role': 'fit' if i in fit_indices else 'holdout'})
    training = [selected[i] for i in fit_indices]
    heldout = [selected[i] for i in holdout_indices]
    fitted = fit_scale(surface_map, training)
    print(json.dumps({'track': args.track, 'fitted_scale': fitted['scale']}, ensure_ascii=False), flush=True)
    validation = {'scale_1': fit_scale(surface_map, heldout, fixed_scale=1.),
                  'fitted_scale': fit_scale(surface_map, heldout, fixed_scale=fitted['scale'])}
    if args.reference_scale is not None:
        validation['other_track_scale'] = fit_scale(surface_map, heldout, fixed_scale=args.reference_scale)
    identified = sum(f['scale_information_after_free_pose'] > .1 and f['scale_information_fraction'] > .01
                     for f in fitted['frames'])
    report = {'algorithm_version': 1, 'track': args.track, 'source_contract': contract,
              'scale_parameterization': 'one shared exp(log_scale); per-frame XYZ/yaw nuisance with existing bounds',
              'fit_indices': fit_indices, 'holdout_indices': holdout_indices,
              'fit': fitted, 'holdout': validation, 'data_identifiable_fit_frames': identified,
              'scale_status': 'experimental_map_consistency_only',
              'production_default_changed': False,
              'synchronization': synchronization, 'selected_sources': provenance,
              'map_cache_sha256': sha256(args.map_cache), 'map_identity': identity,
              'prior_sha256': sha256(args.baseline_run / 'sensor_prior.csv'),
              'baseline_run': str(args.baseline_run), 'command': sys.argv,
              'scope': 'No ground truth, score, per-camera deskew or claim of validated global scale'}
    source_snapshot = args.output / 'sources'
    source_snapshot.mkdir()
    for name in ('three_camera_lab.py', 'three_camera_calibration.py', 'three_camera_source.py'):
        shutil.copy2(Path(__file__).with_name(name), source_snapshot / name)
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report': str(args.output / 'report.json'), 'fitted_scale': fitted['scale'],
                      'identifiable_fit_frames': identified,
                      'holdout_costs': {k: v['cost'] for k, v in validation.items()}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
