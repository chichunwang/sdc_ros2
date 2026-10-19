#!/usr/bin/env python3
"""Position-only prior ablation with blocked GNSS validation; no score labels."""
from output_layout import loads_saved_json, saved_path

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from sensor_fusion import FusionConfig, position_rts
from three_camera_source import sha256


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def rows(path):
    with saved_path(path).open() as stream:
        return list(csv.DictReader(stream))


def export_bag(bag, output):
    from prepare_sensor_prior import load_bag
    gnss, imu = load_bag(bag)
    arrays = {'gnss_ns': np.array([r[0] for r in gnss], dtype=np.int64),
              'xyz': np.array([r[1] for r in gnss], dtype=np.float64),
              'imu_ns': np.array([r[0] for r in imu], dtype=np.int64),
              'quat': np.array([r[1] for r in imu], dtype=np.float64)}
    digest = hashlib.sha256()
    for key in ('gnss_ns', 'xyz', 'imu_ns', 'quat'):
        digest.update(arrays[key].astype('<i8' if key.endswith('_ns') else '<f8').tobytes())
    with Path(output).open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    dump(Path(output).with_suffix('.json'), {'bag': str(bag), 'bag_sha256': sha256(bag),
         'sensor_samples_sha256': digest.hexdigest(), 'integer_timestamps': True})


def predict_positions(stamps, xy, query_ns, training, config=None):
    """Both models see only training rows. Artificial holdout gaps are explicit."""
    config = config or FusionConfig()
    stamps, query_ns = np.asarray(stamps, dtype=np.int64), np.asarray(query_ns, dtype=np.int64)
    training = np.asarray(training, dtype=bool)
    if (len(stamps) != len(xy) or len(training) != len(stamps) or training.sum() < 5
            or np.any(np.diff(stamps) <= 0) or not np.isfinite(xy).all()):
        raise ValueError('Invalid blocked-validation measurements')
    origin = int(stamps[0])
    gt = (stamps[training]-origin)/1e9
    qt = (query_ns-origin)/1e9
    if qt.min() < gt[0] or qt.max() > gt[-1]:
        raise ValueError('Blocked validation must be bracketed; no endpoint extrapolation')
    measured = np.asarray(xy)[training]
    # A removed block breaks SG's equal-time sample assumption. Never apply
    # its five-sample stencil across that artificial gap: doing so can make
    # the baseline look much worse solely because of the validation protocol.
    indices = np.flatnonzero(training)
    breaks = np.flatnonzero(np.diff(indices) > 1)+1
    smooth = measured.copy()
    for group in np.split(np.arange(len(indices)), breaks):
        if len(group) >= 5:
            smooth[group] = savgol_filter(measured[group], 5, 2, axis=0, mode='interp')
    baseline = np.column_stack([np.interp(qt, gt, smooth[:, k]) for k in range(2)])
    motion, _, _, meta = position_rts(gt, measured, qt, config)
    assert meta['position_measurement_updates'] == int(training.sum())
    meta['sg_holdout_gap_policy'] = 'Filter original contiguous training segments separately, then interpolate across withheld interval'
    return baseline, motion, meta


def select_windows(stamps, xy, yaw, frames=48):
    """Choose up to two nonoverlapping windows/class using input motion only."""
    stamps = np.asarray(stamps, dtype=np.int64)
    t = (stamps-stamps[0])/1e9
    speed = np.linalg.norm(np.gradient(xy, t, axis=0), axis=1)
    turn = np.degrees(np.abs(np.gradient(np.unwrap(yaw), t)))
    masks = {'low_speed': speed <= 2., 'straight': (speed >= 5.) & (turn < 2.),
             'turning': (speed > 2.) & (turn >= 2.)}
    selected, unavailable = [], []
    half = frames//2
    for name, mask in masks.items():
        centers = [i for i in range(half, len(t)-half)
                   if t[i] >= 2. and t[-1]-t[i] >= 2.
                   and np.mean(mask[i-half:i-half+frames]) >= .8]
        accepted = []
        for fraction in (.25, .75):
            options = [i for i in centers if all(abs(t[i]-t[j]) >= 3. for j in accepted)
                       and all(abs(i-(w['start']+half)) >= frames for w in selected)]
            if not options:
                continue
            target = np.quantile(t[centers], fraction)
            center = min(options, key=lambda i: (abs(t[i]-target), i))
            accepted.append(center)
            selected.append({'class': name, 'split': 'design' if len(accepted) == 1 else 'holdout',
                'start': center-half, 'frames': frames, 'center': center,
                'center_ns': int(stamps[center]), 'center_s': float(t[center]),
                'speed_median_mps': float(np.median(speed[center-half:center-half+frames])),
                'turn_rate_median_deg_s': float(np.median(turn[center-half:center-half+frames]))})
        if not accepted:
            unavailable.append(name)
    return selected, unavailable


def blocked_validation(gnss_ns, xy, windows):
    output = []
    for window in windows:
        delta = (np.asarray(gnss_ns, dtype=np.int64)-window['center_ns'])/1e9
        validation = np.abs(delta) <= .375
        # Remove an additional .25 s around the validation interval.
        training = np.abs(delta) > .625
        if validation.sum() < 2:
            continue
        baseline, motion, meta = predict_positions(gnss_ns, xy, np.asarray(gnss_ns)[validation], training)
        truth = np.asarray(xy)[validation]  # Noisy held-out sensor readings, NOT truth poses.
        output.append({**window, 'validation_timestamps_ns': np.asarray(gnss_ns)[validation].tolist(),
            'training_rows': int(training.sum()), 'validation_rows': int(validation.sum()),
            'guard_rows': int((~training & ~validation).sum()),
            'heading_rts_prediction_rmse_m': float(np.sqrt(np.mean(np.sum((baseline-truth)**2, axis=1)))),
            'motion_rts_prediction_rmse_m': float(np.sqrt(np.mean(np.sum((motion-truth)**2, axis=1)))),
            'position_measurement_updates': meta['position_measurement_updates']})
    return output


def fixed_time_validation(stamps, xy, track):
    origin=int(stamps[0]);time=(np.asarray(stamps,dtype=np.int64)-origin)/1e9
    windows=[{'class':'fixed_time_grid','split':'design' if i%2==0 else 'holdout',
              'center_ns':origin+int(t*1e9),'center_s':float(t)}
             for i,t in enumerate(np.linspace(time[0]+3,time[-1]-3,12))]
    return {'track':track,'selection':'12 equally spaced time blocks, fixed default parameters, no outcome selection',
            'blocks':blocked_validation(stamps,xy,windows)}


def prepare_track(raw_path, prior_run, output, track):
    output.mkdir(parents=True)
    prior = rows(prior_run/'sensor_prior.csv')
    frame_ns = np.array([int(r['timestamp_ns']) for r in prior], dtype=np.int64)
    config = FusionConfig()
    original_meta = loads_saved_json((prior_run/'sensor_prior.json').read_text())
    raw_meta = loads_saved_json(raw_path.with_suffix('.json').read_text())
    if original_meta['sensor_samples_sha256'] != raw_meta['sensor_samples_sha256']:
        raise ValueError('Raw sensor values differ from original scored inputs')
    with np.load(raw_path) as raw:
        stamps, xyz = raw['gnss_ns'], raw['xyz']
    origin = int(stamps[0])
    positions, velocity, std, notes = position_rts((stamps-origin)/1e9, xyz[:, :2], (frame_ns-origin)/1e9, config)
    quats = np.array([[float(r[k]) for k in ('qx','qy','qz','qw')] for r in prior])
    baseline_xy = np.array([[float(r[k]) for k in ('x','y')] for r in prior])
    windows, unavailable = select_windows(frame_ns, baseline_xy, Rotation.from_quat(quats).as_euler('xyz')[:, 2])
    validation = blocked_validation(stamps, xyz[:, :2], windows)
    # Freeze Z and every quaternion string, not merely numerically close values.
    motion_prior = copy.deepcopy(prior)
    for row, position in zip(motion_prior, positions):
        row.update(x=repr(float(position[0])), y=repr(float(position[1])))
    with (output/'sensor_prior.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prior[0])); writer.writeheader(); writer.writerows(motion_prior)
    diagnostics = rows(prior_run/'sensor_prior.fusion.csv')
    for i, row in enumerate(diagnostics):
        row.update(vx_mps=velocity[i, 0], vy_mps=velocity[i, 1],
                   x_model_std_m=std[i, 0], y_model_std_m=std[i, 1],
                   xy_change_from_fixed_m=float(np.linalg.norm(positions[i]-baseline_xy[i])))
    with (output/'sensor_prior.fusion.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0])); writer.writeheader(); writer.writerows(diagnostics)
    meta = copy.deepcopy(original_meta)
    meta['sensor_fusion'].update(mode='motion_rts', **notes,
        position_method='raw GNSS constant-velocity Kalman + RTS; original heading/Z frozen for position-only ablation',
        max_xy_change_from_fixed_m=float(np.linalg.norm(positions-baseline_xy, axis=1).max()))
    meta['position_only_ablation'] = {'original_prior_run': str(prior_run.resolve()),
        'original_prior_sha256': sha256(prior_run/'sensor_prior.csv'), 'quaternion_and_z_strings_unchanged': True,
        'raw_sensor_samples_sha256': raw_meta['sensor_samples_sha256']}
    dump(output/'sensor_prior.json', meta)
    dump(output/'validation.json', {'track': track, 'profile': 'sensor_guided', 'sensor_fusion': 'motion_rts',
         'frames': len(prior), 'scope': 'Prepared prior values only; no localization or accuracy validation'})
    report = {'track': track, 'windows': windows, 'unavailable_classes': unavailable, 'blocked_validation': validation,
        'xy_change_rms_m': float(np.sqrt(np.mean(np.sum((positions-baseline_xy)**2, axis=1)))),
        'xy_change_max_m': float(np.linalg.norm(positions-baseline_xy, axis=1).max()),
        'frozen': ['original front source', 'source scale 3', 'all original quaternion and Z strings', 'carry', 'directional'],
        'fixed_config': config.__dict__, 'model_parameters_tuned_on_holdout': False,
        'limits': ['Held-out GNSS is noisy input, not ground truth. Common-mode bias cannot be identified.',
                   'Artificial holdout gaps bypass the production coverage gate only in this validation function.',
                   'SG never filters across the removed interval; each contiguous training segment is filtered separately.',
                   'Windows selected from input motion, before computing either method residuals.']}
    dump(output/'ablation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    export = sub.add_parser('export')
    export.add_argument('--bag', type=saved_path, required=True); export.add_argument('--output', type=saved_path, required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--output', type=saved_path, required=True)
    for track in (1, 2):
        prepare.add_argument(f'--raw-track{track}', type=saved_path, required=True)
        prepare.add_argument(f'--prior-track{track}', type=saved_path, required=True)
    args = parser.parse_args()
    if args.mode == 'export':
        export_bag(args.bag, args.output); return
    reports = []
    for track in (1, 2):
        report = prepare_track(getattr(args, f'raw_track{track}'), getattr(args, f'prior_track{track}'), args.output/f'track{track}_motion_prior', track)
        reports.append(report)
        print(json.dumps({'track': track, 'windows': report['windows'], 'blocked_validation': report['blocked_validation']}, indent=2), flush=True)
    dump(args.output/'prior_ablation.json', reports)
    fixed=[]
    for track in (1,2):
        with np.load(getattr(args,f'raw_track{track}')) as raw:
            fixed.append(fixed_time_validation(raw['gnss_ns'],raw['xyz'][:,:2],track))
    dump(args.output/'fixed_time_block_validation.json',fixed)


if __name__ == '__main__':
    main()
