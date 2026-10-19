#!/usr/bin/env python3
"""用固定 rig baseline 三角化的稀疏點，估共享深度尺度；安裝平移不縮放。"""
from output_layout import loads_saved_json, saved_path
import argparse
import json
from pathlib import Path
import numpy as np
from calibrated_rig_geometry import (OPTICAL_TO_FLU, backproject, transform, voxel_indices,
                                     balanced_indices, write_ply)
from three_camera_source import sha256


def depth_scale_consensus(records):
    """Equal group weight in log space; no per-Track fit or pose/height adjustment."""
    training = [r for r in records if r['role'] == 'fit' and len(r['log_depth_ratios']) >= 3]
    if len(training) < 3 or {r['track'] for r in training} != {1, 2}:
        raise ValueError('Insufficient independent fit groups across both Tracks')
    centers = [np.median(r['log_depth_ratios']) for r in training]
    scale = float(np.exp(-np.median(centers)))
    if not .25 <= scale <= 4.:
        raise ValueError('Unphysical/unsupported depth scale')
    return scale


def anchor_records(reconstruction):
    records = []
    for group in sorted(reconstruction.glob('track[12]_*')):
        report = loads_saved_json((group / 'report.json').read_text())
        front = np.load(group / 'rig_0.npz')
        logs, details = [], []
        for tag, camera_id in (('fl', 1), ('fr', 2)):
            check = np.load(group / f'validation_{tag}.npz')
            side = np.load(group / f'rig_{camera_id}.npz')
            good = check['triangulation_eligible'] & (check['angle_deg'] >= 2.) & (check['motion_error_px'] <= 2.)
            xyz = check['triangulated_front_xyz']
            side_xyz = transform(xyz, np.linalg.inv(side['applied_camera_pose']))
            for j in np.flatnonzero(good):
                u, v = np.rint(check['pixels_front'][j, :2]).astype(int)
                su, sv = np.rint(check['pixels_side'][j, :2]).astype(int)
                if not (front['mask'][v, u] and side['mask'][sv, su]):
                    continue
                ratios = np.array([front['depth_z'][v, u] / xyz[j, 2], side['depth_z'][sv, su] / side_xyz[j, 2]])
                if np.any(~np.isfinite(ratios)) or np.any(ratios <= 0) or abs(np.log(ratios[0] / ratios[1])) > np.log(1.25):
                    continue
                log_ratio = float(np.mean(np.log(ratios)))
                logs.append(log_ratio)
                details.append({'camera': tag, 'match_index': int(j), 'angle_deg': float(check['angle_deg'][j]),
                                'sampson_px': float(check['motion_error_px'][j]), 'depth_ratios': ratios.tolist()})
        # The adjacent second frame is never used to fit the shared scale.
        records.append({'track': report['track'], 'original_idx': report['original_idx'],
                        'role': 'fit' if report['original_idx'] % 2 else 'holdout',
                        'log_depth_ratios': logs, 'anchors': details})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reconstruction', type=saved_path, required=True)
    parser.add_argument('--output', type=saved_path, required=True)
    parser.add_argument('--budget', type=int, default=6000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.resolve().relative_to((root / 'outputs').resolve())
    if not (args.reconstruction / 'completion.json').is_file():
        parser.error('Original reconstruction incomplete')
    args.output.mkdir(parents=True, exist_ok=False)
    records = anchor_records(args.reconstruction)
    scale = depth_scale_consensus(records)
    for r in records:
        values = np.exp(r['log_depth_ratios'])
        r['before_median_depth_ratio'] = float(np.median(values)) if len(values) else None
        r['after_median_depth_ratio'] = float(np.median(values * scale)) if len(values) else None
    scale_report = {'depth_scale': scale, 'estimator': 'Median of per-fit-group medians of log(predicted/stereo depth); shared across both Tracks',
                    'thresholds': {'minimum_parallax_deg': 2., 'maximum_sampson_px': 2., 'maximum_front_side_depth_ratio': 1.25},
                    'extrinsic_translation_scaled': False, 'fit_uses_map_or_scores': False,
                    'scope': 'Calibrated sparse stereo consistency, not ground truth; moving objects, fisheye calibration bias and frame/clock assumptions remain',
                    'groups': records, 'image_validation_sha256': sha256(args.reconstruction / 'image_geometry_validation.json')}
    (args.output / 'scale_anchor.json').write_text(json.dumps(scale_report, indent=2, allow_nan=False) + '\n')
    manifests = {}
    for group in sorted(args.reconstruction.glob('track[12]_*')):
        report = loads_saved_json((group / 'report.json').read_text())
        track, index, stamp = report['track'], report['original_idx'], report['timestamp_ns']
        raw_rig = loads_saved_json((args.reconstruction / f'rig_track{track}.json').read_text())['fixed_optical_front_from_camera']
        points, static, colors, ids, confidence = [], [], [], [], []
        for camera_id, tag in enumerate(('f', 'fl', 'fr')):
            saved = np.load(group / f'rig_{camera_id}.npz')
            cam = backproject(saved['depth_z'] * scale, saved['intrinsics'])
            p = transform(cam, saved['applied_camera_pose']) @ OPTICAL_TO_FLU.T
            q = transform(cam, np.array(raw_rig[tag])) @ OPTICAL_TO_FLU.T
            mask = saved['mask']  # identical pixels; scale is the only reconstruction change
            points.append(transform(p, saved['bag_from_header_flu'])[mask])
            static.append(transform(q, saved['bag_from_header_flu'])[mask])
            import cv2
            rgb = cv2.cvtColor(cv2.imread(str(group / f'{tag}_rectified.jpg')), cv2.COLOR_BGR2RGB)
            colors.append(rgb[mask]); ids.append(np.full(mask.sum(), camera_id, dtype=np.uint8)); confidence.append(saved['confidence'][mask])
        points, static, colors, ids, confidence = map(np.concatenate, (points, static, colors, ids, confidence))
        uniform = voxel_indices(points)
        if len(uniform) > args.budget:
            uniform = uniform[np.linspace(0, len(uniform) - 1, args.budget, dtype=int)]
        balanced, sampling_report = balanced_indices(points, ids, confidence, args.budget)
        matched = voxel_indices(points)
        matched = matched[np.linspace(0, len(matched) - 1, min(len(matched), len(balanced)), dtype=int)]
        front = np.flatnonzero(ids == 0)
        front = front[voxel_indices(points[front])]
        if len(front) > args.budget:
            front = front[np.linspace(0, len(front) - 1, args.budget, dtype=int)]
        selections = {'anchored': (uniform, None), 'anchored_static': (uniform, None), 'anchored_front': (front, None),
                      'anchored_uniform_matched': (matched, None), 'anchored_balanced': (balanced, sampling_report)}
        for variant, (selected, sampling) in selections.items():
            directory = args.output / f'{variant}_track{track}'
            directory.mkdir(exist_ok=True)
            name = f'{index:06d}_ts_{stamp}.ply'
            write_ply(directory / name, (static if variant == 'anchored_static' else points)[selected], colors[selected])
            np.savez_compressed(directory / f'{index:06d}_labels.npz', camera_ids=ids[selected])
            key = variant, track
            manifests.setdefault(key, {'schema': 'calibrated_rig_v1', 'track': track, 'variant': variant,
                                       'coordinate_frame': 'front_camera_FLU', 'reference_time': 'front_camera_bag_timestamp',
                                       'ply_scale': 1., 'experimental': True, 'fixed_rig_heights': True,
                                       'per_camera_motion_compensation': variant != 'anchored_static',
                                       'depth_scale': scale, 'scale_anchor_sha256': sha256(args.output / 'scale_anchor.json'),
                                       'source_runtime_sha256': sha256(args.reconstruction / 'runtime.json'), 'frames': []})
            manifests[key]['frames'].append({'original_idx': index, 'timestamp_ns': stamp, 'filename': name,
                                            'sha256': sha256(directory / name), 'points': len(selected),
                                            'camera_counts': {str(i): int(np.sum(ids[selected] == i)) for i in range(3)}, 'sampling': sampling})
    for (variant, track), report in manifests.items():
        (args.output / f'{variant}_track{track}/report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (args.output / 'completion.json').write_text(json.dumps({'source': str(args.reconstruction.resolve()), 'depth_scale': scale, 'groups': len(records)}) + '\n')
    print(json.dumps({'depth_scale': scale, 'groups': [{k: r[k] for k in ('track', 'original_idx', 'role', 'before_median_depth_ratio', 'after_median_depth_ratio')} for r in records]}, indent=2))


if __name__ == '__main__':
    main()
