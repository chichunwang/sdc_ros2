#!/usr/bin/env python3
"""分離場景的三相機偏差驗證；以跨時間剛體一致性篩除可疑尺度錨點。"""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
from calibrated_rig_geometry import PriorTrajectory, rectify, transform
from validate_rig_reconstruction import epipolar_error, summary
from three_camera_source import camera_headers, resolve_camera_timestamps, sha256


def mutual_matches(a, b, ratio=.8):
    if a is None or b is None or min(len(a), len(b)) < 2:
        return []
    matcher = cv2.BFMatcher()
    def accepted(first, second):
        return {pair[0].queryIdx: pair[0].trainIdx for pair in matcher.knnMatch(first, second, k=2)
                if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance}
    f, r = accepted(a, b), accepted(b, a)
    return [(i, j) for i, j in f.items() if r.get(j) == i]


def spatially_unique(pairs, first, second):
    """SIFT may emit several orientations at one location; count one geometric anchor."""
    seen_first, seen_second, result = set(), set(), []
    for i, j in pairs:
        a = tuple(np.rint(first[i].pt).astype(int))
        b = tuple(np.rint(second[j].pt).astype(int))
        if a not in seen_first and b not in seen_second:
            result.append((i,j));seen_first.add(a);seen_second.add(b)
    return result


def triangulate(x1, x2, k1, k2, reference_from_camera):
    other_from_reference = np.linalg.inv(reference_from_camera)
    h = cv2.triangulatePoints(k1 @ np.eye(4)[:3], k2 @ other_from_reference[:3], x1[:, :2].T, x2[:, :2].T).T
    with np.errstate(divide='ignore', invalid='ignore'):
        p = h[:, :3] / h[:, 3:4]
        a, b = p, p - reference_from_camera[:3, 3]
        angle = np.rad2deg(np.arccos(np.clip(np.sum(a*b, axis=1) / (np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)), -1, 1)))
    other = transform(p, other_from_reference)
    eligible = (np.all(np.isfinite(p), axis=1) & (p[:, 2] > 0) & (other[:, 2] > 0)
                & (np.linalg.norm(p, axis=1) > 2) & (np.linalg.norm(p, axis=1) < 60) & (angle >= 2))
    return p, other, angle, eligible


def project(points, k):
    pixels = points @ k.T
    with np.errstate(divide='ignore', invalid='ignore'):
        return pixels[:, :2] / pixels[:, 2:3]


def estimate(records, field):
    fit = [r for r in records if r['role'] == 'fit' and len(r[field]) >= 3]
    centers = [np.median(r[field]) for r in fit]
    if len(fit) < 3 or {r['track'] for r in fit} != {1, 2}:
        return None
    scale = float(np.exp(-np.median(centers)))
    rng = np.random.default_rng(0)
    boot = np.exp(-np.median(rng.choice(centers, size=(2000, len(centers)), replace=True), axis=1))
    held = [r for r in records if r['role'] == 'holdout' and len(r[field]) >= 3]
    errors = [abs(float(np.exp(np.median(r[field])) * scale - 1)) for r in held]
    # Declared engineering checks, not ATE or calibrated sensor accuracy.
    coverage = len(fit) >= 6 and len(held) >= 6
    coverage &= all(sum(r['track'] == t for r in fit) >= 3 and sum(r['track'] == t for r in held) >= 3 for t in (1, 2))
    stable = bool(errors and np.median(errors) <= .1 and np.percentile(errors, 90) <= .2)
    return {'depth_scale': scale, 'fit_groups': len(fit), 'holdout_groups': len(held),
            'bootstrap_group_scale_interval_95': np.percentile(boot, [2.5, 97.5]).tolist(),
            'holdout_absolute_relative_error': summary(errors), 'coverage_passed': bool(coverage),
            'consistency_passed': stable, 'passed': bool(coverage and stable),
            'group_ratios': [{'track': r['track'], 'original_idx': r['original_idx'], 'role': r['role'],
                              'anchors': len(r[field]), 'before': float(np.exp(np.median(r[field]))),
                              'after': float(np.exp(np.median(r[field])) * scale)} for r in records if r[field]]}


def run(args):
    args.output.resolve().relative_to((Path(__file__).resolve().parents[1] / 'outputs').resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text())
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nfeatures=5000)
    trajectories, index, header = {}, {}, {}
    for track in (1, 2):
        trajectories[track] = PriorTrajectory(getattr(args, f'prior_track{track}') / 'sensor_prior.csv')
        index[track] = list(csv.DictReader((args.data_root / f'kaggle_track0{track}/Track{track}_Mapanything_ply/index.csv').open()))
        raw = camera_headers(args.data_root / f'Track{track}/Track{track}.db3')
        header[track] = {tag: resolve_camera_timestamps(raw[tag], [int(r[f'{tag}_ts']) for r in index[track]]) for tag in ('f', 'fl', 'fr')}
    records = []
    for selection in plan['groups']:
        track, idx, later_idx = selection['track'], selection['original_idx'], selection['temporal_check_idx']
        group = args.reconstruction / f'track{track}_{idx:06d}'
        trajectory = trajectories[track]
        front = np.load(group / 'rig_0.npz')
        images = [cv2.imread(str(group / f'{tag}_rectified.jpg'), 0) for tag in ('f', 'fl', 'fr')]
        masks = [cv2.imread(str(group / f'{tag}_valid.png'), 0) for tag in ('f', 'fl', 'fr')]
        features = [sift.detectAndCompute(image, mask) for image, mask in zip(images, masks)]
        row = index[track][later_idx]
        camera_root = args.data_root / f'Track{track}/Track{track}_images'
        later_image = camera_root / f'Track{track}_f_images/{later_idx:06d}_ts_{row["f_ts"]}.jpg'
        rgb, later_k, valid = rectify(later_image, camera_root / 'camera_info/gige_100_f_hdr_camera_info.yaml',
                                      (images[0].shape[1], images[0].shape[0]), camera_root / 'camera_info/gige_100_f_hdr_mask.png')
        later_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kp_later, desc_later = sift.detectAndCompute(later_gray, valid.astype('uint8') * 255)
        temporal = dict(spatially_unique(mutual_matches(features[0][1], desc_later),features[0][0],kp_later))
        next_pose = trajectory.camera_pose(header[track]['f'][idx][1], header[track]['f'][later_idx][1], np.eye(4))
        later_from_front = np.linalg.inv(next_pose)
        record = {**selection, 'logs_all': [], 'logs_static': [], 'logs_temporal': [], 'logs_temporal_all': [], 'pairs': [],
                  'temporal_baseline_m': float(np.linalg.norm(next_pose[:3, 3])), 'later_image_sha256': sha256(later_image),
                  'temporal_header_delta_ns': header[track]['f'][later_idx][1] - header[track]['f'][idx][1]}
        # Check the sensor-to-camera motion contract independently of stereo depth.
        if len(temporal) >= 8:
            ta = np.array([[*features[0][0][i].pt, 1] for i in temporal])
            tb = np.array([[*kp_later[temporal[i]].pt, 1] for i in temporal])
            record['temporal_sensor_epipolar_px'] = summary(epipolar_error(ta, tb, front['intrinsics'], later_k, next_pose))
            na = ta @ np.linalg.inv(front['intrinsics']).T
            nb = tb @ np.linalg.inv(later_k).T
            essential, inliers = cv2.findEssentialMat(na[:, :2], nb[:, :2], np.eye(3), method=cv2.RANSAC,
                                                     prob=.999, threshold=1./float(front['intrinsics'][0,0]))
            if essential is not None and essential.shape == (3, 3):
                image_inliers = inliers[:,0].astype(bool).copy()
                count, rotation, direction, _ = cv2.recoverPose(essential, na[:, :2], nb[:, :2], np.eye(3), mask=inliers)
                delta = rotation @ later_from_front[:3, :3].T
                record['temporal_visual_motion'] = {'cheirality_inliers': int(count),
                    'rotation_disagreement_deg': float(np.rad2deg(np.arccos(np.clip((np.trace(delta)-1)/2,-1,1)))),
                    'translation_direction': direction[:,0].tolist(),
                    'scope': 'Image-only essential geometry; forward motion/dynamics can be degenerate. Not ground truth.'}
                # Independent longer-baseline scale probe, not restricted to the small front/side overlap.
                tp,_,_,valid_temporal = triangulate(ta,tb,front['intrinsics'],later_k,next_pose)
                valid_temporal &= image_inliers & (epipolar_error(ta,tb,front['intrinsics'],later_k,next_pose)<=2.)
                for j in np.flatnonzero(valid_temporal):
                    u,v=np.rint(ta[j,:2]).astype(int)
                    if front['mask'][v,u]:
                        ratio=float(front['depth_z'][v,u]/tp[j,2])
                        if np.isfinite(ratio) and ratio>0:record['logs_temporal_all'].append(float(np.log(ratio)))
        for camera_id, tag in ((1, 'fl'), (2, 'fr')):
            side = np.load(group / f'rig_{camera_id}.npz')
            kp1, d1 = features[0]; kp2, d2 = features[camera_id]
            pairs = spatially_unique(mutual_matches(d1, d2),kp1,kp2)
            if not pairs:
                record['pairs'].append({'camera': tag, 'matches': 0, 'stereo_anchors': 0, 'static_anchors': 0})
                continue
            x1 = np.array([[*kp1[i].pt, 1] for i, j in pairs]); x2 = np.array([[*kp2[j].pt, 1] for i, j in pairs])
            pose, k1, k2 = side['applied_camera_pose'], front['intrinsics'], side['intrinsics']
            p, p2, angles, eligible = triangulate(x1, x2, k1, k2, pose)
            errors = epipolar_error(x1, x2, k1, k2, pose)
            eligible &= errors <= 2.
            logs, static_logs, temporal_logs, entries = [], [], [], []
            for number in np.flatnonzero(eligible):
                i, j = pairs[number]
                u, v = np.rint(x1[number, :2]).astype(int); su, sv = np.rint(x2[number, :2]).astype(int)
                if not (front['mask'][v, u] and side['mask'][sv, su]):
                    continue
                ratios = np.array([front['depth_z'][v, u] / p[number, 2], side['depth_z'][sv, su] / p2[number, 2]])
                if np.any(~np.isfinite(ratios)) or np.any(ratios <= 0) or abs(np.log(ratios[0] / ratios[1])) > np.log(1.25):
                    continue
                log_ratio = float(np.mean(np.log(ratios)))
                logs.append(log_ratio)
                detail = {'front_pixel': x1[number, :2].tolist(), 'side_pixel': x2[number, :2].tolist(),
                          'depth_ratios': ratios.tolist(), 'parallax_deg': float(angles[number]), 'sampson_px': float(errors[number]),
                          'temporal_reprojection_px': None, 'static_compatible': False, 'temporal_depth_ratio': None}
                if i in temporal:
                    observed = np.array(kp_later[temporal[i]].pt)
                    predicted = project(transform(p[number:number+1], later_from_front), later_k)[0]
                    error = float(np.linalg.norm(predicted - observed))
                    detail['temporal_reprojection_px'] = error
                    detail['static_compatible'] = bool(error <= 4.)
                    if detail['static_compatible']:
                        static_logs.append(log_ratio)
                    a = x1[number:number+1]; b = np.array([[*observed, 1]])
                    tp, _, _, tok = triangulate(a, b, k1, later_k, next_pose)
                    te = epipolar_error(a, b, k1, later_k, next_pose)[0]
                    if tok[0] and te <= 4:
                        ratio = float(front['depth_z'][v, u] / tp[0, 2])
                        if np.isfinite(ratio) and ratio > 0:
                            temporal_logs.append(float(np.log(ratio)))
                            detail['temporal_depth_ratio'] = ratio
                entries.append(detail)
            record['logs_all'].extend(logs); record['logs_static'].extend(static_logs); record['logs_temporal'].extend(temporal_logs)
            record['pairs'].append({'camera': tag, 'matches': len(pairs), 'stereo_anchors': len(logs), 'static_anchors': len(static_logs),
                                    'sampson_px_all_matches': summary(errors), 'anchors': entries})
        records.append(record)
        print(track, idx, selection['role'], 'stereo/static/temporal', len(record['logs_all']),len(record['logs_static']),len(record['logs_temporal']), flush=True)
    estimates = {name: estimate(records, field) for name, field in [('stereo', 'logs_all'), ('static_stereo', 'logs_static'),
                                                                ('temporal_front', 'logs_temporal'), ('temporal_front_all','logs_temporal_all')]}
    result = {'scope': 'Separated scene consistency validation. Stereo and temporal motion are calibrated/assumed inputs, not ground truth. Static-compatible does not prove static.',
              'plan_sha256': sha256(args.plan), 'thresholds': {'min_angle_deg': 2, 'max_stereo_sampson_px': 2, 'max_depth_ratio_disagreement': 1.25,
                                                            'max_temporal_reprojection_px': 4, 'min_anchors_per_group': 3,
                                                            'minimum_fit_and_holdout_groups': 6, 'median_relative_error_max': .1, 'p90_relative_error_max': .2},
              'estimates': estimates, 'groups': records}
    (args.output / 'bias_report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(estimates, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan','reconstruction','data-root','prior-track1','prior-track2','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
