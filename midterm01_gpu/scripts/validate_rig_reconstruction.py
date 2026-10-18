#!/usr/bin/env python3
"""獨立影像幾何檢查：固定外參的 epipolar 誤差與有視差點的尺度診斷。"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {'count': len(values), 'median': float(np.median(values)), 'p90': float(np.percentile(values, 90))} if len(values) else {'count': 0, 'median': None, 'p90': None}


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def epipolar_error(x1, x2, k1, k2, front_from_side):
    side_from_front = np.linalg.inv(front_from_side)
    essential = skew(side_from_front[:3, 3]) @ side_from_front[:3, :3]
    f = np.linalg.inv(k2).T @ essential @ np.linalg.inv(k1)
    line2, line1 = x1 @ f.T, x2 @ f
    return np.abs(np.sum(x2 * line2, axis=1)) / np.sqrt(np.sum(line1[:, :2]**2 + line2[:, :2]**2, axis=1))


def validate(directory):
    directory = Path(directory)
    results = []
    sift = cv2.SIFT_create(nfeatures=5000)
    matcher = cv2.BFMatcher()
    for group in sorted(directory.glob('track[12]_*')):
        report = json.loads((group / 'report.json').read_text())
        track = report['track']
        rig = json.loads((directory / f'rig_track{track}.json').read_text())['fixed_optical_front_from_camera']
        images = [cv2.imread(str(group / f'{tag}_rectified.jpg'), 0) for tag in ('f', 'fl', 'fr')]
        valid_masks = [cv2.imread(str(group / f'{tag}_valid.png'), 0) for tag in ('f', 'fl', 'fr')]
        features = [sift.detectAndCompute(image, mask) for image, mask in zip(images, valid_masks)]
        front = np.load(group / 'rig_0.npz')
        k1 = front['intrinsics']
        for camera_id, tag in enumerate(('fl', 'fr'), 1):
            kp1, des1 = features[0]; kp2, des2 = features[camera_id]
            side = np.load(group / f'rig_{camera_id}.npz')
            if des1 is None or des2 is None:
                results.append({'track': track, 'original_idx': report['original_idx'], 'camera': tag, 'matches': 0})
                continue
            # Mutual Lowe-ratio matches, independent of calibration/depth/map.
            forward = {a.queryIdx: a.trainIdx for a, b in matcher.knnMatch(des1, des2, k=2) if a.distance < .8 * b.distance}
            backward = {a.queryIdx: a.trainIdx for a, b in matcher.knnMatch(des2, des1, k=2) if a.distance < .8 * b.distance}
            pairs = [(i, j) for i, j in forward.items() if backward.get(j) == i]
            if not pairs:
                continue
            x1 = np.array([[*kp1[i].pt, 1] for i, _ in pairs])
            x2 = np.array([[*kp2[j].pt, 1] for _, j in pairs])
            static_pose = np.array(rig[tag]); pose = side['applied_camera_pose']; k2 = side['intrinsics']
            static_error = epipolar_error(x1, x2, k1, k2, static_pose)
            motion_error = epipolar_error(x1, x2, k1, k2, pose)
            side_from_front = np.linalg.inv(pose)
            homogeneous = cv2.triangulatePoints(k1 @ np.eye(4)[:3], k2 @ side_from_front[:3], x1[:, :2].T, x2[:, :2].T).T
            with np.errstate(divide='ignore', invalid='ignore'):
                xyz = homogeneous[:, :3] / homogeneous[:, 3:4]
            xyz2 = xyz @ side_from_front[:3, :3].T + side_from_front[:3, 3]
            ray1, ray2 = xyz, xyz - pose[:3, 3]
            with np.errstate(divide='ignore', invalid='ignore'):
                angle = np.rad2deg(np.arccos(np.clip(np.sum(ray1 * ray2, axis=1) / (np.linalg.norm(ray1, axis=1) * np.linalg.norm(ray2, axis=1)), -1, 1)))
            keep = ((motion_error < 3) & (angle > 1) & (xyz[:, 2] > 0) & (xyz2[:, 2] > 0)
                    & (np.linalg.norm(xyz, axis=1) > 2) & (np.linalg.norm(xyz, axis=1) < 60))
            ratios = []
            for j in np.flatnonzero(keep):
                u1, v1 = np.rint(x1[j, :2]).astype(int); u2, v2 = np.rint(x2[j, :2]).astype(int)
                if front['mask'][v1, u1] and side['mask'][v2, u2]:
                    ratios.append([front['depth_z'][v1, u1] / xyz[j, 2], side['depth_z'][v2, u2] / xyz2[j, 2]])
            ratios = np.asarray(ratios).reshape(-1, 2)
            results.append({'track': track, 'original_idx': report['original_idx'], 'camera': tag, 'matches': len(pairs),
                            'static_sampson_px': summary(static_error), 'motion_sampson_px': summary(motion_error),
                            'static_below3px': int(np.sum(static_error < 3)), 'motion_below3px': int(np.sum(motion_error < 3)),
                            'triangulated_depth_ratio_front': summary(ratios[:, 0]), 'triangulated_depth_ratio_side': summary(ratios[:, 1]),
                            'input_baseline_m': float(np.linalg.norm(pose[:3, 3])),
                            'predicted_baseline_m': float(np.linalg.norm(side['model_camera_pose'][:3, 3] - front['model_camera_pose'][:3, 3]))})
            np.savez_compressed(group / f'validation_{tag}.npz', pixels_front=x1, pixels_side=x2,
                                static_error_px=static_error, motion_error_px=motion_error, triangulated_front_xyz=xyz,
                                angle_deg=angle, triangulation_eligible=keep, depth_ratios=ratios)
    output = {'scope': 'Descriptor matches independent of depth/map; no fitted extrinsics or time offset. Ratios may contain moving-object/mismatch bias, not ground truth.',
              'thresholds': {'Lowe_ratio': .8, 'Sampson_px': 3, 'minimum_triangulation_angle_deg': 1, 'range_m': [2, 60]}, 'pairs': results}
    (directory / 'image_geometry_validation.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reconstruction', type=Path)
    args = p.parse_args()
    result = validate(args.reconstruction)
    print(json.dumps(result, indent=2))
