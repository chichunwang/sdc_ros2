#!/usr/bin/env python3
"""以校正三相機 baseline，直接核對原 front PLY；不讀 neural 深度或 map。"""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import scipy
from scipy.spatial import cKDTree

from accuracy.benchmark import create_run_directory
from calibrated_rig_geometry import PriorTrajectory, load_rig, rectify
from output_layout import scored_reference_run
from source_rigidity import RigidityConfig, read_rgb_ply, render_features, match_features, statistics
from three_camera_source import camera_headers, resolve_camera_timestamps, sha256
from validate_rig_reconstruction import epipolar_error


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def features(image, mask):
    keys, descriptors = cv2.SIFT_create(nfeatures=8000, contrastThreshold=.02).detectAndCompute(
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), np.asarray(mask, np.uint8)*255)
    return dict(pixels=np.asarray([k.pt for k in keys]).reshape(-1, 2), descriptors=descriptors)


def spatial_holdout(pixels, size):
    cells = np.floor(np.asarray(pixels)/np.asarray(size)*[4, 3]).astype(int)
    return (cells[:, 0]+2*cells[:, 1]) % 3 == 0


def fit_image_mapping(source, front, config):
    """A direction-only image homography; never receives any XYZ or stereo data."""
    pairs = match_features(source, front, config)
    a, b = source['pixels'][pairs[:, 0]], front['pixels'][pairs[:, 1]]
    holdout = spatial_holdout(a, (config.width, config.height))
    report = dict(matches=len(pairs), fit_matches=int((~holdout).sum()), holdout_matches=int(holdout.sum()))
    h = None
    if min(int((~holdout).sum()), int(holdout.sum())) >= 6 and (~holdout).sum() >= 12:
        cv2.setRNGSeed(20261020)
        h, mask = cv2.findHomography(a[~holdout], b[~holdout], cv2.RANSAC, 3., maxIters=5000, confidence=.999)
        if h is not None and np.isfinite(h).all() and abs(np.linalg.det(h)) > 1e-9:
            predicted = cv2.perspectiveTransform(a[:, None, :], h)[:, 0]
            error = np.linalg.norm(predicted-b, axis=1)
            report.update(homography_front_from_virtual=h.tolist(), fit_inliers=int(mask.sum()),
                          fit_error_px=statistics(error[~holdout]), holdout_error_px=statistics(error[holdout]))
            report['usable'] = bool(mask.sum() >= 12 and np.median(error[holdout]) <= 3.
                                    and np.quantile(error[holdout], .9) <= 6.)
        else:
            h = None
    report.setdefault('usable', False)
    if not report['usable']:
        h = None
    return h, report, dict(virtual_pixels=a, front_pixels=b, holdout=holdout)


def associate_vertices(xyz, front_pixels, virtual_from_front, config):
    """Return only visible original vertices near inverse-mapped image pixels."""
    xyz = np.asarray(xyz, float)
    indices = np.flatnonzero(xyz[:, 0] > 1e-8)
    uv = -xyz[indices, 1:]/xyz[indices, 0, None]*config.focal_px+[config.width/2, config.height/2]
    rounded = np.rint(uv).astype(int)
    valid = ((rounded[:, 0] >= 0) & (rounded[:, 0] < config.width)
             & (rounded[:, 1] >= 0) & (rounded[:, 1] < config.height))
    indices, uv, rounded = indices[valid], uv[valid], rounded[valid]
    ids = np.full(len(front_pixels), -1, int)
    distances = np.full(len(front_pixels), np.inf)
    if len(indices) < 3 or not len(front_pixels):
        return ids, distances
    order = np.argsort(xyz[indices, 0], kind='stable')
    _, first = np.unique(rounded[order, 1]*config.width+rounded[order, 0], return_index=True)
    take = order[first]
    visible = indices[take]
    query = cv2.perspectiveTransform(np.asarray(front_pixels, float)[:, None], virtual_from_front)[:, 0]
    dd, nn = cKDTree(uv[take]).query(query, k=min(5, len(take)))
    if dd.ndim == 1:
        dd, nn = dd[:, None], nn[:, None]
    close = dd <= config.association_radius_px
    depths = xyz[visible[nn], 0]
    lo = np.min(np.where(close, depths, np.inf), axis=1)
    hi = np.max(np.where(close, depths, 0), axis=1)
    good = (close.sum(axis=1) >= 3) & (hi <= config.max_local_depth_ratio*lo)
    seen = set()
    for j in np.flatnonzero(good):
        vertex = int(visible[nn[j, 0]])
        if vertex not in seen:
            ids[j], distances[j] = vertex, dd[j, 0]
            seen.add(vertex)
    return ids, distances


def triangulate(front_pixels, side_pixels, k1, k2, front_from_side):
    """Fixed supplied baseline, no fitted pose; keep thresholds independent of PLY."""
    a, b = np.asarray(front_pixels, float), np.asarray(side_pixels, float)
    inverse = np.linalg.inv(front_from_side)
    x1, x2 = np.c_[a, np.ones(len(a))], np.c_[b, np.ones(len(b))]
    if not len(a):
        return dict(xyz=np.empty((0, 3)), eligible=np.zeros(0, bool), angle_deg=np.zeros(0),
                    sampson_px=np.zeros(0), reprojection_px=np.zeros(0))
    homogeneous = cv2.triangulatePoints(k1 @ np.eye(4)[:3], k2 @ inverse[:3], a.T, b.T).T
    with np.errstate(divide='ignore', invalid='ignore'):
        xyz = homogeneous[:, :3]/homogeneous[:, 3:4]
        side = xyz @ inverse[:3, :3].T+inverse[:3, 3]
        ray1, ray2 = xyz, xyz-front_from_side[:3, 3]
        angle = np.rad2deg(np.arccos(np.clip(np.sum(ray1*ray2, axis=1)/
                                            (np.linalg.norm(ray1, axis=1)*np.linalg.norm(ray2, axis=1)), -1, 1)))
        uv1, uv2 = xyz @ k1.T, side @ k2.T
        reproj = np.maximum(np.linalg.norm(uv1[:, :2]/uv1[:, 2:3]-a, axis=1),
                            np.linalg.norm(uv2[:, :2]/uv2[:, 2:3]-b, axis=1))
    sampson = epipolar_error(x1, x2, k1, k2, front_from_side)
    eligible = (np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0) & (side[:, 2] > 0)
                & (angle >= 2.) & (sampson <= 2.) & (reproj <= 2.)
                & (np.linalg.norm(xyz, axis=1) >= 2.) & (np.linalg.norm(xyz, axis=1) <= 60.))
    return dict(xyz=xyz, eligible=eligible, angle_deg=angle, sampson_px=sampson, reprojection_px=reproj)


def shared_scale(groups, mode):
    """Equal fit-scene weights; the holdout scenes never change the estimate."""
    fit = [g for g in groups if g['role'] == 'fit' and g[mode]['anchors'] >= 6]
    if len(fit) < 3 or {g['track'] for g in fit} != {1, 2}:
        return dict(available=False, reason='insufficient_fit_scenes_across_both_tracks')
    scale = float(np.exp(np.median([g[mode]['median_log_metric_per_raw'] for g in fit])))
    holdout = [g for g in groups if g['role'] == 'holdout' and g[mode]['anchors'] >= 6]
    return dict(available=True, ply_scale=scale, fit_groups=len(fit), holdout_groups=len(holdout),
                holdout_group_relative_scale_error=statistics([abs(np.exp(g[mode]['median_log_metric_per_raw'])/scale-1) for g in holdout]),
                estimator='exp(median(per-fit-scene median(log(stereo range/raw vertex range))))',
                calibrated=False, accepted_for_fusion=False)


def run(args):
    root = Path(__file__).resolve().parents[1]
    output = create_run_directory(root/'outputs/run', 'front_stereo_audit')
    destination = output/'front_stereo'; destination.mkdir()
    print('Run:', output, flush=True)
    start = time.perf_counter(); cv2.setNumThreads(1)
    config = RigidityConfig()
    snapshots = destination/'sources'; snapshots.mkdir()
    source_hashes = {}
    for name in ('audit_front_stereo.py', 'source_rigidity.py', 'calibrated_rig_geometry.py',
                 'three_camera_source.py', 'validate_rig_reconstruction.py', 'output_layout.py'):
        p = Path(__file__).with_name(name); shutil.copy2(p, snapshots/name); source_hashes[name] = sha256(p)
    for relative in ('tests/test_front_stereo.py', 'accuracy/benchmark.py'):
        p = Path(__file__).parent/relative
        shutil.copy2(p, snapshots/relative.replace('/', '_')); source_hashes[relative] = sha256(p)
    (destination/'git_commit.txt').write_bytes(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root))
    (destination/'worktree.patch').write_bytes(subprocess.check_output(['git', 'diff'], cwd=root))
    dump(destination/'params.json', dict(command=sys.argv, config=asdict(config), rectification_size=[args.width, args.height],
        versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, opencv=cv2.__version__),
        source_sha256=source_hashes, thresholds=dict(image_mapping_fit_inliers=12, image_mapping_holdout_median_px=3.,
            image_mapping_holdout_p90_px=6., stereo_sampson_px=2., stereo_reprojection_px=2., minimum_angle_deg=2., range_m=[2, 60]),
        uses_map=False, uses_neural_depth=False, mode_static_uses_gnss=False,
        mode_motion_uses_saved_sensor_prior=True, full_localization=False))
    # Four fixed fractions plus original motion windows, chosen before looking at stereo residuals.
    original = {1: [925, 2306, 1828], 2: [116, 244, 1065, 2588, 1399]}
    groups = []; plan = []; dependencies = {}
    for track in (1, 2):
        baseline = scored_reference_run(root, track)
        frame_rows = list(csv.DictReader((baseline/'frames.csv').open()))
        index_file = args.data_root/f'kaggle_track0{track}/Track{track}_Mapanything_ply/index.csv'
        rows = list(csv.DictReader(index_file.open()))
        raw = camera_headers(args.data_root/f'Track{track}/Track{track}.db3')
        headers = {tag: resolve_camera_timestamps(raw[tag], [int(r[f'{tag}_ts']) for r in rows]) for tag in ('f', 'fl', 'fr')}
        image_root = args.data_root/f'Track{track}/Track{track}_images'
        calibration = image_root/'camera_info'; rig = load_rig(calibration)
        trajectory = PriorTrajectory(baseline/'sensor_prior.csv')
        indices = original[track]+[int(round((len(frame_rows)-1)*f)) for f in (.1, .35, .65, .9)]
        if args.groups:
            indices = [i for t, i in (map(int, value.split(':')) for value in args.groups.split(',')) if t == track]
        dependencies[track] = dict(index_sha256=sha256(index_file), prior_path=str(baseline/'sensor_prior.csv'),
            prior_sha256=sha256(baseline/'sensor_prior.csv'), frame_manifest_sha256=sha256(baseline/'frames.csv'),
            baseline_path=str(baseline), baseline_sha256={name: sha256(baseline/name) for name in
                ('result.txt', 'diagnostic.csv', 'sensor_prior.csv', 'frames.csv', 'test_info.json')},
            bag_sha256=sha256(args.data_root/f'Track{track}/Track{track}.db3'),
            calibration_sha256={p.name: sha256(p) for p in calibration.iterdir() if p.is_file()},
            front_from_side_optical={tag: rig[tag].tolist() for tag in ('fl', 'fr')})
        for ordinal, idx in enumerate(indices):
            role = 'fit' if ordinal % 2 == 0 else 'holdout'
            plan.append(dict(track=track, idx=idx, role=role))
            directory = destination/f'track{track}_{idx:06d}'; directory.mkdir()
            row, frame = rows[idx], frame_rows[idx]
            stamp = int(frame['timestamp_ns'])
            if int(row['f_ts']) != stamp:
                raise ValueError('Original front frame/index exact timestamp mismatch')
            ply = args.data_root/f'Track{track}/raw_ply'/frame['filename']
            xyz, rgb = read_rgb_ply(ply)
            source = render_features(xyz, rgb, config)
            images = {}; image_features = {}; intrinsics = {}; provenance = []
            for tag in ('f', 'fl', 'fr'):
                image_path = image_root/f'Track{track}_{tag}_images/{idx:06d}_ts_{row[tag+"_ts"]}.jpg'
                image, k, valid = rectify(image_path, calibration/f'gige_100_{tag}_hdr_camera_info.yaml',
                                          (args.width, args.height), calibration/'gige_100_f_hdr_mask.png' if tag == 'f' else None)
                images[tag], intrinsics[tag], image_features[tag] = image, k, features(image, valid)
                cv2.imwrite(str(directory/f'{tag}_rectified.jpg'), image[:, :, ::-1])
                provenance.append(dict(camera=tag, path=str(image_path), sha256=sha256(image_path),
                    bag_ns=int(headers[tag][idx][0]), header_ns=int(headers[tag][idx][1]), index_time_basis=headers[tag][idx][3]))
            h, mapping, mapping_arrays = fit_image_mapping(source, image_features['f'], config)
            np.savez_compressed(directory/'mapping.npz', **mapping_arrays)
            cv2.imwrite(str(directory/'virtual.jpg'), source['image'][:, :, ::-1])
            record = dict(track=track, idx=idx, role=role, timestamp_ns=stamp, mapping=mapping,
                ply=dict(path=str(ply), sha256=sha256(ply), vertices=len(xyz)), images=provenance)
            mode_logs = dict(static=[], motion=[]); mode_ranges = dict(static=[], motion=[])
            mode_vertices = dict(static=[], motion=[])
            sides = []
            for tag in ('fl', 'fr'):
                pairs = match_features(image_features['f'], image_features[tag], config)
                a, b = image_features['f']['pixels'][pairs[:, 0]], image_features[tag]['pixels'][pairs[:, 1]]
                vertices, association_distance = (associate_vertices(xyz, a, np.linalg.inv(h), config)
                    if h is not None else (np.full(len(a), -1, int), np.full(len(a), np.inf)))
                associated = vertices >= 0
                side_report = dict(camera=tag, descriptor_matches=len(pairs), associated_vertices=int(associated.sum()), modes={})
                arrays = dict(pixels_front=a, pixels_side=b, vertex_indices=vertices, association_distance_virtual_px=association_distance)
                for mode in ('static', 'motion'):
                    pose = rig[tag] if mode == 'static' else trajectory.camera_pose(headers['f'][idx][1], headers[tag][idx][1], rig[tag])
                    result = triangulate(a, b, intrinsics['f'], intrinsics[tag], pose)
                    keep = result['eligible'] & associated
                    # Depth disagreement never affects this keep mask.
                    raw_range = np.linalg.norm(xyz[vertices[keep]], axis=1)
                    stereo_range = np.linalg.norm(result['xyz'][keep], axis=1)
                    logs = np.log(stereo_range/raw_range)
                    mode_logs[mode].extend(logs.tolist()); mode_ranges[mode].extend(stereo_range.tolist())
                    mode_vertices[mode].extend(vertices[keep].tolist())
                    side_report['modes'][mode] = dict(triangulation_eligible=int(result['eligible'].sum()), anchors=int(keep.sum()),
                        sampson_px=statistics(result['sampson_px']), angle_deg=statistics(result['angle_deg'][keep]),
                        metric_per_raw=statistics(np.exp(logs)), scaled3_over_stereo_range=statistics(3*np.exp(-logs)),
                        input_baseline_m=float(np.linalg.norm(pose[:3, 3])), front_from_side=pose.tolist())
                    for name, value in result.items():
                        arrays[mode+'_'+name] = value
                    arrays[mode+'_anchor'] = keep
                    arrays[mode+'_metric_per_raw'] = np.exp(logs)
                    arrays[mode+'_raw_xyz'] = xyz[vertices[keep]]
                common = arrays['static_anchor'] & arrays['motion_anchor']
                if common.any():
                    static_range = np.linalg.norm(arrays['static_xyz'][common], axis=1)
                    motion_range = np.linalg.norm(arrays['motion_xyz'][common], axis=1)
                    side_report['common_anchor_motion_over_static_range'] = statistics(motion_range/static_range)
                else:
                    side_report['common_anchor_motion_over_static_range'] = None
                np.savez_compressed(directory/f'anchors_{tag}.npz', **arrays)
                sides.append(side_report)
            record['sides'] = sides
            for mode in ('static', 'motion'):
                original_logs, original_ranges = np.asarray(mode_logs[mode]), np.asarray(mode_ranges[mode])
                vertex_ids = np.asarray(mode_vertices[mode])
                unique = np.unique(vertex_ids)
                # Same front vertex seen by both sides receives one scene vote.
                logs = np.array([np.mean(original_logs[vertex_ids == i]) for i in unique])
                ranges = np.array([np.mean(original_ranges[vertex_ids == i]) for i in unique])
                record[mode] = dict(anchors=len(logs), metric_per_raw=statistics(np.exp(logs)),
                    stereo_observations=len(original_logs), same_vertex_observations_deduplicated=True,
                    median_log_metric_per_raw=float(np.median(logs)) if len(logs) else None,
                    range_bins=[dict(range_m=[lo, hi], metric_per_raw=statistics(np.exp(logs[(ranges >= lo) & (ranges < hi)])))
                                for lo, hi in ((2., 10.), (10., 20.), (20., 60.))])
            dump(directory/'report.json', record); groups.append(record)
            print(json.dumps(dict(track=track, idx=idx, role=role, mapping=mapping['usable'],
                anchors={mode: record[mode]['anchors'] for mode in ('static', 'motion')},
                metric_per_raw={mode: record[mode]['metric_per_raw'] for mode in ('static', 'motion')})), flush=True)
    summary = dict(plan=plan, dependencies=dependencies, groups=groups,
        scale_diagnostic={mode: shared_scale(groups, mode) for mode in ('static', 'motion')},
        elapsed_seconds=time.perf_counter()-start, accepted_for_fusion=False,
        limits=['No semantic static labels; moving objects and descriptor mismatches remain possible.',
            'Static mode is independent of GNSS but uses asynchronous captures without motion correction.',
            'Motion mode uses saved GNSS/IMU prior as front-camera motion; lever arm and full body extrinsic unverified.',
            'Only original front PLY vertices are checked; no three-camera learned depth or map posterior enters this audit.',
            'A homography maps texture directions only, with spatial holdout; no depth is manufactured.',
            'Sparse calibrated stereo is a consistency reference, not ATE or ground truth.'])
    summary['baseline_bytes_unchanged'] = all(sha256(Path(item['baseline_path'])/name) == expected
        for item in dependencies.values() for name, expected in item['baseline_sha256'].items())
    dump(destination/'report.json', summary)
    info = json.loads((output/'test_info.json').read_text())
    info.update(algorithm='original_front_stereo_audit_v1', status='completed', reported_score=None,
                report='front_stereo/report.json', full_localization_executed=False)
    dump(output/'test_info.json', info)
    print(json.dumps(summary['scale_diagnostic'], indent=2)); print('Run:', output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    p.add_argument('--data-root', type=Path, default=root.parent/'midterm01/data')
    p.add_argument('--width', type=int, default=896)
    p.add_argument('--height', type=int, default=588)
    p.add_argument('--groups', help='Optional explicit Track:index list; default fixed sixteen route scenes')
    run(p.parse_args())
