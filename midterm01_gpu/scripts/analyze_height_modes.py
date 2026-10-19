#!/usr/bin/env python3
"""Read-only paired height diagnostics; common raw probes, never ground-truth error."""
from output_layout import loads_saved_json, saved_path
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from three_camera_source import sha256
from three_camera_calibration import load_ply, load_binary_map, SurfaceMap


def load_run(path):
    path = Path(path).resolve()
    params = loads_saved_json((path / 'params.json').read_text())
    source = loads_saved_json((path / 'source_input.json').read_text()) if (path / 'source_input.json').exists() else None
    with (path / 'sensor_prior.csv').open() as stream:
        priors = list(csv.DictReader(stream))
    matrices = np.loadtxt(path / 'result.txt').reshape(-1, 3, 4)
    if len(priors) != len(matrices) or not np.isfinite(matrices).all():
        raise ValueError('Incomplete/nonfinite run')
    return path, params, source, priors, matrices


def stats(values):
    values = np.asarray(values)
    return dict(zip(('p10', 'median', 'p90'), np.quantile(values, [.1, .5, .9]).tolist())) if len(values) else None


def probe(scene, raw, pose, selection_rotation=None):
    xyz = raw @ pose[:, :3].T + pose[:, 3]
    distances, ids = scene.tree.query(xyz)
    normals, quality = scene.surfaces(ids)
    valid = (distances <= 3) & (quality > 0) & (np.abs(normals[:, 2]) >= .95)
    residual = np.einsum('ij,ij->i', normals, xyz - scene.points[ids])
    dz = residual[valid] / normals[valid, 2]
    unique = np.array([np.median(dz[ids[valid] == i]) for i in np.unique(ids[valid])])
    selection_rotation = pose[:,:3] if selection_rotation is None else selection_rotation
    relative = raw @ selection_rotation.T
    yaw = np.arctan2(selection_rotation[1,0],selection_rotation[0,0])
    forward = relative[:,:2] @ np.array([np.cos(yaw),np.sin(yaw)])
    lateral = relative[:,:2] @ np.array([-np.sin(yaw),np.cos(yaw)])
    outside = ~((forward >= 1)&(forward <= 15)&(np.abs(lateral) <= 5)&(relative[:,2] >= -4)&(relative[:,2] <= -.3))
    return dict(clipped_nn_rmse_m=float(np.mean(np.minimum(distances, 2)**2)**.5),
                outside_source_surface_fit_nn_rmse_m=float(np.mean(np.minimum(distances[outside],2)**2)**.5) if outside.any() else None,
                outside_source_surface_fit_points=int(outside.sum()),
                nn_inlier_fraction=float(np.mean(distances <= 2)),
                unique_horizontal_targets=len(unique),
                signed_vertical_surface_difference_m=stats(unique),
                median_absolute_vertical_surface_difference_m=float(np.median(np.abs(unique))) if len(unique) else None)


def lower_plane_probe(raw, rotation, yaw=None, min_z=-4.):
    """Dominant lower surface in source coordinates; no semantic road label.

    Fit on IMU-rotated source geometry WITHOUT pose translation. Variation of
    this intercept therefore cannot have been caused by trajectory Z.
    """
    xyz = raw @ rotation.T
    if yaw is None:
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    forward = xyz[:, :2] @ np.array([np.cos(yaw), np.sin(yaw)])
    lateral = xyz[:, :2] @ np.array([-np.sin(yaw), np.cos(yaw)])
    keep = (forward >= 1) & (forward <= 15) & (np.abs(lateral) <= 5) & (xyz[:, 2] >= min_z) & (xyz[:, 2] <= -.3)
    points = np.column_stack([forward[keep], lateral[keep], xyz[keep, 2]])
    if len(points) < 20:
        return dict(status='insufficient_points', candidate_points=len(points))
    design = np.column_stack([points[:, :2], np.ones(len(points))])
    rng = np.random.default_rng(20261019)
    best, best_count = None, 0
    proposals = []
    for _ in range(200):
        ids = rng.choice(len(points), 3, replace=False)
        if abs(np.linalg.det(design[ids])) < .1:
            continue
        model = np.linalg.solve(design[ids], points[ids, 2])
        if np.linalg.norm(model[:2]) > .35:
            continue
        inliers = np.abs(design @ model-points[:, 2]) <= .15
        proposals.append((int(inliers.sum()), model))
        if inliers.sum() > best_count:
            best, best_count = inliers, inliers.sum()
    if best is None or best_count < 20:
        return dict(status='insufficient_plane_support', candidate_points=len(points))
    for _ in range(3):
        model = np.linalg.lstsq(design[best], points[best, 2], rcond=None)[0]
        best = np.abs(design @ model-points[:, 2]) <= .15
        if best.sum() < 20:
            return dict(status='insufficient_plane_support', candidate_points=len(points))
    spread = np.linalg.eigvalsh(np.cov(points[best, :2].T))
    if spread[0] < .15 or np.linalg.norm(model[:2]) > .35:
        return dict(status='insufficient_spatial_span', candidate_points=len(points))
    competing = max((count for count, other in proposals if abs(other[2]-model[2]) >= .4), default=0)/best.sum()
    # Intercept extrapolation matters more than a fixed forward-span rule.
    # Cap effective support at 20 to avoid treating dense correlated pixels as
    # arbitrarily many independent height measurements. This is a conditioning
    # proxy using the 0.15m fitting scale, NOT calibrated covariance.
    intercept_variance_factor=np.linalg.inv(design[best].T @ design[best])[2,2]*max(1.,best.sum()/20.)
    origin_proxy=.15*np.sqrt(max(0.,intercept_variance_factor))
    return dict(status='candidate', candidate_points=len(points), support=int(best.sum()),
                support_span_m=np.ptp(points[best,:2],axis=0).tolist(), competing_plane_support_fraction=float(competing),
                origin_extrapolation_proxy_m=float(origin_proxy),
                relative_z_at_source_origin_m=float(model[2]), forward_slope=float(model[0]), lateral_slope=float(model[1]),
                fit_rmse_m=float(np.mean((design[best] @ model-points[best, 2])**2)**.5),
                semantic_road_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=saved_path, required=True,
                        help='JSON list: track, start, carry, anchored, original, map_cache')
    parser.add_argument('--data-root', type=saved_path, default=Path('../midterm01/data'))
    parser.add_argument('--output', type=saved_path, required=True)
    args = parser.parse_args()
    pairs = loads_saved_json(args.pairs.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(len(pairs), 1, figsize=(11, 2.8 * len(pairs)), squeeze=False)
    surfaces, originals, reports = {}, {}, []
    for pair, axis in zip(pairs, axes.flat):
        carry, anchored = load_run(pair['carry']), load_run(pair.get('candidate',pair.get('anchored')))
        candidate_name = anchored[1].get('sensor_height_mode','anchored')
        # Compare the same observations, geometry and algorithm. Only height
        # policy/output paths may differ. No matching across float timestamps.
        if carry[2] != anchored[2] or carry[3] != anchored[3]:
            raise ValueError('Pair must use identical source manifest and exact sensor prior values')
        ignored = {'save_path', 'sensor_prior_file', 'ply_path', 'sensor_height_mode', 'sensor_height_reference_file'}
        if {k: v for k, v in carry[1].items() if k not in ignored} != {k: v for k, v in anchored[1].items() if k not in ignored}:
            raise ValueError('Pair differs in non-height solver parameters')
        if (carry[0] / 'binary_sha256.txt').read_bytes() != (anchored[0] / 'binary_sha256.txt').read_bytes():
            raise ValueError('Pair must use the same binary')
        key = pair['original']
        if key not in originals:
            originals[key] = load_run(key)
        original = originals[key]
        start, count = pair['start'], len(carry[4])
        if [int(p['timestamp_ns']) for p in original[3][start:start+count]] != [int(p['timestamp_ns']) for p in carry[3]]:
            raise ValueError('Original index/timestamp mismatch')
        cache = Path(pair['map_cache'])
        identity = loads_saved_json(cache.with_suffix(cache.suffix + '.json').read_text())
        if identity['track'] != pair['track'] or identity['map_voxel_leaf_size'] != carry[1]['map_voxel_leaf_size']:
            raise ValueError('Map cache Track/leaf mismatch')
        if identity['map_sha256'] != sha256(args.data_root / f"Track{pair['track']}" / 'map.pcd'):
            raise ValueError('Map cache source hash mismatch')
        if str(cache) not in surfaces:
            surfaces[str(cache)] = SurfaceMap(load_binary_map(cache))
        scene = surfaces[str(cache)]
        matrices = {'carry': carry[4], candidate_name: anchored[4], 'original_full': original[4][start:start+count]}
        frames = []
        # Fixed temporal quantiles, selected before evaluating any residual.
        for i in np.unique(np.linspace(0, count-1, 9, dtype=int)):
            record = carry[2]['selected_frames'][i]
            if record['original_idx'] != start + i:
                raise ValueError('Source original index mismatch')
            ply = args.data_root / f"Track{pair['track']}" / 'raw_ply' / record['filename']
            if sha256(ply) != record['sha256']:
                raise ValueError('Raw PLY hash mismatch')
            raw = load_ply(ply, max_points=6000, leaf=.1) * carry[1]['ply_scale']
            center = np.array([float(carry[3][i][k]) for k in ('x', 'y', 'z')])
            ids = scene.tree.query_ball_point(center, 22)
            yaw = np.arctan2(matrices['carry'][i, 1, 0], matrices['carry'][i, 0, 0])
            frames.append(dict(original_idx=int(start+i), timestamp_ns=record['timestamp_ns'],
                probe_points=len(raw), ply_sha256=record['sha256'],
                source_lower_plane_without_translation=lower_plane_probe(raw, matrices['carry'][i, :, :3]),
                map_lower_plane_relative_to_nominal=lower_plane_probe(scene.points[ids]-center, np.eye(3), yaw=yaw, min_z=-6.),
                modes={name: probe(scene, raw, m[i],selection_rotation=matrices['carry'][i,:,:3]) for name, m in matrices.items()}))
        nominal = np.array([float(p['z']) for p in carry[3]])
        mode_summaries = {}
        for name, m in matrices.items():
            mode_summaries[name] = dict(z_min_m=float(m[:, 2, 3].min()), z_max_m=float(m[:, 2, 3].max()),
                max_abs_offset_from_nominal_m=float(np.max(np.abs(m[:, 2, 3]-nominal))),
                max_abs_adjacent_dz_m=float(np.max(np.abs(np.diff(m[:, 2, 3])))) if count > 1 else 0.,
                mean_probe_clipped_nn_rmse_m=float(np.mean([f['modes'][name]['clipped_nn_rmse_m'] for f in frames])))
            mode_summaries[name]['mean_outside_source_surface_fit_nn_rmse_m'] = float(np.mean([f['modes'][name]['outside_source_surface_fit_nn_rmse_m'] for f in frames]))
            axis.plot(np.arange(start, start+count), m[:, 2, 3], label=name, linewidth=1.2)
        axis.plot(np.arange(start, start+count), nominal, ':', color='black', label='Nominal sensor Z (constant)')
        axis.set(title=f"Track {pair['track']} | original idx {start}-{start+count-1}", xlabel='Original frame index', ylabel='Map Z [m]')
        axis.grid(alpha=.2); axis.legend(fontsize=8, ncol=4)
        xy = np.linalg.norm(carry[4][:, :2, 3]-anchored[4][:, :2, 3], axis=1)
        yaw_a = np.arctan2(carry[4][:, 1, 0], carry[4][:, 0, 0])
        yaw_b = np.arctan2(anchored[4][:, 1, 0], anchored[4][:, 0, 0])
        report = dict(**pair, count=count, same_binary=True, same_prior_values=True, same_source_manifest=True,
                      map_cache_sha256=sha256(cache), mode_summaries=mode_summaries,
                      paired_xy_difference_m=stats(xy), paired_abs_yaw_difference_deg=stats(np.degrees(np.abs(np.arctan2(np.sin(yaw_a-yaw_b), np.cos(yaw_a-yaw_b))))),
                      frames=frames)
        planes = [f['source_lower_plane_without_translation']['relative_z_at_source_origin_m'] for f in frames
                  if f['source_lower_plane_without_translation']['status'] == 'candidate']
        report['source_lower_plane_without_translation'] = dict(usable_probes=len(planes), total_probes=len(frames),
            relative_intercept_m=stats(planes), range_m=float(np.ptp(planes)) if planes else None,
            semantic_road_verified=False)
        reports.append(report)
    original_summaries = []
    for key, (_, params, _, priors, m) in originals.items():
        nominal = np.array([float(p['z']) for p in priors])
        rpy = Rotation.from_matrix(m[:, :, :3]).as_euler('xyz', degrees=True)
        track_dir = Path(params['init_guess_file']).parent.name
        initial = np.loadtxt(args.data_root / track_dir / 'init_guess.txt', delimiter=',')
        sensor_rpy = Rotation.from_quat([float(priors[0][k]) for k in ('qx', 'qy', 'qz', 'qw')]).as_euler('xyz', degrees=True)
        original_summaries.append(dict(run=key, count=len(m),
            sensor_z_min_m=float(nominal.min()), sensor_z_max_m=float(nominal.max()),
            z_min_m=float(m[:, 2, 3].min()), z_max_m=float(m[:, 2, 3].max()),
            z_range_m=float(np.ptp(m[:, 2, 3])), max_abs_offset_from_nominal_m=float(np.max(np.abs(m[:, 2, 3]-nominal))),
            roll_deg=stats(rpy[:, 0]), pitch_deg=stats(rpy[:, 1]),
            supplied_initial_rpy_deg=Rotation.from_matrix(initial[:3, :3]).as_euler('xyz', degrees=True).tolist(),
            first_sensor_prior_rpy_deg=sensor_rpy.tolist(),
            initial_attitude_is_not_verified_extrinsic=True,
            result_sha256=sha256(Path(key) / 'result.txt')))
    figure.tight_layout(); figure.savefig(args.output / 'height_comparison.png', dpi=160); plt.close(figure)
    document = dict(script_sha256=sha256(__file__), pairs_sha256=sha256(args.pairs),
        original_full_runs=original_summaries, pairs=reports,
        method=dict(probe='Same deterministic raw front PLY probes for every pose; 0.1 raw-unit voxel representatives, max 6000; not PCL voxel centroids',
                    surface='NN <=3 m, map PCA reliable, abs(nz)>=0.95; median dz per unique target',
                    height_anchor='Existing nominal sensor Z, sigma=1 m, +/-1.5 m total offset; uncalibrated experimental regularizer'),
        limitations=['No localization ground truth or new Kaggle score.',
                     'Horizontal surface candidates are not labelled roads and can be wrong or on different layers.',
                     'Nominal GNSS altitude is constant and is not a calibrated road-height model.',
                     'Dominant lower source surfaces may change identity; their intercept variation alone is not depth-scale error.',
                     'Short carry runs restart Z and cannot reproduce the full trajectory history.',
                     'Smaller map residual or bounded Z does not prove better XY/yaw or physically correct height.'])
    (args.output / 'report.json').write_text(json.dumps(document, indent=2)+'\n')
    print(json.dumps([dict(track=r['track'], start=r['start'], modes=r['mode_summaries']) for r in reports], indent=2))


if __name__ == '__main__':
    main()
