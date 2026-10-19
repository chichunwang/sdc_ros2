#!/usr/bin/env python3
"""同一 probe／map 的三相機對照；資料信息矩陣不當作定位真值或 covariance。"""
from output_layout import loads_saved_json, saved_path
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from three_camera_calibration import load_binary_map, load_ply
from calibrated_rig_geometry import OPTICAL_TO_FLU, backproject, transform, voxel_indices


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparisons', type=saved_path, nargs='+', required=True)
    p.add_argument('--reconstruction', type=saved_path, required=True)
    p.add_argument('--anchored', type=saved_path, required=True)
    p.add_argument('--data-root', type=saved_path, required=True)
    p.add_argument('--map-cache-track1', type=saved_path, required=True)
    p.add_argument('--map-cache-track2', type=saved_path, required=True)
    p.add_argument('--output', type=saved_path, required=True)
    args = p.parse_args()
    records = [record for path in args.comparisons for record in loads_saved_json(path.read_text())]
    scenes = {t: cKDTree(load_binary_map(getattr(args, f'map_cache_track{t}'))) for t in (1, 2)}
    references, binaries, params, probes, results = {}, set(), {}, {}, []
    depth_scale = loads_saved_json((args.anchored / 'scale_anchor.json').read_text())['depth_scale']
    for r in records:
        run = Path(r['run']); v = loads_saved_json((run / 'validation.json').read_text())
        if r['returncode'] or not v['output_complete_and_finite'] or v['validation_errors']:
            raise ValueError('Failed/invalid comparison')
        key = r['track'], r['kind']
        pose = np.loadtxt(run / 'result.txt').reshape(-1, 3, 4)
        prior = (run / 'sensor_prior.csv').read_bytes()
        binaries.add((run / 'binary_sha256.txt').read_text())
        par = loads_saved_json((run / 'params.json').read_text())
        for field in ('ply_path', 'ply_scale', 'save_path', 'sensor_prior_file'):
            par.pop(field)
        if r['label'] == 'front':
            references[key] = pose, prior
            params[key] = par
        reference, expected_prior = references[key]
        if prior != expected_prior or par != params[key]:
            raise ValueError('Unequal priors/solver parameters')
        if (run / 'frames.csv').read_bytes() != (Path(next(a['run'] for a in records if (a['track'], a['kind'], a['label']) == (*key, 'front'))) / 'frames.csv').read_bytes():
            # Source filenames may differ, but exact local index + timestamp must match.
            a, b = rows(run / 'frames.csv'), rows(Path(next(a['run'] for a in records if (a['track'], a['kind'], a['label']) == (*key, 'front'))) / 'frames.csv')
            if [(x['idx'], x['timestamp_ns']) for x in a] != [(x['idx'], x['timestamp_ns']) for x in b]:
                raise ValueError('Unequal frame times/order')
        metrics = {name: [] for name in ('old_front', 'calibrated_front', 'calibrated_side')}
        for offset, (f, pose_row) in enumerate(zip(rows(run / 'frames.csv'), pose)):
            original = r['start_frame'] + offset
            identity = r['track'], original
            if identity not in probes:
                files = list((args.data_root / f'Track{r["track"]}/raw_ply').glob(f'*_ts_{f["timestamp_ns"]}.ply'))
                if len(files) != 1:
                    raise ValueError('Missing/ambiguous original front probe')
                old = 3 * load_ply(files[0], max_points=6000, leaf=.1)
                group = args.reconstruction / f'track{r["track"]}_{original:06d}'
                new = []
                for camera_id in range(3):
                    saved = np.load(group / f'rig_{camera_id}.npz')
                    cp = backproject(saved['depth_z'] * depth_scale, saved['intrinsics'])
                    q = transform(cp, saved['applied_camera_pose']) @ OPTICAL_TO_FLU.T
                    q = transform(q, saved['bag_from_header_flu'])[saved['mask']]
                    keep = voxel_indices(q, .3)
                    q = q[keep]
                    if len(q) > 3000:
                        q = q[np.linspace(0, len(q) - 1, 3000, dtype=int)]
                    new.append(q)
                probes[identity] = {'old_front': old, 'calibrated_front': new[0], 'calibrated_side': np.vstack(new[1:])}
            for name, source in probes[identity].items():
                distances, _ = scenes[r['track']].query(source @ pose_row[:, :3].T + pose_row[:, 3])
                metrics[name].append(float(np.sqrt(np.mean(np.minimum(distances, 2.)**2))))
        geometry = rows(run / 'geometry.csv')
        xy_information = []
        for g in geometry:
            h = np.array([[float(g[f'scaled_information_r{i}c{j}']) for j in range(4)] for i in range(4)])
            wanted = [0, 1, 3]
            marginal = h[np.ix_(wanted, wanted)] - np.outer(h[wanted, 2], h[2, wanted]) / max(h[2, 2], 1e-12)
            xy_information.append(float(max(0., np.linalg.eigvalsh(marginal)[0])))
        d = pose[:, :, 3] - reference[:, :, 3]
        dyaw = np.arctan2(pose[:, 1, 0], pose[:, 0, 0]) - np.arctan2(reference[:, 1, 0], reference[:, 0, 0])
        dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))
        results.append({**{k: r[k] for k in ('track', 'kind', 'label', 'start_frame', 'run')},
                        'accepted': v['guidance_not_accuracy']['map_accepted_frames'],
                        'rank_histogram': v['geometry_diagnostics_not_accuracy']['rank_histogram'],
                        'input_points': [int(g['input_points']) for g in geometry],
                        'unique_targets': [int(g['unique_targets']) for g in geometry],
                        'xyyaw_information_min_eigenvalue_after_free_z': xy_information,
                        'common_probe_clipped_nn_rmse_m': metrics,
                        'xy_difference_to_front_rms_m': float(np.sqrt(np.mean(np.sum(d[:, :2]**2, axis=1)))),
                        'z_difference_to_front_rms_m': float(np.sqrt(np.mean(d[:, 2]**2))),
                        'yaw_difference_to_front_rms_deg': float(np.rad2deg(np.sqrt(np.mean(dyaw**2)))),
                        'registration_seconds': v['timing_sums']['ndt_seconds'], 'exact_prior_equal': True})
    if len(binaries) != 1:
        raise ValueError('Comparisons used different binaries')
    aggregate = {}
    for label in dict.fromkeys(r['label'] for r in results):
        subset = [r for r in results if r['label'] == label]
        hist = {str(rank): sum(r['rank_histogram'][str(rank)] for r in subset) for rank in range(5)}
        aggregate[label] = {'frames': 2 * len(subset), 'accepted': sum(r['accepted'] for r in subset), 'rank_histogram': hist,
                            'median_xyyaw_min_information': float(np.median([x for r in subset for x in r['xyyaw_information_min_eigenvalue_after_free_z']])),
                            'probe_rmse_m': {name: float(np.sqrt(np.mean([x*x for r in subset for x in r['common_probe_clipped_nn_rmse_m'][name]]))) for name in metrics},
                            'mean_registration_ms': 1000 * sum(r['registration_seconds'] for r in subset) / (2 * len(subset))}
    report = {'scope': 'Identical poses evaluated on common probes; capped NN to same 1m map is NOT ATE. New probes use stereo-anchor scale; old/new source bias remains.',
              'information': 'Data-only normal information, state scales 1m/1m/1m/3deg; eliminate Z before XY/yaw eigenanalysis. Not calibrated covariance.',
              'same_binary_exact_prior_solver_parameters': True, 'runs': results, 'aggregate': aggregate}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
