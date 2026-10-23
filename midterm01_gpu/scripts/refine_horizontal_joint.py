#!/usr/bin/env python3
"""Fine-map surfel factors + planar joint estimation, with fail-closed validation."""
from output_layout import loads_saved_json, saved_path
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import time

import numpy as np
import scipy
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from accuracy.horizontal_cache import prepare_fine_map
from accuracy.surface_cache import binary_xyz_points
from audit_sensor_contract import audit
from horizontal_surfel import (FineSurfelMap, SurfelConfig, select_horizontal_pairs,
                               fit_horizontal_factor, plane_residual_and_jacobian)
from prior_ablation import rows, dump
from kinematic_graph import KinematicConfig
from surfel_joint import (make_problem, interpolate_states, temporal_accept,
                          discount_shared_patches, wrap)
from three_camera_calibration import load_binary_map, load_ply
from three_camera_source import sha256


def select_groups(stamps):
    t = (np.asarray(stamps, np.int64)-int(stamps[0]))/1e9
    groups = []
    for block in np.unique(np.floor(t/2).astype(int)):
        members = np.flatnonzero(np.floor(t/2).astype(int) == block)
        if t[members[-1]]-t[members[0]] < .9: continue
        center = int(members[np.argmin(abs(t[members]-(t[members[0]]+t[members[-1]])/2))])
        ids = [int(np.argmin(abs(t-(t[center]+offset)))) for offset in (-.45, 0., .45)]
        if min(t[ids[1]]-t[ids[0]], t[ids[2]]-t[ids[1]]) >= .35:
            groups.append(ids)
    return groups


def verify_fine_cache(path, data_root, track, required_bounds, config):
    identity = loads_saved_json(path.with_suffix('.pcd.json').read_text())
    if (identity.get('schema') != 'horizontal_fine_map_v1' or identity['track'] != track
            or identity['map_sha256'] != sha256(data_root/f'Track{track}/map.pcd')
            or identity['leaf_m'] != config.leaf_m or identity['cache_sha256'] != sha256(path)):
        raise ValueError('Horizontal map Track/raw-map/leaf/payload identity mismatch')
    actual = np.array([identity['bounds']['min'], identity['bounds']['max']])
    required = np.array([required_bounds['min'], required_bounds['max']])
    if np.any(actual[0] > required[0]) or np.any(actual[1] < required[1]):
        raise ValueError('Explicit fine map does not cover this run association region')
    if binary_xyz_points(path) != identity['points']:
        raise ValueError('Fine map identity point count mismatch')
    return identity


def heldout_at_joint(pool, base, xy, yaw, config):
    test = {k: v[pool['holdout']] for k,v in pool.items()
            if isinstance(v, np.ndarray) and len(v) == len(pool['holdout'])}
    base_yaw = np.arctan2(base[1,0], base[0,0])
    delta = np.r_[xy-base[:2,3], 0., wrap(yaw-base_yaw)/config.yaw_scale_rad]
    # Validate actual proposed XY/yaw while eliminating Z independently.
    def residual(z):
        d = delta.copy(); d[2] = z[0]
        return plane_residual_and_jacobian(d, test, config)[0]
    fit = least_squares(residual, [0.], bounds=(-config.max_translation_m, config.max_translation_m),
                        loss='huber', f_scale=.5)
    before = plane_residual_and_jacobian(np.zeros(4), test, config)[0]
    after = residual(fit.x)
    median = float(np.median(abs(after))); p90 = float(np.quantile(abs(after), .9))
    okay = bool(fit.success and not np.any(fit.active_mask) and median <= .3 and p90 <= .75
                and median <= np.median(abs(before))+.02)
    return {'accepted': okay, 'free_z_m': float(fit.x[0]), 'median_m': median, 'p90_m': p90}


def validate_result(poses, base, stamps, frames):
    if (poses.shape != base.shape or not np.isfinite(poses).all() or len(poses) != len(frames)
            or len(np.unique(stamps)) != len(stamps) or np.any(np.diff(stamps) <= 0)):
        raise ValueError('Final pose/frame integrity mismatch')
    matrices = poses[:, :, :3]
    orthogonality = np.max(abs(np.swapaxes(matrices,1,2)@matrices-np.eye(3)))
    determinants = np.linalg.det(matrices)
    if orthogonality > 1e-5 or np.max(abs(determinants-1)) > 1e-5:
        raise ValueError('Final rotation is not proper orthonormal')
    if not np.array_equal(poses[:,2,:], base[:,2,:]):
        raise ValueError('Postprocessor must preserve original Z and roll/pitch row')
    xy_change = np.linalg.norm(poses[:,:2,3]-base[:,:2,3], axis=1)
    yaw_change = wrap(np.arctan2(poses[:,1,0],poses[:,0,0])-np.arctan2(base[:,1,0],base[:,0,0]))
    return {'frames': len(poses), 'rotation_orthogonality_max_error': float(orthogonality),
            'max_xy_change_m': float(xy_change.max()), 'max_yaw_change_deg': float(np.degrees(abs(yaw_change)).max()),
            'z_and_tilt_row_preserved': True}


def refine(run, data_root, container, fine_map=None):
    start = time.perf_counter(); run, data_root = run.resolve(), data_root.resolve()
    root = Path(__file__).resolve().parents[1]
    run.relative_to((root/'outputs').resolve())
    output = run/'horizontal'; output.mkdir()
    params = loads_saved_json((run/'params.json').read_text())
    validation = loads_saved_json((run/'validation.json').read_text())
    if not validation['output_complete_and_finite']:
        raise ValueError('Base benchmark validation must pass first')
    if (params['localization_profile'] != 'sensor_guided' or params['geometry_mode'] != 'directional'
            or params['sensor_height_mode'] != 'carry' or params['ply_scale'] != 3.
            or validation['sensor_fusion'] != 'motion_rts' or validation['cloud_source'] != 'front'):
        raise ValueError('This ablation requires scored front/motion_rts/directional/carry baseline')
    track = validation['track']; config = SurfelConfig()
    baseline = run/'base'; baseline.mkdir()
    for name in ('result.txt','diagnostic.csv','validation.json'):
        shutil.copyfile(run/name, baseline/name)
    frames = rows(run/'frames.csv'); stamps = np.array([int(r['timestamp_ns']) for r in frames], np.int64)
    base = np.loadtxt(baseline/'result.txt').reshape(-1,3,4)
    validate_result(base, base, stamps, frames)
    source_input = loads_saved_json((run/'source_input.json').read_text())['selected_frames']
    if len(source_input) != len(frames) or any(int(a['timestamp_ns']) != int(b['timestamp_ns'])
            or a['filename'] != b['filename'] for a,b in zip(source_input,frames)):
        raise ValueError('Source manifest mismatch')
    report = {'schema': 'horizontal_surfel_joint_v1', 'track': track, 'config': asdict(config), 'joint_config': asdict(KinematicConfig()),
        'software': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__, 'yaml': yaml.__version__},
        'base_result_sha256': sha256(baseline/'result.txt'), 'base_csv_sha256': sha256(baseline/'diagnostic.csv'),
        'frames_sha256': sha256(run/'frames.csv'), 'source_input_sha256': sha256(run/'source_input.json'),
        'accepted_map_factors': 0, 'applied': False, 'joint_estimation_executed': False,
        'scope': 'experimental planar GNSS/gyro/map joint model, not full IMU preintegration or calibrated covariance',
        'limits': ['Unknown body/IMU/GNSS extrinsics and absolute datum remain unresolved.',
                   'Map holdout and temporal consistency are not ground truth accuracy.',
                   'No GNSS orientation/course, nonholonomic or accelerometer factors; base posterior is not reused as a prior.']}
    # Export current raw bag via ROS; no dependency on historical research runs.
    raw_path = output/'raw_sensors.npz'
    container_dir = '/root/outputs/'+output.relative_to((root/'outputs').resolve()).as_posix()
    command = ['docker','exec',container,'bash','-c',
        'source /opt/ros/humble/setup.bash && exec python3 /root/scripts/audit_input_heading.py export '+
        shlex.join(['--input', f'/root/data/Track{track}/Track{track}.db3', '--output', container_dir+'/raw_sensors.npz'])]
    dump(output/'raw_export_command.json', command)
    with (output/'raw_export.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    from competition_rules import official_pdf_path
    pdf = official_pdf_path()
    dump(output/'sensor_contract.json', audit(raw_path, data_root, track, pdf))
    groups = select_groups(stamps)
    report['temporal_groups'] = len(groups)
    scenes = {}; factors = []; scene_reports = []; group_reports = []
    if groups:
        selected = sorted(set(i for group in groups for i in group)); all_world = []
        raw_clouds = {}
        for i in selected:
            path = data_root/f'Track{track}/raw_ply'/frames[i]['filename']
            if sha256(path) != source_input[i]['sha256']: raise ValueError('PLY changed since baseline run')
            cloud = 3*load_ply(path, max_points=6000, leaf=.1)
            distance = np.linalg.norm(cloud,axis=1); cloud = cloud[(distance >= 1)&(distance <= 30)]
            raw_clouds[i] = cloud
            if len(cloud): all_world.append(cloud@base[i,:,:3].T+base[i,:,3])
        if all_world:
            world = np.concatenate(all_world)
            # 4 m covers the 1.5 m association and 1.8 m normal neighborhoods.
            bounds = {'min': (world.min(axis=0)-4).tolist(), 'max': (world.max(axis=0)+4).tolist()}
            if fine_map:
                fine_map = fine_map.resolve(); fine_map.relative_to((root/'outputs').resolve())
                identity = verify_fine_cache(fine_map, data_root, track, bounds, config)
            else:
                fine_map = prepare_fine_map(output/'fine_map', data_root, track, container, bounds, config.leaf_m)
                identity = loads_saved_json(fine_map.with_suffix('.pcd.json').read_text())
            report['fine_map'] = {'path': str(fine_map), **identity}
            scene = FineSurfelMap(load_binary_map(fine_map), config)
            for i in selected:
                pool = select_horizontal_pairs(scene, raw_clouds[i], base[i])
                diagnostic, factor = fit_horizontal_factor(pool, base[i], config)
                diagnostic.update(idx=i, original_idx=source_input[i]['original_idx'], timestamp_ns=int(stamps[i]))
                scenes[i] = (pool, diagnostic, factor); scene_reports.append(diagnostic)
                print(json.dumps(diagnostic), flush=True)
            for group in groups:
                diagnostics = [scenes[i][1] for i in group]; found = [scenes[i][2] for i in group]
                accepted = temporal_accept(diagnostics, found)
                group_reports.append({'indices': group, 'accepted': accepted,
                    'reason': 'accepted' if accepted else 'spatial_factor_or_temporal_agreement_failed'})
                if accepted: factors.append({**found[1], 'timestamp_ns': int(stamps[group[1]]), 'frame_idx': group[1]})
    dump(output/'observations.json', scene_reports); dump(output/'temporal_groups.json', group_reports)
    factors = discount_shared_patches(factors); dump(output/'map_factors.json', factors)
    report['accepted_map_factors'] = len(factors)
    report['observation_diagnostics_not_accuracy'] = {
        'observations': len(scene_reports),
        'accepted_observations': sum(d['accepted'] for d in scene_reports),
        'fitted_observations': sum('rank' in d for d in scene_reports),
        'rejection_reason_counts': dict(Counter(reason for d in scene_reports for reason in d['reasons'])),
        'rejection_counts_can_overlap': True,
    }
    candidate = base.copy()
    if factors:
        with np.load(raw_path) as source: raw = {k: source[k] for k in source.files}
        problem, origin_ns = make_problem(raw, factors, stamps)
        report['joint_estimation_executed'] = True
        state, solver = problem.solve(); report['solver'] = solver
        query = (stamps-origin_ns)/1e9
        xy, yaw = interpolate_states(problem, state, query)
        base_yaw = np.arctan2(base[:,1,0],base[:,0,0]); correction = wrap(yaw-base_yaw)
        trust = bool(np.max(np.linalg.norm(xy-base[:,:2,3],axis=1)) <= config.max_translation_m
                     and np.max(abs(correction)) <= np.deg2rad(config.max_yaw_deg))
        checks = [heldout_at_joint(scenes[f['frame_idx']][0], base[f['frame_idx']],
                                   xy[f['frame_idx']], yaw[f['frame_idx']], config) for f in factors]
        dump(output/'joint_heldout.json', checks)
        report['joint_local_trust_passed'] = trust
        if solver['success'] and trust and all(c['accepted'] for c in checks):
            candidate[:,:,:3] = Rotation.from_euler('z', correction[:,None]).as_matrix()@base[:,:,:3]
            candidate[:,:2,3] = xy
            candidate[:,2,:] = base[:,2,:]  # exact preservation including decimal-rounding row
            report['applied'] = True; report['reason'] = 'map_joint_and_heldout_checks_passed'
        else: report['reason'] = 'joint_solution_failed_trust_or_heldout_checks; original bytes retained'
        np.savez_compressed(output/'joint_states.npz', time_origin_ns=origin_ns, node_time_s=problem.gt, states=state)
    else: report['reason'] = 'no_reliable_map_factors; original bytes retained, map-free graph skipped'
    report['final_checks'] = validate_result(candidate, base, stamps, frames)
    if report['applied']:
        # Stage and independently reload before publishing either final artifact.
        partial = output/'candidate_result.txt'
        np.savetxt(partial, candidate.reshape(-1,12), fmt='%.12f')
        saved = np.loadtxt(partial).reshape(-1,3,4)
        # Z/tilt row has original precision: serialize it from original lines verbatim.
        lines = partial.read_text().splitlines(); original_lines = (baseline/'result.txt').read_text().splitlines()
        partial.write_text('\n'.join(' '.join(new.split()[:8]+old.split()[8:]) for new,old in zip(lines,original_lines))+'\n')
        saved = np.loadtxt(partial).reshape(-1,3,4); validate_result(saved,base,stamps,frames)
        csv_path = output/'candidate_diagnostic.csv'
        import csv
        with csv_path.open('x',newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['idx','x','y','yaw'])
            writer.writerows((i,p[0,3],p[1,3],np.arctan2(p[1,0],p[0,0])) for i,p in enumerate(saved))
        shutil.copyfile(partial, run/'result.txt'); shutil.copyfile(csv_path,run/'diagnostic.csv')
    report.update(result_sha256=sha256(run/'result.txt'), csv_sha256=sha256(run/'diagnostic.csv'),
        elapsed_seconds=time.perf_counter()-start, output_complete_and_finite=True,
        base_diagnostics_scope='guidance.csv, geometry.csv, height.csv, timings.csv describe base C++ stage; final checks in this report')
    report['csv_changed_from_base'] = report['csv_sha256'] != report['base_csv_sha256']
    report['result_changed_from_base'] = report['result_sha256'] != report['base_result_sha256']
    dump(output/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=saved_path,required=True); parser.add_argument('--data-root',type=saved_path,required=True)
    parser.add_argument('--container',default='sdc_midterm01_gpu'); parser.add_argument('--fine-map',type=saved_path)
    args = parser.parse_args()
    refine(args.run,args.data_root,args.container,args.fine_map)


if __name__ == '__main__': main()
