#!/usr/bin/env python3
"""Experimental absolute height from paired lower surfaces; no fixed mount height."""
from output_layout import loads_saved_json, saved_path
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from analyze_height_modes import lower_plane_probe
from three_camera_calibration import load_ply, load_binary_map, SurfaceMap
from three_camera_source import sha256


def smooth_references(stamps, observations, half_window=.25, max_gap=1., max_edge=.3):
    """Integer timestamps first; bounded interpolation, symmetric time median."""
    stamps = np.asarray(stamps, dtype=np.int64)
    if len(stamps) == 0 or np.any(np.diff(stamps) <= 0):
        raise ValueError('Height timestamps must strictly increase')
    time = (stamps-stamps[0])/1e9
    values = np.asarray(observations, dtype=float)
    good = np.isfinite(values)
    if good.sum() < 2 or np.max(np.diff(time[good]), initial=0) > max_gap:
        raise ValueError('Insufficient surface support or gap exceeds 1 s')
    if time[good][0]-time[0] > max_edge or time[-1]-time[good][-1] > max_edge:
        raise ValueError('Surface references do not cover endpoints within 0.3 s')
    filled = np.interp(time, time[good], values[good])
    return np.array([np.median(filled[np.abs(time-t) <= half_window]) for t in time]), good


def compatible_planes(source, target, max_normal_difference_deg=10.):
    if source['status'] != 'candidate' or target['status'] != 'candidate':
        return False
    for plane in (source, target):
        if plane['support'] < 20 or plane['support']/plane['candidate_points'] < .35:
            return False
        if plane['competing_plane_support_fraction'] >= .8 or plane['origin_extrapolation_proxy_m'] > .3:
            return False
    normals = [np.array([-p['forward_slope'], -p['lateral_slope'], 1.]) for p in (source, target)]
    cosine = np.dot(*normals)/np.prod([np.linalg.norm(n) for n in normals])
    return bool(cosine >= np.cos(np.deg2rad(max_normal_difference_deg)))


def fit_tilt_reference(source, target, rotation, max_angle_deg=15.):
    """Align plane normals with exactly fixed ZYX yaw; bounded geometric tilt.

    No initial-matrix extrinsic or assumed sensor mounting height is used.
    """
    roll0,pitch0,yaw=Rotation.from_matrix(rotation).as_euler('xyz')
    heading=Rotation.from_euler('z',yaw).as_matrix()
    a=np.array([-source['forward_slope'],-source['lateral_slope'],1.]);a/=np.linalg.norm(a)
    b=np.array([-target['forward_slope'],-target['lateral_slope'],1.]);b/=np.linalg.norm(b)
    normal_raw=rotation.T @ heading @ a
    radius=np.hypot(normal_raw[1],normal_raw[2])
    if radius < 1e-8 or abs(b[1]) > radius+1e-10:
        raise ValueError('Plane normal cannot be aligned with fixed yaw')
    phase=np.arctan2(-normal_raw[2],normal_raw[1])
    angle=np.arccos(np.clip(b[1]/radius,-1,1))
    wrap=lambda v: np.arctan2(np.sin(v),np.cos(v))
    roll=min([wrap(phase+angle),wrap(phase-angle)],key=lambda r: abs(wrap(r-roll0)))
    intermediate=Rotation.from_euler('x',roll).apply(normal_raw)
    pitch=wrap(np.arctan2(intermediate[2],intermediate[0])-np.arctan2(b[2],b[0]))
    corrected=Rotation.from_euler('xyz',[roll,pitch,yaw]).as_matrix()
    correction_deg=np.degrees(Rotation.from_matrix(corrected @ rotation.T).magnitude())
    if correction_deg > max_angle_deg+1e-8:
        raise ValueError('Plane tilt correction exceeds angle bound')
    # Plane signed distance from origin is invariant under rotation.
    source_intercept=source['relative_z_at_source_origin_m']/np.sqrt(1+source['forward_slope']**2+source['lateral_slope']**2)/b[2]
    return corrected, source_intercept, correction_deg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=saved_path, required=True)
    parser.add_argument('--data-root', type=saved_path, required=True)
    parser.add_argument('--map-cache', type=saved_path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    source = loads_saved_json((run/'source_input.json').read_text())
    params = loads_saved_json((run/'params.json').read_text())
    fit_tilt=params['sensor_height_mode']=='surface_tilt'
    if source['cloud_source'] != 'front':
        raise ValueError('Surface model is only evaluated for original front PLY')
    with (run/'sensor_prior.csv').open() as stream:
        priors = list(csv.DictReader(stream))
    frames = source['selected_frames']
    if len(frames) != len(priors):
        raise ValueError('Source/prior count mismatch')
    identity = loads_saved_json(args.map_cache.with_suffix(args.map_cache.suffix+'.json').read_text())
    track_dir = Path(params['map_path']).name
    if identity['source_map'] != str((args.data_root/track_dir/'map.pcd').resolve()) or identity['map_voxel_leaf_size'] != .3:
        raise ValueError('Map cache path/leaf mismatch')
    if sha256(args.data_root/track_dir/'map.pcd') != identity['map_sha256']:
        raise ValueError('Map source hash mismatch')
    if identity.get('cache_sha256') and sha256(args.map_cache) != identity['cache_sha256']:
        raise ValueError('Height map cache hash mismatch')
    if 'height_band_bounds' in identity:
        bounds=identity['height_band_bounds']
        for p in priors:
            center=np.array([float(p[k]) for k in ('x','y','z')])
            if np.any(np.asarray(bounds['min']) > center-np.array([22,22,6])+1e-6) or np.any(np.asarray(bounds['max']) < center+np.array([22,22,-.3])-1e-6):
                raise ValueError('Height map crop does not cover this prior window and search band')
    points = load_binary_map(args.map_cache)
    scene = SurfaceMap(points)
    tree = scene.tree
    observations, evidence, stamps, tilt_roll, tilt_pitch, yaw_series = [], [], [], [], [], []
    for index, (frame, prior) in enumerate(zip(frames, priors)):
        stamp = int(prior['timestamp_ns'])
        if stamp != frame['timestamp_ns']:
            raise ValueError('Surface/prior timestamp mismatch')
        path = args.data_root/track_dir/'raw_ply'/frame['filename']
        if sha256(path) != frame['sha256']:
            raise ValueError('Surface input PLY hash mismatch')
        raw = load_ply(path, max_points=6000, leaf=.1)*params['ply_scale']
        rotation = Rotation.from_quat([float(prior[k]) for k in ('qx', 'qy', 'qz', 'qw')]).as_matrix()
        center = np.array([float(prior[k]) for k in ('x', 'y', 'z')])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        a = lower_plane_probe(raw, rotation)
        ids = np.asarray(tree.query_ball_point(center,22),dtype=int)
        normals, quality = scene.surfaces(ids)
        ids = ids[(quality > 0)&(np.abs(normals[:,2]) >= 1/np.sqrt(1+.35**2))]
        b = lower_plane_probe(points[ids]-center, np.eye(3), yaw=yaw, min_z=-6.)
        valid = compatible_planes(a,b,15. if fit_tilt else 10.)
        corrected=rotation; source_intercept=a.get('relative_z_at_source_origin_m'); tilt_angle=None
        if valid and fit_tilt:
            try:
                corrected,source_intercept,tilt_angle=fit_tilt_reference(a,b,rotation)
            except ValueError:
                valid=False
        estimate = center[2]+b['relative_z_at_source_origin_m']-source_intercept if valid else None
        observations.append(estimate if valid else np.nan); stamps.append(stamp)
        rpy=Rotation.from_matrix(corrected).as_euler('xyz')
        tilt_roll.append(rpy[0] if valid else np.nan);tilt_pitch.append(rpy[1] if valid else np.nan);yaw_series.append(yaw)
        evidence.append(dict(idx=index, original_idx=frame['original_idx'], timestamp_ns=stamp,
                             supported=valid, raw_height_reference_z=estimate, source_plane=a, map_plane=b,
                             geometric_tilt_correction_deg=tilt_angle,
                             ply_sha256=frame['sha256']))
    (run/'height_reference.frames.json').write_text(json.dumps(evidence, indent=2)+'\n')
    # Never invent a long missing road segment or silently use GNSS altitude.
    references, good = smooth_references(stamps, observations)
    if fit_tilt:
        roll,_=smooth_references(stamps,tilt_roll);pitch,_=smooth_references(stamps,tilt_pitch)
        orientations=Rotation.from_euler('xyz',np.column_stack([roll,pitch,yaw_series]))
        original=Rotation.from_quat([[float(p[k]) for k in ('qx','qy','qz','qw')] for p in priors])
        if np.degrees((orientations*original.inv()).magnitude()).max() > 15.00001:
            raise ValueError('Smoothed geometric tilt exceeds 15 degree bound')
    with (run/'height_reference.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['timestamp_ns','z','qx','qy','qz','qw'] if fit_tilt else ['timestamp_ns','z'])
        if fit_tilt:
            writer.writerows((s,z,*q) for s,z,q in zip(stamps,references,orientations.as_quat()))
        else:
            writer.writerows(zip(stamps,references))
    report = dict(model='paired_lower_surface_v2', experimental=True, frames=len(frames),
        supported_frames=int(good.sum()), missing_original_indices=[f['original_idx'] for f,g in zip(frames,good) if not g],
        source_origin_height_assumed=False, source_rotation_modified=fit_tilt, source_scale_modified=False,
        max_geometric_tilt_deg=float(np.degrees((orientations*original.inv()).magnitude()).max()) if fit_tilt else 0.,
        geometry_semantic_road_verified=False, nominal_gnss_z_is_truth=False,
        method=dict(reference='map plane intercept minus rotated source plane intercept; invariant source plane distance used when fitting tilt',
                    candidate='forward 1-15 m, lateral +/-5 m; source relative z -4..-0.3, map relative to nominal z -6..-0.3',
                    ransac_trials=200, seed=20261019, vertical_inlier_threshold_m=.15,
                    min_inliers=20, min_support_fraction=.35, max_normal_difference_deg=15 if fit_tilt else 10,
                    max_origin_extrapolation_proxy_m=.3, proxy_effective_support_cap=20,
                    attitude='bounded geometric roll/pitch, yaw fixed to exact sensor prior' if fit_tilt else 'original IMU orientation',
                    map_candidates='Reliable local PCA surfaces only, normal slope <=0.35; walls/volumetric mixtures excluded before plane voting',
                    interpolation_max_gap_s=1., endpoint_max_gap_s=.3, symmetric_median_half_window_s=.25),
        reference_z_min_m=float(references.min()), reference_z_max_m=float(references.max()),
        map_cache_sha256=sha256(args.map_cache), height_map_voxel_leaf_size=.3,
        registration_map_voxel_leaf_size=params['map_voxel_leaf_size'], source_input_sha256=sha256(run/'source_input.json'),
        sensor_prior_sha256=sha256(run/'sensor_prior.csv'), reference_sha256=sha256(run/'height_reference.csv'),
        code_sha256={p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name('analyze_height_modes.py'))},
        limitations=['Lower surfaces can be different road layers, vehicle roofs or reconstruction artifacts.',
                    'Normal agreement and support gates do not verify semantic correspondences.',
                    'Symmetric time median is offline, suppresses short height features, and is not calibrated covariance.'])
    (run/'height_reference.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
