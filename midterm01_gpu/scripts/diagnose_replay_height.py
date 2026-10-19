#!/usr/bin/env python3
"""Read-only saved-cloud height diagnostics; no pose/scale fitting."""
from output_layout import loads_saved_json, saved_path
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from replay_run import sha256
from three_camera_calibration import load_ply, load_binary_map, SurfaceMap


def quantiles(values):
    return dict(zip(('p10', 'median', 'p90'), np.quantile(values, [.1, .5, .9]).tolist())) if len(values) else None


def by_target_median(values, targets):
    return np.array([np.median(values[targets == i]) for i in np.unique(targets)])


def main():
    parser = argparse.ArgumentParser(description='比較保存的影像點雲與 LiDAR 水平表面高度，保留原 pose／尺度。')
    parser.add_argument('replay_run', type=saved_path)
    parser.add_argument('--indices', default='0,166,171,906', help='原定位 run 的 idx，以逗號分隔')
    parser.add_argument('--output', type=saved_path, required=True)
    args = parser.parse_args()
    report = loads_saved_json((args.replay_run/'replay.json').read_text())
    indices = [int(i) for i in args.indices.split(',')]
    rows = {r['idx']: r for r in report['records']}
    if any(i not in rows for i in indices):
        raise ValueError('所選 idx 不在 replay manifest 中')
    cache = report['map_cache']
    if not cache or sha256(cache['file']) != cache['sha256']:
        raise ValueError('需有本次已核對的 binary XYZ map cache')
    scene = SurfaceMap(load_binary_map(cache['file']))
    args.output.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots((len(indices)+1)//2, 2, figsize=(13, 4*((len(indices)+1)//2)), squeeze=False)
    results = []
    for index, axis in zip(indices, axes.flat):
        row = rows[index]
        if sha256(row['path']) != row['sha256']:
            raise ValueError('原始 PLY 已變更')
        raw = load_ply(row['path'], max_points=1000000, leaf=0)
        matrix = np.array(row['pose']).reshape(3,4)
        xyz = raw * report['params']['ply_scale'] @ matrix[:,:3].T + matrix[:,3]
        distances, target = scene.tree.query(xyz)
        normals, quality = scene.surfaces(target)
        # These are horizontal *surface candidates*: ground/roof/other layers
        # are not semantically labelled, and nearest neighbours may be wrong.
        valid = (distances <= 3.) & (quality > 0) & (np.abs(normals[:,2]) >= .95)
        residual = np.einsum('ij,ij->i', normals, xyz-scene.points[target])
        dz = residual[valid] / normals[valid,2]
        unique_dz = by_target_median(dz, target[valid])
        horizontal_range = np.linalg.norm(xyz[:,:2]-matrix[:2,3],axis=1)
        bands = []
        for low, high in [(0,5),(5,10),(10,20),(20,40)]:
            selected = valid & (horizontal_range >= low) & (horizontal_range < high)
            deltas = residual[selected] / normals[selected,2]
            bands.append(dict(range_m=[low,high], source_points=int(selected.sum()),
                              unique_map_targets=int(len(np.unique(target[selected]))),
                              dz_by_unique_target_m=quantiles(by_target_median(deltas,target[selected]))))
        xy_distance = np.linalg.norm(scene.points[:,:2]-matrix[:2,3],axis=1)
        result = dict(idx=index,timestamp_ns=row['timestamp_ns'],source_pose_xyz_m=matrix[:,3].tolist(),
                      source_points=len(xyz),horizontal_candidate_points=int(valid.sum()),
                      unique_horizontal_map_targets=int(len(unique_dz)),
                      signed_vertical_surface_difference_m=quantiles(unique_dz),range_bands=bands,
                      local_map_z_within_3m_xy_m=quantiles(scene.points[xy_distance<3,2]),
                      ply_sha256=row['sha256'])
        results.append(result)
        yaw = np.arctan2(matrix[1,0],matrix[0,0])
        forward = np.array([np.cos(yaw),np.sin(yaw)])
        lateral = np.array([-np.sin(yaw),np.cos(yaw)])
        for points,color,label,size in [(scene.points,'0.55','LiDAR map',2.5),(xyz,'#008bba','Saved image cloud',.7)]:
            relative = points[:,:2]-matrix[:2,3]
            fwd, lat = relative@forward, relative@lateral
            keep = (np.abs(lat)<4)&(fwd>=0)&(fwd<=25)&(points[:,2]-matrix[2,3]>=-4)&(points[:,2]-matrix[2,3]<=6)
            axis.scatter(fwd[keep],points[keep,2]-matrix[2,3],s=size,c=color,alpha=.65,label=label,rasterized=True)
        axis.axhline(0,color='black',linewidth=.6,linestyle='--',label='Source origin Z')
        axis.set(xlim=(0,25),ylim=(-4,6),title=f'Frame {index} | original pose and scale',
                 xlabel='Horizontal distance along saved yaw [m]',ylabel='Map Z minus source-origin Z [m]')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8,loc='upper right')
    for axis in list(axes.flat)[len(indices):]: axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(args.output/'height_profiles.png',dpi=170)
    plt.close(figure)
    document=dict(source_replay=str(args.replay_run.resolve()),source_run=report['run'],map_cache=cache,
                  coordinate_contract=report['coordinate_contract'],pose_and_scale_modified=False,
                  method=dict(max_nn_distance_m=3.,min_absolute_normal_z=.95,
                              aggregation='median per unique map target, then p10/median/p90',
                              signed_difference='n dot (source-map_target) / n_z; positive means source is above selected plane'),
                  limitations=['Nearest neighbours and horizontal candidates are not ground-truth semantic correspondences.',
                               'Map includes roads, walls, roofs and possibly several height layers.',
                               'This is not positioning error and does not justify a constant Z correction.',
                               'Side profiles show a +/-4 m lateral slice, 0-25 m forward, and -4 to +6 m relative Z.'],
                  frames=results)
    (args.output/'height_report.json').write_text(json.dumps(document,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__ == '__main__': main()
