#!/usr/bin/env python3
"""只讀既有點雲的 camera-level 地圖一致性與改善候選來源；不是精度評估。"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from calibrated_rig_geometry import write_ply
from three_camera_calibration import load_ply,SurfaceMap,load_binary_map
from three_camera_source import ply_header,sha256
from analyze_rig_track1 import stats


def read_colored(path):
    lines,offset=ply_header(path)
    if 'format binary_little_endian 1.0' not in lines:raise ValueError('Expected binary calibrated source')
    n=int(next(s.split()[2] for s in lines if s.startswith('element vertex ')))
    packed=np.fromfile(path,dtype=np.dtype([('xyz','<f4',(3,)),('rgb','u1',(3,))]),offset=offset,count=n)
    return packed['xyz'].astype(float),packed['rgb']


def projection_metrics(scene,source,pose,camera_ids):
    rotated=source@pose[:3,:3].T;world=rotated+pose[:3,3]
    dist,targets=scene.tree.query(world);normal,q=scene.surfaces(targets)
    residual=world-scene.points[targets];plane=np.einsum('ij,ij->i',normal,residual)
    reliable=(dist<=2)&(q>0)
    # This independent CPU diagnostic uses the saved pre-C++-voxel source; it is not a replay of the solver.
    info=[]
    for cam in np.unique(camera_ids):
        mask=camera_ids==cam;n=mask.sum();ok=mask&reliable
        jac=np.column_stack((normal,-normal[:,0]*rotated[:,1]+normal[:,1]*rotated[:,0]))
        jac[:,3]*=np.deg2rad(3)
        w=2/(len(source)*.3**2)*q*np.minimum(1,.5/np.maximum(np.abs(plane),1e-12))*reliable
        h=jac[mask].T@(w[mask,None]*jac[mask]);wanted=[0,1,3]
        schur=h[np.ix_(wanted,wanted)]-np.outer(h[wanted,2],h[2,wanted])/max(h[2,2],1e-12)
        grad=(jac[mask].T@(w[mask]*plane[mask]))
        value={'camera':int(cam),'points':int(n),'nn_inlier_fraction':float(np.mean(dist[mask]<=2)),
            'reliable_fraction':float(np.mean(reliable[mask])),'unique_reliable_targets':int(len(np.unique(targets[ok]))),
            'clipped_nn_rmse_m':float(np.sqrt(np.mean(np.minimum(dist[mask],2)**2))),
            'nn_distance_median_m':float(np.median(dist[mask])),
            'reliable_plane_rmse_m':float(np.sqrt(np.mean(plane[ok]**2))) if ok.any() else None,
            'signed_reliable_plane_median_m':float(np.median(plane[ok])) if ok.any() else None,
            'xyyaw_min_information_free_z':float(max(0.,np.linalg.eigvalsh(schur)[0])),
            'gradient_scaled_xyz_yaw':grad.tolist()}
        info.append(value)
    return info


def main(args):
    root=Path(__file__).resolve().parents[1];out=args.output.resolve();out.relative_to((root/'outputs/runs').resolve())
    if (out/'camera_probe.json').exists():raise ValueError('Existing probe cannot be overwritten')
    track=args.track
    full=root/f'outputs/runs/three_camera_full_fast_imported_2610181301/track{track}'
    front=root/'outputs/runs'/({1:'track1_sensor_guided_heading_rts_geometry_directional_cuda_robust_icp_d2d_3011f_70rv9cj4',
                              2:'track2_sensor_guided_heading_rts_geometry_directional_cuda_robust_icp_d2d_3299f_7o_atg6_'}[track])
    rig=Path(json.loads((full/'localization.json').read_text())['run'])
    report=json.loads((full/'report.json').read_text());rposes=np.loadtxt(rig/'result.txt').reshape(-1,3,4);fposes=np.loadtxt(front/'result.txt').reshape(-1,3,4)
    with (front/'sensor_prior.csv').open() as f:prior=list(csv.DictReader(f))
    xyz=np.array([[float(r[k]) for k in ('x','y','z')] for r in prior]);arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz[:,:2],axis=0),axis=1))]
    indices=np.searchsorted(arc,np.linspace(.05,.95,16)*arc[-1]).astype(int)
    map_cache=root/'outputs/runs'/({1:'cuda_ndt_20261017_nlvdk4j5/cache_track1_leaf1.pcd',2:'three_camera_dev_2610181055/cache_track2_leaf1.pcd'}[track])
    scene=SurfaceMap(load_binary_map(map_cache))
    data=root.parent/f'midterm01/data/Track{track}/raw_ply';old_by_stamp={int(p.stem.rsplit('_ts_',1)[1]):p for p in data.glob('*.ply')}
    result=[];provenance=[]
    for idx in indices:
        record=report['frames'][idx];file=full/record['filename'];points,colors=read_colored(file)
        with np.load(file.parent/'labels.npz') as labels:ids=labels['camera_ids'];confidence=labels['confidence']
        if sha256(file)!=record['sha256'] or len(points)!=len(ids):raise ValueError('Invalid source point/camera pairing')
        old=3*load_ply(old_by_stamp[record['timestamp_ns']],max_points=100000,leaf=0)
        probes={'rig_selected':(points,ids),'old_front':(old,np.zeros(len(old),dtype=np.uint8))}
        poses={'front_pose':fposes[idx].copy(),'rig_pose':rposes[idx].copy(),
               'front_xyyaw_rig_z':fposes[idx].copy(),'rig_xyyaw_front_z':rposes[idx].copy()}
        poses['front_xyyaw_rig_z'][2,3]=rposes[idx,2,3];poses['rig_xyyaw_front_z'][2,3]=fposes[idx,2,3]
        for pose_label,pose in poses.items():
            for source_label,(p,c) in probes.items():
                for metrics in projection_metrics(scene,p,pose,c):
                    result.append({'original_idx':int(idx),'split':'audit_fit' if len(provenance)%2==0 else 'audit_holdout',
                        'source':source_label,'pose':pose_label,**metrics})
        provenance.append({'idx':int(idx),'rig_ply_sha256':sha256(file),'labels_sha256':sha256(file.parent/'labels.npz'),
                           'old_front_sha256':sha256(old_by_stamp[record['timestamp_ns']])})
    aggregate={}
    for source in ['rig_selected','old_front']:
        for pose in poses:
            for cam in ([0,1,2] if source=='rig_selected' else [0]):
                subset=[r for r in result if (r['source'],r['pose'],r['camera'])==(source,pose,cam)]
                aggregate[f'{source}/{pose}/camera{cam}']={k:stats([r[k] for r in subset]) for k in
                    ['points','nn_inlier_fraction','reliable_fraction','clipped_nn_rmse_m','unique_reliable_targets','xyyaw_min_information_free_z']}
    document={'scope':'Independent exact CPU NN/PCA on saved selected points before C++ voxel; common probes and poses, not ground-truth errors. 16 distance-quantile scenes, no outcome-based selection.',
        'track':track,'indices':indices.tolist(),'provenance':provenance,'rows':result,'aggregate':aggregate,'map_sha256':sha256(map_cache)}
    (out/'camera_probe.json').write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')
    if args.skip_ablations:
        shutil.copy2(Path(__file__),out/'sources'/Path(__file__).name)
        print(json.dumps({'track':track,'camera_probe_indices':indices.tolist(),'ablation_preparation':False},indent=2));return
    # Four windows selected by route distance only. Same saved depths, calibration and selected-point pool.
    starts=np.searchsorted(arc,np.array([.2,.4,.6,.8])*arc[-1]).astype(int)
    prepared=[]
    for start in starts:
        for name in ['all','front_only','side_only','all_600','front_600']:
            directory=out/'ablation_sources'/f'from{start}_{name}';directory.mkdir(parents=True)
            frames=[]
            for idx in range(start,start+12):
                record=report['frames'][idx];path=full/record['filename'];points,colors=read_colored(path)
                with np.load(path.parent/'labels.npz') as labels:ids=labels['camera_ids']
                selected=np.arange(len(points)) if name.startswith('all') else np.flatnonzero(ids==0 if name.startswith('front') else ids!=0)
                if name.endswith('_600'):
                    if len(selected)<600:raise ValueError('Ablation budget unsupported')
                    selected=selected[np.linspace(0,len(selected)-1,600,dtype=int)]
                file=directory/Path(record['filename']).name;write_ply(file,points[selected],colors[selected])
                frames.append({'original_idx':int(idx),'timestamp_ns':record['timestamp_ns'],'filename':file.name,'sha256':sha256(file),'points':len(selected)})
            manifest={'schema':'calibrated_rig_v1','track':track,'variant':name,'coordinate_frame':'front_camera_FLU',
                'reference_time':'front_camera_bag_timestamp','ply_scale':1.,'experimental':True,
                'per_camera_motion_compensation':True,'fixed_rig_heights':True,'depth_scale':report['depth_scale'],
                'candidate_validation_passed':False,'frames':frames}
            (directory/'report.json').write_text(json.dumps(manifest,indent=2)+'\n')
            prepared.append({'name':name,'start':int(start),'frames':12,'report':str(directory/'report.json')})
    (out/'ablation_plan.json').write_text(json.dumps(prepared,indent=2)+'\n')
    shutil.copy2(Path(__file__),out/'sources'/Path(__file__).name)
    print(json.dumps({'camera_probe_indices':indices.tolist(),'ablation_starts':starts.tolist(),'groups':len(prepared)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--track',type=int,choices=(1,2),default=1);p.add_argument('--skip-ablations',action='store_true')
    main(p.parse_args())
