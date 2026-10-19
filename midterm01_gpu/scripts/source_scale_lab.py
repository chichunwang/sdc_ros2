#!/usr/bin/env python3
"""以時間對應估共用來源尺度，分開檢查跨場景與水平 map；不寫提交 pose。"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from accuracy.benchmark import create_run_directory
from accuracy.horizontal_cache import prepare_fine_map
from horizontal_surfel import FineSurfelMap, SurfelConfig, select_horizontal_pairs, fit_horizontal_factor
from source_rigidity import statistics, split_cells, RigidityConfig
from three_camera_calibration import load_ply, load_binary_map
from three_camera_source import sha256


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def prior_poses(run):
    rows=list(csv.DictReader((run/'sensor_prior.csv').open()))
    poses=np.zeros((len(rows),3,4))
    poses[:,:,:3]=Rotation.from_quat([[float(r[k]) for k in ('qx','qy','qz','qw')] for r in rows]).as_matrix()
    poses[:,:,3]=[[float(r[k]) for k in ('x','y','z')] for r in rows]
    stamps=[int(r['timestamp_ns']) for r in rows]
    if len(set(stamps))!=len(stamps):raise ValueError('Duplicate prior timestamps')
    return dict(zip(stamps,poses))


def scale_terms(a,b,pose_a,pose_b):
    """alpha*(R_b^T R_a a - b) + R_b^T(t_a-t_b) = 0.

    Source points already include the base scale. A common alpha scales both
    times; no frame-dependent scale, map target or submission score is fitted.
    """
    r=pose_b[:,:3].T@pose_a[:,:3]
    t=pose_b[:,:3].T@(pose_a[:,3]-pose_b[:,3])
    return a@r.T-b,np.broadcast_to(t,a.shape).copy()


def fit_common_scale(groups,robust_m=.3):
    if not groups or any(len(d)<4 or d.shape!=t.shape for d,t in groups):
        raise ValueError('Need nonempty paired scale observations')
    if not all(np.isfinite(d).all() and np.isfinite(t).all() for d,t in groups):
        raise ValueError('Nonfinite scale observations')
    if sum(float(np.sum(d*d)) for d,_ in groups)<1e-8:
        raise ValueError('No scale sensitivity')
    def residual(x):
        result=[]
        for d,t in groups:
            e=np.exp(x[0])*d+t
            q=np.sum(e*e,axis=1)/robust_m**2
            w=np.sqrt(2/(np.sqrt(1+q)+1))
            # Each time pair contributes equal weight, regardless of pixels.
            result.append((e*w[:,None]/np.sqrt(len(e))).ravel())
        return np.concatenate(result)
    fit=least_squares(residual,[0.],bounds=([np.log(.25)],[np.log(8.)]),max_nfev=100,
                      ftol=1e-10,xtol=1e-10,gtol=1e-10)
    return dict(multiplier=float(np.exp(fit.x[0])),success=bool(fit.success),at_bound=bool(np.any(fit.active_mask)))


def run(args):
    root=Path(__file__).resolve().parents[1]
    output=create_run_directory(root/'outputs/run','source_scale_validation')
    print('Run:',output,flush=True);start=time.perf_counter()
    source=args.rigidity_run.resolve()/'rigidity'
    evidence=json.loads((source/'report.json').read_text());plan=json.loads((source/'plan.json').read_text())
    config=RigidityConfig(**json.loads((source/'params.json').read_text())['config'])
    snapshot=output/'sources';snapshot.mkdir()
    for name in ('source_scale_lab.py','source_rigidity.py','horizontal_surfel.py','three_camera_calibration.py','three_camera_source.py'):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    dump(output/'params.json',dict(command=sys.argv,rigidity_run=str(args.rigidity_run.resolve()),
        source_report_sha256=sha256(source/'report.json'),source_plan_sha256=sha256(source/'plan.json'),
        fit_selection='Original straight/design motion windows, train image cells only; eligibility independent of map residual',
        min_prior_displacement_m=.3,fit_start_by_track={'1':905,'2':1045},
        candidate_scope='exploratory hypothesis after short-lag audit; not a preregistered calibration',
        source_sha256={p.name:sha256(p) for p in snapshot.iterdir()},
        pose_source='sensor_prior.csv; never result.txt for temporal scale fit'))
    references={int(k):v for k,v in plan['references'].items()};priors={};saved={};frames={}
    for track,ref in references.items():
        baseline=Path(ref['path'])
        for name,digest in ref['sha256'].items():
            if sha256(baseline/name)!=digest:raise ValueError('Baseline evidence changed')
        priors[track]=prior_poses(baseline)
        saved[track]=np.loadtxt(baseline/'result.txt').reshape(-1,3,4)
        frames[track]=list(csv.DictReader((baseline/'frames.csv').open()))
    pairs=[];train_terms=[];train_tracks=set()
    for group in evidence['groups']:
        track=group['track'];idxs=group['indices']
        calibration_group=group['selection']=='original_motion_window' and idxs[1]=={1:925,2:1065}[track]
        for pair in group['pairs']:
            if not all(name in pair['models'] for name in ('rigid','similarity')):continue
            first,last=pair['indices'];path=source/f"group_{group['group_id']:02d}_track{track}"/f'{first}_{last}_pairs.npz'
            data=np.load(path);a,b=data['a'],data['b'];held=split_cells(data['pixels_a'],config)
            pa,pb=[priors[track][int(frames[track][i]['timestamp_ns'])] for i in (first,last)]
            d,t=scale_terms(a,b,pa,pb)
            displacement=float(np.linalg.norm(t[0]))
            eligible=displacement>=.3
            if calibration_group and eligible:
                train_terms.append((d[~held],t[~held]));train_tracks.add(track)
            pairs.append(dict(group=group['group_id'],track=track,indices=[first,last],calibration_group=calibration_group,
                motion_eligible=eligible,prior_displacement_m=displacement,d=d,t=t,holdout=held,
                pairs_file=str(path),pairs_sha256=sha256(path)))
    candidate=fit_common_scale(train_terms)
    alpha=candidate['multiplier'];candidate['ply_scale']=config.ply_scale*alpha
    candidate.update(training_pairs=len(train_terms),training_tracks=sorted(train_tracks),
        cross_track_training_supported=train_tracks=={1,2})
    records=[]
    for p in pairs:
        d,t,held=p['d'],p['t'],p['holdout']
        record={k:v for k,v in p.items() if k not in ('d','t','holdout')}
        record['original_holdout_m']=statistics(np.linalg.norm(d[held]+t[held],axis=1))
        record['candidate_holdout_m']=statistics(np.linalg.norm(alpha*d[held]+t[held],axis=1))
        if p['motion_eligible']:
            local=fit_common_scale([(d[~held],t[~held])]);record['local_training_scale_diagnostic']=local
        records.append(record)
    held=[p for p in records if not p['calibration_group'] and p['motion_eligible']]
    candidate['validation_pairs']=len(held)
    candidate['validation_tracks']=sorted({p['track'] for p in held})
    candidate['validation_scale_relative_disagreement']=statistics([
        abs(p['local_training_scale_diagnostic']['multiplier']/alpha-1) for p in held])
    candidate['validation_original_median_m']=statistics([p['original_holdout_m']['median'] for p in held])
    candidate['validation_candidate_median_m']=statistics([p['candidate_holdout_m']['median'] for p in held])
    candidate['validation_lower_median_pairs']=sum(p['candidate_holdout_m']['median']<p['original_holdout_m']['median'] for p in held)
    candidate['approved_for_production']=False
    dump(output/'temporal_scale.json',dict(candidate=candidate,pairs=records,
        limits=['GNSS-derived prior motion is not truth; common GNSS biases are not calibrated here.',
                'Unknown camera/IMU lever arm and rotation, dynamic matches, relative depth changes may confound scale.',
                'Only train cells of the original straight/design windows fit the shared scale.',
                'Other group-specific scale fits are diagnostics, never applied to source or used to retune shared scale.']))
    print(json.dumps(candidate),flush=True)
    # Frozen one-factor map ablation: same saved pose, same thresholds, same
    # original source vertices. Rescale source only; no solver/gate tuning.
    map_reports=[]
    for track in (1,2):
        cache=getattr(args,f'fine_map_track{track}').resolve()
        identity=json.loads(cache.with_suffix('.pcd.json').read_text())
        if (identity['schema']!='horizontal_fine_map_v1' or identity['track']!=track or identity['leaf_m']!=.3
                or identity['cache_sha256']!=sha256(cache)
                or identity['map_sha256']!=sha256(args.data_root/f'Track{track}/map.pcd')):
            raise ValueError('Fine map identity mismatch')
        prepared=[];required=[]
        margin=SurfelConfig().radius_large_m+SurfelConfig().match_radius_m
        for group in evidence['groups']:
            if group['track']!=track:continue
            idx=group['indices'][1];entry=group['sources'][1];path=Path(entry['path'])
            if sha256(path)!=entry['sha256']:raise ValueError('Source PLY changed')
            raw=load_ply(path,max_points=6000,leaf=.1)*config.ply_scale
            prepared.append((group,entry,raw))
            for multiplier in (1.,alpha):
                required.append(raw*multiplier@saved[track][idx,:,:3].T+saved[track][idx,:,3])
        all_world=np.concatenate(required)
        needed_min=all_world.min(axis=0)-margin;needed_max=all_world.max(axis=0)+margin
        covered=np.all(needed_min>=identity['bounds']['min']) and np.all(needed_max<=identity['bounds']['max'])
        if not covered and args.expand_map_cache:
            bounds=dict(min=np.minimum(np.floor(needed_min),identity['bounds']['min']).tolist(),
                        max=np.maximum(np.ceil(needed_max),identity['bounds']['max']).tolist())
            print(f'Track{track}: expand crop for both frozen scale variants; no localization',flush=True)
            cache=prepare_fine_map(output/'fine_map',args.data_root,track,args.container,bounds)
            identity=json.loads(cache.with_suffix('.pcd.json').read_text())
        map_points=load_binary_map(cache)
        if len(map_points)!=identity['points']:raise ValueError('Fine map point count mismatch')
        scene=FineSurfelMap(map_points)
        dump(output/f'map_identity_track{track}.json',identity)
        for group,entry,raw in prepared:
            idx=group['indices'][1]
            record=dict(track=track,idx=idx,group=group['group_id'],ply_sha256=entry['sha256'],variants={})
            for label,multiplier in [('original',1.),('temporal_shared_scale',alpha)]:
                pool=select_horizontal_pairs(scene,raw*multiplier,saved[track][idx])
                report,factor=fit_horizontal_factor(pool,saved[track][idx])
                world=raw*multiplier@saved[track][idx,:,:3].T+saved[track][idx,:,3]
                margin=scene.config.radius_large_m+scene.config.match_radius_m
                inside=np.all((world>=np.array(identity['bounds']['min'])+margin)
                              &(world<=np.array(identity['bounds']['max'])-margin),axis=1)
                report['source_fraction_inside_cache_with_search_margin']=float(inside.mean())
                report['cache_fully_covers_source_search']=bool(inside.all())
                if not inside.all():
                    report['accepted']=False
                    report['reasons'].append('cache_does_not_cover_all_source_search')
                record['variants'][label]=report
            map_reports.append(record)
            print(json.dumps(dict(track=track,idx=idx,variants={k:dict(patches=v['patches'],accepted=v['accepted'],reasons=v['reasons'])
                for k,v in record['variants'].items()})),flush=True)
            dump(output/'map_scale_ablation.json',map_reports)
    unchanged=all(sha256(Path(ref['path'])/name)==digest for ref in references.values() for name,digest in ref['sha256'].items())
    summary=dict(candidate=candidate,map_frames=len(map_reports),
        map_accepted={label:sum(p['variants'][label]['accepted'] for p in map_reports) for label in ('original','temporal_shared_scale')},
        baseline_bytes_unchanged=unchanged,full_localization_executed=False,new_pose_outputs=0,
        elapsed_seconds=time.perf_counter()-start,
        decision='Research only; do not replace scored source scale. No ground-truth accuracy improvement established.')
    dump(output/'summary.json',summary)
    info=json.loads((output/'test_info.json').read_text());info.update(algorithm='source_temporal_scale_v1',status='completed',reported_score=None)
    dump(output/'test_info.json',info)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);root=Path(__file__).resolve().parents[1]
    p.add_argument('--rigidity-run',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=root.parent/'midterm01/data')
    p.add_argument('--expand-map-cache',action='store_true',help='Explicitly rebuild inadequate crop from raw map in this run')
    p.add_argument('--container',default='sdc_midterm01_gpu')
    for track in (1,2):p.add_argument(f'--fine-map-track{track}',type=Path,required=True)
    run(p.parse_args())
