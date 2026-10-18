#!/usr/bin/env python3
"""已評分 Track 的前相機／三相機完整輸出診斷；不估真值誤差。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def rows(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def stats(values):
    a=np.asarray(values,dtype=float)
    if not len(a) or not np.all(np.isfinite(a)):raise ValueError('Invalid finite diagnostic values')
    return {'mean':float(a.mean()),'median':float(np.median(a)),'p10':float(np.quantile(a,.1)),
            'p90':float(np.quantile(a,.9)),'p99':float(np.quantile(a,.99)),'max':float(a.max()),
            'min':float(a.min()),'rms':float(np.sqrt(np.mean(a*a)))}


def numbers(data,key):return np.array([float(r[key]) for r in data])


def information(data):
    h=np.array([[[float(g[f'scaled_information_r{i}c{j}']) for j in range(4)] for i in range(4)] for g in data])
    wanted=[0,1,3];marginal=h[:,wanted][:,:,wanted]-h[:,wanted,2][:,:,None]*h[:,2,wanted][:,None,:]/np.maximum(h[:,2,2],1e-12)[:,None,None]
    return np.maximum(np.linalg.eigvalsh(marginal),0.)


def analyze(args):
    root=Path(__file__).resolve().parents[1];out=args.output.resolve()
    out.relative_to((root/'outputs/runs').resolve())
    if (out/'full_comparison.json').exists():raise ValueError('Existing diagnostic evidence cannot be overwritten')
    out.mkdir(exist_ok=True)
    sources=out/'sources';sources.mkdir(exist_ok=True);shutil.copy2(Path(__file__),sources/Path(__file__).name)
    track=args.track;count=3011 if track==1 else 3299
    old_runs={1:'track1_sensor_guided_heading_rts_geometry_directional_cuda_robust_icp_d2d_3011f_70rv9cj4',
              2:'track2_sensor_guided_heading_rts_geometry_directional_cuda_robust_icp_d2d_3299f_7o_atg6_'}
    front=root/'outputs/runs'/old_runs[track]
    full=root/f'outputs/runs/three_camera_full_fast_imported_2610181301/track{track}'
    rig=Path(json.loads((full/'localization.json').read_text())['run'])
    sets={};table=[]
    for label,p in [('front',front),('rig',rig)]:
        validation=json.loads((p/'validation.json').read_text())
        if not validation['output_complete_and_finite'] or validation['frames']!=count:raise ValueError('Invalid full Track')
        g=rows(p/'guidance.csv');geom=rows(p/'geometry.csv');f=rows(p/'frames.csv')
        if len(g)!=count or len(geom)!=count or len(f)!=count:raise ValueError('Wrong row count')
        pose=np.loadtxt(p/'result.txt').reshape(-1,3,4)
        xy=np.hypot(numbers(g,'dx'),numbers(g,'dy'));yaw=np.rad2deg(np.abs(numbers(g,'dyaw')))
        n=numbers(geom,'input_points');reliable=numbers(geom,'reliable_points');inliers=numbers(geom,'euclidean_inliers')
        eig=information(geom)
        record={'run':str(p),'rows':len(g),'csv_sha256':sha(p/'diagnostic.csv'),'prior_sha256':sha(p/'sensor_prior.csv'),
            'binary_sha256_record':(p/'binary_sha256.txt').read_text(),
            'accepted':int(numbers(g,'map_accepted').sum()),'xy_correction_m':stats(xy),'abs_yaw_correction_deg':stats(yaw),
            'height_m':stats(pose[:,2,3]),'height_correction_m':stats(numbers(g,'dz')),
            'input_points':stats(n),'inlier_fraction':stats(inliers/n),'reliable_fraction':stats(reliable/n),
            'unique_targets':stats(numbers(geom,'unique_targets')),
            'plane_rmse_m':stats(numbers(geom,'plane_rmse_m')),
            'clipped_nn_rmse_m':stats(numbers(g,'final_clipped_rmse')),
            'objective_reduction':stats(numbers(g,'prior_cost')-numbers(g,'final_cost')),
            'rank_histogram':{str(i):int(np.sum(numbers(geom,'observable_rank')==i)) for i in range(5)},
            'xyyaw_smallest_information_after_free_z':stats(eig[:,0]),
            'fraction_xy_correction_lt_5cm':float(np.mean(xy<.05)),
            'fraction_abs_yaw_correction_lt_0_5deg':float(np.mean(yaw<.5)),
            'fraction_at_xy_bound':float(np.mean(xy>=1.5-1e-5)),
            'fraction_at_yaw_bound':float(np.mean(yaw>=8.-1e-5)),
            'normal_only_vs_prior_diagonal_median':(2*np.median(np.array([[float(row[f'scaled_information_r{i}c{i}']) for i in range(4)] for row in geom]),axis=0)/np.array([4.,4.,1.,1.])).tolist(),
            'information_interpretation':'Scaled prior diag(4,4,1,1). Diagnostic H times normal_gain 2, excludes tangent contribution and uses normal-only robust weights; not full posterior covariance.'}
        table.append({'label':label,**record});sets[label]={'guidance':g,'geometry':geom,'frames':f,'pose':pose,'eigenvalues':eig}
    a,b=sets['front'],sets['rig']
    if [(r['idx'],r['timestamp_ns']) for r in a['frames']]!=[(r['idx'],r['timestamp_ns']) for r in b['frames']]:raise ValueError('Unmatched frames')
    if sha(front/'sensor_prior.csv')!=sha(rig/'sensor_prior.csv'):raise ValueError('Unequal original priors')
    if any(not np.array_equal(numbers(a['guidance'],k),numbers(b['guidance'],k)) for k in ['prior_x','prior_y','prior_yaw']):raise ValueError('Unequal applied XY/yaw prior')
    d=b['pose'][:,:,3]-a['pose'][:,:,3]
    dyaw=np.arctan2(b['pose'][:,1,0],b['pose'][:,0,0])-np.arctan2(a['pose'][:,1,0],a['pose'][:,0,0]);dyaw=np.arctan2(np.sin(dyaw),np.cos(dyaw))
    frames=json.loads((full/'report.json').read_text())['frames']
    counts=np.array([[r['camera_counts'][str(i)] for i in range(3)] for r in frames]);fractions=counts/counts.sum(axis=1)[:,None]
    paired={'xy_delta_m':stats(np.linalg.norm(d[:,:2],axis=1)),'abs_yaw_delta_deg':stats(np.rad2deg(np.abs(dyaw))),
            'z_delta_m':stats(d[:,2]),'fraction_xy_delta_lt_5cm':float(np.mean(np.linalg.norm(d[:,:2],axis=1)<.05)),
            'fraction_xy_delta_lt_10cm':float(np.mean(np.linalg.norm(d[:,:2],axis=1)<.1)),
            'camera_count':{str(i):stats(counts[:,i]) for i in range(3)},'camera_fraction':{str(i):stats(fractions[:,i]) for i in range(3)}}
    per_frame=[]
    for idx in range(count):
        row={'idx':idx,'timestamp_ns':a['frames'][idx]['timestamp_ns'],'xy_delta_m':np.linalg.norm(d[idx,:2]),
             'yaw_delta_deg':np.rad2deg(dyaw[idx]),'z_delta_m':d[idx,2]}
        for label,s in sets.items():
            g=s['guidance'][idx];geom=s['geometry'][idx]
            row.update({label+'_xy_correction_m':np.hypot(float(g['dx']),float(g['dy'])),label+'_yaw_correction_deg':np.rad2deg(float(g['dyaw'])),
                        label+'_reliable_fraction':float(geom['reliable_points'])/float(geom['input_points']),
                        label+'_inlier_fraction':float(geom['euclidean_inliers'])/float(geom['input_points']),
                        label+'_xyyaw_min_information':s['eigenvalues'][idx,0]})
        per_frame.append(row)
    with (out/'paired_frames.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(per_frame[0]));w.writeheader();w.writerows(per_frame)
    old_score,new_score=(1.15275,1.15346) if track==1 else (.30576,.31064)
    report={'track':track,'score':{'source':'User submission screenshot, 2026-10-18; no live leaderboard lookup',
                'front':old_score,'rig':new_score,'difference':new_score-old_score,'relative_increase_pct':100*(new_score/old_score-1),
                'rig_candidate_csv_sha256':sha(full/f'result_{track}.csv'),'submission_file_mapping':'Local candidate named by user IDE and context; screenshot itself has no SHA256'},
            'scope':'Full Track pair diagnostics, not frame-level ground-truth errors. Scores cannot separate translation and rotation contributions.',
            'same_original_prior_bytes':True,'same_frame_times':True,'same_applied_xy_yaw_prior':True,
            'height_prior':'Carried previous map height differs between runs; original prior CSV remains equal.',
            'methods':table,'paired':paired,'source_sha256':{Path(__file__).name:sha(Path(__file__))}}
    (out/'full_comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'score':report['score'],'paired':paired,'methods':table},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--track',type=int,choices=(1,2),default=1)
    analyze(p.parse_args())
