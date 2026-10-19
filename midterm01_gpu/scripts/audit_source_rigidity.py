#!/usr/bin/env python3
"""檢查前相機點雲跨幀 RGB 對應的剛體／尺度一致性；不重算完整定位。"""
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
from scipy.spatial.transform import Rotation

from accuracy.benchmark import create_run_directory
from output_layout import scored_reference_run
from source_rigidity import (RigidityConfig, read_rgb_ply, render_features, match_features,
                             diagnose_matches, statistics)
from three_camera_source import sha256


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run(args):
    root=Path(__file__).resolve().parents[1]
    output=args.output.resolve() if args.output else create_run_directory(root/'outputs/run','source_rigidity_audit')
    if output.parent != (root/'outputs/run').resolve() or not output.name.startswith('track00_test_'):
        raise ValueError('Use a direct outputs/run/track00_test_YYMMDDHHmm directory')
    output.mkdir(exist_ok=True)
    out=output/'rigidity';out.mkdir()
    start=time.perf_counter();config=RigidityConfig();cv2.setNumThreads(1)
    sources=out/'sources';sources.mkdir()
    source_hashes={}
    for name in ('source_rigidity.py','audit_source_rigidity.py','three_camera_source.py','output_layout.py'):
        path=Path(__file__).with_name(name);shutil.copy2(path,sources/name);source_hashes[name]=sha256(path)
    (out/'git_commit.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD'],cwd=root))
    (out/'worktree.patch').write_bytes(subprocess.check_output(['git','diff'],cwd=root))
    dump(out/'params.json',dict(config=asdict(config),command=sys.argv,python=sys.version,numpy=np.__version__,
        scipy=scipy.__version__,opencv=cv2.__version__,source_sha256=source_hashes,
        model_parameters_tuned_on_holdout=False,full_localization=False))
    # The eight original motion windows, plus four fixed route fractions per
    # Track. Chosen before fitting any source rigidity residuals.
    old={1:[905,2286,1808],2:[96,224,1045,2568,1379]}
    plan=[];references={};all_frames={};all_poses={}
    for track in (1,2):
        baseline=getattr(args,f'baseline_track{track}') or scored_reference_run(root,track)
        baseline=baseline.resolve()
        frames=list(csv.DictReader((baseline/'frames.csv').open()))
        poses=np.loadtxt(baseline/'result.txt').reshape(-1,3,4)
        validation=json.loads((baseline/'validation.json').read_text())
        if (len(frames)!=(3011 if track==1 else 3299) or len(poses)!=len(frames)
                or not validation['output_complete_and_finite'] or validation['track']!=track
                or not np.isfinite(poses).all()):
            raise ValueError('Reference must be the complete finite matching Track')
        references[track]=dict(path=str(baseline),sha256={name:sha256(baseline/name)
            for name in ('frames.csv','result.txt','diagnostic.csv','sensor_prior.csv','test_info.json')})
        all_frames[track]=frames;all_poses[track]=poses
        for start_frame in old[track]:
            plan.append(dict(track=track,indices=[start_frame+20+i for i in (-args.stride,0,args.stride)],selection='original_motion_window'))
        for fraction in (.1,.35,.65,.9):
            center=int(round((len(frames)-1)*fraction))
            plan.append(dict(track=track,indices=[center-args.stride,center,center+args.stride],selection='fixed_route_fraction',fraction=fraction))
    dump(out/'plan.json',dict(groups=plan,references=references,scope='Reference poses used only AFTER source-only fits'))
    reports=[]
    for group_id,group in enumerate(plan):
        track=group['track'];frames=all_frames[track];features=[];provenance=[]
        directory=out/f'group_{group_id:02d}_track{track}';directory.mkdir()
        for idx in group['indices']:
            frame=frames[idx];path=args.data_root/f'Track{track}/raw_ply'/frame['filename']
            if not path.name.endswith(f"_ts_{int(frame['timestamp_ns'])}.ply"):
                raise ValueError('Frame filename/timestamp mismatch')
            xyz,rgb=read_rgb_ply(path);feature=render_features(xyz*config.ply_scale,rgb,config)
            features.append(feature)
            cv2.imwrite(str(directory/f'{idx:06d}_virtual.jpg'),feature['image'][:,:,::-1])
            provenance.append(dict(idx=idx,timestamp_ns=int(frame['timestamp_ns']),path=str(path.resolve()),
                sha256=sha256(path),vertices=len(xyz),associated_features=len(feature['pixels']),
                forward_depth_scaled_m=statistics(xyz[:,0]*config.ply_scale)))
        pairs=[]
        for i,j in ((0,1),(1,2),(0,2)):
            match=match_features(features[i],features[j],config)
            a,b=features[i]['xyz'][match[:,0]],features[j]['xyz'][match[:,1]]
            pixels=features[i]['pixels'][match[:,0]]
            report=diagnose_matches(a,b,pixels,config)
            idx_a,idx_b=group['indices'][i],group['indices'][j]
            report.update(indices=[idx_a,idx_b],dt_s=(provenance[j]['timestamp_ns']-provenance[i]['timestamp_ns'])/1e9)
            # Comparison is deliberately downstream: no GNSS/map quantities
            # were available to descriptor matching or either fitted model.
            pa,pb=all_poses[track][[idx_a,idx_b]]
            reference_r=pb[:,:3].T@pa[:,:3]
            reference_t=pb[:,:3].T@(pa[:,3]-pb[:,3])
            report['saved_pose_relative_translation_norm_m']=float(np.linalg.norm(reference_t))
            report['saved_pose_match_error_m']=statistics(np.linalg.norm(a@reference_r.T+reference_t-b,axis=1))
            for model in report['models'].values():
                r=np.array(model['rotation_b_from_a']);t=np.array(model['translation_b_from_a'])
                model['rotation_vs_saved_pose_deg']=float(np.degrees(Rotation.from_matrix(r@reference_r.T).magnitude()))
                model['translation_vs_saved_pose_m']=float(np.linalg.norm(t-reference_t))
            np.savez_compressed(directory/f'{idx_a}_{idx_b}_pairs.npz',a=a,b=b,pixels_a=pixels,
                pixels_b=features[j]['pixels'][match[:,1]],vertex_a=features[i]['vertex_indices'][match[:,0]],
                vertex_b=features[j]['vertex_indices'][match[:,1]])
            pairs.append(report)
        record={**group,'group_id':group_id,'sources':provenance,'pairs':pairs}
        dump(directory/'report.json',record);reports.append(record)
        print(json.dumps(dict(group=group_id,track=track,indices=group['indices'],
            matches=[p['matches'] for p in pairs],rigid_holdout_median_m=[p['models'].get('rigid',{}).get('holdout_error_m',{}).get('median') for p in pairs])),flush=True)
    summary={}
    for track in (1,2):
        pairs=[p for r in reports if r['track']==track for p in r['pairs']]
        fitted=[p for p in pairs if 'rigid' in p['models'] and 'similarity' in p['models']]
        summary[track]=dict(pairs=len(pairs),fitted_pairs=len(fitted),
            rigid_holdout_median_m=statistics([p['models']['rigid']['holdout_error_m']['median'] for p in fitted]),
            similarity_holdout_median_m=statistics([p['models']['similarity']['holdout_error_m']['median'] for p in fitted]),
            relative_scale=statistics([p['models']['similarity']['scale_b_from_a'] for p in fitted]),
            similarity_better_pairs=sum(p['models']['similarity']['holdout_error_m']['median']<p['models']['rigid']['holdout_error_m']['median'] for p in fitted))
    unchanged=all(sha256(Path(v['path'])/name)==expected for v in references.values() for name,expected in v['sha256'].items())
    dump(out/'report.json',dict(groups=reports,summary=summary,baseline_bytes_unchanged=unchanged,
        elapsed_seconds=time.perf_counter()-start,accepted_for_fusion=False,
        limits=['No true static labels; RGB associations may include wrong or moving matches.',
                'A relative similarity scale is not absolute metric calibration.',
                'Source-only consistency is not ATE or an independent absolute map factor.',
                'Virtual camera is a texture lookup; no claim about physical camera/IMU extrinsics.']))
    info=json.loads((output/'test_info.json').read_text()) if (output/'test_info.json').exists() else {}
    info.update(algorithm='source_rigidity_v1',status='completed',reported_score=None,
        source_rigidity_report='rigidity/report.json',full_localization_executed=False)
    dump(output/'test_info.json',info)
    print(json.dumps(summary,indent=2));print('Run:',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    parser.add_argument('--data-root',type=Path,default=root.parent/'midterm01/data')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--stride',type=int,choices=(2,14),default=2,
                        help='Short overlap probe (2) or original ~0.5s baseline (14); same fitting checks')
    for track in (1,2):parser.add_argument(f'--baseline-track{track}',type=Path)
    run(parser.parse_args())
