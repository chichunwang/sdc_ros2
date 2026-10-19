#!/usr/bin/env python3
"""核對完整 CSV／矩陣，彙整同深度消融及共用 probe；不估真值誤差。"""
from output_layout import loads_saved_json, saved_path
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial import cKDTree
from analyze_rig_track1 import rows,stats,information
from three_camera_source import sha256
from three_camera_calibration import load_binary_map,load_ply
from probe_rig_track1_geometry import read_colored


def validate(root,out):
    full=root/'outputs/run/track00_full_2610181301';results=[]
    for track,count in [(1,3011),(2,3299)]:
        base=full/f'track{track}';loc=loads_saved_json((base/'localization.json').read_text());run=Path(loc['run'])
        validation=loads_saved_json((run/'validation.json').read_text());complete=loads_saved_json((base/'completion.json').read_text());report=loads_saved_json((base/'report.json').read_text())
        path=base/f'result_{track}.csv'
        with path.open() as f:
            reader=csv.DictReader(f)
            if reader.fieldnames!=['idx','x','y','yaw']:raise ValueError('CSV header mismatch')
            records=list(reader)
        if len(records)!=count or [int(r['idx']) for r in records]!=list(range(count)):raise ValueError('CSV index/count mismatch')
        values=np.array([[float(r[k]) for k in ['x','y','yaw']] for r in records]);pose=np.loadtxt(run/'result.txt').reshape(-1,3,4)
        if len(pose)!=count or not np.isfinite(values).all() or not np.isfinite(pose).all():raise ValueError('Invalid finite full pose')
        r=pose[:,:,:3];orth=np.max(np.abs(np.swapaxes(r,1,2)@r-np.eye(3)));det=np.linalg.det(r)
        if orth>=1e-5 or np.max(np.abs(det-1))>=1e-5:raise ValueError('Invalid rotations')
        expected=np.column_stack((pose[:,0,3],pose[:,1,3],np.arctan2(pose[:,1,0],pose[:,0,0])))
        yaw_error=np.arctan2(np.sin(values[:,2]-expected[:,2]),np.cos(values[:,2]-expected[:,2]))
        if not np.allclose(values[:,:2],expected[:,:2],atol=1e-5,rtol=0) or np.max(np.abs(yaw_error))>=1e-5:raise ValueError('CSV/TXT mismatch')
        if sha256(path)!=loc['csv_sha256'] or sha256(path)!=sha256(full/f'result_{track}.csv'):raise ValueError('Copied CSV identity mismatch')
        if sha256(base/'report.json')!=complete['report_sha256'] or sha256(base/'report.json')!=loc['source_report_sha256']:raise ValueError('Source report identity mismatch')
        if not (complete['full_frame_set'] and loc['full_frame_set'] and validation['output_complete_and_finite']) or validation['validation_errors']:raise ValueError('Incomplete localization')
        if len(report['frames'])!=count or [f['original_idx'] for f in report['frames']]!=list(range(count)):raise ValueError('Incomplete source manifest')
        results.append({'track':track,'rows':count,'csv_sha256':sha256(path),'root_and_track_csv_identical':True,
            'matrix_and_csv_consistent':True,'rotation_orthogonality_max_error':float(orth),'full_manifest_complete':True,
            'validation':validation['guidance_not_accuracy'],'geometry':validation['geometry_diagnostics_not_accuracy'],
            'scope':'Local integrity and format, not ground-truth accuracy. Scores are user screenshot reports.'})
    (out/'full_outputs_validation.json').write_text(json.dumps(results,indent=2)+'\n')
    return results


def ablations(root,out):
    runs=loads_saved_json((out/'ablation_runs.json').read_text())
    if len(runs)!=24:raise ValueError('Incomplete ablation experiment')
    full=root/'outputs/run/track00_full_2610181301/track1';manifest=loads_saved_json((full/'report.json').read_text())
    oldpaths={int(p.stem.rsplit('_ts_',1)[1]):p for p in (root.parent/'midterm01/data/Track1/raw_ply').glob('*.ply')}
    scene=cKDTree(load_binary_map(root/'outputs/run/track00_test_2610171550/cache_track1_leaf1.pcd'))
    probes={};results=[];binaries=set();prior_windows={}
    for r in runs:
        p=Path(r['run']);v=loads_saved_json((p/'validation.json').read_text());g=rows(p/'guidance.csv');geom=rows(p/'geometry.csv');frames=rows(p/'frames.csv')
        if not v['output_complete_and_finite'] or v['frames']!=12 or len(g)!=12:raise ValueError('Invalid ablation output')
        binary=(p/'binary_sha256.txt').read_text();binaries.add(binary)
        key=r['start'];prior=(p/'sensor_prior.csv').read_bytes()
        if key in prior_windows and prior_windows[key]!=prior:raise ValueError('Unequal ablation priors')
        prior_windows[key]=prior
        pose=np.loadtxt(p/'result.txt').reshape(-1,3,4);errors={k:[] for k in ['old_front','new_front','new_left','new_right']}
        for offset,transform in enumerate(pose):
            idx=key+offset
            if idx not in probes:
                record=manifest['frames'][idx];path=full/record['filename'];points,_=read_colored(path)
                with np.load(path.parent/'labels.npz') as labels:ids=labels['camera_ids']
                probes[idx]={'old_front':3*load_ply(oldpaths[record['timestamp_ns']],max_points=100000,leaf=0),
                             **{name:points[ids==i] for i,name in enumerate(['new_front','new_left','new_right'])}}
            if int(frames[offset]['timestamp_ns'])!=manifest['frames'][idx]['timestamp_ns']:raise ValueError('Ablation timestamp mismatch')
            for name,points in probes[idx].items():
                distances,_=scene.query(points@transform[:,:3].T+transform[:,3]);errors[name].append(float(np.sqrt(np.mean(np.minimum(distances,2)**2))))
        n=np.array([float(a['input_points']) for a in geom]);rel=np.array([float(a['reliable_points']) for a in geom]);inliers=np.array([float(a['euclidean_inliers']) for a in geom])
        results.append({**r,'accepted':int(sum(int(a['map_accepted']) for a in g)),'actual_points':n.tolist(),
            'reliable_fraction':(rel/n).tolist(),'inlier_fraction':(inliers/n).tolist(),'xyyaw_information':information(geom)[:,0].tolist(),
            'final_z':float(pose[-1,2,3]),'xy_correction_m':[float(np.hypot(float(a['dx']),float(a['dy']))) for a in g],
            'abs_yaw_correction_deg':[float(np.rad2deg(abs(float(a['dyaw'])))) for a in g],'probe_errors_m':errors})
    if len(binaries)!=1:raise ValueError('Ablations used unequal binaries')
    aggregate={}
    for name in dict.fromkeys(r['name'] for r in results):
        selected=[r for r in results if r['name']==name]
        aggregate[name]={'frames':12*len(selected),'accepted':sum(r['accepted'] for r in selected),
            **{k:stats([x for r in selected for x in r[k]]) for k in ['actual_points','reliable_fraction','inlier_fraction','xyyaw_information','xy_correction_m','abs_yaw_correction_deg']},
            'common_probe_mean_rmse_m':{k:float(np.mean([x for r in selected for x in r['probe_errors_m'][k]])) for k in errors}}
    document={'scope':'24 runs x 12 frames. Same dense prediction/selected pool for new-camera subsets; same prior per window and binary. Independent windows reset carried Z, not full-trajectory equivalence. Common capped NN probes are not accuracy or comparable losses between sources.',
        'same_binary':True,'same_prior_per_window':True,'aggregate':aggregate,'runs':results}
    (out/'ablation_summary.json').write_text(json.dumps(document,indent=2)+'\n')
    return document


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=saved_path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=args.output.resolve();out.relative_to((root/'outputs/run').resolve())
    if (out/'ablation_summary.json').exists():raise ValueError('Existing diagnosis cannot be overwritten')
    shutil.copy2(Path(__file__),out/'sources'/Path(__file__).name)
    full=validate(root,out);ablation=ablations(root,out)
    print(json.dumps({'full_checks':[{k:r[k] for k in ['track','rows','csv_sha256']} for r in full],'ablation':ablation['aggregate']},indent=2))
