#!/usr/bin/env python3
"""Reproducible heading-only priors and bounded localization checks; never full Track."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from output_layout import loads_saved_json, scored_reference_run
from prepare_sensor_prior import make_priors
from sensor_fusion import wrap
from three_camera_source import sha256

MODES=('motion_rts','turn_heading_rts','pose_yaw')


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def rows(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def prepare(args,root):
    for track in (1,2):
        baseline=scored_reference_run(root,track)
        raw_path=args.raw_directory/f'input_track{track}.npz'
        raw_meta=loads_saved_json(raw_path.with_suffix('.json').read_text())
        if raw_meta['raw_values_sha256']!=sha256(raw_path):raise ValueError('Raw export changed')
        if raw_meta['bag_sha256']!=sha256(root.parent/f'midterm01/data/Track{track}/Track{track}.db3'):
            raise ValueError('Raw export does not match current bag')
        with np.load(raw_path) as source:d={k:source[k] for k in source.files}
        old=rows(baseline/'sensor_prior.csv');stamps=np.array([int(r['timestamp_ns']) for r in old],np.int64)
        base_xyz=np.array([[float(r[k]) for k in ('x','y','z')] for r in old])
        base_quat=np.array([[float(r[k]) for k in ('qx','qy','qz','qw')] for r in old])
        base_rpy=Rotation.from_quat(base_quat).as_euler('xyz')
        for mode in args.modes:
            output=args.experiment/f'track{track}_{mode}_prior';output.mkdir()
            details={}
            xyz,quat,meta=make_priors(d['gnss_bag_ns'],d['gnss_xyz'],d['imu_bag_ns'],d['imu_quat'],
                stamps,fusion=mode,details=details,gnss_quaternions=d['gnss_quat'])
            rpy=Rotation.from_quat(quat).as_euler('xyz')
            if not np.allclose(xyz,base_xyz,rtol=0,atol=2e-10):raise ValueError('Heading-only model changed XYZ')
            if not np.allclose(rpy[:,:2],base_rpy[:,:2],rtol=0,atol=1e-10):raise ValueError('IMU tilt changed')
            if mode=='motion_rts' and not np.allclose(quat,base_quat,rtol=0,atol=1e-12):
                raise ValueError('Original scored heading no longer reproduced')
            with (output/'sensor_prior.csv').open('x',newline='') as f:
                writer=csv.writer(f);writer.writerow(list(old[0]))
                for record,q in zip(old,quat):
                    writer.writerow([record[k] for k in ('timestamp_ns','x','y','z')]+
                        ([record[k] for k in ('qx','qy','qz','qw')] if mode=='motion_rts' else list(q)))
            with (output/'sensor_prior.fusion.csv').open('x',newline='') as f:
                writer=csv.writer(f);writer.writerow(list(details['diagnostic']));writer.writerows(zip(*details['diagnostic'].values()))
            with (output/'sensor_prior.heading_blocks.csv').open('x',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(details['blocks'][0]));writer.writeheader();writer.writerows(details['blocks'])
            meta['heading_only_ablation']=dict(baseline=str(baseline),prior_sha256=sha256(baseline/'sensor_prior.csv'),
                raw_export=str(raw_path),raw_sha256=sha256(raw_path),timestamp_xyz_strings_frozen=True)
            dump(output/'sensor_prior.json',meta)
            dump(output/'validation.json',dict(track=track,profile='sensor_guided',sensor_fusion=mode,prior_only_not_localization=True))
            dyaw=np.degrees(wrap(rpy[:,2]-base_rpy[:,2]))
            report=dict(track=track,mode=mode,prior=str(output),baseline=str(baseline),
                prior_source_sha256=sha256(output/'sensor_prior.csv'),
                raw_xyz_max_delta_m=float(np.max(abs(xyz-base_xyz))),
                yaw_change_rms_deg=float(np.sqrt(np.mean(dyaw**2))),yaw_change_max_deg=float(np.max(abs(dyaw))),
                metadata=meta['sensor_fusion'])
            dump(args.experiment/f'prior_report_track{track}_{mode}.json',report)
            snapshot=args.experiment/f'prepare_sources_track{track}_{mode}';snapshot.mkdir()
            for name in ('heading_upgrade_lab.py','prepare_sensor_prior.py','sensor_fusion.py','direct_heading.py','turn_heading.py'):
                source=Path(__file__).with_name(name)
                if source.exists():shutil.copy2(source,snapshot/name)
            dump(snapshot/'source_sha256.json',{p.name:sha256(p) for p in snapshot.glob('*.py')})
            print(json.dumps({k:v for k,v in report.items() if k!='metadata'}),flush=True)


def shorts(args,root):
    plan=loads_saved_json((args.window_experiment/'prior_ablation.json').read_text())
    windows=[(r['track'],w) for r in plan for w in r['windows']]
    if len(windows)>12 or any(not 1<=w['frames']<=48 for _,w in windows):raise ValueError('Only bounded original windows')
    for mode in args.modes:
        tag=mode if args.ply_scale==3. else f'{mode}_scale{args.ply_scale:.8f}'
        out=args.experiment/f'short_{tag}';out.mkdir()
        records=[]
        for track,window in windows:
            label=f'track{track}_{window["class"]}_{window["split"]}'
            command=[sys.executable,str(root/'scripts/benchmark.py'),'--track',str(track),
                '--start-frame',str(window['start']),'--frames',str(window['frames']),
                '--profile','sensor_guided','--sensor-fusion',mode,'--geometry-mode','directional','--height-mode','carry',
                '--ply-scale',str(args.ply_scale),
                '--prior-run',str(args.experiment/f'track{track}_{mode}_prior'),
                '--map-cache',str(getattr(args,f'map_cache_track{track}').resolve())]
            dump(out/f'{label}.command.json',command)
            print(mode,label,flush=True)
            log_path=out/f'{label}.log'
            with log_path.open('x') as f:result=subprocess.run(command,cwd=root,stdout=f,stderr=subprocess.STDOUT)
            matches=[s[5:] for s in log_path.read_text().splitlines() if s.startswith('Run: ')]
            if result.returncode or not matches:raise RuntimeError(f'Bounded benchmark failed: {log_path}')
            run=Path(matches[-1]);v=loads_saved_json((run/'validation.json').read_text())
            if not v['output_complete_and_finite'] or v['validation_errors'] or v['submission_frame_set_complete']:
                raise ValueError('Invalid bounded result')
            records.append(dict(track=track,**window,mode=mode,ply_scale=args.ply_scale,run=str(run)))
            dump(args.experiment/f'short_{tag}_runs.json',records)


def summarize(args,root):
    references=loads_saved_json((args.control_experiment/'ablation_runs.json').read_text())
    current_controls=loads_saved_json((args.experiment/'short_motion_rts_runs.json').read_text())
    report=[]
    for record_file in sorted(args.experiment.glob('short_*_runs.json')):
        for record in loads_saved_json(record_file.read_text()):
            track,start,mode=record['track'],record['start'],record['mode'];run=Path(record['run'])
            reference=next(r for r in references if r['track']==track and r['start']==start)
            historical_control=Path(reference['runs']['motion_rts'])
            control=Path(next(r for r in current_controls if r['track']==track and r['start']==start)['run'])
            a,b=[np.loadtxt(p/'result.txt').reshape(-1,3,4) for p in (control,run)]
            old,new=[rows(p/'sensor_prior.csv') for p in (control,run)]
            frozen=all([r[k] for k in ('timestamp_ns','x','y','z')]==[s[k] for k in ('timestamp_ns','x','y','z')] for r,s in zip(old,new))
            params=[]
            for p in (control,run):
                value=loads_saved_json((p/'params.json').read_text())
                for key in ('save_path','sensor_prior_file','ply_path','ply_scale'):value.pop(key)
                params.append(value)
            source_before,source_after=[loads_saved_json((p/'source_input.json').read_text()) for p in (control,run)]
            scale=record.get('ply_scale',3.)
            if source_after['ply_scale']!=scale:raise ValueError('Source scale differs from recorded variant')
            source_before.pop('ply_scale');source_after.pop('ply_scale')
            invariants=dict(frozen_timestamp_xyz=frozen,binary_same=(control/'binary_sha256.txt').read_bytes()==(run/'binary_sha256.txt').read_bytes(),
                raw_source_same=source_before==source_after,non_ablation_solver_params_same=params[0]==params[1])
            if not all(invariants.values()):raise ValueError('Ablation controls differ')
            dyaw=np.degrees(wrap(np.arctan2(b[:,1,0],b[:,0,0])-np.arctan2(a[:,1,0],a[:,0,0])))
            record.update(control=str(control),invariants=invariants,
                xy_change_rms_m=float(np.sqrt(np.mean(np.sum((b[:,:2,3]-a[:,:2,3])**2,axis=1)))),
                yaw_change_rms_deg=float(np.sqrt(np.mean(dyaw**2))),
                csv_changed=(control/'diagnostic.csv').read_bytes()!=(run/'diagnostic.csv').read_bytes())
            if mode=='motion_rts' and scale==3.:
                same={name:(historical_control/name).read_bytes()==(run/name).read_bytes() for name in ('result.txt','diagnostic.csv','geometry.csv','height.csv')}
                if not all(same.values()):raise ValueError('Original motion control changed')
                record['control_byte_identical']=same
                record['historical_control']=str(historical_control)
                record['historical_binary_same']=(historical_control/'binary_sha256.txt').read_bytes()==(run/'binary_sha256.txt').read_bytes()
                record['historical_binary_scope']='Historical doc35 predates GICP build; behavioral regression uses exact artifact bytes, paired variants use this run current binary'
            report.append(record)
    dump(args.experiment/'short_summary.json',dict(runs=report,total_poses=sum(r['frames'] for r in report),
        complete=True,full_localization_executed=False,ground_truth_accuracy_established=False))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);root=Path(__file__).resolve().parents[1]
    p.add_argument('action',choices=('prepare','short','summarize'));p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    p.add_argument('--ply-scale',type=float,default=3.,help='One explicitly recorded source-scale ablation, not a parameter sweep')
    p.add_argument('--raw-directory',type=Path,default=root/'outputs/run/track00_test_2610191808')
    p.add_argument('--window-experiment',type=Path,default=root/'outputs/run/track00_test_2610191627')
    p.add_argument('--control-experiment',type=Path,default=root/'outputs/run/track00_test_2610191818')
    for track,default in ((1,'track00_test_2610171550/cache_track1_leaf1.pcd'),(2,'track00_test_2610181055/cache_track2_leaf1.pcd')):
        p.add_argument(f'--map-cache-track{track}',type=Path,default=root/'outputs/run'/default)
    args=p.parse_args();args.experiment=args.experiment.resolve()
    if not np.isfinite(args.ply_scale) or args.ply_scale<=0:p.error('Source scale must be positive and finite')
    args.experiment.relative_to((root/'outputs/run').resolve())
    if not args.experiment.is_dir():p.error('Create a unique track00_test run before the lab')
    globals()[{'prepare':'prepare','short':'shorts','summarize':'summarize'}[args.action]](args,root)
