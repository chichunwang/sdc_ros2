#!/usr/bin/env python3
"""將已停止 reference run 的已提交幀複製到新加速 run；不啟動完整重建。"""
import argparse
import contextlib
import csv
import fcntl
import io
import json
from pathlib import Path
import shutil
import subprocess

import run_rig_full as reference
from run_rig_fast import execution_identity
from three_camera_source import sha256


def migrate(source):
    root=Path(__file__).resolve().parents[1]
    source=source.resolve();source.relative_to((root/'outputs/runs').resolve())
    active=subprocess.run(['pgrep','-f','[p]ython.*scripts/run_rig_(full|fast)\\.py'],capture_output=True)
    if active.returncode==0:raise RuntimeError('Stop the existing reconstruction before importing frames')
    candidate=source/'candidate.json'
    if not candidate.is_file():raise ValueError('Source root candidate.json missing')
    prepared=[]
    with contextlib.ExitStack() as stack:
        for track in (1,2):
            old=source/f'track{track}'
            if not old.exists():continue
            lock=stack.enter_context((old/'.lock').open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            identity=json.loads((old/'identity.json').read_text())
            if 'execution' in identity:raise ValueError('Import accepts the original reference producer only')
            args=argparse.Namespace(track=track,frames=identity['frames'],start_frame=identity['start_frame'],
                data_root=Path(identity['data_root']),prior_run=Path(identity['prior_run']),candidate=candidate,
                checkpoint=root/'outputs/ml_runtime/checkpoint',model_repo=root/'outputs/ml_runtime/map-anything',
                output=root/'outputs/runs/import_preview',map_cache=Path(identity['map_cache']) if identity['map_cache'] else None,
                budget=identity['budget'],save_depth=identity['save_depth'],localize=identity['localize'],
                resume=False,dry_run=True,geometry_backend='cuda',verify_sampling=False,replay_directory=None)
            captured=io.StringIO()
            with contextlib.redirect_stdout(captured):reference.run(args)
            if json.loads(captured.getvalue())!=identity:
                raise ValueError('Original source/model/scale/prior/calibration/runtime has changed; refuse import')
            index=args.data_root/f'kaggle_track0{track}/Track{track}_Mapanything_ply/index.csv'
            with index.open() as stream:rows=list(csv.DictReader(stream))
            records=[]
            for row in rows[args.start_frame:args.start_frame+args.frames]:
                idx,stamp=int(row['index']),int(row['f_ts'])
                folder=old/'frames'/f'{idx:06d}'
                record=reference.completed_frame(folder,idx,stamp)
                if record:
                    reference.verify_committed_inputs(record,row,args.data_root/f'Track{track}/Track{track}_images',track)
                    records.append((folder,record,sha256(folder/'frame.json')))
            prepared.append((old,identity,args,records))
        if not prepared:raise ValueError('No original Track identity found')
        from accuracy.benchmark import create_run_directory
        target=create_run_directory(root/'outputs/runs','three_camera_full_fast_imported')
        shutil.copy2(candidate,target/'candidate.json')
        for old,identity,args,records in prepared:
            new=target/old.name;new.mkdir();(new/'frames').mkdir()
            args.output=new
            origin={'version':1,'source_run':str(old),'identity_sha256':sha256(old/'identity.json'),
                    'original_identity':identity,'completed_frames':len(records),
                    'frame_json_sha256':{str(record['original_idx']):digest for _,record,digest in records},
                    'scope':'Verified original inputs and committed outputs; copied unchanged, no model recomputation. Remaining frames use the new execution backend.'}
            reference.atomic_json(new/'imported_reference.json',origin)
            for folder,record,digest in records:shutil.copytree(folder,new/'frames'/folder.name)
            reference.atomic_json(new/'identity.json',{**identity,'execution':execution_identity(args)})
            shutil.copy2(candidate,new/'candidate.json')
            snapshot=new/'sources';snapshot.mkdir()
            for name in identity['source_sha256']:shutil.copy2(root/'scripts'/name,snapshot/name)
            snapshot=new/'execution_sources';snapshot.mkdir()
            for name in ('run_rig_fast.py','rig_geometry_accel.py','migrate_rig_fast.py'):
                shutil.copy2(root/'scripts'/name,snapshot/name)
            reference.atomic_json(new/'progress.json',{'completed':len(records),'expected':args.frames,
                'last_original_idx':records[-1][1]['original_idx'] if records else None,'imported':True})
        reference.atomic_json(target/'migration.json',{'source_run':str(source),
            'migration_script_sha256':sha256(Path(__file__)),
            'tracks':{old.name:len(records) for old,identity,args,records in prepared},
            'scope':'Import only; no full reconstruction or localization started.'})
    print(f'Run: {target}',flush=True)
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    migrate(parser.parse_args().source)
