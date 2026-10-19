#!/usr/bin/env python3
"""獨立加速入口；保留正在執行的 reference 檔案及其續跑契約。"""
from output_layout import loads_saved_json, saved_path
import argparse
import json
from pathlib import Path
import shutil
import time


def execution_identity(args, chunk_size=256):
    from three_camera_source import sha256
    sources=[Path(__file__),Path(__file__).with_name('rig_geometry_accel.py')]
    identity={'backend':args.geometry_backend,
              'precision':'float64 PCA; integer voxel keys; unchanged float32 weights and bf16 inference',
              'chunk_size':chunk_size if args.geometry_backend!='reference' else None,
              'verify_sampling':args.verify_sampling,
              'sources_sha256':{p.name:sha256(p) for p in sources},
              'mode':'cached_prediction_replay' if args.replay_directory else 'neural_inference'}
    imported=args.output/'imported_reference.json'
    if imported.exists():identity['imported_reference_sha256']=sha256(imported)
    if args.replay_directory:
        identity['prediction_cache']={}
        for i in range(args.start_frame,args.start_frame+args.frames):
            for camera in range(3):
                p=args.replay_directory/f'track{args.track}_{i:06d}/rig_{camera}.npz'
                identity['prediction_cache'][str(p.resolve())]=sha256(p)
    return identity


def run(args):
    import numpy as np
    import torch
    import calibrated_rig_geometry as geometry
    import run_rig_full as reference
    from rig_geometry_accel import CachedRectifier,GeometryAccelerator
    from three_camera_source import sha256
    root=Path(__file__).resolve().parents[1]
    if args.replay_directory and args.frames>24:raise ValueError('Cached prediction replay is limited to 24 diagnostic frames')
    backend=args.geometry_backend
    engine=GeometryAccelerator(backend) if backend!='reference' else None
    cached=CachedRectifier()
    stages={};details={};totals=[]
    old_rectify,old_sample=geometry.rectify,geometry.balanced_indices
    old_backproject,old_transform,old_write=geometry.backproject,geometry.transform,geometry.write_ply
    old_savez=np.savez_compressed
    old_identity,old_json=reference.assert_resume_identity,reference.atomic_json
    extra_sources=[Path(__file__),Path(__file__).with_name('rig_geometry_accel.py')]
    execution=execution_identity(args)
    def timed(name,function):
        def call(*a,**kw):
            before=time.perf_counter();result=function(*a,**kw)
            stages[name]=stages.get(name,0.)+time.perf_counter()-before
            return result
        return call
    def sample(*a,**kw):
        selected,report=(engine.balanced_indices if engine else old_sample)(*a,**kw)
        if engine:details.update(engine.stats)
        if args.verify_sampling and engine:
            expected,expected_report=old_sample(*a,**kw)
            np.testing.assert_array_equal(selected,expected)
            if report!=expected_report:raise ValueError('Sampling report differs from reference')
        return selected,report
    def identity(directory,value,resume):
        value['execution']=execution
        old_identity(directory,value,resume)
        snapshot=Path(directory)/'execution_sources';snapshot.mkdir(exist_ok=True)
        for p in extra_sources:shutil.copy2(p,snapshot/p.name)
    def atomic(path,value):
        if Path(path).name=='frame.json':
            value['stage_seconds']=dict(stages);value['sampling_stage_details']=dict(details)
            totals.append({'original_idx':value['original_idx'],'seconds':value['seconds'],'stages':dict(stages),'sampling':dict(details)})
            stages.clear();details.clear()
        old_json(path,value)
    geometry.rectify=timed('read_decode_rectify',cached if engine else old_rectify)
    geometry.balanced_indices=timed('balanced_sampling',sample)
    geometry.backproject=timed('backproject',old_backproject)
    geometry.transform=timed('point_transforms',old_transform)
    geometry.write_ply=timed('ply_write',old_write)
    np.savez_compressed=timed('labels_or_depth_write',old_savez)
    reference.assert_resume_identity=identity;reference.atomic_json=atomic
    from mapanything import models
    from mapanything.utils import image as image_utils
    old_pretrained=models.MapAnything.from_pretrained
    old_infer=models.MapAnything.infer
    old_preprocess=image_utils.preprocess_inputs
    image_utils.preprocess_inputs=timed('model_input_preprocessing',old_preprocess)
    if args.replay_directory:
        class Replay:
            def __init__(self):
                self.pending=iter(i for i in range(args.start_frame,args.start_frame+args.frames)
                                  if not (args.output/'frames'/f'{i:06d}').exists())
            def eval(self):return self
            def to(self,device):return self
            def infer(self,views,**kwargs):
                idx=next(self.pending)
                predictions=[]
                for camera in range(3):
                    p=args.replay_directory/f'track{args.track}_{idx:06d}/rig_{camera}.npz'
                    with np.load(p) as saved:
                        predictions.append({'depth_z':torch.from_numpy(saved['depth_z'])[None,...,None].to('cuda'),
                                            'conf':torch.from_numpy(saved['confidence'])[None].to('cuda'),
                                            'mask':torch.from_numpy(saved['mask'].astype(np.float32))[None,...,None].to('cuda')})
                return predictions
        models.MapAnything.from_pretrained=lambda *a,**kw:Replay()
    else:
        def infer(model,*a,**kw):
            before=time.perf_counter();result=old_infer(model,*a,**kw);torch.cuda.synchronize()
            stages['model_inference']=time.perf_counter()-before
            return result
        models.MapAnything.infer=infer
    try:
        if args.dry_run:print(json.dumps({'execution':execution},indent=2))
        reference.run(args)
        if not args.dry_run:
            previous=args.output/'performance.json'
            prior_frames=loads_saved_json(previous.read_text())['frames'] if previous.exists() else []
            by_index={r['original_idx']:r for r in prior_frames+totals}
            reference.atomic_json(previous,{'execution':execution,'frames':[by_index[i] for i in sorted(by_index)],
                                  'scope':'Per-frame timers. Cached replay excludes model computation; not full-run speed or accuracy evidence.'})
    finally:
        geometry.rectify=old_rectify;geometry.balanced_indices=old_sample
        geometry.backproject=old_backproject;geometry.transform=old_transform;geometry.write_ply=old_write
        np.savez_compressed=old_savez
        reference.assert_resume_identity=old_identity;reference.atomic_json=old_json
        models.MapAnything.from_pretrained=old_pretrained;models.MapAnything.infer=old_infer
        image_utils.preprocess_inputs=old_preprocess


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--track',type=int,choices=(1,2),required=True)
    p.add_argument('--frames',type=int,required=True)
    p.add_argument('--start-frame',type=int,default=0)
    for name in ('data-root','checkpoint','model-repo','prior-run','candidate','output'):p.add_argument('--'+name,type=saved_path,required=True)
    p.add_argument('--map-cache',type=saved_path)
    p.add_argument('--budget',type=int,default=6000)
    p.add_argument('--resume',action='store_true');p.add_argument('--localize',action='store_true')
    p.add_argument('--save-depth',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--geometry-backend',choices=('reference','cpu','cuda'),default='cuda')
    p.add_argument('--verify-sampling',action='store_true',help='Diagnostic: recompute scalar reference and require exact chosen indices/report; costs extra time')
    p.add_argument('--replay-directory',type=saved_path,help='Diagnostic only: real cached per-view predictions; no new model inference')
    a=p.parse_args()
    if a.budget<100:p.error('Budget must be >=100')
    run(a)
