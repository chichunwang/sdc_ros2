#!/usr/bin/env python3
"""真實短段：dense head batch／allocator cache／PCA chunk 的速度與輸出對照。"""
from output_layout import loads_saved_json, saved_path
import argparse
import csv
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def compare_arrays(reference,actual):
    import numpy as np
    if reference.shape!=actual.shape:return {'shape_equal':False,'identical':False}
    delta=np.abs(actual.astype(np.float64)-reference.astype(np.float64))
    return {'shape_equal':True,'identical':bool(np.array_equal(reference,actual)),
            'changed_fraction':float(np.mean(reference!=actual)),
            'mean_absolute_difference':float(np.mean(delta)),
            'p99_absolute_difference':float(np.quantile(delta,.99)),
            'max_absolute_difference':float(np.max(delta))}


def probe(args):
    import numpy as np
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs
    from calibrated_rig_geometry import PriorTrajectory,load_rig,backproject,transform,OPTICAL_TO_FLU,write_ply
    from rig_geometry_accel import CachedRectifier,GeometryAccelerator
    from three_camera_source import camera_headers,resolve_camera_timestamps,sha256
    root=Path(__file__).resolve().parents[1];out=args.output.resolve()
    out.relative_to((root/'outputs/run').resolve())
    if (out/'identity.json').exists():raise ValueError('Existing probe evidence cannot be overwritten; choose a fresh output directory')
    out.mkdir(exist_ok=True)
    # A complete model cannot be loaded alongside another reconstruction.
    active=subprocess.run(['pgrep','-f','[p]ython.*scripts/run_rig_(full|fast)\\.py'],capture_output=True,text=True)
    if active.returncode==0:raise RuntimeError('Another full reconstruction is active; no competing model loaded')
    checkpoint=root/'outputs/ml_runtime/checkpoint'
    data=root.parent/'midterm01/data';candidate=args.full_run/'candidate.json'
    scale=loads_saved_json(candidate.read_text())['depth_scale']
    paths=[Path(__file__),root/'scripts/rig_geometry_accel.py',root/'scripts/calibrated_rig_geometry.py',root/'scripts/run_rig_fast.py']
    snapshots=out/'sources';snapshots.mkdir(exist_ok=True)
    for p in paths:shutil.copy2(p,snapshots/p.name)
    metadata={'checkpoint_sha256':sha256(checkpoint/'model.safetensors'),'candidate_sha256':sha256(candidate),
        'source_sha256':{p.name:sha256(p) for p in paths},'depth_scale':scale,'resolution_wh':[448,294],
        'runtime_versions':{p:version(p) for p in ('torch','numpy','scipy','opencv-python-headless','mapanything')},
        'upstream_commit':subprocess.check_output(['git','-C',str(root/'outputs/ml_runtime/map-anything'),'rev-parse','HEAD'],text=True).strip(),
        'scope':'4 real groups, interleaved baseline/candidate twice; identical model/input/precision/geometry. No full sequence or accuracy claim.',
        'allocator_environment':{name:os.environ.get(name) for name in ('PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF')},
        'allocator_backend':torch.cuda.get_allocator_backend(),
        'prior_sha256':{},'calibration_sha256':{},'inputs':[]}
    priors={1:root/'outputs/run/track01_full_2610172233',
            2:root/'outputs/run/track02_full_2610172238'}
    samples=[]
    for track,indices in [(1,[1505,1506]),(2,[1661,1662])]:
        with (data/f'kaggle_track0{track}/Track{track}_Mapanything_ply/index.csv').open() as f:rows=list(csv.DictReader(f))
        camera_root=data/f'Track{track}/Track{track}_images';cal=camera_root/'camera_info'
        trajectory=PriorTrajectory(priors[track]/'sensor_prior.csv',max_extrapolation_seconds=.03)
        rig=load_rig(cal);raw=camera_headers(data/f'Track{track}/Track{track}.db3')
        headers={tag:resolve_camera_timestamps(raw[tag],[int(r[tag+'_ts']) for r in rows]) for tag in ('f','fl','fr')}
        metadata['prior_sha256'][str(track)]=sha256(priors[track]/'sensor_prior.csv')
        metadata['calibration_sha256'][str(track)]={p.name:sha256(p) for p in cal.iterdir() if p.is_file()}
        for idx in indices:
            row=rows[idx];stamp=int(row['f_ts']);front=headers['f'][idx][1]
            sample={'track':track,'idx':idx,'timestamp_ns':stamp,'key':f't{track}_{idx}',
                    'bag_from_header':np.linalg.inv(trajectory.pose(stamp))@trajectory.pose(front),'cameras':[]}
            for tag in ('f','fl','fr'):
                path=camera_root/f'Track{track}_{tag}_images/{idx:06d}_ts_{row[tag+"_ts"]}.jpg'
                pose=trajectory.camera_pose(front,headers[tag][idx][1],rig[tag])
                sample['cameras'].append({'tag':tag,'image':path,'calibration':cal/f'gige_100_{tag}_hdr_camera_info.yaml',
                    'exclusion':cal/'gige_100_f_hdr_mask.png' if tag=='f' else None,'pose':pose,
                    'header_ns':headers[tag][idx][1],'bag_ns':headers[tag][idx][0]})
                metadata['inputs'].append({'path':str(path),'sha256':sha256(path),'header_ns':headers[tag][idx][1]})
            samples.append(sample)
    dump(out/'identity.json',metadata)
    model=MapAnything.from_pretrained(str(checkpoint)).eval().to('cuda')
    rectifier=CachedRectifier();engine=GeometryAccelerator('cuda',256)
    old_empty=torch.cuda.empty_cache
    references={};geometry_inputs={};results=[];exports={};suppressed=[0]
    def noop():suppressed[0]+=1

    def measure(sample,variant,folder=None):
        started=time.perf_counter();views=[];valid_masks=[];images=[]
        for c in sample['cameras']:
            rgb,k,mask=rectifier(c['image'],c['calibration'],exclusion_mask=c['exclusion'])
            # Keep the original per-frame image provenance read in the timing scope.
            sha256(c['image'])
            views.append({'img':rgb,'intrinsics':k,'camera_poses':c['pose'].astype(np.float32),'is_metric_scale':True})
            valid_masks.append(mask);images.append(rgb)
        inputs=preprocess_inputs(views,resize_mode='fixed_size',size=(448,294))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before=time.perf_counter()
        predictions=None
        try:
            with torch.inference_mode(),torch.autocast('cuda',enabled=False,cache_enabled=False):
                predictions=model.infer(inputs,memory_efficient_inference=True,minibatch_size=variant['minibatch'],
                    use_amp=True,amp_dtype='bf16',apply_mask=True,mask_edges=True,apply_confidence_mask=True,confidence_percentile=10)
            torch.cuda.synchronize();infer_seconds=time.perf_counter()-before
            points=[];colors=[];ids=[];confidences=[];arrays=[]
            for cam,(prediction,view,camera) in enumerate(zip(predictions,views,sample['cameras'])):
                def array(key):return prediction[key].detach().float().cpu().numpy()[0]
                depth,conf,mask=array('depth_z')[...,0],array('conf'),array('mask')[...,0].astype(bool)
                arrays.append({'depth':depth,'confidence':conf,'mask':mask})
                cam_points=backproject(depth*scale,view['intrinsics'])
                source=transform(transform(cam_points,camera['pose'])@OPTICAL_TO_FLU.T,sample['bag_from_header'])
                raw_range=np.linalg.norm(cam_points,axis=-1)/scale
                valid=valid_masks[cam]&mask&np.isfinite(depth)&(depth>0)&(raw_range>=2)&(raw_range<=60)
                valid &= np.all(np.isfinite(source),axis=-1)&np.isfinite(conf)
                points.append(source[valid]);colors.append(images[cam][valid]);ids.append(np.full(valid.sum(),cam,dtype=np.uint8));confidences.append(conf[valid])
            points,colors,ids,confidence=map(np.concatenate,(points,colors,ids,confidences))
            keep,sampling=engine.balanced_indices(points,ids,confidence,6000)
            row={'frame':sample['key'],'inference_seconds':infer_seconds,'sampling_seconds':engine.stats['total_seconds'],
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                 'selected':len(keep),'sampling':sampling,'pca_details':dict(engine.stats)}
            if folder:
                folder.mkdir(parents=True)
                name=f'{sample["idx"]:06d}_ts_{sample["timestamp_ns"]}.ply'
                write_ply(folder/name,points[keep],colors[keep]);np.savez_compressed(folder/'labels.npz',camera_ids=ids[keep],confidence=confidence[keep])
                row['ply_sha256']=sha256(folder/name);row['labels_sha256']=sha256(folder/'labels.npz')
                row['cloud_path']=str(folder/name)
            row['frame_seconds']=time.perf_counter()-started
            key=sample['key']
            if folder and key not in references:
                references[key]={'arrays':arrays,'keep':keep.copy(),'ply_sha256':row['ply_sha256'],
                                  'labels_sha256':row['labels_sha256'],'sampling':sampling}
                geometry_inputs[key]=(points,ids,confidence)
            if folder:
                ref=references[key]
                row['comparison']={'prediction':[ {k:compare_arrays(ref['arrays'][cam][k],arrays[cam][k]) for k in arrays[cam]} for cam in range(3)],
                    'selected_indices_identical':bool(np.array_equal(ref['keep'],keep)),
                    'sampling_report_identical':ref['sampling']==sampling,
                    'ply_identical':ref['ply_sha256']==row['ply_sha256'],'labels_identical':ref['labels_sha256']==row['labels_sha256']}
            return row
        finally:
            del predictions,inputs
            torch.cuda.empty_cache()

    variants={'baseline':{'minibatch':1,'retain_cache':False},'minibatch2':{'minibatch':2,'retain_cache':False},
              'minibatch3':{'minibatch':3,'retain_cache':False},'retain_cache':{'minibatch':1,'retain_cache':True},
              'minibatch3_retain_cache':{'minibatch':3,'retain_cache':True}}
    try:
        # Interleave repeated baselines to reveal drift; two passes of every candidate.
        schedule=['baseline','minibatch2','baseline','minibatch3','baseline','retain_cache','baseline','minibatch3_retain_cache','baseline']
        for repeat in range(2):
            for phase,name in enumerate(schedule):
                variant=variants[name];gc.collect();old_empty()
                torch.cuda.empty_cache=noop if variant['retain_cache'] else old_empty
                try:
                    measure(samples[0],variant)  # excluded per-phase warmup
                    for sample in samples:
                        folder=out/f'model_{repeat}_{phase}_{name}'/sample['key']
                        row=measure(sample,variant,folder)
                        row.update({'variant':name,'repeat':repeat,'phase':phase,'status':'ok'})
                        results.append(row)
                        if name not in exports:exports[name]={}
                        if sample['key'] not in exports[name]:exports[name][sample['key']]=row
                    print(json.dumps({'repeat':repeat,'phase':phase,'variant':name,'seconds':[r['frame_seconds'] for r in results[-4:]]}),flush=True)
                except torch.OutOfMemoryError as error:
                    results.append({'variant':name,'repeat':repeat,'phase':phase,'status':'oom','error':str(error)})
                    print(json.dumps({'variant':name,'status':'oom'}),flush=True)
                finally:
                    torch.cuda.empty_cache=old_empty;gc.collect();old_empty()
                    dump(out/'model_results.json',results)
        chunk_results=[]
        # Real geometry from baseline predictions, same resident model, vary only PCA chunk.
        for chunk in [256,512,1024]:
            engine=GeometryAccelerator('cuda',chunk)
            for key,(points,ids,confidence) in geometry_inputs.items():
                times=[]
                for iteration in range(4):
                    t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                    keep,report=engine.balanced_indices(points,ids,confidence)
                    seconds=time.perf_counter()-t
                    if not np.array_equal(keep,references[key]['keep']) or report!=references[key]['sampling']:
                        raise ValueError(f'PCA chunk {chunk} changed selected indices/report for {key}')
                    times.append(seconds)
                chunk_results.append({'chunk':chunk,'frame':key,'all_seconds':times,'warm_median_seconds':statistics.median(times[1:]),
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'indices_and_report_identical':True})
            dump(out/'chunk_results.json',chunk_results)
        # Reports allow the unchanged ROS localization to compare actual selected clouds.
        for name,frames in exports.items():
            for track in [1,2]:
                selected=[s for s in samples if s['track']==track]
                if any(s['key'] not in frames for s in selected):continue
                report={'schema':'calibrated_rig_v1','track':track,'variant':name,'coordinate_frame':'front_camera_FLU',
                    'reference_time':'front_camera_bag_timestamp','ply_scale':1.,'experimental':True,
                    'fixed_rig_heights':True,'per_camera_motion_compensation':True,'depth_scale':scale,
                    'candidate_validation_passed':False,'frames':[]}
                for sample in selected:
                    row=frames[sample['key']]
                    report['frames'].append({'original_idx':sample['idx'],'timestamp_ns':sample['timestamp_ns'],
                        'filename':str(Path(row['cloud_path']).relative_to(out)),'sha256':row['ply_sha256'],'points':row['selected']})
                dump(out/f'{name}_track{track}_report.json',report)
        dump(out/'completion.json',{'model_measurements':len(results),'chunk_measurements':len(chunk_results),
                                   'suppressed_empty_cache_calls':suppressed[0],'full_sequence':False})
    finally:
        torch.cuda.empty_cache=old_empty
        del model
        gc.collect();old_empty()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=saved_path,required=True);p.add_argument('--full-run',type=saved_path,required=True)
    probe(p.parse_args())
