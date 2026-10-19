#!/usr/bin/env python3
"""使用者手動的三相機批次重建／定位入口；逐幀原子保存與嚴格中斷續跑。"""
from output_layout import loads_saved_json, saved_path
import argparse
import csv
import fcntl
import gc
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from three_camera_source import sha256, camera_headers, resolve_camera_timestamps


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def completed_frame(directory, original_idx, stamp):
    """A committed frame is immutable; missing/corrupt data is an error, never silently skipped."""
    directory = Path(directory)
    if not directory.exists():
        return None
    record = loads_saved_json((directory / 'frame.json').read_text())
    if record['original_idx'] != original_idx or record['timestamp_ns'] != stamp:
        raise ValueError('Resume frame index/time mismatch')
    for name, digest in record['files_sha256'].items():
        path = (directory / name).resolve()
        path.relative_to(directory.resolve())
        if sha256(path) != digest:
            raise ValueError('Resume frame data hash mismatch')
    return record


def assert_resume_identity(directory, identity, resume):
    path = Path(directory) / 'identity.json'
    if path.exists():
        if not resume or loads_saved_json(path.read_text()) != identity:
            raise ValueError('Existing run requires --resume and identical inputs/model/scale/code/configuration')
    else:
        if resume:
            raise ValueError('Resume identity missing')
        atomic_json(path, identity)


def verify_committed_inputs(record, row, camera_root, track):
    cameras = {c['camera']:c for c in record['cameras']}
    if set(cameras) != {'f','fl','fr'}:raise ValueError('Resume camera provenance incomplete')
    for tag, camera in cameras.items():
        image = Path(camera_root)/f'Track{track}_{tag}_images/{record["original_idx"]:06d}_ts_{row[tag+"_ts"]}.jpg'
        if sha256(image) != camera['image_sha256']:raise ValueError('Resume original image hash mismatch')


def run(args):
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.relative_to((root / 'outputs').resolve())
    prior = args.prior_run.resolve()
    validation = loads_saved_json((prior / 'validation.json').read_text())
    if validation['track'] != args.track or validation['sensor_fusion'] != 'heading_rts':
        raise ValueError('Prior Track/fusion mismatch')
    candidate = loads_saved_json(args.candidate.read_text())
    scale = float(candidate['depth_scale'])
    if not .25 <= scale <= 4.:
        raise ValueError('Candidate scale must be finite within [0.25,4]')
    index_file = args.data_root / f'kaggle_track0{args.track}/Track{args.track}_Mapanything_ply/index.csv'
    rows = list(csv.DictReader(index_file.open()))
    count = 3011 if args.track == 1 else 3299
    if len(rows) != count or [int(r['index']) for r in rows] != list(range(count)):
        raise ValueError('Complete official camera index required')
    if args.start_frame < 0 or args.frames < 1 or args.start_frame+args.frames > count:
        raise ValueError('Frame range outside Track')
    selected = rows[args.start_frame:args.start_frame+args.frames]
    camera_root = args.data_root / f'Track{args.track}/Track{args.track}_images'
    camera_info = camera_root / 'camera_info'
    source_paths = [Path(__file__), Path(__file__).with_name('calibrated_rig_geometry.py'), Path(__file__).with_name('three_camera_source.py')]
    identity = {'version': 1, 'track': args.track, 'start_frame': args.start_frame, 'frames': args.frames,
                'index_sha256': sha256(index_file), 'prior_sha256': sha256(prior / 'sensor_prior.csv'),
                'candidate_sha256': sha256(args.candidate), 'depth_scale': scale, 'candidate_validation_passed': candidate['validation_passed'],
                'checkpoint_sha256': sha256(args.checkpoint / 'model.safetensors'), 'checkpoint_config_sha256': sha256(args.checkpoint / 'config.json'),
                'upstream_commit': subprocess.check_output(['git','-C',str(args.model_repo),'rev-parse','HEAD'],text=True).strip(),
                'calibration_sha256': {p.name: sha256(p) for p in sorted(camera_info.iterdir()) if p.is_file()},
                'source_sha256': {p.name: sha256(p) for p in source_paths}, 'budget': args.budget,
                'runtime_versions': {name:version(name) for name in ('torch','numpy','scipy','opencv-python-headless','mapanything','uniception')},
                'resolution_wh': [448,294], 'max_boundary_extrapolation_seconds': .03, 'save_depth': args.save_depth,
                'data_root': str(args.data_root.resolve()), 'prior_run': str(prior), 'localize': args.localize,
                'map_cache': str(args.map_cache.resolve()) if args.map_cache else None,
                'scope': 'Experimental calibrated rig; candidate scale consistency is not validated localization accuracy'}
    if args.dry_run:
        print(json.dumps(identity,indent=2));return
    if output.exists() and not args.resume:
        raise ValueError('Output already exists; choose a new directory or --resume')
    output.mkdir(parents=True,exist_ok=True)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_resume_identity(output, identity, args.resume)
        if not args.resume:
            (output / 'candidate.json').write_bytes(args.candidate.read_bytes())
            sources = output / 'sources';sources.mkdir()
            for p in source_paths:shutil.copy2(p,sources/p.name)
        frames_root = output / 'frames';frames_root.mkdir(exist_ok=True)
        # Only our own uncommitted temporary frame directories may be removed on resume.
        for stale in frames_root.glob('.frame*_tmp*'):
            if stale.is_dir():shutil.rmtree(stale)
        records = []
        for row in selected:
            i,stamp = int(row['index']),int(row['f_ts'])
            record = completed_frame(frames_root / f'{i:06d}', i, stamp)
            if record:
                verify_committed_inputs(record,row,camera_root,args.track)
                records.append(record)
        remaining = len(selected)-len(records)
        print(json.dumps({'output':str(output),'already_complete':len(records),'remaining':remaining,
                          'candidate_validation_passed':candidate['validation_passed']}),flush=True)
        if remaining:
            import cv2
            import numpy as np
            import torch
            from mapanything.models import MapAnything
            from mapanything.utils.image import preprocess_inputs
            from calibrated_rig_geometry import (PriorTrajectory,load_rig,rectify,backproject,transform,OPTICAL_TO_FLU,balanced_indices,write_ply)
            trajectory = PriorTrajectory(prior / 'sensor_prior.csv', max_extrapolation_seconds=.03)
            rig = load_rig(camera_info)
            raw_header = camera_headers(args.data_root / f'Track{args.track}/Track{args.track}.db3')
            headers = {tag:resolve_camera_timestamps(raw_header[tag],[int(r[f'{tag}_ts']) for r in rows]) for tag in ('f','fl','fr')}
            # Check all requested times before expensive inference; boundaries use recorded, bounded extrapolation.
            for row in selected:
                for tag in ('f','fl','fr'):trajectory.pose(headers[tag][int(row['index'])][1])
            model = MapAnything.from_pretrained(str(args.checkpoint)).eval().to('cuda')
            for row in selected:
                i,stamp = int(row['index']),int(row['f_ts'])
                final = frames_root / f'{i:06d}'
                if completed_frame(final,i,stamp):continue
                started = time.perf_counter()
                with tempfile.TemporaryDirectory(prefix=f'.frame{i:06d}_tmp',dir=frames_root) as temporary:
                    temporary = Path(temporary)
                    views, masks, images, poses, sources = [],[],[],[],[]
                    front_header = headers['f'][i][1]
                    bag_from_header = np.linalg.inv(trajectory.pose(stamp)) @ trajectory.pose(front_header)
                    for tag in ('f','fl','fr'):
                        image = camera_root / f'Track{args.track}_{tag}_images/{i:06d}_ts_{row[tag+"_ts"]}.jpg'
                        excluded = camera_info/'gige_100_f_hdr_mask.png' if tag=='f' else None
                        rgb,k,valid = rectify(image,camera_info/f'gige_100_{tag}_hdr_camera_info.yaml',(448,294),excluded)
                        header_stamp = headers[tag][i][1]
                        pose = trajectory.camera_pose(front_header,header_stamp,rig[tag])
                        views.append({'img':rgb,'intrinsics':k,'camera_poses':pose.astype(np.float32),'is_metric_scale':True})
                        masks.append(valid);images.append(rgb);poses.append(pose)
                        sources.append({'camera':tag,'image_sha256':sha256(image),'header_ns':header_stamp,'bag_ns':headers[tag][i][0],
                                        'applied_camera_pose':pose.tolist(),'extrapolated_motion_seconds':trajectory.extrapolated_queries.get(header_stamp,0.)})
                    inputs = preprocess_inputs(views,resize_mode='fixed_size',size=(448,294))
                    for view,original in zip(inputs,views):
                        np.testing.assert_allclose(view['intrinsics'][0].numpy(),original['intrinsics'],atol=1e-4)
                        if tuple(view['img'].shape[-2:]) != (294,448):raise ValueError('Unexpected image resize')
                    torch.cuda.reset_peak_memory_stats()
                    with torch.inference_mode(), torch.autocast('cuda',enabled=False,cache_enabled=False):
                        predictions = model.infer(inputs,memory_efficient_inference=True,minibatch_size=1,use_amp=True,amp_dtype='bf16',
                                                  apply_mask=True,mask_edges=True,apply_confidence_mask=True,confidence_percentile=10)
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_allocated()
                    points,colors,ids,confidence = [],[],[],[]
                    for cam_id,(prediction,view) in enumerate(zip(predictions,views)):
                        def array(key):return prediction[key].detach().float().cpu().numpy()[0]
                        depth,conf,mask = array('depth_z')[...,0],array('conf'),array('mask')[...,0].astype(bool)
                        cam = backproject(depth*scale,view['intrinsics'])
                        source = transform(transform(cam,poses[cam_id]) @ OPTICAL_TO_FLU.T,bag_from_header)
                        # Match the verification's original-depth mask; near-body exclusion remains conservative.
                        raw_range = np.linalg.norm(cam,axis=-1)/scale
                        valid = masks[cam_id]&mask&np.isfinite(depth)&(depth>0)&(raw_range>=2)&(raw_range<=60)
                        valid &= np.all(np.isfinite(source),axis=-1)&np.isfinite(conf)
                        points.append(source[valid]);colors.append(images[cam_id][valid]);ids.append(np.full(valid.sum(),cam_id,dtype=np.uint8));confidence.append(conf[valid])
                        if args.save_depth:
                            np.savez_compressed(temporary/f'view{cam_id}.npz',depth_z=depth,confidence=conf,mask=valid,intrinsics=view['intrinsics'])
                    points,colors,ids,confidence = map(np.concatenate,(points,colors,ids,confidence))
                    keep,sampling = balanced_indices(points,ids,confidence,args.budget)
                    if len(keep)<10:raise ValueError(f'Insufficient source geometry at frame {i}; no silent fallback')
                    filename = f'{i:06d}_ts_{stamp}.ply'
                    write_ply(temporary/filename,points[keep],colors[keep])
                    np.savez_compressed(temporary/'labels.npz',camera_ids=ids[keep],confidence=confidence[keep])
                    record = {'original_idx':i,'timestamp_ns':stamp,'filename':f'frames/{i:06d}/{filename}',
                              'sha256':sha256(temporary/filename),'points':len(keep),
                              'files_sha256':{p.name:sha256(p) for p in temporary.iterdir()},
                              'camera_counts':{str(c):int(np.sum(ids[keep]==c)) for c in range(3)},'sampling':sampling,
                              'cameras':sources,'seconds':time.perf_counter()-started,'peak_cuda_bytes':peak}
                    atomic_json(temporary/'frame.json',record)
                    temporary.rename(final)
                records.append(record)
                atomic_json(output/'progress.json',{'completed':len(records),'expected':len(selected),'last_original_idx':i})
                print(json.dumps({'frame':i,'completed':len(records),'expected':len(selected),'seconds':record['seconds']}),flush=True)
                del predictions,inputs
                torch.cuda.empty_cache()
            del model
            gc.collect();torch.cuda.empty_cache()
        records.sort(key=lambda r:r['original_idx'])
        if [r['original_idx'] for r in records] != [int(r['index']) for r in selected]:raise ValueError('Incomplete reconstructed frame set')
        report = {'schema':'calibrated_rig_v1','track':args.track,'variant':'bias_candidate_balanced','coordinate_frame':'front_camera_FLU',
                  'reference_time':'front_camera_bag_timestamp','ply_scale':1.,'experimental':True,'fixed_rig_heights':True,
                  'per_camera_motion_compensation':True,'depth_scale':scale,'candidate_validation_passed':candidate['validation_passed'],
                  'identity_sha256':sha256(output/'identity.json'),'frames':records}
        atomic_json(output/'report.json',report)
        atomic_json(output/'completion.json',{'frames':len(records),'first_original_idx':args.start_frame,
                                            'full_frame_set':args.start_frame==0 and args.frames==count,'report_sha256':sha256(output/'report.json')})
        if args.localize:
            previous = output/'localization.json'
            if previous.exists():
                done = loads_saved_json(previous.read_text())
                if done['source_report_sha256'] != sha256(output/'report.json') or done['csv_sha256'] != sha256(done['csv']):
                    raise ValueError('Existing localization result identity mismatch')
                print(json.dumps(done,indent=2));return
            command = [sys.executable,str(root/'scripts/benchmark.py'),'--track',str(args.track),'--frames',str(args.frames),
                       '--start-frame',str(args.start_frame),'--profile','sensor_guided','--sensor-fusion','heading_rts','--geometry-mode','directional',
                       '--cloud-source','reconstruction','--ply-scale','1','--source-report',str(output/'report.json'),'--prior-run',str(prior)]
            if args.map_cache:command+=['--map-cache',str(args.map_cache.resolve())]
            with (output/'localization.log').open('a') as log:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
            if code:raise RuntimeError('Localization failed; inspect localization.log. Reconstruction can be resumed.')
            runs=[line[5:] for line in (output/'localization.log').read_text().splitlines() if line.startswith('Run: ')]
            loc=Path(runs[-1]);check=loads_saved_json((loc/'validation.json').read_text())
            if not check['output_complete_and_finite']:raise ValueError('Invalid localization output')
            csv_path = output/(f'result_{args.track}.csv' if check['submission_frame_set_complete'] else 'diagnostic_subset.csv')
            shutil.copy2(loc/'diagnostic.csv',csv_path)
            done={'run':str(loc),'csv':str(csv_path),'csv_sha256':sha256(csv_path),'full_frame_set':check['submission_frame_set_complete'],
                  'source_report_sha256':sha256(output/'report.json'),'scope':'Experimental candidate; format checks are not accuracy proof'}
            atomic_json(previous,done);print(json.dumps(done,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--track',type=int,choices=(1,2),required=True)
    p.add_argument('--frames',type=int,required=True)
    p.add_argument('--start-frame',type=int,default=0)
    for name in ('data-root','checkpoint','model-repo','prior-run','candidate','output'):
        p.add_argument('--'+name,type=saved_path,required=True)
    p.add_argument('--map-cache',type=saved_path)
    p.add_argument('--budget',type=int,default=6000)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--localize',action='store_true')
    p.add_argument('--save-depth',action='store_true',help='Optional large per-view depth cache; default retains selected point labels and all input/pose hashes')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.budget<100:p.error('Budget must be >=100')
    run(a)
