#!/usr/bin/env python3
"""Forward GPU v2: per-frame map NDT/ICP, balanced source normals, tracking/reacquisition; no IMU."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np

from accuracy.benchmark import create_run_directory
from causal_track3_filter import read_initial_pose
from run_track3_causal import ROOT, TIMING_HEADER, digest, gpu_crop_voxel, select_frames, write_json
from track3_map_only import METHOD, MapOnlyTracker
from track3_registration import Geometry, MapRegistrar, RegistrationConfig

BACKEND = 'cuda_float64_map_ndt_icp_balanced_normals_reacquisition_v2_no_imu'
VRAM_BUDGET_SCOPE = ('Limits only this process PyTorch allocator; excludes NVRTC/driver/cuSOLVER '
    'and other non-Torch allocations, and does not guarantee freedom from OOM.')


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--full', action='store_true')
    group.add_argument('--frames', type=int, default=32)
    p.add_argument('--data-root', type=Path, default=ROOT.parent/'midterm01/data')
    p.add_argument('--map-cache', type=Path, help='Optional SHA-verified lossless XYZ cache under GPU outputs/run')
    p.add_argument('--vram-budget-gib', type=float, default=4.0, help=VRAM_BUDGET_SCOPE)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    if args.frames <= 0:
        p.error('frames must be positive')
    if not np.isfinite(args.vram_budget_gib) or args.vram_budget_gib <= 0:
        p.error('vram-budget-gib must be finite and positive')
    args.data_root = args.data_root.resolve(strict=True)
    return args


def configure_vram_budget(torch, budget_gib):
    # These API calls initialize the CUDA context, before any workload allocation.
    free, total = torch.cuda.mem_get_info()
    budget_bytes = int(budget_gib * 1024**3)
    if budget_bytes > total:
        raise ValueError(f'VRAM budget {budget_gib:g} GiB exceeds device total {total/1024**3:.3f} GiB')
    fraction = budget_bytes / total
    torch.cuda.set_per_process_memory_fraction(float(fraction))
    return dict(budget_gib=budget_gib, budget_bytes=budget_bytes, fraction=fraction,
        initial_free_bytes=free, device_total_bytes=total, scope=VRAM_BUDGET_SCOPE)


def cuda_memory_diagnostics(torch):
    """Best effort: a damaged CUDA context must not hide the original failure."""
    result = {}
    if torch is None:
        return result
    for key, query in (
        ('free_total_bytes', torch.cuda.mem_get_info),
        ('allocated_bytes', torch.cuda.memory_allocated),
        ('reserved_bytes', torch.cuda.memory_reserved),
        ('peak_allocated_bytes', torch.cuda.max_memory_allocated),
        ('peak_reserved_bytes', torch.cuda.max_memory_reserved)):
        try:
            value = query()
            if key == 'free_total_bytes':
                result.update(free_bytes=value[0], total_bytes=value[1])
            else:
                result[key] = value
        except Exception as error:
            result[key+'_error'] = f'{type(error).__name__}: {error}'
    return result


def record_failure(summary, error, torch=None):
    reason = f'{type(error).__name__}: {error}'
    summary['validation_errors'].append(reason)
    summary['failure'] = dict(type=type(error).__name__, message=str(error),
        processed_frames=summary['processed_frames'], failed_original_idx=summary.get('failed_original_idx'))
    summary['cuda_memory'] = cuda_memory_diagnostics(torch)


def validate_output(run, expected):
    errors = []
    matrices = np.loadtxt(run/'result.txt', ndmin=2)
    def records(name):
        with (run/name).open() as stream:
            return list(csv.DictReader(stream))
    rows = records('diagnostic.csv')
    timing = records('timings.csv')
    frames = records('frames.csv')
    obs = [json.loads(x) for x in (run/'observations.jsonl').read_text().splitlines()]
    if matrices.shape != (expected, 12) or not np.isfinite(matrices).all():
        return ['invalid matrix count/shape/finite values']
    for name, records in [('CSV',rows),('timing',timing),('frames',frames),('observations',obs)]:
        if len(records) != expected or [int(x['idx']) for x in records] != list(range(expected)):
            errors.append(name+' count/idx mismatch')
    if errors:
        return errors
    if list(rows[0]) != ['idx','x','y','yaw'] or list(timing[0]) != TIMING_HEADER:
        errors.append('fixed header mismatch')
    poses = matrices.reshape(-1,3,4)
    r = poses[:,:,:3]
    if not np.allclose(r.transpose(0,2,1)@r,np.eye(3),atol=1e-6,rtol=0) or not np.allclose(np.linalg.det(r),1.,atol=1e-6,rtol=0):
        errors.append('improper saved rotation')
    saved = np.array([[float(v[k]) for k in ('x','y','yaw')] for v in rows])
    derived = np.column_stack([poses[:,0,3],poses[:,1,3],np.arctan2(poses[:,1,0],poses[:,0,0])])
    if not np.allclose(saved,derived,atol=1e-8,rtol=0):
        errors.append('matrix/CSV disagreement')
    stamps = [int(f['timestamp_ns']) for f in frames]
    if any(t <= 0 for t in stamps) or any(b <= a for a,b in zip(stamps,stamps[1:])):
        errors.append('nonincreasing/nonpositive frame timestamps')
    for i,(t,f,o) in enumerate(zip(timing,frames,obs)):
        if (int(t['original_idx']) != i or int(t['timestamp_ns']) != int(f['timestamp_ns'])
                or o['timestamp_ns'] != int(f['timestamp_ns']) or o['input_cutoff_ns'] != o['timestamp_ns']
                or o['imu_used'] is not False or not np.isfinite(float(t['scale'])) or float(t['scale']) <= 0):
            errors.append('timestamp/input/scale contract mismatch')
            break
        if not all(np.isfinite(float(t[k])) and float(t[k]) >= 0 for k in ('load_ms','registration_ms','total_ms')):
            errors.append('invalid timings')
            break
    return errors


def main(argv=None, *, method=METHOD, backend=BACKEND, geometry_type=Geometry,
         registrar_type=MapRegistrar, tracker_type=MapOnlyTracker, entrypoint=None, extra_sources=()):
    args = arguments(argv)
    selected = select_frames(args)
    initial = read_initial_pose(args.data_root/'Track3/init_guess.txt')
    plan = dict(method=method,backend=backend,track=3,full=args.full,frames=len(selected),
        registration_config=asdict(RegistrationConfig()),imu_used=False,gnss_used=False,rts_used=False,
        graph_used=False,offline_trajectory_correction=False,deterministic_torch=True,
        prior='previous/current PLY adjacent-cloud rigid registration, then per-frame map NDT + ICP',
        tracking='explicit tracking/lost/reacquisition states; no future-frame corrections',
        source_geometry='training-source normal direction and spatial balancing',
        vram_budget_gib=args.vram_budget_gib,vram_budget_scope=VRAM_BUDGET_SCOPE,
        bootstrap='training P2D NDT/ICP + camera-ray depth selection, three local scale restarts',
        camera_and_depth_assumptions='PLY camera origin/+X forward and raw geometry not independently calibrated',
        ray_median_abs_log_depth_limit=.15,pair_motion_limit_m_s=40.,angular_rate_limit_deg_s=90.,
        replay_supported=False,accuracy='unknown; map/visibility acceptance is not true pose accuracy')
    if args.dry_run:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return 0
    from old_icp_runtime import active_localizations
    active = active_localizations()
    if active:
        raise RuntimeError(f'Another localization is active: {active}')
    run = create_run_directory(ROOT/'outputs/run',method,track=3,full=args.full)
    print('Run: '+str(run),flush=True)
    started = time.perf_counter()
    summary = dict(track=3,method=method,backend=backend,frames=len(selected),expected_frames=len(selected),
        processed_frames=0,map_accepted_frames=0,tracking_lost_frames=0,max_unsupported_seconds=0.,
        tracking_state_counts={},reacquisition_attempted_frames=0,reacquisition_succeeded_frames=0,
        stage_timing_ms={},final_unsupported_seconds=None,final_quality_flags=None,
        returncode=1,output_complete_and_finite=False,submission_frame_set_complete=False,
        validation_errors=[],imu_used=False,localization_executed=False,replay_supported=False,
        submission_compliance_certified=False,accuracy=plan['accuracy'])
    write_json(run/'params.json',plan)
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        plan['vram_allocator']=configure_vram_budget(torch,args.vram_budget_gib)
        torch.use_deterministic_algorithms(True)
        from cuda_scale_input import load_ply_gpu
        write_json(run/'params.json',plan)
        info=json.loads((run/'test_info.json').read_text());info.update(algorithm=method,score=None,localization_executed=False)
        write_json(run/'test_info.json',info)
        (run/'command.sh').write_text(shlex.join([sys.executable,str((Path(__file__) if entrypoint is None else Path(entrypoint)).resolve()),*(sys.argv[1:] if argv is None else argv)])+'\n')
        # Copy the transitive Python source set; source changes fail validation.
        hashes={}
        for source in sorted((ROOT/'scripts').rglob('*.py')):
            if '__pycache__' in source.parts:continue
            relative=source.relative_to(ROOT);target=run/'pipeline_sources'/relative
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
            hashes[str(relative)]=digest(source)
        for source in extra_sources:
            source=Path(source).resolve(strict=True)
            relative=source.relative_to(ROOT);target=run/'pipeline_sources'/relative
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
            hashes[str(relative)]=digest(source)
        write_json(run/'pipeline_source_sha256.json',hashes)
        for name,cmd in [('git_commit.txt',['git','rev-parse','HEAD']),('worktree.patch',['git','diff','--binary','HEAD'])]:
            (run/name).write_bytes(subprocess.check_output(cmd,cwd=ROOT))
        identities={name:dict(path=str(args.data_root/'Track3'/name),sha256=digest(args.data_root/'Track3'/name))
                    for name in ('map.pcd','init_guess.txt')}
        write_json(run/'input_sha256.json',identities)
        prepare=time.perf_counter()
        if args.map_cache:
            from test_track3_map_registration import verified_map_cache
            map_points,cache_info=verified_map_cache(args.map_cache,args.data_root/'Track3/map.pcd')
            write_json(run/'map_cache_reference.json',cache_info)
        else:
            from causal_map_cache import cache_xyz_map
            cache=cache_xyz_map(args.data_root/'Track3/map.pcd',run/'map_cache')
            map_points=np.load(cache,mmap_mode='r',allow_pickle=False)
        geometry=geometry_type('cuda',run/'cuda_cache')
        center=initial[:3,3].copy()
        crop,crop_info=gpu_crop_voxel(map_points,center,torch,np,half_size=65.,leaf=.25)
        scene=registrar_type(crop,geometry=geometry)
        tracker=tracker_type(scene,initial)
        geometry.sync()
        summary['preparation_seconds']=time.perf_counter()-prepare
        write_json(run/'initial_map_tile.json',crop_info)
        write_json(run/'cuda_runtime.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),normal_eigh_batch_size=256,kernel=geometry.kernels.identity,
            cuda_fallback=False,shared_kernel_module_across_frames=True))
        print(f"Preparation: map_points={len(crop)}, time={summary['preparation_seconds']:.3f} s (excluded from frame timings)",flush=True)
        source_rows=[]
        with ((run/'result.txt').open('x') as matrix_file,(run/'diagnostic.csv').open('x',newline='') as csv_file,
              (run/'timings.csv').open('x',newline='') as timing_file,(run/'observations.jsonl').open('x') as obs_file,
              (run/'frames.csv').open('x',newline='') as frame_file):
            cw,tw,fw=csv.writer(csv_file),csv.writer(timing_file),csv.writer(frame_file)
            cw.writerow(['idx','x','y','yaw']);tw.writerow(TIMING_HEADER);fw.writerow(['idx','timestamp_ns','filename'])
            summary['localization_executed']=True
            info['localization_executed']=True;write_json(run/'test_info.json',info)
            for i,(original,ns,path) in enumerate(selected):
                summary['failed_original_idx']=original
                begin=time.perf_counter();raw=load_ply_gpu(path,max_points=12000,leaf=.03)
                loaded=time.perf_counter()
                # Later tile rebuilds are part of this frame's registration time.
                if np.max(abs(tracker.pose[:3,3]-center)) > 20.:
                    center=tracker.pose[:3,3].copy()
                    tracker.scene=None;scene=None;torch.cuda.empty_cache()
                    crop,tile_info=gpu_crop_voxel(map_points,center,torch,np,half_size=65.,leaf=.25)
                    scene=registrar_type(crop,geometry=geometry);tracker.scene=scene
                    write_json(run/f'map_tile_{i:06d}.json',tile_info)
                observation=tracker.step(raw,ns)
                geometry.sync();finished=time.perf_counter()
                pose=np.array(observation['pose']);scale=observation['scale'];accepted=observation['accepted']
                load_ms=(loaded-begin)*1000;reg_ms=(finished-loaded)*1000;total_ms=(finished-begin)*1000
                matrix_file.write(' '.join(format(x,'.12g') for x in pose[:3].ravel())+'\n')
                cw.writerow([i,format(pose[0,3],'.12g'),format(pose[1,3],'.12g'),format(float(np.arctan2(pose[1,0],pose[0,0])),'.12g')])
                tw.writerow([i,original,ns,backend,load_ms,reg_ms,total_ms,accepted,scale]);fw.writerow([i,ns,path.name])
                observation.update(idx=i,original_idx=original,elapsed_seconds=total_ms/1000,
                    load_ms=load_ms,registration_ms=reg_ms,total_ms=total_ms)
                chosen=observation['selected_fit']
                if chosen is not None:
                    observation['heldout_median_m']=chosen['validation']['median_m']
                obs_file.write(json.dumps(observation,allow_nan=False)+'\n')
                for stream in (matrix_file,csv_file,timing_file,obs_file,frame_file):stream.flush()
                # Hash only the CURRENT input, after use; never decode a future cloud.
                source_rows.append(dict(idx=i,original_idx=original,timestamp_ns=ns,filename=path.name,sha256=digest(path)))
                summary['processed_frames']+=1;summary['map_accepted_frames']+=int(accepted)
                summary['tracking_lost_frames']+=int(observation['tracking_lost'])
                summary['max_unsupported_seconds']=max(summary['max_unsupported_seconds'],observation['unsupported_seconds'] or 0.)
                state=observation.get('tracking_state','unknown')
                counts=summary['tracking_state_counts'];counts[state]=counts.get(state,0)+1
                for key in ('reacquisition_attempted','reacquisition_succeeded'):
                    summary[key+'_frames']+=int(bool(observation.get(key,False)))
                for stage,elapsed in observation.get('stage_timing_ms',{}).items():
                    summary['stage_timing_ms'][stage]=summary['stage_timing_ms'].get(stage,0.)+elapsed
                summary['final_unsupported_seconds']=observation.get('unsupported_seconds')
                summary['final_quality_flags']=dict(tracking_lost=bool(observation['tracking_lost']),
                    map_supported_this_frame=bool(accepted),accuracy_certified=False)
                print(f'Frame {i+1}/{len(selected)}: idx={original}, map={"accepted" if accepted else "rejected"}, scale={scale:.6f}, load={load_ms:.2f} ms, registration={reg_ms:.2f} ms, frame={total_ms:.2f} ms',flush=True)
                if not observation['tracking_established']:
                    raise RuntimeError('Bootstrap not supported by current-frame geometry/visibility; no successful localization')
        print('Graph: executed=false, applied=false, time=0.00 ms',flush=True)
        errors=validate_output(run,len(selected))
        for relative,sha in hashes.items():
            if digest(ROOT/relative)!=sha:errors.append('Program changed: '+relative)
        for name,entry in identities.items():
            if digest(entry['path'])!=entry['sha256']:errors.append('Input changed: '+name)
        for row,(_,_,path) in zip(source_rows,selected):
            if digest(path)!=row['sha256']:errors.append('PLY changed: '+path.name)
        if errors:raise ValueError('; '.join(errors))
        summary.pop('failed_original_idx',None)
        summary.update(returncode=0,output_complete_and_finite=True,submission_frame_set_complete=bool(args.full),
            localization_executed=True,csv_sha256=digest(run/'diagnostic.csv'),result_sha256=digest(run/'result.txt'),
            causality='Only current/previous PLY geometry and static map; real prefix comparison recorded separately')
    except (Exception, KeyboardInterrupt) as error:
        record_failure(summary,error,locals().get('torch'))
        print(f'Failed: {type(error).__name__}: {error}',file=sys.stderr,flush=True)
        raise
    finally:
        # Record failures independently; final diagnostics cannot replace an active exception.
        active_exception = sys.exc_info()[0] is not None
        try:
            if 'source_rows' in locals():
                write_json(run/'source_input.json',dict(track=3,frames=source_rows))
            summary['pipeline_elapsed_seconds']=time.perf_counter()-started
            summary['cuda_memory']=cuda_memory_diagnostics(locals().get('torch'))
            write_json(run/'validation.json',summary)
        except Exception as diagnostic_error:
            print(f'Diagnostic write failed: {type(diagnostic_error).__name__}: {diagnostic_error}',file=sys.stderr,flush=True)
            if not active_exception:
                raise
    from run_analysis import generate_after_full
    generate_after_full(run,summary,full=args.full)
    k=summary['map_accepted_frames']
    print(f"Finished: frames={len(selected)}, accepted_map={k}, rejected_map={len(selected)-k}, total={summary['pipeline_elapsed_seconds']:.3f} s",flush=True)
    print('CSV: '+str(run/'diagnostic.csv'),flush=True)
    print('Validation: '+str(run/'validation.json'),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
