#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import signal
import numpy as np

# Resolve track3 from this run's snapshot without replacing the PYTHONPATH
# assembled by the ROS entrypoint (rclpy and sensor_msgs depend on it).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track3.geometry import valid_pose, yaw
from track3.io import frames, load_ply, load_map, sha256, write_json
from track3.localization import PoseState

TIMINGS = 'idx,original_idx,timestamp_ns,backend,load_ms,registration_ms,total_ms,map_accepted,scale'.split(',')
OUTPUT_FILES = ['poses.txt', 'result_3.csv', 'timings.csv', 'frame_manifest.csv', 'observations.jsonl', 'pose_trace.jsonl']


def diagnostic_pose(value):
    """Keep strict JSON even when a rejected solver output is nonfinite."""
    array = np.asarray(value, dtype=float)
    return array.tolist() if array.shape == (4, 4) and np.isfinite(array).all() else None


def capture_progress(info, run, *, accepted_count, consecutive_rejections,
                     max_consecutive_rejections, last_accepted_idx,
                     last_accepted_timestamp_ns, registration_total,
                     localization_start):
    """Preserve measured progress even when preparation or a later frame fails."""
    info.update(map_accepted_frames=accepted_count,
                consecutive_map_rejections=consecutive_rejections,
                max_consecutive_map_rejections=max_consecutive_rejections,
                last_map_accepted_idx=last_accepted_idx,
                last_map_accepted_timestamp_ns=last_accepted_timestamp_ns)
    if registration_total is not None:
        info['registration_seconds'] = registration_total
    if localization_start is not None:
        info['localization_seconds'] = time.perf_counter()-localization_start
    info['output_sha256'] = {name: sha256(run/name) for name in OUTPUT_FILES
                             if (run/name).is_file()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True); p.add_argument('--run-dir',type=Path, required=True)
    mode=p.add_mutually_exclusive_group(required=True); mode.add_argument('--full',action='store_true'); mode.add_argument('--max-frames',type=int)
    p.add_argument('--publish-rviz', action='store_true', help='Publish live map/cloud/pose/path; run.sh enables this')
    a=p.parse_args()
    if a.max_frames is not None and not 0<a.max_frames<3142: p.error('Short test requires 1..3141 frames')
    run=a.run_dir.resolve()
    if not run.is_dir() or not (run/'source_manifest.json').exists(): p.error('Use scripts/run.sh to create an immutable source snapshot')
    if (run/'test_info.json').exists(): p.error('Run directory already used')
    os.environ['ROS_LOG_DIR']=str(run/'ros_log')
    start=time.perf_counter(); info=dict(status='running',mode='full' if a.full else 'test',frame_count=0,version=None,score=None,score_source=None)
    write_json(run/'test_info.json',info)
    visualizer=None; visual_prior=None; tlocal=None; registration_total=None
    accepted_count=0; consecutive_rejections=0; max_consecutive_rejections=0
    last_accepted_idx=None; last_accepted_timestamp_ns=None
    def save_progress():
        capture_progress(info, run, accepted_count=accepted_count,
                         consecutive_rejections=consecutive_rejections,
                         max_consecutive_rejections=max_consecutive_rejections,
                         last_accepted_idx=last_accepted_idx,
                         last_accepted_timestamp_ns=last_accepted_timestamp_ns,
                         registration_total=registration_total, localization_start=tlocal)
    try:
        cfg=json.loads(a.config.read_text()); info['version']=cfg['version']; info['source']='official_front_camera_ply'
        info['imu_used']=False; info['prediction']=cfg['prediction']
        write_json(run/'parameters.json',cfg)
        if a.full and cfg['expected_frames'] != 3142:
            raise ValueError('Full Track3 must contain exactly 3142 frames')
        from track3.gpu import cp, sync, Registration
        device=cp.cuda.runtime.getDeviceProperties(0)
        info['gpu']=dict(name=device['name'].decode(), compute_capability=[device['major'],device['minor']], cupy=cp.__version__)
        data=Path(cfg['data_dir'])
        rows=frames(data/cfg['ply_subdir'], expected=cfg['expected_frames'])
        rows=rows if a.full else rows[:a.max_frames]
        source_files=[data/'map.pcd',data/'init_guess.txt',a.config]
        write_json(run/'input_sources.json',{str(s):sha256(s) for s in source_files})
        T=np.loadtxt(data/'init_guess.txt',delimiter=',')
        if not valid_pose(T): raise ValueError('Initial pose is not a valid rigid transform')
        state=PoseState(T,cfg['initial_scale'],cfg['prediction'],cfg['registration']['scale_bounds'])
        if cfg['prediction']=='visual_pnp':
            from track3.visual_prior import VisualPrior
            visual_prior=VisualPrior(data,cfg['visual_prior'],cfg['expected_frames'])
            camera=data/'Track3_images/camera_info'
            write_json(run/'camera_sources.json',{str(path):sha256(path) for path in [
                camera/'gige_100_f_hdr_camera_info.yaml',camera/'gige_100_f_hdr_mask.png']})
        map_points, count=load_map(data/'map.pcd',cfg['map_leaf_m']); sync()
        info['map_points_read']=count; info['map_points_downsampled']=len(map_points)
        registration=Registration(map_points,cfg['registration'])
        if a.publish_rviz:
            from track3.visualization import RvizPublisher
            visualizer=RvizPublisher()
            visualizer.publish_map(cp.asnumpy(map_points),rows[0]['timestamp_ns'])
        info['preparation_seconds']=time.perf_counter()-start
        warm=time.perf_counter(); cp.sum(cp.arange(4096,dtype=cp.float64)**2).get(); sync()
        info['warmup_seconds']=time.perf_counter()-warm
        info['warmup_scope']='CUDA reduction; first registration includes lazy kernel compilation'
        registration_total=0.; tlocal=time.perf_counter()
        with (run/'poses.txt').open('x') as poses,(run/'result_3.csv').open('x',newline='') as result, \
             (run/'timings.csv').open('x',newline='') as timing,(run/'observations.jsonl').open('x') as observations, \
             (run/'frame_manifest.csv').open('x',newline='') as manifest, \
             (run/'pose_trace.jsonl').open('x') as pose_trace:
            cw=csv.writer(result);cw.writerow(['idx','x','y','yaw'])
            tw=csv.writer(timing);tw.writerow(TIMINGS)
            mw=csv.writer(manifest);mw.writerow(['idx','original_idx','timestamp_ns','path','sha256'])
            for idx,row in enumerate(rows):
                tick=time.perf_counter(); stamp=row['timestamp_ns']
                points=load_ply(row['path']); point_sha=sha256(row['path'])
                motion=None;motion_diagnostics=None;camera_mask_diagnostics=None
                if visual_prior is not None:
                    motion,motion_diagnostics=visual_prior.predict(stamp,state.initialized)
                    motion_diagnostics['image_sha256']=sha256(motion_diagnostics['image'])
                    from track3.visual_prior import filter_by_camera_mask
                    points,camera_mask_diagnostics=filter_by_camera_mask(points,visual_prior.K,visual_prior.mask)
                prior=state.predict(stamp,motion=motion); scale=state.scale
                load_ms=(time.perf_counter()-tick)*1000
                input_scale=float(scale)
                reg_start=time.perf_counter()
                estimate=registration.align(points,prior,scale,initialize=not state.initialized); sync()
                reg_ms=(time.perf_counter()-reg_start)*1000; registration_total+=reg_ms/1000
                decision=state.commit(estimate)
                adopted=decision['accepted']; scale=state.scale
                if adopted:
                    accepted_count+=1; consecutive_rejections=0
                    last_accepted_idx=row['original_idx']; last_accepted_timestamp_ns=stamp
                if not adopted:
                    consecutive_rejections+=1
                    max_consecutive_rejections=max(max_consecutive_rejections,consecutive_rejections)
                map_age_seconds=None if last_accepted_timestamp_ns is None else (stamp-last_accepted_timestamp_ns)*1e-9
                T=state.T.copy()
                if not valid_pose(T): raise ValueError(f'Invalid output pose at frame {idx}')
                # Diagnostics only: record all three poses, regardless of registration acceptance.
                pose_trace.write(json.dumps(dict(
                    idx=idx, original_idx=row['original_idx'], timestamp_ns=stamp,
                    predicted_T=np.asarray(prior, dtype=float).tolist(),
                    registered_T=diagnostic_pose(estimate['T']),
                    final_T=np.asarray(T, dtype=float).tolist(),
                    input_scale=input_scale,
                    proposed_scale=float(estimate['scale']) if np.isfinite(estimate['scale']) else None,
                    final_scale=float(scale),
                    registration_accepted=bool(estimate['accepted']), map_accepted=bool(adopted),
                    registration_reason=decision['reason'], pose_decision=decision['decision'],
                    prediction=decision['prediction'], delta=decision['delta']
                ), allow_nan=False)+'\n'); pose_trace.flush()
                if visualizer is not None:
                    visualizer.publish_frame(getattr(registration,'display_source',points),T,scale,stamp,registration,
                                             accepted=adopted,reason=decision['reason'])
                if visual_prior is not None:
                    visual_prior.store(getattr(registration,'display_source',points),scale)
                poses.write(' '.join(f'{v:.12g}' for v in T[:3].ravel())+'\n');poses.flush()
                cw.writerow([idx,f'{T[0,3]:.12g}',f'{T[1,3]:.12g}',f'{yaw(T[:3,:3]):.12g}']);result.flush()
                total_ms=(time.perf_counter()-tick)*1000
                tw.writerow([idx,row['original_idx'],stamp,'cuda_voxel_ndt_icp',f'{load_ms:.6f}',f'{reg_ms:.6f}',f'{total_ms:.6f}',int(adopted),f'{scale:.12g}']);timing.flush()
                mw.writerow([idx,row['original_idx'],stamp,str(row['path']),point_sha]);manifest.flush()
                cost=estimate.get('cost')
                cost=float(cost) if cost is not None and np.isfinite(cost) else None
                obs=dict(idx=idx,original_idx=row['original_idx'],timestamp_ns=stamp,elapsed_seconds=time.perf_counter()-tlocal,
                         map_accepted=adopted,registration_reason=decision['reason'],coverage=estimate.get('coverage'),
                         registration_converged=estimate.get('converged'),
                         registration_stages=estimate.get('stages'),
                         preprocessing=estimate.get('preprocessing'),initialization=estimate.get('initialization'),
                         visual_prior=motion_diagnostics,
                         camera_mask=camera_mask_diagnostics,
                         model_diagnostics=estimate.get('model_diagnostics'),
                         pose_decision=decision['decision'], prediction=decision['prediction'],
                         delta=decision['delta'],
                         consecutive_map_rejections=consecutive_rejections,
                         map_age_seconds=map_age_seconds,
                         cost=cost,
                         observable=estimate.get('observable'),normalized_min_eigenvalue=estimate.get('normalized_min_eigenvalue'),
                         scale=scale)
                observations.write(json.dumps(obs,allow_nan=False)+'\n');observations.flush()
                info['frame_count']=idx+1
                if idx==0 or (idx+1)%25==0:
                    save_progress(); write_json(run/'test_info.json',info)
                    print(f"idx={row['original_idx']} frames={idx+1}/{len(rows)} map={int(adopted)} scale={scale:.4f} reg={reg_ms:.1f}ms reason={decision['reason']} decision={decision['decision']} rejected={consecutive_rejections}",flush=True)
        info.update(status='success' if accepted_count>0 else 'failed',map_accepted_frames=accepted_count,
                    registration_seconds=registration_total,localization_seconds=time.perf_counter()-tlocal,graph_seconds=0.,
                    elapsed_seconds=time.perf_counter()-start,
                    limitation='No ground truth accuracy validation; frame/scale assumptions in parameters.json')
        if accepted_count==0: info['failure']='No accepted map updates; CSV is not a usable localization result'
        save_progress()
        write_json(run/'test_info.json',info)
        if info['status']!='success': return 2
        if a.full:
            # Plot status and time live separately, preserving final localization evidence.
            plot=subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_run.py')),'--run-dir',str(run)])
            if plot.returncode: print('Localization complete; plots failed, see analysis_status.json',file=sys.stderr)
        print(f"Completed: {run} map updates={accepted_count}/{len(rows)}",flush=True)
        return 0
    except BaseException as error:
        info.update(status='failed',failure=f'{type(error).__name__}: {error}',elapsed_seconds=time.perf_counter()-start)
        save_progress()
        write_json(run/'test_info.json',info);(run/'failure.log').write_text(traceback.format_exc());traceback.print_exc()
        return 130 if isinstance(error,KeyboardInterrupt) else 1
    finally:
        if visualizer is not None: visualizer.close()

if __name__=='__main__':
    def terminate(signum, frame): raise KeyboardInterrupt('Termination signal')
    signal.signal(signal.SIGTERM,terminate)
    sys.exit(main())
