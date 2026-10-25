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
from track3.imu_parameters import resolve_imu_config, resolve_fusion_config

TIMINGS = 'idx,original_idx,timestamp_ns,backend,load_ms,registration_ms,total_ms,map_accepted,scale'.split(',')
OUTPUT_FILES = ['poses.txt', 'result_3.csv', 'timings.csv', 'frame_manifest.csv', 'observations.jsonl', 'pose_trace.jsonl']


def diagnostic_pose(value):
    """Keep strict JSON even when a rejected solver output is nonfinite."""
    array = np.asarray(value, dtype=float)
    return array.tolist() if array.shape == (4, 4) and np.isfinite(array).all() else None


def capture_progress(info, run, *, accepted_count, consecutive_rejections,
                     max_consecutive_rejections, last_accepted_idx,
                     last_accepted_timestamp_ns, registration_total,
                     localization_start, memory=None):
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
    if memory is not None:info['gpu_memory']=memory.summary()
    info['output_sha256'] = {name: sha256(run/name) for name in OUTPUT_FILES+['gpu_memory.jsonl','imu_sources.json']
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
    visualizer=None; visual_prior=None; camera_model=None; imu_reader=None; camera_clock=None; motion_constraints=None; memory=None; tlocal=None; registration_total=None
    accepted_count=0; consecutive_rejections=0; max_consecutive_rejections=0
    last_accepted_idx=None; last_accepted_timestamp_ns=None
    def save_progress():
        if imu_reader is not None:
            info['imu_samples_consumed']=imu_reader.count
            info['imu_input']=imu_reader.summary()
        capture_progress(info, run, accepted_count=accepted_count,
                         consecutive_rejections=consecutive_rejections,
                         max_consecutive_rejections=max_consecutive_rejections,
                         last_accepted_idx=last_accepted_idx,
                         last_accepted_timestamp_ns=last_accepted_timestamp_ns,
                         registration_total=registration_total, localization_start=tlocal,memory=memory)
    try:
        cfg=json.loads(a.config.read_text()); info['version']=cfg['version']; info['source']='official_front_camera_ply'
        info['imu_used']=cfg['prediction']=='imu_eskf'; info['prediction']=cfg['prediction']
        if info['imu_used']:cfg['imu']=resolve_imu_config(resolve_fusion_config(cfg['imu']))
        if info['imu_used'] and cfg.get('motion_constraints',False):
            from track3.motion_parameters import PARAMETERS,VISUAL_VELOCITY_STD_MPS,STATIONARY_VELOCITY_STD_MPS,ENABLE_MOVING_VISUAL_SCALE
            cfg['motion_parameters']=dict(PARAMETERS,visual_velocity_std_mps=VISUAL_VELOCITY_STD_MPS,
                                         stationary_velocity_std_mps=STATIONARY_VELOCITY_STD_MPS,
                                         enable_moving_visual_scale=ENABLE_MOVING_VISUAL_SCALE)
        write_json(run/'parameters.json',cfg)
        if a.full and cfg['expected_frames'] != 3142:
            raise ValueError('Full Track3 must contain exactly 3142 frames')
        from track3.gpu import cp, sync, Registration
        if cfg.get('gpu_memory') is not None:
            from track3.memory import GpuMemoryMonitor
            memory=GpuMemoryMonitor(cp,run/'gpu_memory.jsonl',cfg['gpu_memory'])
            memory.checkpoint('context_ready')
        device=cp.cuda.runtime.getDeviceProperties(0)
        info['gpu']=dict(name=device['name'].decode(), compute_capability=[device['major'],device['minor']], cupy=cp.__version__)
        data=Path(cfg['data_dir'])
        rows=frames(data/cfg['ply_subdir'], expected=cfg['expected_frames'])
        rows=rows if a.full else rows[:a.max_frames]
        source_files=[data/'map.pcd',data/'init_guess.txt',a.config]
        write_json(run/'input_sources.json',{str(s):sha256(s) for s in source_files})
        T=np.loadtxt(data/'init_guess.txt',delimiter=',')
        if not valid_pose(T): raise ValueError('Initial pose is not a valid rigid transform')
        if cfg['prediction']=='imu_eskf':
            from track3.imu import ImuReader
            from track3.inertial import InertialState
            icfg=cfg['imu']
            bag=data/icfg['bag_file']
            write_json(run/'imu_sources.json',dict(
                path=str(bag),sha256=sha256(bag),topic=icfg['topic'],frame=icfg['frame_id'],
                timestamp_contract=('availability=PLY/bag arrival; state and velocity dt=paired camera header; IMU integration=header' if icfg.get('time_basis')=='camera_header' else
                                    'bag arrival clock shared with PLY names; headers retained as evidence'),
                extrinsic_status=icfg['extrinsic_status'],T_ply_imu=icfg['T_ply_imu'],
                gravity_map_mps2=icfg['gravity_map_mps2'],
                noise_source=icfg['noise_source'],orientation_used=False,gnss_used=False))
            state=InertialState(T,cfg['initial_scale'],icfg,cfg['registration']['scale_bounds'])
            imu_reader=ImuReader(bag,icfg['topic'],icfg['frame_id'])
            if icfg.get('time_basis','bag_arrival')=='camera_header':
                from track3.time_alignment import CameraClock,AcquisitionImuReader
                camera_clock=CameraClock(bag)
                imu_reader=AcquisitionImuReader(imu_reader)
                info['state_time_basis']='paired_camera_header; availability_bag_stamp_preserved'
            info['imu_extrinsic_status']=icfg['extrinsic_status']
        else:
            state=PoseState(T,cfg['initial_scale'],cfg['prediction'],cfg['registration']['scale_bounds'])
        if cfg['prediction']=='visual_pnp':
            from track3.visual_prior import VisualPrior
            visual_prior=VisualPrior(data,cfg['visual_prior'],cfg['expected_frames'])
            camera_model=visual_prior
        elif cfg.get('camera_mask',False):
            from track3.visual_prior import CameraMask
            camera_model=CameraMask(data)
        if camera_model is not None:
            camera=data/'Track3_images/camera_info'
            write_json(run/'camera_sources.json',{str(path):sha256(path) for path in [
                camera/'gige_100_f_hdr_camera_info.yaml',camera/'gige_100_f_hdr_mask.png']})
        if imu_reader is not None and cfg.get('motion_constraints',False):
            if camera_model is None:raise ValueError('Motion constraints require camera calibration/mask')
            from track3.motion_constraints import MotionConstraints
            motion_constraints=MotionConstraints(data,camera_model,cfg['expected_frames'])
            info['scale_policy']='map_sim3_while_moving; causal_visual_scale_only_during_verified_rest' if not ENABLE_MOVING_VISUAL_SCALE else 'causal_visual_scale'
            info['stationary_frames']=0
        map_points, count=load_map(data/'map.pcd',cfg['map_leaf_m']); sync()
        if memory is not None:memory.checkpoint('map_ready',trim=True)
        info['map_points_read']=count; info['map_points_downsampled']=len(map_points)
        registration=Registration(map_points,cfg['registration'])
        registration.memory_monitor=memory
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
                if camera_model is not None:
                    from track3.visual_prior import filter_by_camera_mask
                    points,camera_mask_diagnostics=filter_by_camera_mask(points,camera_model.K,camera_model.mask)
                prediction_start=time.perf_counter()
                constraint_diagnostics=None;stationary=False;input_scale=None
                measurement_stamp=stamp if camera_clock is None else camera_clock.for_frame(stamp)
                samples=(imu_reader.take_for_frame(stamp,measurement_stamp) if camera_clock is not None else
                         imu_reader.take_until(stamp) if imu_reader is not None else None)
                if motion_constraints is not None:
                    input_scale,stationary,constraint_diagnostics=motion_constraints.observe(stamp,points,state.scale,samples,state.initialized,
                                                                                           measurement_timestamp_ns=measurement_stamp)
                    if not stationary and not ENABLE_MOVING_VISUAL_SCALE:input_scale=None
                    prior=state.predict(measurement_stamp,samples,input_scale=input_scale,stationary=stationary,
                                        visual_motion=motion_constraints.visual_motion)
                else:
                    prior=(state.predict(measurement_stamp,samples) if imu_reader is not None else state.predict(stamp,motion=motion))
                prediction_ms=(time.perf_counter()-prediction_start)*1000
                scale=state.scale
                load_ms=(time.perf_counter()-tick)*1000
                input_scale=float(scale)
                reg_start=time.perf_counter()
                if motion_constraints is not None:
                    estimate=registration.align(points,prior,scale,initialize=not state.initialized,
                                                fixed_scale=state.initialized and (stationary or ENABLE_MOVING_VISUAL_SCALE))
                else:
                    estimate=registration.align(points,prior,scale,initialize=not state.initialized)
                sync()
                reg_ms=(time.perf_counter()-reg_start)*1000; registration_total+=reg_ms/1000
                memory_record=None if memory is None else memory.checkpoint('frame_end',idx=idx,trim=True)
                update_start=time.perf_counter()
                decision=state.commit(estimate)
                update_ms=(time.perf_counter()-update_start)*1000
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
                    measurement_timestamp_ns=measurement_stamp,
                    predicted_T=np.asarray(prior, dtype=float).tolist(),
                    registered_T=diagnostic_pose(estimate['T']),
                    final_T=np.asarray(T, dtype=float).tolist(),
                    input_scale=input_scale,
                    proposed_scale=float(estimate['scale']) if np.isfinite(estimate['scale']) else None,
                    final_scale=float(scale),
                    registration_accepted=bool(estimate['accepted']), map_accepted=bool(adopted),
                    registration_reason=decision['reason'], pose_decision=decision['decision'],
                    prediction=decision['prediction'], delta=decision['delta'],inertial=decision.get('inertial'),
                    motion_constraints=constraint_diagnostics
                ), allow_nan=False)+'\n'); pose_trace.flush()
                if visualizer is not None:
                    visualizer.publish_frame(getattr(registration,'display_source',points),T,scale,stamp,registration,
                                             accepted=adopted,reason=decision['reason'])
                if visual_prior is not None:
                    visual_prior.store(getattr(registration,'display_source',points),scale)
                if motion_constraints is not None:
                    motion_constraints.store(scale,anchored=state.initialized,
                                             bootstrap=decision['inertial']['update']['bootstrap'])
                    if stationary:info['stationary_frames']+=1
                poses.write(' '.join(f'{v:.12g}' for v in T[:3].ravel())+'\n');poses.flush()
                cw.writerow([idx,f'{T[0,3]:.12g}',f'{T[1,3]:.12g}',f'{yaw(T[:3,:3]):.12g}']);result.flush()
                total_ms=(time.perf_counter()-tick)*1000
                tw.writerow([idx,row['original_idx'],stamp,'cuda_voxel_ndt_icp',f'{load_ms:.6f}',f'{reg_ms:.6f}',f'{total_ms:.6f}',int(adopted),f'{scale:.12g}']);timing.flush()
                mw.writerow([idx,row['original_idx'],stamp,str(row['path']),point_sha]);manifest.flush()
                cost=estimate.get('cost')
                cost=float(cost) if cost is not None and np.isfinite(cost) else None
                obs=dict(idx=idx,original_idx=row['original_idx'],timestamp_ns=stamp,elapsed_seconds=time.perf_counter()-tlocal,
                         measurement_timestamp_ns=measurement_stamp,
                         registration_accepted=bool(estimate['accepted']),
                         map_accepted=adopted,registration_reason=decision['reason'],coverage=estimate.get('coverage'),
                         registration_converged=estimate.get('converged'),
                         registration_stages=estimate.get('stages'),
                         preprocessing=estimate.get('preprocessing'),initialization=estimate.get('initialization'),
                         visual_prior=motion_diagnostics,
                         inertial=decision.get('inertial'),prediction_ms=prediction_ms,measurement_update_ms=update_ms,
                         motion_constraints=constraint_diagnostics,scale_optimized=estimate.get('scale_optimized'),
                         camera_mask=camera_mask_diagnostics,
                         model_diagnostics=estimate.get('model_diagnostics'),
                         pose_decision=decision['decision'], prediction=decision['prediction'],
                         delta=decision['delta'],
                         consecutive_map_rejections=consecutive_rejections,
                         map_age_seconds=map_age_seconds,
                         cost=cost,
                         observable=estimate.get('observable'),normalized_min_eigenvalue=estimate.get('normalized_min_eigenvalue'),
                         scale=scale)
                obs['gpu_memory']=memory_record
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
            if imu_reader is not None and not plot.returncode:
                imu_plot=subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_imu.py')),'--run-dir',str(run)])
                if imu_plot.returncode:print('Localization complete; IMU plots failed, see imu_analysis_status.json',file=sys.stderr)
        print(f"Completed: {run} map updates={accepted_count}/{len(rows)}",flush=True)
        return 0
    except BaseException as error:
        info.update(status='failed',failure=f'{type(error).__name__}: {error}',elapsed_seconds=time.perf_counter()-start)
        save_progress()
        write_json(run/'test_info.json',info);(run/'failure.log').write_text(traceback.format_exc());traceback.print_exc()
        return 130 if isinstance(error,KeyboardInterrupt) else 1
    finally:
        if visualizer is not None: visualizer.close()
        if memory is not None:memory.close()
        if imu_reader is not None:imu_reader.close()
        if camera_clock is not None:camera_clock.close()

if __name__=='__main__':
    def terminate(signum, frame): raise KeyboardInterrupt('Termination signal')
    signal.signal(signal.SIGTERM,terminate)
    sys.exit(main())
