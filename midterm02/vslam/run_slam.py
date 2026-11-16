#!/usr/bin/env python3
"""Midterm02 GPU 視覺－幾何 SLAM 執行入口（在容器 midterm02_vslam 內執行）。

    python vslam/run_slam.py --track 01                # full：全部影像，產生 result_1.csv
    python vslam/run_slam.py --track 01 --frames 200   # 短段測試：前 200 幀
    python vslam/run_slam.py --track 01 --experiment E3_motion_gicp   # 消融實驗（見 gslam/params.py）

每次執行都在 outputs/run/ 建立新的唯一目錄並凍結程式快照；參數改 vslam/gslam/params.py。
"""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vslam'))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np  # noqa: E402

VERSION = 'gslam-0.3.0'


def gpu_snapshot():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=10).stdout.strip()
        return out
    except Exception as exc:  # 沒有 nvidia-smi 不影響計算
        return 'nvidia-smi unavailable: %s' % exc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--track', choices=['01', '02'], required=True)
    ap.add_argument('--frames', type=int, default=0, help='只跑前 N 幀（0 = 全部，即 full）')
    ap.add_argument('--start', type=int, default=0, help='起始幀索引（短段測試用）')
    ap.add_argument('--rviz', default=None, metavar='HOST:PORT',
                    help='把執行狀態串流給 RViz bridge（scripts/dev.sh slam 會自動帶入 rviz:5600）')
    ap.add_argument('--experiment', default=None, help='gslam/params.py 的 EXPERIMENTS 名稱；預設用 EXPERIMENT')
    a = ap.parse_args()

    import check_submission
    import create_run
    from gslam import params as params_mod
    from gslam.dataset import Dataset
    from gslam import outputs

    params, experiment = params_mod.load(a.experiment)
    track = params['track']['track' + a.track]
    ds = Dataset(ROOT / track['data'])
    stop = len(ds) if a.frames <= 0 else min(len(ds), a.start + a.frames)
    indices = list(range(a.start, stop))
    kind = 'full' if len(indices) == len(ds) else 'test'
    run = create_run.create_run(kind, track['code'])
    print('run:', run, flush=True)
    for sub in ('debug/features', 'debug/cloud'):
        (run / sub).mkdir(parents=True, exist_ok=True)
    info_path = run / 'test_info.json'
    outputs.update_json(info_path, status='running', version=VERSION, experiment=experiment, parameters=params,
                        seed=params['seed'], frame_count=len(indices), frame_range=[a.start, stop],
                        input_sha256=ds.input_digest(), missing_images=len(ds.missing_images),
                        command=' '.join(sys.argv), rviz=bool(a.rviz))
    outputs.write_rows(run / 'frame_manifest.csv', ['idx', 'timestamp_ns', 'image', 'ply'], ds.manifest_rows(indices))
    timestamps = [ds.timestamps[i] for i in indices]
    t_all = time.perf_counter()
    try:
        # GPU 驗證：CuPy 在這裡初始化 CUDA 並編譯 kernel；沒有 GPU 會直接失敗，不會改用 CPU
        from gslam.gpu_cloud import device_info
        import cupy as cp
        import cv2
        gpu = device_info()
        print('GPU:', gpu, flush=True)
        outputs.update_json(info_path, gpu=gpu, gpu_before=gpu_snapshot())
        np.random.seed(params['seed'])
        cv2.setRNGSeed(params['seed'])
        from gslam.pipeline import Slam
        viz = None
        if a.rviz:
            from gslam.viz_stream import VizStream
            viz = VizStream(a.rviz, ds.K, ds.size)
            print('RViz stream:', a.rviz, flush=True)
        slam = Slam(ds, params, debug_dir=run / 'debug', viz=viz)
        for local, i in enumerate(indices):
            rec = slam.process(local, i)
            if local % 20 == 0 or rec['tracking_status'] == 'tracking_failure' or rec['loop_status'] == 'accepted':
                print('idx %5d/%d  %-16s fit=%s rmse=%s pnp=%s kf=%d loop=%s  %.0f ms' % (
                    local, len(indices), rec['tracking_status'], outputs._fmt(rec['fitness']),
                    outputs._fmt(rec['inlier_rmse']), rec['pnp_inliers'], len(slam.keyframes), rec['loop_status'] or '-',
                    rec['total_ms']), flush=True)
        slam_seconds = time.perf_counter() - t_all
        t_graph = time.perf_counter()
        odom, optimized, graph_info = slam.finalize()
        graph_seconds = time.perf_counter() - t_graph
        if viz is not None:
            viz.graph([kf.T for kf in slam.keyframes], [(c, q) for c, q, _ in slam.loops])
            viz.final(odom, optimized)

        outputs.write_rows(run / 'timings.csv', outputs.TIMING_COLUMNS, slam.records)
        outputs.write_rows(run / 'frames.csv', outputs.FRAME_COLUMNS, slam.records)
        outputs.write_rows(run / 'keyframes.csv', ['kf_id', 'idx', 'timestamp_ns'],
                           [[kf.id, kf.idx, timestamps[kf.idx]] for kf in slam.keyframes])
        outputs.write_rows(run / 'loop_closures.csv', outputs.LOOP_COLUMNS, slam.loop_rows)
        outputs.write_rows(run / 'loop_candidates.csv', outputs.LOOP_DETAIL_COLUMNS, slam.loop_rows)
        outputs.write_poses(run / 'poses_odometry.csv', timestamps, odom)
        outputs.write_poses(run / 'poses_optimized.csv', timestamps, optimized)
        if slam.bundle_info is not None:    # BA 之前（只有 pose graph）的結果，用來比較 BA 的效果
            outputs.write_poses(run / 'poses_pose_graph.csv', timestamps, slam.pose_graph_poses)
        if slam.odom_scaled is not None:   # 尺度圖修正長度後、尚未做 pose graph 的里程計
            outputs.write_poses(run / 'poses_odometry_scaled.csv', timestamps, slam.odom_scaled)
            outputs.write_rows(run / 'scale_graph.csv', ['idx', 'timestamp_ns', 'log_scale'],
                               [[i, timestamps[i], float(v)] for i, v in enumerate(slam.sigma)])

        kf_indices = [kf.idx for kf in slam.keyframes]
        outputs.save_trajectory(str(run / 'debug' / 'trajectory_preview'), odom, optimized, kf_indices, slam.loops,
                                kf_indices)
        status = [r['tracking_status'] for r in slam.records]
        summary = dict(
            keyframes=len(slam.keyframes), loops_accepted=len(slam.loops), loop_candidates_verified=len(slam.loop_rows),
            graph_optimizations=slam.optimizations, graph=graph_info,
            tracking_failures=status.count('tracking_failure'), recovered=status.count('recovered'),
            visual_ok_frames=int(sum(r['visual_ok'] for r in slam.records)),
            median_registration_ms=float(np.median([r['registration_ms'] for r in slam.records[1:]] or [0])),
            median_total_ms=float(np.median([r['total_ms'] for r in slam.records])),
            gpu_pool_peak_mib=int(cp.get_default_memory_pool().total_bytes() // 2 ** 20))
        timings = dict(slam=slam_seconds, final_graph=graph_seconds)

        result_check = None
        if kind == 'full':
            result = run / track['result']
            outputs.write_poses(result, timestamps, optimized)
            errors = check_submission.check(str(result), str(ds.sample_path)) if ds.sample_path else ['找不到 sample_submission.csv']
            result_check = dict(file=result.name, sha256=outputs.sha256(result), errors=errors[:20], passed=not errors)
            print('submission check:', 'OK' if not errors else 'FAIL %s' % errors[:5], flush=True)
            t_plot = time.perf_counter()
            try:
                outputs.save_analysis(run, odom, optimized, slam.records, kf_indices, slam.loops, slam.loop_rows)
            except Exception as exc:  # 圖表失敗不影響定位結果
                outputs.update_json(run / 'analysis_status.json', status='failed', error=repr(exc),
                                    traceback=traceback.format_exc())
                print('analysis failed:', exc, flush=True)
            timings['analysis'] = time.perf_counter() - t_plot
        timings['total'] = time.perf_counter() - t_all
        outputs.update_json(info_path, status='success', summary=summary, timings_seconds=timings,
                            submission=result_check, gpu_after=gpu_snapshot())
        print('done:', summary, flush=True)
        print('run:', run, flush=True)
    except BaseException as exc:
        outputs.update_json(info_path, status='stopped' if isinstance(exc, KeyboardInterrupt) else 'failed',
                            error=repr(exc), traceback=traceback.format_exc(),
                            timings_seconds=dict(total=time.perf_counter() - t_all))
        raise


if __name__ == '__main__':
    main()
