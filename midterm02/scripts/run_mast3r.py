#!/usr/bin/env python3
"""MASt3R-SLAM 第三個獨立估計器（在 midterm02_mast3r 容器內，以 docker/mast3r/run.sh 啟動）。

    docker/mast3r/run.sh python3 /workspace/midterm02/scripts/run_mast3r.py --experiment M1_mast3r
    docker/mast3r/run.sh python3 /workspace/midterm02/scripts/run_mast3r.py --experiment M2_mast3r_selfcalib --calib-from outputs/run/<E21>

與官方 main.py 的差別（演算法與參數不變，只改執行方式與輸出）：
- 單一程序：後端（因子圖最佳化、影像檢索閉環、重定位）在主程序內、每加一個 keyframe 立刻執行，等同官方 single_thread 模式，結果可重現。
- 官方只輸出 keyframe 位姿；這裡記下每一幀相對其參考 keyframe 的 Sim(3)，最後接到最佳化後的 keyframe 上，
  得到每一幀的位姿（與 DROID 的 traj_filler 同一想法）。
- 內參：M1 用 camera_info.json 的針孔內參（use_calib）；M2 用自校正估出的主點與徑向畸變（MASt3R-SLAM 會自己校正影像）。
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/workspace/midterm02')
sys.path.insert(0, str(ROOT / 'scripts'))
os.chdir('/opt/MASt3R-SLAM')
sys.path.insert(0, '/opt/MASt3R-SLAM')
from create_run import create_run, digest  # noqa: E402
import types  # noqa: E402
sys.modules.setdefault('pyrealsense2', types.ModuleType('pyrealsense2'))   # 官方 dataloader 匯入 RealSense 相機套件；讀影像資料夾用不到

EXPERIMENTS = {
    'M1_mast3r': dict(calib=False),
    'M2_mast3r_selfcalib': dict(calib=True),
}
CKPT = ROOT / 'outputs/cache/mast3r/checkpoints'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument('--track', default='01')
    ap.add_argument('--calib-from', default=None)
    ap.add_argument('--frames', type=int, default=0, help='只跑前 N 幀（測試）')
    a = ap.parse_args()
    exp = EXPERIMENTS[a.experiment]
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    # 官方 checkpoint 內含 argparse.Namespace（訓練參數）；只把它加入 torch.load 的允許清單，不關閉 weights_only 檢查
    torch.serialization.add_safe_globals([argparse.Namespace])
    import torch.multiprocessing as mp
    import lietorch
    from mast3r_slam.config import load_config, config
    load_config('config/calib.yaml')
    config['use_calib'] = True
    config['single_thread'] = True
    config['dataset']['subsample'] = 1
    from mast3r_slam.dataloader import Intrinsics, MonocularDataset
    from mast3r_slam.frame import SharedKeyframes, create_frame
    from mast3r_slam.global_opt import FactorGraph
    from mast3r_slam.lietorch_utils import as_SE3
    from mast3r_slam.mast3r_utils import load_mast3r, load_retriever, mast3r_inference_mono
    from mast3r_slam.tracker import FrameTracker
    from mast3r_slam.mast3r_utils import resize_img

    data = ROOT / ('data/data_VSLAM-T1/Track1/data' if a.track == '01' else 'data/data_VSLAM-T2/Track2/data')
    result_name = 'result_%d.csv' % int(a.track)
    ts = [r[0] for r in list(csv.reader(open(data / 'sample_submission.csv')))[1:]]
    if a.frames:
        ts = ts[:a.frames]
    ci = json.loads((data / 'camera_info.json').read_text())
    calib = [ci['fx'], ci['fy'], ci['cx'], ci['cy']]
    calib_src = None
    if exp['calib']:
        src = json.loads((ROOT / a.calib_from / 'test_info.json').read_text())
        c = src['summary']['graph']['bundle']['calibration']
        calib = [ci['fx'], ci['fy'], c['cx'], c['cy'], c['k1'], c['k2'], 0.0, 0.0]
        calib_src = c

    class Images(MonocularDataset):
        def __init__(self):
            super().__init__()
            self.rgb_files = [str(data / 'images' / ('%s.jpg' % t)) for t in ts]
            self.timestamps = np.arange(len(ts), dtype=np.float64)
            self.use_calibration = True
            self.save_results = False
            self.camera_intrinsics = Intrinsics.from_calib(self.img_size, ci['width'], ci['height'], calib)

    run = create_run('full' if not a.frames else 'test', a.track)
    info_path = run / 'test_info.json'
    info = json.loads(info_path.read_text())
    info.update(status='running', version='mast3r-slam-e6f4e3d', experiment=a.experiment, seed=0, frame_count=len(ts),
                command=' '.join(sys.argv),
                parameters=dict(calibration=calib, calibration_source=calib_src, calib_from=a.calib_from,
                                config=json.loads(json.dumps(config, default=str)),
                                checkpoint_sha256=digest(CKPT / 'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth')))
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')

    ds = Images()
    h, w = ds.get_img_shape()[0]
    manager = mp.Manager()
    keyframes = SharedKeyframes(manager, h, w, buffer=2048)
    device = 'cuda:0'
    model = load_mast3r(str(CKPT / 'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'), device=device)
    K = torch.from_numpy(ds.camera_intrinsics.K_frame).to(device, dtype=torch.float32)
    keyframes.set_intrinsics(K)
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval = load_retriever(model, str(CKPT / 'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth'),
                               device=device)
    tracker = FrameTracker(model, keyframes, device)

    def backend_step(idx):
        """官方 run_backend 對一個新 keyframe 做的事：相鄰 keyframe＋影像檢索候選建邊，再解因子圖。"""
        kf_idx = [idx - 1] if idx > 0 else []
        frame = keyframes[idx]
        kf_idx += retrieval.update(frame, add_after_query=True, k=config['retrieval']['k'],
                                   min_thresh=config['retrieval']['min_thresh'])
        kf_idx = sorted(set(kf_idx) - {idx})
        if kf_idx:
            factor_graph.add_factors(kf_idx, [idx] * len(kf_idx), config['local_opt']['min_match_frac'])
        factor_graph.solve_GN_calib()

    def relocalize(frame):
        """官方 relocalization：用影像檢索找舊 keyframe，成功就把這幀當新 keyframe 接上。"""
        kf_idx = list(retrieval.update(frame, add_after_query=False, k=config['retrieval']['k'],
                                       min_thresh=config['retrieval']['min_thresh']))
        if not kf_idx:
            return False
        keyframes.append(frame)
        n = len(keyframes)
        if factor_graph.add_factors([n - 1] * len(kf_idx), kf_idx, config['reloc']['min_match_frac'],
                                    is_reloc=config['reloc']['strict']):
            retrieval.update(frame, add_after_query=True, k=config['retrieval']['k'],
                             min_thresh=config['retrieval']['min_thresh'])
            keyframes.T_WC[n - 1] = keyframes.T_WC[kf_idx[0]].clone()
            factor_graph.solve_GN_calib()
            return True
        keyframes.pop_last()
        return False

    rel = [None] * len(ts)          # (keyframe 索引, T_kf⁻¹·T_frame)（Sim3 資料）
    mode = 'init'
    last_T = None
    n_reloc = n_reloc_ok = 0
    t0 = time.time()
    for i in range(len(ts)):
        _, img = ds[i]
        T_WC = lietorch.Sim3.Identity(1, device=device) if last_T is None else last_T
        frame = create_frame(i, img, T_WC, img_size=ds.img_size, device=device)
        if mode == 'init':
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            keyframes.append(frame)
            backend_step(len(keyframes) - 1)
            rel[i] = (len(keyframes) - 1, None)
            last_T = frame.T_WC
            mode = 'track'
            continue
        if mode == 'track':
            ref = len(keyframes) - 1
            add_kf, _, try_reloc = tracker.track(frame)
            if try_reloc:
                mode = 'reloc'
            else:
                if add_kf:
                    keyframes.append(frame)
                    backend_step(len(keyframes) - 1)
                    rel[i] = (len(keyframes) - 1, None)
                else:
                    Tk = keyframes.T_WC[ref]
                    rel[i] = (ref, (lietorch.Sim3(Tk).inv() * frame.T_WC).data.clone())
                last_T = frame.T_WC
        if mode == 'reloc':
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            n_reloc += 1
            if relocalize(frame):
                n_reloc_ok += 1
                rel[i] = (len(keyframes) - 1, None)
                last_T = keyframes.T_WC[len(keyframes) - 1]
                mode = 'track'
        if rel[i] is None:   # 追蹤失敗且尚未重定位：沿用上一幀的相對關係
            rel[i] = rel[i - 1]
        if i % 200 == 0:
            print('frame %d/%d  keyframes %d  reloc %d/%d  %.0f s' % (i, len(ts), len(keyframes), n_reloc_ok, n_reloc,
                                                                      time.time() - t0), flush=True)
    t1 = time.time()
    poses = []
    for i in range(len(ts)):
        k, r = rel[i]
        Tk = lietorch.Sim3(keyframes.T_WC[k])
        T = Tk if r is None else Tk * lietorch.Sim3(r)
        poses.append(as_SE3(T).matrix().cpu().numpy().reshape(4, 4))
    with open(run / result_name, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['timestamp'] + ['m%d%d' % (i, j) for i in range(4) for j in range(4)])
        for name, T in zip(ts, poses):
            T = T.astype(np.float64)
            U, _, Vt = np.linalg.svd(T[:3, :3])
            T[:3, :3] = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
            T[3] = [0, 0, 0, 1]
            wr.writerow([name] + ['%.10g' % v for v in T.ravel()])
    import subprocess
    status = 'success'
    chk_out = ''
    if not a.frames:
        chk = subprocess.run([sys.executable, str(ROOT / 'scripts/check_submission.py'), str(run / result_name),
                              str(data / 'sample_submission.csv')], capture_output=True, text=True)
        status = 'success' if chk.returncode == 0 else 'failed'
        chk_out = chk.stdout[-1000:] + chk.stderr[-1000:]
    info.update(status=status, timings_seconds=dict(slam=t1 - t0),
                summary=dict(keyframes=len(keyframes), reloc_attempts=n_reloc, reloc_success=n_reloc_ok,
                             factor_graph_edges=int(len(factor_graph.ii))),
                submission=dict(check_output=chk_out))
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    # 成功的 full 自動產生唯讀圖表（失敗只記 analysis_status.json，不影響定位結果）
    subprocess.run([sys.executable, str(ROOT / 'scripts/analyze_external_run.py'), str(run)])
    print('run:', run)
    print('submission check:', status, chk_out[-200:])


if __name__ == '__main__':
    main()
