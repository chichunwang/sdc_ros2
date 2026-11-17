#!/usr/bin/env python3
"""VGGT 分支：讓 3D 基礎模型「看影像直接輸出相機位置」（在 midterm02_vggt 容器內，以 docker/vggt/run.sh 啟動）。

    docker/vggt/run.sh python3 scripts/run_vggt.py --experiment V1_vggt
    docker/vggt/run.sh python3 scripts/run_vggt.py --experiment V2_vggt_selfcalib --calib-from outputs/run/<E21>
    docker/vggt/run.sh python3 scripts/run_vggt.py --experiment V1_vggt --start 0 --length 132   # 只跑一段（測試）

VGGT（Meta 2025，facebook/VGGT-1B）一次看 S 張影像，前饋輸出每張的相機位姿（camera-from-world，OpenCV 軸）與內參，
沒有特徵匹配、沒有 BA。一次能看的張數受 GPU 記憶體限制，所以整段 1314 幀切成重疊的段：
- 每段 chunk 幀、相鄰段重疊 overlap 幀；每段的世界座標與尺度都不同（第一張為原點、尺度任意）。
- 段與段的接法（Sim(3)）：重疊幀的相機中心距離比的中位數 → 尺度；重疊幀旋轉的弦平均 → 旋轉；中心平均 → 平移。
  只用中心做 Umeyama 在直線段時繞行進方向的旋轉不可觀測，所以旋轉改由相機姿態決定。
- 重疊幀的位置取兩段的平均（線性權重淡入淡出）。
- 目前沒有閉環：段與段一路接下去，誤差會累積（這是 Private 的風險；Public 只看一段）。
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path('/workspace/midterm02')
sys.path.insert(0, str(ROOT / 'scripts'))
from create_run import create_run  # noqa: E402

EXPERIMENTS = {
    'V1_vggt': dict(calib=False, chunk=64, overlap=24),   # 段長依單段測試：24–80 幀時誤差約 2–7 cm，132 幀時 14–33 cm
    'V2_vggt_selfcalib': dict(calib=True, chunk=64, overlap=24),
}


def load_images(paths, K, dist, w=518, h=336):
    out = []
    for p in paths:
        img = cv2.imread(str(p))
        if dist is not None:
            img = cv2.undistort(img, K, dist, None, K)
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_AREA)
        out.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
    return torch.stack(out)


def infer(model, imgs, device):
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        x = imgs.to(device)[None]
        tokens, ps_idx = model.aggregator(x)
        pose_enc = model.camera_head(tokens)[-1]
    E, Kp = pose_encoding_to_extri_intri(pose_enc.float(), x.shape[-2:])
    E = E[0].cpu().numpy().astype(np.float64)      # (S,3,4) camera-from-world
    T = np.tile(np.eye(4), (len(E), 1, 1))
    T[:, :3, :3] = np.transpose(E[:, :3, :3], (0, 2, 1))
    T[:, :3, 3] = -np.einsum('nji,nj->ni', E[:, :3, :3], E[:, :3, 3])
    return T, Kp[0].cpu().numpy()                  # world-from-camera


def project_so3(M):
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt


def chain_sim3(A, B):
    """求把 B（新段）接到 A（已接好）上的 Sim(3)，A、B 是同一批重疊幀的 world-from-camera。"""
    ca, cb = A[:, :3, 3], B[:, :3, 3]
    ia, ib = np.triu_indices(len(ca), 1)
    da = np.linalg.norm(ca[ia] - ca[ib], axis=1)
    db = np.linalg.norm(cb[ia] - cb[ib], axis=1)
    ok = db > 1e-6
    s = float(np.median(da[ok] / db[ok]))
    R = project_so3(sum(A[i, :3, :3] @ B[i, :3, :3].T for i in range(len(A))))
    t = ca.mean(0) - s * (R @ cb.mean(0))
    return s, R, t


def apply_sim3(T, s, R, t):
    out = T.copy()
    out[:, :3, :3] = R @ T[:, :3, :3]
    out[:, :3, 3] = s * T[:, :3, 3] @ R.T + t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument('--track', default='01')
    ap.add_argument('--calib-from', default=None)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--length', type=int, default=0, help='只跑一段（單一 chunk 測試，不接段）')
    ap.add_argument('--chunk', type=int, default=0)
    a = ap.parse_args()
    exp = dict(EXPERIMENTS[a.experiment])
    if a.chunk:
        exp['chunk'] = a.chunk
    torch.manual_seed(0)
    np.random.seed(0)
    data = ROOT / ('data/data_VSLAM-T1/Track1/data' if a.track == '01' else 'data/data_VSLAM-T2/Track2/data')
    result_name = 'result_%d.csv' % int(a.track)
    ts = [r[0] for r in list(csv.reader(open(data / 'sample_submission.csv')))[1:]]
    ci = json.loads((data / 'camera_info.json').read_text())
    K = np.array([[ci['fx'], 0, ci['cx']], [0, ci['fy'], ci['cy']], [0, 0, 1.0]])
    dist, calib_src = None, None
    if exp['calib']:
        c = json.loads((ROOT / a.calib_from / 'test_info.json').read_text())['summary']['graph']['bundle']['calibration']
        K[0, 2], K[1, 2] = c['cx'], c['cy']
        dist = np.array([c['k1'], c['k2'], 0, 0, 0.0])
        calib_src = c
    paths = [data / 'images' / ('%s.jpg' % t) for t in ts]

    from vggt.models.vggt import VGGT
    device = 'cuda'
    model = VGGT.from_pretrained('facebook/VGGT-1B').to(device).eval()

    t0 = time.time()
    if a.length:
        sel = list(range(a.start, a.start + a.length))
        T, Kp = infer(model, load_images([paths[i] for i in sel], K, dist), device)
        out = ROOT / ('outputs/cache/vggt_test_%s_%d_%d.npz' % (a.experiment, a.start, a.length))
        np.savez(out, idx=np.array(sel), T=T, K=Kp)
        print('wrote %s  %.1f s  peak GPU %.1f GB  fx(pred, 518 px) %.1f' % (
            out, time.time() - t0, torch.cuda.max_memory_allocated() / 1e9, np.median(Kp[:, 0, 0])))
        return

    run = create_run('full', a.track)
    info_path = run / 'test_info.json'
    info = json.loads(info_path.read_text())
    info.update(status='running', version='vggt-a288dd0', experiment=a.experiment, seed=0, frame_count=len(ts),
                command=' '.join(sys.argv),
                parameters=dict(exp, calib_from=a.calib_from, calibration_source=calib_src, image_size=[518, 336],
                                weights='facebook/VGGT-1B'))
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')

    N, L, O = len(ts), exp['chunk'], exp['overlap']
    starts = list(range(0, max(N - O, 1), L - O))
    if starts[-1] + L < N:
        starts.append(N - L)
    poses = np.full((N, 4, 4), np.nan)
    weight = np.zeros(N)
    acc_c = np.zeros((N, 3))
    scales = []
    prev = None
    for k, s0 in enumerate(starts):
        idx = np.arange(s0, min(s0 + L, N))
        T, _ = infer(model, load_images([paths[i] for i in idx], K, dist), device)
        if prev is not None:
            ov = np.flatnonzero(np.isin(idx, prev))
            gi = idx[ov]
            s, R, t = chain_sim3(poses[gi], T[ov])
            T = apply_sim3(T, s, R, t)
            scales.append(s)
        # 重疊區位置淡入淡出；旋轉用新段（新段在重疊區後半更居中）
        w = np.ones(len(idx))
        if prev is not None:
            n_ov = int(np.isin(idx, prev).sum())
            w[:n_ov] = (np.arange(n_ov) + 1) / (n_ov + 1)
        acc_c[idx] += w[:, None] * T[:, :3, 3]
        weight[idx] += w
        poses[idx, :3, :3] = T[:, :3, :3]
        poses[idx, 3] = [0, 0, 0, 1]
        poses[idx, :3, 3] = acc_c[idx] / weight[idx, None]
        prev = idx
        print('chunk %d/%d frames %d–%d  %.0f s' % (k + 1, len(starts), idx[0], idx[-1], time.time() - t0), flush=True)
    t1 = time.time()
    with open(run / result_name, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['timestamp'] + ['m%d%d' % (i, j) for i in range(4) for j in range(4)])
        for name, T in zip(ts, poses):
            T = T.copy()
            T[:3, :3] = project_so3(T[:3, :3])
            wr.writerow([name] + ['%.10g' % v for v in T.ravel()])
    import subprocess
    chk = subprocess.run([sys.executable, str(ROOT / 'scripts/check_submission.py'), str(run / result_name),
                          str(data / 'sample_submission.csv')], capture_output=True, text=True)
    status = 'success' if chk.returncode == 0 else 'failed'
    info.update(status=status, timings_seconds=dict(vggt=t1 - t0),
                summary=dict(chunks=len(starts), chain_scales=scales),
                submission=dict(check_output=chk.stdout[-1000:] + chk.stderr[-1000:]))
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    # 成功的 full 自動產生唯讀圖表（失敗只記 analysis_status.json，不影響定位結果）
    subprocess.run([sys.executable, str(ROOT / 'scripts/analyze_external_run.py'), str(run)])
    print('run:', run)
    print('submission check:', status, chk.stdout[-200:])


if __name__ == '__main__':
    main()
