"""全部 frame 的 6-DOF point-to-plane ICP（多程序）。輸出 npz：每幀 pose、資訊矩陣、殘差統計。
初值：作者先驗 x,y,yaw；z = 地圖路面 + 相機離地；roll/pitch 0。兩階段：粗（voxel 0.4、對應 2 m）→ 細（0.2、0.8 m）。"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sdcloc.cloud import PlyCache, ground_level_m, preprocess  # noqa: E402
from sdcloc.register import icp_point_to_plane, observability  # noqa: E402

G = {}


def init_worker(map_path, plys_path, prior_path, scale, estimate_scale=False, scale_file=None, init_path=None, class_match=False,
                accumulate=0, traj_path=None, lever=(0.0, 0.0), cam_yaw_offset_deg=0.0, max_range=50.0, depth_weight=0.0, huber=0.3):
    G['estimate_scale'] = estimate_scale
    G['max_range'] = max_range
    G['depth_weight'] = depth_weight
    G['huber'] = huber
    G['scale_per_frame'] = np.load(scale_file) if scale_file else None
    G['accumulate'] = accumulate
    if init_path:
        prev = np.load(init_path)
        G['init'] = {int(f): (prev['rotvec'][k], prev['t'][k]) for k, f in enumerate(prev['frame']) if prev['ok'][k]}
    else:
        G['init'] = None
    if accumulate > 0:
        # 每幀相機在地圖中的位姿：x/y/yaw 來自平滑軌跡（參考點扣 lever），z/roll/pitch 來自上一輪配準
        traj = pl.read_csv(traj_path).to_numpy()
        vy = traj[:, 3]
        cy = vy + np.deg2rad(cam_yaw_offset_deg)
        cx = traj[:, 1] - (np.cos(vy) * lever[0] - np.sin(vy) * lever[1])
        cyy = traj[:, 2] - (np.sin(vy) * lever[0] + np.cos(vy) * lever[1])
        poses = {}
        for k in range(len(traj)):
            if k not in G['init']:
                continue
            rv, t_prev = G['init'][k]
            R_prev = Rotation.from_rotvec(rv).as_matrix()
            yaw_prev = np.arctan2(R_prev[1, 0], R_prev[0, 0])
            R_rest = Rotation.from_euler('z', -yaw_prev).as_matrix() @ R_prev      # 去掉 yaw 剩 roll/pitch
            R = Rotation.from_euler('z', cy[k]).as_matrix() @ R_rest
            poses[k] = (R, np.array([cx[k], cyy[k], t_prev[2]]))
        G['acc_poses'] = poses
    G['prior'] = pl.read_csv(prior_path).to_numpy()
    m = np.load(map_path)
    xyz = m['xyz']
    # 只留軌跡周圍 ±100 m 的地圖，省記憶體
    lo = G['prior'][:, 1:3].min(axis=0) - 100; hi = G['prior'][:, 1:3].max(axis=0) + 100
    keep = (xyz[:, 0] >= lo[0]) & (xyz[:, 0] <= hi[0]) & (xyz[:, 1] >= lo[1]) & (xyz[:, 1] <= hi[1])
    G['mxyz'] = xyz[keep].astype(np.float64)
    G['mn'] = m['normals'][keep].astype(np.float32)
    del m, xyz, keep
    G['tree'] = cKDTree(G['mxyz'])
    gm = G['mn'][:, 2] > 0.9
    G['map_ground'] = gm
    G['gxyz'] = G['mxyz'][gm]
    G['gtree'] = cKDTree(G['gxyz'][:, :2])
    G['class_match'] = class_match
    G['plys'] = PlyCache(plys_path)
    G['scale'] = scale


def accumulated_cloud(k, voxel):
    """把 k±N 幀各自前處理後，用平滑軌跡的相對位姿疊到第 k 幀的相機座標（公尺）。地面／非地面分開再 voxel。"""
    from sdcloc.mapio import voxel_downsample
    Rk, tk = G['acc_poses'][k]
    parts_g, parts_n = [], []
    n = len(G['plys'])
    for j in range(max(0, k - G['accumulate']), min(n, k + G['accumulate'] + 1)):
        if j not in G['acc_poses']:
            continue
        raw_j, _ = G['plys'].get(j)
        s_j = float(G['scale_per_frame'][j]) if G['scale_per_frame'] is not None else G['scale']
        src_j, g_j = preprocess(raw_j, s_j, voxel=voxel, max_range=G['max_range'])
        if len(src_j) == 0:
            continue
        Rj, tj = G['acc_poses'][j]
        p = (src_j @ Rj.T + tj - tk) @ Rk          # 到第 k 幀相機座標
        parts_g.append(p[g_j]); parts_n.append(p[~g_j])
    out, flags = [], []
    for parts, flag in ((parts_g, True), (parts_n, False)):
        if parts:
            pts = np.vstack(parts)
            if len(pts):
                ds, _, _ = voxel_downsample(pts, voxel)
                out.append(ds); flags.append(np.full(len(ds), flag))
    if not out:
        return np.zeros((0, 3)), np.zeros(0, bool)
    return np.vstack(out), np.concatenate(flags)


def register_frame(i):
    raw, _ = G['plys'].get(i)
    s = float(G['scale_per_frame'][i]) if G['scale_per_frame'] is not None else G['scale']
    g0 = ground_level_m(raw * s) / s      # 原始單位的地面高度（公尺窗口判斷）
    x, y, yaw = G['prior'][i, 1], G['prior'][i, 2], G['prior'][i, 3]
    _, j = G['gtree'].query([x, y], k=20)
    rz = float(np.median(G['gxyz'][j, 2]))
    if G['accumulate'] > 0 and i in G['acc_poses']:
        R, t = G['acc_poses'][i]
        R, t = R.copy(), t.copy()
    elif G['init'] is not None and i in G['init']:
        rv, t_prev = G['init'][i]
        R = Rotation.from_rotvec(rv).as_matrix()
        t = np.array(t_prev, dtype=float)
    else:
        R = Rotation.from_euler('z', yaw).as_matrix()
        t = np.array([x, y, rz - g0 * s])
    out = None
    s_est = 1.0      # 相對於 s 的倍率
    for voxel, corr, iters in ((0.4, 2.0, 15), (0.2, 0.8, 25)):
        if G['accumulate'] > 0 and i in G['acc_poses']:
            src, is_g = accumulated_cloud(i, voxel)
        else:
            src, is_g = preprocess(raw, s, voxel=voxel, max_range=G['max_range'])
        if len(src) < 50:
            break
        # 單目深度誤差約隨距離平方長大：權重 1/(1+(r/d0)²)，d0 處權重減半；0＝不加權（只靠 max_range 硬切）。
        # 正規化成平均 1：只改相對輕重，資訊矩陣的整體大小才和不加權時可比（不然 smoother 會以為配準很不可靠）
        w = None
        if G['depth_weight'] > 0:
            w = 1.0 / (1.0 + (np.linalg.norm(src, axis=1) / G['depth_weight']) ** 2)
            w /= w.mean()
        out = icp_point_to_plane(src, G['mxyz'], G['mn'], G['tree'], R, t, max_iter=iters, max_corr=corr, huber=G['huber'],
                                 estimate_scale=G['estimate_scale'], s0=s_est, workers=2, src_weights=w,
                                 src_ground=is_g if G['class_match'] else None,
                                 map_ground=G['map_ground'] if G['class_match'] else None)
        R, t, s_est = out['R'], out['t'], out['s']
    if out is None or 'rmse' not in out:
        return {'frame': i, 'ok': False}
    p = (s_est * src[~is_g]) @ R.T + t
    d, _ = G['tree'].query(p, workers=1) if len(p) else (np.array([]), None)
    ob = observability(out['information'])
    return {'frame': i, 'ok': True, 't': t, 'rotvec': Rotation.from_matrix(R).as_rotvec(),
            'yaw': float(np.arctan2(R[1, 0], R[0, 0])), 'information': out['information'],
            'rmse': out['rmse'], 'inlier_frac': out['inlier_frac'], 'inliers': out['inliers'], 'iters': out['iters'],
            'converged': out['converged'], 'n_src': len(src), 'n_nonground': int((~is_g).sum()),
            'ng_median_d': float(np.median(d)) if len(d) else np.nan, 'ng_frac03': float(np.mean(d < .3)) if len(d) else np.nan,
            'xy_min_info': ob['xy_min'], 'xy_max_info': ob['xy_max'], 'yaw_info': ob['yaw'], 'rank': ob['rank'],
            'prior_xy': np.array([x, y]), 'prior_yaw': yaw, 'road_z': rz, 'cam_h': -g0 * s * s_est,
            'scale_est': s * s_est, 'scale_info': out['scale_info']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', type=Path, required=True)
    ap.add_argument('--plys', type=Path, required=True)
    ap.add_argument('--prior', type=Path, required=True)
    ap.add_argument('--scale', type=float, default=8.6)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--frames', default='all', help='all 或 start:stop:step')
    ap.add_argument('--procs', type=int, default=3)
    ap.add_argument('--estimate-scale', action='store_true')
    ap.add_argument('--scale-file', type=Path, help='逐幀尺度 npy（smooth_scale.py 的輸出）；給了就忽略 --scale')
    ap.add_argument('--init', type=Path, help='用這個 register_all npz 的 pose 當初值')
    ap.add_argument('--class-match', action='store_true', help='地面只配地面、非地面只配非地面')
    ap.add_argument('--accumulate', type=int, default=0, help='前後各 N 幀疊起來配（需要 --traj、--init）')
    ap.add_argument('--traj', type=Path, help='平滑軌跡 CSV（idx,x,y,yaw，參考點）用來算幀間相對運動與初值')
    ap.add_argument('--lever', default='0,0', help='參考點相對相機的 lever "along,lat"')
    ap.add_argument('--cam-yaw-offset-deg', type=float, default=0.0, help='相機 yaw − 車體 yaw')
    ap.add_argument('--max-range', type=float, default=50.0, help='只用這個距離內的點（m）；遠點的單目深度誤差大')
    ap.add_argument('--huber', type=float, default=0.3, help='點到面殘差的 Huber 門檻（m）')
    ap.add_argument('--depth-weight', type=float, default=0.0,
                    help='依深度加權：權重 1/(1+(r/d0)²) 的 d0（m）；0＝不加權。10/25 測過沒用：Track1 25 m 加權（d0 5/8/12）對 I_ns 都比不加權差，'
                         'Track3 12 m 內幾乎沒變；硬切 --max-range 12 比較好')
    a = ap.parse_args()
    if a.accumulate > 0 and (not a.traj or not a.init):
        ap.error('--accumulate 需要 --traj 與 --init')
    n = len(PlyCache(a.plys))
    idx = list(range(n)) if a.frames == 'all' else list(range(*[int(v) for v in a.frames.split(':')]))
    t0 = time.time()
    with Pool(a.procs, initializer=init_worker,
              initargs=(str(a.map), str(a.plys), str(a.prior), a.scale, a.estimate_scale,
                        str(a.scale_file) if a.scale_file else None, str(a.init) if a.init else None, a.class_match,
                        a.accumulate, str(a.traj) if a.traj else None, tuple(float(v) for v in a.lever.split(',')),
                        a.cam_yaw_offset_deg, a.max_range, a.depth_weight, a.huber)) as pool:
        results = []
        for k, r in enumerate(pool.imap(register_frame, idx, chunksize=4)):
            results.append(r)
            if k % 100 == 0:
                print(f'{k}/{len(idx)} {time.time()-t0:.0f}s', flush=True)
    keys = [k for k in results[0] if k not in ('frame', 'ok')]
    ok = np.array([r['ok'] for r in results])
    arrays = {'frame': np.array([r['frame'] for r in results]), 'ok': ok}
    for k in keys:
        vals = [r.get(k) for r in results]
        proto = next(v for v, o in zip(vals, ok) if o)
        fill = np.full(np.shape(proto), np.nan) if isinstance(proto, np.ndarray) else np.nan
        arrays[k] = np.array([v if o else fill for v, o in zip(vals, ok)])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, scale=a.scale, **arrays)
    print('saved', a.out, 'ok', int(ok.sum()), '/', len(ok), f'{time.time()-t0:.0f}s')
    print(json.dumps({'ng_frac03_mean': float(np.nanmean(arrays['ng_frac03'])), 'rank6_frac': float(np.mean(arrays['rank'] == 6)),
                      'median_shift_xy': float(np.nanmedian(np.linalg.norm(arrays['t'][:, :2] - arrays['prior_xy'], axis=1)))}, indent=1))


if __name__ == '__main__':
    main()
