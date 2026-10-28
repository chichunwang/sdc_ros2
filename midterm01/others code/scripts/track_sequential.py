"""純點雲追蹤（Track03 模擬：沒有 GNSS／IMU）。
- 預測：非完整約束等速——速度 v 與角速度 ω 由最近 --hist 幀「接受的 ICP 位姿」用中位數估（不是濾波器自己的信念，
  所以不會像 EKF 那樣把一次配準偏差當成速度、越走越偏；10/21 實測 EKF 版會往後退）。
- 量測：ICP 帶固定 σ 的預測先驗（先驗資訊 = c/σ²，c 是 ICP 資訊矩陣的過度自信倍率）：正常路段地圖資訊遠大於先驗、
  等於沒加；特徵少的路段先驗接手，位姿貼著預測走。尺度固定（逐幀估會把前進吸收成縮放而停在原地）。
- 起點：init_guess 直接當初值，貼合差才試 yaw 多點起始；不在 xy 多點起始（Track1 會貼到平行的錯誤結構，實測偏 2.7 m）。
  Track3 的 init_guess 雜訊大（斜 20°、yaw 差 17°、xy 差 1.9 m）→ --init-search 改水平姿態大範圍格點搜尋。
- 起步沒有速度歷史（v=0），ICP 沿車向谷底又太平，前 ~50 幀會落後 ~1 m → 正向跑完再從第 --backfill 幀往回追，取代前段。
- 門檻不過就用預測；連續失敗太多就在預測附近多點起始找回來。
輸出 npz 與 register_all 相容；另輸出相機軌跡 CSV（給 register_all --traj，lever 0,0、yaw 偏角 0）。"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sdcloc.cloud import PlyCache, ground_level_m, preprocess  # noqa: E402
from sdcloc.register import icp_point_to_plane, observability  # noqa: E402


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def set_yaw(R, yaw):
    return Rotation.from_euler('z', yaw - yaw_of(R)).as_matrix() @ R


def rot2(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', type=Path, required=True)
    ap.add_argument('--plys', required=True)
    ap.add_argument('--init-guess', type=Path, required=True)
    ap.add_argument('--scale0', type=float, default=8.6, help='尺度（LoGeR 前相機約 8.6；MapAnything 約 1）')
    ap.add_argument('--scale-file', type=Path, help='逐幀尺度 npy（scale_from_height.py）；給了就逐幀用它，不用 --scale0')
    ap.add_argument('--max-range', type=float, default=25.0)
    ap.add_argument('--overconf', type=float, default=50.0, help='ICP 資訊矩陣的過度自信倍率 c')
    ap.add_argument('--sigma-prior', default='0.5,0.3,1.0', help='預測先驗 σ：along m, lat m, yaw deg（固定、不累積）')
    ap.add_argument('--sigma-z', type=float, default=0.3)
    ap.add_argument('--sigma-rp-deg', type=float, default=1.0)
    ap.add_argument('--hist', type=int, default=10, help='估速度／角速度用最近幾幀接受的位姿')
    ap.add_argument('--max-jump', type=float, default=2.0)
    ap.add_argument('--max-dyaw-deg', type=float, default=5.0)
    ap.add_argument('--max-fails', type=int, default=20, help='連續失敗幾幀就做多點起始找回')
    ap.add_argument('--init-search', action='store_true',
                    help='第 0 幀不信 init_guess 的姿態：roll/pitch 歸零、z 貼路面，在 yaw ±45°、xy ±4 m 格點粗配挑最好的'
                         '（Track3 的 init_guess 斜 20°、yaw 差 17°、xy 差 1.9 m；Track1 只差 0.35 m／1°，不用開）')
    ap.add_argument('--backfill', type=int, default=100,
                    help='正向跑完後從第 N 幀往回追到第 0 幀、取代前 N 幀（0＝關）。Track1：15–45 幀沿車向落後 −1.15/−0.88/−0.54 → '
                         '−0.60/−0.38/−0.01 m，全段 xy RMSE 0.601→0.596（第 N 幀之後逐位元相同）')
    ap.add_argument('--frames', type=int, default=0, help='0＝全部')
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    c = a.overconf
    sp = [float(v) for v in a.sigma_prior.split(',')]
    info_body = np.diag([c / sp[0] ** 2, c / sp[1] ** 2])
    info_yaw = c / np.deg2rad(sp[2]) ** 2
    info_z, info_rp = c / a.sigma_z ** 2, c / np.deg2rad(a.sigma_rp_deg) ** 2

    m = np.load(a.map)
    T0 = np.loadtxt(a.init_guess, delimiter=',')
    xyz = m['xyz']
    keep = (np.abs(xyz[:, 0] - T0[0, 3]) < 2000) & (np.abs(xyz[:, 1] - T0[1, 3]) < 2000)
    mxyz = xyz[keep].astype(np.float64); mn = m['normals'][keep].astype(np.float32)
    del m, xyz
    tree = cKDTree(mxyz)
    map_ground = mn[:, 2] > 0.9
    plys = PlyCache(a.plys)
    n = len(plys) if a.frames == 0 else min(a.frames, len(plys))
    scales = np.load(a.scale_file) if a.scale_file else np.full(len(plys), a.scale0)
    print('map points', len(mxyz), 'frames', n, f'{time.time()-t0:.0f}s', flush=True)

    def make_prior(t_pred, yaw_pred, scale_xy=1.0, scale_yaw=1.0):
        info = np.zeros((6, 6))
        info[:2, :2] = rot2(yaw_pred) @ info_body @ rot2(yaw_pred).T * scale_xy
        info[5, 5] = info_yaw * scale_yaw
        info[2, 2] = info_z; info[3, 3] = info_rp; info[4, 4] = info_rp
        return {'mean_t': t_pred.copy(), 'mean_yaw': yaw_pred, 'info': info}

    def run_icp(raw, R0, t0_, s_k, prior=None, coarse_only=False):
        out = None; Rc, tc = R0.copy(), t0_.copy(); src = is_g = None
        for voxel, corr, iters in ((0.4, 2.0, 15),) if coarse_only else ((0.4, 2.0, 15), (0.2, 0.8, 25)):
            src, is_g = preprocess(raw, s_k, voxel=voxel, max_range=a.max_range)
            if len(src) < 50:
                return None, None, None
            out = icp_point_to_plane(src, mxyz, mn, tree, Rc, tc, max_iter=iters, max_corr=corr, huber=0.3, prior=prior,
                                     workers=2, src_ground=is_g, map_ground=map_ground)
            Rc, tc = out['R'], out['t']
        return out, src, is_g

    def nonground_fit(out, src, is_g):
        p = src[~is_g] @ out['R'].T + out['t']
        if len(p) == 0:
            return np.array([np.nan])
        d, _ = tree.query(p, workers=2)
        return d

    def good(out):
        return out is not None and 'rmse' in out and np.isfinite(out['rmse'])

    def multistart(raw, R_pred, t_pred, s_k, dyaws, dalongs, dlats):
        """在預測附近粗配一輪，取非地面貼合率最高的。"""
        yaw_p = yaw_of(R_pred); fwd = np.array([np.cos(yaw_p), np.sin(yaw_p), 0.0]); left = np.array([-np.sin(yaw_p), np.cos(yaw_p), 0.0])
        best = None
        for dy in dyaws:
            Rg = set_yaw(R_pred, yaw_p + np.deg2rad(dy))
            for da in dalongs:
                for dl in dlats:
                    out, src, is_g = run_icp(raw, Rg, t_pred + da * fwd + dl * left, s_k, coarse_only=True)
                    if not good(out):
                        continue
                    score = float(np.mean(nonground_fit(out, src, is_g) < .5))
                    if best is None or score > best[0]:
                        best = (score, out['R'], out['t'], out['inlier_frac'])
        return best

    def motion_from_history(hist):
        """最近接受的位姿 → 每幀沿車向位移 v 與轉角 ω（中位數）。"""
        if len(hist) < 3:
            return 0.0, 0.0
        H = np.array(hist[-a.hist:])              # (frame, x, y, yaw)
        dk = np.diff(H[:, 0]); dxy = np.diff(H[:, 1:3], axis=0) / dk[:, None]
        ym = H[:-1, 3] + 0.5 * wrap(H[1:, 3] - H[:-1, 3])
        along = np.cos(ym) * dxy[:, 0] + np.sin(ym) * dxy[:, 1]
        dpsi = wrap(H[1:, 3] - H[:-1, 3]) / dk
        return float(np.median(along)), float(np.median(dpsi))

    # ---- 狀態 ----
    R = T0[:3, :3].copy(); t = T0[:3, 3].copy()
    if a.init_search:
        raw0, _ = plys.get(0)
        R = Rotation.from_euler('z', yaw_of(R)).as_matrix()
        gxyz = mxyz[map_ground]
        _, j = cKDTree(gxyz[:, :2]).query(t[:2], k=20)
        t[2] = float(np.median(gxyz[j, 2])) - ground_level_m(raw0 * float(scales[0]))
        best = multistart(raw0, R, t, float(scales[0]), np.arange(-45, 46, 3), (-4, -2, 0, 2, 4), (-4, -2, 0, 2, 4))
        if best is None:
            raise SystemExit('init search: 沒有任何起點收斂')
        print(f'init search: fit {best[0]:.3f}, moved {np.linalg.norm(best[2][:2] - T0[:2, 3]):.2f} m, '
              f'yaw {np.degrees(yaw_of(best[1])):.2f} (init_guess {np.degrees(yaw_of(T0[:3, :3])):.2f}) {time.time()-t0:.0f}s', flush=True)
        R, t = best[1].copy(), best[2].copy()

    def track_pass(order, R, t, hist, sgn, first_special):
        """照 order 逐幀追。sgn=+1 正向、−1 反向：v、w 都是正向時間的量（hist 照處理順序排，差分時 dk 帶正負號），
        反向時預測往回走。first_special：這一趟的第一幀用放寬的先驗與 yaw 多點起始（只給正向從 init 起步用）。"""
        rows = []; n_fail = 0; n_consec = 0; n_recover = 0
        for i, k in enumerate(order):
            first = first_special and i == 0
            row, R, t, hist, ok, recovered = track_frame(k, R, t, hist, sgn, first, n_consec >= a.max_fails)
            rows.append(row)
            if ok:
                n_consec = 0; n_recover += int(recovered)
            else:
                n_fail += 1; n_consec += 1
            if k % 100 == 0:
                print(f'{k}/{n} fails {n_fail} consec {n_consec} v {row["v"]*27.8:.1f} m/s w {np.degrees(row["w"])*27.8:+.1f} deg/s {time.time()-t0:.0f}s', flush=True)
        return rows, n_fail, n_recover

    def track_frame(k, R, t, hist, sgn, first, try_recover):
        raw, _ = plys.get(k)
        s_k = float(scales[k])
        # ---- 預測 ----
        v, w = motion_from_history(hist)
        yaw_prev = yaw_of(R); yaw_pred = yaw_prev + sgn * w; pm = yaw_prev + 0.5 * sgn * w
        t_pred = t + sgn * v * np.array([np.cos(pm), np.sin(pm), 0.0])
        R_pred = set_yaw(R, yaw_pred)
        # 起點先驗放寬（init_guess 可能差 0.5–1 m、1–2°）
        prior = make_prior(t_pred, yaw_pred, scale_xy=0.1, scale_yaw=0.25) if first else make_prior(t_pred, yaw_pred)
        out, src, is_g = run_icp(raw, R_pred, t_pred, s_k, prior=prior)
        ok = good(out) and out['inlier_frac'] > 0.4
        recovered = False
        if ok:
            d = nonground_fit(out, src, is_g)
            jump = float(np.linalg.norm(out['t'][:2] - t_pred[:2])); dyaw = float(np.degrees(abs(wrap(yaw_of(out['R']) - yaw_pred))))
            ok = jump < a.max_jump and dyaw < a.max_dyaw_deg and np.mean(d < .3) > 0.3
            if first and np.mean(d < .5) < 0.5:
                best = multistart(raw, R_pred, t_pred, s_k, (-9, -6, -3, 3, 6, 9), (0.0,), (0.0,))
                if best is not None and best[0] > np.mean(d < .5):
                    out, src, is_g = run_icp(raw, best[1], best[2], s_k, prior=prior)
                    ok = good(out)
                    if ok:
                        d = nonground_fit(out, src, is_g)
                    print(f'frame 0 yaw multi-start: fit {best[0]:.2f}, yaw {np.degrees(yaw_of(best[1]) - yaw_pred):+.1f} deg', flush=True)
        if not ok and try_recover:
            best = multistart(raw, R_pred, t_pred, s_k, (-8, -4, 0, 4, 8), (-3, -1.5, 0, 1.5, 3), (-1.5, 0, 1.5))
            if best is not None and best[0] >= 0.6 and best[3] >= 0.5:
                out, src, is_g = run_icp(raw, best[1], best[2], s_k)
                ok = good(out)
                if ok:
                    d = nonground_fit(out, src, is_g)
                    recovered = True
                    hist = []                     # 速度重新估
                    print(f'frame {k}: recovered by multi-start, fit {best[0]:.2f}, moved {np.linalg.norm(out["t"][:2]-t_pred[:2]):.2f} m', flush=True)
        if ok:
            R, t = out['R'].copy(), out['t'].copy()
            yaw_now = yaw_of(R)
            hist.append((k, t[0], t[1], yaw_now))
            ob = observability(out['information'])
            row = {'frame': k, 'ok': True, 't': t.copy(), 'rotvec': Rotation.from_matrix(R).as_rotvec(),
                   'yaw': yaw_now, 'information': out['information'],
                   'rmse': out['rmse'], 'inlier_frac': out['inlier_frac'], 'inliers': out['inliers'], 'iters': out['iters'],
                   'converged': out['converged'], 'n_src': len(src), 'n_nonground': int((~is_g).sum()),
                   'ng_median_d': float(np.median(d)), 'ng_frac03': float(np.mean(d < .3)),
                   'xy_min_info': ob['xy_min'], 'xy_max_info': ob['xy_max'], 'yaw_info': ob['yaw'], 'rank': ob['rank'],
                   'prior_xy': t_pred[:2].copy(), 'prior_yaw': yaw_pred,
                   'road_z': float(np.nan), 'cam_h': float(-ground_level_m(raw * s_k)), 'scale_est': s_k, 'scale_info': out['scale_info'],
                   'v': v, 'w': w, 'recovered': recovered}
        else:
            R, t = R_pred.copy(), t_pred.copy()
            row = {'frame': k, 'ok': False, 't': t.copy(), 'rotvec': Rotation.from_matrix(R).as_rotvec(),
                   'yaw': yaw_of(R), 'scale_est': s_k, 'v': v, 'w': w, 'recovered': False,
                   'prior_xy': t_pred[:2].copy(), 'prior_yaw': yaw_pred}
        if a.debug:
            fwd = np.array([np.cos(yaw_pred), np.sin(yaw_pred)])
            inn = (t[:2] - t_pred[:2]) @ fwd if ok else float('nan')
            print(f'k {k} ok {int(ok)} pred ({t_pred[0]:.2f},{t_pred[1]:.2f}) icp ({t[0]:.2f},{t[1]:.2f}) inn_along {inn:+.3f} '
                  f'v {v:+.3f} w {np.degrees(w):+.2f} xymin {row.get("xy_min_info", float("nan")):.0f} yawinfo {row.get("yaw_info", float("nan")):.0f} yaw {np.degrees(yaw_of(R)):.2f}', flush=True)
        return row, R, t, hist, ok, recovered

    rows, n_fail, n_recover = track_pass(range(n), R, t, [], 1, True)
    if a.backfill > 0 and n > a.backfill + a.hist:
        # 起步時沒有速度歷史（v=0）車卻每幀走 0.3 m，加上 ICP 沿車向谷底太平，前 ~50 幀會落後 ~1 m 拉不回來；
        # 從第 N 幀（已收斂）帶著速度往回追，取代前 N 幀
        N = a.backfill
        seed = [(r['frame'], r['t'][0], r['t'][1], r['yaw']) for r in rows[N + a.hist:N - 1:-1] if r['ok']]
        back, nf_b, nr_b = track_pass(range(N - 1, -1, -1), Rotation.from_rotvec(rows[N]['rotvec']).as_matrix(),
                                      rows[N]['t'].copy(), seed, -1, False)
        back = back[::-1]
        dal = np.array([(rb['t'][:2] - rf['t'][:2]) @ np.array([np.cos(rf['yaw']), np.sin(rf['yaw'])]) for rb, rf in zip(back, rows[:N])])
        print(f'backfill {N}: backward - forward along (m): frames 0-14 {dal[:15].mean():+.2f}, 15-45 {dal[15:46].mean():+.2f}, '
              f'46-{N - 1} {dal[46:].mean():+.2f}; backward fails {nf_b} recoveries {nr_b}', flush=True)
        rows = back + rows[N:]
        n_fail = sum(1 for r in rows if not r['ok'])
    okm = np.array([r['ok'] for r in rows])
    proto = next(r for r in rows if r['ok'])
    keys = [k_ for k_ in proto if k_ not in ('frame', 'ok')]
    arrays = {'frame': np.array([r['frame'] for r in rows]), 'ok': okm}
    for key in keys:
        fill = np.full(np.shape(proto[key]), np.nan) if isinstance(proto[key], np.ndarray) else np.nan
        arrays[key] = np.array([r[key] if key in r else fill for r in rows])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, scale=a.scale0, **arrays)
    csv = a.out.with_suffix('.cam.csv')
    with csv.open('w', newline='') as f:
        f.write('idx,x,y,yaw\n')
        for r in rows:
            f.write(f"{r['frame']},{r['t'][0]:.6f},{r['t'][1]:.6f},{r['yaw']:.8f}\n")
    print('saved', a.out, 'and', csv, 'ok', int(okm.sum()), '/', n, 'fails', n_fail, 'recoveries', n_recover, f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
