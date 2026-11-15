"""輸出：官方格式 pose CSV、run 目錄內各紀錄檔、headless 圖表（Matplotlib Agg，不需要顯示器）。"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HEADER = ['timestamp'] + ['m%d%d' % (r, c) for r in range(4) for c in range(4)]
TIMING_COLUMNS = ['idx', 'timestamp_ns', 'backend', 'load_ms', 'preprocess_ms', 'registration_ms', 'total_ms',
                  'is_keyframe', 'fitness', 'inlier_rmse']
FRAME_COLUMNS = ['idx', 'timestamp_ns', 'points_raw', 'points_used', 'keypoints', 'matches', 'essential_inliers',
                 'pnp_inliers', 'visual_inlier_ratio', 'visual_reprojection_px', 'visual_ok', 'prior_used',
                 'registration_method', 'attempts', 'converged', 'iterations', 'correspondences', 'fitness',
                 'inlier_rmse', 'normalized_min_eigenvalue', 'gate_reasons', 'tracking_status', 'is_keyframe',
                 'loop_status', 'estimated_scale', 'visual_scale_ratio', 'feature_ms', 'matching_ms',
                 'registration_ms', 'total_ms', 'step_translation_m', 'step_rotation_deg', 'ground_height', 'cloud_scale',
                 'vg_translation_m', 'vg_rotation_deg', 'visual_mode', 'flow_px', 'visual_length_m',
                 'visual_length_mad_m', 'pnp_length_m']
LOOP_COLUMNS = ['query_kf', 'match_kf', 'accepted', 'fitness', 'inlier_rmse', 'reason']
LOOP_DETAIL_COLUMNS = LOOP_COLUMNS + ['votes', 'matches', 'essential_inliers', 'pnp_inliers',
                                      'translation_disagreement_m', 'rotation_disagreement_deg', 'T_rel']


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return '%.6g' % v
    return str(v)


def write_rows(path, columns, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([_fmt(r[c]) for c in columns] if isinstance(r, dict) else [_fmt(x) for x in r])


def write_poses(path, timestamps, poses):
    """官方格式：timestamp（整數字串）＋ 4×4 camera-to-world，row-major；最後一列固定 0,0,0,1。"""
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for t, T in zip(timestamps, poses):
            T = np.asarray(T, dtype=np.float64)
            if not np.all(np.isfinite(T)):
                raise ValueError('pose 含 NaN/Inf：timestamp %s' % t)
            values = [repr(float(x)) for x in T[:3].ravel()] + ['0.0', '0.0', '0.0', '1.0']
            w.writerow([str(t)] + values)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def update_json(path, **fields):
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    data.update(fields)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def save_cloud_topdown(path, target, source, title=''):
    """俯視圖（x 右、z 前）：灰＝局部子圖，紅＝配準後的目前幀。"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 7))
    step_t = max(1, len(target) // 60000)
    step_s = max(1, len(source) // 30000)
    ax.scatter(target[::step_t, 0], target[::step_t, 2], s=0.3, c='0.6', label='local submap')
    ax.scatter(source[::step_s, 0], source[::step_s, 2], s=0.3, c='tab:red', label='registered frame')
    ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('z [m]'); ax.set_title(title)
    ax.legend(markerscale=20, loc='upper right')
    fig.savefig(str(path), dpi=110, bbox_inches='tight')
    plt.close(fig)


def _xyz(poses):
    return np.array([T[:3, 3] for T in poses])


def save_trajectory(path_stem, odom, optimized, kf_indices, loops, kf_frame_index):
    plt = _plt()
    a, b = _xyz(odom), _xyz(optimized)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(a[:, 0], a[:, 2], color='0.6', lw=1, label='odometry')
    ax.plot(b[:, 0], b[:, 2], color='tab:blue', lw=1.2, label='optimized')
    ax.scatter(b[kf_indices, 0], b[kf_indices, 2], s=4, c='tab:orange', label='keyframes', zorder=3)
    for n, (c, q, _) in enumerate(loops):
        i, j = kf_frame_index[c], kf_frame_index[q]
        ax.plot([b[i, 0], b[j, 0]], [b[i, 2], b[j, 2]], color='tab:green', lw=0.8,
                label='loop edges' if n == 0 else None)
    ax.set_aspect('equal'); ax.set_xlabel('x [m] (camera right at start)'); ax.set_ylabel('z [m] (camera forward at start)')
    ax.set_title('Top-view trajectory (scale and world frame are arbitrary)')
    ax.legend()
    for ext in ('png', 'pdf'):
        fig.savefig('%s.%s' % (path_stem, ext), dpi=130, bbox_inches='tight')
    plt.close(fig)


def save_analysis(run, odom, optimized, records, kf_indices, loops, loop_rows):
    """成功 full 後的四張圖與 manifest；只讀結果，不改任何 pose。"""
    plt = _plt()
    out = Path(run) / 'analysis'
    out.mkdir(exist_ok=True)
    save_trajectory(str(out / '01_trajectory'), odom, optimized, kf_indices, loops, kf_indices)
    idx = np.array([r['idx'] for r in records])

    def col(name):
        return np.array([float(r[name]) if r[name] != '' else np.nan for r in records])

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(idx, col('step_translation_m'), lw=0.8); ax[0].set_ylabel('translation per frame [m]')
    ax[1].plot(idx, col('step_rotation_deg'), lw=0.8, color='tab:red'); ax[1].set_ylabel('rotation per frame [deg]')
    bad = np.array([r['tracking_status'] == 'tracking_failure' for r in records])
    for a in ax:
        for i in idx[bad]:
            a.axvline(i, color='k', alpha=0.25, lw=0.6)
        a.grid(alpha=0.3)
    ax[1].set_xlabel('frame idx (vertical lines: tracking failure)')
    for ext in ('png', 'pdf'):
        fig.savefig(str(out / ('02_pose_changes.' + ext)), dpi=130, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(idx, col('registration_ms'), lw=0.8, label='registration (GPU)')
    ax[0].plot(idx, col('feature_ms'), lw=0.8, label='feature extraction')
    ax[0].plot(idx, col('matching_ms'), lw=0.8, label='matching (GPU)')
    ax[0].plot(idx, col('total_ms'), lw=0.8, color='k', alpha=0.5, label='total')
    ax[0].set_ylabel('ms'); ax[0].legend(ncol=4, fontsize=8)
    ax[1].plot(idx, col('fitness'), lw=0.8); ax[1].set_ylabel('fitness')
    ax[2].plot(idx, col('inlier_rmse'), lw=0.8, color='tab:red'); ax[2].set_ylabel('inlier RMSE [m]')
    ax[2].set_xlabel('frame idx')
    for a in ax:
        a.grid(alpha=0.3)
    for ext in ('png', 'pdf'):
        fig.savefig(str(out / ('03_registration_timing.' + ext)), dpi=130, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    if loop_rows:
        q = np.array([r['query_kf'] for r in loop_rows]); m = np.array([r['match_kf'] for r in loop_rows])
        acc = np.array([r['accepted'] == 1 for r in loop_rows])
        ax.scatter(q[~acc], m[~acc], s=10, c='tab:red', label='rejected candidates')
        ax.scatter(q[acc], m[acc], s=14, c='tab:green', label='accepted loops')
        ax.legend()
    ax.set_xlabel('query keyframe'); ax.set_ylabel('matched keyframe'); ax.grid(alpha=0.3)
    ax.set_title('Loop closure candidates (%d accepted / %d verified)' % (len(loops), len(loop_rows)))
    for ext in ('png', 'pdf'):
        fig.savefig(str(out / ('04_loop_closures.' + ext)), dpi=130, bbox_inches='tight')
    plt.close(fig)

    sources = {}
    for name in ('poses_odometry.csv', 'poses_optimized.csv', 'frames.csv', 'timings.csv', 'keyframes.csv',
                 'loop_closures.csv'):
        if (Path(run) / name).exists():
            sources[name] = sha256(Path(run) / name)
    (out / 'manifest.json').write_text(json.dumps(dict(sources_sha256=sources), indent=2) + '\n', encoding='utf-8')
    (out / 'README.md').write_text(
        '# 圖表說明\n\n'
        '- `01_trajectory`：俯視軌跡（x–z 平面）。灰＝純 odometry，藍＝pose graph 最佳化後，橘點＝keyframe，綠線＝閉環邊。\n'
        '- `02_pose_changes`：逐幀位移與轉角；直線標出追蹤失敗的幀。\n'
        '- `03_registration_timing`：各階段耗時、配準 fitness 與 inlier RMSE。\n'
        '- `04_loop_closures`：閉環候選（紅＝驗證未過，綠＝接受）。\n\n'
        '圖表只讀取本 run 的輸出（SHA 見 manifest.json），不修改 pose。world frame 與尺度任意；沒有 ground truth，'
        '這些圖不能當作精度證明。\n', encoding='utf-8')
