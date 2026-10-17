#!/usr/bin/env python3
"""唯讀既有 sensor_guided run，另存離線濾波候選；不重跑定位或推估 Kaggle 分數。"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_run(run, track):
    validation = json.loads((run / 'validation.json').read_text())
    if (validation.get('track') != track or validation.get('profile') != 'sensor_guided'
            or validation.get('returncode') != 0 or validation.get('validation_errors')
            or validation.get('output_complete_and_finite') is not True):
        raise ValueError('需要 Track 相符且驗證通過的 sensor_guided run')
    frames = read_rows(run / 'frames.csv')
    stamps = np.array([int(row['timestamp_ns']) for row in frames], dtype=np.int64)
    if (len(stamps) < 7 or np.any(stamps <= 0) or np.any(np.diff(stamps) <= 0)
            or [int(row['idx']) for row in frames] != list(range(len(frames)))):
        raise ValueError('frames 必須有至少 7 筆、連續 idx 與遞增的整數 timestamp')
    values = np.loadtxt(run / 'result.txt', ndmin=2)
    if values.shape != (len(frames), 12) or not np.all(np.isfinite(values)):
        raise ValueError('result.txt 必須逐幀對應 12 個有限數值')
    if validation.get('frames') != len(frames) or validation.get('expected_frames') != len(frames):
        raise ValueError('validation 與實際筆數不符')
    poses = values.reshape(-1, 3, 4)
    rotation = poses[:, :, :3]
    if (np.max(np.abs(rotation.transpose(0, 2, 1) @ rotation - np.eye(3))) > 1e-4
            or np.max(np.abs(np.linalg.det(rotation) - 1)) > 1e-4):
        raise ValueError('輸入旋轉不是合法 SO(3) 矩陣')
    t = (stamps - stamps[0]) / 1e9
    if np.max(np.diff(t)) > .2:
        raise ValueError('幀缺口超過 0.2 秒，應先分段處理')
    yaw = np.unwrap(np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]))
    return frames, t, poses, np.column_stack((poses[:, :2, 3], yaw))


def local_polynomial(t, values, window_seconds):
    """時間域的二次局部多項式：SG 類平滑，直接處理非等間距，不用幀號代替時間。"""
    if not np.isfinite(window_seconds) or window_seconds <= 0 or window_seconds > t[-1] - t[0]:
        raise ValueError('多項式視窗須為正數且不超過軌跡時間範圍')
    output = np.empty_like(values, dtype=float)
    for i, stamp in enumerate(t):
        # 端點將完整時間窗移入資料範圍，以多項式內插處理，不填零。
        start = np.clip(stamp - window_seconds / 2, t[0], t[-1] - window_seconds)
        left, right = np.searchsorted(t, [start, start + window_seconds], side='left')
        if right < len(t) and t[right] <= start + window_seconds + 1e-12:
            right += 1
        if right - left < 3:
            raise ValueError('視窗內不足 3 筆資料，請加大 --window-seconds')
        tau = (t[left:right] - stamp) / window_seconds
        design = np.column_stack((np.ones(len(tau)), tau, tau * tau))
        # 先移除當下值，避免 map 大座標影響最小平方精度。
        coefficients = np.linalg.lstsq(design, values[left:right] - values[i], rcond=None)[0]
        output[i] = values[i] + coefficients[0]
    return output


def ema_control(values, alpha):
    """因果 IIR 對照，刻意保留延遲以量化直接濾最終 pose 的影響。"""
    output = values.copy()
    for i in range(1, len(values)):
        output[i] = (1 - alpha) * output[i - 1] + alpha * values[i]
    return output


def rts_smooth(t, values, observation_std, acceleration_psd, velocity_std):
    """逐軸 [位置,速度] Kalman + RTS；yaw 軸是 [展開角度,角速度]，非 IMU 融合。"""
    output = np.empty_like(values)
    for axis in range(values.shape[1]):
        z = values[:, axis] - values[0, axis]
        variance = observation_std[axis] ** 2
        q = acceleration_psd[axis]
        means = np.empty((len(t), 2))
        covariances = np.empty((len(t), 2, 2))
        predicted_means = np.empty_like(means)
        predicted_covariances = np.empty_like(covariances)
        means[0] = [z[0], (z[1] - z[0]) / (t[1] - t[0])]
        covariances[0] = np.diag([variance, velocity_std[axis] ** 2])
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            f = np.array([[1., dt], [0., 1.]])
            process = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
            predicted_means[i] = f @ means[i - 1]
            predicted_covariances[i] = f @ covariances[i - 1] @ f.T + process
            p = predicted_covariances[i]
            gain = p[:, 0] / (p[0, 0] + variance)
            means[i] = predicted_means[i] + gain * (z[i] - predicted_means[i, 0])
            # Joseph form keeps the covariance symmetric and positive semidefinite.
            a = np.eye(2) - np.outer(gain, [1., 0.])
            covariances[i] = a @ p @ a.T + variance * np.outer(gain, gain)
        smoothed = means.copy()
        for i in range(len(t) - 2, -1, -1):
            dt = t[i + 1] - t[i]
            f = np.array([[1., dt], [0., 1.]])
            gain = np.linalg.solve(predicted_covariances[i + 1], f @ covariances[i]).T
            smoothed[i] += gain @ (smoothed[i + 1] - predicted_means[i + 1])
        output[:, axis] = smoothed[:, 0] + values[0, axis]
    return output


def filtered_poses(poses, values):
    result = poses.copy()
    original_yaw = np.arctan2(poses[:, 1, 0], poses[:, 0, 0])
    # 左乘合法 Rz，保留原始 roll/pitch；禁止逐元素平滑旋轉矩陣。
    delta = np.arctan2(np.sin(values[:, 2] - original_yaw), np.cos(values[:, 2] - original_yaw))
    result[:, :, :3] = Rotation.from_euler('z', delta).as_matrix() @ poses[:, :, :3]
    result[:, :2, 3] = values[:, :2]
    return result


def metrics(t, original, filtered):
    xy = np.linalg.norm(filtered[:, :2] - original[:, :2], axis=1)
    yaw = np.rad2deg(np.arctan2(np.sin(filtered[:, 2] - original[:, 2]),
                               np.cos(filtered[:, 2] - original[:, 2])))
    dt = np.diff(t)
    speed = np.linalg.norm(np.diff(filtered[:, :2], axis=0) / dt[:, None], axis=1)
    yaw_rate = np.rad2deg(np.diff(filtered[:, 2]) / dt)
    out = {'xy_change_rms_m': float(np.sqrt(np.mean(xy**2))),
           'xy_change_max_m': float(np.max(xy)),
           'yaw_change_rms_deg': float(np.sqrt(np.mean(yaw**2))),
           'yaw_change_max_deg': float(np.max(np.abs(yaw))),
           'endpoint_xy_change_m': xy[[0, -1]].tolist(),
           'speed_p99_mps': float(np.quantile(speed, .99)),
           'yaw_rate_p99_degps': float(np.quantile(np.abs(yaw_rate), .99))}
    base_speed = np.gradient(original[:, :2], t, axis=0)
    base_rate = np.rad2deg(np.gradient(original[:, 2], t))
    speed_at_frame = np.linalg.norm(base_speed, axis=1)
    masks = {'slow': speed_at_frame < 2,
             'turn': (speed_at_frame >= 2) & (np.abs(base_rate) > 3),
             'moving_straight': (speed_at_frame >= 2) & (np.abs(base_rate) <= 3)}
    out['segments_from_original_estimate_not_truth'] = {
        name: {'frames': int(mask.sum()),
               'xy_change_rms_m': float(np.sqrt(np.mean(xy[mask]**2))) if mask.any() else None,
               'yaw_change_rms_deg': float(np.sqrt(np.mean(yaw[mask]**2))) if mask.any() else None}
        for name, mask in masks.items()}
    return out


def heading_audit(bag, first_stamp):
    # ROS imports stay optional: basic postprocessing needs only NumPy/SciPy.
    from prepare_sensor_prior import load_bag
    from scipy.signal import savgol_filter
    gnss, imu = load_bag(bag)
    gt = (np.array([row[0] for row in gnss], dtype=np.int64) - first_stamp) / 1e9
    it = (np.array([row[0] for row in imu], dtype=np.int64) - first_stamp) / 1e9
    positions = np.array([row[1] for row in gnss])
    smooth = savgol_filter(positions, 5, 2, axis=0, mode='interp')
    velocity = np.gradient(smooth[:, :2], gt, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    imu_yaw = np.interp(gt, it, np.unwrap(Rotation.from_quat([row[1] for row in imu]).as_euler('xyz')[:, 2]))
    course = np.arctan2(velocity[:, 1], velocity[:, 0])
    offsets = np.arctan2(np.sin(course - imu_yaw), np.cos(course - imu_yaw))
    mask = (speed > 5) & (np.abs(np.gradient(imu_yaw, gt)) < np.deg2rad(2)) & (np.abs(offsets) < np.deg2rad(15))
    blocks = []
    for start in np.arange(0, gt[-1], 20):
        selected = mask & (gt >= start) & (gt < start + 20)
        if np.count_nonzero(selected) >= 10:
            median = np.median(offsets[selected])
            blocks.append({'start_s': float(start), 'samples': int(selected.sum()),
                           'course_minus_imu_median_deg': float(np.rad2deg(median)),
                           'mad_deg': float(np.rad2deg(np.median(np.abs(offsets[selected] - median))))})
    digest = hashlib.sha256()
    for series in (gnss, imu):
        digest.update(np.array([r[0] for r in series], dtype='<i8').tobytes())
        digest.update(np.array([r[1] for r in series], dtype='<f8').tobytes())
    return {'bag': str(bag), 'bag_bytes': bag.stat().st_size,
            'sensor_samples_sha256': digest.hexdigest(),
            'gnss_median_dt_s': float(np.median(np.diff(gt))),
            'gnss_samples': len(gt), 'imu_samples': len(it),
            'calibration_selection': 'speed > 5 m/s; |IMU yaw rate| < 2 deg/s; |offset| < 15 deg',
            'blocks': blocks, 'interpretation': 'course is not body yaw or ground truth; variation does not prove IMU bias drift'}


def plot_comparison(output, t, original, candidates, audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), constrained_layout=True)
    for name, values in candidates.items():
        if name in ('baseline', 'ema_lag_control'):
            continue
        axes[0].plot(t, np.linalg.norm(values[:, :2] - original[:, :2], axis=1), label=name, lw=.8)
        axes[2].plot(t, np.rad2deg(values[:, 2] - original[:, 2]), label=name, lw=.8)
    axes[0].set(ylabel='XY change (m)', title='Offline smoothing of existing outputs: changes, not accuracy')
    axes[1].plot(t, np.linalg.norm(candidates['ema_lag_control'][:, :2] - original[:, :2], axis=1), color='#bb4433')
    axes[1].set(ylabel='XY change (m)', title='Causal EMA on final pose: lag control')
    axes[2].set(ylabel='Yaw change (deg)')
    if audit:
        blocks = audit['blocks']
        axes[3].errorbar([b['start_s'] + 10 for b in blocks],
                         [b['course_minus_imu_median_deg'] for b in blocks],
                         yerr=[b['mad_deg'] for b in blocks], marker='o', capsize=3)
        axes[3].set(ylabel='Course - IMU yaw (deg)', title='Straight / moving inputs: 20 s median +/- MAD; not ground truth')
    else:
        axes[3].plot(t, np.rad2deg(original[:, 2]), lw=.8)
        axes[3].set(ylabel='Original unwrapped yaw (deg)')
    for ax in axes:
        ax.set_xlabel('Data time (s)')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--track', type=int, choices=(1, 2), required=True)
    parser.add_argument('--window-seconds', type=float, default=.5)
    parser.add_argument('--ema-alpha', type=float, default=.25)
    parser.add_argument('--rts-xy-std', type=float, default=.1, help='exploratory observation std, m; not calibrated covariance')
    parser.add_argument('--rts-yaw-std-deg', type=float, default=.2)
    parser.add_argument('--rts-xy-accel-psd', type=float, default=4., help='white acceleration PSD, m^2/s^3')
    parser.add_argument('--rts-yaw-accel-psd-deg2', type=float, default=25., help='angular acceleration PSD, deg^2/s^3')
    parser.add_argument('--bag', type=Path, help='optional read-only heading audit; requires sourced ROS environment')
    parser.add_argument('--plot', action='store_true', help='requires Matplotlib')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run.resolve()
    try:
        run.relative_to((root / 'outputs/runs').resolve())
        positive = (args.window_seconds, args.rts_xy_std, args.rts_yaw_std_deg,
                    args.rts_xy_accel_psd, args.rts_yaw_accel_psd_deg2)
        if not all(math.isfinite(x) and x > 0 for x in positive) or not 0 < args.ema_alpha <= 1:
            raise ValueError('濾波參數必須為有限正數，EMA alpha 必須介於 0 與 1')
        frames, t, poses, original = load_run(run, args.track)
        sg = local_polynomial(t, original, args.window_seconds)
        sg_xy, sg_yaw = original.copy(), original.copy()
        sg_xy[:, :2] = sg[:, :2]
        sg_yaw[:, 2] = sg[:, 2]
        rts = rts_smooth(t, original,
                         [args.rts_xy_std, args.rts_xy_std, np.deg2rad(args.rts_yaw_std_deg)],
                         [args.rts_xy_accel_psd] * 2 + [args.rts_yaw_accel_psd_deg2 * (np.pi / 180)**2],
                         [5., 5., np.deg2rad(30)])
        audit = heading_audit(args.bag.resolve(), int(frames[0]['timestamp_ns'])) if args.bag else None
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    candidates = {'baseline': original, 'sg_xy': sg_xy, 'sg_yaw': sg_yaw,
                  'sg_xy_yaw': sg, 'rts_xy_yaw': rts,
                  'ema_lag_control': ema_control(original, args.ema_alpha)}
    output = Path(tempfile.mkdtemp(prefix=f'trajectory_filters_track{args.track}_', dir=root / 'outputs/runs'))
    # docker exec 以 root 執行時，仍讓主機 VS Code 可讀新 run；只改本次產生的目錄。
    output.chmod(0o755)
    summary = {'source_run': str(run), 'track': args.track, 'frames': len(t),
               'complete_track_row_count': len(t) == {1: 3011, 2: 3299}[args.track],
               'scope': 'postprocessing existing poses only; no localization rerun; no ground truth; no Kaggle score',
               'postprocessing_map_objective_revalidated': False,
               'warning': 'Candidates may violate original per-frame map objective / correction bounds; RTS assumes independent output observations although GNSS interpolation is correlated. All settings are exploratory.',
               'duration_s': float(t[-1]), 'frame_dt_quantiles_s': np.quantile(np.diff(t), [0, .5, 1]).tolist(),
               'parameters': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
               'software': {'numpy': np.__version__, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
               'source_sha256': {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                                 for name in ('frames.csv', 'result.txt', 'validation.json', 'guidance.csv', 'sensor_prior.csv', 'sensor_prior.json')},
               'heading_audit': audit, 'methods': {}}
    for name, values in candidates.items():
        matrix = poses.copy() if name == 'baseline' else filtered_poses(poses, values)
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f'Nonfinite candidate: {name}')
        csv_path = output / (name + '.csv')
        with csv_path.open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['idx', 'x', 'y', 'yaw'])
            for i, pose in enumerate(matrix):
                writer.writerow([i, format(pose[0, 3], '.12g'), format(pose[1, 3], '.12g'),
                                 format(np.arctan2(pose[1, 0], pose[0, 0]), '.12g')])
        np.savetxt(output / (name + '.txt'), matrix.reshape(-1, 12), fmt='%.12g')
        summary['methods'][name] = metrics(t, original, values)
        summary['methods'][name]['csv_sha256'] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    (output / 'frames.csv').write_bytes((run / 'frames.csv').read_bytes())
    (output / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write_json(output / 'report.json', summary)
    if args.plot:
        plot_comparison(output, t, original, candidates, audit)
    print(f'Output: {output}')
    print(json.dumps({name: {key: stats[key] for key in ('xy_change_rms_m', 'yaw_change_rms_deg')}
                      for name, stats in summary['methods'].items()}, indent=2))


if __name__ == '__main__':
    main()
