"""原始 GNSS／IMU 的離線融合；沒有讀取 pose 結果、地圖或評分標籤。"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


@dataclass(frozen=True)
class FusionConfig:
    gnss_std_m: float = .3
    xy_accel_psd: float = 4.0  # white acceleration PSD, m^2/s^3
    bias_rw_deg_sqrt_s: float = .12
    course_window_s: float = 2.0
    min_course_speed_mps: float = 5.0
    max_turn_rate_degps: float = 2.0
    max_heading_span_deg: float = 4.0
    course_std_floor_deg: float = .5
    max_observation_deg: float = 15.0
    max_bias_deg: float = 10.0
    max_bias_rate_degps: float = .2
    robust_gate_sigma: float = 3.0
    min_heading_blocks: int = 3

    def validate(self):
        if not all(np.isfinite(value) and value > 0 for value in asdict(self).values()):
            raise ValueError('Fusion configuration must contain finite positive values')
        if self.course_window_s < 1.0 or self.min_heading_blocks < 3:
            raise ValueError('Course window must be >= 1 s and at least 3 blocks are required')


def heading_observations(gt, xy, it, imu_yaw, config):
    """以互不重疊的 GNSS 時間區塊估 course，避免將 28 Hz 內插值當獨立觀測。"""
    rows = []
    bins = np.floor((gt - gt[0]) / config.course_window_s).astype(int)
    for block in np.unique(bins):
        indices = np.flatnonzero(bins == block)
        t = gt[indices]
        row = {'time_s': float(np.mean(t)), 'samples': int(len(t)), 'accepted': False,
               'reason': '', 'speed_mps': None, 'offset_deg': None, 'std_deg': None}
        if len(t) < 5 or t[-1] - t[0] < 1.0 or t[0] < it[0] or t[-1] > it[-1]:
            row['reason'] = 'insufficient_samples_or_imu_coverage'
            rows.append(row)
            continue
        tau = t - np.mean(t)
        design = np.column_stack((np.ones(len(t)), tau))
        points = xy[indices] - xy[indices[0]]
        weights = np.ones(len(t))
        for _ in range(4):
            fit = np.linalg.lstsq(design * np.sqrt(weights[:, None]), points * np.sqrt(weights[:, None]), rcond=None)[0]
            residual = points - design @ fit
            weights = np.minimum(1., config.robust_gate_sigma * config.gnss_std_m /
                                 np.maximum(np.linalg.norm(residual, axis=1), 1e-12))
        speed = float(np.linalg.norm(fit[1]))
        heading = np.interp(t, it, imu_yaw)
        turn_rate = np.gradient(heading, t)
        offset = float(wrap(np.arctan2(fit[1, 1], fit[1, 0]) - np.mean(heading)))
        row.update(speed_mps=speed, offset_deg=float(np.rad2deg(offset)))
        if speed <= config.min_course_speed_mps:
            row['reason'] = 'low_speed'
        elif (np.max(np.abs(turn_rate)) >= np.deg2rad(config.max_turn_rate_degps)
              or np.ptp(heading) >= np.deg2rad(config.max_heading_span_deg)):
            row['reason'] = 'turning'
        elif abs(offset) >= np.deg2rad(config.max_observation_deg):
            row['reason'] = 'heading_disagreement_or_reverse'
        elif np.count_nonzero(weights < .5) > 1:
            row['reason'] = 'inconsistent_gnss_block'
        else:
            normal = np.array([-fit[1, 1], fit[1, 0]]) / speed
            perpendicular = residual @ normal
            scatter = max(config.gnss_std_m, float(np.sqrt(np.average(perpendicular**2, weights=weights))))
            # Effective slope information, with residual scatter and a noise floor.
            sxx = float(np.sum(weights * (tau - np.average(tau, weights=weights))**2))
            std = max(np.deg2rad(config.course_std_floor_deg), scatter / (np.sqrt(sxx) * speed))
            row.update(accepted=True, reason='accepted', std_deg=float(np.rad2deg(std)))
        rows.append(row)
    return rows


def bias_rts(times, observations, std, queries, config):
    """慢變 bias random-walk KF + RTS。先作逐筆 innovation 降權，再雙向平滑。"""
    process = np.deg2rad(config.bias_rw_deg_sqrt_s)**2
    count = len(times)
    mean, variance = np.empty(count), np.empty(count)
    prediction, prediction_variance = np.empty(count), np.empty(count)
    inflated = np.ones(count)
    current, covariance = 0., np.deg2rad(10.)**2  # diffuse prior; no reused global median measurement
    for i in range(count):
        dt = times[i] - times[i - 1] if i else 0.
        covariance += process * dt
        prediction[i], prediction_variance[i] = current, covariance
        innovation = float(wrap(observations[i] - current))
        scale = abs(innovation) / np.sqrt(covariance + std[i]**2)
        inflated[i] = max(1., (scale / config.robust_gate_sigma)**2)
        measurement_variance = std[i]**2 * inflated[i]
        gain = covariance / (covariance + measurement_variance)
        current += gain * innovation
        covariance = (1 - gain)**2 * covariance + gain**2 * measurement_variance
        mean[i], variance[i] = current, covariance
    smooth, smooth_var = mean.copy(), variance.copy()
    for i in range(count - 2, -1, -1):
        gain = variance[i] / prediction_variance[i + 1]
        smooth[i] += gain * (smooth[i + 1] - prediction[i + 1])
        smooth_var[i] += gain**2 * (smooth_var[i + 1] - prediction_variance[i + 1])
    bounded = np.clip(smooth, -np.deg2rad(config.max_bias_deg), np.deg2rad(config.max_bias_deg))
    # 安全限制只約束校正，不改 IMU 量到的真實轉向；輸出 covariance 僅作模型診斷。
    rate = np.deg2rad(config.max_bias_rate_degps)
    for indices, neighbor_offset in ((range(1, count), -1), (range(count - 2, -1, -1), 1)):
        for i in indices:
            neighbor = i + neighbor_offset
            allowance = rate * abs(times[i] - times[neighbor])
            bounded[i] = np.clip(bounded[i], bounded[neighbor] - allowance, bounded[neighbor] + allowance)
    result = np.interp(queries, times, bounded)
    # Between observations, include Brownian-bridge process uncertainty. Outside,
    # carry the boundary mean and grow uncertainty instead of extrapolating drift.
    var = np.interp(queries, times, np.maximum(smooth_var, 0.))
    right = np.searchsorted(times, queries, side='right')
    interior = (right > 0) & (right < count)
    left_dt = queries[interior] - times[right[interior] - 1]
    right_dt = times[right[interior]] - queries[interior]
    var[interior] += process * left_dt * right_dt / (left_dt + right_dt)
    var += process * np.maximum(times[0] - queries, 0.) + process * np.maximum(queries - times[-1], 0.)
    return result, np.sqrt(var), {'innovation_downweighted_blocks': int(np.count_nonzero(inflated > 1)),
                                 'bounded_blocks': int(np.count_nonzero(np.abs(smooth - bounded) > 1e-12))}


def position_rts(gt, xy, queries, config):
    """原始 GNSS 每筆只更新一次的 [x,y,vx,vy] KF/RTS；frame 時刻只有 prediction。"""
    grid = np.unique(np.concatenate((gt, queries)))
    measurement_at = np.searchsorted(grid, gt)
    query_at = np.searchsorted(grid, queries)
    origin = xy[0].copy()
    points = xy - origin
    n = len(grid)
    means = np.empty((n, 4))
    covariances = np.empty((n, 4, 4))
    predicted = np.empty_like(means)
    predicted_cov = np.empty_like(covariances)
    h = np.column_stack((np.eye(2), np.zeros((2, 2))))
    base_r = np.eye(2) * config.gnss_std_m**2
    state = np.r_[points[0], (points[1] - points[0]) / (gt[1] - gt[0])]
    state[:2] += state[2:] * (grid[0] - gt[0])
    p = np.diag([100., 100., 100., 100.])
    observation = 0
    downweighted = 0
    identity = np.eye(4)
    for i, stamp in enumerate(grid):
        dt = stamp - grid[i - 1] if i else 0.
        f = np.eye(4)
        f[:2, 2:] = np.eye(2) * dt
        q = config.xy_accel_psd * np.block([[np.eye(2) * dt**3 / 3, np.eye(2) * dt**2 / 2],
                                           [np.eye(2) * dt**2 / 2, np.eye(2) * dt]])
        state, p = f @ state, f @ p @ f.T + q
        predicted[i], predicted_cov[i] = state, p
        if observation < len(gt) and measurement_at[observation] == i:
            innovation = points[observation] - h @ state
            s = h @ p @ h.T + base_r
            mahalanobis = float(innovation @ np.linalg.solve(s, innovation))
            inflation = max(1., mahalanobis / config.robust_gate_sigma**2)
            downweighted += inflation > 1
            r = base_r * inflation
            gain = np.linalg.solve(h @ p @ h.T + r, h @ p).T
            state = state + gain @ innovation
            a = identity - gain @ h
            p = a @ p @ a.T + gain @ r @ gain.T
            observation += 1
        means[i], covariances[i] = state, (p + p.T) / 2
    smooth, smooth_cov = means.copy(), covariances.copy()
    for i in range(n - 2, -1, -1):
        dt = grid[i + 1] - grid[i]
        f = np.eye(4)
        f[:2, 2:] = np.eye(2) * dt
        gain = np.linalg.solve(predicted_cov[i + 1], f @ covariances[i]).T
        smooth[i] += gain @ (smooth[i + 1] - predicted[i + 1])
        smooth_cov[i] += gain @ (smooth_cov[i + 1] - predicted_cov[i + 1]) @ gain.T
    return (smooth[query_at, :2] + origin, smooth[query_at, 2:],
            np.sqrt(np.maximum(np.diagonal(smooth_cov[query_at, :2, :2], axis1=1, axis2=2), 0.)),
            {'position_measurement_updates': int(observation), 'position_downweighted_updates': int(downweighted)})


def fuse_priors(gnss_ns, positions, imu_ns, quaternions, frame_ns, baseline_xyz,
                baseline_meta, mode, config):
    config.validate()
    if mode not in ('heading_rts', 'motion_rts'):
        raise ValueError('Unknown fusion mode')
    # GNSS origin makes block boundaries independent of requested image count.
    origin = int(gnss_ns[0])
    gt = (np.asarray(gnss_ns, dtype=np.int64) - origin) / 1e9
    it = (np.asarray(imu_ns, dtype=np.int64) - origin) / 1e9
    ft = (np.asarray(frame_ns, dtype=np.int64) - origin) / 1e9
    positions = np.asarray(positions, dtype=float)
    rotations = Rotation.from_quat(quaternions)
    imu_yaw = np.unwrap(rotations.as_euler('xyz')[:, 2])
    rows = heading_observations(gt, positions[:, :2], it, imu_yaw, config)
    selected = [row for row in rows if row['accepted']]
    fixed = np.deg2rad(baseline_meta['heading_bias_deg'])
    notes = {}
    if len(selected) >= config.min_heading_blocks:
        bias, uncertainty, notes = bias_rts(np.array([row['time_s'] for row in selected]),
                np.deg2rad([row['offset_deg'] for row in selected]),
                np.deg2rad([row['std_deg'] for row in selected]), ft, config)
        fallback = None
    else:
        bias = np.full(len(ft), fixed)
        uncertainty = np.full(len(ft), np.deg2rad(10.))
        fallback = 'insufficient reliable heading blocks; explicitly retained fixed baseline bias'
    orientation = Rotation.from_euler('z', bias) * Slerp(it, rotations)(np.clip(ft, it[0], it[-1]))
    xyz = baseline_xyz.copy()
    diagnostic = {'timestamp_ns': np.asarray(frame_ns, dtype=np.int64),
                  'bias_deg': np.rad2deg(bias), 'bias_model_std_deg': np.rad2deg(uncertainty),
                  'fixed_bias_deg': np.full(len(ft), np.rad2deg(fixed)),
                  'xy_change_from_fixed_m': np.zeros(len(ft))}
    if mode == 'motion_rts':
        xy, velocity, std, position_meta = position_rts(gt, positions[:, :2], ft, config)
        xyz[:, :2] = xy
        diagnostic.update(vx_mps=velocity[:, 0], vy_mps=velocity[:, 1],
                          x_model_std_m=std[:, 0], y_model_std_m=std[:, 1],
                          xy_change_from_fixed_m=np.linalg.norm(xy - baseline_xyz[:, :2], axis=1))
        notes.update(position_meta)
    meta = {'algorithm_version': 1, 'mode': mode, 'config': asdict(config),
            'heading_observation_blocks': len(rows), 'accepted_heading_blocks': len(selected),
            'heading_observation_range_deg': [min(row['offset_deg'] for row in selected), max(row['offset_deg'] for row in selected)] if selected else None,
            'heading_support_seconds_from_gnss_start': [selected[0]['time_s'], selected[-1]['time_s']] if selected else None,
            'heading_fallback': fallback, 'bias_range_deg': [float(np.min(diagnostic['bias_deg'])), float(np.max(diagnostic['bias_deg']))],
            'max_xy_change_from_fixed_m': float(np.max(diagnostic['xy_change_from_fixed_m'])),
            'position_method': 'raw GNSS constant-velocity Kalman + RTS' if mode == 'motion_rts' else 'unchanged fixed-mode GNSS Savitzky-Golay + linear interpolation',
            'correlation': 'course is derived from GNSS; no independent XY-heading covariance claim; no map posterior reused as GNSS observation',
            'uncertainty': 'model diagnostics only; not calibrated accuracy or passed as independent map covariance',
            'imu_usage': 'quaternion orientation only; no raw acceleration/gyro integration', **notes}
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(orientation.as_quat())):
        raise ValueError('Nonfinite fused prior')
    return xyz, orientation.as_quat(), meta, diagnostic, rows
