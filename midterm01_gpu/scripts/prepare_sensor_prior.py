#!/usr/bin/env python3
"""離線同步允許使用的 GNSS／IMU 輸入；不讀取定位真值或 Kaggle 標籤。"""
from output_layout import saved_path
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


def interpolate_positions(times, positions, queries, max_edge=0.3):
    """以相對秒數計算；端點僅容許短時間等速外推，不靜默延用過期資料。"""
    if len(times) < 2 or np.any(np.diff(times) <= 0) or np.max(np.diff(times)) > 0.6:
        raise ValueError('GNSS timestamps must increase with gaps <= 0.6 seconds')
    if queries[0] < times[0] - max_edge or queries[-1] > times[-1] + max_edge:
        raise ValueError('GNSS does not cover the requested frames')
    out = np.column_stack([np.interp(queries, times, positions[:, axis]) for axis in range(3)])
    for mask, a, b in [(queries < times[0], 0, 1), (queries > times[-1], -2, -1)]:
        out[mask] = positions[a] + (queries[mask, None] - times[a]) * ((positions[b] - positions[a]) / (times[b] - times[a]))
    return out


def make_priors(gnss_ns, positions, imu_ns, quaternions, frame_ns,
                fusion='fixed', fusion_config=None, details=None, gnss_quaternions=None):
    if fusion not in ('fixed', 'heading_rts', 'motion_rts', 'pose_heading_rts', 'independent_heading_rts', 'pose_yaw', 'turn_heading_rts'):
        raise ValueError('Unknown sensor fusion mode')
    for name, stamps, minimum in [('frame', frame_ns, 1), ('GNSS', gnss_ns, 2), ('IMU', imu_ns, 2)]:
        if (len(stamps) < minimum or any(not isinstance(s, (int, np.integer)) or s <= 0 for s in stamps)
                or any(b <= a for a, b in zip(stamps, stamps[1:]))):
            raise ValueError(f'{name} timestamps must be positive increasing integer nanoseconds')
    if np.asarray(positions).shape != (len(gnss_ns), 3) or np.asarray(quaternions).shape != (len(imu_ns), 4):
        raise ValueError('Sensor timestamp/value shape mismatch')
    # Integer subtraction precedes conversion to float to preserve nanosecond ordering.
    origin = int(frame_ns[0])
    gt = (np.asarray(gnss_ns, dtype=np.int64) - origin) / 1e9
    it = (np.asarray(imu_ns, dtype=np.int64) - origin) / 1e9
    ft = (np.asarray(frame_ns, dtype=np.int64) - origin) / 1e9
    positions, quaternions = np.asarray(positions), np.asarray(quaternions)
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(quaternions)):
        raise ValueError('Nonfinite sensor measurements')
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(np.abs(norms - 1) > 0.02):
        raise ValueError('Invalid IMU quaternion')
    if np.any(np.diff(it) <= 0) or np.max(np.diff(it)) > .05:
        raise ValueError('IMU timestamps must increase with gaps <= 0.05 seconds')
    if ft[0] < it[0] - .03 or ft[-1] > it[-1] + .03:
        raise ValueError('IMU does not cover the requested frames')
    smooth = savgol_filter(positions, 5, 2, axis=0, mode='interp') if len(positions) >= 5 else positions.copy()
    xyz = interpolate_positions(gt, smooth, ft)
    rotations = Rotation.from_quat(quaternions)
    rpy = rotations.as_euler('xyz')
    # Estimate a fixed heading bias using only moving, almost straight GNSS segments.
    # This is a heading observation from sensor input, not an extrinsic or label fit.
    velocity = np.gradient(smooth[:, :2], gt, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    imu_yaw = np.interp(gt, it, np.unwrap(rpy[:, 2]))
    yaw_rate = np.gradient(imu_yaw, gt)
    course = np.arctan2(velocity[:, 1], velocity[:, 0])
    offsets = np.arctan2(np.sin(course - imu_yaw), np.cos(course - imu_yaw))
    mask = (speed > 5) & (np.abs(yaw_rate) < np.deg2rad(2)) & (np.abs(offsets) < np.deg2rad(15))
    bias = float(np.median(offsets[mask])) if np.count_nonzero(mask) >= 20 else 0.0
    spread = float(np.median(np.abs(offsets[mask] - bias))) if np.any(mask) else None
    if spread is not None and spread > np.deg2rad(4):
        bias = 0.0
    interpolated = Slerp(it, rotations)(np.clip(ft, it[0], it[-1]))
    orientations = Rotation.from_euler('z', bias) * interpolated
    meta = {'time_basis': 'integer bag timestamp; relative seconds for interpolation',
            'offline': True, 'gnss_position_smoothing': '5-sample quadratic Savitzky-Golay',
            'gnss_interpolation': 'linear; endpoint constant-velocity extrapolation <= 0.3 s',
            'imu_interpolation': 'quaternion SLERP; endpoint clamp <= 0.03 s',
            'heading_bias_deg': float(np.degrees(bias)), 'heading_calibration_samples': int(np.count_nonzero(mask)),
            'heading_offset_mad_deg': None if spread is None else float(np.degrees(spread)),
            'gnss_count': len(gt), 'imu_count': len(it), 'frames': len(ft),
            'gnss_max_gap_s': float(np.max(np.diff(gt))), 'imu_max_gap_s': float(np.max(np.diff(it))),
            'gnss_is_input_not_ground_truth': True,
            'frame_assumption': 'GNSS map coordinates and IMU forward/up axes compatible with supplied source pose; no extra body-camera extrinsic applied'}
    quat = orientations.as_quat()
    if fusion != 'fixed':
        from sensor_fusion import FusionConfig, fuse_priors
        xyz, quat, fusion_meta, diagnostic, blocks = fuse_priors(
            gnss_ns, positions, imu_ns, quaternions, frame_ns, xyz, meta,
            fusion, fusion_config or FusionConfig(), gnss_quaternions)
        meta['sensor_fusion'] = fusion_meta
        meta['heading_bias_deg_role'] = 'fixed reference only; actual per-frame correction is in sensor_prior.fusion.csv bias_deg'
        if fusion in ('motion_rts', 'pose_heading_rts', 'independent_heading_rts', 'pose_yaw', 'turn_heading_rts'):
            meta['gnss_position_smoothing'] = 'raw GNSS constant-velocity Kalman + RTS'
            meta['gnss_interpolation'] = 'prediction-only at frame timestamps, each raw GNSS updates once'
        if details is not None:
            details.update(diagnostic=diagnostic, blocks=blocks)
    return xyz, quat, meta


def load_bag(path, include_gnss_orientation=False):
    from rclpy.serialization import deserialize_message
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Imu
    connection = sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True)
    try:
        topics = {name: (ident, kind) for ident, name, kind in connection.execute('SELECT id,name,type FROM topics')}
        series = []
        gnss_orientation = []
        for topic, cls, expected in [('/localizer_node/gnss_pose', PoseStamped, 'geometry_msgs/msg/PoseStamped'),
                                     ('/imu/data', Imu, 'sensor_msgs/msg/Imu')]:
            ident, kind = topics[topic]
            if kind != expected:
                raise ValueError(f'Wrong sensor type: {topic}: {kind}')
            rows = []
            for stamp, packed in connection.execute('SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp', (ident,)):
                message = deserialize_message(packed, cls)
                if cls is PoseStamped:
                    p = message.pose.position
                    rows.append((stamp, [p.x, p.y, p.z]))
                    if include_gnss_orientation:
                        q = message.pose.orientation
                        gnss_orientation.append([q.x, q.y, q.z, q.w])
                else:
                    q = message.orientation
                    if message.orientation_covariance[0] < 0:
                        raise ValueError('IMU reports unavailable orientation')
                    rows.append((stamp, [q.x, q.y, q.z, q.w]))
            series.append(rows)
        return (*series, gnss_orientation) if include_gnss_orientation else series
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=saved_path, required=True)
    parser.add_argument('--frames', type=saved_path, required=True)
    parser.add_argument('--output', type=saved_path, required=True)
    parser.add_argument('--fusion', choices=('fixed', 'heading_rts', 'motion_rts', 'pose_heading_rts', 'independent_heading_rts', 'pose_yaw', 'turn_heading_rts'), default='fixed')
    parser.add_argument('--pose-heading-std-deg', type=float, default=3., help='experimental GNSS orientation block std floor, degrees; uncalibrated')
    parser.add_argument('--gnss-std', type=float, default=.3, help='fusion model GNSS std in m, not calibrated covariance')
    parser.add_argument('--xy-accel-psd', type=float, default=4., help='motion_rts white acceleration PSD, m^2/s^3')
    parser.add_argument('--bias-rw-deg', type=float, default=.12, help='heading bias random walk std, deg/sqrt(s)')
    parser.add_argument('--course-window', type=float, default=2., help='nonoverlapping GNSS course window, s')
    args = parser.parse_args()
    from sensor_fusion import FusionConfig
    config = FusionConfig(gnss_std_m=args.gnss_std, xy_accel_psd=args.xy_accel_psd,
                          bias_rw_deg_sqrt_s=args.bias_rw_deg, course_window_s=args.course_window,
                          pose_heading_std_deg=args.pose_heading_std_deg)
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    with args.frames.open() as stream:
        frames = list(csv.DictReader(stream))
    if [int(row['idx']) for row in frames] != list(range(len(frames))):
        parser.error('Frame manifest idx must be contiguous from zero')
    frame_ns = [int(row['timestamp_ns']) for row in frames]
    experimental_heading = args.fusion in ('pose_heading_rts', 'independent_heading_rts', 'pose_yaw')
    loaded = load_bag(args.bag, include_gnss_orientation=experimental_heading)
    gnss, imu = loaded[:2]
    gnss_quaternions = loaded[2] if experimental_heading else None
    details = {}
    xyz, quat, meta = make_priors([r[0] for r in gnss], [r[1] for r in gnss],
                                  [r[0] for r in imu], [r[1] for r in imu], frame_ns,
                                  args.fusion, config, details, gnss_quaternions)
    with args.output.open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['timestamp_ns', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        writer.writerows((ns, *p, *q) for ns, p, q in zip(frame_ns, xyz, quat))
    if details:
        with args.output.with_suffix('.fusion.csv').open('x', newline='') as stream:
            writer = csv.writer(stream)
            columns = list(details['diagnostic'])
            writer.writerow(columns)
            writer.writerows(zip(*(details['diagnostic'][name] for name in columns)))
        with args.output.with_suffix('.heading_blocks.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(details['blocks'][0]))
            writer.writeheader()
            writer.writerows(details['blocks'])
    meta.update({'bag': str(args.bag), 'frames_manifest': str(args.frames)})
    sensor_digest = hashlib.sha256()
    for series in (gnss, imu):
        sensor_digest.update(np.asarray([row[0] for row in series], dtype='<i8').tobytes())
        sensor_digest.update(np.asarray([row[1] for row in series], dtype='<f8').tobytes())
    if experimental_heading:
        sensor_digest.update(np.asarray(gnss_quaternions, dtype='<f8').tobytes())
        meta['sensor_hash_includes_gnss_orientation'] = True
    meta.update({'bag_bytes': args.bag.stat().st_size,
                 'sensor_samples_sha256': sensor_digest.hexdigest(),
                 'frames_manifest_sha256': hashlib.sha256(args.frames.read_bytes()).hexdigest()})
    args.output.with_suffix('.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == '__main__':
    main()
