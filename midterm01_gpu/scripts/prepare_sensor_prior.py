#!/usr/bin/env python3
"""離線同步允許使用的 GNSS／IMU 輸入；不讀取定位真值或 Kaggle 標籤。"""
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


def make_priors(gnss_ns, positions, imu_ns, quaternions, frame_ns):
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
    return xyz, orientations.as_quat(), meta


def load_bag(path):
    from rclpy.serialization import deserialize_message
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Imu
    connection = sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True)
    try:
        topics = {name: (ident, kind) for ident, name, kind in connection.execute('SELECT id,name,type FROM topics')}
        series = []
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
                else:
                    q = message.orientation
                    if message.orientation_covariance[0] < 0:
                        raise ValueError('IMU reports unavailable orientation')
                    rows.append((stamp, [q.x, q.y, q.z, q.w]))
            series.append(rows)
        return series
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.frames.open() as stream:
        frame_ns = [int(row['timestamp_ns']) for row in csv.DictReader(stream)]
    gnss, imu = load_bag(args.bag)
    xyz, quat, meta = make_priors([r[0] for r in gnss], [r[1] for r in gnss],
                                  [r[0] for r in imu], [r[1] for r in imu], frame_ns)
    with args.output.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['timestamp_ns', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        writer.writerows((ns, *p, *q) for ns, p, q in zip(frame_ns, xyz, quat))
    meta.update({'bag': str(args.bag), 'frames_manifest': str(args.frames)})
    sensor_digest = hashlib.sha256()
    for series in (gnss, imu):
        sensor_digest.update(np.asarray([row[0] for row in series], dtype='<i8').tobytes())
        sensor_digest.update(np.asarray([row[1] for row in series], dtype='<f8').tobytes())
    meta.update({'bag_bytes': args.bag.stat().st_size,
                 'sensor_samples_sha256': sensor_digest.hexdigest(),
                 'frames_manifest_sha256': hashlib.sha256(args.frames.read_bytes()).hexdigest()})
    args.output.with_suffix('.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == '__main__':
    main()
