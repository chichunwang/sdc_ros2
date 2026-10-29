#!/usr/bin/env python3
"""SDC HW4：radar/GPS EKF；GT callback 只存檔，完全不影響濾波。"""
import csv
import bisect
import hashlib
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

from EKF import RadarGpsFilter, covariance, vector
import EKF


def position_errors(rows, ground_truth):
    """唯讀評估：整數 ns 對齊 GT，只在重疊範圍內線性插值，不改估計。"""
    samples = {int(row[0]): np.asarray(row[1:3], dtype=float) for row in rows}
    truth = {int(row[0]): np.asarray(row[1:3], dtype=float) for row in ground_truth}
    times = sorted(truth)
    errors, largest_gap = [], 0
    if times:
        for stamp in sorted(samples):
            if stamp < times[0] or stamp > times[-1]:
                continue
            index = bisect.bisect_left(times, stamp)
            if times[index] == stamp:
                reference = truth[stamp]
            else:
                left, right = times[index-1], times[index]
                largest_gap = max(largest_gap, right-left)
                alpha = (stamp-left)/(right-left)
                reference = truth[left] + alpha*(truth[right]-truth[left])
            error = samples[stamp]-reference
            errors.append((stamp, float(error[0]), float(error[1]), float(np.linalg.norm(error))))
    metrics = dict(samples=len(errors), input_unique_timestamps=len(samples),
                   gt_unique_timestamps=len(times),
                   max_interpolated_gt_gap_s=largest_gap/1e9,
                   duplicate_policy='last arrival at each timestamp',
                   alignment='GT linear interpolation at each sensor timestamp; no extrapolation')
    if errors:
        distances = np.asarray([row[3] for row in errors])
        metrics.update(rmse_xy_m=float(np.sqrt(np.mean(distances**2))),
                       mean_xy_m=float(np.mean(distances)),
                       median_xy_m=float(np.median(distances)),
                       p95_xy_m=float(np.percentile(distances, 95)),
                       max_xy_m=float(np.max(distances)),
                       first_stamp_ns=errors[0][0], last_stamp_ns=errors[-1][0])
    return metrics, errors


def save_analysis(records, result_dir):
    """GT 只在輸出後評估；不同 sensor 的統計使用各自時戳。"""
    report = dict(comparison='Each sensor uses its own timestamps, not paired samples.', sensors={})
    fig, ax = plt.subplots(figsize=(10, 4.5))
    any_errors = False
    all_stamps = [int(row[0]) for row in records['gt']]
    origin = min(all_stamps) if all_stamps else 0
    for sensor in ('radar', 'gps', 'estimate'):
        metrics, errors = position_errors(records[sensor], records['gt'])
        report['sensors'][sensor] = metrics
        with (result_dir / f'{sensor}_position_errors.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['stamp_ns', 'error_x_m', 'error_y_m', 'error_xy_m'])
            writer.writerows(errors)
        if errors:
            any_errors = True
            ax.plot([(row[0]-origin)/1e9 for row in errors], [row[3] for row in errors],
                    label=sensor, linewidth=1, alpha=.8)
    report['status'] = 'available' if any_errors else 'no_overlapping_ground_truth'
    (result_dir / 'analysis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if any_errors:
        ax.set(xlabel='Time from first GT sample [s]', ylabel='Position error [m]',
               title='Position error against time-aligned GT (evaluation only)')
        ax.grid(True, alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(result_dir / '02_position_error.png', dpi=160)
    plt.close(fig)
    return report


def stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def yaw_from_quaternion(q):
    values = vector([q.x, q.y, q.z, q.w], 4)
    norm = np.linalg.norm(values)
    if norm < 1e-12:
        raise ValueError('Invalid zero quaternion')
    x, y, z, w = values / norm
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


class Fusion(Node):
    def __init__(self):
        super().__init__('kf')
        self.declare_parameter('result_dir', '/ros_ws/hw4/result')
        self.declare_parameter('fixed_frame', 'origin')
        # 僅當 bag covariance 全零／未提供時使用；不是實測標定值。
        self.declare_parameter('gps_fallback_std_m', 3.0)
        self.declare_parameter('radar_fallback_std_m', 0.1)
        self.declare_parameter('radar_fallback_std_yaw_rad', math.radians(1.0))
        self.frame = self.get_parameter('fixed_frame').value
        self.filter = RadarGpsFilter()
        self.records = dict(estimate=[], gps=[], radar=[], gt=[])
        self.warning_keys = set()
        self.invalid_messages = 0
        self.fallback_counts = dict(gps=0, radar=0)
        self.covariance_statistics = {}
        self.result_dir = Path(os.path.expanduser(self.get_parameter('result_dir').value))
        self.result_dir.mkdir(parents=True, exist_ok=True)
        # 原生 launch 也不可覆寫歷史結果；wrapper 每輪自建唯一目錄。
        if any((self.result_dir / name).exists() for name in ('source', 'result.png', 'summary.json')):
            raise FileExistsError(f'Result directory already used: {self.result_dir}; '
                                  'use scripts/dev.sh launch to create a new run.')
        source_dir = self.result_dir / 'source'
        source_dir.mkdir()
        sources = {}
        for source in (Path(__file__), Path(EKF.__file__)):
            content = source.read_bytes()
            (source_dir / source.name).write_bytes(content)
            sources[source.name] = dict(sha256=hashlib.sha256(content).hexdigest(),
                                        loaded_from=str(source.resolve()))
        manifest = dict(algorithm='planar_radar_gps_ekf',
                        created_at_utc=datetime.now(timezone.utc).isoformat(),
                        python=sys.version, sources=sources, fixed_frame=self.frame,
                        fallback_std=dict(gps_position_m=self.get_parameter('gps_fallback_std_m').value,
                                          radar_position_m=self.get_parameter('radar_fallback_std_m').value,
                                          radar_yaw_rad=self.get_parameter('radar_fallback_std_yaw_rad').value))
        (self.result_dir / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        self.posePub = self.create_publisher(Odometry, '/pred', 100)
        # 同時相容 reliable 和 best effort 的 bag publisher，保留較大 queue。
        qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(PoseWithCovarianceStamped, '/gps', self.gpsCallback, qos)
        self.create_subscription(Odometry, '/radar_odometry', self.odometryCallback, qos)
        self.create_subscription(Odometry, '/gt_odom', self.gtCallback, qos)
        self.get_logger().info('EKF ready: radar + GPS only; Ctrl+C saves result.png and CSV.')

    def warn_once(self, key, message):
        if key not in self.warning_keys:
            self.warning_keys.add(key)
            self.get_logger().warning(message)

    def check_frame(self, header):
        if header.frame_id != self.frame:
            raise ValueError(f'frame_id={header.frame_id!r}, expected {self.frame!r}; '
                             'verify the common coordinate frame; no implicit transform')

    def noise(self, values, indices, sensor):
        full = np.asarray(values, dtype=float).reshape(6, 6)
        selected = full[np.ix_(indices, indices)]
        raw_selected = selected.copy()
        fallback = bool(np.all(selected == 0))
        if fallback:
            self.fallback_counts[sensor] += 1
            self.warn_once(sensor + '_cov', f'{sensor} covariance is all zero; using '
                           'documented fallback standard deviations. Inspect bag before reporting.')
            if sensor == 'gps':
                std = [float(self.get_parameter('gps_fallback_std_m').value)] * 2
            else:
                std = [float(self.get_parameter('radar_fallback_std_m').value)] * 2
                std.append(float(self.get_parameter('radar_fallback_std_yaw_rad').value))
            if not np.isfinite(std).all() or min(std) <= 0:
                raise ValueError('Fallback standard deviations must be finite and positive')
            selected = np.diag(np.square(std))
        effective = covariance(selected, len(indices))
        if sensor not in self.covariance_statistics:
            self.covariance_statistics[sensor] = dict(
                messages=0, fallback_messages=0, indices=indices,
                first_message_covariance_6x6=full.tolist(),
                first_selected_covariance=raw_selected.tolist(),
                first_effective_pose_covariance=effective.tolist(),
                raw_diagonal_min=np.diag(raw_selected).tolist(),
                raw_diagonal_max=np.diag(raw_selected).tolist())
        stats = self.covariance_statistics[sensor]
        stats['messages'] += 1
        stats['fallback_messages'] += int(fallback)
        stats['raw_diagonal_min'] = np.minimum(stats['raw_diagonal_min'], np.diag(raw_selected)).tolist()
        stats['raw_diagonal_max'] = np.maximum(stats['raw_diagonal_max'], np.diag(raw_selected)).tolist()
        return effective

    def odometryCallback(self, data):
        try:
            self.check_frame(data.header)
            p = data.pose.pose
            pose = vector([p.position.x, p.position.y, yaw_from_quaternion(p.orientation)], 3)
            noise = self.noise(data.pose.covariance, [0, 1, 5], 'radar')
            stamp = stamp_ns(data.header.stamp)
            self.records['radar'].append((stamp, *pose))
            self.publish(self.filter.push_radar(stamp, pose, noise))
        except ValueError as error:
            self.invalid_messages += 1
            self.warn_once('radar_' + str(error), f'Radar message rejected: {error}')

    def gpsCallback(self, data):
        try:
            self.check_frame(data.header)
            p = data.pose.pose.position
            position = vector([p.x, p.y], 2)
            noise = self.noise(data.pose.covariance, [0, 1], 'gps')
            stamp = stamp_ns(data.header.stamp)
            self.records['gps'].append((stamp, *position))
            self.publish(self.filter.push_gps(stamp, position, noise))
        except ValueError as error:
            self.invalid_messages += 1
            self.warn_once('gps_' + str(error), f'GPS message rejected: {error}')

    def gtCallback(self, data):
        # 只記錄評估資料；不得初始化 EKF、校正、估 yaw 或調整 covariance。
        try:
            self.check_frame(data.header)
            p = data.pose.pose.position
            self.records['gt'].append((stamp_ns(data.header.stamp), *vector([p.x, p.y], 2)))
        except ValueError as error:
            self.warn_once('gt_' + str(error), f'GT plot sample rejected: {error}')

    def publish(self, estimates):
        for stamp, pose, noise in estimates:
            msg = Odometry()
            msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
            msg.header.frame_id = self.frame
            msg.child_frame_id = 'ekf_base_link'
            msg.pose.pose.position.x, msg.pose.pose.position.y = map(float, pose[:2])
            msg.pose.pose.orientation.z = math.sin(pose[2]/2)
            msg.pose.pose.orientation.w = math.cos(pose[2]/2)
            full = np.zeros((6, 6))
            full[np.ix_([0, 1, 5], [0, 1, 5])] = noise
            # 未估計的 z/roll/pitch 不宣稱零不確定性。
            full[2, 2] = full[3, 3] = full[4, 4] = 1e6
            msg.pose.covariance = full.ravel().tolist()
            self.posePub.publish(msg)
            self.records['estimate'].append((stamp, *pose))

    def save_results(self):
        result_dir = self.result_dir
        for name, rows in self.records.items():
            with (result_dir / (name + '.csv')).open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['stamp_ns', 'x', 'y'] + (['yaw'] if name in ('estimate', 'radar') else []))
                writer.writerows(rows)
        fig, ax = plt.subplots(figsize=(10, 8))
        styles = dict(gt=('black', 3, 0.35), radar=('tab:green', 1, 0.7),
                      gps=('tab:orange', 0, 0.5), estimate=('tab:blue', 1.5, 1.0))
        for name in ('gt', 'radar', 'gps', 'estimate'):
            if self.records[name]:
                values = np.asarray(self.records[name])
                color, width, alpha = styles[name]
                ax.plot(values[:, 1], values[:, 2], label=name, color=color,
                        linewidth=width, alpha=alpha, marker='.' if name == 'gps' else None)
        ax.set(xlabel='x [m]', ylabel='y [m]', title='Radar / GPS EKF localization')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True)
        if any(self.records.values()):
            ax.legend()
        fig.tight_layout()
        fig.savefig(result_dir / 'result.png', dpi=160)
        plt.close(fig)
        (result_dir / 'covariance_statistics.json').write_text(
            json.dumps(self.covariance_statistics, indent=2), encoding='utf-8')
        analysis = save_analysis(self.records, result_dir)
        stats = dict(self.filter.stats)
        stats.update(invalid_messages=self.invalid_messages,
                     fallback_covariance_messages=self.fallback_counts,
                     pending_gps_after_last_radar=len(self.filter.pending_gps),
                     samples={key: len(rows) for key, rows in self.records.items()},
                     evaluation=analysis,
                     gt_usage='evaluation and plotting only; never a filter input',
                     data_status=('has_sensor_data' if all(self.records.values())
                                  and self.filter.stats['gps_updates'] > 0 else 'incomplete'),
                     recording_note='Bag completion, terminal identity and RViz video require user evidence.')
        (result_dir / 'summary.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
        print(f'\nSaved HW4 results: {result_dir}', flush=True)
        print(json.dumps(stats, indent=2), flush=True)


def main(args=None):
    rclpy.init(args=args)
    fusion = Fusion()
    try:
        rclpy.spin(fusion)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            fusion.save_results()
        finally:
            fusion.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
