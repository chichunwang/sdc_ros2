#!/usr/bin/env python3
"""SDC HW4：radar/GPS EKF；GT callback 只存檔，完全不影響濾波。"""
import csv
import json
import math
import os
from pathlib import Path

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
        if np.all(selected == 0):
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
        return covariance(selected, len(indices))

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
        result_dir = Path(os.path.expanduser(self.get_parameter('result_dir').value))
        result_dir.mkdir(parents=True, exist_ok=True)
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
        stats = dict(self.filter.stats)
        stats.update(invalid_messages=self.invalid_messages,
                     fallback_covariance_messages=self.fallback_counts,
                     pending_gps_after_last_radar=len(self.filter.pending_gps),
                     samples={key: len(rows) for key, rows in self.records.items()},
                     evaluation='No time-aligned accuracy metric computed; inspect trajectories.')
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
