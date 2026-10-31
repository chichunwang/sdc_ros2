"""唯讀 SQLite / CDR。每次僅傳回已抵達且量測時間不晚於影像的事件。

ROS 反序列化沿用既有容器；保留 GNSS XYZ 與有效 quaternion；遲到事件不回修。
"""
from pathlib import Path
import sqlite3
import numpy as np


class SensorStream:
    def __init__(self, path, gnss_frame=''):
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import Imu
        from geometry_msgs.msg import PoseStamped
        self.db = sqlite3.connect(Path(path).resolve(strict=True).as_uri()+'?mode=ro', uri=True)
        self.db.execute('PRAGMA query_only=ON')
        self.decode = deserialize_message
        self.classes = {'imu': Imu, 'gnss': PoseStamped}
        names = {'/imu/data': ('imu', 'sensor_msgs/msg/Imu'),
                 '/localizer_node/gnss_pose': ('gnss', 'geometry_msgs/msg/PoseStamped')}
        self.kinds = {}
        for name, (kind, expected) in names.items():
            rows = self.db.execute('SELECT id,type,serialization_format FROM topics WHERE name=?', (name,)).fetchall()
            if len(rows) != 1 or rows[0][1:] != (expected, 'cdr'):
                raise ValueError(f'Invalid required sensor topic: {name}')
            self.kinds[rows[0][0]] = kind
        ids = list(self.kinds)
        self.cursor = self.db.execute('SELECT id,timestamp,topic_id,data FROM messages WHERE topic_id IN (?,?) ORDER BY timestamp,id', ids)
        self.next = self.cursor.fetchone()
        self.queue = []
        self.last_arrival = self.last_camera = None
        self.gnss_frame = gnss_frame
        self.counts = dict(imu=0, gnss=0, late=0, duplicate=0)

    def take(self, arrival_ns, camera_ns):
        if type(arrival_ns) is not int or type(camera_ns) is not int or camera_ns > arrival_ns:
            raise ValueError('Integer camera acquisition and arrival clocks required')
        if self.last_arrival is not None and (arrival_ns <= self.last_arrival or camera_ns <= self.last_camera):
            raise ValueError('Clocks must strictly increase')
        while self.next is not None and self.next[1] <= arrival_ns:
            mid, arrival, tid, blob = self.next
            kind = self.kinds[tid]
            m = self.decode(blob, self.classes[kind])
            stamp = int(m.header.stamp.sec)*1_000_000_000 + int(m.header.stamp.nanosec)
            if not 0 < stamp <= arrival:
                raise ValueError(f'Invalid {kind} measurement clock')
            row = dict(kind=kind, timestamp_ns=stamp, arrival_ns=int(arrival), message_id=int(mid))
            if kind == 'imu':
                if m.header.frame_id != 'imu_link':
                    raise ValueError('Unexpected IMU frame')
                if m.angular_velocity_covariance[0] < 0 or m.linear_acceleration_covariance[0] < 0:
                    raise ValueError('Unavailable IMU measurement')
                if m.orientation_covariance[0] < 0:
                    raise ValueError('Original guided model requires available IMU orientation')
                quat = [m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w]
                if not np.isfinite(quat).all() or abs(np.linalg.norm(quat)-1.) > .02:
                    raise ValueError('Invalid IMU orientation quaternion')
                row.update(quaternion=quat, gyro=[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z],
                           accel=[m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z],
                           orientation_covariance=list(m.orientation_covariance),
                           gyro_covariance=list(m.angular_velocity_covariance))
            else:
                if m.header.frame_id != self.gnss_frame:
                    raise ValueError('Unexpected GNSS frame; verify map transform before use')
                row['xy'] = [m.pose.position.x, m.pose.position.y]
                row['z'] = m.pose.position.z
                q = [m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w]
                valid_heading = bool(np.isfinite(q).all() and abs(np.linalg.norm(q)-1.) <= .02)
                # Invalid orientation must not discard valid GNSS XYZ.
                row['heading_quaternion'] = q if valid_heading else None
                row['heading_status'] = 'available_unknown_covariance' if valid_heading else 'invalid_orientation'
            values = row['gyro']+row['accel'] if kind == 'imu' else row['xy']+[row['z']]
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite sensor data')
            if self.last_camera is not None and stamp <= self.last_camera:
                self.counts['late'] += 1
            else:
                self.queue.append(row)
            self.next = self.cursor.fetchone()
        ready = [r for r in self.queue if r['timestamp_ns'] <= camera_ns]
        self.queue = [r for r in self.queue if r['timestamp_ns'] > camera_ns]
        ready.sort(key=lambda r: (r['timestamp_ns'], r['message_id']))
        unique = {}
        for r in ready:
            key = (r['kind'], r['timestamp_ns'])
            self.counts['duplicate'] += int(key in unique)
            unique[key] = r
        ready = sorted(unique.values(), key=lambda r: (r['timestamp_ns'], r['message_id']))
        for r in ready:
            self.counts[r['kind']] += 1
        self.last_arrival, self.last_camera = arrival_ns, camera_ns
        return ready

    def summary(self):
        return dict(self.counts, buffered=len(self.queue),
                    policy='arrival<=PLY bag time AND header<=camera header; late discarded; no interpolation')

    def close(self):
        self.db.close()
