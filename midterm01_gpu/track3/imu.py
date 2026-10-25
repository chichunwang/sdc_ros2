"""Read-only, arrival-causal ROS 2 SQLite IMU input; no orientation/GNSS use."""
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import numpy as np


@dataclass
class ImuSample:
    timestamp_ns: int                 # bag arrival clock, also used by PLY names
    header_timestamp_ns: int          # evidence only; never silently substituted
    angular_velocity: np.ndarray     # rad/s, in imu_link
    linear_acceleration: np.ndarray  # specific force, m/s^2, in imu_link
    frame_id: str
    message_id: int = 0
    arrival_timestamp_ns: int | None = None


class ImuReader:
    def __init__(self, path, topic='/imu/data', expected_frame='imu_link'):
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import Imu
        self.decode = lambda blob: deserialize_message(blob, Imu)
        self.path = Path(path).resolve(strict=True)
        self.db = sqlite3.connect(self.path.as_uri()+'?mode=ro', uri=True)
        self.db.execute('PRAGMA query_only=ON')
        try:
            topics = self.db.execute('SELECT id,type,serialization_format FROM topics WHERE name=?', (topic,)).fetchall()
            if len(topics) != 1 or topics[0][1:] != ('sensor_msgs/msg/Imu', 'cdr'):
                raise ValueError('IMU topic must uniquely contain sensor_msgs/msg/Imu in CDR')
            self.cursor = self.db.execute('SELECT id,timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id', (topics[0][0],))
            self.next = self.cursor.fetchone()
            if self.next is None:
                raise ValueError('Empty IMU topic')
        except BaseException:
            self.db.close()
            raise
        self.expected_frame = expected_frame
        self.last_cutoff = None
        self.count = 0
        self.zero_gyro_covariance_count = 0
        self.zero_accel_covariance_count = 0

    def take_until(self, timestamp_ns):
        if not isinstance(timestamp_ns, (int, np.integer)):
            raise ValueError('IMU cutoff must be integer nanoseconds')
        if self.last_cutoff is not None and timestamp_ns <= self.last_cutoff:
            raise ValueError('IMU cutoffs must strictly increase')
        samples = []
        while self.next is not None and self.next[1] <= timestamp_ns:
            message_id, arrival, blob = self.next
            m = self.decode(blob)
            if m.header.frame_id != self.expected_frame:
                raise ValueError(f'Unexpected IMU frame: {m.header.frame_id!r}')
            if m.angular_velocity_covariance[0] < 0 or m.linear_acceleration_covariance[0] < 0:
                raise ValueError('IMU reports unavailable angular velocity or acceleration')
            gyro = np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
            accel = np.array([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
            if not np.isfinite(gyro).all() or not np.isfinite(accel).all():
                raise ValueError('Nonfinite IMU measurement')
            self.zero_gyro_covariance_count += int(not np.any(m.angular_velocity_covariance))
            self.zero_accel_covariance_count += int(not np.any(m.linear_acceleration_covariance))
            header = int(m.header.stamp.sec)*1_000_000_000 + int(m.header.stamp.nanosec)
            samples.append(ImuSample(int(arrival), header, gyro, accel, m.header.frame_id, int(message_id)))
            self.count += 1
            self.next = self.cursor.fetchone()
        self.last_cutoff = int(timestamp_ns)
        return samples

    def summary(self):
        return dict(samples_consumed=self.count,
                    unknown_zero_gyro_covariance_samples=self.zero_gyro_covariance_count,
                    unknown_zero_accel_covariance_samples=self.zero_accel_covariance_count,
                    covariance_policy='Configured continuous noise densities; zero covariance never means noiseless')

    def close(self):
        self.db.close()
