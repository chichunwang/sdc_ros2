import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import numpy as np
from plyfile import PlyData


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def frames(directory, suffix='.ply', expected=3142):
    rows = []
    for p in Path(directory).glob('*'+suffix):
        m = re.fullmatch(r'(\d+)_ts_(\d+)', p.stem)
        if not m: raise ValueError(f'Invalid frame name: {p.name}')
        rows.append(dict(original_idx=int(m[1]), timestamp_ns=int(m[2]), path=p))
    rows.sort(key=lambda r: r['timestamp_ns'])
    if len(rows) != expected or [r['original_idx'] for r in rows] != list(range(expected)):
        raise ValueError(f'Expected contiguous 0..{expected-1} frames, found {len(rows)}')
    if any(b['timestamp_ns'] <= a['timestamp_ns'] for a,b in zip(rows, rows[1:])):
        raise ValueError('Duplicate or decreasing frame timestamp')
    return rows


def load_ply(path):
    v = PlyData.read(str(path))['vertex']
    xyz = np.column_stack([v[a] for a in ('x','y','z')]).astype(np.float64)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) < 100: raise ValueError(f'Too few finite points: {path}')
    return xyz


def load_map(path, leaf):
    import pandas as pd
    import cupy as cp
    from .gpu import voxel_mean
    header = []; offset = 0
    with Path(path).open('rb') as f:
        while True:
            line = f.readline().decode('ascii').strip(); header.append(line); offset += 1
            if line.startswith('DATA '): break
            if offset > 100: raise ValueError('Invalid PCD header')
    fields = next(s.split()[1:] for s in header if s.startswith('FIELDS '))
    if header[-1] != 'DATA ascii': raise ValueError('This streaming loader requires ASCII PCD')
    # Explicitly select XYZ; intensity and timestamp-related map fields never enter localization.
    cols = [fields.index(a) for a in ('x', 'y', 'z')]
    pieces, weights = [], []; read_count = 0
    for chunk in pd.read_csv(path, sep=r'\s+', skiprows=offset, header=None,
                             usecols=cols, dtype=np.float32, chunksize=1_000_000):
        p = chunk.to_numpy(); p = p[np.isfinite(p).all(axis=1)]; read_count += len(p)
        mean, count = voxel_mean(p, leaf)
        pieces.append(cp.asnumpy(mean)); weights.append(cp.asnumpy(count))
        if len(pieces) >= 4:
            mean, count = voxel_mean(np.concatenate(pieces), leaf, np.concatenate(weights))
            pieces = [cp.asnumpy(mean)]; weights = [cp.asnumpy(count)]
    if not pieces: raise ValueError('Empty map')
    mean, _ = voxel_mean(np.concatenate(pieces), leaf, np.concatenate(weights))
    return mean, read_count


class ImuStream:
    """Only /imu/data; both availability (bag) and measurement (header) clocks gated."""
    def __init__(self, bag):
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import Imu
        self.decode = lambda x: deserialize_message(x, Imu)
        self.db = sqlite3.connect(f'file:{Path(bag).resolve()}?mode=ro', uri=True)
        topics = self.db.execute('select id,name,type from topics').fetchall()
        matches = [r for r in topics if r[1] == '/imu/data' and r[2] == 'sensor_msgs/msg/Imu']
        if len(matches) != 1: raise ValueError('Expected /imu/data of type sensor_msgs/msg/Imu')
        self.cursor = self.db.execute('select timestamp,data from messages where topic_id=? order by timestamp,id', (matches[0][0],))
        self.next = self.cursor.fetchone(); self.pending = []; self.invalid = 0

    def until(self, frame_ns):
        while self.next is not None and self.next[0] <= frame_ns:
            bag_ns, raw = self.next; m = self.decode(raw)
            stamp = int(m.header.stamp.sec)*1_000_000_000+int(m.header.stamp.nanosec)
            q = np.array([m.orientation.x,m.orientation.y,m.orientation.z,m.orientation.w])
            w = np.array([m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z])
            if (stamp > 0 and np.isfinite(q).all() and np.isfinite(w).all() and
                    0.9 < np.linalg.norm(q) < 1.1 and m.orientation_covariance[0] != -1 and
                    m.angular_velocity_covariance[0] != -1):
                from scipy.spatial.transform import Rotation
                # Quaternion's gravity/world z axis projects gyro without assuming camera/IMU yaw extrinsics.
                rate = float((Rotation.from_quat(q).as_matrix() @ w)[2])
                self.pending.append((stamp, int(bag_ns), rate))
            else: self.invalid += 1
            self.next = self.cursor.fetchone()
        ready = [s for s in self.pending if s[0] <= frame_ns]
        self.pending = [s for s in self.pending if s[0] > frame_ns]
        return sorted(ready)

    def close(self): self.db.close()
