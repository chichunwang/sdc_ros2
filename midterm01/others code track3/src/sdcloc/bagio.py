"""讀 rosbag2 的 sqlite 檔，不靠 ROS：自己解 CDR 拿 GNSS 與 IMU。"""
import sqlite3
import struct
from pathlib import Path

import numpy as np

GNSS_TOPIC = '/localizer_node/gnss_pose'
IMU_TOPIC = '/imu/data'
GNSS_TYPE = 'geometry_msgs/msg/PoseStamped'
IMU_TYPE = 'sensor_msgs/msg/Imu'


class _Cdr:
    # 對齊是相對 payload 起點算的（4 byte 的 encapsulation 不算）
    def __init__(self, blob):
        if blob[:4] != b'\x00\x01\x00\x00':
            raise ValueError(f'unsupported CDR encapsulation {blob[:4].hex()}')
        self.buf = blob[4:]
        self.pos = 0

    def _align(self, n):
        self.pos += -self.pos % n

    def u32(self):
        self._align(4)
        v, = struct.unpack_from('<I', self.buf, self.pos)
        self.pos += 4
        return v

    def i32(self):
        self._align(4)
        v, = struct.unpack_from('<i', self.buf, self.pos)
        self.pos += 4
        return v

    def string(self):
        n = self.u32()
        s = self.buf[self.pos:self.pos + n - 1].decode()
        self.pos += n
        return s

    def doubles(self, n):
        self._align(8)
        v = struct.unpack_from(f'<{n}d', self.buf, self.pos)
        self.pos += 8 * n
        return v

    def header(self):
        sec, nsec = self.i32(), self.u32()
        return sec * 1_000_000_000 + nsec, self.string()


def decode_pose_stamped(blob):
    c = _Cdr(blob)
    stamp, frame = c.header()
    d = c.doubles(7)
    return stamp, frame, d[:3], d[3:]


def decode_imu(blob):
    c = _Cdr(blob)
    stamp, frame = c.header()
    d = c.doubles(4 + 9 + 3 + 9 + 3 + 9)
    return {'stamp': stamp, 'frame_id': frame, 'quat': d[0:4], 'ori_cov0': d[4],
            'gyro': d[13:16], 'acc': d[25:28]}


def _find_db3(target):
    target = Path(target)
    if target.is_file():
        return target
    found = sorted(target.glob('*.db3'))
    if len(found) != 1:
        raise FileNotFoundError(f'expected one .db3 in {target}, found {len(found)}')
    return found[0]


def load_sensors(target, need_gnss=True):
    """回傳 gnss／imu 兩組 numpy 陣列（照 bag 時間排序）和各 topic 的筆數。
    need_gnss=False：bag 沒有 GNSS（Track03）時 gnss 回空陣列。"""
    path = _find_db3(target)
    db = sqlite3.connect(f'file:{path.as_posix()}?mode=ro', uri=True)
    try:
        topics = {name: (tid, kind) for tid, name, kind in db.execute('SELECT id,name,type FROM topics')}
        counts = dict(db.execute('SELECT topic_id, COUNT(*) FROM messages GROUP BY topic_id'))
        info = {name: (kind, counts.get(tid, 0)) for name, (tid, kind) in topics.items()}
        for topic, kind in ((GNSS_TOPIC, GNSS_TYPE), (IMU_TOPIC, IMU_TYPE)):
            if topic == GNSS_TOPIC and not need_gnss and topic not in topics:
                continue
            if topic not in topics:
                raise KeyError(f'{topic} not in bag; topics: {sorted(topics)}')
            if topics[topic][1] != kind:
                raise ValueError(f'{topic} is {topics[topic][1]}, expected {kind}')

        def rows(topic):
            return db.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp',
                              (topics[topic][0],))

        g_t, g_ht, g_xyz, g_q, g_frames = [], [], [], [], set()
        for t, blob in rows(GNSS_TOPIC) if GNSS_TOPIC in topics else ():
            stamp, frame, xyz, q = decode_pose_stamped(blob)
            g_t.append(t); g_ht.append(stamp); g_xyz.append(xyz); g_q.append(q); g_frames.add(frame)
        i_t, i_ht, i_q, i_g, i_a, i_cov, i_frames = [], [], [], [], [], [], set()
        for t, blob in rows(IMU_TOPIC):
            m = decode_imu(blob)
            i_t.append(t); i_ht.append(m['stamp']); i_q.append(m['quat']); i_g.append(m['gyro'])
            i_a.append(m['acc']); i_cov.append(m['ori_cov0']); i_frames.add(m['frame_id'])
    finally:
        db.close()
    return {
        'path': str(path),
        'topics': info,
        'gnss': {'t': np.array(g_t, dtype=np.int64), 'header_t': np.array(g_ht, dtype=np.int64),
                 'xyz': np.array(g_xyz, dtype=float).reshape(-1, 3), 'quat': np.array(g_q, dtype=float).reshape(-1, 4),
                 'frame_ids': sorted(g_frames)},
        'imu': {'t': np.array(i_t, dtype=np.int64), 'header_t': np.array(i_ht, dtype=np.int64),
                'quat': np.array(i_q, dtype=float).reshape(-1, 4), 'gyro': np.array(i_g, dtype=float).reshape(-1, 3),
                'acc': np.array(i_a, dtype=float).reshape(-1, 3), 'ori_cov0': np.array(i_cov, dtype=float),
                'frame_ids': sorted(i_frames)},
    }
