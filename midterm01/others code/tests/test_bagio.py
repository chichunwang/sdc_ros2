import sqlite3
import struct

import numpy as np
import pytest

from sdcloc.bagio import decode_imu, decode_pose_stamped, load_sensors


def header(sec, nsec, frame):
    fid = frame.encode() + b'\x00'
    body = struct.pack('<iI', sec, nsec) + struct.pack('<I', len(fid)) + fid
    return body + b'\x00' * (-len(body) % 8)   # 對齊到 8 才放 double


def pose_msg(sec, nsec, frame, xyz, quat):
    return b'\x00\x01\x00\x00' + header(sec, nsec, frame) + struct.pack('<7d', *xyz, *quat)


def imu_msg(sec, nsec, frame, quat, gyro, acc, ori_cov0=0.01):
    cov = [ori_cov0] + [0.0] * 8
    doubles = [*quat, *cov, *gyro, *([0.0] * 9), *acc, *([0.0] * 9)]
    return b'\x00\x01\x00\x00' + header(sec, nsec, frame) + struct.pack(f'<{len(doubles)}d', *doubles)


def test_decode_pose_stamped():
    stamp, frame, xyz, quat = decode_pose_stamped(pose_msg(12, 345, 'map', (1.5, -2.0, 3.25), (0, 0, 0.6, 0.8)))
    assert stamp == 12_000_000_345
    assert frame == 'map'
    assert xyz == pytest.approx((1.5, -2.0, 3.25))
    assert quat == pytest.approx((0, 0, 0.6, 0.8))


def test_decode_pose_stamped_empty_frame_id():
    # frame_id 空字串時長度欄是 1（只有結尾 0），對齊位置不同，要能正確解
    stamp, frame, xyz, _ = decode_pose_stamped(pose_msg(1, 2, '', (7.0, 8.0, 9.0), (0, 0, 0, 1)))
    assert frame == ''
    assert xyz == pytest.approx((7.0, 8.0, 9.0))


def test_decode_imu():
    m = decode_imu(imu_msg(3, 4, 'imu_link', (0.1, 0.2, 0.3, 0.927), (0.01, -0.02, 0.5), (0.3, 0.1, 9.8)))
    assert m['stamp'] == 3_000_000_004
    assert m['frame_id'] == 'imu_link'
    assert m['quat'] == pytest.approx((0.1, 0.2, 0.3, 0.927))
    assert m['gyro'] == pytest.approx((0.01, -0.02, 0.5))
    assert m['acc'] == pytest.approx((0.3, 0.1, 9.8))
    assert m['ori_cov0'] == pytest.approx(0.01)


def test_unknown_encapsulation_rejected():
    bad = b'\x00\x07\x00\x00' + pose_msg(0, 0, 'm', (0, 0, 0), (0, 0, 0, 1))[4:]
    with pytest.raises(ValueError):
        decode_pose_stamped(bad)


def make_bag(path):
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT, serialization_format TEXT, offered_qos_profiles TEXT)')
    db.execute('CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB)')
    db.execute("INSERT INTO topics VALUES (1, '/localizer_node/gnss_pose', 'geometry_msgs/msg/PoseStamped', 'cdr', '')")
    db.execute("INSERT INTO topics VALUES (2, '/imu/data', 'sensor_msgs/msg/Imu', 'cdr', '')")
    db.execute("INSERT INTO topics VALUES (3, '/cam', 'sensor_msgs/msg/CompressedImage', 'cdr', '')")
    # 故意亂序寫入，讀出來要照 bag 時間排序
    db.execute('INSERT INTO messages(topic_id,timestamp,data) VALUES (1, 2000, ?)', (pose_msg(0, 2000, '', (2, 0, 0), (0, 0, 0, 1)),))
    db.execute('INSERT INTO messages(topic_id,timestamp,data) VALUES (1, 1000, ?)', (pose_msg(0, 1000, '', (1, 0, 0), (0, 0, 0, 1)),))
    for t in (1000, 1500, 2000):
        db.execute('INSERT INTO messages(topic_id,timestamp,data) VALUES (2, ?, ?)',
                   (t, imu_msg(0, t, 'imu_link', (0, 0, 0, 1), (0, 0, 0.1), (0, 0, 9.8))))
    db.execute("INSERT INTO messages(topic_id,timestamp,data) VALUES (3, 1200, x'00')")
    db.commit()
    db.close()


def test_load_sensors_from_file_and_dir(tmp_path):
    bag = tmp_path / 'bag' / 'Track.db3'
    bag.parent.mkdir()
    make_bag(bag)
    for target in (bag, bag.parent):
        s = load_sensors(target)
        assert s['gnss']['t'].tolist() == [1000, 2000]
        assert s['gnss']['xyz'][:, 0].tolist() == [1.0, 2.0]
        assert s['imu']['t'].tolist() == [1000, 1500, 2000]
        assert np.allclose(s['imu']['gyro'][:, 2], 0.1)
        assert s['topics']['/cam'] == ('sensor_msgs/msg/CompressedImage', 1)
