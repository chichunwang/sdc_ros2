#!/usr/bin/env python3
"""在 Humble 容器內檢查真實 sensor topic、frame、時間與 covariance。"""
import json
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def main():
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=sys.argv[1], storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    topics = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    expected = {'/gps': 'geometry_msgs/msg/PoseWithCovarianceStamped',
                '/radar_odometry': 'nav_msgs/msg/Odometry',
                '/gt_odom': 'nav_msgs/msg/Odometry'}
    for topic, message_type in expected.items():
        if topics.get(topic) != message_type:
            raise RuntimeError(f'{topic}: expected {message_type}, got {topics.get(topic)}')
    if '/clock' not in topics:
        raise RuntimeError('此 bag 沒有 /clock，與講義不同；請確認資料版本後再播放。')
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(expected)))
    types = {topic: get_message(name) for topic, name in expected.items()}
    report = {}
    previous_stamp = {}
    while reader.has_next():
        topic, serialized, record_ns = reader.read_next()
        msg = deserialize_message(serialized, types[topic])
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        full = np.asarray(msg.pose.covariance).reshape(6, 6)
        indices = [0, 1] if topic == '/gps' else [0, 1, 5]
        selected = full[np.ix_(indices, indices)]
        if topic not in report:
            report[topic] = dict(count=0, frames=[], first_stamp_ns=stamp,
                                 last_stamp_ns=stamp, nonincreasing_stamps=0,
                                 all_zero_covariances=0, invalid_covariances=0,
                                 first_covariance_6x6=full.tolist(),
                                 selected_diagonal_min=np.diag(selected).tolist(),
                                 selected_diagonal_max=np.diag(selected).tolist(),
                                 max_record_header_offset_seconds=0.0)
        item = report[topic]
        item['count'] += 1
        if msg.header.frame_id not in item['frames']:
            item['frames'].append(msg.header.frame_id)
        item['last_stamp_ns'] = stamp
        if topic in previous_stamp and stamp <= previous_stamp[topic]:
            item['nonincreasing_stamps'] += 1
        previous_stamp[topic] = stamp
        item['all_zero_covariances'] += int(np.all(selected == 0))
        valid = (np.isfinite(selected).all() and np.allclose(selected, selected.T)
                 and np.linalg.eigvalsh((selected+selected.T)/2)[0] >= -1e-9)
        item['invalid_covariances'] += int(not valid)
        item['selected_diagonal_min'] = np.minimum(item['selected_diagonal_min'], np.diag(selected)).tolist()
        item['selected_diagonal_max'] = np.maximum(item['selected_diagonal_max'], np.diag(selected)).tolist()
        item['max_record_header_offset_seconds'] = max(item['max_record_header_offset_seconds'],
                                                       abs(record_ns-stamp)/1e9)
    missing = set(expected) - set(report)
    if missing:
        raise RuntimeError(f'Topics have no messages: {sorted(missing)}')
    for item in report.values():
        duration = (item['last_stamp_ns'] - item['first_stamp_ns']) / 1e9
        item['mean_rate_hz'] = (item['count']-1)/duration if duration > 0 else None
    print(json.dumps(dict(bag=sys.argv[1], has_clock=True, sensors=report), indent=2))


if __name__ == '__main__':
    main()
