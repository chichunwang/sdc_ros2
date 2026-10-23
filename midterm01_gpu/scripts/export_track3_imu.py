#!/usr/bin/env python3
"""Export unmodified, availability-cutoff IMU measurements; no calibration or poses."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3

import numpy as np

RUN_ROOT = Path(__file__).resolve().parents[1] / 'outputs/run'

COVARIANCES = ('orientation_covariance', 'angular_velocity_covariance',
               'linear_acceleration_covariance')


def validate_arrays(arrays, end_ns):
    """Validate the exact NPZ schema without reordering or modifying measurements."""
    if not isinstance(end_ns, int) or isinstance(end_ns, bool) or end_ns <= 0:
        raise ValueError('end_ns must be a positive integer')
    shapes = {'availability_ns': (), 'measurement_ns': (), 'quaternion_xyzw': (4,),
              'gyro_rad_s': (3,), 'accel_m_s2': (3,), 'frame_id': ()}
    shapes.update({key: (9,) for key in COVARIANCES})
    if set(arrays) != set(shapes):
        raise ValueError('unexpected payload schema')
    n = len(arrays['availability_ns'])
    if n == 0:
        raise ValueError('no IMU messages before cutoff')
    for key, tail in shapes.items():
        value = arrays[key]
        expected = np.dtype('int64') if key.endswith('_ns') else np.dtype('float64')
        if value.shape != (n,) + tail:
            raise ValueError(f'{key}: invalid shape')
        if key == 'frame_id':
            if value.dtype.kind != 'U':
                raise ValueError('frame_id must be unicode')
        elif value.dtype != expected or not np.isfinite(value).all():
            raise ValueError(f'{key}: invalid dtype or nonfinite values')
    availability = arrays['availability_ns']
    measurement = arrays['measurement_ns']
    if np.any(availability <= 0) or np.any(measurement <= 0):
        raise ValueError('timestamps must be positive')
    if np.any(availability[1:] <= availability[:-1]):
        raise ValueError('availability_ns must be strictly increasing')
    if np.any(availability > end_ns) or np.any(measurement > availability):
        raise ValueError('measurement or availability exceeds causal cutoff')
    if np.any(np.abs(np.linalg.norm(arrays['quaternion_xyzw'], axis=1) - 1) > 1e-3):
        raise ValueError('quaternion norm differs from one by more than 1e-3')
    if len(np.unique(arrays['frame_id'])) != 1:
        raise ValueError('multiple frame_id values')
    for key in COVARIANCES:
        if np.any(arrays[key][:, 0] < 0):
            raise ValueError(f'{key}: unavailable covariance (negative first element)')
    return arrays


def extract_messages(rows, deserialize, end_ns):
    """Convert (bag timestamp, serialized Imu) rows using an injected decoder."""
    values = {key: [] for key in ('availability_ns', 'measurement_ns',
              'quaternion_xyzw', 'gyro_rad_s', 'accel_m_s2', 'frame_id') + COVARIANCES}
    for timestamp, payload in rows:
        msg = deserialize(payload)
        stamp = msg.header.stamp
        if not (0 <= stamp.nanosec < 1_000_000_000):
            raise ValueError('invalid header nanosec')
        values['availability_ns'].append(timestamp)
        values['measurement_ns'].append(stamp.sec * 1_000_000_000 + stamp.nanosec)
        values['frame_id'].append(msg.header.frame_id)
        values['quaternion_xyzw'].append([getattr(msg.orientation, k) for k in 'xyzw'])
        values['gyro_rad_s'].append([getattr(msg.angular_velocity, k) for k in 'xyz'])
        values['accel_m_s2'].append([getattr(msg.linear_acceleration, k) for k in 'xyz'])
        for key in COVARIANCES:
            values[key].append(getattr(msg, key))
    arrays = {key: np.asarray(value, dtype=(np.int64 if key.endswith('_ns') else
              str if key == 'frame_id' else np.float64)) for key, value in values.items()}
    return validate_arrays(arrays, end_ns)


def query_rows(connection, end_ns):
    """Select only the declared Imu topic and the available prefix."""
    if not isinstance(end_ns, int) or isinstance(end_ns, bool) or end_ns <= 0:
        raise ValueError('end_ns must be a positive integer')
    topics = connection.execute('SELECT id, type FROM topics WHERE name = ?',
                                ('/imu/data',)).fetchall()
    if len(topics) != 1 or topics[0][1] != 'sensor_msgs/msg/Imu':
        raise ValueError('expected exactly one /imu/data topic of sensor_msgs/msg/Imu')
    return connection.execute(
        'SELECT timestamp, data FROM messages WHERE topic_id = ? AND timestamp <= ? '
        'ORDER BY timestamp, id', (topics[0][0], end_ns))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_metadata(arrays, end_ns, bag):
    validate_arrays(arrays, end_ns)
    bag = Path(bag).resolve(strict=True)
    offset = arrays['measurement_ns'] - arrays['availability_ns']
    return {'count': len(offset), 'cutoff_ns': end_ns,
            'availability_range_ns': [int(arrays['availability_ns'].min()), int(arrays['availability_ns'].max())],
            'measurement_range_ns': [int(arrays['measurement_ns'].min()), int(arrays['measurement_ns'].max())],
            'header_minus_bag_offset_ns': {'min': int(offset.min()), 'max': int(offset.max())},
            'all_zero_covariance_message_counts': {key: int(np.all(arrays[key] == 0, axis=1).sum()) for key in COVARIANCES},
            'zero_covariance_meaning': 'unknown', 'calibrated': False,
            'future_messages_used': False, 'topic': '/imu/data', 'type': 'sensor_msgs/msg/Imu',
            'payload_schema': {key: {'dtype': str(value.dtype), 'shape': list(value.shape)} for key, value in arrays.items()},
            'script_sha256': sha256(__file__),
            'bag': {'path': str(bag), 'size_bytes': bag.stat().st_size, 'sha256': sha256(bag)}}


def validate_output(output):
    output = Path(output).absolute()
    root = RUN_ROOT.resolve()
    parent = output.parent.resolve(strict=True)
    relative = parent.relative_to(root)
    if not relative.parts or not re.fullmatch(r'track0[0-3]_(?:test|full)_\d{10}(?:_\d{4})?', relative.parts[0]):
        raise ValueError('output must be inside an existing run under GPU outputs/run')
    run = root / relative.parts[0]
    if any((run / name).exists() for name in ('result.txt', 'diagnostic.csv', 'validation.json')):
        raise ValueError('run already contains localization results')
    if output.suffix != '.npz':
        raise ValueError('output must have .npz suffix')
    output = parent / output.name
    metadata = output.with_suffix('.json')
    if output.exists() or output.is_symlink() or metadata.exists() or metadata.is_symlink():
        raise FileExistsError('output or metadata already exists')
    return output, metadata


def write_export(output, arrays, metadata):
    output, sidecar = validate_output(output)
    # Both files are exclusively created; never replace an existing result.
    with sidecar.open('x', encoding='utf-8') as stream:
        with output.open('xb') as payload:
            np.savez_compressed(payload, **arrays)
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', required=True, type=Path, help='single ROS2 SQLite .db3 file')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--end-ns', required=True, type=int)
    args = parser.parse_args()
    validate_output(args.output)
    if args.end_ns <= 0:
        parser.error('--end-ns must be positive')
    bag = args.bag.resolve(strict=True)
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu
    with sqlite3.connect(bag.as_uri() + '?mode=ro', uri=True) as connection:
        arrays = extract_messages(query_rows(connection, args.end_ns),
                                  lambda data: deserialize_message(data, Imu), args.end_ns)
    write_export(args.output, arrays, build_metadata(arrays, args.end_ns, bag))
    print(f'Exported {len(arrays["availability_ns"])} IMU messages: {args.output}')


if __name__ == '__main__':
    main()
