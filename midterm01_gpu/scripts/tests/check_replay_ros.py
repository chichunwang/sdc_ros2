#!/usr/bin/env python3
"""Run inside GPU container against a paused, three-frame replay.

Checks actual ROS/TF/PLY contracts. It does not rerun localization or score poses.
"""
from output_layout import loads_saved_json, saved_path
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as RosPath
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import MarkerArray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('replay_directory', type=saved_path)
    args = parser.parse_args()
    report = loads_saved_json((args.replay_directory/'replay.json').read_text())
    assert len(report['records']) >= 3
    rclpy.init()
    node = rclpy.create_node('check_saved_replay')
    messages = {}
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    subscriptions = []
    for topic, kind in [('/replay/pose', PoseStamped), ('/replay/path', RosPath), ('/replay/source', PointCloud2),
                        ('/replay/submap', PointCloud2), ('/replay/markers', MarkerArray)]:
        subscriptions.append(node.create_subscription(kind, topic, lambda msg, topic=topic: messages.__setitem__(topic, msg), qos))
    subscriptions.append(node.create_subscription(TFMessage, '/tf', lambda msg: messages.__setitem__('/tf', msg), 10))

    def until(predicate):
        deadline = time.monotonic() + 35
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
            if predicate(): return
        raise AssertionError('Timed out waiting for replay ROS messages')

    def check(index):
        expected = report['records'][index]
        until(lambda: len(messages) == 6 and len(messages['/replay/path'].poses) == index + 1
              and messages['/replay/source'].header.stamp == messages['/replay/pose'].header.stamp
              and messages['/replay/submap'].header.stamp == messages['/replay/pose'].header.stamp
              and messages['/replay/markers'].markers[0].header.stamp == messages['/replay/pose'].header.stamp
              and f"frame {expected['idx']} / original" in messages['/replay/markers'].markers[0].text)
        pose = messages['/replay/pose'].pose
        np.testing.assert_allclose([pose.position.x, pose.position.y, pose.position.z], [expected['pose'][i] for i in (3,7,11)], atol=1e-7)
        actual_q = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        assert abs(abs(np.dot(actual_q, expected['quaternion'])) - 1) < 1e-7
        tf = next(t for t in messages['/tf'].transforms if t.child_frame_id == 'replay_source')
        np.testing.assert_allclose([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z],
                                   [pose.position.x, pose.position.y, pose.position.z], atol=1e-7)
        assert tf.header.frame_id == 'map'
        assert messages['/replay/source'].header.frame_id == 'map'
        assert messages['/replay/submap'].header.frame_id == 'map'
        assert messages['/replay/source'].width > 0 and messages['/replay/submap'].width > 0
        text = messages['/replay/markers'].markers[0].text
        assert str(expected['timestamp_ns']) in text and 'RPY [deg]' in text
        heading = messages['/replay/markers'].markers[1].pose.orientation
        assert abs(math.atan2(2 * heading.w * heading.z, 1 - 2 * heading.z**2) - expected['rpy'][2]) < 1e-7
        return expected

    first = check(0)
    # Independently recover local coordinates from world geometry, then verify
    # scale-once voxel centroids against the original ASCII/binary PLY.
    with Path(first['container_path']).open('rb') as stream:
        count, format_name, properties = 0, '', []
        while True:
            line = stream.readline().decode('ascii').strip()
            if line.startswith('format '): format_name = line.split()[1]
            if line.startswith('element vertex '): count = int(line.split()[-1])
            if line.startswith('property '): properties.append(line.split()[1:])
            if line == 'end_header': break
            assert line, 'Truncated PLY'
        if format_name == 'ascii':
            raw = np.loadtxt(stream, max_rows=count, usecols=(0,1,2), dtype=np.float32)
        else:
            assert format_name == 'binary_little_endian'
            types = {'float':'<f4', 'uchar':'u1'}
            packed = np.fromfile(stream, dtype=[(name, types[kind]) for kind, name in properties], count=count)
            raw = np.column_stack([packed[name] for name in ('x','y','z')])
    raw *= np.float32(report['params']['ply_scale'])
    raw = raw[np.isfinite(raw).all(axis=1)]
    cloud = messages['/replay/source']
    values = np.array(list(point_cloud2.read_points(cloud, field_names=('x','y','z'), skip_nans=True)), dtype=[('x','f4'),('y','f4'),('z','f4')])
    xyz = np.column_stack([values[name] for name in ('x','y','z')])
    matrix = np.array(first['pose']).reshape(3,4)
    xyz = (xyz.astype(float) - matrix[:,3]) @ np.linalg.inv(matrix[:,:3]).T
    inverse = np.float32(1 / np.float32(report['params']['input_voxel_leaf_size']))
    bins = np.floor(raw * inverse).astype(np.int64)
    tested = 0
    for point in xyz[::max(1, len(xyz)//24)]:
        key = np.floor(point * inverse).astype(np.int64)
        matches = raw[(bins == key).all(axis=1)]
        if not len(matches): continue
        np.testing.assert_allclose(point, matches.mean(axis=0), atol=8e-4)
        tested += 1
    assert tested >= 8
    client = node.create_client(SetParameters, '/localization_replay/set_parameters')
    assert client.wait_for_service(timeout_sec=10)
    def set_parameters(parameters):
        request = SetParameters.Request()
        request.parameters = [p.to_parameter_msg() for p in parameters]
        return client.call_async(request)
    def set_index(index):
        future = set_parameters([Parameter('frame_index', value=index)])
        until(future.done)
        assert all(r.successful for r in future.result().results)
        check(index)
    set_index(2)
    set_index(1)
    set_index(0)
    future = set_parameters([Parameter('frame_index', value=999999)])
    until(future.done)
    assert not future.result().results[0].successful
    output = dict(status='passed', track=report['track'], run=report['run'],
                  checks=['late subscriber while paused', 'saved XYZ and full quaternion', 'TF frame/direction',
                          'original integer timestamp in HUD', 'map XY heading', 'map-frame PLY and independent inverse-pose voxel centroid check',
                          'seek forward/back/restart and path prefix', 'reject invalid seek'], centroid_bins=tested,
                  scope='Saved-output replay contracts, not localization accuracy')
    (args.replay_directory/'ros_check.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__': main()
