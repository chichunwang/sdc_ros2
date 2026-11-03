"""CPU/ROS exact wire-equivalence checks; no CUDA or localization required."""
from array import array
import numpy as np
import pytest
from rclpy.serialization import serialize_message, deserialize_message
from sensor_msgs.msg import PointCloud2
from ros_wire import canonical_pointcloud_wire
from track3.visualization import cloud_message
from track2.visualization import cloud_message as track2_message


@pytest.mark.parametrize('count', [0, 1, 79, 6000])
@pytest.mark.parametrize('color', [None, (70, 235, 100), (255, 160, 40)])
@pytest.mark.parametrize('transformed', [False, True])
def test_fast_payload_preserves_full_ros_wire_and_inputs(count, color, transformed):
    points = np.random.default_rng(71).normal(size=(count, 3))
    # A strided input exercises packing and no input mutation.
    buf = np.empty((count, 6)); buf[:, ::2] = points; points = buf[:, ::2]
    before = points.copy()
    pose = np.eye(4); pose[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    pose[:3, 3] = [3272., -984., 1.6]
    stamp = 1727241181291090775
    kwargs = dict(pose=pose if transformed else None, scale=7.3, color=color)
    baseline = cloud_message(points, stamp, **kwargs)
    actual = track2_message(points, stamp, **kwargs)
    assert isinstance(actual.data, array) and actual.data.typecode == 'B'
    assert actual == baseline
    assert bytes(actual.data) == bytes(baseline.data)
    assert canonical_pointcloud_wire(serialize_message(actual))[0] == canonical_pointcloud_wire(serialize_message(baseline))[0]
    assert deserialize_message(serialize_message(actual), PointCloud2) == actual
    np.testing.assert_array_equal(points, before)
    assert actual.row_step == actual.width*actual.point_step
    assert actual.header.stamp.sec*10**9+actual.header.stamp.nanosec == stamp


def test_payload_handles_nonfinite_values_and_preserves_density_flag():
    points = np.array([[np.nan, 0., 1.], [2., np.inf, -np.inf]], dtype=np.float32)
    for color in (None, (255, 0, 40)):
        baseline = cloud_message(points, 1727241181291090775, color=color)
        actual = track2_message(points, 1727241181291090775, color=color)
        assert not actual.is_dense
        assert bytes(actual.data) == bytes(baseline.data)
        assert canonical_pointcloud_wire(serialize_message(actual))[0] == canonical_pointcloud_wire(serialize_message(baseline))[0]


def test_cdr_normalization_only_ignores_alignment_not_point_payload():
    msg = cloud_message(np.array([[1., 2., 3.]]), 1727241181291090775, color=(70, 235, 100))
    wire = serialize_message(msg)
    expected, padding = canonical_pointcloud_wire(wire)
    changed = bytearray(wire)
    assert padding
    for i in padding: changed[i] ^= 0xff
    assert canonical_pointcloud_wire(changed)[0] == expected
    # Last meaningful byte is is_dense; changing it cannot be normalized away.
    changed = bytearray(wire); changed[-1] ^= 1
    assert canonical_pointcloud_wire(changed)[0] != expected
    # Nor can a changed coordinate payload be ignored.
    changed = bytearray(wire); changed[-5] ^= 1
    assert canonical_pointcloud_wire(changed)[0] != expected
