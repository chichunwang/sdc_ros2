from types import SimpleNamespace
import numpy as np
from track3.visualization import cloud_message, RvizPublisher
from track3.geometry import rz


def test_display_cloud_matches_written_pose_and_exact_timestamp():
    raw = np.array([[1.,2.,3.],[-2.,0.,1.]])
    original = raw.copy(); T = np.eye(4)
    T[:3,:3]=rz(.3);T[:3,3]=[660.,-222.,10.]
    stamp = 1727239790258245070
    cloud = cloud_message(raw,stamp,T,3.)
    actual = np.frombuffer(cloud.data,dtype='<f4').reshape(-1,3)
    np.testing.assert_allclose(actual,raw@T[:3,:3].T*3+T[:3,3],atol=3e-5)
    np.testing.assert_array_equal(raw,original)
    assert cloud.header.frame_id=='map'
    assert cloud.header.stamp.sec*10**9+cloud.header.stamp.nanosec==stamp
    assert cloud.width==2 and cloud.point_step==12 and cloud.row_step==24


def test_ros_publisher_lifecycle_with_generated_data():
    node = RvizPublisher()
    try:
        stamp = 1727239790258245070
        raw = np.array([[1.,0.,0.],[0.,1.,0.]])
        node.publish_map(raw,stamp)
        node.publish_frame(raw,np.eye(4),3.,stamp,SimpleNamespace(center=None))
        assert len(node.path.poses)==1
        assert node.path.header.frame_id=='map'
    finally:
        node.close()
