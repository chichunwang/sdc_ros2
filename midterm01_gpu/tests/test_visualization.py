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
        clouds=[];statuses=[];markers=[]
        node.cloud_pub=SimpleNamespace(publish=clouds.append)
        node.status_pub=SimpleNamespace(publish=statuses.append)
        node.marker_pub=SimpleNamespace(publish=markers.append)
        stamp = 1727239790258245070
        raw = np.array([[1.,0.,0.],[0.,1.,0.]])
        node.publish_map(raw,stamp)
        node.publish_frame(raw,np.eye(4),3.,stamp,SimpleNamespace(center=None),accepted=True,reason='accepted')
        node.publish_frame(raw,np.eye(4),3.,stamp+1,SimpleNamespace(center=None),
                           accepted=False,reason='low_coverage')
        assert len(node.path.poses)==2
        assert node.path.header.frame_id=='map'
        assert clouds[0].width==len(raw) and clouds[1].width==len(raw)
        accepted_rgb=np.frombuffer(clouds[0].data,dtype='<u4').reshape(-1,4)[:,3]
        prior_rgb=np.frombuffer(clouds[1].data,dtype='<u4').reshape(-1,4)[:,3]
        assert np.all(accepted_rgb==(70<<16)|(235<<8)|100)
        assert np.all(prior_rgb==(255<<16)|(160<<8)|40)
        assert markers[0].text=='ACCEPTED'
        assert markers[1].text=='PRIOR ONLY: low_coverage'
        import json
        assert json.loads(statuses[1].data)==dict(timestamp_ns=stamp+1,map_accepted=False,
                                                reason='low_coverage',scale=3.)
    finally:
        node.close()
