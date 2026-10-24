"""RViz publishers only; never alter localization poses or sensor inputs."""
import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation


def cloud_message(points, timestamp_ns, pose=None, scale=1.):
    xyz = np.asarray(points, dtype=np.float64)
    if pose is not None:
        xyz = xyz @ np.asarray(pose)[:3,:3].T * scale + np.asarray(pose)[:3,3]
    xyz = np.asarray(xyz, dtype='<f4').reshape(-1,3)
    msg = PointCloud2()
    msg.header.frame_id = 'map'
    msg.header.stamp.sec = int(timestamp_ns)//1_000_000_000
    msg.header.stamp.nanosec = int(timestamp_ns)%1_000_000_000
    msg.height = 1; msg.width = len(xyz)
    msg.fields = [PointField(name=name, offset=offset, datatype=PointField.FLOAT32, count=1)
                  for name,offset in [('x',0),('y',4),('z',8)]]
    msg.is_bigendian = False; msg.point_step = 12; msg.row_step = 12*len(xyz)
    msg.is_dense = bool(np.isfinite(xyz).all()); msg.data = xyz.tobytes()
    return msg


class RvizPublisher:
    def __init__(self):
        self.context = Context(); rclpy.init(context=self.context)
        self.node = rclpy.create_node('track3_live_localization', context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.map_pub = self.node.create_publisher(PointCloud2, '/track3/map', qos)
        self.submap_pub = self.node.create_publisher(PointCloud2, '/track3/submap', qos)
        self.cloud_pub = self.node.create_publisher(PointCloud2, '/track3/current_cloud', qos)
        self.pose_pub = self.node.create_publisher(PoseStamped, '/track3/pose', 10)
        self.path_pub = self.node.create_publisher(Path, '/track3/path', qos)
        self.tf = TransformBroadcaster(self.node)
        self.path = Path(); self.path.header.frame_id = 'map'
        self.submap_center = None

    def publish_map(self, points, timestamp_ns):
        self.map_pub.publish(cloud_message(points, timestamp_ns))
        self.spin()

    def publish_frame(self, points, T, scale, timestamp_ns, registration):
        self.cloud_pub.publish(cloud_message(points, timestamp_ns, T, scale))
        if registration.center is not None and (self.submap_center is None or
                not np.array_equal(self.submap_center, registration.center)):
            import cupy as cp
            model = registration.models[min(registration.models)]
            self.submap_pub.publish(cloud_message(cp.asnumpy(model.means), timestamp_ns))
            self.submap_center = registration.center.copy()
        p = PoseStamped(); p.header.frame_id = 'map'
        p.header.stamp.sec = int(timestamp_ns)//1_000_000_000
        p.header.stamp.nanosec = int(timestamp_ns)%1_000_000_000
        p.pose.position.x, p.pose.position.y, p.pose.position.z = map(float,T[:3,3])
        q = Rotation.from_matrix(T[:3,:3]).as_quat()
        p.pose.orientation.x,p.pose.orientation.y,p.pose.orientation.z,p.pose.orientation.w = map(float,q)
        self.pose_pub.publish(p)
        self.path.header = p.header; self.path.poses.append(p); self.path_pub.publish(self.path)
        transform = TransformStamped(); transform.header = p.header
        transform.child_frame_id = 'track3_camera'
        transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z = map(float,T[:3,3])
        transform.transform.rotation = p.pose.orientation
        self.tf.sendTransform(transform)
        self.spin()

    def spin(self):
        self.executor.spin_once(timeout_sec=0.)

    def close(self):
        self.executor.shutdown(); self.node.destroy_node(); self.context.shutdown()
