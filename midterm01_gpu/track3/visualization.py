"""RViz publishers only; never alter localization poses or sensor inputs."""
import json
import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation


def cloud_message(points, timestamp_ns, pose=None, scale=1., color=None):
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
    msg.is_bigendian = False
    msg.is_dense = bool(np.isfinite(xyz).all())
    if color is None:
        msg.point_step=12;msg.data=xyz.tobytes()
    else:
        packed=np.empty((len(xyz),4),dtype='<f4');packed[:,:3]=xyz
        rgb=(int(color[0])<<16)|(int(color[1])<<8)|int(color[2])
        packed[:,3]=np.asarray(rgb,dtype='<u4').view('<f4')
        msg.fields.append(PointField(name='rgb',offset=12,datatype=PointField.FLOAT32,count=1))
        msg.point_step=16;msg.data=packed.tobytes()
    msg.row_step=msg.point_step*len(xyz)
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
        self.status_pub = self.node.create_publisher(String, '/track3/status', qos)
        self.marker_pub = self.node.create_publisher(Marker, '/track3/registration_status', qos)
        self.tf = TransformBroadcaster(self.node)
        self.path = Path(); self.path.header.frame_id = 'map'
        self.submap_center = None

    def publish_map(self, points, timestamp_ns):
        self.map_pub.publish(cloud_message(points, timestamp_ns))
        self.spin()

    def publish_frame(self, points, T, scale, timestamp_ns, registration, *, accepted, reason):
        # Show the current frame at its emitted pose in both cases. Orange
        # explicitly identifies a prior-only cloud, useful for overlap diagnosis.
        stationary=reason=='stationary_visual_imu_hold'
        color=(80,190,255) if stationary else (70,235,100) if accepted else (255,160,40)
        self.cloud_pub.publish(cloud_message(points,timestamp_ns,T,scale,color=color))
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
        status = String()
        status.data = json.dumps(dict(timestamp_ns=int(timestamp_ns), map_accepted=bool(accepted),
                                      reason=reason, scale=float(scale)))
        self.status_pub.publish(status)
        marker = Marker(); marker.header = p.header
        marker.ns = 'registration_status'; marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING; marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, T[:3, 3]+[0, 0, 3])
        marker.pose.orientation.w = 1.; marker.scale.z = 1.
        marker.color.r = 0.2 if accepted else 1.
        marker.color.g = 1. if accepted else 0.5
        marker.color.b = 0.2 if accepted else 0.1; marker.color.a = 1.
        if stationary:marker.color.r,marker.color.g,marker.color.b=.3,.75,1.
        marker.text = 'STATIONARY (IMAGE + IMU)' if stationary else 'ACCEPTED' if accepted else f'PRIOR ONLY: {reason}'
        self.marker_pub.publish(marker)
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
