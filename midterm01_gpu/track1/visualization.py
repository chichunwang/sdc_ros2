"""Track1 ROS 顯示。只讀已估計 pose，不影響濾波或配準。"""
import json
import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation
from track3.visualization import cloud_message


class Track1RvizPublisher:
    def __init__(self):
        self.context = Context(); rclpy.init(context=self.context)
        self.node = rclpy.create_node('track1_live_localization', context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        for name, msg_type in [('map', PointCloud2), ('submap', PointCloud2),
                               ('current_cloud', PointCloud2), ('pose', PoseStamped),
                               ('path', Path), ('status', String), ('registration_status', Marker)]:
            setattr(self, name+'_pub', self.node.create_publisher(msg_type, '/track1/'+name, qos))
        self.tf = TransformBroadcaster(self.node)
        self.path = Path(); self.path.header.frame_id = 'map'
        self.submap_center = None

    def publish_map(self, points, timestamp_ns):
        self.map_pub.publish(cloud_message(points, timestamp_ns))
        self.spin()

    def publish_submap(self, points, timestamp_ns):
        self.submap_pub.publish(cloud_message(points, timestamp_ns))

    def publish_frame(self, raw, T, scale, timestamp_ns, *, accepted, reason, stationary=False,
                      speed_mps=None, gyro_bias_radps=None, scale_validated=None):
        color = (70, 235, 100) if accepted else (255, 160, 40)
        self.current_cloud_pub.publish(cloud_message(raw, timestamp_ns, T, scale, color=color))
        pose = PoseStamped(); pose.header.frame_id = 'map'
        pose.header.stamp.sec, pose.header.stamp.nanosec = divmod(int(timestamp_ns), 1_000_000_000)
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, T[:3, 3])
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = map(float, q)
        self.pose_pub.publish(pose)
        self.path.header = pose.header; self.path.poses.append(pose); self.path_pub.publish(self.path)
        status = String(); status.data = json.dumps(dict(timestamp_ns=int(timestamp_ns), map_accepted=bool(accepted),
            reason=reason, stationary=bool(stationary), scale=float(scale), scale_validated=scale_validated,
            speed_mps=speed_mps, gyro_bias_radps=gyro_bias_radps))
        self.status_pub.publish(status)
        marker = Marker(); marker.header = pose.header; marker.ns = 'registration_status'; marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING; marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, T[:3, 3]+[0, 0, 3])
        marker.pose.orientation.w = 1.; marker.scale.z = 1.; marker.color.a = 1.
        marker.color.r, marker.color.g, marker.color.b = [c/255. for c in color]
        marker.text = ('MAP FUSED' if accepted else 'CORRECTION MEMORY: '+reason
                       if reason in ('ground_only_prior', 'ground_support_failed', 'temporal_innovation_rejected')
                       else 'SENSOR PRIOR: '+reason)
        marker.text += f' | scale={scale:.3f}'
        if scale_validated is False: marker.text += ' (UNVALIDATED)'
        if stationary: marker.text += ' | STATIONARY'
        self.registration_status_pub.publish(marker)
        transform = TransformStamped(); transform.header = pose.header; transform.child_frame_id = 'track1_camera'
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = map(float, T[:3, 3])
        transform.transform.rotation = pose.pose.orientation; self.tf.sendTransform(transform)
        self.spin()

    def spin(self):
        self.executor.spin_once(timeout_sec=0.)

    def close(self):
        self.executor.shutdown(); self.node.destroy_node(); self.context.shutdown()
