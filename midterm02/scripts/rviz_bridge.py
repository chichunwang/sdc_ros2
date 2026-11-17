#!/usr/bin/env python3
"""RViz bridge：接收 SLAM 的 TCP 串流（vslam/gslam/viz_stream.py），發佈成 ROS 2 topic 給 RViz。

在 ROS 2 Humble 容器 midterm02_rviz 內執行（由 scripts/dev.sh rviz 啟動），不在 SLAM 容器內。只做顯示，不回傳任何東西給 SLAM。

座標：SLAM 的世界座標是第一幀的 OpenCV 相機軸（x 右、y 下、z 前），直接看會上下顛倒。
這裡發佈靜態 TF map → world_cv，把它轉成 RViz 慣用的 x 前、y 左、z 上；所有資料都掛在 world_cv 或 camera。

Topics（frame）：
  /vslam/camera/image_raw、/vslam/camera/camera_info（camera）  原始影像，供 RViz Camera 顯示把 3D 疊到影像上
  /vslam/features/image                                        特徵點、匹配與狀態文字
  /vslam/current_cloud（camera）    目前幀的彩色點雲
  /vslam/submap（world_cv）         配準用的局部子圖
  /vslam/map（world_cv）            所有 keyframe 依目前 pose 拼成的彩色地圖（閉環後會重排）
  /vslam/path_live、/vslam/path_odometry、/vslam/path_optimized
  /vslam/keyframes（PoseArray）、/vslam/markers（閉環邊、相機視錐）
  TF：map → world_cv → camera

執行緒（2026-11-17）：接收執行緒只做便宜的事（目前影像／點雲／TF、記下 keyframe 與位姿），
路徑與整張地圖改由定時執行緒發佈（路徑每 PATH_PERIOD_S 秒、地圖每 MAP_PERIOD_S 秒最多一次，點數上限 MAP_MAX_POINTS）。
以前每 10 幀就在接收執行緒裡重組並發佈整張地圖：Track2 到 1000 多個 keyframe 時一次要約 8 秒，
這段時間不讀 socket，SLAM 的送出就被卡住，整體慢了一倍以上。
"""
import json
import socket
import struct
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, TransformStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

PORT = 5600
WORLD, CAMERA = 'world_cv', 'camera'
# p_map = M · p_cv：map 的 x = cv 的 z（前），y = −x（左），z = −y（上）
M_MAP_CV = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
PATH_PERIOD_S = 1.0          # 路徑（live／odometry）重發間隔
MAP_PERIOD_S = 5.0           # 整張地圖重組的最短間隔（閉環修正後也遵守）
MAP_MAX_POINTS = 3_000_000   # 地圖點數上限：超過時每個 keyframe 等間隔取樣（只影響顯示）


def quaternion(R):
    """旋轉矩陣 -> (x, y, z, w)。"""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
    return [float(v) for v in q]


def to_pose(T):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in T[:3, 3])
    q = quaternion(T[:3, :3])
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
    return pose


def read_exact(conn, n):
    chunks = []
    while n:
        data = conn.recv(min(n, 1 << 20))
        if not data:
            raise ConnectionError('closed')
        chunks.append(data)
        n -= len(data)
    return b''.join(chunks)


def read_message(conn):
    header = json.loads(read_exact(conn, struct.unpack('>I', read_exact(conn, 4))[0]))
    arrays = {}
    for spec in header['arrays']:
        dtype = np.dtype(spec['dtype'])
        count = int(np.prod(spec['shape'])) if spec['shape'] else 1
        arrays[spec['name']] = np.frombuffer(read_exact(conn, count * dtype.itemsize), dtype=dtype).reshape(spec['shape'])
    return header['type'], header['meta'], arrays


class Bridge(Node):
    def __init__(self):
        super().__init__('vslam_rviz_bridge')
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_image = self.create_publisher(Image, '/vslam/camera/image_raw', 5)
        self.pub_info = self.create_publisher(CameraInfo, '/vslam/camera/camera_info', 5)
        self.pub_feat = self.create_publisher(Image, '/vslam/features/image', 5)
        self.pub_cloud = self.create_publisher(PointCloud2, '/vslam/current_cloud', 5)
        self.pub_submap = self.create_publisher(PointCloud2, '/vslam/submap', latched)
        self.pub_map = self.create_publisher(PointCloud2, '/vslam/map', latched)
        self.pub_live = self.create_publisher(Path, '/vslam/path_live', latched)
        self.pub_odom = self.create_publisher(Path, '/vslam/path_odometry', latched)
        self.pub_opt = self.create_publisher(Path, '/vslam/path_optimized', latched)
        self.pub_kf = self.create_publisher(PoseArray, '/vslam/keyframes', latched)
        self.pub_marker = self.create_publisher(MarkerArray, '/vslam/markers', latched)
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        static = TransformStamped()
        static.header.stamp = self.get_clock().now().to_msg()
        static.header.frame_id, static.child_frame_id = 'map', WORLD
        q = quaternion(M_MAP_CV)
        static.transform.rotation.x, static.transform.rotation.y, static.transform.rotation.z, static.transform.rotation.w = q
        self.static_tf.sendTransform(static)
        self.lock = threading.Lock()
        self.reset(None)
        threading.Thread(target=self.serve, daemon=True).start()
        threading.Thread(target=self.publish_loop, daemon=True).start()
        self.get_logger().info('waiting for SLAM on tcp port %d' % PORT)

    # ------------------------------------------------------------ 狀態
    def reset(self, meta):
        self.final_sent = False
        self.info = meta
        self.kf_clouds = {}            # kf_id -> (xyz, rgb)（相機座標）
        self.kf_poses = np.zeros((0, 4, 4))
        self.live, self.odom = [], []
        self.map_dirty = False
        self.path_dirty = False

    def serve(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', PORT))
        server.listen(1)
        while rclpy.ok():
            conn, addr = server.accept()
            self.get_logger().info('SLAM connected from %s' % (addr,))
            try:
                while rclpy.ok():
                    kind, meta, arrays = read_message(conn)
                    getattr(self, 'on_' + kind)(meta, arrays)
            except ConnectionError:
                self.get_logger().info('SLAM disconnected; display keeps the last state')
            except Exception as exc:  # noqa: BLE001  顯示端出錯不應中斷，記錄後等下一次連線
                self.get_logger().error('bridge error: %r' % exc)
            finally:
                conn.close()

    # ------------------------------------------------------------ 訊息轉換
    def header(self, frame):
        from std_msgs.msg import Header
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = frame
        return h

    def cloud_msg(self, frame, xyz, rgb=None, stamp=None):
        msg = PointCloud2()
        msg.header = self.header(frame)
        if stamp is not None:
            msg.header.stamp = stamp
        n = len(xyz)
        msg.height, msg.width = 1, n
        msg.is_dense, msg.is_bigendian = True, False
        fields = [PointField(name=a, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, a in enumerate('xyz')]
        data = np.zeros((n, 4), dtype=np.float32)
        data[:, :3] = xyz
        if rgb is not None:
            packed = (rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2].astype(np.uint32)
            data[:, 3] = packed.view(np.float32)
            fields.append(PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1))
        msg.fields = fields
        msg.point_step, msg.row_step = 16, 16 * n
        msg.data = data.tobytes()
        return msg

    def image_msg(self, bgr, stamp):
        msg = Image()
        msg.header = self.header(CAMERA)
        msg.header.stamp = stamp
        msg.height, msg.width = int(bgr.shape[0]), int(bgr.shape[1])
        msg.encoding, msg.is_bigendian, msg.step = 'bgr8', 0, int(bgr.shape[1]) * 3
        msg.data = bgr.tobytes()
        return msg

    def path_msg(self, poses):
        msg = Path()
        msg.header = self.header(WORLD)
        for T in poses:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose = to_pose(T)
            msg.poses.append(ps)
        return msg

    # ------------------------------------------------------------ 處理各類訊息
    def on_reset(self, meta, arrays):
        self.reset(meta)
        clear = Marker()
        clear.action = Marker.DELETEALL
        self.pub_marker.publish(MarkerArray(markers=[clear]))
        empty = np.zeros((0, 3), dtype=np.float32)
        # 地圖的空訊息也要帶 rgb 欄位：RViz 收到沒有 rgb 的點雲會把上色方式從 RGB8 改成純色（變成全白）
        self.pub_map.publish(self.cloud_msg(WORLD, empty, np.zeros((0, 3), dtype=np.uint8)))
        self.pub_submap.publish(self.cloud_msg(WORLD, empty))
        for pub in (self.pub_live, self.pub_odom, self.pub_opt):
            pub.publish(self.path_msg([]))

    def on_frame(self, meta, a):
        stamp = self.get_clock().now().to_msg()
        T = a['T']
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id, tf.child_frame_id = WORLD, CAMERA
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = (float(v) for v in T[:3, 3])
        q = quaternion(T[:3, :3])
        tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = q
        self.tf.sendTransform(tf)
        if 'image' in a:
            self.pub_image.publish(self.image_msg(a['image'], stamp))
            self.pub_feat.publish(self.image_msg(a['annotated'], stamp))
            K = np.array(self.info['K'])
            sx = a['image'].shape[1] / float(self.info['width'])
            sy = a['image'].shape[0] / float(self.info['height'])
            info = CameraInfo()
            info.header = self.header(CAMERA)
            info.header.stamp = stamp
            info.height, info.width = int(a['image'].shape[0]), int(a['image'].shape[1])
            info.distortion_model = 'plumb_bob'
            info.d = [0.0] * 5
            fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy
            info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            self.pub_info.publish(info)
        self.pub_cloud.publish(self.cloud_msg(CAMERA, a['cloud_xyz'], a['cloud_rgb'], stamp))
        with self.lock:
            self.live.append(T.copy())
            self.odom.append(a['odom_T'].copy())
            self.path_dirty = True
        self.publish_frustum()

    def on_keyframe(self, meta, a):
        with self.lock:
            self.kf_clouds[meta['kf_id']] = (a['xyz'].copy(), a['rgb'].copy())

    def on_graph(self, meta, a):
        with self.lock:
            self.kf_poses = a['kf_poses'].copy()
            self.map_dirty = True
        poses = PoseArray()
        poses.header = self.header(WORLD)
        poses.poses = [to_pose(T) for T in self.kf_poses]
        self.pub_kf.publish(poses)
        if 'submap' in a:
            self.pub_submap.publish(self.cloud_msg(WORLD, a['submap']))
        marker = Marker()
        marker.header = self.header(WORLD)
        marker.ns, marker.id, marker.type, marker.action = 'loops', 0, Marker.LINE_LIST, Marker.ADD
        marker.scale.x = 0.08
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.1, 1.0, 0.2, 1.0
        marker.pose.orientation.w = 1.0
        for c, q in a['loops']:
            for k in (int(c), int(q)):
                p = self.kf_poses[k][:3, 3]
                marker.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
        self.pub_marker.publish(MarkerArray(markers=[marker]))

    def on_final(self, meta, a):
        with self.lock:
            self.path_dirty = False      # 最終路徑由這裡發，定時執行緒不要再用 live／odom 蓋掉
            self.final_sent = True
        self.pub_odom.publish(self.path_msg(a['odom']))
        self.pub_opt.publish(self.path_msg(a['optimized']))
        self.publish_map()

    def publish_loop(self):
        """定時執行緒：路徑與地圖的重發不佔用接收執行緒。"""
        last_path = last_map = 0.0
        while rclpy.ok():
            time.sleep(0.2)
            now = time.time()
            try:
                if self.path_dirty and now - last_path >= PATH_PERIOD_S and not getattr(self, 'final_sent', False):
                    with self.lock:
                        live, odom = list(self.live), list(self.odom)
                        self.path_dirty = False
                    self.pub_live.publish(self.path_msg(live))
                    self.pub_odom.publish(self.path_msg(odom))
                    last_path = now
                if self.map_dirty and now - last_map >= MAP_PERIOD_S:
                    self.publish_map()
                    last_map = time.time()
            except Exception as exc:  # noqa: BLE001  顯示端出錯不應中斷
                self.get_logger().error('publish error: %r' % exc)

    def publish_map(self):
        with self.lock:
            items = sorted(self.kf_clouds.items())
            poses = self.kf_poses
            self.map_dirty = False
        items = [(k, p, c) for k, (p, c) in items if k < len(poses)]
        total = sum(len(p) for _, p, _ in items)
        step = max(1, int(np.ceil(total / MAP_MAX_POINTS)))
        xyz, rgb = [], []
        for k, p, c in items:
            T = poses[k]
            xyz.append(p[::step] @ T[:3, :3].T.astype(np.float32) + T[:3, 3].astype(np.float32))
            rgb.append(c[::step])
        if xyz:
            self.pub_map.publish(self.cloud_msg(WORLD, np.concatenate(xyz), np.concatenate(rgb)))

    def publish_frustum(self):
        """相機視錐（在 camera frame，跟著 TF 移動）。"""
        if self.info is None:
            return
        K, w, h, depth = np.array(self.info['K']), self.info['width'], self.info['height'], 1.5
        corners = [np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0]) * depth
                   for u, v in ((0, 0), (w, 0), (w, h), (0, h))]
        marker = Marker()
        marker.header = self.header(CAMERA)
        marker.ns, marker.id, marker.type, marker.action = 'frustum', 1, Marker.LINE_LIST, Marker.ADD
        marker.scale.x = 0.05
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.85, 0.1, 1.0
        marker.pose.orientation.w = 1.0
        origin = np.zeros(3)
        segments = [(origin, c) for c in corners] + [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
        for a, b in segments:
            marker.points.append(Point(x=float(a[0]), y=float(a[1]), z=float(a[2])))
            marker.points.append(Point(x=float(b[0]), y=float(b[1]), z=float(b[2])))
        self.pub_marker.publish(MarkerArray(markers=[marker]))


def main():
    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
