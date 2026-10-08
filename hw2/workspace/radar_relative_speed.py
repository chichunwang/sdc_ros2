import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


RADAR_TOPIC = '/ars548/radar_front/detections'
MARKER_TOPIC = '/radar_speed_markers'


class RadarRelativeSpeed(Node):

    def __init__(self):
        super().__init__('radar_relative_speed')

        self.subscription = self.create_subscription(
            PointCloud2,
            RADAR_TOPIC,
            self.radar_callback,
            qos_profile_sensor_data
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            MARKER_TOPIC,
            10
        )

        self.get_logger().info(
            f'Subscribed to: {RADAR_TOPIC}'
        )

        self.get_logger().info(
            f'Publishing markers to: {MARKER_TOPIC}'
        )

    def estimate_ego_speed(self, azimuth, range_rate):
        """
        靜止物體的 Radar RangeRate 大約滿足：

            RangeRate ≈ -v_ego * cos(azimuth)

        因此：

            v_ego ≈ -RangeRate / cos(azimuth)

        因為大部分 Radar detections 通常是靜止背景，
        使用 median 可以降低移動車輛 outlier 的影響。
        """

        cos_az = np.cos(azimuth)

        # 避免接近 ±90 度的 detection，
        # 因為 cos(theta) 太小會造成除法不穩定。
        mask = np.abs(cos_az) > 0.5

        if np.count_nonzero(mask) < 10:
            return 0.0

        ego_candidates = (
            -range_rate[mask] / cos_az[mask]
        )

        # 合理的車速範圍
        ego_candidates = ego_candidates[
            (ego_candidates > -5.0)
            & (ego_candidates < 50.0)
        ]

        if len(ego_candidates) < 10:
            return 0.0

        return float(np.median(ego_candidates))

    def radar_callback(self, msg):

        required_fields = (
            'x',
            'y',
            'z',
            'AzimuthAngle',
            'RangeRate',
            'ObjectID'
        )

        available_fields = {
            field.name for field in msg.fields
        }

        missing = set(required_fields) - available_fields

        if missing:
            self.get_logger().error(
                f'Missing Radar fields: {missing}'
            )
            return

        points = point_cloud2.read_points(
            msg,
            field_names=required_fields,
            skip_nans=True
        )

        if len(points) == 0:
            return

        x = points['x'].astype(np.float64)
        y = points['y'].astype(np.float64)
        z = points['z'].astype(np.float64)

        azimuth = points['AzimuthAngle'].astype(np.float64)

        range_rate = points['RangeRate'].astype(np.float64)

        object_id = points['ObjectID'].astype(np.int64)

        # ----------------------------------------------------------
        # 1. Estimate ego vehicle speed
        # ----------------------------------------------------------

        ego_speed = self.estimate_ego_speed(
            azimuth,
            range_rate
        )

        # ----------------------------------------------------------
        # 2. Predict RangeRate of stationary background
        # ----------------------------------------------------------

        stationary_rr = (
            -ego_speed * np.cos(azimuth)
        )

        # ----------------------------------------------------------
        # 3. Remove stationary background
        #
        # residual ≈ 0:
        #   很可能是靜止背景
        #
        # residual 大:
        #   很可能物體本身正在移動
        # ----------------------------------------------------------

        residual = (
            range_rate - stationary_rr
        )

        # 約 5.4 km/h
        MOVING_THRESHOLD = 1.5

        moving_mask = (
            np.abs(residual) > MOVING_THRESHOLD
        )

        # 只顯示比較合理的道路區域
        roi_mask = (
            (x > 2.0)
            & (x < 100.0)
            & (np.abs(y) < 20.0)
            & (z > -5.0)
            & (z < 5.0)
        )

        valid_mask = moving_mask & roi_mask

        if np.count_nonzero(valid_mask) == 0:
            self.publish_empty(msg)
            return

        # ----------------------------------------------------------
        # 4. Group detections by ObjectID
        # ----------------------------------------------------------

        marker_array = MarkerArray()

        valid_ids = np.unique(
            object_id[valid_mask]
        )

        marker_id = 0

        for oid in valid_ids:

            # 0 / 65535 常被拿來表示 invalid / unassociated
            if oid == 0 or oid == 65535:
                continue

            object_mask = (
                valid_mask
                & (object_id == oid)
            )

            count = np.count_nonzero(object_mask)

            # 只有一個 detection 很容易是雜訊
            if count < 2:
                continue

            obj_x = float(
                np.median(x[object_mask])
            )

            obj_y = float(
                np.median(y[object_mask])
            )

            obj_z = float(
                np.median(z[object_mask])
            )

            obj_range_rate = float(
                np.median(range_rate[object_mask])
            )

            obj_residual = float(
                np.median(residual[object_mask])
            )

            # ------------------------------------------------------
            # 額外檢查：
            # 如果整個 object 的 residual 不夠大，就不要顯示
            # ------------------------------------------------------

            if abs(obj_residual) < MOVING_THRESHOLD:
                continue

            # Radar RangeRate 是 m/s
            # 轉成 km/h
            relative_speed_kmh = (
                obj_range_rate * 3.6
            )

            marker = Marker()

            marker.header = msg.header

            marker.ns = 'radar_relative_speed'
            marker.id = marker_id

            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD

            marker.pose.position.x = obj_x
            marker.pose.position.y = obj_y

            # 文字放在 detection 上方
            marker.pose.position.z = obj_z + 2.0

            marker.pose.orientation.w = 1.0

            marker.scale.z = 1.2

            # 白色文字
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            marker.text = (
                f'{relative_speed_kmh:+.1f} km/h'
            )

            # 如果之後沒有更新，自動消失
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 300000000

            marker_array.markers.append(marker)

            marker_id += 1

        # ----------------------------------------------------------
        # 顯示一些 debug 資訊
        # ----------------------------------------------------------

        self.get_logger().info(
            f'Ego speed estimate: '
            f'{ego_speed * 3.6:.1f} km/h | '
            f'Moving objects: {marker_id}'
        )

        self.marker_pub.publish(marker_array)

    def publish_empty(self, msg):

        marker_array = MarkerArray()

        marker = Marker()

        marker.header = msg.header
        marker.action = Marker.DELETEALL

        marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)


def main(args=None):

    rclpy.init(args=args)

    node = RadarRelativeSpeed()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()