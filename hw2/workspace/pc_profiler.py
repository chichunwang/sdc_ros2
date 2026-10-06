import argparse

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class PointCloudProfiler(Node):

    def __init__(self, topic_name):
        super().__init__('pc_profiler')

        self.topic_name = topic_name

        self.subscription = self.create_subscription(
            PointCloud2,
            self.topic_name,
            self.pointcloud_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            f'Subscribed to: {self.topic_name}'
        )

    def pointcloud_callback(self, msg):
        # 檢查 PointCloud2 是否包含 x, y, z
        available_fields = [field.name for field in msg.fields]

        required_fields = {'x', 'y', 'z'}
        missing_fields = required_fields - set(available_fields)

        if missing_fields:
            self.get_logger().error(
                f'Missing fields: {missing_fields}. '
                f'Available fields: {available_fields}'
            )
            return

        # 從 PointCloud2 取出 x, y, z
        points = point_cloud2.read_points_numpy(
            msg,
            field_names=['x', 'y', 'z'],
            skip_nans=True
        )

        # 原始 PointCloud2 中的點數
        total_points = msg.width * msg.height

        # 移除 NaN 後有效的 XYZ 點數
        valid_points = len(points)

        # Timestamp
        timestamp = (
            msg.header.stamp.sec
            + msg.header.stamp.nanosec * 1e-9
        )

        print('\n' + '=' * 60)
        print(f'Topic: {self.topic_name}')
        print(f'Timestamp: {timestamp:.9f}')
        print(f'Total points: {total_points}')
        print(f'Valid XYZ points: {valid_points}')

        if valid_points == 0:
            print('No valid XYZ points in this frame.')
            print('=' * 60)
            return

        # XYZ 各欄
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        print(f'X range: [{np.min(x):.3f}, {np.max(x):.3f}] m')
        print(f'Y range: [{np.min(y):.3f}, {np.max(y):.3f}] m')
        print(f'Z range: [{np.min(z):.3f}, {np.max(z):.3f}] m')
        print('=' * 60)


def main():
    parser = argparse.ArgumentParser(
        description='ROS 2 PointCloud2 profiler'
    )

    parser.add_argument(
        'topic',
        help='PointCloud2 topic to subscribe to'
    )

    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    node = PointCloudProfiler(args.topic)

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()