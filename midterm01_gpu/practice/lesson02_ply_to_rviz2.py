#!/usr/bin/env python3
"""Publish an ASCII XYZ-first PLY file as a ROS 2 PointCloud2 topic for RViz2.

Usage (in a sourced ROS 2 shell):
    python3 lesson02_ply_to_rviz2.py /absolute/path/to/raw_ply/file.ply --step 5

Assumptions: ASCII PLY, first three vertex columns are x, y, z.
"""

import argparse
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


def load_xyz(path: Path, step: int) -> np.ndarray:
    with path.open("r", encoding="ascii") as stream:
        for line in stream:
            if line.strip() == "end_header":
                break
        else:
            raise ValueError("Invalid PLY: no end_header. This example expects ASCII PLY.")
        points = np.loadtxt(stream, usecols=(0, 1, 2), dtype=np.float32)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    return points[::step]


class PlyPointCloudPublisher(Node):
    def __init__(self, points: np.ndarray) -> None:
        super().__init__("raw_ply_publisher")
        self.publisher = self.create_publisher(PointCloud2, "/raw_ply/points", 1)
        self.cloud = point_cloud2.create_cloud_xyz32(
            Header(frame_id="raw_ply"), points.tolist()
        )
        self.timer = self.create_timer(2.0, self.publish_cloud)
        self.get_logger().info(
            f"Loaded {len(points)} points; publishing /raw_ply/points (frame_id=raw_ply)"
        )
        self.publish_cloud()

    def publish_cloud(self) -> None:
        self.cloud.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.cloud)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply_path", type=Path, help="Path to the ASCII PLY file")
    parser.add_argument("--step", type=int, default=5, help="Keep every Nth point (default: 5)")
    args = parser.parse_args()
    if args.step < 1:
        parser.error("--step must be >= 1")
    if not args.ply_path.is_file():
        parser.error(f"PLY not found inside this environment: {args.ply_path}")

    points = load_xyz(args.ply_path, args.step)
    if len(points) == 0:
        parser.error("No finite XYZ points were read")
    rclpy.init()
    node = PlyPointCloudPublisher(points)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
