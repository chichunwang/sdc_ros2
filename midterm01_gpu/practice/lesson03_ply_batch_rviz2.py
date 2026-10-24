#!/usr/bin/env python3
"""Publish all ASCII XYZ-first PLY files in a directory to RViz2.

Modes:
  sequence: publish one file at a time, sorted by filename, looping from start.
  merge:    combine sampled points from every file into one cloud for PREVIEW ONLY.

WARNING: merge does not register point clouds or transform their coordinate frames.
Only use its geometry as a map if all input points already share a metric frame.

Example (ROS 2 Humble):
  python3 lesson03_ply_batch_rviz2.py /workspace/midterm01/data/Track3/raw_ply --mode sequence --rate 1
  python3 lesson03_ply_batch_rviz2.py /workspace/midterm01/data/Track3/raw_ply --mode merge --max-total-points 200000
"""

import argparse
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


def read_xyz(path: Path, step: int = 5, cap: int | None = None) -> np.ndarray:
    """Read ASCII PLY vertex XYZ, assuming the first 3 properties are x,y,z."""
    with path.open("r", encoding="ascii") as stream:
        fmt = None
        vertex_count = None
        properties = []
        within_vertex = False
        first = stream.readline().strip()
        if first != "ply":
            raise ValueError(f"{path.name}: missing 'ply' header")
        for raw_line in stream:
            line = raw_line.strip()
            parts = line.split()
            if line == "end_header":
                break
            if parts[:1] == ["format"]:
                fmt = parts[1] if len(parts) > 1 else None
            elif len(parts) >= 3 and parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                within_vertex = True
            elif parts[:1] == ["element"]:
                within_vertex = False
            elif within_vertex and parts[:1] == ["property"]:
                properties.append(parts[-1])
        else:
            raise ValueError(f"{path.name}: missing end_header")

        if fmt != "ascii":
            raise ValueError(f"{path.name}: only ASCII PLY is supported (got {fmt})")
        if vertex_count is None:
            raise ValueError(f"{path.name}: vertex count not found")
        if properties[:3] != ["x", "y", "z"]:
            raise ValueError(
                f"{path.name}: expected first three vertex fields x y z, got {properties[:3]}"
            )
        if vertex_count == 0:
            return np.empty((0, 3), dtype=np.float32)

        xyz = np.loadtxt(
            stream,
            usecols=(0, 1, 2),
            dtype=np.float32,
            max_rows=vertex_count,
            ndmin=2,
        )

    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    xyz = xyz[::step]
    # In merge mode, sample all files fairly within the total point budget.
    if cap is not None and len(xyz) > cap:
        indices = np.linspace(0, len(xyz) - 1, cap, dtype=np.int64)
        xyz = xyz[indices]
    return xyz


class BatchPlyPublisher(Node):
    def __init__(self, args: argparse.Namespace, paths: list[Path]) -> None:
        super().__init__("batch_ply_publisher")
        self.args = args
        self.paths = paths
        self.publisher = self.create_publisher(PointCloud2, args.topic, 10)
        self.index = 0
        self.last_cloud = None

        self.get_logger().info(
            f"Found {len(paths)} PLY files; mode={args.mode}; topic={args.topic}; "
            f"frame_id={args.frame_id}"
        )
        if args.mode == "merge":
            self.build_merged_cloud()
            self.timer = self.create_timer(2.0, self.publish_merged)
            self.publish_merged()
        else:
            self.timer = self.create_timer(1.0 / args.rate, self.publish_next)
            self.publish_next()

    def make_cloud(self, xyz: np.ndarray) -> PointCloud2:
        return point_cloud2.create_cloud_xyz32(
            Header(frame_id=self.args.frame_id), xyz.tolist()
        )

    def publish_merged(self) -> None:
        self.last_cloud.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.last_cloud)

    def build_merged_cloud(self) -> None:
        cap_per_file = max(1, self.args.max_total_points // len(self.paths))
        chunks = []
        for i, path in enumerate(self.paths, 1):
            xyz = read_xyz(path, step=self.args.step, cap=cap_per_file)
            if len(xyz):
                chunks.append(xyz)
            if i % 100 == 0 or i == len(self.paths):
                self.get_logger().info(f"Processed {i}/{len(self.paths)} files")
        if not chunks:
            raise ValueError("All PLY files are empty after filtering")
        merged = np.concatenate(chunks, axis=0)
        self.last_cloud = self.make_cloud(merged)
        self.get_logger().warn(
            "MERGE IS ONLY A VISUAL PREVIEW: no pose/scale alignment was applied."
        )
        self.get_logger().info(
            f"Merged {len(chunks)} files into {len(merged)} sampled points "
            f"(budget={self.args.max_total_points})."
        )

    def publish_next(self) -> None:
        # Loops over the sequence to let RViz2 show every frame repeatedly.
        path = self.paths[self.index]
        xyz = read_xyz(path, step=self.args.step)
        if len(xyz):
            msg = self.make_cloud(xyz)
            msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(msg)
            self.get_logger().info(
                f"Frame {self.index + 1}/{len(self.paths)}: {path.name}, "
                f"{len(xyz)} points"
            )
        else:
            self.get_logger().warn(f"Skipping empty frame: {path.name}")
        self.index = (self.index + 1) % len(self.paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply_dir", type=Path, help="Folder containing *.ply")
    parser.add_argument("--mode", choices=["sequence", "merge"], default="sequence")
    parser.add_argument("--rate", type=float, default=1.0, help="FPS for sequence mode")
    parser.add_argument("--step", type=int, default=5, help="Keep every Nth XYZ point")
    parser.add_argument(
        "--max-total-points", type=int, default=200000,
        help="Merge mode point budget; samples from EACH file (default: 200000)",
    )
    parser.add_argument("--limit-files", type=int, default=0, help="0 = all files")
    parser.add_argument("--topic", default="/raw_ply/batch_points")
    parser.add_argument("--frame-id", default="raw_ply")
    args = parser.parse_args()

    if not args.ply_dir.is_dir():
        parser.error(f"PLY directory not found: {args.ply_dir}")
    if args.rate <= 0 or args.step < 1 or args.max_total_points < 1 or args.limit_files < 0:
        parser.error("rate must be >0; step/max-total-points >=1; limit-files >=0")
    paths = sorted(args.ply_dir.glob("*.ply"))
    if args.limit_files:
        paths = paths[: args.limit_files]
    if not paths:
        parser.error(f"No *.ply files in {args.ply_dir}")
    if args.mode == "merge" and len(paths) > args.max_total_points:
        parser.error("Increase --max-total-points to at least the number of files")

    rclpy.init()
    node = None
    try:
        node = BatchPlyPublisher(args, paths)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
