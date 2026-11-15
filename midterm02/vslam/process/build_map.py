"""Stitch the image point clouds with your camera poses into one map, to check your trajectory visually
(walls should line up, revisited places should overlap). The map is not submitted.
Usage: python process/build_map.py [Track1] [--every N] [--voxel V]"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.io import load_ply

ap = argparse.ArgumentParser()
ap.add_argument("target", nargs="?", default="Track1")
ap.add_argument("--every", type=int, default=5, help="use every N-th image")
ap.add_argument("--voxel", type=float, default=0.1, help="voxel size of the output map")
args = ap.parse_args()

df = pd.read_csv(f"{args.target}/result/submission.csv", dtype={"timestamp": str})
poses = df.iloc[:, 1:].to_numpy(np.float64).reshape(-1, 4, 4)

pts, rgb = [], []
for t, T in list(zip(df["timestamp"], poses))[:: args.every]:
    p, c = load_ply(f"{args.target}/data/raw_ply/{t}.ply", color=True)
    pts.append(p @ T[:3, :3].T + T[:3, 3])
    rgb.append(c)
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts)))
pcd.colors = o3d.utility.Vector3dVector(np.concatenate(rgb))
pcd = pcd.voxel_down_sample(args.voxel)
o3d.io.write_point_cloud(f"{args.target}/result/map.ply", pcd)
print(f"map: {len(pcd.points)} points -> {args.target}/result/map.ply")
o3d.visualization.draw_geometries([pcd])
