import json
import os

import numpy as np
import open3d as o3d
import pandas as pd


def load_ply(file, color=False):
    """image point cloud (camera frame, unknown scale) -> (N, 3), or ((N, 3), (N, 3) RGB in [0, 1]) with color=True"""
    pcd = o3d.io.read_point_cloud(file)
    if color:
        return np.asarray(pcd.points), np.asarray(pcd.colors)
    return np.asarray(pcd.points)


def load_folder(folder):
    files = sorted(os.listdir(folder))
    return [os.path.join(folder, f) for f in files if f.endswith(".ply")]


def load_camera_info(file):
    """intrinsics K (3x3) and image size (width, height)"""
    c = json.load(open(file))
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]])
    return K, (c["width"], c["height"])


def load_imu(file):
    """IMU as recorded (Track1 only): timestamp [ns], orientation, angular_velocity [rad/s], linear_acceleration [m/s^2]"""
    return pd.read_csv(file, dtype={"timestamp": np.int64})
