"""點雲快取讀取與前處理（縮放、裁範圍、voxel、地面分離）。"""
import numpy as np

from .mapio import voxel_downsample


class PlyCache:
    """快取目錄：<stem>_xyz.npy（mmap）、<stem>_offsets.npy、<stem>_ts.npy；多程序共用，不各自載入。"""

    def __init__(self, stem):
        stem = str(stem)
        if stem.endswith('.npz'):
            stem = stem[:-4]
        self.xyz = np.load(stem + '_xyz.npy', mmap_mode='r')
        self.offsets = np.load(stem + '_offsets.npy')
        self.ts = np.load(stem + '_ts.npy')

    def __len__(self):
        return len(self.offsets) - 1

    def get(self, i):
        a, b = self.offsets[i], self.offsets[i + 1]
        return np.array(self.xyz[a:b], dtype=np.float64), None


def ground_level_m(xyz_m):
    """公尺座標的地面高度：車前 4–17 m、|y|<8 m 的點取最低 10% 的中位數（兩種點雲共用）。"""
    g = xyz_m[(xyz_m[:, 0] > 4.0) & (xyz_m[:, 0] < 17.0) & (np.abs(xyz_m[:, 1]) < 8.0)]
    if len(g) < 20:
        g = xyz_m
    k = max(10, len(g) // 10)
    return float(np.median(np.sort(g[:, 2])[:k]))


def ground_level(xyz_raw, scale=8.7):
    """舊介面：原始單位的地面高度（內部換算成公尺再算）。"""
    return ground_level_m(xyz_raw * scale) / scale


def preprocess(xyz_raw, scale, voxel=0.2, min_range=1.0, max_range=40.0, ground_margin=0.3):
    """縮放成公尺 → 範圍裁切 → voxel；回傳 (xyz_m, is_ground)。is_ground 依離地高度判斷。"""
    p = xyz_raw * scale
    g0 = ground_level_m(p)
    r = np.linalg.norm(p, axis=1)
    p = p[(r >= min_range) & (r <= max_range)]
    if len(p) == 0:
        return p, np.zeros(0, bool)
    ds, _, _ = voxel_downsample(p, voxel)
    is_ground = ds[:, 2] < g0 + ground_margin
    return ds, is_ground
