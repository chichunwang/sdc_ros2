"""資料集讀取：以奈秒時間戳（字串，不轉 float）對齊影像與點雲，逐幀載入，不一次讀進記憶體。"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


class Dataset:
    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        if not self.dir.is_dir():
            raise FileNotFoundError('找不到資料目錄：%s（請先把官方資料放到 midterm02/data/）' % self.dir)
        ply = {p.stem: p for p in (self.dir / 'raw_ply').glob('*.ply')}
        img = {p.stem: p for p in (self.dir / 'images').glob('*.jpg')}
        sample = self.dir / 'sample_submission.csv'
        if sample.exists():
            with sample.open(newline='', encoding='utf-8') as f:
                self.timestamps = [row[0].strip() for row in list(csv.reader(f))[1:] if row]
        else:
            self.timestamps = sorted(ply, key=int)
        if len(set(self.timestamps)) != len(self.timestamps):
            raise ValueError('timestamp 有重複')
        self.timestamps = sorted(self.timestamps, key=int)  # 追蹤依時間順序進行
        missing_ply = [t for t in self.timestamps if t not in ply]
        missing_img = [t for t in self.timestamps if t not in img]
        if missing_ply:
            raise FileNotFoundError('缺少 %d 份點雲，例如 %s' % (len(missing_ply), missing_ply[:3]))
        self.missing_images = missing_img  # 缺影像的幀仍可用純幾何追蹤
        self.ply = ply
        self.img = img
        self.sample_path = sample if sample.exists() else None
        info = json.loads((self.dir / 'camera_info.json').read_text())
        self.camera_info = info
        self.K = np.array([[info['fx'], 0, info['cx']], [0, info['fy'], info['cy']], [0, 0, 1]], dtype=np.float64)
        self.size = (int(info['width']), int(info['height']))
        self.imu_path = self.dir / 'imu.csv' if (self.dir / 'imu.csv').exists() else None

    def __len__(self):
        return len(self.timestamps)

    def load_cloud(self, i):
        """回傳 (xyz float64 (N,3), rgb float32 (N,3) in [0,1])，相機座標。"""
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(self.ply[self.timestamps[i]]))
        xyz = np.asarray(pcd.points, dtype=np.float64)
        rgb = np.asarray(pcd.colors, dtype=np.float32)
        if len(rgb) != len(xyz):
            rgb = np.zeros((len(xyz), 3), dtype=np.float32)
        return xyz, rgb

    def load_image(self, i):
        """回傳 BGR uint8 影像；該幀沒有影像時回傳 None。"""
        import cv2
        path = self.img.get(self.timestamps[i])
        return None if path is None else cv2.imread(str(path), cv2.IMREAD_COLOR)

    def manifest_rows(self, indices):
        for local, i in enumerate(indices):
            t = self.timestamps[i]
            yield [local, t, 'images/%s.jpg' % t if t in self.img else '', 'raw_ply/%s.ply' % t]

    def input_digest(self):
        """輸入的輕量指紋：內參、IMU、sample 檔內容，加上所有幀檔名與大小（不重讀數 GB 的資料）。"""
        h = hashlib.sha256()
        for name in ('camera_info.json', 'imu.csv', 'sample_submission.csv'):
            p = self.dir / name
            if p.exists():
                h.update(name.encode()); h.update(p.read_bytes())
        for t in self.timestamps:
            h.update(('%s:%d:%d;' % (t, self.ply[t].stat().st_size,
                                     self.img[t].stat().st_size if t in self.img else -1)).encode())
        return h.hexdigest()
