"""Track2 官方前相機 ego mask；沿用共用投影規則，點雲留在 CUDA。"""
from pathlib import Path
import cv2
import numpy as np
import cupy as cp
import yaml
from track3.io import sha256


def device_mask(points, K, valid_mask):
    """與共用相機遮罩相同的投影；保留 GPU 點雲供配準直接使用。"""
    raw = cp.asarray(points, dtype=cp.float64)
    finite = cp.all(cp.isfinite(raw), axis=1) & (raw[:, 0] > 0)
    safe = cp.where(finite, raw[:, 0], 1.)
    u = cp.rint(K[0, 2]-K[0, 0]*raw[:, 1]/safe).astype(cp.int64)
    v = cp.rint(K[1, 2]-K[1, 1]*raw[:, 2]/safe).astype(cp.int64)
    h, w = valid_mask.shape
    inside = finite & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    allowed = cp.asarray(valid_mask)[cp.clip(v, 0, h-1), cp.clip(u, 0, w-1)] > 0
    kept = raw[inside & allowed]
    return kept, dict(input_points=len(raw), kept_points=len(kept),
        removed_points=len(raw)-len(kept), method='provided_front_ego_mask_and_rectified_FOV')


class FrontMask:
    def __init__(self, data):
        folder = Path(data)/'Track2_images/camera_info'
        info = folder/'gige_100_f_hdr_camera_info.yaml'
        mask_path = folder/'gige_100_f_hdr_mask.png'
        c = yaml.safe_load(info.read_text())
        K = np.asarray(c['camera_matrix']['data']).reshape(3, 3)
        D = np.asarray(c['distortion_coefficients']['data'])
        self.K = np.asarray(c['projection_matrix']['data']).reshape(3, 4)[:, :3].copy()
        size = (c['image_width'], c['image_height'])
        maps = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), self.K, size, cv2.CV_32FC1)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError('Missing Track2 official front mask')
        rectified = cv2.remap(mask, *maps, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        self.mask = cv2.erode(np.where(rectified == 0, 255, 0).astype('uint8'), np.ones((7, 7), dtype='uint8'))
        self.device_mask = cp.asarray(self.mask)
        self.sources = {str(f): sha256(f) for f in (info, mask_path)}

    def apply(self, raw):
        return device_mask(raw, self.K, self.device_mask)
