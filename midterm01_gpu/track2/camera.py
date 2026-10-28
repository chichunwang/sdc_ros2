"""Track2 官方前相機 ego mask；重用 Track3 的 CUDA 投影實作。"""
from pathlib import Path
import cv2
import numpy as np
import yaml
from track3.io import sha256
from track3.visual_prior import filter_by_camera_mask


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
        self.sources = {str(f): sha256(f) for f in (info, mask_path)}

    def apply(self, raw):
        return filter_by_camera_mask(raw, self.K, self.mask)
