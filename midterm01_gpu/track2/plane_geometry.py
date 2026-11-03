"""CUDA 局部平面快取；沒有可靠二維鄰域的 LiDAR 點不產生法向量測。"""
import numpy as np
import cupy as cp
from .guided import KNN
from . import parameters as p


class PlaneGeometry:
    def __init__(self, nn):
        self.nn = nn
        n = len(nn.points)
        self.done = cp.zeros(n, dtype=cp.bool_)
        self.reliable = cp.zeros(n, dtype=cp.bool_)
        self.confidence = cp.zeros(n, dtype=cp.float64)
        self.normal = cp.zeros((n, 3), dtype=cp.float64)
        self.center = cp.zeros((n, 3), dtype=cp.float64)
        self.thickness_variance = cp.zeros(n, dtype=cp.float64)
        self.line_reliable = cp.zeros(n, dtype=cp.bool_)
        self.line_direction = cp.zeros((n, 3), dtype=cp.float64)
        self.line_confidence = cp.zeros(n, dtype=cp.float64)

    def ensure(self, ids):
        ids = cp.unique(ids[ids >= 0]); ids = ids[~self.done[ids]]
        if not len(ids):
            return
        k = p.PLANE_NEIGHBORS
        if not 6 <= k <= 16:
            raise ValueError('CUDA plane neighbors must be 6..16')
        query = cp.ascontiguousarray(self.nn.points[ids])
        neighbors = cp.empty((len(ids), k), dtype=cp.int64)
        KNN(((len(ids)+127)//128,), (128,), (query, np.int32(len(ids)), self.nn.points,
            self.nn.key, self.nn.start, self.nn.end, np.int32(len(self.nn.key)),
            np.float64(self.nn.cell), np.float64(p.PLANE_RADIUS_M), np.int32(k), neighbors))
        good = neighbors >= 0; count = good.sum(axis=1)
        local = self.nn.points[cp.maximum(neighbors, 0)]-query[:, None, :]
        local = cp.where(good[:, :, None], local, 0.)
        mean = local.sum(axis=1)/cp.maximum(count, 1)[:, None]
        centered = cp.where(good[:, :, None], local-mean[:, None, :], 0.)
        cov = cp.einsum('nki,nkj->nij', centered, centered)/cp.maximum(count, 1)[:, None, None]
        values, vectors = cp.linalg.eigh(cov); values = cp.maximum(values, 0.)
        planarity = (values[:, 1]-values[:, 0])/cp.maximum(values[:, 2], 1e-30)
        reliable = ((count >= p.PLANE_MIN_NEIGHBORS)
            & (values[:, 1] >= p.PLANE_MIN_SECOND_EIGEN_M2)
            & (values[:, 0] <= p.PLANE_MAX_THICKNESS_RATIO*values[:, 1])
            & (planarity >= p.PLANE_MIN_PLANARITY))
        self.done[ids] = True
        self.reliable[ids] = reliable
        self.confidence[ids] = cp.clip(planarity, 0., 1.)
        self.normal[ids] = vectors[:, :, 0]
        self.center[ids] = query+mean
        self.thickness_variance[ids] = values[:, 0]
        # Line and plane classes are disjoint. A line has TWO transverse
        # residuals; no residual is invented along its unobserved axis.
        linearity = 1-values[:, 1]/cp.maximum(values[:, 2],1e-30)
        direction = vectors[:, :, 2]
        line = ((~reliable) & (count >= p.LINE_MIN_NEIGHBORS)
            & (values[:, 2] >= p.LINE_MIN_VARIANCE_M2)
            & (values[:, 1] <= p.LINE_MAX_SECOND_RATIO*values[:, 2])
            & (cp.abs(direction[:, 2]) >= np.cos(np.deg2rad(p.LINE_MAX_VERTICAL_ANGLE_DEG))))
        self.line_reliable[ids] = line
        self.line_direction[ids] = direction
        self.line_confidence[ids] = cp.clip(linearity,0.,1.)
