"""CUDA bounded search for a reliable, source-compatible map surface.

Only replace an unreliable nearest point. Original reliable plane matches remain
unchanged; extra search distance is limited by one map voxel and the current
correspondence radius. This does not optimize or alter any pose.
"""
import numpy as np
import cupy as cp
from .guided import KNN
from . import parameters as p


def bounded_surface_fallback(nn, geometry, query, ids, distance2, source_good,
                             source_normals, rotation, radius, *, normal_max_deg=15.):
    result = ids.copy()
    rescued = cp.zeros(len(ids), dtype=cp.bool_)
    eligible = (ids >= 0) & (~geometry.reliable[cp.maximum(ids, 0)]) & source_good
    rows = cp.flatnonzero(eligible)
    if not len(rows):
        return result, rescued
    k = p.PLANE_NEIGHBORS
    if not 6 <= k <= 16:
        raise ValueError('CUDA surface candidate count must be 6..16')
    q = cp.ascontiguousarray(query[rows])
    neighbors = cp.empty((len(rows), k), dtype=cp.int64)
    KNN(((len(rows)+127)//128,), (128,), (q, np.int32(len(rows)), nn.points,
        nn.key, nn.start, nn.end, np.int32(len(nn.key)), np.float64(nn.cell),
        np.float64(radius), np.int32(k), neighbors))
    geometry.ensure(neighbors.ravel())
    safe = cp.maximum(neighbors, 0)
    normals = source_normals[rows] @ cp.asarray(rotation).T
    compatible = cp.abs(cp.einsum('ni,nki->nk', normals, geometry.normal[safe])) >= np.cos(np.deg2rad(normal_max_deg))
    d2 = cp.sum((q[:, None, :]-nn.points[safe])**2, axis=2)
    limit = cp.minimum(radius, cp.sqrt(distance2[rows])+p.METRIC_MAP_LEAF_M)
    valid = ((neighbors >= 0) & geometry.reliable[safe] & compatible
             & (d2 <= limit[:, None]**2+1e-12))
    chosen = cp.argmin(cp.where(valid, d2, cp.inf), axis=1)
    present = valid.any(axis=1)
    result[rows] = cp.where(present, neighbors[cp.arange(len(rows)), chosen], ids[rows])
    rescued[rows] = present
    return result, rescued
