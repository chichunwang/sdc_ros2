"""Numerical primitives only. No tracking, files, seed selection or prior."""
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from .contracts import GeometryUnavailable

def xyz(value):
    p = np.ascontiguousarray(value, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or len(p) < 8:
        raise ValueError('Expected at least eight finite XYZ points')
    return p


def checked_pose(value):
    p = np.array(value, dtype=np.float64, copy=True)
    if (p.shape != (4, 4) or not np.isfinite(p).all()
            or not np.allclose(p[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or np.linalg.norm(p[:3, :3].T @ p[:3, :3]-np.eye(3)) > 1e-5
            or abs(np.linalg.det(p[:3, :3])-1) > 1e-5):
        raise ValueError('Expected proper T_map_source')
    return p


def block_validation(raw):
    """Immutable source-space blocks; raw .2 units, not a calibrated metre."""
    c = np.floor(xyz(raw)/.2).astype(np.int64)
    return ((c[:, 0]*73856093 ^ c[:, 1]*19349663 ^ c[:, 2]*83492791) % 5) == 0


def source_weights(raw):
    """One total vote per immutable .2-raw-unit spatial block, before any fit."""
    _, inverse, count = torch.unique(torch.floor(raw/.2).to(torch.int64), dim=0,
                                     return_inverse=True, return_counts=True)
    weight = 1./count[inverse].to(raw.dtype)
    return weight/weight.mean()


class Geometry:
    def __init__(self, backend, cache_dir=None):
        if backend not in ('cuda', 'cpu_reference'):
            raise ValueError('Use cuda or explicit cpu_reference')
        self.backend = backend
        self.device = 'cuda' if backend == 'cuda' else 'cpu'
        self.kernels = None
        if backend == 'cuda':
            from .cuda_grid import CudaKernels
            self.kernels = CudaKernels(Path(cache_dir))

    def tensor(self, value):
        return torch.as_tensor(np.ascontiguousarray(value), device=self.device, dtype=torch.float64)

    def tree(self, points, leaf=1.5):
        if self.backend == 'cuda':
            from .cuda_grid import CudaGrid
            return CudaGrid(points, self.kernels, leaf=leaf)
        return cKDTree(points)

    def query(self, tree, points, k, radius):
        if self.backend == 'cuda':
            return tree.query_gpu(points, k, radius)
        d, i = tree.query(points.numpy(), k=k, distance_upper_bound=radius, workers=1)
        d, i = np.asarray(d).reshape(-1, k), np.asarray(i).reshape(-1, k)
        return self.tensor(d), torch.as_tensor(np.where(np.isfinite(d), i, 0), dtype=torch.int64)

    def eigh(self, c):
        # Keep the established bounded CUDA eigensolver workspace.
        if self.backend == 'cuda':
            from .cuda_grid import _normal_eigh
            return _normal_eigh(c)
        return torch.linalg.eigh(c)

    def sync(self):
        if self.backend == 'cuda':
            torch.cuda.synchronize()


def voxel_statistics(points, resolution, geometry):
    """CUDA centroid/covariance reductions in local coordinates, float64."""
    origin = points.amin(0)
    local = points-origin
    cells = torch.floor(local/resolution).to(torch.int64)
    _, inverse, count = torch.unique(cells, dim=0, return_inverse=True, return_counts=True)
    n = len(count)
    sums = torch.zeros((n, 3), device=points.device, dtype=points.dtype)
    sums.index_add_(0, inverse, local)
    means = sums/count[:, None]
    delta = local-means[inverse]
    cov = torch.zeros((n, 9), device=points.device, dtype=points.dtype)
    cov.index_add_(0, inverse, (delta[:, :, None]*delta[:, None, :]).reshape(-1, 9))
    cov = cov.reshape(-1, 3, 3)/count[:, None, None]
    good = count >= 6
    mean, cov = means[good]+origin, cov[good]
    if len(mean) < 8:
        raise GeometryUnavailable('Insufficient supported NDT voxels')
    eigen, vectors = geometry.eigh(cov)
    # Discretization regularization, not calibrated sensor covariance.
    floor = torch.maximum(eigen[:, -1]*.01, torch.full_like(eigen[:, 0], (.05*resolution)**2))
    eigen = torch.maximum(eigen, floor[:, None])
    precision = (vectors/eigen[:, None, :]) @ vectors.transpose(1, 2)
    return mean, precision


def target_planes(points, tree, geometry, radius):
    d, ids = geometry.query(tree, points, min(24, len(points)), radius)
    valid = torch.isfinite(d)
    count = valid.sum(1).clamp_min(1)
    local = points[ids]-points[:, None]
    mean = (local*valid[:, :, None]).sum(1)/count[:, None]
    centered = (local-mean[:, None])*valid[:, :, None]
    cov = centered.transpose(1, 2) @ centered/count[:, None, None]
    eigen, vectors = geometry.eigh(cov)
    good = (count >= 8) & (eigen[:, 1] > .0025) & (eigen[:, 0] < .2*eigen[:, 1])
    return vectors[:, :, 0], good


def point_jacobian(rotated, estimate_scale=True):
    """d(R*s*p+t)/d[dt,dtheta,dlog_s], rotation about SOURCE origin."""
    n = len(rotated)
    j = torch.zeros((n, 3, 7 if estimate_scale else 6), device=rotated.device, dtype=rotated.dtype)
    j[:, :, :3] = torch.eye(3, device=rotated.device, dtype=rotated.dtype)
    x, y, z = rotated.unbind(1)
    j[:, 0, 4], j[:, 0, 5] = z, -y
    j[:, 1, 3], j[:, 1, 5] = -z, x
    j[:, 2, 3], j[:, 2, 4] = y, -x
    if estimate_scale:
        j[:, :, 6] = rotated
    return j


