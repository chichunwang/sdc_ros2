"""CUDA deterministic voxel selection; retain the original parser and XYZ values."""
import numbers

import numpy as np
import torch

from three_camera_calibration import load_ply


def load_ply_gpu(path, max_points=12000, leaf=.03):
    """Read PLY on CPU and select original rows with CUDA float64/int64 voxels.

    Near-integer voxel boundaries use the original NumPy floor semantics as an
    explicit boundary decision. CUDA failures propagate without a CPU fallback.
    """
    if isinstance(max_points, bool) or not isinstance(max_points, numbers.Integral) or max_points <= 0:
        raise ValueError('max_points must be a positive integer')
    if not isinstance(leaf, numbers.Real) or not np.isfinite(leaf) or leaf < 0:
        raise ValueError('leaf must be finite and nonnegative')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for load_ply_gpu; CPU fallback is disabled')
    points = np.asarray(load_ply(path, max_points=2**63-1, leaf=0), dtype=np.float64)
    keep = np.arange(len(points), dtype=np.int64)
    if leaf and len(points):
        raw = torch.as_tensor(points, dtype=torch.float64, device='cuda')
        scaled = raw / float(leaf)
        floors = torch.floor(scaled)
        near = torch.abs(scaled - torch.round(scaled)) <= 1e-10
        # Only ambiguous coordinates are adjudicated by the original CPU rule.
        boundary = torch.nonzero(near, as_tuple=False)
        if boundary.numel():
            coordinates = boundary.cpu().numpy()
            values = np.floor(points[coordinates[:, 0], coordinates[:, 1]] / leaf)
            floors[boundary[:, 0], boundary[:, 1]] = torch.as_tensor(
                values, dtype=torch.float64, device=raw.device)
        if not bool(torch.isfinite(floors).all().item()):
            raise ValueError('Nonfinite voxel coordinates')
        if bool(((floors < -(2**63)) | (floors >= 2**63)).any().item()):
            raise ValueError('Voxel coordinates exceed int64 range')
        cells = floors.to(torch.int64)
        minimum = cells.amin(dim=0).cpu().tolist()
        maximum = cells.amax(dim=0).cpu().tolist()
        sizes = [int(high)-int(low)+1 for low, high in zip(minimum, maximum)]
        product = sizes[0] * sizes[1] * sizes[2]
        if product >= 2**63:
            raise ValueError('Exact voxel key packing product exceeds int64 range')
        offsets = cells - torch.tensor(minimum, dtype=torch.int64, device=cells.device)
        keys = (offsets[:, 0] * sizes[1] + offsets[:, 1]) * sizes[2] + offsets[:, 2]
        # Stable sorting preserves earliest original index within each 3D cell.
        order = torch.argsort(keys, stable=True)
        ordered_keys = keys[order]
        first = torch.ones(len(order), dtype=torch.bool, device=order.device)
        first[1:] = ordered_keys[1:] != ordered_keys[:-1]
        keep = torch.sort(order[first], stable=True).values.cpu().numpy()
    if len(keep) > max_points:
        keep = keep[np.linspace(0, len(keep)-1, int(max_points), dtype=int)]
    torch.cuda.synchronize()
    # Never return GPU-computed coordinates: original float64 XYZ is preserved.
    return np.array(points[keep], dtype=np.float64, copy=True)
