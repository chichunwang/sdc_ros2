"""Training-only sampling weights, frozen across all current-frame candidates."""
import hashlib
import numpy as np
import torch
from .geometry import block_validation, source_weights, target_planes, xyz

def structural_weights(raw, scale, rotation, geometry):
    raw = xyz(raw)
    training = ~block_validation(raw)
    selected = raw[training]
    local = geometry.tensor(selected*float(scale))
    tree = geometry.tree(selected*float(scale))
    normal, valid = target_planes(local, tree, geometry, radius=1.5)
    world_normal = normal @ geometry.tensor(np.asarray(rotation)).T
    labels = world_normal.abs().argmax(1)
    labels = torch.where(valid, labels, torch.full_like(labels, 3))
    # The spatial membership is fixed in raw source coordinates, independent
    # of the optimized scale and any subsequently changing map association.
    spatial = source_weights(geometry.tensor(selected))
    mass = torch.zeros(4, dtype=spatial.dtype, device=spatial.device)
    mass.index_add_(0, labels, spatial)
    balanced = spatial/mass[labels]
    balanced = balanced/balanced.mean()
    output = np.ones(len(raw), dtype=np.float64)
    output[training] = balanced.cpu().numpy()
    counts = torch.bincount(labels, minlength=4).cpu().tolist()
    weighted_mass = torch.zeros(4, dtype=spatial.dtype, device=spatial.device)
    weighted_mass.index_add_(0, labels, balanced)
    fractions = (weighted_mass/balanced.sum()).cpu().tolist()
    geometry.sync()
    return output, dict(method='equal_normal_axis_strata_then_raw_spatial_blocks',
        strata=['map_x_dominant_normal','map_y_dominant_normal',
                'map_z_dominant_normal','unclassified'],
        training_points=len(selected), heldout_excluded=int((~training).sum()),
        source_training_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
        scale_for_pca=float(scale), pca_radius_m=1.5, counts=counts,
        weight_fractions=fractions, immutable_during_all_current_frame_fits=True,
        covariance_calibrated=False)


