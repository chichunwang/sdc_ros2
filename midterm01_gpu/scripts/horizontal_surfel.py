"""Fine-map horizontal surfels and data-only factors with spatial holdout gates."""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SurfelConfig:
    leaf_m: float = .3
    radius_small_m: float = .9
    radius_large_m: float = 1.8
    patch_m: float = 1.2
    match_radius_m: float = 1.5
    plane_std_m: float = .3
    min_fit_patches: int = 12
    min_holdout_patches: int = 6
    max_points_per_normal_bin: int = 60
    # Include sloped roadside structures as well as facades. Pure horizontal
    # planes remain excluded; free-Z information decides which XY/yaw axes exist.
    max_normal_z: float = .98
    max_normal_disagreement_deg: float = 12.
    max_translation_m: float = .75
    max_yaw_deg: float = 3.
    yaw_scale_rad: float = np.deg2rad(3.)


class FineSurfelMap:
    def __init__(self, points, config=None):
        self.config = config or SurfelConfig()
        self.points = np.asarray(points, float)
        if self.points.ndim != 2 or self.points.shape[1] != 3 or len(self.points) < 64 or not np.isfinite(self.points).all():
            raise ValueError('Surfel map requires at least 64 finite XYZ points')
        self.tree = cKDTree(self.points)
        self.normals = np.zeros_like(self.points)
        self.quality = np.zeros(len(points))
        self.computed = np.zeros(len(points), bool)

    def surfaces(self, indices):
        pending = np.unique(indices[~self.computed[indices]])
        for first in range(0, len(pending), 2048):
            selected = pending[first:first+2048]
            distances, ids = self.tree.query(self.points[selected], k=64, workers=1)
            local = self.points[ids]-self.points[selected, None, :]
            normals, qualities = [], []
            for radius in (self.config.radius_small_m, self.config.radius_large_m):
                valid = distances <= radius
                count = valid.sum(axis=1)
                mean = np.sum(local*valid[..., None], axis=1)/np.maximum(count[:, None], 1)
                centered = (local-mean[:, None, :])*valid[..., None]
                covariance = np.einsum('nki,nkj->nij', centered, centered)/np.maximum(count[:, None, None], 1)
                eigenvalues, vectors = np.linalg.eigh(covariance)
                eigenvalues = np.maximum(eigenvalues, 0.)
                planarity = (eigenvalues[:, 1]-eigenvalues[:, 0])/np.maximum(eigenvalues[:, 2], 1e-12)
                reliable = ((count >= 8) & (eigenvalues[:, 1] >= (self.config.leaf_m*.5)**2)
                            & (eigenvalues[:, 0] <= .2*eigenvalues[:, 1]) & (planarity >= .2))
                normals.append(vectors[:, :, 0]); qualities.append(planarity*reliable)
            agreement = np.abs(np.sum(normals[0]*normals[1], axis=1)) >= np.cos(np.deg2rad(self.config.max_normal_disagreement_deg))
            quality = np.minimum(*qualities)*agreement
            self.normals[selected] = normals[0]
            self.quality[selected] = np.clip(quality, 0., 1.)
            self.computed[selected] = True
        return self.normals[indices], self.quality[indices]


def patch_ids(points, size):
    return np.floor(np.asarray(points)/size).astype(np.int64)


def heldout_patches(ids):
    # Stable integer spatial split; no residual or pose-label based selection.
    hashed = np.bitwise_xor.reduce(ids*np.array([73856093, 19349663, 83492791], dtype=np.int64), axis=1)
    return hashed % 4 == 0


def select_horizontal_pairs(scene, raw, pose):
    config = scene.config
    rotated = np.asarray(raw)@pose[:, :3].T
    world = rotated+pose[:, 3]
    distances, indices = scene.tree.query(world)
    normals, quality = scene.surfaces(indices)
    valid = ((distances <= config.match_radius_m) & (quality > 0) & (np.abs(normals[:, 2]) <= config.max_normal_z))
    available = np.flatnonzero(valid)
    # One nearest source per map patch/orientation. Dense pixels cannot create
    # arbitrary independent copies of the same map plane.
    ids = patch_ids(scene.points[indices], config.patch_m)
    bins = np.floor(((np.arctan2(normals[:, 1], normals[:, 0])+np.pi/8) % np.pi)/(np.pi/4)).astype(int)
    chosen = []; seen = set()
    for index in available[np.argsort(distances[available], kind='stable')]:
        key = (*ids[index], int(bins[index]))
        if key not in seen:
            seen.add(key); chosen.append(index)
    balanced = []
    # Deterministic round-robin over spatially sorted patches in each normal bin.
    for normal_bin in range(4):
        selected = [i for i in chosen if bins[i] == normal_bin]
        selected.sort(key=lambda i: tuple(ids[i]))
        if len(selected) > config.max_points_per_normal_bin:
            selected = [selected[j] for j in np.linspace(0, len(selected)-1, config.max_points_per_normal_bin, dtype=int)]
        balanced.extend(selected)
    selected = np.array(balanced, int)
    if len(selected):
        _, inverse, counts = np.unique(ids[selected], axis=0, return_inverse=True, return_counts=True)
        balanced_quality = quality[selected]/counts[inverse]
    else: balanced_quality = quality[selected]
    return {'rotated': rotated[selected], 'targets': scene.points[indices[selected]]-pose[:, 3],
            'normals': normals[selected], 'quality': balanced_quality, 'patch_ids': ids[selected],
            'normal_bins': bins[selected], 'holdout': heldout_patches(ids[selected]),
            'initial_distances': distances[selected], 'source_index': selected,
            'raw_points': len(raw), 'nonhorizontal_before_dedup': int(valid.sum())}


def plane_residual_and_jacobian(delta, pool, config):
    angle = delta[3]*config.yaw_scale_rad
    rotated = pool['rotated']@Rotation.from_euler('z', angle).as_matrix().T
    error = rotated+delta[:3]-pool['targets']
    residual = np.sum(pool['normals']*error, axis=1)
    yaw_column = np.sum(pool['normals']*np.column_stack((-rotated[:, 1], rotated[:, 0], np.zeros(len(rotated)))), axis=1)*config.yaw_scale_rad
    jacobian = np.column_stack((pool['normals'], yaw_column))
    return residual, jacobian


def horizontal_information(jacobian, residual, quality, config):
    # Eliminate unconstrained Z from the DATA, without a GNSS/height prior.
    weights = quality*np.minimum(1., .5/np.maximum(np.abs(residual), 1e-12))/config.plane_std_m**2
    j = jacobian*np.sqrt(weights[:, None])
    projected = j[:, [0, 1, 3]]
    nuisance = j[:, 2]
    if nuisance@nuisance > 1e-10:
        projected = projected-nuisance[:, None]*(nuisance@projected)/(nuisance@nuisance)
    information = projected.T@projected
    values, vectors = np.linalg.eigh(information)
    values = np.maximum(values, 0.)
    # Retain only actually measured directions, never inflate a weak axis.
    selected = values >= max(1., .02*values[-1])
    return values, vectors, selected, projected, weights


def fit_horizontal_factor(pool, pose, config=None):
    config = config or SurfelConfig()
    holdout = pool['holdout']; train = ~holdout
    fit_count = len(np.unique(pool['patch_ids'][train], axis=0))
    holdout_count = len(np.unique(pool['patch_ids'][holdout], axis=0))
    report = {'accepted': False, 'raw_points': pool['raw_points'],
              'nonhorizontal_before_dedup': pool['nonhorizontal_before_dedup'], 'patches': len(holdout),
              'fit_patches': fit_count, 'holdout_patches': holdout_count,
              'unique_spatial_patches': fit_count+holdout_count,
              'normal_bins': len(np.unique(pool['normal_bins'])), 'reasons': []}
    if fit_count < config.min_fit_patches or holdout_count < config.min_holdout_patches:
        report['reasons'] = ['insufficient_spatial_support']; return report, None
    def subset(mask):
        return {key: value[mask] for key, value in pool.items() if isinstance(value, np.ndarray) and len(value) == len(holdout)}
    def solve(mask):
        selected = subset(mask)
        quality = np.sqrt(selected['quality'])
        bounds = [config.max_translation_m]*3+[config.max_yaw_deg/np.degrees(config.yaw_scale_rad)]
        fit = least_squares(lambda d: plane_residual_and_jacobian(d, selected, config)[0]*quality,
            np.zeros(4), jac=lambda d: plane_residual_and_jacobian(d, selected, config)[1]*quality[:, None],
            bounds=(-np.array(bounds), np.array(bounds)), loss='huber', f_scale=.5, max_nfev=30,
            ftol=1e-7, xtol=1e-7, gtol=1e-7)
        return fit, selected
    fit, selected = solve(train)
    test_fit, _ = solve(holdout)  # validation only; never added to the factor
    before, _ = plane_residual_and_jacobian(np.zeros(4), subset(holdout), config)
    after, _ = plane_residual_and_jacobian(fit.x, subset(holdout), config)
    residual, jacobian = plane_residual_and_jacobian(fit.x, selected, config)
    values, vectors, supported, _, _ = horizontal_information(jacobian, residual, selected['quality'], config)
    difference = (fit.x-test_fit.x)[[0, 1, 3]]
    directions = vectors[:, supported].T
    disagreement = np.linalg.norm(directions@difference) if len(directions) else None
    report.update(delta_xyz_scaled_yaw=fit.x.tolist(), horizontal_eigenvalues=values.tolist(), rank=int(supported.sum()),
        heldout_before_median_m=float(np.median(np.abs(before))), heldout_after_median_m=float(np.median(np.abs(after))),
        heldout_after_p90_m=float(np.quantile(np.abs(after), .9)), fit_vs_holdout_supported_scaled_distance= None if disagreement is None else float(disagreement),
        success=bool(fit.success and test_fit.success), fit_at_bound=bool(np.any(fit.active_mask)), active_bounds=fit.active_mask.tolist())
    checks = {'optimizer': fit.success and test_fit.success, 'interior_solution': not np.any(fit.active_mask),
        'measured_horizontal_directions': int(supported.sum()) >= 2,
        'heldout_consistency': np.median(np.abs(after)) <= .3 and np.quantile(np.abs(after), .9) <= .75,
        'heldout_not_worsened': np.median(np.abs(after)) <= np.median(np.abs(before))+.02,
        'split_solution_agreement': disagreement is not None and disagreement <= .3}
    report['reasons'] = [key for key, value in checks.items() if not value]
    report['accepted'] = all(checks.values())
    if not report['accepted']: return report, None
    # Linearized map-only factor. Half strength is a conservative allowance for
    # unknown depth/model correlations, not an asserted calibrated covariance.
    sqrt_information = np.sqrt(values[supported])[:, None]*vectors[:, supported].T/np.sqrt(2.)
    return report, {'reference_xy_yaw': [*pose[:2, 3], float(np.arctan2(pose[1, 0], pose[0, 0]))],
                    'delta_xy_scaled_yaw': fit.x[[0, 1, 3]].tolist(),
                    'sqrt_information': sqrt_information.tolist(), 'yaw_scale_rad': config.yaw_scale_rad,
                    'train_patch_ids': selected['patch_ids'].tolist(),
                    'scope': 'map data only, Z marginalized; geometry/model information, not calibrated covariance'}
