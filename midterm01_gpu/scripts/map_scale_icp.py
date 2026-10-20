"""Independent layered Sim(3) map observations; no GNSS residual in the fit.

The seventh unknown is log camera-depth scale. GNSS/IMU supply association seeds
and finite trust bounds only. Geometry information is diagnostic, not covariance.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from horizontal_surfel import heldout_patches


@dataclass(frozen=True)
class ScaleICPConfig:
    initial_scales: tuple = (6., 8.7, 11.)
    min_scale: float = 4.
    max_scale: float = 12.
    max_range_m: float = 55.
    max_translation_m: float = 5.
    max_rotation_deg: float = 15.
    match_radii_m: tuple = (3., 1.5)
    iterations_per_level: int = 14
    huber_m: float = .4
    patch_m: float = 1.5
    max_pairs_per_bin: int = 160
    min_train_patches: int = 24
    min_test_patches: int = 8
    min_nonground_patches: int = 12
    holdout_median_m: float = .45
    holdout_p90_m: float = 1.
    min_scaled_singular: float = .12


def normals_at(points, tree, indices, radius=2.5):
    distance, ids = tree.query(points[indices], k=min(24, len(points)), workers=1)
    local = points[ids]-points[indices, None]
    valid = distance <= radius
    count = valid.sum(axis=1)
    mean = (local*valid[..., None]).sum(axis=1)/np.maximum(count[:, None], 1)
    centered = (local-mean[:, None])*valid[..., None]
    cov = np.einsum('nki,nkj->nij', centered, centered)/np.maximum(count[:, None, None], 1)
    eigen, vectors = np.linalg.eigh(cov)
    quality = (eigen[:, 1]-eigen[:, 0])/np.maximum(eigen[:, 2], 1e-12)
    good = (count >= 8) & (eigen[:, 1] > .015) & (eigen[:, 0] < .3*eigen[:, 1]) & (quality > .15)
    return vectors[:, :, 0], quality*good


class LayeredScaleMap:
    def __init__(self, points, config=None):
        self.config = config or ScaleICPConfig()
        self.points = np.asarray(points, float)
        if self.points.ndim != 2 or self.points.shape[1] != 3 or len(points) < 64 or not np.isfinite(points).all():
            raise ValueError('Layered map requires finite XYZ points')
        self.tree = cKDTree(self.points)
        self.normals = np.zeros_like(self.points)
        self.quality = np.zeros(len(points))
        self.computed = np.zeros(len(points), bool)

    def surfaces(self, ids):
        pending = np.unique(ids[~self.computed[ids]])
        for first in range(0, len(pending), 4096):
            selected = pending[first:first+4096]
            self.normals[selected], self.quality[selected] = normals_at(self.points, self.tree, selected)
            self.computed[selected] = True
        return self.normals[ids], self.quality[ids]

    def road_height(self, prior):
        ids = np.asarray(self.tree.query_ball_point(prior[:3, 3], 12.), int)
        if not len(ids):
            return float(prior[2, 3]-1.5)
        normal, quality = self.surfaces(ids)
        points = self.points[ids]
        okay = ((quality > 0) & (abs(normal[:, 2]) > .85)
                & (np.linalg.norm(points[:, :2]-prior[:2, 3], axis=1) < 8.)
                & (points[:, 2] < prior[2, 3]+.5))
        # A local lower horizontal surface estimate, not a fixed map Z offset.
        return float(np.quantile(points[okay, 2], .35)) if okay.sum() >= 8 else float(prior[2, 3]-1.5)


def source_geometry(raw, prior, config):
    raw = np.asarray(raw, float)
    if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
        raise ValueError('Source must be finite Nx3')
    # Fixed membership across scale starts prevents choosing scale by dropping far points.
    distance = np.linalg.norm(raw, axis=1)
    raw = raw[(distance > .15) & (distance*config.max_scale <= config.max_range_m)]
    if len(raw) < 64:
        return raw, np.zeros_like(raw), np.zeros(len(raw), bool), np.zeros(len(raw))
    metric = raw*config.initial_scales[1]
    normal, quality = normals_at(metric, cKDTree(metric), np.arange(len(raw)))
    rotated = metric@prior[:3, :3].T
    world_normals = normal@prior[:3, :3].T
    low = (quality > 0) & (abs(world_normals[:, 2]) > .85) & (rotated[:, 2] < 0)
    height = np.quantile(rotated[low, 2], .35) if low.sum() >= 8 else -1.5
    ground = (abs(world_normals[:, 2]) > .85) & (abs(rotated[:, 2]-height) < .8)
    return raw, normal, ground, quality


def pairs_at(scene, raw, source_normals, source_ground, source_quality, pose, scale, road_z, radius, config):
    rotated = scale*raw@pose[:3, :3].T
    world = rotated+pose[:3, 3]
    distance, ids = scene.tree.query(world, k=8, workers=1)
    normal, quality = scene.surfaces(ids.ravel())
    normal, quality = normal.reshape(-1, 8, 3), quality.reshape(-1, 8)
    targets = scene.points[ids]
    ground = (abs(normal[:, :, 2]) > .85) & (abs(targets[:, :, 2]-road_z) < .8)
    source_n = source_normals@pose[:3, :3].T
    agreement = abs(np.einsum('nki,ni->nk', normal, source_n)) > np.cos(np.deg2rad(55.))
    allowed = ((ground == source_ground[:, None]) & (quality > 0) & agreement
               & (source_quality[:, None] > 0) & (distance < radius))
    best = np.argmin(np.where(allowed, distance, np.inf), axis=1)
    available = np.flatnonzero(allowed[np.arange(len(raw)), best])
    target_ids = ids[np.arange(len(raw)), best]
    n = normal[np.arange(len(raw)), best]
    d = distance[np.arange(len(raw)), best]
    patch = np.floor(scene.points[target_ids]/config.patch_m).astype(np.int64)
    # Three unsigned dominant-normal axes balance road and differently oriented walls.
    bins = np.argmax(abs(n), axis=1)
    selected = []
    for b in range(3):
        candidates = available[bins[available] == b]
        candidates = candidates[np.argsort(d[candidates], kind='stable')]
        seen = set(); unique = []
        for i in candidates:
            key = tuple(patch[i])
            if key not in seen:
                seen.add(key); unique.append(i)
        if len(unique) > config.max_pairs_per_bin:
            unique = np.asarray(unique)[np.linspace(0, len(unique)-1, config.max_pairs_per_bin, dtype=int)]
        selected.extend(unique)
    selected = np.asarray(selected, int)
    return {'rotated': rotated[selected], 'world': world[selected],
            'target': scene.points[target_ids[selected]], 'normal': n[selected],
            'patch': patch[selected], 'ground': source_ground[selected],
            'source_index': selected, 'holdout': heldout_patches(patch[selected]),
            'quality': np.minimum(scene.quality[target_ids[selected]], source_quality[selected])}


def linear_terms(pool):
    n, q = pool['normal'], pool['rotated']
    residual = np.einsum('ni,ni->n', n, pool['world']-pool['target'])
    # Left SO(3) increment about the camera center; log-scale derivative is R*s*p.
    jacobian = np.column_stack((n, np.cross(q, n), np.einsum('ni,ni->n', n, q)))
    return residual, jacobian


def fit_seed(scene, raw, normal, ground, quality, prior, seed, road_z, config):
    pose = prior.copy(); scale = float(seed); stable = False
    for radius in config.match_radii_m:
        stable = False
        for _ in range(config.iterations_per_level):
            pool = pairs_at(scene, raw, normal, ground, quality, pose, scale, road_z, radius, config)
            train = ~pool['holdout']
            if train.sum() < config.min_train_patches:
                break
            residual, jac = linear_terms(pool)
            residual, jac = residual[train], jac[train]
            weight = pool['quality'][train]*np.minimum(1., config.huber_m/np.maximum(abs(residual), 1e-12))
            # Numerical units only, no prior or damping residual in the map estimate.
            units = np.array([1., 1., 1., .05, .05, .05, .1])
            step = np.linalg.lstsq(jac*units*np.sqrt(weight[:, None]), -residual*np.sqrt(weight), rcond=1e-5)[0]*units
            step[:3] *= min(1., .5/max(np.linalg.norm(step[:3]), 1e-12))
            step[3:6] *= min(1., np.deg2rad(2.)/max(np.linalg.norm(step[3:6]), 1e-12))
            step[6] = np.clip(step[6], -.08, .08)
            proposed = pose.copy()
            proposed[:3, :3] = Rotation.from_rotvec(step[3:6]).as_matrix()@pose[:3, :3]
            proposed[:3, 3] += step[:3]
            new_scale = scale*np.exp(step[6])
            angle = Rotation.from_matrix(proposed[:3, :3]@prior[:3, :3].T).magnitude()
            if (not config.min_scale < new_scale < config.max_scale
                    or np.linalg.norm(proposed[:3, 3]-prior[:3, 3]) > config.max_translation_m
                    or angle > np.deg2rad(config.max_rotation_deg)):
                return pose, scale, False
            pose, scale = proposed, new_scale
            if np.linalg.norm(step[:3]) < .003 and np.linalg.norm(step[3:6]) < .0003 and abs(step[6]) < .0005:
                stable = True
                break
    return pose, scale, stable


def register_frame(scene, raw, prior, config=None):
    config = config or scene.config
    prior = np.asarray(prior, float)
    if (prior.shape != (4, 4) or not np.isfinite(prior).all()
            or not np.allclose(prior[3], [0, 0, 0, 1])
            or not np.allclose(prior[:3, :3].T@prior[:3, :3], np.eye(3), atol=1e-6)):
        raise ValueError('Invalid source-to-map seed')
    raw, normal, ground, quality = source_geometry(raw, prior, config)
    report = {'accepted': False, 'pose': prior.tolist(), 'scale': float(config.initial_scales[1]),
              'reasons': [], 'source_points': len(raw), 'source_ground_points': int(ground.sum()),
              'source_nonground_points': int(((~ground)&(quality > 0)).sum())}
    if len(raw) < 64:
        report['reasons'] = ['insufficient_source']; return report
    road_z = scene.road_height(prior)
    candidates = []
    for seed in config.initial_scales:
        pose, scale, stable = fit_seed(scene, raw, normal, ground, quality, prior, seed, road_z, config)
        pool = pairs_at(scene, raw, normal, ground, quality, pose, scale, road_z, config.match_radii_m[-1], config)
        residual, jac = linear_terms(pool)
        train, test = ~pool['holdout'], pool['holdout']
        # Candidate selection uses TRAINING only, with a missing-support penalty.
        fit_cost = float(np.mean(np.minimum(residual[train]**2, 1.))) if train.any() else 1.
        fit_cost += 1.-min(1., train.sum()/max(1, config.min_train_patches*3))
        candidates.append((fit_cost, pose, scale, stable, pool, residual, jac))
    _, pose, scale, stable, pool, residual, jac = min(candidates, key=lambda x: x[0])
    train, test = ~pool['holdout'], pool['holdout']
    # Marginalize free Z, roll, pitch and scale before checking XY/yaw support.
    units = np.array([1., 1., 1., .05, .05, .05, .1])
    j = jac[train]*units*np.sqrt(pool['quality'][train, None])
    if len(j):
        nuisance, horizontal = j[:, [2, 3, 4, 6]], j[:, [0, 1, 5]]
        projected = horizontal-nuisance@np.linalg.lstsq(nuisance, horizontal, rcond=1e-6)[0]
        singular = np.linalg.svd(projected, compute_uv=False)
        scale_col, other = j[:, 6], j[:, :6]
        scale_info = float(np.linalg.norm(scale_col-other@np.linalg.lstsq(other, scale_col, rcond=1e-6)[0]))
    else:
        singular, scale_info = np.zeros(3), 0.
    test_residual = abs(residual[test])
    median = float(np.median(test_residual)) if len(test_residual) else None
    p90 = float(np.quantile(test_residual, .9)) if len(test_residual) else None
    nonground = len(np.unique(pool['patch'][train & ~pool['ground']], axis=0))
    checks = {'train_support': len(np.unique(pool['patch'][train], axis=0)) >= config.min_train_patches,
              'test_support': len(np.unique(pool['patch'][test], axis=0)) >= config.min_test_patches,
              'nonground_support': nonground >= config.min_nonground_patches,
              'horizontal_observability': len(singular) == 3 and singular[-1] >= config.min_scaled_singular,
              'scale_observability': scale_info >= config.min_scaled_singular,
              'heldout_residual': median is not None and median <= config.holdout_median_m and p90 <= config.holdout_p90_m,
              'interior': config.min_scale+.02 < scale < config.max_scale-.02
                          and np.linalg.norm(pose[:3, 3]-prior[:3, 3]) < config.max_translation_m-.02}
    report.update(accepted=bool(all(checks.values())), pose=pose.tolist(), scale=float(scale),
        reasons=[k for k, v in checks.items() if not v], optimizer_step_small=bool(stable),
        train_patches=len(np.unique(pool['patch'][train], axis=0)), test_patches=len(np.unique(pool['patch'][test], axis=0)),
        nonground_train_patches=nonground, heldout_median_m=median, heldout_p90_m=p90,
        horizontal_singular_values=singular.tolist(), marginalized_scale_singular=scale_info,
        road_height_m=road_z, translation_from_seed_m=(pose[:3, 3]-prior[:3, 3]).tolist(),
        rotation_from_seed_deg=float(np.degrees(Rotation.from_matrix(pose[:3, :3]@prior[:3, :3].T).magnitude())),
        fit_scope='map only; GNSS/IMU seed and trust bounds; geometry diagnostics are not accuracy/covariance')
    return report
