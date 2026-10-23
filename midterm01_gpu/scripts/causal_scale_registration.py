"""Current-frame bounded Sim(3) observations for the forward Track3 filter.

CUDA performs exact kNN, normals and balanced pairing. CPU only solves at most
7 unknowns and checks numerical bounds. An explicit CPU reference is available
for synthetic mechanism tests; production never silently falls back to it.
"""
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from causal_track3_filter import proper_pose
from map_scale_icp import (LayeredScaleMap, ScaleICPConfig, linear_terms,
                           normals_at, pairs_at)
from horizontal_surfel import heldout_patches


@dataclass(frozen=True)
class CausalScaleConfig:
    min_scale: float = .5
    max_scale: float = 16.
    max_log_scale_step: float = .15
    startup_translation_m: float = 5.
    tracking_translation_m: float = 3.
    startup_rotation_deg: float = 24.  # 3 * filter's 8 degree initial uncertainty
    tracking_rotation_deg: float = 8.
    position_std_floor_m: float = .25
    rotation_std_floor_deg: float = 1.
    log_scale_std_floor: float = .05
    iterations_per_level: int = 40

    def __post_init__(self):
        if (any(not np.isfinite(v) or v <= 0 for v in asdict(self).values())
                or self.min_scale >= self.max_scale):
            raise ValueError('Invalid positive scale/trust/noise bounds')
        if not isinstance(self.iterations_per_level, int):
            raise ValueError('iterations_per_level must be an integer')


def _loss(residual, quality, huber):
    a = abs(residual)
    return float(np.sum(quality*np.where(a <= huber, .5*a*a, huber*(a-.5*huber))))


def fit_bounded(scene, raw, normal, ground, quality, prior, seed, road_z,
                cfg, pair_function, *, estimate_scale, initial_pose=None):
    pose = prior.copy() if initial_pose is None else initial_pose.copy()
    scale = float(seed)
    stable = False
    levels = []
    for radius in cfg.match_radii_m:
        stable = False
        plateau = 0
        status = {'radius_m': radius, 'iterations': 0, 'stop': 'iteration_limit'}
        for iteration in range(cfg.iterations_per_level):
            status['iterations'] = iteration+1
            pool = pair_function(scene, raw, normal, ground, quality, pose, scale, road_z, radius, cfg)
            train = ~pool['holdout']
            if train.sum() < cfg.min_train_patches:
                status['stop'] = 'insufficient_training_support'
                break
            residual, jac = linear_terms(pool)
            r, j, w = residual[train], jac[train], pool['quality'][train]
            if not estimate_scale:
                j = j[:, :6]
            weight = w*np.minimum(1., cfg.huber_m/np.maximum(abs(r), 1e-12))
            units = np.array([1., 1., 1., .05, .05, .05] + ([.1] if estimate_scale else []))
            step = np.linalg.lstsq(j*units*np.sqrt(weight[:, None]), -r*np.sqrt(weight), rcond=1e-5)[0]*units
            step[:3] *= min(1., .5/max(np.linalg.norm(step[:3]), 1e-12))
            step[3:6] *= min(1., np.deg2rad(2.)/max(np.linalg.norm(step[3:6]), 1e-12))
            if estimate_scale:
                step[6] = np.clip(step[6], -.08, .08)
            if (np.linalg.norm(step[:3]) < .003 and np.linalg.norm(step[3:6]) < .0003
                    and (not estimate_scale or abs(step[6]) < .0005)):
                stable = True
                status['stop'] = 'small_step'
                break
            old_cost = _loss(r, w, cfg.huber_m)
            moved = False
            for alpha in (1., .5, .25, .125, .0625, .03125, .015625):
                proposed = pose.copy()
                proposed[:3, :3] = Rotation.from_rotvec(alpha*step[3:6]).as_matrix() @ pose[:3, :3]
                proposed[:3, 3] += alpha*step[:3]
                new_scale = scale*np.exp(alpha*step[6]) if estimate_scale else scale
                angle = Rotation.from_matrix(proposed[:3, :3] @ prior[:3, :3].T).magnitude()
                if (not cfg.min_scale <= new_scale <= cfg.max_scale
                        or np.linalg.norm(proposed[:3, 3]-prior[:3, 3]) > cfg.max_translation_m
                        or angle > np.deg2rad(cfg.max_rotation_deg)):
                    continue
                # Descent is evaluated on the SAME training correspondences.
                # Validation patches never select a seed, step or line search.
                selected = pool['source_index'][train]
                world = new_scale*raw[selected] @ proposed[:3, :3].T + proposed[:3, 3]
                new_r = np.einsum('ni,ni->n', pool['normal'][train], world-pool['target'][train])
                new_cost = _loss(new_r, w, cfg.huber_m)
                if new_cost <= old_cost+1e-12:
                    pose, scale, moved = proposed, float(new_scale), True
                    relative_drop = (old_cost-new_cost)/max(old_cost, 1e-12)
                    # Test the ACTUAL accepted increment, not the unscaled
                    # Gauss-Newton proposal. Require sustained cost stability
                    # and increments below 1/10 of the measurement noise floors.
                    small = (alpha*np.linalg.norm(step[:3]) < .025
                             and alpha*np.linalg.norm(step[3:6]) < np.deg2rad(.1)
                             and (not estimate_scale or alpha*abs(step[6]) < .005))
                    plateau = plateau+1 if small and relative_drop < 1e-3 else 0
                    status.update(last_relative_cost_drop=relative_drop,
                                  last_translation_step_m=float(alpha*np.linalg.norm(step[:3])),
                                  last_rotation_step_deg=float(np.rad2deg(alpha*np.linalg.norm(step[3:6]))),
                                  last_log_scale_step=float(alpha*step[6]) if estimate_scale else 0.)
                    break
            if not moved:
                status['stop'] = 'line_search_or_constraint_blocked'
                break
            if plateau >= 3:
                stable = True
                status['stop'] = 'cost_and_step_stable'
                break
        status['converged'] = stable
        levels.append(status)
    return pose, scale, stable, levels


def _up_normal(normal, up):
    """Sign-invariant median of local surface normals, oriented toward up."""
    n = normal.copy()
    n *= np.where((n @ up)[:, None] < 0, -1., 1.)
    value = np.median(n, axis=0)
    return value/max(np.linalg.norm(value), 1e-12)


class CausalScaleRegistrar:
    def __init__(self, points, config=None, *, cache_dir=None, backend='cuda'):
        self.config = config or CausalScaleConfig()
        # Keep established spatial/normal/heldout acceptance thresholds. No
        # Track-specific ranges, frame overrides or leaderboard-driven tuning.
        self.base = ScaleICPConfig()
        if backend == 'cuda':
            from cuda_scale_geometry import CudaLayeredMap, CudaGrid, cuda_normals, cuda_pairs
            self.scene = CudaLayeredMap(points, self.base, cache_dir=cache_dir)
            self.tree_factory = lambda p: CudaGrid(p, self.scene.kernels)
            self.normals = cuda_normals
            self.pairs = cuda_pairs
        elif backend == 'cpu_reference':
            self.scene = LayeredScaleMap(points, self.base)
            self.tree_factory = cKDTree
            self.normals = normals_at
            self.pairs = pairs_at
        else:
            raise ValueError('Explicit backend must be cuda or cpu_reference')
        self.backend = backend
        self.training_tree = None

    def _coverage_cost(self, raw, pose, scale):
        """Score a FIXED source set, including points that lost correspondences.

        A low residual on one wall cannot win by discarding the front wall.
        Only training map patches enter this nearest-surface score. Distances
        are capped at the existing PCA/search radius, with missing points paying
        that same cap rather than disappearing from candidate selection.
        """
        if self.training_tree is None:
            train = ~heldout_patches(np.floor(self.scene.points/self.base.patch_m).astype(np.int64))
            if not train.any():
                return 2.5**2
            self.training_tree = self.tree_factory(self.scene.points[train])
        world = scale*raw @ pose[:3, :3].T+pose[:3, 3]
        distance, _ = self.training_tree.query(world, k=1, workers=1)
        return float(np.mean(np.minimum(np.asarray(distance).ravel(), 2.5)**2))

    def _geometry(self, raw, scale, pose):
        """Metric PCA thresholds must be evaluated at this candidate's scale."""
        metric = raw*scale
        normal, quality = self.normals(metric, self.tree_factory(metric), np.arange(len(raw)))
        rotated = metric @ pose[:3, :3].T
        world_normals = normal @ pose[:3, :3].T
        low = (quality > 0) & (abs(world_normals[:, 2]) > .85) & (rotated[:, 2] < 0)
        height = np.quantile(rotated[low, 2], .35) if low.sum() >= 8 else -1.5
        ground = (abs(world_normals[:, 2]) > .85) & (abs(rotated[:, 2]-height) < .8)
        return normal, ground, quality

    def _ground_initialization(self, raw, prior, scale, road_z):
        """A current-frame seed, NOT a ground-truth pose or scale measurement.

        Align the lower source plane with a local map plane. Target patches used
        for final validation are excluded from this initialization. The supplied
        initial height only seeds scale; the seven-unknown map fit remains free.
        """
        n, ground, quality = self._geometry(raw, scale, prior)
        source_ids = ground & (quality > 0)
        ids = np.asarray(self.scene.tree.query_ball_point(prior[:3, 3], 12.), int)
        if source_ids.sum() < 24 or not len(ids):
            return None
        mn, mq = self.scene.surfaces(ids)
        mp = self.scene.points[ids]
        train = ~heldout_patches(np.floor(mp/self.base.patch_m).astype(np.int64))
        keep = (train & (mq > 0) & (abs(mn[:, 2]) > .85) & (abs(mp[:, 2]-road_z) < .8))
        if keep.sum() < 24:
            return None
        source_n = _up_normal(n[source_ids], prior[:3, :3].T @ np.array([0., 0., 1.]))
        map_n = _up_normal(mn[keep], np.array([0., 0., 1.]))
        world_n = prior[:3, :3] @ source_n
        axis = np.cross(world_n, map_n)
        angle = np.arctan2(np.linalg.norm(axis), world_n @ map_n)
        if angle >= np.deg2rad(self.config.startup_rotation_deg):
            return None
        pose = prior.copy()
        if np.linalg.norm(axis) > 1e-12:
            pose[:3, :3] = Rotation.from_rotvec(axis/np.linalg.norm(axis)*angle).as_matrix() @ prior[:3, :3]
        source_offset = float(np.median(raw[source_ids] @ source_n))
        target_offset = float(np.median(mp[keep] @ map_n))
        seed = ((target_offset-map_n @ prior[:3, 3])/source_offset
                if abs(source_offset) > 1e-6 else np.nan)
        return pose, seed, {'tilt_correction_deg': float(np.rad2deg(angle)),
                            'source_plane_points': int(source_ids.sum()), 'map_training_plane_points': int(keep.sum()),
                            'scale_from_initial_height': float(seed) if np.isfinite(seed) else None,
                            'status': 'training geometry initialization only; initial height accuracy unknown'}

    def register(self, raw, prior, predicted_scale, *, bootstrap=False, estimate_scale=True):
        prior = proper_pose(prior)
        c = self.config
        if not c.min_scale <= predicted_scale <= c.max_scale:
            raise ValueError('Predicted scale outside configured global domain')
        if bootstrap and estimate_scale:
            low, high = c.min_scale, c.max_scale
            # Log-spaced numerical initialization in an explicit uncalibrated
            # domain, chosen on training only; not a Kaggle parameter sweep.
            seeds = np.unique(np.r_[np.geomspace(low*1.01, high/1.01, 7), predicted_scale])
        elif estimate_scale:
            low = max(c.min_scale, predicted_scale*np.exp(-c.max_log_scale_step))
            high = min(c.max_scale, predicted_scale*np.exp(c.max_log_scale_step))
            seeds = np.unique([low+(high-low)*.1, predicted_scale, high-(high-low)*.1])
        else:
            low, high, seeds = c.min_scale, c.max_scale, [predicted_scale]
        cfg = replace(self.base, min_scale=float(low), max_scale=float(high),
                      iterations_per_level=c.iterations_per_level,
                      max_translation_m=c.startup_translation_m if bootstrap else c.tracking_translation_m,
                      max_rotation_deg=c.startup_rotation_deg if bootstrap else c.tracking_rotation_deg)
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
            raise ValueError('Expected finite raw Nx3 points')
        distance = np.linalg.norm(raw, axis=1)
        # Fixed source membership across all seeds, based on the PREDICTION.
        # Missing matches incur a training cost instead of disappearing from it.
        raw = np.ascontiguousarray(raw[(distance > .15) & (distance*predicted_scale <= cfg.max_range_m)])
        report = {'accepted': False, 'pose': prior.tolist(), 'scale': float(predicted_scale),
                  'reasons': [], 'source_points': len(raw), 'backend': self.backend,
                  'scale_estimated': bool(estimate_scale), 'scale_bounds': [float(low), float(high)],
                  'fit_scope': 'current map only, bounded by past-state prediction; no future frames',
                  'measurement_covariance_status': 'linearized heuristic with explicit floors, not calibrated'}
        if len(raw) < 64:
            report['reasons'] = ['insufficient_source']
            return report
        road_z = self.scene.road_height(prior)
        ground_seed = self._ground_initialization(raw, prior, predicted_scale, road_z) if bootstrap else None
        starts = [(float(seed), prior, 'supplied_pose') for seed in seeds]
        if ground_seed is not None:
            ground_pose, height_scale, detail = ground_seed
            report['ground_initialization'] = detail
            ground_scales = list(seeds)
            if estimate_scale and np.isfinite(height_scale) and low < height_scale < high:
                ground_scales.append(height_scale)
            starts.extend((float(seed), ground_pose, 'ground_plane') for seed in np.unique(ground_scales))
        candidates = []
        for seed, seed_pose, seed_type in starts:
            normal, ground, quality = self._geometry(raw, seed, seed_pose)
            pose, scale, stable, levels = fit_bounded(self.scene, raw, normal, ground, quality,
                                            prior, seed, road_z, cfg, self.pairs,
                                            estimate_scale=estimate_scale, initial_pose=seed_pose)
            # Validate geometry at the fitted metric scale and orientation,
            # rather than reusing a quality mask computed at the startup scale.
            normal, ground, quality = self._geometry(raw, scale, pose)
            pool = self.pairs(self.scene, raw, normal, ground, quality, pose, scale,
                              road_z, cfg.match_radii_m[-1], cfg)
            residual, jac = linear_terms(pool)
            train = ~pool['holdout']
            fit_cost = float(np.mean(np.minimum(residual[train]**2, 1.))) if train.any() else 1.
            fit_cost += 1.-min(1., train.sum()/max(1, cfg.min_train_patches*3))
            coverage_cost = self._coverage_cost(raw, pose, scale)
            fit_cost += coverage_cost
            candidates.append((fit_cost, pose, scale, stable, pool, residual, jac,
                               {'seed': seed, 'seed_type': seed_type, 'training_cost': fit_cost,
                                'fixed_source_coverage_cost': coverage_cost,
                                'converged': stable, 'levels': levels,
                                'valid_source_normals': int((quality > 0).sum())}))
        # Initialization selection depends on training geometry only.
        _, pose, scale, stable, pool, residual, jac, chosen = min(candidates, key=lambda v: v[0])
        report['optimizer'] = chosen
        report['candidate_summaries'] = [v[-1] for v in candidates]
        train, test = ~pool['holdout'], pool['holdout']
        units = np.array([1., 1., 1., .05, .05, .05, .1])
        j = jac[train]*units*np.sqrt(pool['quality'][train, None])
        nuisance_columns = [2, 3, 4] + ([6] if estimate_scale else [])
        if len(j):
            nuisance, horizontal = j[:, nuisance_columns], j[:, [0, 1, 5]]
            projected = horizontal-nuisance @ np.linalg.lstsq(nuisance, horizontal, rcond=1e-6)[0]
            singular = np.linalg.svd(projected, compute_uv=False)
            scale_col, other = j[:, 6], j[:, :6]
            scale_info = float(np.linalg.norm(scale_col-other @ np.linalg.lstsq(other, scale_col, rcond=1e-6)[0]))
        else:
            singular, scale_info = np.zeros(3), 0.
        test_residual = abs(residual[test])
        median = float(np.median(test_residual)) if len(test_residual) else None
        p90 = float(np.quantile(test_residual, .9)) if len(test_residual) else None
        ntrain = len(np.unique(pool['patch'][train], axis=0))
        ntest = len(np.unique(pool['patch'][test], axis=0))
        nonground = len(np.unique(pool['patch'][train & ~pool['ground']], axis=0))
        checks = {'optimizer_converged': stable,
                  'train_support': ntrain >= cfg.min_train_patches,
                  'test_support': ntest >= cfg.min_test_patches,
                  'nonground_support': nonground >= cfg.min_nonground_patches,
                  'horizontal_observability': len(singular) == 3 and singular[-1] >= cfg.min_scaled_singular,
                  'scale_observability': not estimate_scale or scale_info >= cfg.min_scaled_singular,
                  'heldout_residual': median is not None and median <= cfg.holdout_median_m and p90 <= cfg.holdout_p90_m,
                  'interior': (not estimate_scale or low*1.001 < scale < high/1.001)
                              and np.linalg.norm(pose[:3, 3]-prior[:3, 3]) < cfg.max_translation_m-.02
                              and Rotation.from_matrix(pose[:3, :3] @ prior[:3, :3].T).magnitude()
                                  < np.deg2rad(cfg.max_rotation_deg)-.0003}
        # Covariance uses training only. Weak directions get LARGE variance,
        # never zero variance from a pseudoinverse. Floors reflect unknown model
        # and patch correlation; this is not a calibrated sampling covariance.
        r = residual[train]
        weight = pool['quality'][train]*np.minimum(1., cfg.huber_m/np.maximum(abs(r), 1e-12))
        weighted = jac[train]*units*np.sqrt(weight[:, None])
        if not estimate_scale:
            weighted = weighted[:, :6]
        hessian = weighted.T @ weighted
        eigen, vectors = np.linalg.eigh(hessian)
        sigma2 = max(.05**2, float(np.mean(r*r)) if len(r) else 1.)
        dim = 7 if estimate_scale else 6
        inv = (vectors*(sigma2/np.maximum(eigen, 1e-8))) @ vectors.T
        covariance = np.eye(7)*1e6
        covariance[:dim, :dim] = units[:dim, None]*inv*units[None, :dim]
        floors = [c.position_std_floor_m]*3 + [np.deg2rad(c.rotation_std_floor_deg)]*3 + [c.log_scale_std_floor]
        covariance += np.diag(np.square(floors))
        report.update(accepted=bool(all(checks.values())), pose=pose.tolist(), scale=float(scale),
                      reasons=[k for k, v in checks.items() if not v], optimizer_converged=bool(stable),
                      train_patches=ntrain, test_patches=ntest, nonground_train_patches=nonground,
                      heldout_median_m=median, heldout_p90_m=p90,
                      horizontal_singular_values=singular.tolist(), marginalized_scale_singular=scale_info,
                      measurement_covariance=covariance.tolist(), road_height_m=float(road_z),
                      rotation_from_prediction_deg=float(np.rad2deg(Rotation.from_matrix(pose[:3, :3] @ prior[:3, :3].T).magnitude())))
        return report
