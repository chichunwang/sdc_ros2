"""Map-only Sim(3) registration laboratory. No IMU, motion filter or trajectory.

P2D NDT uses regularized voxel Gaussians and a bounded robust Mahalanobis
objective (not PCL's exact likelihood). Fine ICP uses robust point-to-point
distance, with target planes used ONLY for observability. Fixed membership pays an outlier cost for lost
matches. Only training blocks select steps/seeds; validation is diagnostic.
Dense geometry runs in float64 CUDA. cpu_reference is explicit, for tests only.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch


@dataclass(frozen=True)
class RegistrationConfig:
    ndt_resolutions: tuple = (2., 1.)
    icp_distances: tuple = (1.5, .75)
    iterations: int = 150
    min_scale: float = .5
    max_scale: float = 16.
    translation_domain_m: float = 8.
    rotation_domain_deg: float = 30.
    normal_radius_m: float = 1.5
    min_support: int = 80
    min_coverage: float = .6
    validation_median_m: float = .35
    validation_p90_m: float = 1.
    robust_c2: float = 9.
    consensus_translation_m: float = .5
    consensus_rotation_deg: float = 2.
    consensus_log_scale: float = .1

    def __post_init__(self):
        values = [v for v in asdict(self).values() if not isinstance(v, tuple)]
        if (not all(np.isfinite(v) and v > 0 for v in values)
                or self.min_scale >= self.max_scale or self.iterations != int(self.iterations)
                or not self.ndt_resolutions or not self.icp_distances
                or any(not np.isfinite(v) or v <= 0 for v in self.ndt_resolutions+self.icp_distances)
                or self.min_coverage > 1):
            raise ValueError('Invalid registration configuration')


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
            from cuda_scale_geometry import CudaKernels
            self.kernels = CudaKernels(Path(cache_dir))

    def tensor(self, value):
        return torch.as_tensor(np.ascontiguousarray(value), device=self.device, dtype=torch.float64)

    def tree(self, points, leaf=1.5):
        if self.backend == 'cuda':
            from cuda_scale_geometry import CudaGrid
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
            from cuda_scale_geometry import _normal_eigh
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
        raise ValueError('Insufficient supported NDT voxels')
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


class MapRegistrar:
    def __init__(self, points, *, backend='cuda', cache_dir=None, config=None, geometry=None):
        self.config = config or RegistrationConfig()
        self.geometry = geometry if geometry is not None else Geometry(backend, cache_dir)
        if self.geometry.backend != backend:
            raise ValueError('Shared geometry backend mismatch')
        self.points = xyz(points)
        self.target = self.geometry.tensor(self.points)
        self.tree = self.geometry.tree(self.points)
        self.normal, self.planar = target_planes(self.target, self.tree, self.geometry,
                                                  self.config.normal_radius_m)
        self.ndt = []
        for resolution in self.config.ndt_resolutions:
            mean, precision = voxel_statistics(self.target, resolution, self.geometry)
            self.ndt.append((resolution, mean, precision,
                             self.geometry.tree(mean.cpu().numpy(), leaf=resolution)))
        self.geometry.sync()

    def _pairs(self, raw, pose, scale, kind, data, radius):
        g = self.geometry
        rotated = scale*raw @ g.tensor(pose[:3, :3]).T
        world = rotated+g.tensor(pose[:3, 3])
        if kind == 'ndt':
            _, means, precisions, tree = data
            d, ids = g.query(tree, world, 4, radius)
            residual = world[:, None]-means[ids]
            squared = torch.einsum('nki,nkij,nkj->nk', residual, precisions[ids], residual)
            squared = torch.where(torch.isfinite(d), squared, torch.inf)
            best = squared.argmin(1)
            row = torch.arange(len(raw), device=raw.device)
            ids = ids[row, best]
            good = torch.isfinite(squared[row, best])
            return rotated, world-means[ids], precisions[ids], good, ids
        d, ids = g.query(self.tree, world, 1, radius)
        ids = ids[:, 0]
        eye = torch.eye(3, device=raw.device, dtype=raw.dtype)
        precision = (eye/.15**2).expand(len(raw), -1, -1)
        good = torch.isfinite(d[:, 0])
        return rotated, world-self.target[ids], precision, good, ids

    def _loss(self, residual, precision, good, cap=None, weights=None):
        r2 = torch.einsum('ni,nij,nj->n', residual, precision, residual).clamp_min(0)
        c2 = self.config.robust_c2
        # Bounded Geman-McClure. Missing points pay exactly 1, never vanish.
        # ICP is CONTINUOUS at its search radius: a point crossing the radius
        # pays the same saturated cost on either side of that boundary.
        if cap is None:
            rho = r2/(r2+c2)
        else:
            limited = r2.clamp_max(cap)
            rho = (limited/(limited+c2))/(cap/(cap+c2))
        rho = torch.where(good, rho, torch.ones_like(r2))
        return (rho.mean() if weights is None else (rho*weights).sum()/weights.sum()), r2

    def _inside(self, pose, scale, anchor):
        c = self.config
        return (c.min_scale < scale < c.max_scale
                and np.linalg.norm(pose[:3, 3]-anchor[:3, 3]) < c.translation_domain_m
                and Rotation.from_matrix(pose[:3, :3] @ anchor[:3, :3].T).magnitude()
                    < np.deg2rad(c.rotation_domain_deg))

    def fit(self, raw, initial, scale, *, estimate_scale=True, use_ndt=True, train_mask=None,
            training_weights=None):
        raw = xyz(raw)
        initial = checked_pose(initial)
        if not np.isfinite(scale) or not self.config.min_scale < scale < self.config.max_scale:
            raise ValueError('Scale seed must be inside explicit domain')
        mask = ~block_validation(raw) if train_mask is None else np.asarray(train_mask, bool)
        if mask.shape != (len(raw),) or mask.sum() < self.config.min_support:
            raise ValueError('Insufficient training source blocks')
        source = self.geometry.tensor(raw[mask])
        if training_weights is None:
            weights = source_weights(source)
        else:
            supplied = np.asarray(training_weights, dtype=np.float64)
            if (supplied.shape != (len(raw),) or not np.isfinite(supplied[mask]).all()
                    or np.any(supplied[mask] <= 0)):
                raise ValueError('Expected positive finite immutable training weights per source row')
            weights = self.geometry.tensor(supplied[mask])
            weights = weights/weights.mean()
        pose, scale = initial.copy(), float(scale)
        levels = []
        stages = [('ndt', d, 2*d[0]) for d in self.ndt] if use_ndt else []
        stages += [('icp', None, d) for d in self.config.icp_distances]
        for kind, data, radius in stages:
            status = dict(kind=kind, radius_m=radius, converged=False, iterations=0, stop='iteration_limit')
            plateau = 0
            for it in range(self.config.iterations):
                rotated, residual, precision, good, ids = self._pairs(source, pose, scale, kind, data, radius)
                status.update(iterations=it+1, support=int(good.sum()))
                if int(good.sum()) < self.config.min_support:
                    status['stop'] = 'insufficient_support'
                    break
                cap = (radius/.15)**2 if kind == 'icp' else None
                cost, r2 = self._loss(residual, precision, good, cap, weights)
                jac = point_jacobian(rotated, estimate_scale)
                weight = self.config.robust_c2/(r2+self.config.robust_c2).square()*good*weights
                jt = jac.transpose(1, 2) @ precision
                h = (jt @ jac*weight[:, None, None]).sum(0).cpu().numpy()
                b = ((jt @ residual[:, :, None])[:, :, 0]*weight[:, None]).sum(0).cpu().numpy()
                units = np.r_[np.ones(3), [.05]*3, [.1] if estimate_scale else []]
                # Tiny at most 7x7 solve; damping is numerical, not a pose prior.
                hs = h*units[:, None]*units[None, :]
                step = np.linalg.lstsq(hs+np.eye(len(units))*max(np.trace(hs), 1.)*1e-8,
                                     -b*units, rcond=1e-8)[0]*units
                factor = min(1., .75/max(np.linalg.norm(step[:3]), 1e-12),
                             np.deg2rad(3)/max(np.linalg.norm(step[3:6]), 1e-12))
                if estimate_scale:
                    factor = min(factor, .08/max(abs(step[6]), 1e-12))
                step *= factor
                if (np.linalg.norm(step[:3]) < .001 and np.linalg.norm(step[3:6]) < 1e-4
                        and (not estimate_scale or abs(step[6]) < 1e-4)):
                    status.update(converged=True, stop='small_step')
                    break
                moved = False
                for alpha in (1., .5, .25, .125, .0625, .03125):
                    candidate = pose.copy()
                    candidate[:3, :3] = Rotation.from_rotvec(alpha*step[3:6]).as_matrix() @ pose[:3, :3]
                    candidate[:3, 3] += alpha*step[:3]
                    s = scale*np.exp(alpha*step[6]) if estimate_scale else scale
                    if not self._inside(candidate, s, initial):
                        continue
                    # Full FIXED training source, fresh matches, same loss.
                    _, rr, pp, gg, _ = self._pairs(source, candidate, s, kind, data, radius)
                    new_cost, _ = self._loss(rr, pp, gg, cap, weights)
                    if float(new_cost) < float(cost)-1e-12:
                        drop = float(cost-new_cost)
                        pose, scale, moved = candidate, float(s), True
                        small = (alpha*np.linalg.norm(step[:3]) < .01
                                 and alpha*np.linalg.norm(step[3:6]) < .001
                                 and (not estimate_scale or alpha*abs(step[6]) < .002))
                        plateau = plateau+1 if drop < 1e-5 and small else 0
                        status.update(cost=float(new_cost), last_step=(alpha*step).tolist())
                        break
                if not moved:
                    status['stop'] = 'no_descent_or_domain'
                    break
                if plateau >= 3:
                    status.update(converged=True, stop='cost_and_step_stable')
                    break
            levels.append(status)
        report = self.evaluate(raw, pose, scale, mask, estimate_scale)
        c = self.config
        interior = (scale > c.min_scale*1.01 and scale < c.max_scale/1.01
                    and np.linalg.norm(pose[:3, 3]-initial[:3, 3]) < c.translation_domain_m-.05
                    and Rotation.from_matrix(pose[:3, :3] @ initial[:3, :3].T).magnitude()
                        < np.deg2rad(c.rotation_domain_deg)-.001)
        checks = dict(converged=levels[-1]['converged'], interior=bool(interior),
                      coverage=report['validation']['coverage'] >= c.min_coverage,
                      validation=report['validation']['median_m'] <= c.validation_median_m
                                 and report['validation']['p90_m'] <= c.validation_p90_m,
                      observable=report['physical_rank'] == (7 if estimate_scale else 6))
        report.update(pose=pose.tolist(), scale=scale, levels=levels, checks=checks,
                      accepted=bool(all(checks.values())), reasons=[k for k,v in checks.items() if not v],
                      backend=self.geometry.backend, estimate_scale=estimate_scale,
                      accuracy='unknown: validation is geometry, not ground-truth pose error')
        self.geometry.sync()
        return report

    def evaluate(self, raw, pose, scale, train_mask, estimate_scale):
        source = self.geometry.tensor(raw)
        world = scale*source @ self.geometry.tensor(pose[:3, :3]).T+self.geometry.tensor(pose[:3, 3])
        distance, _ = self.geometry.query(self.tree, world, 1, 5.)
        d = distance[:, 0].clamp_max(5.).cpu().numpy()
        result = {}
        for name, keep in [('training', train_mask), ('validation', ~train_mask)]:
            v = d[keep]
            result[name] = dict(points=len(v), coverage=float(np.mean(v < .75)) if len(v) else 0.,
                                median_m=float(np.median(v)) if len(v) else 5.,
                                p90_m=float(np.quantile(v, .9)) if len(v) else 5.,
                                capped_rmse_m=float(np.sqrt(np.mean(v*v))) if len(v) else 5.)
        rotated, residual, precision, good, ids = self._pairs(source, pose, scale, 'icp', None, .75)
        # Data-only point-to-plane Hessian; never invent observability from the
        # point-sampling correspondences, damping or a motion prior.
        good &= torch.as_tensor(train_mask, device=source.device) & self.planar[ids]
        j = torch.einsum('ni,nij->nj', self.normal[ids[good]], point_jacobian(rotated[good], estimate_scale))
        units = self.geometry.tensor(np.r_[np.ones(3), [.05]*3, [.1] if estimate_scale else []])
        h = ((j*units).T @ (j*units)).cpu().numpy()/max(len(j), 1)
        eigen = np.linalg.eigvalsh(h).clip(min=0)
        rank = int(np.sum(eigen > max(eigen[-1]*1e-5, 1e-8)))
        result.update(physical_rank=rank, physical_hessian_eigenvalues=eigen.tolist(),
                      physical_support=len(j), training_cost=result['training']['capped_rmse_m'])
        return result


def choose_training_candidate(candidates):
    """Converged/interior candidates first; validation never selects the seed."""
    if not candidates:
        raise ValueError('No registration candidates')
    return min(candidates, key=lambda r: (not (r['checks']['converged'] and r['checks']['interior']), r['training_cost']))


def assess_multistart(candidates, config=None):
    """Reject geometrically plausible competing basins, without validation tuning.

    These are reproducibility tolerances, NOT calibrated pose-error bounds.
    Passing local checks is insufficient if independent starts disagree.
    """
    c = config or RegistrationConfig()
    chosen = choose_training_candidate(candidates)
    usable = [r for r in candidates if r['checks']['converged'] and r['checks']['interior']
              and r['training']['coverage'] >= c.min_coverage
              and r['training']['median_m'] <= c.validation_median_m
              and r['training']['p90_m'] <= c.validation_p90_m]
    best = min((r['training_cost'] for r in usable), default=0.)
    plausible = [r for r in usable if r['training_cost'] <= best*1.5+.05]
    discrepancies = []
    for i, a in enumerate(plausible):
        for b in plausible[i+1:]:
            p, q = np.array(a['pose']), np.array(b['pose'])
            dt = float(np.linalg.norm(p[:3, 3]-q[:3, 3]))
            dr = float(np.rad2deg(Rotation.from_matrix(p[:3, :3] @ q[:3, :3].T).magnitude()))
            ds = float(abs(np.log(a['scale']/b['scale'])))
            if dt > c.consensus_translation_m or dr > c.consensus_rotation_deg or ds > c.consensus_log_scale:
                discrepancies.append(dict(translation_m=dt, rotation_deg=dr, log_scale=ds,
                                          scale_a=a['scale'], scale_b=b['scale']))
    enough = len(plausible) >= 2
    report = dict(selected=chosen, plausible_training_basins=len(plausible),
                  initial_value_consistent=enough and not discrepancies,
                  ambiguities=discrepancies, accepted=bool(chosen['accepted'] and enough and not discrepancies),
                  status='geometry only; consensus does not establish real pose accuracy')
    report['reasons'] = list(chosen['reasons'])
    if not enough:
        report['reasons'].append('insufficient_independent_initializations')
    if discrepancies:
        report['reasons'].append('ambiguous_initialization')
    return report
