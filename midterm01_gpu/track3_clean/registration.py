"""Stateless current-source bounded NDT/ICP; no knowledge of trajectory history."""
from copy import copy
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .config import RegistrationConfig
from .contracts import GeometryUnavailable
from .geometry import (Geometry, xyz, checked_pose, block_validation, source_weights,
                       target_planes, voxel_statistics, point_jacobian)

class MapRegistrar:
    def with_config(self, config):
        """Share immutable map tensors/index, without mutating another fit's domain."""
        view = copy(self)
        view.config = config
        if config.ndt_resolutions != self.config.ndt_resolutions or config.normal_radius_m != self.config.normal_radius_m:
            raise ValueError('Changing map geometry requires a new registrar')
        return view

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
            raise GeometryUnavailable('Insufficient training source blocks')
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


