"""CUDA Gaussian coarse registration and exact-point fine ICP.

Coarse residuals use local Gaussian cells; fine residuals and nearest-point
selection both use Euclidean distance. Surface normals only test observability.
Point work stays on CUDA; the 7x7 pose/scale solve stays on the host.
"""
import itertools
import numpy as np
import cupy as cp
from .geometry import valid_pose
from .line_search import backtrack, step_is_small

OFFSET = 1 << 20
# +/-2 contains every centroid within the 1.5-leaf truncation radius, even
# when the query crosses a voxel boundary.
NEIGHBORS = cp.asarray(list(itertools.product(range(-2, 3), repeat=3)), dtype=cp.int64)


def sync():
    cp.cuda.get_current_stream().synchronize()


def keys(c):
    c = c.astype(cp.int64) + OFFSET
    if bool(cp.any((c < 0) | (c >= 2 * OFFSET))):
        raise ValueError('Point outside signed 21-bit voxel index range')
    return (c[..., 0] << 42) | (c[..., 1] << 21) | c[..., 2]


def voxel_mean(points, leaf, weights=None):
    p = cp.asarray(points, dtype=cp.float64)
    if not len(p): raise ValueError('Empty cloud')
    k = keys(cp.floor(p / leaf))
    order = cp.argsort(k); k = k[order]; p = p[order]
    starts = cp.r_[cp.array([0], dtype=cp.int64), cp.flatnonzero(k[1:] != k[:-1])+1]
    if weights is None: w = cp.ones(len(p), dtype=cp.float64)
    else: w = cp.asarray(weights, dtype=cp.float64)[order]
    count = cp.add.reduceat(w, starts)
    means = cp.add.reduceat(p*w[:, None], starts, axis=0) / count[:, None]
    return means, count


class SparseVoxelModelError(ValueError):
    """Expected absence of geometric support, distinct from numerical/CUDA errors."""
    def __init__(self, leaf, point_count, populated_voxels):
        self.diagnostics = dict(voxel_m=float(leaf), point_count=int(point_count),
                                populated_voxels=int(populated_voxels),
                                min_points_per_voxel=5, min_populated_voxels=7)
        super().__init__(f'Insufficient populated map voxels: {self.diagnostics}')


class VoxelModel:
    def __init__(self, points, leaf):
        self.leaf = leaf
        p = cp.asarray(points, dtype=cp.float64)
        if not len(p):
            raise SparseVoxelModelError(leaf, 0, 0)
        k = keys(cp.floor(p / leaf)); order = cp.argsort(k)
        k = k[order]; p = p[order]
        starts = cp.r_[cp.array([0], dtype=cp.int64), cp.flatnonzero(k[1:] != k[:-1])+1]
        counts = cp.diff(cp.r_[starts, len(p)])
        good = counts >= 5
        if int(good.sum()) < 7:
            raise SparseVoxelModelError(leaf, len(p), int(good.sum()))
        means = cp.add.reduceat(p, starts, axis=0) / counts[:, None]
        second = cp.add.reduceat(p[:, :, None]*p[:, None, :], starts, axis=0) / counts[:, None, None]
        cov = second-means[:, :, None]*means[:, None, :]
        cov = cov[good]; self.means = means[good]; self.keys = k[starts][good]
        val, vec = cp.linalg.eigh(cov)
        floor = cp.maximum(val[:, -1]*0.01, (leaf*0.05)**2)
        val = cp.maximum(val, floor[:, None])
        self.precision = (vec/val[:, None, :]) @ vec.transpose(0, 2, 1)
        self.normals = vec[:, :, 0]

    def correspond(self, q, mode='icp'):
        cells = cp.floor(q/self.leaf).astype(cp.int64)
        candidate_keys = keys(cells[:, None, :]+NEIGHBORS[None, :, :])
        ids = cp.searchsorted(self.keys, candidate_keys)
        ids = cp.minimum(ids, len(self.keys)-1)
        present = self.keys[ids] == candidate_keys
        residual = self.means[ids]-q[:, None, :]
        d2 = cp.sum(residual**2, axis=2)
        present &= d2 < (1.5*self.leaf)**2
        metric = cp.einsum('nci,ncij,ncj->nc',residual,self.precision[ids],residual) if mode=='ndt' else d2
        metric = cp.where(present,metric,cp.inf)
        choice = cp.argmin(metric, axis=1); rows = cp.arange(len(q))
        return ids[rows,choice],cp.isfinite(metric[rows,choice])


class PointModel:
    """Exact map-point ICP; normals are used separately for the quality gate."""
    def __init__(self, points, leaf):
        from .nearest import GpuNearest
        normal_model=VoxelModel(points,leaf)
        chunks=[normal_model.correspond(points[start:start+8192]) for start in range(0,len(points),8192)]
        ids=cp.concatenate([part[0] for part in chunks]);good=cp.concatenate([part[1] for part in chunks])
        normals=normal_model.normals[ids[good]]
        self.leaf=leaf
        self.nn=GpuNearest(points[good],cell=leaf)
        # GpuNearest sorts its map points by voxel keys.
        self.means=self.nn.points
        self.normals=normals[self.nn.order]

    def correspond(self,q,mode='icp'):
        ids,_=self.nn.query(q,1.5*self.leaf)
        return cp.maximum(ids,0),ids>=0


def jacobian(v):
    J = cp.zeros((len(v), 3, 7), dtype=cp.float64)
    J[:, :, :3] = cp.eye(3)
    J[:, 0, 4] = v[:, 2]; J[:, 0, 5] = -v[:, 1]
    J[:, 1, 3] = -v[:, 2]; J[:, 1, 5] = v[:, 0]
    J[:, 2, 3] = v[:, 1]; J[:, 2, 4] = -v[:, 0]
    J[:, :, 6] = v
    return J


def linearize(source, T, scale, model, mode, robust_m):
    v = source @ cp.asarray(T[:3, :3]).T * scale
    q = v + cp.asarray(T[:3, 3]); ids, good = model.correspond(q,mode)
    n = int(good.sum())
    if n < 30: return None
    ids = ids[good]; residual = q[good]-model.means[ids]
    J = jacobian(v[good])
    if mode == 'ndt':
        W = model.precision[ids]
        error2 = cp.einsum('ni,nij,nj->n', residual, W, residual)
        w = 1/(1+error2/9.0)
        H = cp.einsum('nki,nkl,nlj,n->ij', J, W, J, w)
        g = cp.einsum('nki,nkl,nl,n->i', J, W, residual, w)
        cost = cp.sum(9*cp.log1p(error2/9.0))
    else:
        # Correspondence AND the minimized objective use Euclidean distance.
        # The old Euclidean-match / plane-loss combination had discontinuous
        # cost at normal changes and could fail even at tiny line-search steps.
        error2=cp.sum(residual**2,axis=1)
        w=1/(1+error2/robust_m**2)
        H=cp.einsum('nki,nkj,n->ij',J,J,w)
        g=cp.einsum('nki,nk,n->i',J,residual,w)
        cost=cp.sum(robust_m**2*cp.log1p(error2/robust_m**2))
    # Full-cloud cost penalizes lost support. Never compare only surviving residuals.
    penalty = 9*np.log1p(9) if mode == 'ndt' else robust_m**2*np.log1p((1.5*model.leaf/robust_m)**2)
    score = (float(cost)+(len(source)-n)*penalty)/len(source)
    return cp.asnumpy(H), cp.asnumpy(g), score, n/len(source)


def normal_information(source,T,scale,model,robust_m):
    """Check surface observability independently of point-to-point ICP's H."""
    v=source@cp.asarray(T[:3,:3]).T*scale
    q=v+cp.asarray(T[:3,3]);ids,good=model.correspond(q)
    residual=q[good]-model.means[ids[good]]
    A=cp.einsum('ni,nij->nj',model.normals[ids[good]],jacobian(v[good]))
    w=1/(1+cp.sum(residual**2,axis=1)/robust_m**2)
    return cp.asnumpy((A.T*w)@A)


class Registration:
    def __init__(self, map_points, cfg):
        self.map = cp.asarray(map_points); self.cfg = cfg
        self.center = None; self.models = {}

    def align(self, source, prior, initial_scale, initialize=False):
        cfg = self.cfg; T = prior.copy(); scale = float(initial_scale)
        preprocessing=None;initialization=None
        if not valid_pose(T) or not np.isfinite(scale) or not cfg['scale_bounds'][0] < scale < cfg['scale_bounds'][1]:
            raise ValueError('Invalid registration prior pose or scale')
        p = cp.asarray(source, dtype=cp.float64)
        p = p[cp.all(cp.isfinite(p), axis=1)]
        if cfg.get('preprocessing',{}).get('enabled',False):
            from .preprocessing import filter_camera_cloud
            p,preprocessing,_=filter_camera_cloud(p,cfg['preprocessing'])
        self.display_source=cp.asnumpy(p)
        if len(p) < 30:
            return dict(accepted=False, reason='insufficient_source_points', T=T, scale=scale,
                        converged=False, observable=False, coverage=0., stages=[])
        if initialize and cfg.get('initializer',{}).get('enabled',False):
            from .initialization import GroundInitializer,GroundInitializationError
            try:
                seed=GroundInitializer(self.map,cfg['initializer']).align(p,T,scale)
            except GroundInitializationError as error:
                return dict(accepted=False,reason='initialization_support_failed',T=T,scale=scale,
                            preprocessing=preprocessing,initialization=dict(failure=str(error)),stages=[])
            initialization={k:v for k,v in seed.items() if k not in ('T','scale')}
            initialization.update(T=seed['T'].tolist(),scale=seed['scale'])
            T=seed['T'];scale=seed['scale']
        scale_reference=scale
        # Use a causal prior to select map region. No trajectory-derived crop.
        if self.center is None or np.max(np.abs(T[:3,3]-self.center)) > cfg['submap_rebuild_distance_m']:
            half = cfg['submap_half_extent_m']
            local = self.map[cp.all(cp.abs(self.map-cp.asarray(T[:3, 3])) < cp.asarray(half), axis=1)]
            if len(local) < 100: return dict(accepted=False, reason='empty_submap', T=T, scale=scale)
            # Build transactionally: a failed rebuild must not leave a mixed cache.
            try:
                models = {stage['voxel_m']: (VoxelModel(local,stage['voxel_m']) if stage['mode']=='ndt'
                          else PointModel(local,stage['voxel_m'])) for stage in cfg['stages']}
            except SparseVoxelModelError as error:
                return dict(accepted=False, reason='insufficient_populated_map_voxels',
                            converged=False, observable=False, coverage=0., T=T, scale=scale,
                            model_diagnostics=error.diagnostics, stages=[])
            self.models = models
            self.center = T[:3,3].copy()
        converged = False; coverage = 0.; score = np.inf; Hfinal = None
        stage_records = []
        for stage in cfg['stages']:
            leaf, mode = stage['voxel_m'], stage['mode']
            model = self.models[leaf]
            sample, _ = voxel_mean(p*scale, max(leaf/3, 0.15)); sample = sample/scale
            if len(sample) > cfg['max_source_points']:
                sample = sample[cp.linspace(0, len(sample)-1, cfg['max_source_points']).astype(cp.int64)]
            converged = False; Hfinal = None; coverage = 0.; score = np.inf
            record = dict(mode=mode, voxel_m=leaf, source_points=len(sample),
                          iterations=0, stop_reason='max_iterations')
            def evaluate(t, s):
                data = linearize(sample, t, s, model, mode, cfg['robust_m'])
                if data is None: return None
                H, g, cost, cov = data
                # Log-scale random walk prior, not a metric calibration claim.
                sig = cfg['initial_log_scale_sigma'] if initialize else cfg['log_scale_sigma']
                e = np.log(s/scale_reference)
                H[6, 6] += len(sample)/sig**2
                g[6] += len(sample)*e/sig**2
                return H, g, cost+e*e/sig**2, cov
            for iteration in range(stage['iterations']):
                record['iterations'] = iteration+1
                before = evaluate(T, scale)
                if before is None:
                    record['stop_reason'] = 'insufficient_correspondences'
                    break
                H, g, score, coverage = before
                record.setdefault('initial_cost', float(score))
                diag = np.maximum(np.diag(H), 1e-6)
                delta = np.linalg.solve(H+np.diag(diag)*1e-5, -g)
                # Trust-region limits in metres, radians and log scale.
                f = max(1., np.linalg.norm(delta[:3])/2., np.linalg.norm(delta[3:6])/np.deg2rad(5), abs(delta[6])/0.1)
                delta /= f
                record['proposed_step'] = dict(translation_m=float(np.linalg.norm(delta[:3])),
                                              rotation_rad=float(np.linalg.norm(delta[3:6])),
                                              log_scale=float(delta[6]))
                candidate,trials = backtrack(evaluate,T,scale,delta,score,g,len(sample),cfg['scale_bounds'])
                record['last_line_search'] = trials
                if candidate is None:
                    converged = step_is_small(delta)
                    Hfinal = H
                    record['stop_reason'] = 'step_small_no_descent' if converged else 'backtracking_failed'
                    break
                T,scale,after,factor = candidate
                score,coverage = after[2:]; Hfinal = after[0]
                record['min_accepted_factor'] = min(record.get('min_accepted_factor',1.),factor)
                # A tiny line-search fraction alone is not evidence of stationarity.
                converged = step_is_small(delta)
                if converged: record['stop_reason'] = 'step_small'
                if converged: break
            record.update(converged=bool(converged), final_cost=float(score) if np.isfinite(score) else None,
                          coverage=float(coverage))
            stage_records.append(record)
        observable = False; min_eigenvalue = None
        if Hfinal is not None:
            Hgeometry=normal_information(sample,T,scale,model,cfg['robust_m'])
            Hgeometry[6,6]+=len(sample)/(cfg['initial_log_scale_sigma'] if initialize else cfg['log_scale_sigma'])**2
            spatial = Hgeometry[:6,:6] - np.outer(Hgeometry[:6,6],Hgeometry[6,:6])/max(Hgeometry[6,6],1e-12)
            diagonal = np.diag(spatial)
            if np.min(diagonal)>1e-10:
                normalized = spatial / np.sqrt(diagonal[:,None]*diagonal[None,:])
                min_eigenvalue = float(np.linalg.eigvalsh(normalized).min())
                observable = min_eigenvalue > 1e-6
        candidate_valid = bool(valid_pose(T) and np.isfinite(scale) and cfg['scale_bounds'][0] < scale < cfg['scale_bounds'][1])
        accepted = bool(converged and observable and coverage >= cfg['min_coverage'] and candidate_valid)
        reason = ('accepted' if accepted else
                  'invalid_candidate_pose_or_scale' if not candidate_valid else
                  'not_converged:'+stage_records[-1]['stop_reason'] if not converged else
                  'degenerate_geometry' if not observable else 'low_coverage')
        return dict(T=T, scale=scale, accepted=accepted, converged=bool(converged), reason=reason,
                    preprocessing=preprocessing,initialization=initialization,
                    stages=stage_records,
                    observable=observable, normalized_min_eigenvalue=min_eigenvalue,
                    coverage=coverage, cost=float(score) if np.isfinite(score) else None, hessian=Hfinal)
