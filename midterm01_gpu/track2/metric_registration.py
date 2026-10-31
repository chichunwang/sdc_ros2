"""因果地面基準 Sim(3)：當幀地面定傾斜，非地面結構解 XY/yaw/scale。

GNSS 是有雜訊的 XY/yaw prior；相機高度由官方 initial pose 與 XYZ 地面
初始化，之後只用過去接受的幾何估計更新。Ground-only 的尺度不冒稱
非地面 map registration。所有點級搜尋、plane、H/g 在 CUDA。
"""
import time
import numpy as np
import cupy as cp
from scipy.spatial.transform import Rotation
from track3.gpu import voxel_mean
from track3.initialization import fit_plane, plane_basis, transform4, GroundInitializationError
from track3.nearest import GpuNearest
from . import parameters as p
from .correction_filter import CausalCorrectionFilter, pose_correction, yaw
from .plane_geometry import PlaneGeometry
from .observability import checked_covariance, geometry_observation


def wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


class MetricRegistration:
    def __init__(self, map_xyz, initial_pose):
        self.map = cp.asarray(map_xyz, dtype=cp.float64)
        self.initial_pose = np.asarray(initial_pose).copy()
        self.height = None
        self.height_variance = p.CAMERA_HEIGHT_STD_M**2
        self.last_stamp = None
        self.last_scale = float(p.PLY_SCALE)
        self.center = None
        self.nn = None
        self.source_features = None
        self.correction_filter = CausalCorrectionFilter()

    def prepare(self, raw, prior, prior_covariance):
        self.prior_pose = np.asarray(prior).copy()
        self.prior_covariance = checked_covariance(prior_covariance)
        self.prior_precision = np.linalg.inv(self.prior_covariance)
        self.yaw_search_bound = np.deg2rad(max(p.METRIC_MAX_YAW_CORRECTION_DEG,
            min(p.METRIC_SEARCH_YAW_HARD_DEG,
                p.METRIC_SEARCH_YAW_SIGMAS*np.degrees(np.sqrt(self.prior_covariance[2, 2])))))
        raw = cp.asarray(raw, dtype=cp.float64)
        raw = raw[cp.all(cp.isfinite(raw), axis=1) & (raw[:, 0] > 0)]
        if len(raw) < 100:
            raise GroundInitializationError('Insufficient masked forward source points')
        center = cp.asarray(prior[:3, 3])
        local = self.map[cp.all(cp.abs(self.map-center) < cp.asarray(p.METRIC_SUBMAP_HALF_EXTENT_M), axis=1)]
        if len(local) < 100:
            raise GroundInitializationError('Insufficient local map XYZ')
        near = local[cp.linalg.norm(local[:, :2]-center[:2], axis=1) < p.GROUND_RADIUS_M]
        if len(near) < 100:
            raise GroundInitializationError('Insufficient map ground candidates')
        cells = cp.floor(near[:, :2]/p.GROUND_GRID_M).astype(cp.int64)
        order = cp.lexsort(cp.stack((near[:, 2], cells[:, 1], cells[:, 0])))
        cells = cells[order]
        starts = cp.r_[0, cp.flatnonzero(cp.any(cells[1:] != cells[:-1], axis=1))+1]
        nm, dm, md = fit_plane(near[order[starts]], p.MAP_GROUND_THRESHOLD_M, True)
        ns, ds, sd = fit_plane(raw, p.SOURCE_GROUND_THRESHOLD_RAW)
        if nm[2] < .95 or ns[2] < .85 or ds <= .02:
            raise GroundInitializationError('Ground axes/height invalid for FLU camera')
        if md['mad'] > p.MAX_MAP_GROUND_MAD_M or sd['inliers'] < p.MIN_SOURCE_GROUND_POINTS:
            raise GroundInitializationError('Unreliable map/source ground plane')
        if self.height is None:
            # Uses supplied initial pose, never an accumulated fixed-scale z.
            initial_height = float(nm @ self.initial_pose[:3, 3]+dm)
            if not p.CAMERA_HEIGHT_BOUNDS_M[0] < initial_height < p.CAMERA_HEIGHT_BOUNDS_M[1]:
                raise GroundInitializationError('Initial pose does not provide a plausible camera-ground height')
            self.height = initial_height
        scale_seed = self.height/ds
        if not p.SCALE_BOUNDS[0] < scale_seed < p.SCALE_BOUNDS[1]:
            raise GroundInitializationError('Ground-height scale seed outside explicit bounds')
        if sd['mad']*scale_seed > p.MAX_SOURCE_GROUND_MAD_M:
            raise GroundInitializationError('Metric source ground thickness exceeds allowance')
        Bs, Bm = plane_basis(ns), plane_basis(nm)
        origin = prior[:3, 3]-nm*(nm @ prior[:3, 3]+dm)
        source = raw @ cp.asarray(Bs)
        source[:, 2] += ds
        non_ground = source[:, 2] > max(p.MIN_SOURCE_FEATURE_HEIGHT_RAW, 6*sd['mad'])
        distance = cp.linalg.norm(raw, axis=1)*scale_seed
        features = source[non_ground & (distance <= p.FEATURE_MAX_RANGE_M)]
        if len(features) < 30:
            raise GroundInitializationError('Insufficient non-ground source structure')
        sample, _ = voxel_mean(features, p.SOURCE_FEATURE_LEAF_RAW)
        if len(sample) > p.MAX_METRIC_SOURCE_POINTS:
            sample = sample[cp.linspace(0, len(sample)-1, p.MAX_METRIC_SOURCE_POINTS).astype(cp.int64)]
        target = (local-cp.asarray(origin)) @ cp.asarray(Bm)
        target = target[target[:, 2] > max(p.MIN_MAP_FEATURE_HEIGHT_M, 3*md['mad'])]
        if len(target) < 30:
            raise GroundInitializationError('Insufficient non-ground map structure')
        rays = sample-cp.asarray([0., 0., ds])
        az = cp.arctan2(rays[:, 1], rays[:, 0])
        el = cp.arctan2(rays[:, 2], cp.linalg.norm(rays[:, :2], axis=1))
        bins = cp.floor(cp.stack([az, el], axis=1)/p.ANGULAR_CELL_RAD).astype(cp.int64)
        # CuPy 13 unique(axis=0) compares rows in a Python loop with repeated
        # device synchronization. Encode each angular cell as one int64 key,
        # so sorting/counting stays on CUDA and avoids quadratic host work.
        offset = 1 << 20
        if bool(cp.any((bins < -offset) | (bins >= offset))):
            raise ValueError('Angular cell index outside signed 21-bit range')
        angular_key = ((bins[:,0]+offset) << 21) | (bins[:,1]+offset)
        unique, inv, counts = cp.unique(angular_key, return_inverse=True, return_counts=True)
        # Each angular cell contributes one aggregate depth observation; this
        # engineering correlation model avoids counting every pixel independently.
        weight = 1/counts[inv]
        self.source_features = sample
        self.raw_features = (sample-cp.asarray([0., 0., ds])) @ cp.asarray(Bs).T
        self.weight = weight
        self.origin = origin; self.Bm = Bm; self.Bs = Bs
        self.ds = ds; self.nm = nm; self.ns = ns; self.dm = dm
        self.feature_nn = GpuNearest(target, cell=1.5)
        self.geometry = PlaneGeometry(self.feature_nn)
        # RViz presents the local XYZ map, rather than the ground-relative model.
        self.nn = type('DisplaySubmap', (), {'points': local})()
        self.center = prior[:3, 3].copy()
        f = Bm.T @ (prior[:3, :3] @ Bs[:, 0])
        yaw0 = float(np.arctan2(f[1], f[0]))
        self.seed = np.array([0., 0., yaw0, np.log(scale_seed)])
        self.seed = self.state_for_correction(prior, np.zeros(3), scale_seed)
        return dict(source_ground=dict(normal=ns.tolist(), offset=ds, **sd),
                    map_ground=dict(normal=nm.tolist(), offset=dm, **md),
                    scale_seed=scale_seed, height_prior_m=self.height,
                    height_std_m=float(np.sqrt(self.height_variance)),
                    source_features=len(sample), angular_cells=len(unique), map_features=len(target),
                    feature_max_range_m=p.FEATURE_MAX_RANGE_M,
                    prior_covariance_xy_yaw=self.prior_covariance.tolist(),
                    prior_frame='map [x_m,y_m,yaw_rad] with full cross covariance',
                    correlation_model='one aggregate observation per angular cell; uncalibrated')

    def pose(self, x):
        scale = float(np.exp(x[3]))
        T = np.eye(4)
        T[:3, :3] = self.Bm @ Rotation.from_euler('z', x[2]).as_matrix() @ self.Bs.T
        T[:3, 3] = self.origin+self.Bm @ np.array([x[0], x[1], scale*self.ds])
        return T, scale

    def correction_jacobian(self, x):
        """[plane dx,dy,yaw,logscale] -> world XY/yaw correction。"""
        T, scale = self.pose(x)
        forward = T[:3, 0]
        derivative = np.cross(self.nm, forward)
        dyaw = (forward[0]*derivative[1]-forward[1]*derivative[0])/np.dot(forward[:2], forward[:2])
        J = np.zeros((3, 4))
        J[:2, :2] = self.Bm[:2, :2]
        J[:2, 3] = self.nm[:2]*scale*self.ds
        J[2, 2] = dyaw
        return J

    def state_for_correction(self, prior, correction, scale):
        """將 world correction 放回當幀地面模型，保留地面法向／高度關係。

不能直接把上一幀 plane dx/dy 搬到新基底，也不能直接旋轉 world Z
破壞 ground alignment。用小矩陣反解 XY，以單變數 Newton 對齊 world yaw。
"""
        x = self.seed.copy(); x[3] = np.log(scale)
        target_xy = prior[:2, 3]+correction[:2]
        x[:2] = np.linalg.solve(self.Bm[:2, :2], target_xy-self.origin[:2]-self.nm[:2]*scale*self.ds)
        target_yaw = yaw(prior)+correction[2]
        for _ in range(6):
            T, _ = self.pose(x)
            error = wrap(target_yaw-yaw(T))
            if abs(error) < 1e-12:
                break
            x[2] += error/self.correction_jacobian(x)[2, 2]
        return x

    def plane_residuals(self, q, radius):
        ids, d2 = self.feature_nn.query(q, radius)
        self.geometry.ensure(ids)
        safe = cp.maximum(ids, 0)
        good = (ids >= 0) & self.geometry.reliable[safe]
        normals = self.geometry.normal[safe]
        residual = cp.sum(normals*(q-self.geometry.center[safe]), axis=1)
        return ids, d2, good, normals, residual, self.geometry.confidence[safe]

    def evaluate(self, x, radius, robust, *, derivatives=False, include_prior=True,
                 association=None, diagnostics=True):
        # Solver trials only consume cost, matches and H/g. Sorting residuals
        # for medians and copying diagnostic scalars forces needless CUDA work
        # and host synchronization at every line-search trial.
        q, J = transform4(self.source_features, x, jacobian=derivatives)
        if association is None:
            ids, d2, good, normals, residual, confidence = self.plane_residuals(q, radius)
            centers = self.geometry.center[cp.maximum(ids, 0)]
            association = dict(ids=ids, good=good, normals=normals, centers=centers, confidence=confidence)
        else:
            ids = association['ids']; good = association['good']
            normals = association['normals']; confidence = association['confidence']
            residual = cp.sum(normals*(q-association['centers']), axis=1)
            if diagnostics:
                d2 = cp.minimum(cp.sum((q-self.feature_nn.points[cp.maximum(ids, 0)])**2, axis=1), radius**2)
        w = self.weight
        unmatched = .5*robust**2*np.log1p(radius**2/robust**2)/p.FEATURE_STD_M**2
        loss = .5*robust**2*cp.log1p(residual**2/robust**2)/p.FEATURE_STD_M**2
        loss = cp.where(good, unmatched-confidence*(unmatched-loss), unmatched)
        data_cost = float(cp.sum(w*loss))
        H = np.zeros((4, 4)); g = np.zeros(4)
        matches = int(good.sum()) if derivatives or diagnostics else None
        if derivatives and matches >= 20:
            projected_J = cp.einsum('ni,nij->nj', normals[good], J[good])
            ww = w[good]*confidence[good]/(1+residual[good]**2/robust**2)/p.FEATURE_STD_M**2
            H = cp.asnumpy(cp.einsum('ni,nj,n->ij', projected_J, projected_J, ww))
            g = cp.asnumpy(cp.einsum('ni,n,n->i', projected_J, residual[good], ww))
        data_H = H.copy()
        prior_cost = 0.
        if include_prior:
            T, scale = self.pose(x)
            r = pose_correction(T, self.prior_pose)
            A = self.correction_jacobian(x)
            h = scale*self.ds; hr = h-self.height
            hvar = self.height_variance+p.HEIGHT_OBSERVATION_FLOOR_M**2
            prior_cost = float(.5*r @ self.prior_precision @ r+.5*hr**2/hvar)
            H += A.T @ self.prior_precision @ A
            g += A.T @ self.prior_precision @ r
            H[3, 3] += h*h/hvar; g[3] += h*hr/hvar
        result = dict(cost=data_cost+prior_cost, data_cost=data_cost, prior_cost=prior_cost,
                      matches=matches, association=association, H=H, data_H=data_H, g=g)
        if not diagnostics:
            return result
        coverage = float(cp.sum(w*good)/w.sum())
        plane_median = float(cp.median(cp.abs(residual[good]))) if bool(good.any()) else float(radius)
        result.update(feature_cost=2*data_cost*p.FEATURE_STD_M**2/float(w.sum()),
                    coverage=coverage, median_capped_distance_m=float(cp.median(cp.sqrt(d2))),
                    nn_coverage=float(cp.sum(w*(ids >= 0))/w.sum()),
                    median_plane_residual_m=plane_median,
                    angular_support=float(cp.sum(w*good)))
        return result

    def score_pose(self, T, scale):
        """Same fixed raw sample, current map, and point-to-plane loss; no output changes."""
        q = self.raw_features @ cp.asarray(T[:3,:3]).T*scale+cp.asarray(T[:3,3])
        q = (q-cp.asarray(self.origin)) @ cp.asarray(self.Bm)
        radius, robust, _ = p.METRIC_STAGES[-1]
        ids, d2, good, _, residual, confidence = self.plane_residuals(q, radius)
        unmatched = robust**2*np.log1p(radius**2/robust**2)
        loss = robust**2*cp.log1p(residual**2/robust**2)
        loss = cp.where(good, unmatched-confidence*(unmatched-loss), unmatched)
        return dict(feature_cost=float(cp.sum(self.weight*loss)/self.weight.sum()),
                    coverage=float(cp.sum(self.weight*good)/self.weight.sum()),
                    median_capped_distance_m=float(cp.median(cp.sqrt(d2))),
                    median_plane_residual_m=float(cp.median(cp.abs(residual[good]))) if bool(good.any()) else float(radius),
                    source_features=len(q), comparison='same_raw_features_and_map_point_to_plane_with_unmatched_penalty')

    def valid(self, x):
        if not np.isfinite(x).all():
            return False
        s = np.exp(x[3]); h = s*self.ds
        return bool(p.SCALE_BOUNDS[0] < s < p.SCALE_BOUNDS[1]
            and np.linalg.norm(x[:2]-self.seed[:2]) <= p.METRIC_MAX_XY_CORRECTION_M
            and abs(wrap(x[2]-self.seed[2])) <= self.yaw_search_bound
            and abs(h-self.height) <= p.METRIC_MAX_HEIGHT_CHANGE_M)

    def refine(self, raw, prior, timestamp_ns, *, prior_covariance, speed_mps=None, rotation_delta_rad=None, reference_delta=None):
        started = time.perf_counter()
        if self.last_stamp is not None and timestamp_ns <= self.last_stamp:
            raise ValueError('Metric registration timestamps must strictly increase')
        dt = 0. if self.last_stamp is None else (timestamp_ns-self.last_stamp)*1e-9
        self.height_variance += p.CAMERA_HEIGHT_RW_M_SQRT_S**2*dt
        self.last_stamp = int(timestamp_ns)
        predicted_correction = self.correction_filter.predict(prior, timestamp_ns, speed_mps,
            prior_covariance=prior_covariance, rotation_delta_rad=rotation_delta_rad, reference_delta=reference_delta)
        try:
            planes = self.prepare(raw, prior, prior_covariance)
        except GroundInitializationError as error:
            correction, temporal = self.correction_filter.update(missing_reason='ground_support_failed')
            T = prior.copy(); T[:2, 3] += correction[:2]
            T[:3, :3] = Rotation.from_euler('z', correction[2]).as_matrix() @ prior[:3, :3]
            return T, self.last_scale, dict(accepted=False, geometry_accepted=False, reason='ground_support_failed',
                failure=str(error), scale_validated=False, scale_source='previous_or_unverified_seed',
                candidate_pose=None, correction_filter=temporal,
                elapsed_seconds=time.perf_counter()-started)
        cp.cuda.get_current_stream().synchronize()
        prepared = time.perf_counter()
        x = self.seed.copy(); stages = []
        # Search from the predicted *world-frame* correction only if its
        # current-frame objective is better. Output fusion occurs below,
        # inside localization and before the only emission of this frame.
        carry = self.state_for_correction(prior, predicted_correction, np.exp(self.seed[3]))
        radius0, robust0, _ = p.METRIC_STAGES[0]
        seeds = [x, carry]
        if self.correction_filter.last_update is None:
            sigma = np.sqrt(self.prior_covariance[2, 2])
            bound = self.yaw_search_bound
            for multiple in p.MAP_STARTUP_YAW_SEED_SIGMAS:
                c = np.array([0., 0., np.clip(multiple*sigma, -bound, bound)])
                seeds.append(self.state_for_correction(prior, c, np.exp(self.seed[3])))
        eligible = [(self.evaluate(seed_x, radius0, robust0, diagnostics=False)['cost'], seed_x) for seed_x in seeds if self.valid(seed_x)]
        x = min(eligible, key=lambda pair: pair[0])[1].copy()
        for radius, robust, iterations in p.METRIC_STAGES:
            converged = False; reason = 'iteration_limit'; count = 0
            for it in range(iterations):
                before = self.evaluate(x, radius, robust, derivatives=True, diagnostics=False)
                if before['matches'] < 20:
                    reason = 'insufficient_matches'; break
                damping = np.diag(np.maximum(np.diag(before['H']), 1e-8))*1e-5
                delta = np.linalg.solve(before['H']+damping, -before['g'])
                cap = max(1., np.linalg.norm(delta[:2])/.5, abs(delta[2])/.05, abs(delta[3])/.1)
                delta /= cap
                small = np.linalg.norm(delta[:2]) < .002 and abs(delta[2]) < .0002 and abs(delta[3]) < .0002
                chosen = None
                for half in range(14):
                    factor = 2.**(-half); trial = x+factor*delta
                    if self.valid(trial):
                        after = self.evaluate(trial, radius, robust, association=before['association'], diagnostics=False)
                        if after['cost'] <= before['cost']+1e-12:
                            chosen = trial; break
                count = it+1
                if chosen is None:
                    converged = bool(small); reason = 'stationary' if small else 'line_search_failed'; break
                actual_delta = chosen-x
                x = chosen
                actual_small = (np.linalg.norm(actual_delta[:2]) < .002
                    and abs(actual_delta[2]) < .0002 and abs(actual_delta[3]) < .0002)
                if actual_small:
                    converged = True; reason = 'stationary'; break
            end = self.evaluate(x, radius, robust)
            stages.append(dict(radius_m=radius, robust_m=robust, iterations=count,
                               converged=converged, reason=reason, feature_cost=end['feature_cost'], coverage=end['coverage']))
        final = self.evaluate(x, radius, robust, derivatives=True)
        seed = self.evaluate(self.seed, radius, robust, association=final['association'])
        optimized = time.perf_counter()
        eig = np.linalg.eigvalsh(final['data_H'])
        quality_ok = bool(self.valid(x) and final['matches'] >= p.MIN_METRIC_MATCHES
            and final['median_plane_residual_m'] <= p.MAX_PLANE_MEDIAN_M
            and final['cost'] <= seed['cost']+1e-9 and stages[-1]['converged'])
        observation = covariance = None; observable = None
        if final['coverage'] > 0:
            observation, covariance, observable = geometry_observation(
                final['data_H'], self.correction_jacobian(x), self.prior_covariance, final)
        support_required = 0. if observable is None else p.PLANE_MIN_SUPPORT_PER_VARIABLE*(observable['rank']+1)
        geometry_accepted = bool(quality_ok and observable is not None and observable['rank'] > 0
            and final['angular_support'] >= support_required)
        scale_info = 0. if observable is None else observable['scale_information']
        scale_std = 1/np.sqrt(max(scale_info, 1e-20))
        candidate_T, candidate_scale = self.pose(x)
        candidate_sensitivity = None
        if geometry_accepted:
            # Candidate is sensor-regularized. Its response to a changed prior
            # reference is NOT a full -I data-only observation response.
            J = self.correction_jacobian(x)
            candidate_sensitivity = np.eye(3)-J @ np.linalg.solve(final['H'], J.T @ self.prior_precision)
        correction, temporal = self.correction_filter.update(
            pose_correction(candidate_T, prior) if geometry_accepted else None,
            covariance if geometry_accepted else None,
            observation_matrix=observation if geometry_accepted else None,
            missing_reason='unobservable_map_geometry' if quality_ok else 'geometry_rejected',
            candidate_sensitivity=candidate_sensitivity)
        accepted = temporal['accepted']
        scale = candidate_scale if accepted and scale_std <= p.MAX_SCALE_LOG_STD else float(np.exp(self.seed[3]))
        emitted_x = self.state_for_correction(prior, correction, scale)
        T, scale = self.pose(emitted_x)
        # The seed is an independent current ground measurement with a past
        # height anchor; it must be distinguished from feature-validated scale.
        self.last_scale = scale
        if accepted and scale_std <= p.MAX_SCALE_LOG_STD:
            measured_height = scale*self.ds
            measurement_var = max((measured_height*scale_std)**2, p.HEIGHT_OBSERVATION_FLOOR_M**2)
            gain = self.height_variance/(self.height_variance+measurement_var)
            self.height += gain*(measured_height-self.height)
            self.height_variance = (1-gain)**2*self.height_variance+gain*gain*measurement_var
        else:
            gain = 0.
        diag = {k:v for k,v in final.items() if k not in ('H','g','data_H','association')}
        diag.update(accepted=accepted, geometry_accepted=geometry_accepted,
            reason='feature_metric_update' if accepted else 'temporal_innovation_rejected' if geometry_accepted else 'ground_only_prior',
            scale_validated=bool(accepted and scale_std <= p.MAX_SCALE_LOG_STD),
            scale_source='current_features_and_ground' if accepted and scale_std <= p.MAX_SCALE_LOG_STD else 'current_ground_and_past_height',
            scale=scale, scale_seed=planes['scale_seed'], scale_log_std=float(scale_std),
            scale_information=scale_info, data_information_eigenvalues=eig.tolist(),
            seed_feature_cost=seed['feature_cost'], stages=stages, planes=planes,
            height_anchor_after_m=self.height, height_update_gain=float(gain),
            correction_plane_xy_yaw=(emitted_x[:3]-self.seed[:3]).tolist(),
            candidate_pose=candidate_T.tolist(), candidate_scale=candidate_scale,
            candidate_correction_plane_xy_yaw=(x[:3]-self.seed[:3]).tolist(),
            candidate_data_information=final['data_H'].tolist(), correction_filter=temporal,
            observability=observable, angular_support_required=float(support_required),
            association_policy='fixed inside line search; rematch at outer ICP iteration; seed compared on final identical planes',
            prior_covariance_xy_yaw=self.prior_covariance.tolist(),
            measurement_covariance_model='point-to-plane data-only Schur modes; current-prior whitening and error floors; uncalibrated',
            feature_quality_scope='unfiltered_geometry_candidate', emitted_feature_quality=self.score_pose(T, scale),
            ground_normal_difference_before_deg=float(np.degrees(np.arccos(np.clip(self.nm @ (prior[:3,:3] @ self.ns),-1,1)))),
            model='causal_observable_point_to_plane_with_correction_memory', elapsed_seconds=time.perf_counter()-started)
        cp.cuda.get_current_stream().synchronize()
        finished = time.perf_counter()
        diag['elapsed_seconds'] = finished-started
        diag['phase_timings_ms'] = dict(
            prepare_including_prediction=(prepared-started)*1000,
            optimization_and_final_diagnostics=(optimized-prepared)*1000,
            fusion_and_emitted_quality=(finished-optimized)*1000,
            clock='synchronized wall time; includes Python control and CUDA waits')
        return T, scale, diag
