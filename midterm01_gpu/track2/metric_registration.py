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
from .correction_filter import CausalCorrectionFilter, correction_covariance, pose_correction, yaw


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

    def prepare(self, raw, prior):
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
        # RViz presents the local XYZ map, rather than the ground-relative model.
        self.nn = type('DisplaySubmap', (), {'points': local})()
        self.center = prior[:3, 3].copy()
        f = Bm.T @ (prior[:3, :3] @ Bs[:, 0])
        yaw0 = float(np.arctan2(f[1], f[0]))
        self.seed = np.array([0., 0., yaw0, np.log(scale_seed)])
        return dict(source_ground=dict(normal=ns.tolist(), offset=ds, **sd),
                    map_ground=dict(normal=nm.tolist(), offset=dm, **md),
                    scale_seed=scale_seed, height_prior_m=self.height,
                    height_std_m=float(np.sqrt(self.height_variance)),
                    source_features=len(sample), angular_cells=len(unique), map_features=len(target),
                    feature_max_range_m=p.FEATURE_MAX_RANGE_M,
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

    def evaluate(self, x, radius, robust, *, derivatives=False, include_prior=True):
        q, J = transform4(self.source_features, x)
        ids, d2 = self.feature_nn.query(q, radius)
        good = ids >= 0
        w = self.weight
        data_cost = float(cp.sum(w*robust**2*cp.log1p(d2/robust**2))/p.FEATURE_STD_M**2)
        H = np.zeros((4, 4)); g = np.zeros(4)
        if derivatives and int(good.sum()) >= 20:
            residual = q[good]-self.feature_nn.points[ids[good]]
            ww = w[good]/(1+d2[good]/robust**2)/p.FEATURE_STD_M**2
            H = cp.asnumpy(cp.einsum('nki,nkj,n->ij', J[good], J[good], ww))
            g = cp.asnumpy(cp.einsum('nki,nk,n->i', J[good], residual, ww))
        data_H = H.copy()
        prior_cost = 0.
        if include_prior:
            yaw_r = wrap(x[2]-self.seed[2])
            h = float(np.exp(x[3])*self.ds)
            h_r = h-self.height
            hvar = self.height_variance+p.HEIGHT_OBSERVATION_FLOOR_M**2
            r = np.array([x[0]/p.METRIC_POSITION_STD_M, x[1]/p.METRIC_POSITION_STD_M,
                          yaw_r/np.deg2rad(p.METRIC_YAW_STD_DEG), h_r/np.sqrt(hvar)])
            A = np.diag([1/p.METRIC_POSITION_STD_M]*2+
                        [1/np.deg2rad(p.METRIC_YAW_STD_DEG), h/np.sqrt(hvar)])
            prior_cost = float(.5*r @ r)
            # data_cost uses rho(r^2), whose derivative is 2*J^T W r.
            # H/g are half of the full objective derivatives throughout.
            H += .5*A.T @ A; g += .5*A.T @ r
        coverage = float(cp.sum(w*good)/w.sum())
        return dict(cost=data_cost+prior_cost, data_cost=data_cost, prior_cost=prior_cost,
                    feature_cost=data_cost*p.FEATURE_STD_M**2/float(w.sum()),
                    coverage=coverage, median_capped_distance_m=float(cp.median(cp.sqrt(d2))),
                    matches=int(good.sum()), H=H, data_H=data_H, g=g)

    def score_pose(self, T, scale):
        """Read-only comparison on this frame's ONE fixed raw feature sample."""
        q = self.raw_features @ cp.asarray(T[:3,:3]).T*scale+cp.asarray(T[:3,3])
        q = (q-cp.asarray(self.origin)) @ cp.asarray(self.Bm)
        radius, robust, _ = p.METRIC_STAGES[-1]
        ids, d2 = self.feature_nn.query(q, radius)
        return dict(feature_cost=float(cp.sum(self.weight*robust**2*cp.log1p(d2/robust**2))/self.weight.sum()),
                    coverage=float(cp.sum(self.weight*(ids>=0))/self.weight.sum()),
                    median_capped_distance_m=float(cp.median(cp.sqrt(d2))),
                    source_features=len(q), comparison='same_raw_features_and_map_with_unmatched_penalty')

    def valid(self, x):
        if not np.isfinite(x).all():
            return False
        s = np.exp(x[3]); h = s*self.ds
        return bool(p.SCALE_BOUNDS[0] < s < p.SCALE_BOUNDS[1]
            and np.linalg.norm(x[:2]) <= p.METRIC_MAX_XY_CORRECTION_M
            and abs(wrap(x[2]-self.seed[2])) <= np.deg2rad(p.METRIC_MAX_YAW_CORRECTION_DEG)
            and abs(h-self.height) <= p.METRIC_MAX_HEIGHT_CHANGE_M)

    def refine(self, raw, prior, timestamp_ns, *, speed_mps=None):
        started = time.perf_counter()
        if self.last_stamp is not None and timestamp_ns <= self.last_stamp:
            raise ValueError('Metric registration timestamps must strictly increase')
        dt = 0. if self.last_stamp is None else (timestamp_ns-self.last_stamp)*1e-9
        self.height_variance += p.CAMERA_HEIGHT_RW_M_SQRT_S**2*dt
        self.last_stamp = int(timestamp_ns)
        predicted_correction = self.correction_filter.predict(prior, timestamp_ns, speed_mps)
        try:
            planes = self.prepare(raw, prior)
        except GroundInitializationError as error:
            correction, temporal = self.correction_filter.update(missing_reason='ground_support_failed')
            T = prior.copy(); T[:2, 3] += correction[:2]
            T[:3, :3] = Rotation.from_euler('z', correction[2]).as_matrix() @ prior[:3, :3]
            return T, self.last_scale, dict(accepted=False, geometry_accepted=False, reason='ground_support_failed',
                failure=str(error), scale_validated=False, scale_source='previous_or_unverified_seed',
                candidate_pose=None, correction_filter=temporal,
                elapsed_seconds=time.perf_counter()-started)
        x = self.seed.copy(); stages = []
        # Search from the predicted *world-frame* correction only if its
        # current-frame objective is better. Output fusion occurs below,
        # inside localization and before the only emission of this frame.
        carry = self.state_for_correction(prior, predicted_correction, np.exp(self.seed[3]))
        radius0, robust0, _ = p.METRIC_STAGES[0]
        if self.valid(carry) and self.evaluate(carry, radius0, robust0)['cost'] < self.evaluate(x, radius0, robust0)['cost']:
            x = carry
        for radius, robust, iterations in p.METRIC_STAGES:
            converged = False; reason = 'iteration_limit'; count = 0
            for it in range(iterations):
                before = self.evaluate(x, radius, robust, derivatives=True)
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
                        after = self.evaluate(trial, radius, robust)
                        if after['cost'] <= before['cost']+1e-12:
                            chosen = trial; break
                count = it+1
                if chosen is None:
                    converged = bool(small); reason = 'stationary' if small else 'line_search_failed'; break
                x = chosen
                if small:
                    converged = True; reason = 'stationary'; break
            end = self.evaluate(x, radius, robust)
            stages.append(dict(radius_m=radius, robust_m=robust, iterations=count,
                               converged=converged, reason=reason, feature_cost=end['feature_cost'], coverage=end['coverage']))
        final = self.evaluate(x, radius, robust, derivatives=True)
        seed = self.evaluate(self.seed, radius, robust)
        eig = np.linalg.eigvalsh(final['data_H'])
        # Diagnose scale information after marginalizing other map variables.
        dH = final['data_H']
        scale_info = float(dH[3,3]-dH[3,:3] @ np.linalg.pinv(dH[:3,:3], rcond=1e-8) @ dH[:3,3])
        scale_std = 1/np.sqrt(max(scale_info, 1e-20))
        geometry_accepted = bool(self.valid(x) and final['matches'] >= p.MIN_METRIC_MATCHES
            and final['coverage'] >= p.MIN_METRIC_COVERAGE
            and final['median_capped_distance_m'] <= p.MAX_FEATURE_MEDIAN_M
            and final['cost'] <= seed['cost']+1e-9
            and stages[-1]['converged'])
        candidate_T, candidate_scale = self.pose(x)
        covariance = (correction_covariance(final['data_H'], self.correction_jacobian(x), final)
                      if geometry_accepted else None)
        correction, temporal = self.correction_filter.update(
            pose_correction(candidate_T, prior) if geometry_accepted else None, covariance)
        accepted = temporal['accepted']
        scale = candidate_scale if accepted else float(np.exp(self.seed[3]))
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
        diag = {k:v for k,v in final.items() if k not in ('H','g','data_H')}
        diag.update(accepted=accepted, geometry_accepted=geometry_accepted,
            reason='feature_metric_update' if accepted else 'temporal_innovation_rejected' if geometry_accepted else 'ground_only_prior',
            scale_validated=bool(accepted and scale_std <= p.MAX_SCALE_LOG_STD),
            scale_source='current_features_and_ground' if accepted else 'current_ground_and_past_height',
            scale=scale, scale_seed=planes['scale_seed'], scale_log_std=float(scale_std),
            scale_information=scale_info, data_information_eigenvalues=eig.tolist(),
            seed_feature_cost=seed['feature_cost'], stages=stages, planes=planes,
            height_anchor_after_m=self.height, height_update_gain=float(gain),
            correction_plane_xy_yaw=(emitted_x[:3]-self.seed[:3]).tolist(),
            candidate_pose=candidate_T.tolist(), candidate_scale=candidate_scale,
            candidate_correction_plane_xy_yaw=(x[:3]-self.seed[:3]).tolist(),
            candidate_data_information=final['data_H'].tolist(), correction_filter=temporal,
            measurement_covariance_model='data-only angular-cell GN approximation with error floors; uncalibrated',
            feature_quality_scope='unfiltered_geometry_candidate', emitted_feature_quality=self.score_pose(T, scale),
            ground_normal_difference_before_deg=float(np.degrees(np.arccos(np.clip(self.nm @ (prior[:3,:3] @ self.ns),-1,1)))),
            model='causal_grounded_metric_sim3_with_correction_memory', elapsed_seconds=time.perf_counter()-started)
        return T, scale, diag
