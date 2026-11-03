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
from .observability import checked_covariance, geometry_observation, likelihood_observation, height_observation
from .height_filter import fuse_height
from .ground_support import checked_ground


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
        fused_mad, ground_support = sd['mad'], None
        if not p.uses_v52_fusion():
            ns, ds, fused_mad, ground_support = checked_ground(
                raw, ns, ds, sd, nm, prior[:3, :3], scale_seed)
        scale_seed = self.height/ds
        if not p.SCALE_BOUNDS[0] < scale_seed < p.SCALE_BOUNDS[1]:
            raise GroundInitializationError('Sensor-supported ground scale outside explicit bounds')
        Bs, Bm = plane_basis(ns), plane_basis(nm)
        origin = prior[:3, 3]-nm*(nm @ prior[:3, 3]+dm)
        source = raw @ cp.asarray(Bs)
        source[:, 2] += ds
        non_ground = source[:, 2] > max(p.MIN_SOURCE_FEATURE_HEIGHT_RAW, 6*fused_mad)
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
        self.source_line_good = None
        if p.LINE_FEATURES_ENABLED and not p.uses_v52_fusion():
            metric_sample = sample*scale_seed
            source_nn = GpuNearest(metric_sample,cell=1.5)
            source_geometry = PlaneGeometry(source_nn)
            source_ids, _ = source_nn.query(metric_sample,1e-6)
            source_geometry.ensure(source_ids)
            safe_source = cp.maximum(source_ids,0)
            self.source_line_good = (source_ids >= 0)&source_geometry.line_reliable[safe_source]
            self.source_line_direction = source_geometry.line_direction[safe_source]
        # RViz presents the local XYZ map, rather than the ground-relative model.
        self.nn = type('DisplaySubmap', (), {'points': local})()
        self.center = prior[:3, 3].copy()
        f = Bm.T @ (prior[:3, :3] @ Bs[:, 0])
        yaw0 = float(np.arctan2(f[1], f[0]))
        self.seed = np.array([0., 0., yaw0, np.log(scale_seed)])
        self.seed = self.state_for_correction(prior, np.zeros(3), scale_seed)
        return dict(source_ground=dict(**sd, normal=ns.tolist(), offset=ds,
                        fitted_mad_raw=sd['mad'], fused_mad_raw=fused_mad,
                        support_definition='inliers of original measured plane; fused normal and median offset used for pose'),
                    ground_support=ground_support,
                    map_ground=dict(normal=nm.tolist(), offset=dm, **md),
                    scale_seed=scale_seed, height_prior_m=self.height,
                    height_std_m=float(np.sqrt(self.height_variance)),
                    source_features=len(sample), angular_cells=len(unique), map_features=len(target),
                    source_line_features=0 if self.source_line_good is None else int(self.source_line_good.sum()),
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

    def feature_association(self, q, radius, rotation):
        """Disjoint plane OR source/target-validated vertical line constraints."""
        ids,d2,good,normals,residual,confidence = self.plane_residuals(q,radius)
        safe = cp.maximum(ids,0)
        centers = self.geometry.center[safe]
        association = dict(ids=ids,good=good,normals=normals,centers=centers,confidence=confidence)
        if getattr(self,'source_line_good',None) is not None:
            source_direction = self.source_line_direction @ cp.asarray(rotation).T
            direction = self.geometry.line_direction[safe]
            compatible = cp.abs(cp.sum(source_direction*direction,axis=1)) >= np.cos(np.deg2rad(p.LINE_MAX_DIRECTION_DIFFERENCE_DEG))
            lines = (ids >= 0)&(~good)&self.source_line_good&self.geometry.line_reliable[safe]&compatible
            if bool(lines.any()):
                projector = cp.einsum('ni,nj->nij',normals,normals)
                line_projector = cp.eye(3)[None,:,:]-cp.einsum('ni,nj->nij',direction,direction)
                projector = cp.where(lines[:,None,None],line_projector,projector)
                vector = cp.einsum('nij,nj->ni',projector,q-centers)
                residual = cp.where(lines,cp.linalg.norm(vector,axis=1),residual)
                confidence = cp.where(lines,self.geometry.line_confidence[safe],confidence)
                good = good|lines
                association.update(good=good,confidence=confidence,projector=projector,line_matches=lines)
        return association,d2,residual

    def evaluate(self, x, radius, robust, *, derivatives=False, include_prior=True,
                 association=None, diagnostics=True):
        # Solver trials only consume cost, matches and H/g. Sorting residuals
        # for medians and copying diagnostic scalars forces needless CUDA work
        # and host synchronization at every line-search trial.
        q, J = transform4(self.source_features, x, jacobian=derivatives)
        if association is None:
            c,s = np.cos(x[2]),np.sin(x[2])
            association,d2,residual = self.feature_association(q,radius,np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]]))
        else:
            ids = association['ids']; good = association['good']
            normals = association['normals']; confidence = association['confidence']
            if 'projector' in association:
                residual = cp.linalg.norm(cp.einsum('nij,nj->ni',association['projector'],q-association['centers']),axis=1)
            else:
                residual = cp.sum(normals*(q-association['centers']), axis=1)
            if diagnostics:
                d2 = cp.minimum(cp.sum((q-self.feature_nn.points[cp.maximum(ids, 0)])**2, axis=1), radius**2)
        ids=association['ids'];good=association['good'];normals=association['normals'];confidence=association['confidence']
        w = self.weight
        unmatched = .5*robust**2*np.log1p(radius**2/robust**2)/p.FEATURE_STD_M**2
        loss = .5*robust**2*cp.log1p(residual**2/robust**2)/p.FEATURE_STD_M**2
        loss = cp.where(good, unmatched-confidence*(unmatched-loss), unmatched)
        data_cost = float(cp.sum(w*loss))
        H = np.zeros((4, 4)); g = np.zeros(4)
        matches = int(good.sum()) if derivatives or diagnostics else None
        if derivatives and matches >= 20:
            ww = w[good]*confidence[good]/(1+residual[good]**2/robust**2)/p.FEATURE_STD_M**2
            if 'projector' in association:
                projected_J = cp.einsum('nij,njk->nik',association['projector'][good],J[good])
                vector = cp.einsum('nij,nj->ni',association['projector'][good],q[good]-association['centers'][good])
                H = cp.asnumpy(cp.einsum('nki,nkj,n->ij',projected_J,projected_J,ww))
                g = cp.asnumpy(cp.einsum('nki,nk,n->i',projected_J,vector,ww))
            else:
                projected_J = cp.einsum('ni,nij->nj', normals[good], J[good])
                H = cp.asnumpy(cp.einsum('ni,nj,n->ij', projected_J, projected_J, ww))
                g = cp.asnumpy(cp.einsum('ni,n,n->i', projected_J, residual[good], ww))
        data_H = H.copy(); data_g = g.copy()
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
                      matches=matches, association=association, H=H, data_H=data_H, g=g, data_g=data_g)
        if not diagnostics:
            return result
        coverage = float(cp.sum(w*good)/w.sum())
        plane_median = float(cp.median(cp.abs(residual[good]))) if bool(good.any()) else float(radius)
        result.update(feature_cost=2*data_cost*p.FEATURE_STD_M**2/float(w.sum()),
                    line_matches=0 if 'line_matches' not in association else int(association['line_matches'].sum()),
                    residual_model='disjoint point-to-plane or validated vertical point-to-line; no line-axis residual',
                    coverage=coverage, median_capped_distance_m=float(cp.median(cp.sqrt(d2))),
                    nn_coverage=float(cp.sum(w*(ids >= 0))/w.sum()),
                    median_plane_residual_m=plane_median,
                    median_geometric_residual_m=plane_median,
                    angular_support=float(cp.sum(w*good)))
        return result

    def score_pose(self, T, scale):
        """Same fixed raw sample, current map, and point-to-plane loss; no output changes."""
        q = self.raw_features @ cp.asarray(T[:3,:3]).T*scale+cp.asarray(T[:3,3])
        q = (q-cp.asarray(self.origin)) @ cp.asarray(self.Bm)
        radius, robust, _ = p.METRIC_STAGES[-1]
        rotation = self.Bm.T @ T[:3,:3] @ self.Bs
        association,d2,residual = self.feature_association(q,radius,rotation)
        ids=association['ids'];good=association['good'];confidence=association['confidence']
        unmatched = robust**2*np.log1p(radius**2/robust**2)
        loss = robust**2*cp.log1p(residual**2/robust**2)
        loss = cp.where(good, unmatched-confidence*(unmatched-loss), unmatched)
        return dict(feature_cost=float(cp.sum(self.weight*loss)/self.weight.sum()),
                    coverage=float(cp.sum(self.weight*good)/self.weight.sum()),
                    median_capped_distance_m=float(cp.median(cp.sqrt(d2))),
                    median_plane_residual_m=float(cp.median(cp.abs(residual[good]))) if bool(good.any()) else float(radius),
                    source_features=len(q), comparison='same_raw_features_and_map_plane_or_validated_line_with_unmatched_penalty')

    def valid(self, x):
        if not np.isfinite(x).all():
            return False
        s = np.exp(x[3]); h = s*self.ds
        return bool(p.SCALE_BOUNDS[0] < s < p.SCALE_BOUNDS[1]
            and np.linalg.norm(x[:2]-self.seed[:2]) <= p.METRIC_MAX_XY_CORRECTION_M
            and abs(wrap(x[2]-self.seed[2])) <= self.yaw_search_bound
            and abs(h-self.height) <= p.METRIC_MAX_HEIGHT_CHANGE_M)

    def _finish_v52(self, x, prior, final, seed, stages, planes, eig,
                    quality_ok, observation, covariance, observable, support_required,
                    geometry_accepted, scale_info, scale_std, started, prepared, optimized):
        """v5.2 fusion unchanged; directional expiry is the sole numerical change."""
        candidate_T, candidate_scale = self.pose(x)
        candidate_sensitivity = None
        if geometry_accepted:
            # Candidate is sensor-regularized. Its response to a changed prior
            # reference is NOT a full -I data-only observation response.
            J = self.correction_jacobian(x)
            candidate_sensitivity = np.eye(3)-J @ np.linalg.solve(final['H'], J.T @ self.prior_precision)
        validator = None
        if p.V511_ENABLED and p.VALIDATE_EMITTED_MAP_PROPOSAL and geometry_accepted:
            radius, robust, _ = p.METRIC_STAGES[-1]
            emitted_scale = candidate_scale if scale_std <= p.MAX_SCALE_LOG_STD else float(np.exp(self.seed[3]))
            baseline = self.state_for_correction(prior, self.correction_filter.predicted_mean, float(np.exp(self.seed[3])))
            baseline_loss = self.evaluate(baseline,radius,robust,include_prior=False,association=final['association'],diagnostics=False)['data_cost']
            def validator(proposal):
                trial = self.state_for_correction(prior,proposal,emitted_scale)
                valid = self.valid(trial)
                cost = self.evaluate(trial,radius,robust,include_prior=False,association=final['association'],diagnostics=False)['data_cost'] if valid else None
                ok = bool(cost is not None and np.isfinite(cost) and cost <= baseline_loss+1e-12)
                return dict(accepted=ok,reason='posterior_fixed_plane_loss_not_increased' if ok else 'posterior_fixed_plane_loss_increased' if valid else 'posterior_outside_search',
                    baseline_data_cost=baseline_loss,proposal_data_cost=cost,
                    comparison='actual emitted pose/scale vs prediction on same final associations; no rematching or discarded points')
        correction, temporal = self.correction_filter.update(
            pose_correction(candidate_T, prior) if geometry_accepted else None,
            covariance if geometry_accepted else None,
            observation_matrix=observation if geometry_accepted else None,
            missing_reason='unobservable_map_geometry' if quality_ok else 'geometry_rejected',
            candidate_sensitivity=candidate_sensitivity,proposal_validator=validator)
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
        diag = {k:v for k,v in final.items() if k not in ('H','g','data_H','data_g','association')}
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
            model='v52_point_to_plane_with_directional_correction_memory', elapsed_seconds=time.perf_counter()-started)
        diag['candidate_selection'] = getattr(self,'candidate_selection',None)
        diag['fusion_profile'] = p.FUSION_PROFILE
        diag['height_variance_after_m2'] = float(self.height_variance)
        cp.cuda.get_current_stream().synchronize()
        finished = time.perf_counter()
        diag['timing_components_ms'] = dict(preparation=(prepared-started)*1000,
            optimization_and_final_diagnostics=(optimized-prepared)*1000,
            fusion_and_emitted_diagnostics=(finished-optimized)*1000)
        return T, scale, diag

    def height_proposal(self, x, final, candidate_scale, scale_std, radius, robust, pose_eligible):
        """Preview without state changes; commit together with a valid pose proposal."""
        height, variance = self.height, self.height_variance
        height_likelihood = None
        height_update = dict(accepted=False, reason='pose_or_scale_untrusted', gain=0.)
        if pose_eligible and scale_std <= p.MAX_SCALE_LOG_STD:
            measured_height, measurement_var, height_likelihood = height_observation(
                final['data_H'], final['data_g'], candidate_scale*self.ds, final)
            if measured_height is not None:
                physical = p.CAMERA_HEIGHT_BOUNDS_M[0] < measured_height < p.CAMERA_HEIGHT_BOUNDS_M[1]
                profile = x.copy()
                if physical:
                    profile[:3] += np.asarray(height_likelihood['profiled_pose_step'])
                    profile[3] = np.log(measured_height/self.ds)
                within_bounds = bool(physical and self.valid(profile))
                height_loss = (self.evaluate(profile, radius, robust, include_prior=False,
                    association=final['association'], diagnostics=False) if within_bounds else None)
                loss_valid = bool(height_loss is not None and np.isfinite(height_loss['data_cost'])
                    and height_loss['data_cost'] <= final['data_cost']+1e-12)
                height_likelihood.update(within_existing_search_bounds=within_bounds,
                    candidate_data_cost=final['data_cost'],
                    profiled_data_cost=None if height_loss is None else height_loss['data_cost'],
                    fixed_association_loss_valid=loss_valid)
                if loss_valid:
                    height, variance, height_update = fuse_height(
                        self.height, self.height_variance, measured_height, measurement_var)
                else:
                    height_update['reason'] = 'height_likelihood_outside_search' if not within_bounds else 'height_likelihood_loss_increased'
            else:
                height_update['reason'] = 'height_unobservable'
        return height, variance, height_update, height_likelihood

    def _solve_candidate(self, x):
        """Complete all stages on one current-frame hypothesis; no filter commit."""
        x = x.copy(); stages = []
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
        eig = np.linalg.eigvalsh(final['data_H'])
        quality_ok = bool(self.valid(x) and final['matches'] >= p.MIN_METRIC_MATCHES
            and final['median_plane_residual_m'] <= p.MAX_PLANE_MEDIAN_M
            and final['cost'] <= seed['cost']+1e-9 and stages[-1]['converged'])
        observation = covariance = None; observable = None
        if final['coverage'] > 0:
            observation, covariance, observable = geometry_observation(
                final['data_H'], self.correction_jacobian(x), self.prior_covariance, final)
        support_required = 0. if observable is None else p.PLANE_MIN_SUPPORT_PER_VARIABLE*(observable['strong_rank']+1)
        geometry_accepted = bool(quality_ok and observable is not None and observable['rank'] > 0
            and final['angular_support'] >= support_required)
        scale_info = 0. if observable is None else observable['scale_information']
        scale_std = 1/np.sqrt(max(scale_info, 1e-20))
        return dict(x=x, stages=stages, final=final, seed=seed, eig=eig, quality_ok=quality_ok, observation=observation, covariance=covariance, observable=observable, support_required=support_required, geometry_accepted=geometry_accepted, scale_info=scale_info, scale_std=scale_std)

    def _select_current_candidates(self, prior, carry):
        enabled = p.V511_ENABLED and p.uses_v52_fusion()
        startup = self.correction_filter.last_update is None
        seeds = [self.seed.copy(), carry.copy()]
        if startup:
            sigma = np.sqrt(self.prior_covariance[2, 2])
            for multiple in p.MAP_STARTUP_YAW_SEED_SIGMAS:
                seeds.append(self.state_for_correction(prior, [0.,0.,np.clip(multiple*sigma,-self.yaw_search_bound,self.yaw_search_bound)], np.exp(self.seed[3])))
        radius0, robust0, _ = p.METRIC_STAGES[0]
        if not enabled:
            eligible = [(self.evaluate(x,radius0,robust0,diagnostics=False)['cost'],x) for x in seeds if self.valid(x)]
            return self._solve_candidate(min(eligible,key=lambda pair:pair[0])[1])
        if startup:
            yaw_offsets = []
            if p.STARTUP_INCLUDE_GIVEN_YAW:
                yaw_offsets.append(wrap(yaw(self.initial_pose)-yaw(prior)))
            if p.STARTUP_INCLUDE_MAP_AXIS:
                covariance = cp.cov(self.feature_nn.points[:,:2].T)
                axis = cp.asnumpy(cp.linalg.eigh(covariance)[1][:,-1])
                axis_yaw = float(np.arctan2(axis[1],axis[0]))
                current = self.seed[2]
                offset = wrap(axis_yaw+np.pi*round((current-axis_yaw)/np.pi)-current)
                yaw_offsets.append(offset)
            for delta in yaw_offsets:
                seeds.append(self.state_for_correction(prior,[0.,0.,delta],np.exp(self.seed[3])))
            # Height hypotheses are seeds inside the SAME objective and bounds,
            # never independent height observations or fixed metric calibration.
            originals = [x.copy() for x in seeds]
            for x in originals:
                for multiple in p.STARTUP_HEIGHT_SEED_SIGMAS:
                    height = self.height+multiple*np.sqrt(self.height_variance)
                    if p.CAMERA_HEIGHT_BOUNDS_M[0] < height < p.CAMERA_HEIGHT_BOUNDS_M[1]:
                        trial = x.copy(); trial[3] = np.log(height/self.ds); seeds.append(trial)
        unique=[]; invalid=0
        for x in seeds:
            if not self.valid(x): invalid+=1;continue
            # Same tolerances used by the solver's convergence test.
            if any(np.linalg.norm(x[:2]-v[:2])<.002 and abs(wrap(x[2]-v[2]))<.0002 and abs(x[3]-v[3])<.0002 for v in unique):continue
            unique.append(x)
        initial = [(self.evaluate(x,radius0,robust0,diagnostics=False)['cost'],x) for x in unique]
        first = min(initial,key=lambda row:(row[0],tuple(row[1])))[1]
        trials=[self._solve_candidate(first)]
        if startup or (p.TRACKING_RETRY_ON_GEOMETRY_FAILURE and not trials[0]['geometry_accepted']):
            for x in unique:
                if np.array_equal(x,first):continue
                trials.append(self._solve_candidate(x))
        for trial in trials:
            T, scale = self.pose(trial['x'])
            trial['common_score'] = self.score_pose(T,scale)['feature_cost']
        eligible = [t for t in trials if t['geometry_accepted']]
        chosen = min(eligible or trials,key=lambda t:(t['common_score'],tuple(t['x'])))
        chosen['selection'] = dict(startup=startup,fully_refined=len(trials),invalid_seeds=invalid,
            selected_index=next(i for i,t in enumerate(trials) if t is chosen),
            candidates=[dict(geometry_accepted=t['geometry_accepted'],rank=0 if t['observable'] is None else t['observable']['rank'],
                common_feature_cost=t['common_score'],candidate_state=t['x'].tolist(),selected=t is chosen) for t in trials],
            policy='complete each startup hypothesis before common raw-sample comparison; qualified candidates first; one temporal commit')
        return chosen

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
                ground_support=getattr(error, 'diagnostic', None),
                candidate_pose=None, correction_filter=temporal,
                elapsed_seconds=time.perf_counter()-started)
        cp.cuda.get_current_stream().synchronize()
        prepared = time.perf_counter()
        carry = self.state_for_correction(prior, predicted_correction, np.exp(self.seed[3]))
        candidates = self._select_current_candidates(prior, carry)
        x, stages, final, seed, eig, quality_ok, observation, covariance, observable, support_required, geometry_accepted, scale_info, scale_std = (
            candidates[name] for name in ('x','stages','final','seed','eig','quality_ok','observation','covariance','observable','support_required','geometry_accepted','scale_info','scale_std'))
        self.candidate_selection = candidates.get('selection')
        optimized = time.perf_counter()
        radius, robust, _ = p.METRIC_STAGES[-1]
        if p.uses_v52_fusion():
            return self._finish_v52(x, prior, final, seed, stages, planes, eig,
                quality_ok, observation, covariance, observable, support_required,
                geometry_accepted, scale_info, scale_std, started, prepared, optimized)
        candidate_T, candidate_scale = self.pose(x)
        candidate_sensitivity = None; measurement = None; likelihood = None
        likelihood_valid = False
        if geometry_accepted:
            measurement, observation, covariance, likelihood = likelihood_observation(
                final['data_H'], final['data_g'], self.correction_jacobian(x),
                self.prior_covariance, final, pose_correction(candidate_T, prior))
            # A map-only likelihood is world anchored. A changed sensor
            # reference shifts its relative correction by the SAME delta.
            candidate_sensitivity = np.eye(3)
            logscale = x[3]+likelihood['likelihood_nuisance_logscale_step']
            scale_in_bounds = bool(np.isfinite(logscale) and np.log(p.SCALE_BOUNDS[0]) < logscale < np.log(p.SCALE_BOUNDS[1]))
            likelihood_x = (self.state_for_correction(prior, measurement, float(np.exp(logscale)))
                            if scale_in_bounds else None)
            within_bounds = bool(likelihood_x is not None and self.valid(likelihood_x))
            likelihood['within_existing_search_bounds'] = within_bounds
            # H/g is a LOCAL quadratic, not a guarantee about a nonlinear pose.
            # Require the unregularized loss to improve on the SAME final planes
            # (no rematching/dropping difficult points to manufacture improvement).
            validated = None
            if within_bounds:
                validated = self.evaluate(likelihood_x, radius, robust, include_prior=False,
                                          association=final['association'], diagnostics=False)
            likelihood_valid = bool(validated is not None and np.isfinite(validated['data_cost'])
                                    and validated['data_cost'] <= final['data_cost']+1e-12)
            likelihood.update(fixed_association_data_cost=None if validated is None else validated['data_cost'],
                              candidate_data_cost=final['data_cost'],
                              fixed_association_loss_valid=likelihood_valid,
                              validation='profiled scale; existing search bounds; actual nonlinear data loss on unchanged final planes')
        # A weak mode's unregularized likelihood center can be far away even
        # though its Bayesian update is tiny. Validate the ACTUAL finite gain
        # proposal inside existing bounds, not the inverse-H center. The same
        # final planes/sample remain fixed so lost matches cannot hide loss.
        proposed_height, proposed_height_variance, height_update, height_likelihood = self.height_proposal(
            x, final, candidate_scale, scale_std, radius, robust, geometry_accepted)
        baseline_x = self.state_for_correction(prior, predicted_correction, self.height/self.ds)
        baseline_loss = self.evaluate(baseline_x, radius, robust, include_prior=False,
            association=final['association'], diagnostics=False)['data_cost']
        def validate_proposal(proposal):
            trial = self.state_for_correction(prior, proposal, proposed_height/self.ds)
            valid = self.valid(trial)
            value = (self.evaluate(trial, radius, robust, include_prior=False,
                association=final['association'], diagnostics=False)['data_cost'] if valid else None)
            ok = bool(value is not None and np.isfinite(value) and value <= baseline_loss+1e-12)
            return dict(accepted=ok, reason='posterior_map_loss_increased' if valid else 'posterior_outside_search',
                baseline_data_cost=baseline_loss, proposal_data_cost=value,
                comparison='actual limited-gain pose and once-fused height proposal vs prediction; same fixed planes')
        measurement_accepted = geometry_accepted
        correction, temporal = self.correction_filter.update(
            measurement if measurement_accepted else None,
            covariance if measurement_accepted else None,
            observation_matrix=observation if measurement_accepted else None,
            missing_reason='unobservable_map_geometry' if quality_ok else 'geometry_rejected',
            candidate_sensitivity=candidate_sensitivity,
            memory_observation_matrix=observable['strong_observation_matrix'] if measurement_accepted else None,
            proposal_validator=validate_proposal if measurement_accepted else None,
            local_linear_likelihood=True)
        accepted = temporal['accepted']
        if accepted:
            self.height, self.height_variance = proposed_height, proposed_height_variance
        else:
            height_update = dict(height_update, accepted=False, gain=0.,
                reason='pose_update_rejected' if height_update['accepted'] else height_update['reason'])
        # h is a physical state, not another MAP candidate to fuse a second
        # time. Emit THIS frame's height posterior and its current ground scale.
        scale = self.height/self.ds
        height_accepted = height_update['accepted']
        emitted_x = self.state_for_correction(prior, correction, scale)
        T, scale = self.pose(emitted_x)
        # The seed is an independent current ground measurement with a past
        # height anchor; it must be distinguished from feature-validated scale.
        self.last_scale = scale
        gain = height_update['gain']
        diag = {k:v for k,v in final.items() if k not in ('H','g','data_H','data_g','association')}
        diag.update(accepted=accepted, geometry_accepted=geometry_accepted,
            reason=('feature_metric_update' if accepted else 'temporal_innovation_rejected'
                    if geometry_accepted else 'ground_only_prior'),
            scale_validated=bool(height_accepted),
            scale_source='current_data_height_posterior_and_ground' if height_accepted else 'current_ground_and_past_height',
            scale=scale, scale_seed=planes['scale_seed'], scale_log_std=float(scale_std),
            scale_information=scale_info, data_information_eigenvalues=eig.tolist(),
            seed_feature_cost=seed['feature_cost'], stages=stages, planes=planes,
            height_anchor_after_m=self.height, height_update_gain=float(gain),
            height_variance_after_m2=float(self.height_variance),
            height_likelihood=height_likelihood, height_update=height_update,
            correction_plane_xy_yaw=(emitted_x[:3]-self.seed[:3]).tolist(),
            candidate_pose=candidate_T.tolist(), candidate_scale=candidate_scale,
            candidate_correction_plane_xy_yaw=(x[:3]-self.seed[:3]).tolist(),
            candidate_data_information=final['data_H'].tolist(), correction_filter=temporal,
            observability=observable, angular_support_required=float(support_required),
            association_policy='fixed inside line search; rematch at outer ICP iteration; seed compared on final identical planes',
            prior_covariance_xy_yaw=self.prior_covariance.tolist(),
            measurement_covariance_model='plane/line data-only likelihood H/g Schur modes; sensor prior once; uncalibrated',
            likelihood_observation=likelihood,
            candidate_data_gradient=final['data_g'].tolist(),
            feature_quality_scope='unfiltered_geometry_candidate', emitted_feature_quality=self.score_pose(T, scale),
            ground_normal_difference_before_deg=float(np.degrees(np.arccos(np.clip(self.nm @ (prior[:3,:3] @ self.ns),-1,1)))),
            model='causal_observable_plane_line_with_directional_motion', elapsed_seconds=time.perf_counter()-started)
        cp.cuda.get_current_stream().synchronize()
        finished = time.perf_counter()
        diag['elapsed_seconds'] = finished-started
        diag['phase_timings_ms'] = dict(
            prepare_including_prediction=(prepared-started)*1000,
            optimization_and_final_diagnostics=(optimized-prepared)*1000,
            fusion_and_emitted_quality=(finished-optimized)*1000,
            clock='synchronized wall time; includes Python control and CUDA waits')
        return T, scale, diag
