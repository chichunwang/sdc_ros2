"""Causal road-texture speed, never an absolute map pose or heading.

Each disjoint image interval contributes at most one observation. Static flow
can support zero velocity; v5.13 metric depth uses a trusted past height/ground model independently of
map-pose acceptance; full-image relative fits also observe lateral velocity
and yaw change. Cloud rasterization stays on CUDA; sparse
image tracking/PnP and small filters use CPU. No reference trajectory is read.
"""
from pathlib import Path
import cv2
import cupy as cp
import numpy as np
from scipy.spatial.transform import Rotation
from track3.io import sha256
from track3.visual_prior import relative_pose_from_pnp
from . import parameters as p

# 原有 LK 視窗；路面深度假設至少需要這個 finest-level patch 的支持。
LK_WINDOW = (21, 21)


def road_patch_mask(mask):
    """Necessary finest LK patch support; mask is not semantic ground truth.

    A ground-labelled centre beside a car/curb can track that object's patch.
    Coarser pyramid support remains an approximation; no new depth/inlier gate.
    """
    return cv2.erode(mask, np.ones(LK_WINDOW, np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)


def ground_points(pixels, K, normal, height):
    """Intersect sparse rectified FLU rays with n.p + height = 0."""
    uv = np.asarray(pixels, dtype=float).reshape(-1, 2)
    normal = np.asarray(normal, dtype=float)
    rays = np.c_[np.ones(len(uv)), -(uv[:, 0]-K[0, 2])/K[0, 0],
                 -(uv[:, 1]-K[1, 2])/K[1, 1]]
    denominator = rays @ normal
    depth = -float(height)/np.where(abs(denominator) > 1e-12, denominator, 1.)
    good = ((denominator < -1e-6) & (depth >= p.IMAGE_SPEED_GROUND_DEPTH_M[0])
            & (depth <= p.IMAGE_SPEED_GROUND_DEPTH_M[1]))
    return rays*depth[:, None], good


def track_images(before, after, mask, *, min_distance=8):
    points = cv2.goodFeaturesToTrack(before, maxCorners=p.IMAGE_SPEED_MAX_FEATURES,
        qualityLevel=.01, minDistance=min_distance, mask=mask)
    if points is None:
        return np.empty((0, 2)), np.empty((0, 2))
    lk = dict(winSize=LK_WINDOW, maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    end, status, _ = cv2.calcOpticalFlowPyrLK(before, after, points, None, **lk)
    if end is None:
        return np.empty((0, 2)), np.empty((0, 2))
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(after, before, end, None, **lk)
    if back is None:
        return np.empty((0, 2)), np.empty((0, 2))
    good = ((status.ravel() != 0) & (reverse.ravel() != 0)
            & np.isfinite(end.reshape(-1, 2)).all(axis=1)
            & (np.linalg.norm(back-points, axis=2).ravel() <= p.IMAGE_SPEED_FB_PX))
    a, b = points.reshape(-1, 2)[good], end.reshape(-1, 2)[good]
    h, w = after.shape
    good = (b[:, 0] >= 0) & (b[:, 0] < w) & (b[:, 1] >= 0) & (b[:, 1] < h)
    return a[good], b[good]


def grid_support(pixels, shape):
    if not len(pixels):
        return 0
    h, w = shape
    cells = np.floor(np.asarray(pixels)/[w/8, h/6]).astype(int)
    return len(np.unique(cells, axis=0))


def ground_speed(before, after, K, normal, height, height_variance, dt, shape):
    """One sparse planar PnP observation; no map-frame trajectory output."""
    objects, good = ground_points(before, K, normal, height)
    objects, pixels = objects[good], np.asarray(after)[good]
    diag = dict(accepted=False, reason='insufficient_ground_tracks', ground_tracks=len(objects))
    if len(objects) < p.IMAGE_SPEED_MIN_TRACKS:
        return diag
    cv2.setRNGSeed(752)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(objects, pixels, K, None,
        iterationsCount=200, reprojectionError=p.IMAGE_SPEED_REPROJECTION_PX,
        confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    count = 0 if inliers is None else len(inliers)
    diag.update(inliers=count, inlier_fraction=count/len(objects))
    if not ok or count < p.IMAGE_SPEED_MIN_TRACKS or count < .6*len(objects):
        return dict(diag, reason='pnp_outliers')
    keep = inliers.ravel()
    ok, rvec, tvec = cv2.solvePnP(objects[keep], pixels[keep], K, None,
        rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)
    projected, jacobian = cv2.projectPoints(objects[keep], rvec, tvec, K, None)
    error = np.linalg.norm(projected.reshape(-1, 2)-pixels[keep], axis=1)
    relative = relative_pose_from_pnp(rvec, tvec)
    translation = relative[:3, 3]
    normal = np.asarray(normal, dtype=float)
    planar = translation-normal*float(normal @ translation)
    speed = float(np.linalg.norm(planar)/dt)
    turn = float(Rotation.from_matrix(relative[:3, :3]).magnitude()/dt)
    # A camera supported by the same road cannot move far through that road.
    vertical_allowance = 3*np.sqrt(height_variance+p.HEIGHT_OBSERVATION_FLOOR_M**2)
    normal_std = np.deg2rad(np.sqrt(max(np.asarray(p.IMU_ORIENTATION_UNKNOWN_STD_DEG)[:2]**2
                                      + np.asarray(p.EXTRINSIC_ROTATION_STD_DEG)[:2]**2)))
    vertical_allowance += np.linalg.norm(translation)*np.sin(normal_std)
    transformed = objects[keep] @ cv2.Rodrigues(rvec)[0].T + tvec.ravel()
    valid = (ok and np.isfinite(relative).all() and np.all(transformed[:, 2] > 0)
        and np.median(error) <= p.IMAGE_SPEED_REPROJECTION_PX
        and speed <= p.IMAGE_SPEED_MAX_MPS and turn <= p.IMAGE_SPEED_MAX_ROTATION_RATE_RADPS
        and abs(float(normal @ translation)) <= vertical_allowance)
    # Interval-average speed approximates instantaneous speed. Account for
    # GNSS-model acceleration, uncertain physical height and image residuals.
    depth = float(np.median(np.linalg.norm(objects[keep], axis=1)))
    pixel_std = max(.5, float(np.median(error)))
    # Road occupies a trapezoid, not eight arbitrary full-image grid cells.
    # Assess the actual pose Jacobian instead; clustered/collinear points do
    # not manufacture a precise speed. Preserve metre/radian units explicitly.
    J = jacobian[:, :6]
    units = np.diag([.05]*3+[1.]*3)
    information = units @ (J.T @ J/pixel_std**2) @ units
    values, basis = np.linalg.eigh((information+information.T)*.5)
    rank = int(np.count_nonzero(values > max(1e-10, float(values.max())*1e-10)))
    diag.update(pnp_information_rank=rank, ground_grid_cells=grid_support(pixels[keep], shape))
    if rank < 6:
        return dict(diag, reason='pnp_geometry_rank_deficient')
    pose_covariance = units @ ((basis/values) @ basis.T) @ units
    theta = np.r_[rvec.ravel(), tvec.ravel()]
    def planar_velocity(x):
        relative = relative_pose_from_pnp(x[:3], x[3:])
        step = relative[:3, 3]
        return (step-normal*float(normal @ step))/dt
    derivative = np.column_stack([(planar_velocity(theta+np.eye(6)[i]*1e-6)
                                  -planar_velocity(theta-np.eye(6)[i]*1e-6))/2e-6 for i in range(6)])
    velocity_covariance = derivative @ pose_covariance @ derivative.T
    velocity_covariance = (velocity_covariance+velocity_covariance.T)*.5
    # Norm's central derivative vanishes at zero speed. Propagate the VECTOR
    # first and use its largest planar variance, so a clustered zero-motion
    # fit cannot claim spurious speed certainty.
    image_variance = max(0., float(np.linalg.eigvalsh(velocity_covariance).max()))
    # Ray/plane depth is also sensitive to plane tilt: its first-order
    # relative error grows approximately as range * angle_std / height.
    # Reuse the existing uncalibrated ground-normal model floor, rather than
    # treating the fitted plane as exact because it contains many pixels.
    normal_variance = np.deg2rad(p.GROUND_NORMAL_FLOOR_DEG)**2
    variance = (p.IMAGE_SPEED_STD_MPS**2
        + speed**2*(height_variance+depth**2*normal_variance)/height**2
        + image_variance + p.GNSS_XY_ACCEL_PSD*dt/4)
    return dict(diag, accepted=bool(valid), reason='image_ground_speed' if valid else 'pnp_physical_gate',
        kind='speed', speed_mps=speed, variance_m2ps2=float(variance),
        ground_normal_variance_rad2=float(normal_variance),
        image_speed_variance_m2ps2=image_variance,
        pnp_planar_velocity_covariance=velocity_covariance.tolist(),
        reprojection_median_px=float(np.median(error)), rotation_rate_radps=turn,
        vertical_motion_m=float(normal @ translation), vertical_allowance_m=float(vertical_allowance),
        model='past metric ground + current image tracks; interval-average speed, not absolute pose/heading')


class CausalGroundSpeed:
    def __init__(self, data, camera):
        self.folder = Path(data)/'Track2_images/Track2_f_images'
        factor = p.IMAGE_SPEED_DOWNSAMPLE
        if type(factor) is not int or factor < 1:
            raise ValueError('Image downsample must be a positive integer')
        self.maps = tuple(m[::factor, ::factor] for m in camera.maps)
        self.K = camera.K.copy(); self.K[:2] /= factor
        self.mask = camera.mask[::factor, ::factor].copy()
        self.anchor = None; self.current = None; self.gyro_angle = 0.
        self.last_stamp = None; self.last_arrival = None
        self.gyro_R=np.eye(3);self.interval_sequence=0;self.interval_produced=False
        self.gyro_std_bound=0.
        self.anchor_diagnostic=None

    def observe(self, arrival_ns, camera_ns, gyro_rotation_rad, *, rotation_previous_current=None,gyro_std_bound_rad=0.):
        if (type(arrival_ns) is not int or type(camera_ns) is not int or camera_ns > arrival_ns
                or (self.last_stamp is not None and camera_ns <= self.last_stamp)
                or (self.last_arrival is not None and arrival_ns <= self.last_arrival)):
            raise ValueError('Image speed requires increasing acquisition/arrival clocks')
        if not np.isfinite(gyro_rotation_rad) or gyro_rotation_rad < 0:
            raise ValueError('Invalid causal gyro interval')
        self.last_stamp = camera_ns; self.last_arrival = arrival_ns
        candidates = list(self.folder.glob(f'*_ts_{arrival_ns}.jpg'))
        if len(candidates) != 1:
            raise ValueError('Image must exactly match current PLY arrival timestamp')
        path = candidates[0]; raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise ValueError(f'Unreadable current image: {path}')
        gray = cv2.remap(raw, *self.maps, cv2.INTER_LINEAR)
        self.current = dict(stamp=camera_ns, arrival_ns=arrival_ns, gray=gray, path=str(path), sha256=sha256(path))
        self.gyro_angle += gyro_rotation_rad
        if not np.isfinite(gyro_std_bound_rad) or gyro_std_bound_rad<0:raise ValueError('Invalid gyro uncertainty bound')
        self.gyro_std_bound+=gyro_std_bound_rad
        if rotation_previous_current is not None:
            rotation_previous_current=np.asarray(rotation_previous_current,dtype=float)
            if rotation_previous_current.shape!=(3,3) or not np.allclose(rotation_previous_current.T@rotation_previous_current,np.eye(3),atol=1e-8):
                raise ValueError('Invalid causal gyro rotation matrix')
            self.gyro_R=self.gyro_R@rotation_previous_current
        self.interval_produced=False
        diag = dict(accepted=False, reason='awaiting_keyframe', current_image=str(path),
                    current_image_sha256=self.current['sha256'], timestamp_ns=camera_ns, arrival_ns=arrival_ns)
        if self.anchor is None:
            self.refresh = True
            return diag
        dt = (camera_ns-self.anchor['stamp'])*1e-9
        diag.update(interval_s=dt, keyframe_timestamp_ns=self.anchor['stamp'],
                    keyframe_image=self.anchor['path'], keyframe_image_sha256=self.anchor['sha256'])
        self.refresh = dt >= p.IMAGE_SPEED_INTERVAL_S
        if not self.refresh:
            return dict(diag, reason='nonoverlapping_interval_pending')
        if dt > p.IMAGE_SPEED_MAX_INTERVAL_S:
            return dict(diag, reason='image_interval_too_long')
        self.interval_sequence+=1
        # Distributed full-image tracks permit a zero-motion observation even
        # before a metric ground anchor is available. Spatial support reduces
        # single-object failures; it does not prove a scene is static. The
        # independent GNSS innovation gate must also accept zero velocity.
        a, b = track_images(self.anchor['gray'], gray, self.mask)
        flow = np.linalg.norm(b-a, axis=1)
        cells = grid_support(a, gray.shape)
        diag.update(tracks=len(a), grid_cells=cells, gyro_rate_radps=self.gyro_angle/dt)
        if len(a) >= p.IMAGE_SPEED_MIN_TRACKS and cells >= p.IMAGE_SPEED_MIN_GRID_CELLS:
            median, p90 = float(np.median(flow)), float(np.quantile(flow, .9))
            diag.update(flow_median_px=median, flow_p90_px=p90)
            if (median <= p.IMAGE_SPEED_QUIET_MEDIAN_PX and p90 <= p.IMAGE_SPEED_QUIET_P90_PX
                    and self.gyro_angle/dt <= p.IMAGE_SPEED_QUIET_GYRO_RADPS):
                self.interval_produced=True
                return dict(diag, accepted=True, reason='distributed_image_and_gyro_stationary', kind='stationary',
                    variance_m2ps2=p.IMAGE_SPEED_STATIONARY_STD_MPS**2)
        if p.v513_enabled() and (p.IMAGE_RELATIVE_TRANSLATION_ENABLED or p.IMAGE_RELATIVE_YAW_ENABLED):
            relative=self._relative_observation(a,b,diag,dt,gray.shape,rotation_previous_current is not None)
            if not p.v514_enabled() or relative.get('accepted') or self.anchor.get('ground') is None:
                return relative
            # The PLY depth fit and road-plane fit are alternative models of
            # ONE image interval, never independent observations. A failed
            # relative fit must not suppress the existing road-speed channel.
            fallback=self._ground_observation(gray,diag,dt)
            fallback.update(solver_comparison=relative.get('solver_comparison',{}),
                relative_channel='ground_speed_fallback',image_interval_model='road_plane_fallback',
                relative_attempt={k:relative.get(k) for k in ('reason','kind','inliers','depth_correspondences')},
                candidate_policy='relative fit first; road-plane speed only if relative rejected; one observation')
            self.interval_produced=bool(fallback.get('accepted'))
            return fallback
        if self.anchor.get('ground') is None:
            return dict(diag, reason='no_past_validated_ground_scale')
        return self._ground_observation(gray,diag,dt)

    def _ground_observation(self,gray,diag,dt):
        ground = self.anchor['ground']
        mask = ground['mask']
        patch_diagnostic = None
        if p.v5163_enabled():
            mask = road_patch_mask(mask)
            patch_diagnostic = dict(window_px=list(LK_WINDOW),
                source_mask_pixels=int(np.count_nonzero(ground['mask'])),
                supported_mask_pixels=int(np.count_nonzero(mask)),
                policy='entire finest LK window inside past road mask; coarse pyramid/semantics not guaranteed')
        a, b = track_images(self.anchor['gray'], gray, mask,
                            min_distance=p.IMAGE_SPEED_GROUND_FEATURE_SPACING_PX)
        measured = ground_speed(a, b, self.K, ground['normal'], ground['height'],
                                ground['height_variance'], dt, gray.shape)
        if measured.get('accepted'):
            sigma = (np.linalg.norm(np.deg2rad(p.IMU_GYRO_UNKNOWN_STD_DEGPS))*dt)**2
            sigma += (p.IMAGE_SPEED_REPROJECTION_PX/self.K[0, 0])**2
            allowance = self.gyro_angle+3*np.sqrt(sigma)
            measured['rotation_allowance_rad'] = float(allowance)
            if measured['rotation_rate_radps']*dt > allowance:
                measured.update(accepted=False, reason='visual_rotation_exceeds_gyro_budget')
        measured.update(height_reference_m=ground['height'], height_reference_variance_m2=ground['height_variance'])
        if patch_diagnostic is not None:
            measured['road_patch_support'] = patch_diagnostic
        return dict(diag, **measured)

    def store(self, raw, scale, registration, *, reference_rotation=None):
        """Store only this emitted frame, without any look-ahead or old edits."""
        if self.current is None:
            raise RuntimeError('Store requires a current image')
        if not self.refresh:
            if p.v513_enabled() and self.anchor_diagnostic is not None:
                self.anchor_diagnostic=dict(self.anchor_diagnostic,store_attempted=False,
                    current_image_timestamp_ns=self.current['stamp'])
            return
        if p.v513_enabled() and p.IMAGE_ANCHOR_MEMORY_ENABLED:
            from .image_anchor import metric_reference
            reference=metric_reference(registration)
            ground=None;depth=None
            if reference['accepted']:
                ground=dict(mask=self._visible_ground_mask(raw,reference['normal'],reference['offset_raw'],reference['threshold_raw']),
                    normal=np.asarray(reference['normal']),height=reference['height_m'],height_variance=reference['height_variance_m2'],
                    normal_variance_rad2=reference['ground_normal_model_variance_rad2'],mad_raw=reference['ground_mad_raw'])
                depth=self._visible_depth(raw,reference['scale'])
            # Retrying a longer interval is allowed only if the prior attempt
            # produced NO observation. An already used interval is never reused.
            retain=(not reference['accepted'] and self.anchor is not None
                and self.anchor.get('depth') is not None and not self.interval_produced
                and (self.current['stamp']-self.anchor['stamp'])*1e-9<p.IMAGE_SPEED_MAX_INTERVAL_S)
            active_reference=self.anchor['metric_reference'] if retain else reference
            self.anchor_diagnostic=dict(active_reference,store_attempted=True,
                current_reference=reference,anchor_retained=bool(retain),
                anchor_available=bool(active_reference['accepted'] and (retain or depth is not None)),
                current_image_timestamp_ns=self.current['stamp'],
                image_timestamp_ns=self.anchor['stamp'] if retain else self.current['stamp'])
            if retain:return
            self.anchor=dict(self.current,ground=ground,depth=depth,metric_reference=reference,
                reference_rotation=None if reference_rotation is None else np.asarray(reference_rotation).copy())
            self.gyro_angle=0.;self.gyro_R=np.eye(3);self.gyro_std_bound=0.
            return
        ground = None; planes = registration.get('planes')
        if registration.get('scale_validated') and planes is not None:
            measured = planes['source_ground']; angle = registration.get('ground_normal_difference_before_deg')
            sigma = np.sqrt(max(np.asarray(p.IMU_ORIENTATION_UNKNOWN_STD_DEG)[:2]**2
                                + np.asarray(p.EXTRINSIC_ROTATION_STD_DEG)[:2]**2))
            if angle is not None and (angle/sigma)**2 <= p.GROUND_NORMAL_NIS_REJECT:
                mask = self._visible_ground_mask(raw, measured['normal'], measured['offset'], measured['threshold'])
                ground = dict(mask=mask, normal=np.asarray(measured['normal']),
                    height=float(scale*measured['offset']),
                    height_variance=float(registration['height_variance_after_m2']+p.HEIGHT_OBSERVATION_FLOOR_M**2))
        self.anchor = dict(self.current, ground=ground)
        if p.v513_enabled() and ground is not None:
            measured=planes['source_ground'];ds=float(measured['offset'])
            self.anchor.update(depth=self._visible_depth(raw,scale),
                metric_reference=dict(accepted=True,reason='legacy_validated_scale',
                    scale_log_variance=float(ground['height_variance']/ground['height']**2+(measured.get('mad',0.)/ds)**2)),
                reference_rotation=None if reference_rotation is None else np.asarray(reference_rotation).copy())
        self.gyro_angle = 0.
        self.gyro_R=np.eye(3)
        self.gyro_std_bound=0.

    def _visible_depth(self,raw,scale):
        raw=cp.asarray(raw,dtype=cp.float64);h,w=self.mask.shape
        good=cp.isfinite(raw).all(axis=1)&(raw[:,0]>0)
        safe=cp.where(good,raw[:,0],1.)
        uv=cp.stack([self.K[0,2]-self.K[0,0]*raw[:,1]/safe,self.K[1,2]-self.K[1,1]*raw[:,2]/safe],axis=1)
        pixels=cp.rint(uv).astype(cp.int64);metric_range=cp.linalg.norm(raw,axis=1)*scale
        good&=(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
        good&=(metric_range>=p.IMAGE_SPEED_GROUND_DEPTH_M[0])&(metric_range<=p.FEATURE_MAX_RANGE_M)
        raw=raw[good];uv=uv[good];pixels=pixels[good]
        ids=pixels[:,1]*w+pixels[:,0];nearest=cp.full(h*w,cp.inf)
        cp.minimum.at(nearest,ids,raw[:,0]);visible=raw[:,0]<=nearest[ids]+1e-10
        return dict(points=raw[visible]*scale,pixels=uv[visible])

    def _relative_observation(self,before,after,diag,dt,shape,has_gyro):
        from .relative_geometry import compare_relative
        from track3.search import closest_pixels
        if self.anchor.get('depth') is None:
            return dict(diag,reason='no_trusted_metric_depth_anchor',anchor_model=self.anchor.get('metric_reference'))
        if self.anchor.get('reference_rotation') is None:
            return dict(diag,reason='missing_anchor_reference_rotation')
        depth=self.anchor['depth']
        if not len(before) or not len(depth['points']):return dict(diag,reason='insufficient_background_depth_tracks')
        ids,d2=closest_pixels(before,depth['pixels'])
        keep=cp.asnumpy(d2)<=p.IMAGE_SPEED_REPROJECTION_PX**2
        objects=cp.asnumpy(depth['points'][ids[cp.asarray(keep)]])
        association_error=cp.asnumpy(d2)[keep]
        if p.v514_enabled():
            from .relative_geometry import depth_on_feature_rays
            objects=depth_on_feature_rays(objects,before[keep],self.K)
        pixels=after[keep]
        reference=self.anchor['metric_reference']
        sigma=max(self.gyro_std_bound**2,(np.linalg.norm(np.deg2rad(p.IMU_GYRO_UNKNOWN_STD_DEGPS))*dt)**2)
        sigma+=float(np.sum(np.deg2rad(p.EXTRINSIC_ROTATION_STD_DEG)**2))*self.gyro_angle**2
        compared=compare_relative(objects,pixels,self.K,dt,shape,self.gyro_R if has_gyro else None,sigma,reference['scale_log_variance'])
        both=p.IMAGE_RELATIVE_TRANSLATION_ENABLED and p.IMAGE_RELATIVE_YAW_ENABLED
        channel=('relative_yaw' if self.interval_sequence%2==0 else 'vector_velocity') if both else ('vector_velocity' if p.IMAGE_RELATIVE_TRANSLATION_ENABLED else 'relative_yaw')
        if p.IMAGE_RELATIVE_SOLVER not in ('gyro_translation','pnp'):raise ValueError('Unknown relative-motion solver')
        selected=compared['pnp' if channel=='relative_yaw' else p.IMAGE_RELATIVE_SOLVER]
        self.interval_produced=bool(selected['accepted'])
        result=dict(diag,**selected,kind=channel,interval_s=dt,depth_correspondences=len(objects),
            solver_comparison=compared,anchor_reference_rotation=self.anchor['reference_rotation'].tolist(),
            anchor_model=reference,measurement_scope='one channel per disjoint accepted image interval; relative motion only')
        if p.v514_enabled():
            result.update(depth_association_rms_px=float(np.sqrt(np.mean(association_error))) if len(association_error) else None,
                depth_association_model='nearest PLY forward depth on exact anchor feature ray; depth errors retain model floor')
        return result

    def _visible_ground_mask(self, raw, normal, offset, threshold):
        raw = cp.asarray(raw, dtype=cp.float64)
        h, w = self.mask.shape
        positive = cp.isfinite(raw).all(axis=1) & (raw[:, 0] > 0)
        safe = cp.where(positive, raw[:, 0], 1.)
        u = cp.rint(self.K[0, 2]-self.K[0, 0]*raw[:, 1]/safe).astype(cp.int64)
        v = cp.rint(self.K[1, 2]-self.K[1, 1]*raw[:, 2]/safe).astype(cp.int64)
        good = positive & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ids = v[good]*w+u[good]; depth = raw[good, 0]
        surface = cp.abs(raw[good] @ cp.asarray(normal)+offset) < threshold
        nearest = cp.full(h*w, cp.inf); cp.minimum.at(nearest, ids, depth)
        visible = surface & (depth <= nearest[ids]+1e-10)
        ground = cp.zeros(h*w, dtype=cp.uint8); ground[ids[visible]] = 255
        mask = cp.asnumpy(ground.reshape(h, w))
        return cv2.bitwise_and(cv2.dilate(mask, np.ones((5, 5), np.uint8)), self.mask)
