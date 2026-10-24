"""Forward adjacent-cloud initialization + map ICP, no IMU or pose smoothing."""
from dataclasses import replace
import time
import numpy as np
from scipy.spatial.transform import Rotation

from track3_registration import (MapRegistrar, RegistrationConfig, assess_multistart,
                                  block_validation, checked_pose)
from track3_visibility import ray_depth_diagnostic
from track3_structural_weights import structural_weights

METHOD = 'track3_forward_gpu_structural_ndt_icp_reacquisition_v2'


class MapOnlyTracker:
    def __init__(self, scene, initial, initial_scale=3.):
        self.scene = scene
        self.pose = checked_pose(initial)
        self.initial = self.pose.copy()
        self.scale = float(initial_scale)
        self.previous_raw = None
        self.previous_ns = None
        self.last_map_ns = None
        self.last_map_pose = None
        self.consecutive_map_rejections = 0
        self.state = 'initializing'

    def _ray(self, raw, result):
        result['ray_depth'] = ray_depth_diagnostic(self.scene, raw, result['pose'], result['scale'], ~block_validation(raw))
        return result

    @staticmethod
    def _training_supported(result):
        ray = result['ray_depth']
        return (all(result['checks'][k] for k in ('converged', 'interior', 'observable'))
                and result['training']['coverage'] >= .6 and result['training']['median_m'] <= .35
                and result['training']['p90_m'] <= 1. and ray['available']
                and ray['training']['support_fraction'] >= .6
                and ray['training']['median_abs_log_depth_ratio'] is not None
                and ray['training']['median_abs_log_depth_ratio'] <= .15)

    def _choose_and_check(self, raw, candidates):
        supported = [r for r in candidates if self._training_supported(r)]
        if not supported:
            return None, dict(accepted=False, reasons=['no_training_supported_visible_map_fit'])
        # Candidate and restart selection uses TRAINING geometry only.
        chosen = min(supported, key=lambda r:r['ray_depth']['training']['median_abs_log_depth_ratio'])
        consensus = assess_multistart(supported)
        # The ray-selected winner must itself belong to the RMS-plausible
        # consensus group. Two cheap fits cannot certify a third, different
        # winner outside that group's training-cost band.
        best_cost = min(r['training_cost'] for r in supported)
        plausible = [r for r in supported if r['training_cost'] <= best_cost*1.5+.05]
        selected_pose = np.array(chosen['pose'])
        winner_consistent = any(r is chosen for r in plausible) and all(
            np.linalg.norm(np.array(r['pose'])[:3,3]-selected_pose[:3,3]) <= .5
            and Rotation.from_matrix(np.array(r['pose'])[:3,:3] @ selected_pose[:3,:3].T).magnitude() <= np.deg2rad(2.)
            and abs(np.log(r['scale']/chosen['scale'])) <= .1 for r in plausible)
        ray = chosen['ray_depth']['validation']
        ray_pass = (ray['support_fraction'] >= .6 and ray['median_abs_log_depth_ratio'] is not None
                    and ray['median_abs_log_depth_ratio'] <= .15)
        reasons = list(chosen['reasons'])
        consistent = consensus['initial_value_consistent'] and winner_consistent
        if not consistent:
            reasons.append('restart_inconsistent')
        if not ray_pass:
            reasons.append('heldout_ray_depth')
        return chosen, dict(accepted=not reasons, reasons=reasons,
                            restart_consensus=bool(consistent),
                            ambiguities=consensus['ambiguities'], plausible_starts=len(supported))

    def _bootstrap(self, raw):
        from old_icp_runtime import initial_ground_tilt
        variants = [('official', self.initial)]
        try:
            # Unlike early exploratory scripts, initialization EXCLUDES the
            # reserved source blocks as well as the optimizer/seed selection.
            ground, _ = initial_ground_tilt(raw[~block_validation(raw)], self.initial, self.scale)
            variants.append(('training_ground_hypothesis', ground))
        except ValueError:
            pass
        coarse = []
        for kind, initial in variants:
            for scale in sorted(set((3., 6., 9., self.scale))):
                fit = self._ray(raw, self.scene.fit(raw, initial, scale))
                fit.update(seed_type=kind, seed_scale=scale)
                coarse.append(fit)
        supported = [r for r in coarse if self._training_supported(r)]
        if not supported:
            return None, dict(accepted=False, reasons=['bootstrap_visible_geometry_unsupported']), coarse
        best = min(supported, key=lambda r:r['ray_depth']['training']['median_abs_log_depth_ratio'])
        restarts = []
        for factor in (.9, 1., 1.1):
            seed = float(np.clip(best['scale']*factor, self.scene.config.min_scale*1.001, self.scene.config.max_scale/1.001))
            fit = self._ray(raw, self.scene.fit(raw, best['pose'], seed, use_ndt=False))
            fit.update(seed_type='local_scale_restart', seed_scale=seed)
            restarts.append(fit)
        chosen, decision = self._choose_and_check(raw, restarts)
        decision['scope'] = 'local restart reproducibility after training visibility selection; not global uniqueness or calibrated accuracy'
        return chosen, decision, coarse+restarts

    def _new_pair_registrar(self, world):
        return MapRegistrar(world, config=RegistrationConfig(), geometry=self.scene.geometry,
                            backend=self.scene.geometry.backend)

    def _map_candidates(self, raw, prior, config, weights):
        candidates = []
        # Include the actual predicted scale, not just two displaced
        # scale seeds. Every start has the same fixed training weights.
        for factor in (.95, 1., 1.05):
            seed = float(np.clip(self.scale*factor, config.min_scale*1.001, config.max_scale/1.001))
            fit = self._ray(raw, self.scene.fit(raw, prior, seed, use_ndt=True,
                                               training_weights=weights))
            fit.update(seed_type='adjacent_cloud_prior', seed_scale=seed)
            candidates.append(fit)
        return candidates

    def step(self, raw, timestamp_ns):
        raw = np.asarray(raw, dtype=np.float64)
        ns = int(timestamp_ns)
        if ns <= 0 or (self.previous_ns is not None and ns <= self.previous_ns):
            raise ValueError('Only increasing current-frame timestamps are allowed')
        previous = self.pose.copy()
        state_before = self.state
        stages = {}
        sampling = None
        reacquisition_attempted = False
        motion_report = None
        pair = None
        if self.last_map_ns is None:
            started = time.perf_counter()
            chosen, decision, candidates = self._bootstrap(raw)
            stages['bootstrap'] = 1000*(time.perf_counter()-started)
        else:
            dt = (ns-self.previous_ns)/1e9
            started = time.perf_counter()
            weights, sampling = structural_weights(raw, self.scale, previous[:3,:3], self.scene.geometry)
            stages['source_geometry'] = 1000*(time.perf_counter()-started)
            started = time.perf_counter()
            world = self.scale*self.previous_raw @ previous[:3, :3].T+previous[:3, 3]
            relative = self._new_pair_registrar(world)
            pair = relative.fit(raw, previous, self.scale, estimate_scale=False,
                                training_weights=weights)
            relative = None  # release previous-cloud grid before map fits
            pair_pose = np.array(pair['pose'])
            pair_translation = np.linalg.norm(pair_pose[:3, 3]-previous[:3, 3])
            pair_rotation = Rotation.from_matrix(pair_pose[:3, :3] @ previous[:3, :3].T).magnitude()
            pair_motion = pair_translation <= max(.25, 40*dt) and pair_rotation <= max(np.deg2rad(.5), np.deg2rad(90)*dt)
            pair['motion_gate'] = bool(pair_motion)
            pair['accepted'] = bool(pair['accepted'] and pair_motion)
            prior = pair_pose if pair['accepted'] else previous
            stages['pair_registration'] = 1000*(time.perf_counter()-started)
            # All spatial/relative-scale domains are safety assumptions, not
            # calibrated uncertainty or leaderboard-derived corrections.
            reacquisition_attempted = self.consecutive_map_rejections > 0 or not pair['accepted']
            # Tracking uses the established local domain. After a rejected map
            # update, reacquire in the EXISTING bootstrap domain, using only the
            # current cloud/static map. Published past poses never change.
            domain = (8., 30.) if reacquisition_attempted else (2., 8.)
            config = replace(RegistrationConfig(), min_scale=max(.5, self.scale*np.exp(-.15)),
                             max_scale=min(16., self.scale*np.exp(.15)),
                             translation_domain_m=domain[0], rotation_domain_deg=domain[1])
            original_config = self.scene.config
            self.scene.config = config
            started = time.perf_counter()
            try:
                candidates = self._map_candidates(raw, prior, config, weights)
                chosen, decision = self._choose_and_check(raw, candidates)
            finally:
                self.scene.config = original_config
            stages['map_reacquisition' if reacquisition_attempted else 'map_tracking'] = 1000*(time.perf_counter()-started)
            if chosen is not None:
                p = np.array(chosen['pose'])
                # Bound physical motion since the last MAP anchor. A correction
                # of accumulated pair-prior error is not one frame of vehicle
                # travel. Rates/floors remain unchanged; the elapsed interval and
                # reference are now the same for both sides of this inequality.
                anchor_dt = (ns-self.last_map_ns)/1e9
                anchor = self.last_map_pose
                distance = float(np.linalg.norm(p[:3,3]-anchor[:3,3]))
                angle = float(Rotation.from_matrix(p[:3,:3] @ anchor[:3,:3].T).magnitude())
                translation_limit = max(.5, 40*anchor_dt)
                rotation_limit = max(np.deg2rad(1.), np.deg2rad(90)*anchor_dt)
                motion = distance <= translation_limit and angle <= rotation_limit
                motion_report = dict(reference='last_accepted_map_anchor',
                    elapsed_seconds=anchor_dt, displacement_m=distance,
                    rotation_deg=float(np.rad2deg(angle)), translation_limit_m=translation_limit,
                    rotation_limit_deg=float(np.rad2deg(rotation_limit)),
                    correction_from_pair_prior_m=float(np.linalg.norm(p[:3,3]-prior[:3,3])),
                    accepted=bool(motion))
                if not motion:
                    decision['accepted'] = False
                    decision['reasons'].append('map_motion_gate')
            if not decision['accepted']:
                self.pose = prior.copy()
        if decision['accepted']:
            self.pose, self.scale = np.array(chosen['pose']), float(chosen['scale'])
            self.last_map_ns = ns
            self.last_map_pose = self.pose.copy()
            self.consecutive_map_rejections = 0
            self.state = 'tracking'
        else:
            self.consecutive_map_rejections += 1
            self.state = 'relocalizing' if self.last_map_ns is not None else 'initializing'
        self.pose = checked_pose(self.pose)
        self.previous_raw, self.previous_ns = raw.copy(), ns
        unsupported = (ns-self.last_map_ns)/1e9 if self.last_map_ns is not None else None
        if unsupported is not None and unsupported > 2.:
            self.state = 'lost'
        return dict(accepted=decision['accepted'], pose=self.pose.tolist(), scale=self.scale,
                    map_decision=decision, selected_fit=chosen, candidate_fits=candidates,
                    pair_fit=pair, imu_used=False, timestamp_ns=ns, input_cutoff_ns=ns,
                    tracking_established=self.last_map_ns is not None,
                    unsupported_seconds=unsupported, tracking_lost=unsupported is None or unsupported > 2.,
                    tracking_state=self.state, previous_tracking_state=state_before,
                    consecutive_map_rejections=self.consecutive_map_rejections,
                    reacquisition_attempted=reacquisition_attempted,
                    reacquisition_succeeded=bool(reacquisition_attempted and decision['accepted']),
                    map_motion=motion_report, source_sampling=sampling, stage_timing_ms=stages,
                    accuracy='unknown; geometry and motion gates are not truth-error certification')
