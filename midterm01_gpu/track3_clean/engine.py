"""One current frame -> pair prediction -> map observation -> one state commit."""
from dataclasses import replace
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .config import RegistrationConfig, TrackingConfig
from .contracts import BootstrapUnsupported, GeometryUnavailable, Similarity, TrackingState
from .geometry import block_validation, xyz
from .initialization import initial_ground_tilt
from .quality import decide
from .registration import MapRegistrar
from .sampling import structural_weights
from .state import TrackingMemory
from .visibility import ray_depth_diagnostic


class LocalizationEngine:
    def __init__(self, scene, initial_pose, config=None):
        self.scene = scene
        self.config = config or TrackingConfig()
        self.memory = TrackingMemory(Similarity(initial_pose, self.config.initial_scale),
                                     lost_after_seconds=self.config.lost_after_seconds)

    def _fit_starts(self, scene, raw, initial, seeds, weights=None, *, ray=True):
        candidates = []
        for seed in seeds:
            fit = scene.fit(raw, initial, seed, training_weights=weights)
            fit['seed_scale'] = seed
            fit['seed_pose'] = np.asarray(initial).tolist()
            if ray:
                fit['ray_depth'] = ray_depth_diagnostic(scene, raw, fit['pose'], fit['scale'], ~block_validation(raw))
            candidates.append(fit)
        return candidates

    def _bootstrap(self, raw):
        m, c = self.memory, self.config
        poses = [m.published.pose]
        try:
            ground, _ = initial_ground_tilt(raw[~block_validation(raw)], poses[0], c.initial_scale)
            poses.append(ground)
        except ValueError:
            pass
        candidates = []
        for pose in poses:
            candidates.extend(self._fit_starts(self.scene, raw, pose, c.bootstrap_scales))
        # Coarse alternatives are included in the final ambiguity check.
        # Local restarts alone must not erase a competing supported coarse basin.
        first = decide(candidates, config=self.scene.config)
        if first.selected is not None:
            best = first.selected
            seeds = self._seeds(best['scale'], self.scene.config)
            candidates.extend(self._fit_starts(self.scene, raw, best['pose'], seeds))
        return decide(candidates, config=self.scene.config), candidates

    def _seeds(self, scale, config):
        return tuple(float(np.clip(scale*f, config.min_scale*1.001, config.max_scale/1.001))
                     for f in self.config.restart_factors)

    def _bounded(self, scale, *, reacquiring):
        c, base = self.config, self.scene.config
        return replace(base, min_scale=max(base.min_scale, scale*np.exp(-c.max_log_scale_change)),
                       max_scale=min(base.max_scale, scale*np.exp(c.max_log_scale_change)),
                       translation_domain_m=c.reacquisition_translation_m if reacquiring else c.local_translation_m,
                       rotation_domain_deg=c.reacquisition_rotation_deg if reacquiring else c.local_rotation_deg)

    def _pair(self, raw, ns, weights):
        c, ref = self.config, self.memory.reference
        # The target is in previous CAMERA METRES, not huge map coordinates.
        # Estimate current scale as well as R,t; raw PLY is not certified rigid/metric.
        config = self._bounded(ref.estimate.scale, reacquiring=True)
        target = ref.estimate.scale*ref.raw
        scene = MapRegistrar(target, backend=self.scene.geometry.backend,
                             geometry=self.scene.geometry, config=config)
        candidates = self._fit_starts(scene, raw, np.eye(4), self._seeds(ref.estimate.scale, config), weights, ray=False)
        decision = decide(candidates, require_ray=False, config=config)
        estimate = None
        motion = False
        if decision.accepted:
            local = Similarity(decision.selected['pose'], decision.selected['scale'])
            elapsed = (ns-ref.timestamp_ns)/1e9
            motion = (np.linalg.norm(local.pose[:3, 3]) <= max(c.pair_translation_floor_m, c.max_speed_mps*elapsed)
                      and Rotation.from_matrix(local.pose[:3, :3]).magnitude()
                      <= np.deg2rad(max(c.pair_rotation_floor_deg, c.max_rotation_degps*elapsed)))
            if motion:
                estimate = local.in_map(ref.estimate)
        return estimate, dict(accepted=estimate is not None, reasons=list(decision.reasons)+
                              (['pair_motion'] if decision.accepted and not motion else []),
                              reference_timestamp_ns=ref.timestamp_ns, decision=decision.diagnostics,
                              candidates=candidates)

    def step(self, raw, timestamp_ns):
        raw = xyz(raw)
        ns = self.memory.check_time(timestamp_ns)
        before = self.memory.state.value
        started = time.perf_counter()
        stages, pair, pair_estimate = {}, None, None
        if self.memory.last_map is None:
            decision, candidates = self._bootstrap(raw)
            stages['bootstrap_ms'] = 1000*(time.perf_counter()-started)
            prediction = self.memory.published
            reacquiring = False
        else:
            ref = self.memory.reference
            weights, sampling = structural_weights(raw, ref.estimate.scale, ref.estimate.pose[:3, :3], self.scene.geometry)
            stages['source_geometry_ms'] = 1000*(time.perf_counter()-started)
            stage = time.perf_counter()
            try:
                pair_estimate, pair = self._pair(raw, ns, weights)
            except GeometryUnavailable as error:
                # Missing pair support is an observation rejection. CUDA errors
                # and programming errors still propagate; there is no fallback.
                pair = dict(accepted=False, reasons=['pair_geometry_unavailable'],
                            detail=str(error), reference_timestamp_ns=ref.timestamp_ns)
            stages['pair_ms'] = 1000*(time.perf_counter()-stage)
            pair['sampling'] = sampling
            prediction = pair_estimate or self.memory.published
            reacquiring = self.memory.state != TrackingState.TRACKING or pair_estimate is None
            config = self._bounded(prediction.scale, reacquiring=reacquiring)
            stage = time.perf_counter()
            candidates = self._fit_starts(self.scene.with_config(config), raw, prediction.pose,
                                          self._seeds(prediction.scale, config), weights)
            decision = decide(candidates, config=config)
            stages['map_ms'] = 1000*(time.perf_counter()-stage)
        map_estimate = None
        motion = None
        if decision.accepted:
            proposed = Similarity(decision.selected['pose'], decision.selected['scale'])
            anchor = self.memory.last_map
            motion = True
            if anchor is not None:
                c = self.config
                elapsed = (ns-anchor.timestamp_ns)/1e9
                dp = np.linalg.norm(proposed.pose[:3, 3]-anchor.estimate.pose[:3, 3])
                dr = Rotation.from_matrix(proposed.pose[:3, :3] @ anchor.estimate.pose[:3, :3].T).magnitude()
                motion = dp <= c.map_translation_slack_m+c.max_speed_mps*elapsed and dr <= np.deg2rad(c.map_rotation_slack_deg+c.max_rotation_degps*elapsed)
            if motion:
                map_estimate = proposed
        diagnostics = dict(state_before=before, reacquisition_attempted=reacquiring,
                           pair=pair, map=dict(accepted=map_estimate is not None,
                           reasons=list(decision.reasons)+(['map_anchor_motion'] if decision.accepted and not motion else []),
                           decision=decision.diagnostics, candidates=candidates), stage_timing_ms=stages,
                           prediction=dict(pose=prediction.pose.tolist(), scale=prediction.scale))
        if self.memory.last_map is None and map_estimate is None:
            raise BootstrapUnsupported(diagnostics)
        return self.memory.commit(raw, ns, map_estimate=map_estimate,
                                  pair_estimate=pair_estimate, diagnostics=diagnostics)
