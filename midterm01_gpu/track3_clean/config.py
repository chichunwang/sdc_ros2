"""Explicit engineering domains, carried from v2; not calibrated uncertainty."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class TrackingConfig:
    bootstrap_scales: tuple = (3., 6., 9.)
    restart_factors: tuple = (.95, 1., 1.05)
    initial_scale: float = 3.
    max_log_scale_change: float = .15
    local_translation_m: float = 2.
    local_rotation_deg: float = 8.
    reacquisition_translation_m: float = 8.
    reacquisition_rotation_deg: float = 30.
    max_speed_mps: float = 40.
    max_rotation_degps: float = 90.
    pair_translation_floor_m: float = .25
    pair_rotation_floor_deg: float = .5
    map_translation_slack_m: float = .5
    map_rotation_slack_deg: float = 3.
    lost_after_seconds: float = 2.
    map_half_size_m: float = 65.
    map_recenter_distance_m: float = 20.
    map_leaf_m: float = .25
    source_leaf_raw: float = .03
    source_max_points: int = 12000

    def __post_init__(self):
        for v in asdict(self).values():
            values = v if isinstance(v, tuple) else (v,)
            if not values or any(isinstance(x, bool) or not np.isfinite(x) or x <= 0 for x in values):
                raise ValueError('Tracking parameters must be finite and positive')
        if self.source_max_points != int(self.source_max_points):
            raise ValueError('source_max_points must be an integer')

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


