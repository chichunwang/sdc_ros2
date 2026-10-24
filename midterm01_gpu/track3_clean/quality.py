"""Training-only selection followed by held-out veto; no state mutation."""
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation

from .config import RegistrationConfig
from .contracts import Decision


def differences(a, b):
    p, q = np.asarray(a['pose']), np.asarray(b['pose'])
    return (float(np.linalg.norm(p[:3, 3]-q[:3, 3])),
            float(np.rad2deg(Rotation.from_matrix(p[:3, :3] @ q[:3, :3].T).magnitude())),
            float(abs(np.log(a['scale']/b['scale']))))


def training_failures(candidate, config, require_ray):
    reasons = [k for k in ('converged', 'interior', 'observable') if not candidate['checks'][k]]
    t = candidate['training']
    if (t['coverage'] < config.min_coverage or t['median_m'] > config.validation_median_m
            or t['p90_m'] > config.validation_p90_m):
        reasons.append('training_geometry')
    if require_ray:
        ray = candidate.get('ray_depth', {})
        r = ray.get('training', {})
        depth = r.get('median_abs_log_depth_ratio')
        if (not ray.get('available') or r.get('support_fraction', 0) < config.min_coverage
                or depth is None or depth > .15):
            reasons.append('training_visibility')
    return reasons


def decide(candidates, *, require_ray=True, config=None):
    """Never choose or resolve ambiguity using held-out measurements.

    Every plausible training basin must agree with the actual selected winner.
    Candidate-level rejection counts remain available even when none survive.
    """
    c = config or RegistrationConfig()
    rejected = [training_failures(r, c, require_ray) for r in candidates]
    usable = [r for r, failures in zip(candidates, rejected) if not failures]
    diagnostics = dict(candidate_count=len(candidates), training_supported=len(usable),
                       candidate_rejection_counts=dict(Counter(k for rr in rejected for k in rr)))
    if not usable:
        return Decision(False, None, ('no_training_supported_fit',), diagnostics)
    key = ((lambda r: r['ray_depth']['training']['median_abs_log_depth_ratio'])
           if require_ray else (lambda r: r['training_cost']))
    chosen = min(usable, key=key)
    best = min(r['training_cost'] for r in usable)
    plausible = [r for r in usable if r['training_cost'] <= best*1.5+.05]
    limits = (c.consensus_translation_m, c.consensus_rotation_deg, c.consensus_log_scale)
    gaps = [differences(a, b) for i, a in enumerate(plausible) for b in plausible[i+1:]]
    reasons = []
    identities = {(tuple(np.asarray(r['seed_pose']).ravel()), r['seed_scale'])
                  if 'seed_pose' in r else ('unspecified_test_seed', i)
                  for i, r in enumerate(plausible)}
    if len(identities) < 2:
        reasons.append('insufficient_independent_initializations')
    if (not any(chosen is r for r in plausible)
            or any(any(v > limit for v, limit in zip(gap, limits)) for gap in gaps)):
        reasons.append('ambiguous_training_basins')
    # The selected winner is now fixed. Held-out evidence may only veto it.
    reasons.extend(k for k, value in chosen['checks'].items() if not value)
    if require_ray:
        r = chosen['ray_depth'].get('validation', {})
        depth = r.get('median_abs_log_depth_ratio')
        if r.get('support_fraction', 0) < c.min_coverage or depth is None or depth > .15:
            reasons.append('heldout_visibility')
    diagnostics.update(plausible_starts=len(plausible), independent_starts=len(identities), basin_differences=gaps,
                       selected_seed=chosen.get('seed_scale'),
                       accuracy='unknown; local consensus is not global uniqueness or truth')
    return Decision(not reasons, chosen, tuple(dict.fromkeys(reasons)), diagnostics)
