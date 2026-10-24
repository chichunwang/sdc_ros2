"""Causal pose state: predict a prior, then commit or discard a registration.

No sensor fusion. A rejected candidate can never leak its rotation, translation
or scale into the next frame. All transforms are T_map_ply; scale is separate.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import valid_pose


class PoseState:
    def __init__(self, initial_pose, initial_scale, prediction='previous_pose',
                 scale_bounds=(0.2, 30.)):
        if not valid_pose(initial_pose):
            raise ValueError('Invalid initial rigid pose')
        if prediction not in ('previous_pose', 'constant_velocity','visual_pnp'):
            raise ValueError(f'Unknown prediction mode: {prediction}')
        self.bounds = tuple(scale_bounds)
        if not self.bounds[0] < initial_scale < self.bounds[1]:
            raise ValueError('Initial scale outside registration bounds')
        self.T = initial_pose.copy()
        self.scale = float(initial_scale)
        self.prediction = prediction
        self.initialized = False
        self.stamp = None
        self.accepted_history = []
        self.pending = None

    def predict(self, timestamp_ns, motion=None):
        if not isinstance(timestamp_ns, (int, np.integer)):
            raise ValueError('Frame timestamp must be integer nanoseconds')
        if self.pending is not None:
            raise RuntimeError('Commit the preceding frame before predicting again')
        if self.stamp is not None and timestamp_ns <= self.stamp:
            raise ValueError('Frame timestamps must strictly increase')
        prior = self.T.copy()
        kind = 'initial_guess' if self.stamp is None else 'previous_pose'
        if self.prediction=='visual_pnp' and motion is not None:
            if not valid_pose(motion):raise ValueError('Invalid visual relative pose')
            prior=self.T@motion
            kind='visual_pnp'
        # Only two consecutive accepted observations establish a velocity.
        # Rejection clears history; uncorrected priors never train the model.
        if self.prediction == 'constant_velocity' and len(self.accepted_history) == 2:
            (ta, a), (tb, b) = self.accepted_history
            ratio = (int(timestamp_ns)-tb)/(tb-ta)
            prior[:3, 3] += ratio*(b[:3, 3]-a[:3, 3])
            omega = Rotation.from_matrix(b[:3, :3] @ a[:3, :3].T).as_rotvec()
            prior[:3, :3] = Rotation.from_rotvec(ratio*omega).as_matrix() @ b[:3, :3]
            kind = 'constant_velocity'
        if not valid_pose(prior):
            raise ValueError('Prediction produced an invalid pose')
        self.pending = (int(timestamp_ns), prior.copy(), self.scale, kind)
        return prior

    def commit(self, estimate):
        if self.pending is None:
            raise RuntimeError('Predict a prior before committing')
        stamp, prior, prior_scale, prediction_kind = self.pending
        candidate = np.asarray(estimate['T'], dtype=float)
        scale = float(estimate['scale'])
        accepted = bool(estimate['accepted'])
        reason = estimate['reason']
        candidate_valid = (valid_pose(candidate) and np.isfinite(scale) and
                           self.bounds[0] < scale < self.bounds[1])
        if accepted and not candidate_valid:
            accepted = False
            reason = 'invalid_candidate_pose_or_scale'
        delta = None
        if candidate_valid:
            delta = dict(translation_m=(candidate[:3, 3]-prior[:3, 3]).tolist(),
                         rotation_rad=Rotation.from_matrix(
                             candidate[:3, :3] @ prior[:3, :3].T).as_rotvec().tolist(),
                         log_scale=float(np.log(scale/prior_scale)))
        self.T = candidate.copy() if accepted else prior.copy()
        self.scale = scale if accepted else prior_scale
        if accepted:
            self.initialized = True
            self.accepted_history.append((stamp, self.T.copy()))
            self.accepted_history = self.accepted_history[-2:]
        else:
            self.accepted_history.clear()
        self.stamp = stamp
        self.pending = None
        return dict(accepted=accepted, reason=reason, prediction=prediction_kind,
                    decision='accepted_registration' if accepted else 'retained_prior',
                    delta=delta)
