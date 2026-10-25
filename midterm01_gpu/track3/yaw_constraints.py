"""Current-frame robust map-heading weighting inside the existing ESKF update.

This is a one-pass Huber-style covariance inflation, not a second heading
measurement or a trajectory smoother. It cannot identify a common gyro/map
bias, and does not establish that the gyro prediction is ground truth.
"""
import numpy as np


def yaw_jacobian(R):
    """d atan2(R10,R00) / d left/global rotation error, away from vertical."""
    x, y, z = np.asarray(R, dtype=float)[:, 0]
    horizontal2 = x*x + y*y
    if horizontal2 <= np.finfo(float).eps:
        raise ValueError('Camera forward axis is vertical; yaw is undefined')
    return np.array([-x*z/horizontal2, -y*z/horizontal2, 1.])


class YawConstraint:
    def __init__(self, cfg):
        self.enabled = cfg.get('enabled', False)
        self.huber_sigma = float(cfg.get('huber_sigma', 3.))
        self.min_horizontal_norm = float(cfg.get('min_horizontal_norm', .1))
        if not isinstance(self.enabled, bool):
            raise ValueError('yaw_constraint.enabled must be a bool')
        if not np.isfinite(self.huber_sigma) or self.huber_sigma <= 0:
            raise ValueError('yaw_constraint.huber_sigma must be finite and positive')
        if not np.isfinite(self.min_horizontal_norm) or not 0 < self.min_horizontal_norm <= 1:
            raise ValueError('yaw_constraint.min_horizontal_norm must be in (0,1]')

    def covariance(self, predicted_R, registered_R, orientation_covariance, camera_covariance):
        """Return camera-origin map covariance and finite JSON diagnostics.

        The map-axis angular covariance gains a PSD rank-one term. Its projection
        onto measured yaw has variance R_yaw / weight; orthogonal directions are
        unchanged. The IMU lever-arm transformation is applied by the caller
        afterwards, preserving the induced position/orientation covariance.
        """
        C = np.array(camera_covariance, dtype=float, copy=True)
        diag = dict(enabled=self.enabled, weight=1., covariance_inflated=False,
                    huber_sigma=self.huber_sigma, reason='disabled')
        if not self.enabled:
            return C, diag
        predicted_R = np.asarray(predicted_R, dtype=float)
        registered_R = np.asarray(registered_R, dtype=float)
        P = np.asarray(orientation_covariance, dtype=float)
        if (C.shape != (6,6) or P.shape != (3,3) or predicted_R.shape != (3,3) or
                registered_R.shape != (3,3) or not all(np.isfinite(a).all() for a in
                                                     (C,P,predicted_R,registered_R))):
            raise ValueError('Invalid finite yaw covariance inputs')
        norms = [float(np.linalg.norm(R[:2,0])) for R in (predicted_R,registered_R)]
        if min(norms) < self.min_horizontal_norm:
            diag.update(reason='near_vertical_forward_axis', horizontal_forward_norms=norms)
            return C, diag
        jp, jm = yaw_jacobian(predicted_R), yaw_jacobian(registered_R)
        difference = (np.arctan2(registered_R[1,0],registered_R[0,0]) -
                      np.arctan2(predicted_R[1,0],predicted_R[0,0]))
        innovation = float(np.arctan2(np.sin(difference),np.cos(difference)))
        prior_variance = float(jp @ P @ jp)
        map_variance = float(jm @ C[3:,3:] @ jm)
        if prior_variance < -1e-12 or map_variance <= 0:
            raise ValueError('Yaw variances must be nonnegative/positive')
        prior_variance = max(prior_variance,0.)
        total = prior_variance + map_variance
        if not np.isfinite(total) or total <= 0:
            raise ValueError('Invalid yaw innovation variance')
        normalized = abs(innovation) / np.sqrt(total)
        weight = min(1., self.huber_sigma / normalized) if normalized > 0 else 1.
        direction = jm / (jm @ jm)
        extra = map_variance * (1./weight - 1.)
        C[3:,3:] += extra * np.outer(direction,direction)
        diag.update(reason='huber_downweighted' if weight < 1 else 'within_huber_region',
                    weight=float(weight), covariance_inflated=bool(weight < 1),
                    innovation_rad=innovation, innovation_deg=float(np.degrees(innovation)),
                    normalized_innovation=float(normalized),
                    predicted_std_deg=float(np.degrees(np.sqrt(prior_variance))),
                    base_map_std_deg=float(np.degrees(np.sqrt(map_variance))),
                    effective_map_std_deg=float(np.degrees(np.sqrt(map_variance/weight))),
                    prior_yaw_jacobian=jp.tolist(), map_yaw_jacobian=jm.tolist())
        return C, diag
