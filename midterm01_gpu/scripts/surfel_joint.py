"""Experimental planar GNSS/gyro batch graph with independently gated map factors.

The base map posterior is initialization/association only, never an extra prior.
No course, GNSS orientation, accelerometer integration or nonholonomic factors.
"""
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, vstack

from kinematic_graph import KinematicProblem


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class SurfelJointProblem(KinematicProblem):
    def __init__(self, gt, xy, it, quat, gyro, training, factors, config=None):
        super().__init__(gt, xy, it, quat, gyro, training=training, config=config)
        self.lateral_weight[:] = 0.  # body forward axis/antenna lever arm unverified
        self.factors = factors
        self.map_rows = []
        self.map_columns = []
        self.map_values = []
        offset = 0
        for factor in factors:
            node = int(factor['node'])
            whitening = np.asarray(factor['sqrt_information'], float)
            reference = np.asarray(factor['reference_xy_yaw'], float)
            delta = np.asarray(factor['delta_xy_scaled_yaw'], float)
            scale = float(factor['yaw_scale_rad'])
            if (not 0 <= node < self.n or whitening.ndim != 2 or whitening.shape[1] != 3
                    or reference.shape != (3,) or delta.shape != (3,) or scale <= 0
                    or not all(np.isfinite(a).all() for a in (whitening, reference, delta, scale))):
                raise ValueError('Invalid map factor')
            for row in range(len(whitening)):
                for axis, column in enumerate((0, 1, 4)):
                    self.map_rows.append(offset+row); self.map_columns.append(6*node+column)
                    self.map_values.append(whitening[row, axis]/(scale if axis == 2 else 1.))
            offset += len(whitening)
        self.map_jacobian = coo_matrix((self.map_values, (self.map_rows, self.map_columns)),
                                      shape=(offset, 6*self.n)).tocsr()

    def map_residual(self, vector):
        state = np.asarray(vector).reshape(self.n, 6)
        errors = []
        for factor in self.factors:
            s = state[factor['node']]
            ref = np.asarray(factor['reference_xy_yaw'])
            relative = np.r_[s[:2]+self.origin-ref[:2], wrap(s[4]-ref[2])/factor['yaw_scale_rad']]
            errors.append(np.asarray(factor['sqrt_information'])@(relative-factor['delta_xy_scaled_yaw']))
        return np.concatenate(errors) if errors else np.zeros(0)

    def evaluate(self, vector, jacobian=False):
        base = super().evaluate(vector, jacobian)
        return vstack((base, self.map_jacobian), format='csr') if jacobian else np.r_[base, self.map_residual(vector)]

    def solve(self):
        if not self.factors:
            raise ValueError('Map-free graph is not an approved replacement for the scored baseline')
        result = least_squares(self.evaluate, self.initial.ravel(), jac=lambda x: self.evaluate(x, True),
            x_scale='jac', max_nfev=80, ftol=1e-6, xtol=1e-7, gtol=1e-6,
            tr_options={'maxiter': 400, 'atol': 1e-7, 'btol': 1e-7})
        state = result.x.reshape(self.n, 6).copy(); state[:, :2] += self.origin
        return state, {'success': bool(result.success), 'nfev': result.nfev, 'cost': float(result.cost),
            'optimality': float(result.optimality), 'message': result.message,
            'gnss_measurements_used': int(self.training.sum()), 'map_factors': len(self.factors),
            'course_factors': 0, 'gnss_orientation_factors': 0, 'nonholonomic_factors': 0,
            'bias': 'effective yaw-rate bias, not calibrated three-axis IMU bias',
            'accelerometer_preintegration': False}


def make_problem(raw, factors, frame_ns, config=None):
    """One raw GNSS factor per retained timestamp; map nodes at exact integer times."""
    if not factors: return None
    gnss_ns, imu_ns = raw['gnss_bag_ns'], raw['imu_bag_ns']
    covered = ((gnss_ns >= max(imu_ns[0], int(frame_ns[0])-1_000_000_000)) &
               (gnss_ns <= min(imu_ns[-1], int(frame_ns[-1])+1_000_000_000)))
    gnss_ns = gnss_ns[covered]
    if len(gnss_ns) < 5 or np.any(np.diff(gnss_ns) <= 0):
        raise ValueError('Insufficient strictly ordered GNSS coverage')
    factor_ns = np.array([f['timestamp_ns'] for f in factors], np.int64)
    if factor_ns.min() < imu_ns[0] or factor_ns.max() > imu_ns[-1]:
        raise ValueError('Map factors outside IMU coverage')
    nodes = np.unique(np.r_[gnss_ns, factor_ns])
    origin_ns = int(nodes[0])
    gt = (nodes-origin_ns)/1e9
    it = (imu_ns-origin_ns)/1e9
    training = np.isin(nodes, gnss_ns)
    observed_xy = raw['gnss_xyz'][covered, :2]
    xy = np.column_stack([np.interp(gt, (gnss_ns-origin_ns)/1e9, observed_xy[:, a]) for a in range(2)])
    mapped = [{**f, 'node': int(np.searchsorted(nodes, f['timestamp_ns']))} for f in factors]
    problem = SurfelJointProblem(gt, xy, it, raw['imu_quat'], raw['angular_velocity'], training, mapped, config)
    return problem, origin_ns


def interpolate_states(problem, state, query):
    """Hermite XY; gyro propagation plus node correction for continuous yaw."""
    query = np.asarray(query, float)
    gt = problem.gt
    if query.min() < gt[0]-.3 or query.max() > gt[-1]+.3:
        raise ValueError('Joint output beyond 0.3 s edge extrapolation')
    clipped = np.clip(query, gt[0], gt[-1])
    xy = CubicHermiteSpline(gt, state[:, :2], state[:, 2:4])(clipped)
    for side, outside in ((0, query < gt[0]), (-1, query > gt[-1])):
        xy[outside] = state[side, :2]+(query[outside]-gt[side])[:, None]*state[side, 2:4]
    # Within each edge the bias is linear, so its integral is analytic.
    edge = np.clip(np.searchsorted(gt, clipped, side='right')-1, 0, len(gt)-2)
    dt = clipped-gt[edge]; duration = gt[edge+1]-gt[edge]
    gyro = np.interp(clipped, problem.it, problem.integral)-np.interp(gt[edge], problem.it, problem.integral)
    bias_integral = state[edge, 5]*dt+.5*(state[edge+1, 5]-state[edge, 5])*dt**2/duration
    predicted_edge = problem.gyro_angle-.5*(state[:-1, 5]+state[1:, 5])*np.diff(gt)
    correction = state[1:, 4]-state[:-1, 4]-predicted_edge
    yaw = state[edge, 4]+gyro-bias_integral+correction[edge]*dt/duration
    # Constant endpoint yaw is conservative at uncovered sub-frame edges.
    yaw[query < gt[0]] = state[0, 4]; yaw[query > gt[-1]] = state[-1, 4]
    return xy, yaw


def temporal_accept(reports, factors):
    if len(reports) != 3 or any(f is None for f in factors): return False
    center = factors[1]
    _, singular, right = np.linalg.svd(np.asarray(center['sqrt_information']), full_matrices=False)
    directions = right[singular > 1e-10]
    delta = np.asarray(center['delta_xy_scaled_yaw'])
    return all(np.linalg.norm(directions@(np.asarray(f['delta_xy_scaled_yaw'])-delta)) <= .3 for f in factors)


def discount_shared_patches(factors):
    counts = {}
    for factor in factors:
        for patch in set(map(tuple, factor['train_patch_ids'])): counts[patch] = counts.get(patch, 0)+1
    result = []
    for factor in factors:
        overlap = max((counts[p] for p in set(map(tuple, factor['train_patch_ids']))), default=1)
        result.append({**factor, 'sqrt_information': (np.asarray(factor['sqrt_information'])/np.sqrt(overlap)).tolist(),
                       'maximum_patch_reuse': overlap})
    return result
