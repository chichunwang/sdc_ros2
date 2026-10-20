"""Conservative map-minus-GNSS block corrections and an eight-state batch graph.

z_GNSS = p + b_GNSS + noise. The map supplies slow bias/yaw anchors only;
raw gyro supplies relative yaw. Shared model settings are not calibrated noise.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

from kinematic_graph import KinematicConfig, KinematicProblem


def wrap(x):
    return (np.asarray(x)+np.pi) % (2*np.pi)-np.pi


@dataclass(frozen=True)
class SlowMapConfig:
    window_s: float = 12.
    min_block_span_s: float = 1.
    min_block_observations: int = 8
    max_anchor_gap_s: float = 3.
    displacement_gate_m: float = .5
    displacement_gate_mps: float = 1.
    gnss_bias_correlation_s: float = 25.
    gnss_bias_stationary_std_m: float = 1.5
    initial_gnss_bias_std_m: float = 3.
    map_correction_std_m: float = .5
    map_yaw_std_deg: float = 1.
    bounded_loss_sigma: float = 3.


def bounded_vector(value, cap):
    """Bounded squared cost and its EXACT derivative (Geman-McClure)."""
    value = np.asarray(value, float)
    scale = 1./np.sqrt(1.+np.dot(value, value)/cap**2)
    return value*scale, scale*np.eye(len(value))-scale**3*np.outer(value, value)/cap**2


def temporal_mask(times, map_xy, gnss_xy, accepted, config):
    """Require agreement with a genuinely adjacent source frame, not a distant one."""
    times = np.asarray(times, float); accepted = np.asarray(accepted, bool)
    okay = np.zeros(len(times), bool)
    discrepancy = np.linalg.norm(np.diff(map_xy, axis=0)-np.diff(gnss_xy, axis=0), axis=1)
    dt = np.diff(times)
    edge_ok = (accepted[:-1] & accepted[1:] & (dt > 0) & (dt <= config.max_anchor_gap_s)
               & (discrepancy <= config.displacement_gate_m+config.displacement_gate_mps*dt))
    okay[:-1] |= edge_ok; okay[1:] |= edge_ok
    return okay, discrepancy


def slow_anchors(times, map_xy, map_yaw, gnss_xy, reference_yaw, accepted, config=None):
    config = config or SlowMapConfig()
    times = np.asarray(times, float)
    mask, discrepancy = temporal_mask(times, map_xy, gnss_xy, accepted, config)
    anchors = []
    # Nonoverlapping windows: correlated sliding medians must not masquerade as
    # hundreds of independent absolute observations. No endpoint extrapolation.
    blocks = np.floor((times-times[0])/config.window_s).astype(int)
    for block in np.unique(blocks):
        ids = np.flatnonzero((blocks == block)&mask)
        if len(ids) < config.min_block_observations or times[ids[-1]]-times[ids[0]] < config.min_block_span_s:
            continue
        # Split at uncovered gaps so a missing interval is never bridged by a median.
        for segment in np.split(ids, np.flatnonzero(np.diff(times[ids]) > config.max_anchor_gap_s)+1):
            if len(segment) < config.min_block_observations or times[segment[-1]]-times[segment[0]] < config.min_block_span_s:
                continue
            delta = map_xy[segment]-gnss_xy[segment]
            correction = np.median(delta, axis=0)
            yaw_delta = wrap(map_yaw[segment]-reference_yaw[segment])
            # Circular centering before the scalar median handles the +/-pi cut.
            center = np.angle(np.mean(np.exp(1j*yaw_delta)))
            yaw_correction = center+np.median(wrap(yaw_delta-center))
            stamp = float(np.median(times[segment]))
            anchors.append({'time_s': stamp, 'bias_xy': (-correction).tolist(),
                'yaw': float(np.interp(stamp, times, reference_yaw)+yaw_correction),
                'observations': len(segment), 'span_s': float(times[segment[-1]]-times[segment[0]]),
                'correction_mad_xy_m': np.median(abs(delta-correction), axis=0).tolist(),
                'yaw_mad_deg': float(np.degrees(np.median(abs(wrap(yaw_delta-yaw_correction))))),
                'first_frame': int(segment[0]), 'last_frame': int(segment[-1])})
    return anchors, mask, discrepancy


class SlowBiasProblem:
    def __init__(self, gt, xy, it, quat, gyro, anchors, config=None):
        self.config = config or SlowMapConfig()
        if not all(np.isfinite(v) and v > 0 for v in asdict(self.config).values()):
            raise ValueError('Invalid slow-map configuration')
        self.base = KinematicProblem(gt, xy, it, quat, gyro, config=KinematicConfig())
        self.n = self.base.n; self.gt = self.base.gt
        self.anchors = anchors
        self.initial = np.zeros((self.n, 8)); self.initial[:, :6] = self.base.initial
        # Seed yaw close to the existing course-corrected heading only in caller;
        # this is initialization, not another independent observation.
        self.alpha = np.exp(-self.base.dt/self.config.gnss_bias_correlation_s)
        self.bias_std = self.config.gnss_bias_stationary_std_m*np.sqrt(1-self.alpha**2)
        self.anchor_edges = []
        for anchor in anchors:
            t = anchor['time_s']
            if t < self.gt[0] or t > self.gt[-1]:
                raise ValueError('Slow anchor outside GNSS coverage')
            edge = min(np.searchsorted(self.gt, t, side='right')-1, self.n-2)
            fraction = (t-self.gt[edge])/(self.gt[edge+1]-self.gt[edge])
            self.anchor_edges.append((edge, fraction))

    def evaluate(self, vector, jacobian=False):
        state = np.asarray(vector).reshape(self.n, 8)
        # Reuse the tested motion, raw-gyro, effective gyro-bias and soft NHC
        # factors, replacing its GNSS rows by p+b-z. No GNSS factor is duplicated.
        base_vector = state[:, :6].ravel()
        base_residual = self.base.evaluate(base_vector)
        rows, cols, values = [], [], []
        if jacobian:
            jac = self.base.evaluate(base_vector, True).tocoo()
            keep = jac.row >= 2*self.n
            rows.extend(jac.row[keep]); cols.extend((jac.col[keep]//6)*8+jac.col[keep]%6)
            values.extend(jac.data[keep])
        residual = list(base_residual)
        difference = (state[:, :2]+state[:, 6:8]-self.base.measured)/self.base.config.gnss_std_m
        for i, d in enumerate(difference):
            # Huber GNSS as in the base model; differentiable vector norm gate.
            norm = np.linalg.norm(d); gate = self.base.config.robust_gate_sigma
            scale = 1. if norm <= gate else np.sqrt((2*gate*norm-gate**2)/norm**2)
            derivative = 0. if norm <= gate else (-gate/norm**3+gate**2/norm**4)/scale
            block = (scale*np.eye(2)+derivative*np.outer(d, d))/self.base.config.gnss_std_m
            residual[2*i:2*i+2] = d*scale
            if jacobian:
                for a in range(2):
                    for b in range(2):
                        for shift in (0, 6):
                            rows.append(2*i+a); cols.append(8*i+b+shift); values.append(block[a, b])
        offset = len(residual)
        rw = (state[1:, 6:8]-self.alpha[:, None]*state[:-1, 6:8])/self.bias_std[:, None]
        residual.extend(rw.ravel())
        if jacobian:
            for i in range(self.n-1):
                for a in range(2):
                    rows.extend([offset+2*i+a]*2); cols.extend([8*i+6+a, 8*(i+1)+6+a])
                    values.extend([-self.alpha[i]/self.bias_std[i], 1/self.bias_std[i]])
        offset += 2*(self.n-1)
        residual.extend(state[0, 6:8]/self.config.initial_gnss_bias_std_m)
        if jacobian:
            rows.extend([offset, offset+1]); cols.extend([6, 7]); values.extend([1/self.config.initial_gnss_bias_std_m]*2)
        offset += 2
        for anchor, (edge, fraction) in zip(self.anchors, self.anchor_edges):
            weights = (1-fraction, fraction)
            bias = weights[0]*state[edge, 6:8]+weights[1]*state[edge+1, 6:8]
            xy_r, xy_j = bounded_vector((bias-np.asarray(anchor['bias_xy']))/self.config.map_correction_std_m,
                                        self.config.bounded_loss_sigma)
            # Raw gyro propagation to the anchor; interpolate residual node
            # correction, avoiding a delayed smoothed-quaternion yaw shape.
            t = anchor['time_s']; duration = self.base.dt[edge]; delta_t = t-self.gt[edge]
            integral = np.interp(t, self.base.it, self.base.integral)-np.interp(self.gt[edge], self.base.it, self.base.integral)
            endpoint_integral = self.base.gyro_angle[edge]
            yaw = (weights[0]*state[edge, 4]+weights[1]*state[edge+1, 4]+integral-fraction*endpoint_integral
                   +.5*fraction*(1-fraction)*duration*(state[edge+1, 5]-state[edge, 5]))
            yaw_std = np.deg2rad(self.config.map_yaw_std_deg)
            yaw_r, yaw_j = bounded_vector(np.array([wrap(yaw-anchor['yaw'])/yaw_std]), self.config.bounded_loss_sigma)
            residual.extend(xy_r); residual.extend(yaw_r)
            if jacobian:
                for side in (0, 1):
                    for a in range(2):
                        for b in range(2):
                            rows.append(offset+a); cols.append(8*(edge+side)+6+b)
                            values.append(weights[side]*xy_j[a, b]/self.config.map_correction_std_m)
                    rows.append(offset+2); cols.append(8*(edge+side)+4)
                    values.append(weights[side]*yaw_j[0, 0]/yaw_std)
                    rows.append(offset+2); cols.append(8*(edge+side)+5)
                    values.append(((-1 if side == 0 else 1)*.5*fraction*(1-fraction)*duration)*yaw_j[0, 0]/yaw_std)
            offset += 3
        if jacobian:
            return coo_matrix((values, (rows, cols)), shape=(len(residual), 8*self.n)).tocsr()
        return np.asarray(residual)

    def solve(self):
        solution = least_squares(self.evaluate, self.initial.ravel(), jac=lambda x: self.evaluate(x, True),
            x_scale='jac', ftol=1e-6, xtol=1e-7, gtol=1e-6, max_nfev=80,
            tr_options={'maxiter': 400, 'atol': 1e-7, 'btol': 1e-7})
        state = solution.x.reshape(self.n, 8).copy(); state[:, :2] += self.base.origin
        report = {'success': bool(solution.success), 'status': int(solution.status), 'message': solution.message,
                  'nfev': int(solution.nfev), 'cost': float(solution.cost), 'optimality': float(solution.optimality),
                  'gnss_measurements_used_once': self.n, 'slow_map_anchors': len(self.anchors),
                  'initial_cost': float(.5*np.sum(self.evaluate(self.initial.ravel())**2)),
                  'gnss_bias_range_xy_m': [state[:, 6:8].min(axis=0).tolist(), state[:, 6:8].max(axis=0).tolist()]}
        return state, report


def smooth_trajectory(raw, frame_ns, prior_poses, observations, config=None):
    config = config or SlowMapConfig()
    frame_ns = np.asarray(frame_ns, np.int64); prior_poses = np.asarray(prior_poses, float)
    if (len(frame_ns) < 2 or len(observations) != len(frame_ns) or prior_poses.shape != (len(frame_ns), 4, 4)
            or np.any(np.diff(frame_ns) <= 0) or not np.isfinite(prior_poses).all()):
        raise ValueError('Invalid frame/prior/observation contract')
    gnss_ns = np.asarray(raw['gnss_bag_ns'], np.int64); imu_ns = np.asarray(raw['imu_bag_ns'], np.int64)
    # Keep original timestamps as int64 until after subtracting a shared origin.
    origin = int(gnss_ns[0]); gt_all = (gnss_ns-origin)/1e9; it = (imu_ns-origin)/1e9
    covered = (gnss_ns >= imu_ns[0])&(gnss_ns <= imu_ns[-1])
    gt = gt_all[covered]; xy = np.asarray(raw['gnss_xyz'])[covered, :2]
    times = (frame_ns-origin)/1e9
    if times.min() < gt[0]-.3 or times.max() > gt[-1]+.3:
        raise ValueError('Frames outside GNSS edge allowance')
    map_pose = np.array([r['pose'] for r in observations], float)
    good = np.array([r['accepted'] for r in observations], bool)
    map_xy = map_pose[:, :2, 3]; map_yaw = np.arctan2(map_pose[:, 1, 0], map_pose[:, 0, 0])
    gnss_at_frames = np.column_stack([np.interp(times, gt, xy[:, a]) for a in range(2)])
    base = KinematicProblem(gt, xy, it, raw['imu_quat'], raw['angular_velocity'])
    prior_yaw = np.unwrap(np.arctan2(prior_poses[:, 1, 0], prior_poses[:, 0, 0]))
    # Only relative gyro content within a block matters here. A constant seed
    # rate bias is removed; the graph subsequently estimates its varying bias.
    seed_bias = float(np.median(base.initial[:, 5]))
    ref = np.interp(times, it, base.integral)-seed_bias*(times-times[0])
    ref += prior_yaw[0]-ref[0]
    anchors, mask, discrepancy = slow_anchors(times, map_xy, map_yaw, gnss_at_frames, ref, good, config)
    anchors = [a for a in anchors if gt[0] <= a['time_s'] <= gt[-1]]
    report = {'schema': 'layered_scale_slow_graph_v1', 'config': asdict(config),
              'motion_config': asdict(base.config), 'geometric_map_frames': int(good.sum()),
              'temporal_map_frames': int(mask.sum()), 'slow_anchors': anchors,
              'applied': False, 'joint_estimation_executed': False,
              'limits': ['Shared noise/bounds/time constants are experimental, not calibrated accuracy.',
                         'Forward FLU axis and gyro/quaternion same-axis assumptions remain unverified extrinsics.',
                         'Slow map-minus-GNSS anchors share GNSS noise; conservative block weight is not exact covariance.',
                         'No GNSS orientation/course factors or accelerometer integration; NHC is a soft vehicle assumption.',
                         'A coherent wrong map cluster can still pass consistency; bounded loss does not certify truth.']}
    arrays = {'frame_time_ns': frame_ns, 'temporal_map_mask': mask, 'adjacent_discrepancy_m': discrepancy}
    if not anchors:
        report['reason'] = 'no_supported_slow_map_anchors; preserve motion_rts sensor initialization'
        return prior_poses.copy(), report, arrays
    problem = SlowBiasProblem(gt, xy, it, raw['imu_quat'], raw['angular_velocity'], anchors, config)
    # Initialization only, never repeated posterior factors.
    # Outside a short test window retain the IMU turn shape and extend only
    # its heading-offset seed, rather than clamping the entire trajectory yaw.
    heading_offset = prior_yaw-np.interp(times, gt, base.imu_yaw)
    problem.initial[:, 4] = base.imu_yaw+np.interp(gt, times, heading_offset)
    state, solver = problem.solve()
    report.update(solver=solver, joint_estimation_executed=True)
    if not solver['success'] or not np.isfinite(state).all():
        raise RuntimeError('Slow map batch solver failed: '+str(solver))
    query = np.clip(times, gt[0], gt[-1])
    new_xy = CubicHermiteSpline(gt, state[:, :2], state[:, 2:4])(query)
    edges = np.clip(np.searchsorted(gt, query, side='right')-1, 0, len(gt)-2)
    dt = query-gt[edges]; duration = gt[edges+1]-gt[edges]; f = dt/duration
    integral = np.interp(query, it, base.integral)-np.interp(gt[edges], it, base.integral)
    yaw = ((1-f)*state[edges, 4]+f*state[edges+1, 4]+integral-f*base.gyro_angle[edges]
           +.5*f*(1-f)*duration*(state[edges+1, 5]-state[edges, 5]))
    for edge, outside in ((0, times < gt[0]), (-1, times > gt[-1])):
        new_xy[outside] = state[edge, :2]+(times[outside]-gt[edge])[:, None]*state[edge, 2:4]
        edge_gyro = np.interp(times[outside], it, base.integral)-np.interp(gt[edge], it, base.integral)
        yaw[outside] = state[edge, 4]+edge_gyro-state[edge, 5]*(times[outside]-gt[edge])
    final = prior_poses.copy()
    final[:, :2, 3] = new_xy
    final[:, :3, :3] = Rotation.from_euler('z', wrap(yaw-prior_yaw)[:, None]).as_matrix()@prior_poses[:, :3, :3]
    report.update(applied=True, reason='slow_map_bias_and_yaw_anchors_with_raw_gyro_NHC_batch',
        xy_change_rms_m=float(np.sqrt(np.mean(np.sum((new_xy-prior_poses[:, :2, 3])**2, axis=1)))),
        yaw_change_rms_deg=float(np.degrees(np.sqrt(np.mean(wrap(yaw-prior_yaw)**2)))))
    arrays.update(node_time_s=gt, states=state, time_origin_ns=np.int64(origin))
    return final, report, arrays
