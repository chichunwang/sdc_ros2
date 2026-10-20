"""平面 EKF 與時間對齊；只依賴 NumPy，不接收 ground truth。"""
import heapq
import math

import numpy as np


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=float)


def covariance(matrix, size):
    """拒絕非有限／明顯非 PSD covariance，只修正浮點數等級的誤差。"""
    value = np.asarray(matrix, dtype=float)
    if value.shape != (size, size) or not np.isfinite(value).all():
        raise ValueError(f"Expected finite {size}x{size} covariance")
    if not np.allclose(value, value.T, rtol=1e-6, atol=1e-9):
        raise ValueError("Covariance is not symmetric")
    value = (value + value.T) * 0.5
    eigenvalues, vectors = np.linalg.eigh(value)
    tolerance = 1e-9 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if eigenvalues[0] < -tolerance:
        raise ValueError("Covariance has negative eigenvalues")
    return (vectors * np.maximum(eigenvalues, 1e-12)) @ vectors.T


def vector(value, size):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected finite vector of length {size}")
    return result.copy()


class ExtendedKalmanFilter:
    """狀態 [x,y,yaw]，控制 [局部 dx,局部 dy,delta yaw]。

    延用老師符號：S=state covariance、R=control covariance、Q=GPS
    covariance；與某些課本的 Q/R 命名相反。R 經 B 轉到 global frame。
    """
    def __init__(self, x, y, yaw=0.0):
        self.pose = vector([x, y, wrap_angle(yaw)], 3)
        self.A = np.eye(3)
        self.B = np.eye(3)
        self.S = np.eye(3)
        self.C = np.array([[1., 0., 0.], [0., 1., 0.]])
        self.R = np.eye(3)
        self.Q = np.eye(2)

    def predict(self, u):
        dx, dy, dyaw = vector(u, 3)
        yaw = self.pose[2]
        c, s = math.cos(yaw), math.sin(yaw)
        self.A = np.array([[1., 0., -s*dx-c*dy],
                           [0., 1., c*dx-s*dy], [0., 0., 1.]])
        self.B = np.eye(3)
        self.B[:2, :2] = rotation(yaw)
        self.pose += np.array([c*dx-s*dy, s*dx+c*dy, dyaw])
        self.pose[2] = wrap_angle(self.pose[2])
        self.S = covariance(self.A @ self.S @ self.A.T
                            + self.B @ covariance(self.R, 3) @ self.B.T, 3)
        return self.pose.copy(), self.S.copy()

    def update(self, z):
        measurement = vector(z, 2)
        noise = covariance(self.Q, 2)
        innovation = measurement - self.C @ self.pose
        innovation_cov = self.C @ self.S @ self.C.T + noise
        gain = np.linalg.solve(innovation_cov, self.C @ self.S).T
        self.pose += gain @ innovation
        self.pose[2] = wrap_angle(self.pose[2])
        residual = np.eye(3) - gain @ self.C
        # Joseph form 保持 covariance 的對稱性與半正定性。
        self.S = covariance(residual @ self.S @ residual.T
                            + gain @ noise @ gain.T, 3)
        return self.pose.copy(), self.S.copy()


def relative_control(previous, current):
    """T_previous^-1 T_current；局部控制不含 global 座標原點。"""
    previous, current = vector(previous, 3), vector(current, 3)
    result = np.empty(3)
    result[:2] = rotation(-previous[2]) @ (current[:2] - previous[:2])
    result[2] = wrap_angle(current[2] - previous[2])
    return result


def relative_covariance(previous, current, previous_cov, current_cov):
    """以相鄰 absolute radar poses 的一階 Jacobian 傳遞到局部控制。

    bag 未提供相鄰 pose 的 cross covariance，這裡明確假設其誤差獨立。
    累積里程計通常相關，因此這只是可解釋的近似，不是校準過的噪聲。
    """
    dx, dy, _ = relative_control(previous, current)
    c, s = math.cos(previous[2]), math.sin(previous[2])
    ja = np.array([[-c, -s, dy], [s, -c, -dx], [0., 0., -1.]])
    jb = np.eye(3)
    jb[:2, :2] = rotation(-previous[2])
    return covariance(ja @ covariance(previous_cov, 3) @ ja.T
                      + jb @ covariance(current_cov, 3) @ jb.T, 3)


class RadarGpsFilter:
    """GPS 暫存到下一個 radar，依 header timestamp 插值並按時間融合。

    radar interval 內使用線性位置、最短角差插值，最多一個 radar interval
    延遲。晚於已處理時間才送達的 GPS 丟棄並計數，不做過期量測平滑。
    """
    def __init__(self):
        self.ekf = None
        self.last_radar = None
        self.pending_gps = []
        self.sequence = 0
        self.stats = dict(radar=0, gps=0, gps_updates=0, late_gps=0,
                          gps_before_radar=0, nonincreasing_radar=0)

    def push_gps(self, stamp_ns, position, noise):
        position, noise = vector(position, 2), covariance(noise, 2)
        self.stats['gps'] += 1
        if self.last_radar is not None and stamp_ns < self.last_radar[0]:
            self.stats['late_gps'] += 1
            return []
        if self.last_radar is not None and stamp_ns == self.last_radar[0]:
            self._update(position, noise)
            return [self._snapshot(stamp_ns)]
        heapq.heappush(self.pending_gps, (stamp_ns, self.sequence, position, noise))
        self.sequence += 1
        return []

    def _update(self, position, noise):
        self.ekf.Q = noise
        self.ekf.update(position)
        self.stats['gps_updates'] += 1

    def _snapshot(self, stamp_ns):
        return stamp_ns, self.ekf.pose.copy(), self.ekf.S.copy()

    def push_radar(self, stamp_ns, pose, noise):
        pose, noise = vector(pose, 3), covariance(noise, 3)
        self.stats['radar'] += 1
        if self.last_radar is None:
            # yaw 一律由 radar 初始化，避免 GPS 先到時假設車頭朝東。
            self.ekf = ExtendedKalmanFilter(*pose)
            self.ekf.S = noise
            while self.pending_gps and self.pending_gps[0][0] <= stamp_ns:
                stamp, _, position, gps_noise = heapq.heappop(self.pending_gps)
                if stamp == stamp_ns:
                    self._update(position, gps_noise)
                else:
                    self.stats['gps_before_radar'] += 1
            self.last_radar = stamp_ns, pose, noise
            return [self._snapshot(stamp_ns)]

        start_ns, start, start_cov = self.last_radar
        if stamp_ns <= start_ns:
            self.stats['nonincreasing_radar'] += 1
            return []
        total_noise = relative_covariance(start, pose, start_cov, noise)
        previous, previous_alpha = start, 0.0
        outputs = []

        def advance(alpha, target):
            nonlocal previous, previous_alpha
            fraction = alpha - previous_alpha
            if fraction > 0:
                # 按時間分配 interval covariance，並換到 segment 起點局部 frame。
                basis = np.eye(3)
                basis[:2, :2] = rotation(start[2] - previous[2])
                self.ekf.R = fraction * (basis @ total_noise @ basis.T)
                self.ekf.predict(relative_control(previous, target))
            previous, previous_alpha = target, alpha

        while self.pending_gps and self.pending_gps[0][0] <= stamp_ns:
            gps_ns, _, position, gps_noise = heapq.heappop(self.pending_gps)
            if gps_ns < start_ns:
                self.stats['late_gps'] += 1
                continue
            alpha = (gps_ns - start_ns) / (stamp_ns - start_ns)
            interpolated = start + alpha * (pose - start)
            interpolated[2] = wrap_angle(start[2] + alpha * wrap_angle(pose[2]-start[2]))
            advance(alpha, interpolated)
            self._update(position, gps_noise)
            outputs.append(self._snapshot(gps_ns))
        advance(1.0, pose)
        outputs.append(self._snapshot(stamp_ns))
        self.last_radar = stamp_ns, pose, noise
        return outputs
