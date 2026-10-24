import numpy as np
from scipy.spatial.transform import Rotation


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def yaw(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def rz(a):
    return Rotation.from_rotvec([0, 0, a]).as_matrix()


def apply_delta(T, scale, delta):
    out = T.copy()
    out[:3, 3] += delta[:3]
    out[:3, :3] = Rotation.from_rotvec(delta[3:6]).as_matrix() @ T[:3, :3]
    return out, scale * np.exp(delta[6])


def valid_pose(T):
    return bool(T.shape == (4, 4) and np.isfinite(T).all() and
            np.allclose(T[3], [0, 0, 0, 1], atol=1e-7) and
            np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5) and
            abs(np.linalg.det(T[:3, :3])-1) < 1e-5)
