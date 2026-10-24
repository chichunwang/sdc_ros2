"""Frame and unit contracts shared by numerical code, state and output."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import numbers

import numpy as np


class GeometryUnavailable(ValueError):
    """Expected lack of geometric support; distinct from CUDA/programming failures."""


class BootstrapUnsupported(RuntimeError):
    def __init__(self, diagnostics):
        super().__init__('Current-frame bootstrap has no unambiguous map support')
        self.diagnostics = diagnostics


def checked_timestamp(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError('timestamp_ns must be a positive integer, never float seconds')
    return int(value)


@dataclass(frozen=True)
class Similarity:
    """p_target_m = scale_m_per_raw * R_target_source @ p_raw + t_target_m.

    The 4x4 pose stores ONLY R,t. Scale is never hidden in its rotation.
    """
    pose: np.ndarray
    scale: float

    def __post_init__(self):
        p = np.array(self.pose, dtype=np.float64, copy=True)
        if (p.shape != (4, 4) or not np.isfinite(p).all()
                or not np.allclose(p[3], [0, 0, 0, 1], rtol=0, atol=1e-9)
                or not np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), rtol=0, atol=1e-6)
                or abs(np.linalg.det(p[:3, :3])-1) > 1e-6):
            raise ValueError('Expected a proper source-to-target R,t pose')
        if isinstance(self.scale, bool) or not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Expected finite positive metres per raw unit')
        p.setflags(write=False)
        object.__setattr__(self, 'pose', p)
        object.__setattr__(self, 'scale', float(self.scale))

    def transform(self, raw):
        return self.scale*np.asarray(raw) @ self.pose[:3, :3].T+self.pose[:3, 3]

    def in_map(self, previous):
        """Compose a fit into PREVIOUS CAMERA METRES with previous map R,t.

        The local target was previous.scale * previous_raw, so self.scale
        already has metres/raw units: multiplying previous.scale again is wrong.
        """
        return Similarity(previous.pose @ self.pose, self.scale)


@dataclass(frozen=True)
class Frame:
    original_idx: int
    timestamp_ns: int
    path: Path


class TrackingState(str, Enum):
    INITIALIZING = 'initializing'
    TRACKING = 'tracking'
    REACQUIRING = 'reacquiring'
    LOST = 'lost'


@dataclass(frozen=True)
class Decision:
    accepted: bool
    selected: dict | None
    reasons: tuple[str, ...]
    diagnostics: dict


@dataclass(frozen=True)
class FrameResult:
    estimate: Similarity
    timestamp_ns: int
    state: TrackingState
    map_accepted: bool
    pair_accepted: bool
    source: str
    unsupported_seconds: float
    diagnostics: dict
