"""Causal state transitions, independent of registration and file formats."""
from dataclasses import dataclass

import numpy as np

from .contracts import FrameResult, Similarity, TrackingState, checked_timestamp


@dataclass(frozen=True)
class Reference:
    estimate: Similarity
    timestamp_ns: int
    raw: np.ndarray


class TrackingMemory:
    def __init__(self, initial, *, lost_after_seconds=2.):
        self.published = initial
        self.last_output_ns = None
        self.reference = None
        self.last_map = None
        self.state = TrackingState.INITIALIZING
        self.lost_after_seconds = lost_after_seconds

    def check_time(self, timestamp_ns):
        ns = checked_timestamp(timestamp_ns)
        if self.last_output_ns is not None and ns <= self.last_output_ns:
            raise ValueError('Cannot revisit a published frame')
        return ns

    def commit(self, raw, timestamp_ns, *, map_estimate=None, pair_estimate=None, diagnostics=None):
        ns = self.check_time(timestamp_ns)
        map_ok, pair_ok = map_estimate is not None, pair_estimate is not None
        if self.last_map is None and not map_ok:
            raise RuntimeError('Current-frame bootstrap has no map support')
        estimate = map_estimate or pair_estimate or self.published
        if map_ok or pair_ok:
            cloud = np.array(raw, dtype=np.float64, copy=True)
            cloud.setflags(write=False)
            self.reference = Reference(estimate, ns, cloud)
        # An unsupported cloud NEVER replaces the last supported reference.
        if map_ok:
            self.last_map = Reference(estimate, ns, self.reference.raw)
        unsupported = (ns-self.last_map.timestamp_ns)/1e9
        self.state = (TrackingState.TRACKING if map_ok else
                      TrackingState.LOST if unsupported > self.lost_after_seconds else
                      TrackingState.REACQUIRING)
        self.published, self.last_output_ns = estimate, ns
        return FrameResult(estimate, ns, self.state, map_ok, pair_ok,
                           'map' if map_ok else 'pair' if pair_ok else 'held',
                           unsupported, diagnostics or {})
