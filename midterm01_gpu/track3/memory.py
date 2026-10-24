"""Sample CuPy pool memory without importing or allocating GPU arrays."""

import json
import math
import time


class GpuMemoryMonitor:
    """Bound the allocator pool and optionally release its unused cached blocks.

    Measurements are checkpoint samples, not an absolute allocation peak. Pool
    statistics exclude CUDA context, libraries and allocations outside this pool.
    """

    DEFAULTS = {
        "pool_limit_mib": 4096,
        "trim_above_mib": 3072,
        "reserve_mib": 1024,
        "min_trim_mib": 256,
    }

    def __init__(self, cp, log_path=None, settings=None):
        values = dict(self.DEFAULTS)
        if settings is not None:
            unknown = set(settings) - set(values)
            if unknown:
                raise ValueError("Unknown GPU memory settings: " + ", ".join(sorted(unknown)))
            values.update(settings)
        for name, value in values.items():
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(name + " must be a finite nonnegative MiB value")
        if values["pool_limit_mib"] <= 0:
            raise ValueError("pool_limit_mib must be greater than zero")
        if values["trim_above_mib"] > values["pool_limit_mib"]:
            raise ValueError("trim_above_mib must not exceed pool_limit_mib")
        if values["min_trim_mib"] > values["trim_above_mib"]:
            raise ValueError("min_trim_mib must not exceed trim_above_mib")
        self.settings = values
        self._bytes = {key: int(value * 1024 * 1024) for key, value in values.items()}
        self._cp = cp
        self._pool = cp.get_default_memory_pool()
        free, _ = cp.cuda.runtime.memGetInfo()
        limit = min(self._bytes["pool_limit_mib"],
                    int(free) + int(self._pool.total_bytes()) - self._bytes["reserve_mib"])
        existing_limit=int(self._pool.get_limit())
        if existing_limit>0:limit=min(limit,existing_limit)
        used = int(self._pool.used_bytes())
        if limit <= 0 or limit < used:
            raise MemoryError(
                "GPU pool budget cannot preserve the requested device reserve: "
                f"limit={limit} bytes, in_use={used} bytes; no fallback applied")
        self._log = None
        if log_path is not None:
            self._log = open(log_path, "x", encoding="utf-8")
        try:
            self._pool.set_limit(size=limit)
        except BaseException:
            self.close()
            raise
        self._started = time.monotonic()
        self._count = 0
        self._max_used = None
        self._max_reserved = None
        self._max_cached = None
        self._min_free = None

    def _stats(self):
        free, total = self._cp.cuda.runtime.memGetInfo()
        used = int(self._pool.used_bytes())
        reserved = int(self._pool.total_bytes())
        return {
            "pool_used_bytes": used,
            "pool_reserved_bytes": reserved,
            "pool_cached_bytes": max(0, reserved - used),
            "pool_limit_bytes": int(self._pool.get_limit()),
            "device_free_bytes": int(free),
            "device_total_bytes": int(total),
        }

    def _sample(self, stats):
        self._max_used = max(self._max_used or 0, stats["pool_used_bytes"])
        self._max_reserved = max(self._max_reserved or 0, stats["pool_reserved_bytes"])
        self._max_cached = max(self._max_cached or 0, stats["pool_cached_bytes"])
        free = stats["device_free_bytes"]
        self._min_free = free if self._min_free is None else min(self._min_free, free)

    def checkpoint(self, stage, idx=None, trim=False):
        """Sample memory; trimming releases cached blocks only, never in-use data."""
        stats = self._stats()
        self._sample(stats)
        before_reserved = stats["pool_reserved_bytes"]
        attempted = bool(
            trim and stats["pool_cached_bytes"] >= self._bytes["min_trim_mib"]
            and (before_reserved >= self._bytes["trim_above_mib"]
                 or stats["device_free_bytes"] < self._bytes["reserve_mib"]))
        if attempted:
            self._pool.free_all_blocks()
            self._cp.get_default_pinned_memory_pool().free_all_blocks()
            stats = self._stats()
            self._sample(stats)
        released = max(0, before_reserved - stats["pool_reserved_bytes"])
        record = {
            "elapsed_seconds": time.monotonic() - self._started,
            "stage": stage,
            "idx": idx,
            **stats,
            "trim_attempted": attempted,
            "trimmed": released > 0,
            "released_cached_bytes": released,
        }
        if self._log is not None:
            self._log.write(json.dumps(record, allow_nan=False) + "\n")
            self._log.flush()
        self._count += 1
        return record

    def summary(self):
        return {
            "sample_count": self._count,
            "sampled_max_pool_used_bytes": self._max_used,
            "sampled_max_pool_reserved_bytes": self._max_reserved,
            "sampled_max_pool_cached_bytes": self._max_cached,
            "sampled_min_device_free_bytes": self._min_free,
            "measurement_scope": (
                "Checkpoint samples, including pre/post trim reads; not an absolute "
                "allocation peak or all CUDA process memory. Pool limit excludes "
                "CUDA context, libraries and allocations outside this pool."
            ),
        }

    def close(self):
        if self._log is not None:
            self._log.close()
            self._log = None
