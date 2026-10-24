"""CPU-only tests: no CuPy import or CUDA runtime calls."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track3.memory import GpuMemoryMonitor

MIB = 1024 * 1024


class FakePool:
    def __init__(self, used=512, reserved=1024):
        self.used = used * MIB
        self.reserved = reserved * MIB
        self.limit = 0
        self.frees = 0
        self.limits = []

    def used_bytes(self):
        return self.used

    def total_bytes(self):
        return self.reserved

    def get_limit(self):
        return self.limit

    def set_limit(self, *, size):
        self.limit = size
        self.limits.append(size)

    def free_all_blocks(self):
        self.frees += 1
        self.reserved = self.used


class FakeCp:
    def __init__(self, free=4096, used=512, reserved=1024):
        self.pool = FakePool(used, reserved)
        self.pinned = FakePool(0, 0)
        self.free = free * MIB
        self.total = 8192 * MIB
        self.cuda = SimpleNamespace(runtime=SimpleNamespace(memGetInfo=self.mem_info))

    def mem_info(self):
        return self.free, self.total

    def get_default_memory_pool(self):
        return self.pool

    def get_default_pinned_memory_pool(self):
        return self.pinned


class GpuMemoryMonitorTests(unittest.TestCase):
    def test_limit_includes_existing_reserved_memory(self):
        cp = FakeCp(free=2048, reserved=2048)
        monitor = GpuMemoryMonitor(cp)
        self.assertEqual(cp.pool.limit, 3072 * MIB)
        self.assertEqual(monitor.summary()["sample_count"], 0)
        self.assertIsNone(monitor.summary()["sampled_max_pool_used_bytes"])

    def test_configured_limit_caps_available_budget(self):
        cp = FakeCp(free=7000)
        GpuMemoryMonitor(cp)
        self.assertEqual(cp.pool.limit, 4096 * MIB)

    def test_existing_stricter_pool_limit_is_respected(self):
        cp=FakeCp(free=7000);cp.pool.limit=2048*MIB
        GpuMemoryMonitor(cp)
        self.assertEqual(cp.pool.limit,2048*MIB)

    def test_trim_releases_only_cache_and_pinned_blocks(self):
        cp = FakeCp(used=1024, reserved=3500)
        monitor = GpuMemoryMonitor(cp)
        record = monitor.checkpoint("frame_end", idx=2, trim=True)
        self.assertTrue(record["trimmed"])
        self.assertTrue(record["trim_attempted"])
        self.assertEqual(record["released_cached_bytes"], 2476 * MIB)
        self.assertEqual(record["pool_used_bytes"], 1024 * MIB)
        self.assertEqual(record["pool_reserved_bytes"], 1024 * MIB)
        self.assertEqual(record["pool_cached_bytes"], 0)
        self.assertEqual(cp.pinned.frees, 1)
        self.assertEqual(record["idx"], 2)
        for key, value in record.items():
            if key.endswith("_bytes"):
                self.assertIsInstance(value, int)

    def test_low_device_free_triggers_below_reserved_threshold(self):
        cp = FakeCp(free=2048, used=512, reserved=1024)
        monitor = GpuMemoryMonitor(cp)
        cp.free = 800 * MIB
        record = monitor.checkpoint("pressure", trim=True)
        self.assertTrue(record["trimmed"])
        self.assertEqual(cp.pool.used, 512 * MIB)

    def test_no_trim_without_request_or_pressure_or_enough_cache(self):
        cases = [(3500, 512, 4096, False), (1024, 512, 4096, True),
                 (3500, 3400, 4096, True)]
        for reserved, used, free, trim in cases:
            with self.subTest(reserved=reserved, used=used, trim=trim):
                cp = FakeCp(free=free, used=used, reserved=reserved)
                record = GpuMemoryMonitor(cp).checkpoint("end", trim=trim)
                self.assertFalse(record["trimmed"])
                self.assertFalse(record["trim_attempted"])
                self.assertEqual(cp.pool.frees, 0)
                self.assertEqual(cp.pinned.frees, 0)

    def test_unreleasable_split_cache_is_not_reported_as_freed(self):
        cp = FakeCp(reserved=3500)
        cp.pool.free_all_blocks = lambda: None
        record = GpuMemoryMonitor(cp).checkpoint("end", trim=True)
        self.assertTrue(record["trim_attempted"])
        self.assertFalse(record["trimmed"])
        self.assertEqual(record["released_cached_bytes"], 0)

    def test_summary_preserves_pretrim_sample_and_counts_checkpoints(self):
        cp = FakeCp(reserved=3500)
        monitor = GpuMemoryMonitor(cp)
        monitor.checkpoint("first", trim=True)
        cp.free = 2000 * MIB
        monitor.checkpoint("second")
        result = monitor.summary()
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["sampled_max_pool_reserved_bytes"], 3500 * MIB)
        self.assertEqual(result["sampled_max_pool_used_bytes"], 512 * MIB)
        self.assertEqual(result["sampled_min_device_free_bytes"], 2000 * MIB)
        self.assertIn("not an absolute", result["measurement_scope"])

    def test_exclusive_jsonl_and_flush_and_idempotent_close(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.jsonl"
            monitor = GpuMemoryMonitor(FakeCp(), path)
            record = monitor.checkpoint("start")
            self.assertEqual(json.loads(path.read_text()), record)
            another_cp = FakeCp()
            with self.assertRaises(FileExistsError):
                GpuMemoryMonitor(another_cp, path)
            self.assertEqual(another_cp.pool.limits, [])
            self.assertEqual(json.loads(path.read_text()), record)
            monitor.close()
            monitor.close()

    def test_impossible_budget_raises_without_changing_limit(self):
        for free, used, reserved in [(0, 512, 512), (600, 800, 1000)]:
            cp = FakeCp(free=free, used=used, reserved=reserved)
            with self.assertRaisesRegex(MemoryError, "in_use=.*no fallback"):
                GpuMemoryMonitor(cp)
            self.assertEqual(cp.pool.limits, [])

    def test_invalid_settings(self):
        for settings in [{"pool_limit_mib": 0}, {"reserve_mib": -1},
                         {"min_trim_mib": float("nan")}, {"reserve_mib": True},
                         {"pool_limit_mib": 2000}, {"min_trim_mib": 4000},
                         {"unknown": 1}]:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    GpuMemoryMonitor(FakeCp(), settings=settings)


if __name__ == "__main__":
    unittest.main(verbosity=2)
