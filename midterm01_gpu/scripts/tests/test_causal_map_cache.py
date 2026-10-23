"""Small synthetic ASCII PCD tests, restricted to the assigned test run."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'causal_map_cache.py'
spec = importlib.util.spec_from_file_location('causal_map_cache', SCRIPT)
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)
from causal_test_support import fixture_run
TEST_RUN = fixture_run()
HEADER = '''# synthetic PCD
VERSION .7
FIELDS intensity z x y
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH 2
HEIGHT 1
POINTS 2
DATA ascii
'''
PAYLOAD = '8 3.123456789 1.123456789 2.123456789\n9 6 4 5\n'


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='map_cache_', dir=TEST_RUN)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / 'synthetic.pcd'
        self.source.write_text(HEADER + PAYLOAD)
        self.output = self.directory / 'cache'

    def test_exact_xyz_and_metadata(self):
        original = self.source.read_bytes()
        # Exercise multiple streaming batches without constructing a large cloud.
        with patch.object(cache, 'BATCH_ROWS', 1):
            path = cache.cache_xyz_map(self.source, self.output)
        data = np.load(path, allow_pickle=False)
        np.testing.assert_array_equal(data, np.asarray([[1.123456789, 2.123456789, 3.123456789],
                                                      [4, 5, 6]], dtype=np.float32))
        self.assertEqual(data.dtype, np.dtype('float32'))
        self.assertEqual(data.shape, (2, 3))
        self.assertEqual(self.source.read_bytes(), original)
        self.assertFalse((self.output / 'map_xyz.partial.npy').exists())
        metadata = json.loads((self.output / 'map_xyz.json').read_text())
        self.assertEqual(metadata['points'], 2)
        self.assertEqual(metadata['source_sha256'], cache._sha(self.source))
        self.assertEqual(metadata['cache_sha256'], cache._sha(path))
        self.assertFalse(metadata['voxel_applied'])
        self.assertFalse(metadata['pose_processing_applied'])

    def test_invalid_payload_preserves_partial(self):
        for index, payload in enumerate(('8 NaN 1 2\n9 6 4 5\n',
                                         '8 3 1 2\n', PAYLOAD + '9 9 9 9\n',
                                         '8 3 1\n9 6 4 5\n', '8 3 word 2\n9 6 4 5\n')):
            with self.subTest(payload=payload):
                self.source.write_text(HEADER + payload)
                output = self.directory / f'bad_{index}'
                with self.assertRaises(ValueError):
                    cache.cache_xyz_map(self.source, output)
                self.assertTrue((output / 'map_xyz.partial.npy').exists())
                self.assertFalse((output / 'map_xyz.npy').exists())
                self.assertFalse((output / 'map_xyz.json').exists())

    def test_header_rejections(self):
        for index, header in enumerate((HEADER.replace('DATA ascii', 'DATA binary'),
                                        HEADER.replace('POINTS 2', 'POINTS 3'),
                                        HEADER.replace('WIDTH 2', 'WIDTH 0'),
                                        HEADER.replace('SIZE 4 4 4 4', 'SIZE 4 4 8 4'),
                                        HEADER.replace('COUNT 1 1 1 1', 'COUNT 1 1 2 1'))):
            with self.subTest(header=header):
                self.source.write_text(header + PAYLOAD)
                with self.assertRaises(ValueError):
                    cache.cache_xyz_map(self.source, self.directory / f'header_{index}')

    def test_no_overwrite_any_artifact(self):
        for name in ('map_xyz.npy', 'map_xyz.partial.npy', 'map_xyz.json'):
            output = self.directory / name.replace('.', '_')
            output.mkdir()
            artifact = output / name
            artifact.write_bytes(b'preserve')
            with self.assertRaises(FileExistsError):
                cache.cache_xyz_map(self.source, output)
            self.assertEqual(artifact.read_bytes(), b'preserve')

    def test_source_change_rejected(self):
        original_sha = cache._sha
        calls = []
        def changed(path):
            calls.append(path)
            return original_sha(path) if len(calls) == 1 else 'changed'
        with patch.object(cache, '_sha', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                cache.cache_xyz_map(self.source, self.output)
        self.assertTrue((self.output / 'map_xyz.partial.npy').exists())
        self.assertFalse((self.output / 'map_xyz.json').exists())

    def test_scope(self):
        with self.assertRaises(ValueError):
            cache.cache_xyz_map(self.source, SCRIPT.parent / 'forbidden')


if __name__ == '__main__':
    unittest.main()
