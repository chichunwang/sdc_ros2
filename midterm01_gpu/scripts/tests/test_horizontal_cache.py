"""Do not certify a failed, truncated or mismatched fine localization map."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from accuracy.horizontal_cache import prepare_fine_map


class FineCacheTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[2]
        self.temp=tempfile.TemporaryDirectory(dir=root/'outputs')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.data=self.root/'data'
        source=self.data/'Track1/map.pcd'; source.parent.mkdir(parents=True); source.write_bytes(b'raw map identity')
        self.output=self.root/'fine'; self.bounds={'min':[-1.,-2.,-3.],'max':[1.,2.,3.]}

    def test_success_reuse_and_identity_mismatch(self):
        def builder(*args,**kwargs):
            p=self.output/'fine_track1.partial.pcd'
            header=b'FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA binary\n'
            p.write_bytes(header+struct.pack('<fff',1,2,3))
            return subprocess.CompletedProcess(args[0],0)
        with patch('accuracy.horizontal_cache.subprocess.run',side_effect=builder) as build:
            cache=prepare_fine_map(self.output,self.data,1,'test',self.bounds)
            self.assertEqual(build.call_count,1)
            self.assertEqual(cache,prepare_fine_map(self.output,self.data,1,'test',self.bounds))
            self.assertEqual(build.call_count,1)
            with self.assertRaisesRegex(ValueError,'identity'):
                prepare_fine_map(self.output,self.data,1,'test',{'min':[-2.,-2.,-3.],'max':[1.,2.,3.]})
            cache.write_bytes(cache.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError,'identity'):
                prepare_fine_map(self.output,self.data,1,'test',self.bounds)

    def test_outside_output_is_rejected_before_mkdir(self):
        with tempfile.TemporaryDirectory() as outside:
            requested=Path(outside)/'must-not-create'
            with self.assertRaises(ValueError):
                prepare_fine_map(requested,self.data,1,'test',self.bounds)
            self.assertFalse(requested.exists())

    def test_failed_build_has_no_certified_output(self):
        def failed(*args,**kwargs):
            (self.output/'fine_track1.partial.pcd').write_bytes(b'incomplete')
            return subprocess.CompletedProcess(args[0],1)
        with patch('accuracy.horizontal_cache.subprocess.run',side_effect=failed):
            with self.assertRaisesRegex(RuntimeError,'construction failed'):
                prepare_fine_map(self.output,self.data,1,'test',self.bounds)
        self.assertFalse((self.output/'fine_track1.pcd').exists())
        self.assertFalse((self.output/'fine_track1.pcd.json').exists())
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            prepare_fine_map(self.output,self.data,1,'test',self.bounds)


if __name__=='__main__': unittest.main()
