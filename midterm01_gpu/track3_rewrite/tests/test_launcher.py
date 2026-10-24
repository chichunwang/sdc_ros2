"""Standalone launcher contracts against generated native I/O artifacts."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_launcher',ROOT/'track3_rewrite/run.py')
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
RUN=Path(os.environ['TRACK3_REWRITE_TEST_RUN']).resolve()

class LauncherContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=RUN);self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)
    def copied_output(self):
        out=self.path/'out';shutil.copytree(RUN/'native_synthetic_final_3',out)
        raw=self.path/'inputs';shutil.copytree(RUN/'generated_inputs_final',raw)
        frames=[(i,1_000_000_000+i*40_000_000,raw/f'{i}_ts_{1_000_000_000+i*40_000_000}.ply') for i in range(3)]
        return out,frames
    def test_real_full_dryrun_without_build_or_cuda(self):
        with patch.object(sys,'argv',['run.py','--full','--dry-run']),patch.object(launch,'refuse_running',side_effect=AssertionError('process execution')),patch('sys.stdout',new_callable=io.StringIO) as out:
            self.assertEqual(launch.main(),0)
        plan=json.loads(out.getvalue());self.assertEqual(plan['frames'],3142);self.assertFalse(plan['cuda_initialized']);self.assertFalse(plan['localization_executed'])
    def test_native_success_outputs_validate(self):
        out,frames=self.copied_output();status=launch.validate(out,frames,{})
        self.assertEqual(status['processed_frames'],3)
    def test_future_timestamp_tampering_is_rejected(self):
        out,frames=self.copied_output();p=out/'observations.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()];rows[1]['input_cutoff_ns']+=1;p.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
        with self.assertRaisesRegex(ValueError,'causality'):launch.validate(out,frames,{})
    def test_improper_rotation_is_rejected(self):
        out,frames=self.copied_output();p=out/'result.txt';rows=p.read_text().splitlines();values=rows[0].split();values[0]=str(float(values[0])*2);rows[0]=' '.join(values);p.write_text('\n'.join(rows)+'\n')
        with self.assertRaises(ValueError):launch.validate(out,frames,{})
    def test_changed_current_source_is_rejected(self):
        out,frames=self.copied_output();p=frames[1][2];p.write_bytes(p.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'SHA'):launch.validate(out,frames,{})
    def test_new_map_extractor_is_lossless_and_rejects_processed_cache(self):
        data=self.path/'data';track=data/'Track3';track.mkdir(parents=True)
        p=track/'map.pcd';p.write_text('FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA ascii\n1.25 -2.5 3.75 9\n4 5 6 10\n')
        cache=launch.extract_map(p,self.path/'cache');checked,_=launch.map_identity(cache,data);self.assertEqual(checked,cache)
        import numpy as np
        np.testing.assert_array_equal(np.load(cache),[[1.25,-2.5,3.75],[4,5,6]])
        meta=cache.parent/'map_xyz.json';record=json.loads(meta.read_text());record['voxel_applied']=True;launch.write_json(meta,record)
        with self.assertRaises(ValueError):launch.map_identity(cache,data)
    def test_native_refuses_existing_output_without_modification(self):
        out,frames=self.copied_output();before={p.name:launch.sha(p) for p in out.iterdir() if p.is_file()}
        container='/root/outputs/'+out.relative_to(ROOT/'outputs').as_posix()
        result=subprocess.run(['docker','exec','sdc_midterm01_gpu','/root/outputs/run/'+RUN.name+'/native_build/track3_rewrite','--out',container],capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertIn('refusing existing output',result.stderr)
        self.assertEqual(before,{p.name:launch.sha(p) for p in out.iterdir() if p.is_file()})
    def test_plot_and_status_write_failure_preserve_localization(self):
        with patch.object(launch,'logged',return_value=1):
            launch.analyze_full(self.path)
        status=json.loads((self.path/'analysis_status.json').read_text())
        self.assertEqual(status['status'],'failed');self.assertTrue(status['localization_success'])
        with patch.object(launch,'logged',return_value=1),patch.object(launch,'write_json',side_effect=OSError('injected disk error')),patch('sys.stderr',new_callable=io.StringIO):
            launch.analyze_full(self.path)
    def test_signal_pending_before_child_pid_is_forwarded(self):
        pidfile=self.path/'child.pid';attempts=[];previous=signal.getsignal(signal.SIGINT)
        child_code='import os,signal,sys,time; signal.signal(signal.SIGINT,lambda *_:sys.exit(7)); time.sleep(.2); open(sys.argv[1],"w").write(str(os.getpid())); print("ready",flush=True); time.sleep(5)'
        def forward():
            attempts.append(1);os.kill(int(pidfile.read_text()),signal.SIGINT)
        def send_when_installed():
            deadline=time.monotonic()+5
            while signal.getsignal(signal.SIGINT) is previous:
                if time.monotonic()>deadline:return
                time.sleep(.005)
            os.kill(os.getpid(),signal.SIGINT)
        thread=threading.Thread(target=send_when_installed);thread.start()
        try:
            code=launch.logged([sys.executable,'-c',child_code,str(pidfile)],self.path/'child.log',interrupt_command=forward)
        finally:
            thread.join(timeout=6)
        self.assertEqual(code,7);self.assertGreaterEqual(len(attempts),2)
        self.assertIs(signal.getsignal(signal.SIGINT),previous)

if __name__=='__main__':unittest.main()
