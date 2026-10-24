"""Fast v2 source reuse, candidate ordering/cleanup and bounded CUDA streams."""
from concurrent.futures import Future,ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
import sys,threading,time,unittest
from types import SimpleNamespace
from unittest.mock import patch,Mock
import numpy as np

SCRIPTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCRIPTS))
from run_track3_map_only_fast import main as fast_entrypoint
from track3_registration_fast import join_candidates,FastGeometry
from run_track3_map_only import arguments

class RuntimeChecks(unittest.TestCase):
    def test_retains_candidate_order(self):
        a,b,c=Future(),Future(),Future()
        c.set_result(3);a.set_result(1);b.set_result(2)
        self.assertEqual(join_candidates([a,b,c]),[1,2,3])

    def test_worker_error_waits_for_other_workers(self):
        finished=threading.Event()
        def fail():raise RuntimeError('controlled worker failure')
        def finish():time.sleep(.03);finished.set();return 2
        with ThreadPoolExecutor(2) as executor:
            a,b=executor.submit(fail),executor.submit(finish)
            with self.assertRaisesRegex(RuntimeError,'controlled worker failure'):
                join_candidates([a,b])
            self.assertTrue(finished.is_set())
            self.assertTrue(a.done() and b.done())

    def test_interrupt_waits_before_propagation(self):
        finished=threading.Event()
        def finish():time.sleep(.03);finished.set();return 1
        import track3_registration_fast as module
        original=module.wait
        calls=[]
        def interrupted(futures):
            calls.append(1)
            if len(calls)==1:raise KeyboardInterrupt('controlled interrupt')
            return original(futures)
        with ThreadPoolExecutor(1) as executor,patch.object(module,'wait',interrupted):
            pending=executor.submit(finish)
            with self.assertRaises(KeyboardInterrupt):join_candidates([pending])
            self.assertTrue(finished.is_set() and pending.done())
            self.assertEqual(len(calls),2)

    def test_immutable_source_is_reused_and_input_change_invalidates(self):
        geometry=FastGeometry('cpu_reference')
        raw=np.arange(300,dtype=float).reshape(-1,3)/100
        mask=np.arange(100)%2==0
        weights=np.ones(100)
        a=geometry.prepare(raw,mask,weights)
        b=geometry.prepare(raw,mask.copy(),weights)
        self.assertIs(a,b)
        c=geometry.prepare(raw,~mask,weights)
        self.assertIsNot(a,c)
        d=geometry.prepare(raw,~mask,weights.copy())
        self.assertIsNot(c,d)
        np.testing.assert_array_equal(d['source'],raw[~mask])

    def test_one_persistent_stream_per_worker_over_repeated_frames(self):
        from track3_registration_fast import parallel_map_candidates
        from track3_registration import RegistrationConfig
        import torch
        barrier=threading.Barrier(3)
        created=[]
        class Stream:
            def __init__(self):
                self.cuda_stream=len(created)+1;created.append(self)
            def synchronize(self):pass
        geometry=SimpleNamespace(backend='cuda',parallel_candidates=True,
            _local=threading.local(),sync=Mock())
        def fit(raw,prior,seed,**kwargs):
            barrier.wait(timeout=3)
            return dict(pose=prior.tolist(),scale=seed)
        scene=SimpleNamespace(geometry=geometry,fit=fit)
        raw=np.arange(36,dtype=float).reshape(12,3)/10
        with ThreadPoolExecutor(3) as executor,patch.object(torch.cuda,'Stream',Stream), \
             patch.object(torch.cuda,'stream',side_effect=lambda stream:nullcontext()), \
             patch('track3_visibility_fast.ray_depth_diagnostic',return_value={}):
            geometry.executor=executor
            for _ in range(20):
                results=parallel_map_candidates(scene,raw,np.eye(4),3.,RegistrationConfig(),np.ones(12))
                self.assertEqual(len({r['candidate_execution']['stream'] for r in results}),3)
        self.assertEqual(len(created),3)

    def test_cli_defaults_and_invalid_budgets_preserved(self):
        self.assertEqual(arguments([]).vram_budget_gib,4.)
        for value in ('0','nan','-1'):
            with self.assertRaises(SystemExit):arguments(['--vram-budget-gib',value])


if __name__=='__main__':unittest.main()
