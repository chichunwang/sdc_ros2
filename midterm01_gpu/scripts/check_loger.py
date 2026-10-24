#!/usr/bin/env python3
"""Synthetic LoGeR environment check. No dataset image is loaded."""
import argparse
from pathlib import Path
import time
import traceback
from track3.io import write_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--record-dir',type=Path,required=True);a=p.parse_args()
    started=time.perf_counter();a.record_dir.mkdir(parents=True,exist_ok=True)
    info=dict(test_type='synthetic 2-frame inference; no Track3 data or localization')
    try:
        import torch
        from track3.loger_source import load_model,VENDOR
        info.update(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),architectures=torch.cuda.get_arch_list(),checkpoint_sha256=sha256(VENDOR/'latest.pt'))
        print('Loading checkpoint with strict state dict',flush=True)
        model=load_model()
        print('Forward on two generated RGB frames',flush=True)
        torch.manual_seed(7)
        images=torch.rand(1,2,3,280,504,device='cuda')
        for count in (1,2):
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                out=model(images[:,:count],window_size=count,overlap_size=0,se3=False,sim3=False)
            for key in ['local_points','conf']:
                if not torch.isfinite(out[key]).all(): raise ValueError(f'Nonfinite {key} with {count} frames')
        torch.cuda.synchronize()
        shapes={k:list(v.shape) for k,v in out.items() if isinstance(v,torch.Tensor)}
        for k in ['local_points','conf']:
            if not torch.isfinite(out[k]).all():raise ValueError(f'Nonfinite {k}')
        assert out['local_points'].shape[:2]==(1,2)
        info.update(status='success',output_shapes=shapes,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    except Exception as e:
        info.update(status='failed',failure=f'{type(e).__name__}: {e}')
        (a.record_dir/'loger_failure.log').write_text(traceback.format_exc());traceback.print_exc()
    info['elapsed_seconds']=time.perf_counter()-started
    write_json(a.record_dir/'loger_check.json',info);print(info,flush=True)
    return 0 if info['status']=='success' else 1

if __name__=='__main__':raise SystemExit(main())
