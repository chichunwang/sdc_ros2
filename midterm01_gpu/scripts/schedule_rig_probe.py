#!/usr/bin/env python3
"""在已授權完整 run 的兩 Track 之間插入短測，finally 恢復原 wrapper。"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def start_ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]


def schedule(args):
    root=Path(__file__).resolve().parents[1]
    wrapper=Path(f'/proc/{args.wrapper}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    child=Path(f'/proc/{args.child}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    if 'run_three_camera_full_fast.sh' not in wrapper or 'run_rig_fast.py --track 1 ' not in child:
        raise ValueError('Expected active Track1 and its original fast wrapper')
    status=Path(f'/proc/{args.child}/status').read_text()
    if int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))!=args.wrapper:
        raise ValueError('Wrapper/child parent mismatch')
    token=start_ticks(args.wrapper)
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Scheduler interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    held=False
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        os.kill(args.wrapper,signal.SIGSTOP);held=True
        (args.output/'scheduling.json').write_text(json.dumps({'wrapper':args.wrapper,'child':args.child,
            'wrapper_start_ticks':token,'action':'Delay Track2 only; Track1 child continues normally',
            'status':'waiting_for_track1'},indent=2)+'\n')
        print('Track1 continues; its wrapper is held before Track2.',flush=True)
        deadline=time.monotonic()+900
        while Path(f'/proc/{args.child}').exists():
            state=Path(f'/proc/{args.child}/stat').read_text().rsplit(')',1)[1].split()[0]
            if state=='Z':break
            if time.monotonic()>deadline:raise TimeoutError('Track1 did not finish within 15 minutes')
            time.sleep(2)
        validation=args.run/'track1/localization.json'
        if not validation.exists():raise RuntimeError('Track1 did not complete localization; no probe started')
        deadline=time.monotonic()+300
        while not (args.output/'probe_ready').exists():
            if time.monotonic()>deadline:raise TimeoutError('Probe preparation exceeded 5 minutes')
            time.sleep(2)
        command=[str(root/'outputs/ml_runtime/venv/bin/python'),str(root/'scripts/probe_rig_memory.py'),
                 '--output',str(args.output),'--full-run',str(args.run)]
        with (args.output/'probe.log').open('w') as log:
            environment={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':args.allocator_config}
            subprocess.run(command,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        print('Memory probe completed.',flush=True)
    finally:
        if held and Path(f'/proc/{args.wrapper}').exists() and start_ticks(args.wrapper)==token:
            os.kill(args.wrapper,signal.SIGCONT)
            (args.output/'wrapper_resumed.json').write_text(json.dumps({'wrapper':args.wrapper,'resumed':True},indent=2)+'\n')
            print('Original full-run wrapper resumed.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wrapper',type=int,required=True);p.add_argument('--child',type=int,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allocator-config',default='expandable_segments:True',help='Match the production full-run wrapper; use expandable_segments:False to reproduce the initial native-allocator probe')
    schedule(p.parse_args())
