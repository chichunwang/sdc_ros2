#!/usr/bin/env python3
"""Reuse preselected small windows to test the optional horizontal joint pipeline."""
from output_layout import loads_saved_json, saved_path, scored_reference_run
import argparse
import json
from pathlib import Path
import subprocess
import sys

from prior_ablation import dump


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=saved_path,required=True)
    parser.add_argument('--control-experiment',type=saved_path,required=True)
    for track in (1,2):
        parser.add_argument(f'--coarse-track{track}',type=saved_path,required=True)
        parser.add_argument(f'--fine-track{track}',type=saved_path,required=True)
    args=parser.parse_args(); root=Path(__file__).resolve().parents[1]
    output=args.experiment.resolve()/'short_runs'; output.mkdir()
    old=loads_saved_json((args.control_experiment/'ablation_runs.json').read_text())
    if not 1<=len(old)<=12 or any(not 1<=r['frames']<=48 for r in old):
        raise ValueError('Only preselected small windows; never full sequences')
    records=[]
    for previous in old:
        track=previous['track']; label=f'track{track}_{previous["class"]}_{previous["split"]}'
        cmd=[sys.executable,str(root/'scripts/benchmark.py'),'--track',str(track),
            '--frames',str(previous['frames']),'--start-frame',str(previous['start']),
            '--profile','sensor_guided','--sensor-fusion','motion_rts','--geometry-mode','directional',
            '--height-mode','carry','--horizontal-refinement','surfel_joint',
            '--prior-run',str(scored_reference_run(root,track)),
            '--map-cache',str(getattr(args,f'coarse_track{track}').resolve()),
            '--horizontal-map-cache',str(getattr(args,f'fine_track{track}').resolve())]
        dump(output/f'{label}.command.json',cmd)
        print(label,flush=True)
        log=output/f'{label}.log'
        with log.open('x') as stream: result=subprocess.run(cmd,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
        found=[line[5:] for line in log.read_text().splitlines() if line.startswith('Run: ')]
        if result.returncode or not found: raise RuntimeError(f'Short benchmark failed: {log}')
        run=Path(found[-1]); validation=loads_saved_json((run/'validation.json').read_text())
        if not validation['output_complete_and_finite'] or validation['submission_frame_set_complete']:
            raise ValueError('Invalid short-run output')
        control=Path(previous['runs']['directional'])
        same={f: (run/'base'/f).read_bytes()==(control/f).read_bytes() for f in ('result.txt','diagnostic.csv')}
        same.update({f:(run/f).read_bytes()==(control/f).read_bytes() for f in ('geometry.csv','height.csv','sensor_prior.csv')})
        if not all(same.values()): raise ValueError(f'Base algorithm changed: {same}')
        report=loads_saved_json((run/'horizontal/report.json').read_text())
        record={k:previous[k] for k in ('track','class','split','start','frames')}
        record.update(run=str(run),control=str(control),base_control_byte_identical=same,
                      map_factors=report['accepted_map_factors'],applied=report['applied'],reason=report['reason'])
        records.append(record); dump(args.experiment/'ablation_runs.json',records)
    dump(args.experiment/'short_summary.json',{'runs':records,'poses':sum(r['frames'] for r in records),
        'control_poses_reused':sum(r['frames'] for r in records),'full_localization_started':False,
        'accuracy_verified':False})


if __name__=='__main__': main()
