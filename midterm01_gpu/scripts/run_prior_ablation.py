#!/usr/bin/env python3
"""Run only preselected bounded windows; never a full Track or submission."""
from output_layout import loads_saved_json, saved_path

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def read(path):
    return loads_saved_json(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=saved_path, required=True)
    for track in (1,2):
        parser.add_argument(f'--map-cache-track{track}', type=saved_path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    experiment = args.experiment.resolve()
    experiment.relative_to((root/'outputs/run').resolve())
    reports = read(experiment/'prior_ablation.json')
    if (experiment/'ablation_runs.json').exists():
        raise ValueError('Ablation record exists; preserve the previous experiment')
    planned = [(r['track'], w) for r in reports for w in r['windows']]
    if not 1 <= len(planned) <= 12 or any(not 1 <= w['frames'] <= 48 for _,w in planned):
        raise ValueError('This entry supports at most twelve 48-frame windows, never full sequences')
    output = experiment/'short_runs'; output.mkdir(exist_ok=True)
    records = []
    for track, window in planned:
        candidate = experiment/f'track{track}_motion_prior'
        baseline = Path(read(candidate/'sensor_prior.json')['position_only_ablation']['original_prior_run'])
        pair = {'track':track, **window, 'runs':{}}
        for label, prior in [('heading_rts',baseline),('motion_rts',candidate)]:
            name = f'track{track}_{window["class"]}_{window["split"]}_{label}'
            command = [sys.executable,str(root/'scripts/benchmark.py'),'--track',str(track),
                '--frames',str(window['frames']),'--start-frame',str(window['start']),
                '--profile','sensor_guided','--sensor-fusion',label,'--geometry-mode','directional',
                '--height-mode','carry','--prior-run',str(prior),
                '--map-cache',str(getattr(args,f'map_cache_track{track}').resolve())]
            print(f'{name}: original idx {window["start"]}..{window["start"]+window["frames"]-1}',flush=True)
            (output/f'{name}.command.json').write_text(json.dumps(command,indent=2)+'\n')
            log_path = output/f'{name}.log'
            with log_path.open('x') as log:
                completed = subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            matches = [line[5:] for line in log_path.read_text().splitlines() if line.startswith('Run: ')]
            if not matches or completed.returncode:
                raise RuntimeError(f'Failed short run; inspect {log_path}')
            run = Path(matches[-1]); validation = read(run/'validation.json')
            if not validation['output_complete_and_finite'] or validation['submission_frame_set_complete']:
                raise ValueError('Expected complete short-window output, not a submission')
            (output/name).symlink_to(Path('../../')/run.name,target_is_directory=True)
            pair['runs'][label] = str(run)
        a,b = [Path(pair['runs'][k]) for k in ('heading_rts','motion_rts')]
        if (a/'source_input.json').read_bytes() != (b/'source_input.json').read_bytes():
            raise ValueError('Ablation source clouds differ')
        if (a/'binary_sha256.txt').read_bytes() != (b/'binary_sha256.txt').read_bytes():
            raise ValueError('Ablation binaries differ')
        with (a/'sensor_prior.csv').open() as f: pa=list(csv.DictReader(f))
        with (b/'sensor_prior.csv').open() as f: pb=list(csv.DictReader(f))
        if any(x[k]!=y[k] for x,y in zip(pa,pb) for k in ('timestamp_ns','z','qx','qy','qz','qw')):
            raise ValueError('Position-only ablation changed Z or attitude')
        params_a,params_b=read(a/'params.json'),read(b/'params.json')
        for params in (params_a,params_b):
            for key in ('save_path','sensor_prior_file','ply_path'):params.pop(key)
        if params_a!=params_b:raise ValueError('Non-prior solver parameters changed')
        records.append(pair)
        (experiment/'ablation_runs.json').write_text(json.dumps(records,indent=2)+'\n')
    print(f'Completed {len(records)} matched pairs; no full Track started.',flush=True)


if __name__=='__main__':
    main()
