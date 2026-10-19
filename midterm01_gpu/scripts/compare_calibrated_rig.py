#!/usr/bin/env python3
"""三相機小片段公平消融；主機呼叫既有 GPU benchmark，不啟動完整 Track。"""
from output_layout import saved_path
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reconstruction', type=saved_path, required=True)
    p.add_argument('--prior-track1', type=saved_path, required=True)
    p.add_argument('--prior-track2', type=saved_path, required=True)
    p.add_argument('--map-cache-track1', type=saved_path, required=True)
    p.add_argument('--map-cache-track2', type=saved_path, required=True)
    p.add_argument('--output', type=saved_path, required=True)
    p.add_argument('--labels', default='front,single,predicted,rig_front,rig_static,rig,rig_uniform_matched,balanced')
    p.add_argument('--label-prefix', default='', help='Distinguish e.g. a lower point-budget control in the report')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.resolve().relative_to((root / 'outputs').resolve())
    if not (args.reconstruction / 'completion.json').is_file():
        p.error('Reconstruction must complete first; do not compare a failed/partial run')
    args.output.mkdir(parents=True, exist_ok=False)
    windows = [(1, 'straight', 1505), (1, 'turn', 1823), (2, 'straight', 1661), (2, 'turn', 1397)]
    labels = args.labels.split(',')
    if len(labels) != len(set(labels)) or any(not label.replace('_', '').isalnum() for label in labels):
        p.error('Labels must be unique simple variant names')
    records = []
    for track, kind, start in windows:
        for label in labels:
            command = [sys.executable, str(root / 'scripts/benchmark.py'), '--track', str(track), '--frames', '2',
                       '--start-frame', str(start), '--profile', 'sensor_guided', '--sensor-fusion', 'heading_rts',
                       '--geometry-mode', 'directional', '--prior-run', str(getattr(args, f'prior_track{track}').resolve()),
                       '--map-cache', str(getattr(args, f'map_cache_track{track}').resolve()), '--ply-scale', '3' if label == 'front' else '1']
            if label != 'front':
                command += ['--cloud-source', 'reconstruction', '--source-report', str((args.reconstruction / f'{label}_track{track}/report.json').resolve())]
            logfile = args.output / f'track{track}_{kind}_{label}.log'
            with logfile.open('x') as log:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            output = logfile.read_text()
            run = next((line[5:] for line in output.splitlines() if line.startswith('Run: ')), None)
            record = {'track': track, 'kind': kind, 'label': args.label_prefix + label, 'start_frame': start, 'frames': 2,
                      'returncode': code, 'run': run, 'command': command, 'log': str(logfile.resolve())}
            records.append(record)
            (args.output / 'comparison_runs.json').write_text(json.dumps(records, indent=2) + '\n')
            print(json.dumps(record), flush=True)
            if code:
                raise SystemExit(f'Comparison failed; inspect {logfile}')
    (args.output / 'scope.json').write_text(json.dumps({
        'frames': 2 * len(windows) * len(labels), 'windows': windows, 'same_binary_map_prior_parameters': True,
        'selection': 'Two adjacent frames at sensor-selected straight/turn centers from the previous experiment; no score-based selection',
        'isolation': {'rig_front_vs_rig': 'Same inferred front depth, add side points',
                      'rig_uniform_matched_vs_balanced': 'Same reconstruction, equal exported point count; change sampling only',
                      'rig_static_vs_rig': 'Same motion-conditioned inferred depths, change merge transforms only'},
        'scope': 'No full Track, ground truth, Kaggle or formal accuracy claim'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
