#!/usr/bin/env python3
"""Read-only live output viewer, including runs without a built-in publisher.

No CUDA import, registration, filtering, output pose rewrite or source change.
Only the newest written PLY is loaded for display; path keeps every emitted pose.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from track2.visualization import Track2RvizPublisher
from track2.map_io import xyz_chunks
from track3.io import load_ply, write_json


def complete_json_lines(stream):
    rows = []
    while True:
        offset = stream.tell(); line = stream.readline()
        if not line:
            break
        if not line.endswith('\n'):
            stream.seek(offset); break
        rows.append(json.loads(line))
    return rows


def frame_paths(run):
    with (run/'frame_manifest.csv').open() as f:
        return {int(r['idx']): Path(r['path']) for r in csv.DictReader(f) if r.get('sha256')}


def display_map(path, max_points=120000):
    # Display-only deterministic sampling of XYZ on CPU, never a registration map.
    from track2.map_io import pcd_header
    header, _ = pcd_header(path)
    step = max(1, (int(header['POINTS'][0])+max_points-1)//max_points)
    arrays = []; offset = 0
    for xyz in xyz_chunks(path):
        start = (-offset) % step
        arrays.append(xyz[start::step]); offset += len(xyz)
    return np.concatenate(arrays)[:max_points]


def watch(run, record):
    if run.resolve() == record.resolve():
        raise ValueError('Observer records must be separate from localization run')
    inputs = json.loads((run/'input_sources.json').read_text())
    map_path = Path(inputs['sources']['map.pcd']['path'])
    view = Track2RvizPublisher()
    start = time.perf_counter()
    count = 0; last = None
    write_json(record/'viewer_status.json', dict(status='starting', observed_run=str(run), read_only=True))
    try:
        view.publish_map(display_map(map_path), 0)
        with (run/'pose_trace.jsonl').open() as stream:
            while True:
                rows = complete_json_lines(stream)
                if rows:
                    # Build full emitted path without repeatedly loading old PLYs.
                    paths = frame_paths(run)
                    for row in rows[:-1]:
                        view.publish_frame(np.empty((0, 3)), np.asarray(row['final']), row['scale'], row['timestamp_ns'],
                                           accepted=row['map_accepted'], reason='emitted_pose')
                    last = rows[-1]
                    raw = load_ply(paths[last['idx']])
                    view.publish_frame(raw, np.asarray(last['final']), last['scale'], last['timestamp_ns'],
                                       accepted=last['map_accepted'], reason='read_only_live_outputs')
                    count += len(rows)
                    write_json(record/'viewer_status.json', dict(status='running', observed_run=str(run), read_only=True,
                        displayed_frames=count, latest_idx=last['idx'], elapsed_seconds=time.perf_counter()-start))
                view.spin()
                time.sleep(.25)
    except KeyboardInterrupt:
        pass
    finally:
        view.close()
        write_json(record/'viewer_status.json', dict(status='stopped', observed_run=str(run), read_only=True,
            displayed_frames=count, latest_idx=None if last is None else last['idx'], elapsed_seconds=time.perf_counter()-start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--record-dir', type=Path, required=True)
    args = parser.parse_args()
    watch(args.run_dir.resolve(), args.record_dir.resolve())
