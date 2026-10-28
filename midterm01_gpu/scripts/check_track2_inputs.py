#!/usr/bin/env python3
"""Read-only Track2 time/source audit; never constructs a filter or registration."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track3.io import frames, sha256, write_json
from track3.time_alignment import CameraClock
from track2.sensors import SensorStream
from track2.map_io import pcd_header
from track2 import parameters as p


def inspect(run):
    start = time.perf_counter()
    data = Path(p.DATA_DIR)
    rows = frames(data/'raw_ply', expected=p.EXPECTED_FRAMES)
    header, offset = pcd_header(data/'map.pcd')
    clock = CameraClock(data/'Track2.db3')
    sensors = SensorStream(data/'Track2.db3', p.GNSS_EXPECTED_FRAME)
    manifest = []
    event_count = 0
    try:
        for row in rows:
            arrival = row['timestamp_ns']; camera = clock.for_frame(arrival)
            events = sensors.take(arrival, camera)
            if any(e['arrival_ns'] > arrival or e['timestamp_ns'] > camera for e in events):
                raise AssertionError('Noncausal sensor delivery')
            event_count += len(events)
            manifest.append(dict(idx=row['original_idx'], timestamp_ns=arrival, camera_timestamp_ns=camera,
                                 path=str(row['path']), sha256=sha256(row['path'])))
        info = dict(status='passed', frames=len(rows), sensors=sensors.summary(), delivered_events=event_count,
                    pcd_header=header, pcd_payload_offset=offset,
                    input_sources={n: dict(path=str(data/n), sha256=sha256(data/n))
                                   for n in ('map.pcd', 'Track2.db3', 'init_guess.txt')},
                    kind='read_only_input_contract_check_no_localization',
                    localization_executed=False, elapsed_seconds=time.perf_counter()-start)
        write_json(run/'data_contract.json', info)
        write_json(run/'input_frame_manifest.json', manifest)
        print(json.dumps({k: v for k, v in info.items() if k not in ('input_sources', 'pcd_header')}, indent=2))
    finally:
        clock.close(); sensors.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--record-dir', type=Path, required=True)
    args = parser.parse_args()
    if (args.record_dir/'data_contract.json').exists():
        raise FileExistsError('Existing audit preserved')
    inspect(args.record_dir)
