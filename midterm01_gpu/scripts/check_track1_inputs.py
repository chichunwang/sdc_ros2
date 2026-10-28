#!/usr/bin/env python3
"""唯讀 Track1 clocks/感測器/相機契約；不載入 PLY、不構造定位器或CUDA。"""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from track3.io import frames, sha256, write_json
from track3.geometry import valid_pose
from track3.time_alignment import CameraClock
from track1.camera import FrontMask
from track1.sensors import SensorStream
from track1.map_io import pcd_header
from track1 import parameters as p


def inspect(run):
    if (run/'data_contract.json').exists():
        raise FileExistsError('Existing audit preserved')
    start = time.perf_counter(); data = Path(p.DATA_DIR)
    rows = frames(data/'raw_ply', expected=p.EXPECTED_FRAMES)
    initial = np.loadtxt(data/'init_guess.txt', delimiter=',')
    if not valid_pose(initial):
        raise ValueError('Invalid Track1 initial pose')
    header, offset = pcd_header(data/'map.pcd')
    if not all(c in header['FIELDS'] for c in ('x', 'y', 'z')) or header['DATA'][0] not in ('ascii', 'binary'):
        raise ValueError('Unsupported Track1 map contract')
    mask = FrontMask(data)  # Rectification only; apply() and GPU are never used.
    if mask.mask.shape != (928, 1440) or not np.any(mask.mask):
        raise ValueError('Invalid front-camera mask')
    clock = CameraClock(data/'Track1.db3')
    sensors = SensorStream(data/'Track1.db3', p.GNSS_EXPECTED_FRAME)
    manifest = []; delivered = 0; latest_imu = None; ages = []; counts = []
    first_gnss = None
    try:
        for row in rows:
            arrival = row['timestamp_ns']; camera = clock.for_frame(arrival)
            events = sensors.take(arrival, camera)
            for event in events:
                if event['arrival_ns'] > arrival or event['timestamp_ns'] > camera:
                    raise AssertionError('Noncausal sensor availability')
                if event['kind'] == 'imu': latest_imu = event['timestamp_ns']
                elif first_gnss is None: first_gnss = event['xy']
            if latest_imu is None:
                raise ValueError('No available IMU at Track1 frame')
            age = (camera-latest_imu)*1e-9
            if not 0 <= age <= p.IMU_MAX_AGE_S:
                raise ValueError('Track1 IMU age exceeds runner contract')
            delivered += len(events); ages.append(age)
            image = data/'Track1_images/Track1_f_images'/row['path'].with_suffix('.jpg').name
            if not image.is_file():
                raise FileNotFoundError(image)
            manifest.append(dict(idx=row['original_idx'], original_idx=row['original_idx'],
                timestamp_ns=arrival, camera_timestamp_ns=camera, path=str(row['path']),
                size_bytes=row['path'].stat().st_size, content_read=False))
            counts.append(dict(idx=row['original_idx'], camera_timestamp_ns=camera,
                               imu_age_seconds=age, delivered_events=len(events)))
        sources = {n: dict(path=str(data/n), sha256=sha256(data/n))
                   for n in ('Track1.db3', 'init_guess.txt', 'metadata.yaml')}
        # Large map/PLY payloads are not read by this contract-only check.
        # The localization runner records their actual content SHA when used.
        sources['map.pcd'] = dict(path=str(data/'map.pcd'), size_bytes=(data/'map.pcd').stat().st_size,
                                  sha256=None, payload_read=False)
        result = dict(status='passed', version=p.VERSION, track=1, frames=len(rows),
            sensors=sensors.summary(), delivered_events=delivered,
            max_imu_age_seconds=max(ages), first_available_gnss_xy=first_gnss,
            gnss_expected_frame=p.GNSS_EXPECTED_FRAME,
            gnss_frame_semantics='empty header assumed map; frame/lever/covariance uncalibrated',
            pcd_header=header, pcd_payload_offset=offset, input_sources=sources,
            camera_sources=mask.sources, mask_shape=list(mask.mask.shape),
            kind='read_only_input_contract_check_no_localization',
            localization_executed=False, cuda_used=False, elapsed_seconds=time.perf_counter()-start)
        write_json(run/'data_contract.json', result)
        write_json(run/'input_frame_manifest.json', manifest)
        write_json(run/'input_timing.json', counts)
        print(json.dumps({k:v for k,v in result.items() if k not in ('input_sources','pcd_header','camera_sources')},indent=2))
    finally:
        clock.close(); sensors.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--record-dir',type=Path,required=True)
    inspect(parser.parse_args().record_dir.resolve())
