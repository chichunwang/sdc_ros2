#!/usr/bin/env python3
"""Read-only front PLY / XYZ map / initial-pose preflight; no bag or registration."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from track3.io import frames, sha256, write_json
from track3.geometry import valid_pose


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'config/track3.json')
    a=p.parse_args(); run=a.record_dir.resolve()
    if not (run/'source_manifest.json').is_file():
        p.error('先使用 create_run.py 建立唯一的 run 及來源快照')
    if any((run/name).exists() for name in ('test_info.json','data_inspection.json','frame_manifest.csv')):
        p.error('不覆寫已使用的 run，請建立新 run')
    start=time.perf_counter()
    cfg=json.loads(a.config.read_text());data=Path(cfg['data_dir'])
    info=dict(status='running',version=cfg['version'],score=None,score_source=None,
              test_type='read-only dataset inspection; no localization',imu_used=False)
    write_json(run/'test_info.json',info)
    write_json(run/'parameters.json',cfg)
    try:
        rows=frames(data/cfg['ply_subdir'],expected=cfg['expected_frames'])
        pose=np.loadtxt(data/'init_guess.txt',delimiter=',')
        if not valid_pose(pose):raise ValueError('Invalid initial pose')
        # Inspect only the PCD header. No timestamp, intensity or trajectory
        # fields are used to infer pose; the registration loader selects XYZ.
        with (data/'map.pcd').open('rb') as f:
            header=[]
            for _ in range(100):
                line=f.readline().decode('ascii').strip();header.append(line)
                if line.startswith('DATA '):break
        fields=next(line.split()[1:] for line in header if line.startswith('FIELDS '))
        if not all(axis in fields for axis in ('x','y','z')) or header[-1]!='DATA ascii':
            raise ValueError('Map requires XYZ and ASCII PCD for the streaming loader')
        sources=[a.config,data/'init_guess.txt',data/'map.pcd',rows[0]['path'],rows[-1]['path']]
        source_hashes={str(path):sha256(path) for path in sources}
        with (run/'frame_manifest.csv').open('x',newline='') as f:
            writer=csv.writer(f);writer.writerow(['idx','original_idx','timestamp_ns','path'])
            for idx,row in enumerate(rows):writer.writerow([idx,row['original_idx'],row['timestamp_ns'],str(row['path'])])
        report=dict(frames=len(rows),first_timestamp_ns=rows[0]['timestamp_ns'],
                    last_timestamp_ns=rows[-1]['timestamp_ns'],initial_pose_valid=True,
                    map_fields=fields,localization_map_fields=['x','y','z'],imu_used=False,
                    input_sha256=source_hashes,
                    sha_scope='config, initial pose, map, first and last PLY; interior PLY contents not read',
                    caveat='PLY FLU and T_map_ply conventions retained; evaluation frame and metric scale not independently verified')
        write_json(run/'data_inspection.json',report)
        info.update(status='success',frame_count=0,inspected_frames=len(rows),
                    elapsed_seconds=time.perf_counter()-start,output_sha256={name:sha256(run/name) for name in ('data_inspection.json','frame_manifest.csv')})
        print(json.dumps(report,ensure_ascii=False,indent=2))
    except Exception as error:
        info.update(status='failed',failure=f'{type(error).__name__}: {error}',elapsed_seconds=time.perf_counter()-start)
        raise
    finally:
        write_json(run/'test_info.json',info)


if __name__=='__main__':main()
