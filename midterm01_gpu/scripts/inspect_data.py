#!/usr/bin/env python3
"""Read-only dataset/clock/frame preflight. Never runs registration."""
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu
from track3.io import frames, sha256, write_json
from track3.geometry import valid_pose


def main():
    p=argparse.ArgumentParser();p.add_argument('--record-dir',type=Path,required=True);a=p.parse_args()
    a.record_dir.mkdir(parents=True,exist_ok=True)
    data=Path('/workspace/midterm01/data/Track3');r=frames(data/'raw_ply')
    cams={}
    for camera in ('f','fl','fr'):
        rows=frames(data/f'Track3_images/Track3_{camera}_images',suffix='.jpg')
        cams[camera]=dict(count=len(rows),same_timestamps_as_ply=[v['timestamp_ns'] for v in rows]==[v['timestamp_ns'] for v in r])
    db=sqlite3.connect(f'file:{data/"Track3.db3"}?mode=ro',uri=True)
    topics=db.execute('select id,name,type from topics').fetchall()
    imu_id=next(i for i,n,t in topics if n=='/imu/data')
    deltas=[];last=-1;valid=0;total=0;zero_covariance=0
    for bag,raw in db.execute('select timestamp,data from messages where topic_id=? order by timestamp,id',(imu_id,)):
        m=deserialize_message(raw,Imu);stamp=int(m.header.stamp.sec)*10**9+int(m.header.stamp.nanosec)
        q=np.array([m.orientation.x,m.orientation.y,m.orientation.z,m.orientation.w])
        if stamp<=last:raise ValueError('Non-monotonic IMU header time')
        last=stamp;total+=1;deltas.append(int(bag)-stamp)
        valid+=int(np.isfinite(q).all() and .9<np.linalg.norm(q)<1.1 and m.orientation_covariance[0]!=-1)
        zero_covariance+=int(not any(m.orientation_covariance))
    report=dict(status='success',test_type='read-only dataset inspection; no localization',
                frames=len(r),first_timestamp_ns=r[0]['timestamp_ns'],last_timestamp_ns=r[-1]['timestamp_ns'],
                cameras=cams,initial_pose_valid=valid_pose(np.loadtxt(data/'init_guess.txt',delimiter=',')),
                topics=topics,imu_messages=total,valid_orientation=valid,orientation_covariance_unknown=zero_covariance,
                bag_minus_header_ns=dict(min=min(deltas),max=max(deltas)),
                sha256={str(p):sha256(p) for p in (data/'init_guess.txt',data/'metadata.yaml')},
                caveat='FLU inferred from bundled exporter; official PLY provenance and Kaggle reference point are not independently verified')
    if not report['initial_pose_valid']:raise ValueError('Invalid initial pose')
    write_json(a.record_dir/'data_inspection.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
