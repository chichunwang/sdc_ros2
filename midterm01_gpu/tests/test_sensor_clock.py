"""Arrival jitter must not turn 36 ms camera motion into a 12 ms velocity."""
import sqlite3
import json
from pathlib import Path
import numpy as np
import pytest
from sensor_msgs.msg import Imu,Image
from rclpy.serialization import serialize_message
from track3.imu import ImuReader
from track3.time_alignment import CameraClock,AcquisitionImuReader
from track3.inertial import InertialState

STAMP=1727239790226594548


def database(tmp_path):
    path=tmp_path/'clock.db3';db=sqlite3.connect(path)
    db.execute('CREATE TABLE topics(id INTEGER,name TEXT,type TEXT,serialization_format TEXT)')
    db.execute('CREATE TABLE messages(id INTEGER,topic_id INTEGER,timestamp INTEGER,data BLOB)')
    db.executemany('INSERT INTO topics VALUES(?,?,?,?)',[(1,'/imu/data','sensor_msgs/msg/Imu','cdr'),
        (2,'/lucid_cameras_x00/gige_100_f_hdr/h265','sensor_msgs/msg/Image','cdr')])
    for i,(header,arrival) in enumerate(((0,32),(36,44),(72,80))):
        msg=Image();t=STAMP+header*1_000_000;msg.header.stamp.sec=t//10**9;msg.header.stamp.nanosec=t%10**9
        db.execute('INSERT INTO messages VALUES(?,?,?,?)',(100+i,2,STAMP+arrival*1_000_000,serialize_message(msg)))
    for i,header in enumerate((-5,5,15,25,35,45,55,65,75,10)):
        msg=Imu();msg.header.frame_id='imu_link';msg.linear_acceleration.z=9.80665
        t=STAMP+header*1_000_000;msg.header.stamp.sec=t//10**9;msg.header.stamp.nanosec=t%10**9
        arrival=header+1 if i!=9 else 50
        db.execute('INSERT INTO messages VALUES(?,?,?,?)',(i+1,1,STAMP+arrival*1_000_000,serialize_message(msg)))
    db.commit();db.close();return path


def test_acquisition_clock_buffers_future_and_discards_late_without_old_pose_edit(tmp_path):
    path=database(tmp_path);before=path.read_bytes();cam=CameraClock(path);imu=AcquisitionImuReader(ImuReader(path))
    try:
        c0=cam.for_frame(STAMP+32_000_000);a=imu.take_for_frame(STAMP+32_000_000,c0)
        assert c0==STAMP and [s.timestamp_ns for s in a]==[STAMP-5_000_000]
        assert len(imu.queue)==3  # Already arrived, but captured after image0.
        c1=cam.for_frame(STAMP+44_000_000);b=imu.take_for_frame(STAMP+44_000_000,c1)
        assert c1-c0==36_000_000
        assert [s.timestamp_ns-STAMP for s in b]==[5_000_000,15_000_000,25_000_000,35_000_000]
        assert all(s.arrival_timestamp_ns<=STAMP+44_000_000 and s.timestamp_ns<=c1 for s in b)
        c2=cam.for_frame(STAMP+80_000_000);c=imu.take_for_frame(STAMP+80_000_000,c2)
        assert [s.timestamp_ns-STAMP for s in c]==[45_000_000,55_000_000,65_000_000]
        assert imu.late==1 and imu.summary()['future_acquisition_samples_buffered']==1
        with pytest.raises(ValueError):cam.for_frame(STAMP+80_000_000)
    finally:cam.close();imu.close()
    assert path.read_bytes()==before


def test_visual_speed_uses_image_time_and_preserves_availability_gate(tmp_path):
    path=database(tmp_path);cam=CameraClock(path);imu=AcquisitionImuReader(ImuReader(path))
    cfg=json.loads((Path(__file__).resolve().parents[1]/'config/track3.json').read_text())['imu']
    # Synthetic measurements below use identity IMU axes, independent of user RPY.
    cfg.update(extrinsic_source='config', T_ply_imu=np.eye(4).tolist(),
               extrinsic_status='synthetic_identity_fixture')
    s=InertialState(np.eye(4),7.5,cfg)
    try:
        c0=cam.for_frame(STAMP+32_000_000);s.predict(c0,imu.take_for_frame(STAMP+32_000_000,c0))
        s.commit(dict(T=np.eye(4),scale=7.5,accepted=True,reason='accepted'))
        motion=np.eye(4);motion[0,3]=.3
        c1=cam.for_frame(STAMP+44_000_000)
        s.predict(c1,imu.take_for_frame(STAMP+44_000_000,c1),visual_motion=motion)
        d=s.prediction_diagnostics
        assert d['visual_velocity_update']['velocity_map_mps'][0]==pytest.approx(.3/.036)
        assert s.v[0]<9.0  # The arrival-based value would be 25 m/s.
        assert d['latest_bag_timestamp_ns']<=STAMP+44_000_000
        assert d['latest_header_timestamp_ns']<=c1
        assert d['integration_clock']=='imu_header_ns_zero_order_hold'
    finally:cam.close();imu.close()


def test_future_or_noninteger_acquisition_is_rejected(tmp_path):
    path=database(tmp_path);imu=AcquisitionImuReader(ImuReader(path))
    try:
        with pytest.raises(ValueError):imu.take_for_frame(STAMP,STAMP+1)
        with pytest.raises(ValueError):imu.take_for_frame(STAMP,float(STAMP))
    finally:imu.close()
