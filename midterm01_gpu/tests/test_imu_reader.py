"""Generated real ROS CDR messages in SQLite; unrelated topics are never decoded."""
import sqlite3
import numpy as np
import pytest
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Imu
from track3.imu import ImuReader


def bag(tmp_path, bad=None):
    path=tmp_path/'generated.db3'
    db=sqlite3.connect(path)
    db.execute('CREATE TABLE topics(id INTEGER,name TEXT,type TEXT,serialization_format TEXT)')
    db.execute('CREATE TABLE messages(id INTEGER,topic_id INTEGER,timestamp INTEGER,data BLOB)')
    db.executemany('INSERT INTO topics VALUES(?,?,?,?)',[(1,'/imu/data','sensor_msgs/msg/Imu','cdr'),(2,'/gnss','fake','cdr')])
    stamp=1727239790258245070
    for i in range(4):
        m=Imu();m.header.frame_id='wrong' if bad=='frame' else 'imu_link'
        m.header.stamp.sec=1727239790;m.header.stamp.nanosec=258245000+i
        m.angular_velocity.z=.1*i;m.linear_acceleration.z=9.80665
        if bad=='unavailable':m.angular_velocity_covariance[0]=-1
        if bad=='nan':m.linear_acceleration.x=float('nan')
        db.execute('INSERT INTO messages VALUES(?,?,?,?)',(i+1,1,stamp+i*10,serialize_message(m)))
    db.execute('INSERT INTO messages VALUES(99,2,?,?)',(stamp,b'not CDR'))
    db.commit();db.close()
    return path,stamp


def test_readonly_integer_clock_and_availability_cutoff(tmp_path):
    path,stamp=bag(tmp_path);before=path.read_bytes();reader=ImuReader(path)
    assert reader.take_until(stamp-1)==[]
    first=reader.take_until(stamp+10)
    assert len(first)==2 and first[-1].timestamp_ns==stamp+10
    assert first[0].header_timestamp_ns==1727239790258245000
    assert [s.message_id for s in first]==[1,2]
    np.testing.assert_allclose(first[1].angular_velocity,[0,0,.1])
    with pytest.raises(sqlite3.OperationalError):reader.db.execute('DELETE FROM messages')
    with pytest.raises(ValueError):reader.take_until(stamp+10)
    with pytest.raises(ValueError):reader.take_until(float(stamp+100))
    assert len(reader.take_until(stamp+20))==1
    reader.close();assert before==path.read_bytes()


@pytest.mark.parametrize('bad',['frame','unavailable','nan'])
def test_bad_sensor_data_fails_explicitly(tmp_path,bad):
    path,stamp=bag(tmp_path,bad);reader=ImuReader(path)
    try:
        with pytest.raises(ValueError):reader.take_until(stamp)
    finally:reader.close()
