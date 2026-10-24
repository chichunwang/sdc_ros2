import sqlite3
import numpy as np
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Imu
from track3.io import ImuStream


def test_both_clocks_and_unknown_orientation(tmp_path):
    p=tmp_path/'imu.db3';db=sqlite3.connect(p)
    db.executescript('create table topics(id integer,name text,type text);create table messages(id integer,topic_id integer,timestamp integer,data blob);')
    db.execute("insert into topics values(1,'/imu/data','sensor_msgs/msg/Imu')")
    def row(i,bag,header,unknown=False):
        m=Imu();m.header.stamp.sec=1;m.header.stamp.nanosec=header
        m.orientation.w=1.;m.angular_velocity.z=.2
        if unknown:m.orientation_covariance[0]=-1.
        db.execute('insert into messages values(?,?,?,?)',(i,1,1_000_000_000+bag,serialize_message(m)))
    row(1,10,100);row(2,200,150);row(3,300,250,True);db.commit();db.close()
    s=ImuStream(p)
    assert s.until(1_000_000_050)==[]  # header in future even though bag available
    got=s.until(1_000_000_100)
    assert len(got)==1 and got[0][:2]==(1_000_000_100,1_000_000_010)
    assert s.until(1_000_000_175)==[]  # bag availability in future
    assert len(s.until(1_000_000_200))==1
    assert s.until(1_000_000_400)==[] and s.invalid==1
    s.close()
