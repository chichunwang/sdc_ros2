"""生成感測器測試：保留GNSS orientation、雙偏差可觀測性及雙時間因果。"""
import json,sqlite3
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from track2.causal_prior import CausalSensorPrior,wrap
from track2 import parameters as p


def imu(stamp,angle=-5.,gyro=0.):
    return dict(kind='imu',timestamp_ns=stamp,quaternion=Rotation.from_euler('z',angle,degrees=True).as_quat().tolist(),gyro=[0.,0.,np.deg2rad(gyro)])


def gnss(stamp,angle=5.,x=0.):
    return dict(kind='gnss',timestamp_ns=stamp,xy=[x,0.],z=1.6,heading_quaternion=Rotation.from_euler('z',angle,degrees=True).as_quat().tolist())


def test_two_unknown_biases_fuse_current_orientation_without_claiming_exact_heading():
    s=CausalSensorPrior(np.eye(4),1_000_000_000)
    s.event(imu(1_000_000_000));s.event(gnss(1_000_000_000));T,d=s.for_frame(1_000_000_000)
    # Independent equal unknown bias budgets, not treating GNSS yaw as truth.
    variance=10.**2;relative_noise=3.**2+1.**2
    expected=variance/(2*variance+relative_noise)*10.
    assert np.degrees(s.bias)==pytest.approx(expected)
    assert np.degrees(np.arctan2(T[1,0],T[0,0]))==pytest.approx(-5.+expected)
    assert d['gnss_heading_updates']==1 and d['accepted_heading_blocks']==0
    assert np.degrees(np.sqrt(s.bias_var))>7.
    assert d['gnss_heading_decision']['delivered_timestamp_ns']==1_000_000_000


def test_held_orientation_does_not_repeatedly_shrink_common_bias_uncertainty(monkeypatch):
    monkeypatch.setattr(p,'HEADING_BIAS_RW_DEG_SQRT_S',0.)
    monkeypatch.setattr(p,'GNSS_HEADING_BIAS_RW_DEG_SQRT_S',0.)
    s=CausalSensorPrior(np.eye(4),1_000_000_000)
    for k in range(160):
        t=1_000_000_000+k*50_000_000;s.event(imu(t))
        if k%5==0:s.event(gnss(t))
        _,d=s.for_frame(t)
    common=np.array([1.,1.])/np.sqrt(2)
    assert common@s.heading_covariance@common==pytest.approx(np.deg2rad(10.)**2,abs=1e-14)
    assert s.gnss_heading_updates==1 and s.accepted_blocks==0
    assert d['gnss_heading_decision']['reason']=='held_gnss_orientation_already_used'


def simulate(n,with_heading=True):
    s=CausalSensorPrior(np.eye(4),1_000_000_000);outputs=[]
    for k in range(n):
        t=1_000_000_000+k*50_000_000;s.event(imu(t,angle=-4.))
        if k%5==0:
            row=gnss(t,angle=7.,x=max(0.,k*.05-2.)*8.)
            if not with_heading:row.pop('heading_quaternion')
            s.event(row)
        T,d=s.for_frame(t);outputs.append((T,json.dumps(d,sort_keys=True,allow_nan=False)))
    return s,outputs


def test_course_later_observes_common_bias_and_prefix_never_changes():
    short,a=simulate(70);long,b=simulate(240)
    for x,y in zip(a,b):np.testing.assert_array_equal(x[0],y[0]);assert x[1]==y[1]
    assert long.accepted_blocks>0
    assert np.degrees(long.bias)==pytest.approx(4.,abs=.5)
    # A finite prior pulls the uncertain GNSS bias toward zero; it is not exact truth.
    assert abs(long.heading_bias[1]-np.deg2rad(-7.)) < np.sqrt(long.heading_covariance[1,1])
    common=np.array([1.,1.])/np.sqrt(2)
    assert common@long.heading_covariance@common < common@short.heading_covariance@common
    assert np.linalg.eigvalsh(long.heading_covariance).min()>0


def test_orientation_pair_uses_actual_limited_gain_in_joint_joseph(monkeypatch):
    monkeypatch.setattr(p,'HEADING_MAX_BIAS_DEG',.5)
    s=CausalSensorPrior(np.eye(4),1_000_000_000)
    s.event(imu(1_000_000_000,angle=-8.));s.event(gnss(1_000_000_000,angle=12.))
    _,d=s.for_frame(1_000_000_000);q=d['gnss_heading_decision']
    H=np.array(q['observation_matrix']);K=np.array(q['heading_gain']);P=np.array(q['covariance_before']);A=np.eye(2)-np.outer(K,H)
    assert np.degrees(s.bias)==pytest.approx(.5)
    np.testing.assert_allclose(s.heading_covariance,A@P@A.T+np.outer(K,K)*q['measurement_variance'],atol=1e-15)
    assert q['bias_gain']<q['unconstrained_bias_gain']


def test_turning_gnss_pair_is_rejected_without_losing_position():
    s=CausalSensorPrior(np.eye(4),1_000_000_000)
    s.event(imu(1_000_000_000,gyro=3.));s.event(gnss(1_000_000_000,x=5.))
    _,d=s.for_frame(1_000_000_000)
    assert s.gnss_updates==1 and s.gnss_heading_updates==0
    assert d['gnss_heading_decision']['reason']=='turning_or_held_gnss_heading'
    assert s.position[0]>4.


def test_disable_or_missing_orientation_keeps_the_same_sensor_prior(monkeypatch):
    _,old=simulate(80,with_heading=False)
    monkeypatch.setattr(p,'GNSS_HEADING_ENABLED',False)
    _,disabled=simulate(80,with_heading=True)
    for a,b in zip(old,disabled):np.testing.assert_array_equal(a[0],b[0]);assert a[1]==b[1]


def test_stream_preserves_orientation_and_waits_for_arrival_and_camera_cutoffs(tmp_path):
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import Imu
    from geometry_msgs.msg import PoseStamped
    from track2.sensors import SensorStream
    db=sqlite3.connect(tmp_path/'bag.db3');db.execute('CREATE TABLE topics(id INTEGER,name TEXT,type TEXT,serialization_format TEXT)');db.execute('CREATE TABLE messages(id INTEGER,timestamp INTEGER,topic_id INTEGER,data BLOB)')
    db.executemany('INSERT INTO topics VALUES(?,?,?,?)',[(1,'/imu/data','sensor_msgs/msg/Imu','cdr'),(2,'/localizer_node/gnss_pose','geometry_msgs/msg/PoseStamped','cdr')])
    for mid,topic,arrival,stamp,angle in [(1,1,110,100,-4.),(2,1,210,190,-4.),(3,2,220,200,7.)]:
        m=Imu() if topic==1 else PoseStamped();m.header.stamp.sec,m.header.stamp.nanosec=divmod(stamp,1_000_000_000)
        q=Rotation.from_euler('z',angle,degrees=True).as_quat()
        if topic==1:m.header.frame_id='imu_link';a=m.orientation
        else:a=m.pose.orientation
        a.x,a.y,a.z,a.w=q.tolist();db.execute('INSERT INTO messages VALUES(?,?,?,?)',(mid,arrival,topic,serialize_message(m)))
    m=PoseStamped();m.header.stamp.nanosec=300;m.pose.position.x=5.;m.pose.orientation.w=0.
    db.execute('INSERT INTO messages VALUES(?,?,?,?)',(4,320,2,serialize_message(m)))
    db.commit();db.close();stream=SensorStream(tmp_path/'bag.db3');s=CausalSensorPrior(np.eye(4),100)
    for event in stream.take(225,150):s.event(event)
    T,d=s.for_frame(150);saved=T.copy();assert d['gnss_heading_updates']==0
    for event in stream.take(260,230):s.event(event)
    _,d=s.for_frame(230);assert d['gnss_heading_updates']==1
    assert d['gnss_heading_decision']['timestamp_ns']==200
    np.testing.assert_array_equal(T,saved)
    invalid=stream.take(340,330)
    assert len(invalid)==1 and invalid[0]['heading_quaternion'] is None
    assert invalid[0]['heading_status']=='invalid_orientation' and invalid[0]['xy'][0]==5.
    s.event(invalid[0]);_,d=s.for_frame(330)
    assert s.gnss_updates==2 and d['gnss_heading_updates']==1
    assert d['gnss_heading_decision']['reason']=='invalid_or_missing_gnss_orientation'
    stream.close()
    # The header alone is insufficient: GNSS stamp200 arrives at220, after cutoff215.
    guard=SensorStream(tmp_path/'bag.db3')
    assert all(row['kind']=='imu' for row in guard.take(215,205))
    assert guard.take(260,230)==[] and guard.counts['late']==1
    guard.close()


def test_heading_pair_wraps_across_pi():
    s=CausalSensorPrior(np.eye(4),1_000_000_000)
    s.event(imu(1_000_000_000,angle=179.));s.event(gnss(1_000_000_000,angle=-179.))
    T,d=s.for_frame(1_000_000_000)
    assert d['gnss_heading_updates']==1
    assert abs(np.degrees(wrap(np.arctan2(T[1,0],T[0,0])-np.pi)))<1.1
