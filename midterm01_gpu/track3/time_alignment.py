"""Arrival-causal sensor acquisition clocks; original PLY bag stamps stay intact."""
from pathlib import Path
import sqlite3
from .imu import ImuSample


class CameraClock:
    def __init__(self,path,topic='/lucid_cameras_x00/gige_100_f_hdr/h265'):
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import Image
        self.decode=lambda data:deserialize_message(data,Image)
        self.db=sqlite3.connect(Path(path).resolve(strict=True).as_uri()+'?mode=ro',uri=True)
        self.db.execute('PRAGMA query_only=ON')
        rows=self.db.execute('SELECT id,type,serialization_format FROM topics WHERE name=?',(topic,)).fetchall()
        if len(rows)!=1 or rows[0][1:]!=('sensor_msgs/msg/Image','cdr'):
            self.db.close();raise ValueError('Front camera topic must uniquely contain sensor_msgs/msg/Image CDR')
        self.cursor=self.db.execute('SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id',(rows[0][0],))
        self.next=self.cursor.fetchone();self.last_arrival=None;self.last_header=None

    def for_frame(self,arrival_ns):
        if type(arrival_ns) is not int or (self.last_arrival is not None and arrival_ns<=self.last_arrival):
            raise ValueError('Camera arrival clock must strictly increase in integer ns')
        while self.next is not None and self.next[0]<arrival_ns:self.next=self.cursor.fetchone()
        if self.next is None or self.next[0]!=arrival_ns:raise ValueError('PLY stamp does not exactly match front-camera arrival')
        message=self.decode(self.next[1])
        stamp=int(message.header.stamp.sec)*1_000_000_000+int(message.header.stamp.nanosec)
        if stamp<=0 or stamp>arrival_ns or (self.last_header is not None and stamp<=self.last_header):
            raise ValueError('Invalid, future, or non-increasing camera acquisition clock')
        self.last_arrival=arrival_ns;self.last_header=stamp;self.next=self.cursor.fetchone()
        return stamp

    def close(self):self.db.close()


class AcquisitionImuReader:
    """Only already-arrived samples whose sensor time is <= the current image.

    Samples ahead of image acquisition stay buffered. Late samples never modify
    a past pose: they are discarded and counted, rather than retro-integrated.
    """
    def __init__(self,reader):
        self.reader=reader;self.queue=[];self.last_measurement=None
        self.delivered=0;self.late=0

    @property
    def count(self):return self.reader.count

    def take_for_frame(self,arrival_ns,measurement_ns):
        if type(measurement_ns) is not int or measurement_ns>arrival_ns:
            raise ValueError('Image acquisition must be integer ns and no later than availability')
        if self.last_measurement is not None and measurement_ns<=self.last_measurement:
            raise ValueError('Acquisition clock must strictly increase')
        for raw in self.reader.take_until(arrival_ns):
            stamp=raw.header_timestamp_ns
            if stamp<=0 or stamp>raw.timestamp_ns:raise ValueError('Invalid or future IMU acquisition timestamp')
            if self.last_measurement is not None and stamp<=self.last_measurement:
                self.late+=1;continue
            self.queue.append(ImuSample(stamp,stamp,raw.angular_velocity,raw.linear_acceleration,
                                        raw.frame_id,raw.message_id,raw.timestamp_ns))
        self.queue.sort(key=lambda s:(s.timestamp_ns,s.message_id))
        ready=[];future=[]
        for sample in self.queue:
            if sample.timestamp_ns<=measurement_ns:
                if ready and ready[-1].timestamp_ns==sample.timestamp_ns:ready[-1]=sample
                else:ready.append(sample)
            else:future.append(sample)
        self.queue=future;self.delivered+=len(ready);self.last_measurement=measurement_ns
        return ready

    def summary(self):
        return dict(self.reader.summary(),acquisition_samples_delivered=self.delivered,
                    late_acquisition_samples_discarded=self.late,future_acquisition_samples_buffered=len(self.queue),
                    integration_clock='IMU header; frame state time=paired camera header; availability gate=bag arrival')

    def close(self):self.reader.close()
