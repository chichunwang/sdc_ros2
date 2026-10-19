#!/usr/bin/env python3
"""Audit unused PoseStamped orientation; never assume heading provenance or truth."""
from output_layout import saved_path

import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
from scipy.spatial.transform import Rotation

from sensor_fusion import FusionConfig, heading_observations
from three_camera_source import sha256


def wrap(x):
    return np.arctan2(np.sin(x),np.cos(x))


def stats(v):
    v=np.asarray(v,float)
    return {'count':len(v),'rms':float(np.sqrt(np.mean(v*v))),
            'median':float(np.median(v)),'absolute_p90':float(np.quantile(np.abs(v),.9))} if len(v) else {'count':0}


def export(bag,output):
    from rclpy.serialization import deserialize_message
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Imu
    data={};frames={};counts={}
    with sqlite3.connect(f'file:{bag.resolve()}?mode=ro',uri=True)as connection:
        topics={name:ident for ident,name in connection.execute('SELECT id,name FROM topics')}
        for prefix,topic,cls in [('gnss','/localizer_node/gnss_pose',PoseStamped),('imu','/imu/data',Imu)]:
            series=list(connection.execute('SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp',(topics[topic],)))
            messages=[deserialize_message(blob,cls)for _,blob in series]
            data[prefix+'_bag_ns']=np.array([stamp for stamp,_ in series],dtype=np.int64)
            data[prefix+'_header_ns']=np.array([m.header.stamp.sec*1000000000+m.header.stamp.nanosec for m in messages],dtype=np.int64)
            quats=[m.pose.orientation if prefix=='gnss' else m.orientation for m in messages]
            data[prefix+'_quat']=np.array([[q.x,q.y,q.z,q.w]for q in quats])
            frames[prefix]=sorted({m.header.frame_id for m in messages});counts[prefix]=len(messages)
            if prefix=='gnss':
                data['gnss_xyz']=np.array([[m.pose.position.x,m.pose.position.y,m.pose.position.z]for m in messages])
            else:
                for key in ('angular_velocity','linear_acceleration'):
                    data[key]=np.array([[getattr(m,key).x,getattr(m,key).y,getattr(m,key).z]for m in messages])
                    data[key+'_covariance']=np.array([getattr(m,key+'_covariance')for m in messages])
                data['orientation_covariance']=np.array([m.orientation_covariance for m in messages])
    with output.open('xb')as stream:np.savez_compressed(stream,**data)
    output.with_suffix('.json').write_text(json.dumps({'source_bag':str(bag),'bag_sha256':sha256(bag),
        'frames':frames,'samples':counts,'raw_values_sha256':sha256(output),'orientation_source_verified':False},indent=2)+'\n')


def analyze(raw_path,output):
    with np.load(raw_path)as raw:d={k:raw[k]for k in raw.files}
    gnss_ns,imu_ns=d['gnss_bag_ns'],d['imu_bag_ns'];origin=int(gnss_ns[0])
    gt,it=(gnss_ns-origin)/1e9,(imu_ns-origin)/1e9
    norm=np.linalg.norm(d['gnss_quat'],axis=1)
    if np.any(np.abs(norm-1)>.02):raise ValueError('GNSS Pose contains invalid quaternion; cannot audit yaw')
    yaw=np.unwrap(Rotation.from_quat(d['gnss_quat']).as_euler('xyz')[:,2])
    imu_yaw=np.unwrap(Rotation.from_quat(d['imu_quat']).as_euler('xyz')[:,2])
    covered=(gt>=it[0])&(gt<=it[-1])
    imu_at=np.interp(gt,it,imu_yaw)
    xy=d['gnss_xyz'][:,:2];dt=np.diff(gt)
    velocity=np.gradient(xy,gt,axis=0);speed=np.linalg.norm(velocity,axis=1)
    turning=np.degrees(np.abs(np.gradient(imu_at,gt)))
    moving=(speed>5)&covered;straight=moving&(turning<2)
    backward=np.arctan2(np.diff(xy[:,1]),np.diff(xy[:,0]));back_speed=np.linalg.norm(np.diff(xy,axis=0),axis=1)/dt
    centered=np.arctan2(velocity[:,1],velocity[:,0])
    cfg=FusionConfig();blocks=heading_observations(gt,xy,it,imu_yaw,cfg)
    bins=np.floor((gt-gt[0])/cfg.course_window_s).astype(int);block_rows=[]
    for row,block in zip(blocks,np.unique(bins)):
        if not row['accepted']:continue
        indices=np.flatnonzero(bins==block)
        provided=float(np.arctan2(np.mean(np.sin(yaw[indices])),np.mean(np.cos(yaw[indices]))))
        course=np.deg2rad(row['offset_deg'])+float(np.mean(np.interp(gt[indices],it,imu_yaw)))
        block_rows.append({'time_s':row['time_s'],'provided_minus_course_deg':float(np.degrees(wrap(provided-course))),
            'course_minus_imu_deg':row['offset_deg'],'provided_minus_imu_deg':float(np.degrees(wrap(provided-float(np.mean(np.interp(gt[indices],it,imu_yaw))))))})
    gyro=d['angular_velocity'];yaw_rate=np.gradient(imu_yaw,it)
    report={'raw_export_sha256':sha256(raw_path),'gnss_samples':len(gt),'imu_samples':len(it),
        'gnss_quaternion_norm_min_max':[float(norm.min()),float(norm.max())],
        'gnss_roll_pitch_yaw_span_deg':np.ptp(Rotation.from_quat(d['gnss_quat']).as_euler('xyz',degrees=True),axis=0).tolist(),
        'provided_heading_minus_imu_deg':{'moving':stats(np.degrees(wrap(yaw[moving]-imu_at[moving]))),
            'straight':stats(np.degrees(wrap(yaw[straight]-imu_at[straight])))},
        'provided_heading_minus_position_course_deg':{
            'backward_difference_moving':stats(np.degrees(wrap(yaw[1:][back_speed>5]-backward[back_speed>5]))),
            'centered_difference_moving':stats(np.degrees(wrap(yaw[moving]-centered[moving]))),
            'robust_two_second_straight_blocks':stats([r['provided_minus_course_deg']for r in block_rows])},
        'robust_course_blocks':block_rows,
        'gyro_z_minus_quaternion_yaw_rate_rad_s':stats(gyro[:,2]-yaw_rate),
        'gyro_z_vs_quaternion_yaw_rate_correlation':float(np.corrcoef(gyro[:,2],yaw_rate)[0,1]),
        'zero_covariance_arrays':{key:bool(np.all(d[key]==0))for key in
            ('angular_velocity_covariance','linear_acceleration_covariance','orientation_covariance')},
        'bag_minus_header_ms':{prefix:stats((d[prefix+'_bag_ns']-d[prefix+'_header_ns'])/1e6)for prefix in ('gnss','imu')},
        'source_verified':False,'accepted_for_fusion':False,
        'limits':['PoseStamped orientation has values but its publisher/source and uncertainty are unverified.',
            'Agreement with course or IMU does not make the heading an independent measurement or truth.',
            'All-zero IMU covariance is unspecified covariance, not perfect accuracy.',
            'Quaternion yaw derivative and gyro are correlated IMU outputs; RMS is consistency only.']}
    with output.open('x')as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in report.items()if k not in ('robust_course_blocks','limits')},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('export','analyze'))
    parser.add_argument('--input',type=saved_path,required=True);parser.add_argument('--output',type=saved_path,required=True)
    args=parser.parse_args()
    if args.mode=='export':export(args.input,args.output)
    else:analyze(args.input,args.output)


if __name__=='__main__':main()
