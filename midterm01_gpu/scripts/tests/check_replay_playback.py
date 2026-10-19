#!/usr/bin/env python3
"""Host-side, isolated-domain RViz regression on a short saved-run manifest."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from output_layout import loads_saved_json, saved_path
import replay
from replay_run import container_path


STREAM_CHECK = '''import rclpy,time,json,hashlib,numpy as np
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
rclpy.init();n=rclpy.create_node('inspect_playback_stream')
q=QoSProfile(depth=10,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
sources=[];poses=[]
def source(m):
 fields={f.name:f.offset for f in m.fields}
 dtype=np.dtype(dict(names=['x','y','z'],formats=['<f4']*3,offsets=[fields[n] for n in ['x','y','z']],itemsize=m.point_step))
 view=np.frombuffer(m.data,dtype=dtype,count=m.width*m.height)
 xyz=np.column_stack([view[n] for n in ['x','y','z']])
 sources.append((time.monotonic(),m.header.stamp.sec*1000000000+m.header.stamp.nanosec,m.header.frame_id,hashlib.sha256(xyz.tobytes()).hexdigest()))
def pose(m):poses.append(m.header.stamp.sec*1000000000+m.header.stamp.nanosec)
s=n.create_subscription(PointCloud2,'/replay/source',source,q)
p=n.create_subscription(PoseStamped,'/replay/pose',pose,q)
end=time.monotonic()+10
while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.05)
result=dict(source_messages=len(sources),distinct_clouds=len(set(v[3] for v in sources)),pose_messages=len(poses),
 same_stamp_pairs=len(set(v[1] for v in sources)&set(poses)),source_frames=sorted(set(v[2] for v in sources)),
 max_cloud_arrival_gap_s=max([b[0]-a[0] for a,b in zip(sources,sources[1:])],default=None))
print(json.dumps(result));n.destroy_node();rclpy.shutdown()
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('prepared_replay',type=saved_path)
    parser.add_argument('--output',type=saved_path,required=True)
    parser.add_argument('--before-binary',type=saved_path)
    parser.add_argument('--domain',type=int,default=141,help='Temporary diagnostic domain; project domain stays unchanged')
    args=parser.parse_args()
    root=replay.ROOT
    data_root,container,_=replay.environment()
    args.output.mkdir(parents=True,exist_ok=True)
    meta=loads_saved_json((args.prepared_replay/'replay.json').read_text())
    assert 3<=len(meta['records'])<=100,'Use only a bounded diagnostic segment'
    cfg=yaml.safe_load((root/'workspace/src/localization/config/replay.rviz').read_text())
    params=dict(frame_manifest=container_path(args.prepared_replay/'frames.tsv',root,data_root),
                map_file=meta['map_file'],frame_index=0,paused=False,speed=1.,loop=True,exit_at_end=False)
    for key in ('ply_scale','input_voxel_leaf_size','map_voxel_leaf_size','submap_half_size','submap_half_height'):
        params[key]=float(meta['params'][key])
    if meta['map_cache']:params['map_voxel_leaf_size']=0.
    results=[]
    variants=([('before',container_path(args.before_binary,root,data_root))] if args.before_binary else [])+[('after',replay.BINARY)]
    for label,binary in variants:
        output=args.output/label
        output.mkdir(exist_ok=False)
        config=output/'playback.rviz'
        config.write_text(yaml.safe_dump(cfg,sort_keys=False))
        (output/'params.yaml').write_text(json.dumps({'localization_replay':{'ros__parameters':params}},indent=2)+'\n')
        replay.RVIZ=container_path(config,root,data_root)
        gui=None
        process=None
        with (output/'publisher.log').open('wb') as log:
            try:
                gui=replay.start_gui(container,str(args.domain),output,data_root)
                command=['docker','exec',gui,'bash','-c','source /opt/ros/humble/setup.bash && exec '+shlex.quote(binary)
                         +' --ros-args --params-file '+shlex.quote(container_path(output/'params.yaml',root,data_root))]
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                time.sleep(2)
                inspection=subprocess.check_output(['docker','exec',gui,'bash','-c',
                    'source /opt/ros/humble/setup.bash && exec python3 -c '+shlex.quote(STREAM_CHECK)],text=True)
                stream=loads_saved_json(inspection.strip().splitlines()[-1])
                logs=subprocess.check_output(['docker','logs',gui],stderr=subprocess.STDOUT,text=True)
                (output/'rviz.log').write_text(logs)
                drops=[line for line in logs.splitlines() if 'Message Filter dropping message' in line]
                result=dict(variant=label,binary=binary,domain=args.domain,selected_frames=len(meta['records']),
                            stream=stream,rviz_message_filter_drop_log_lines=len(drops))
                (output/'playback_check.json').write_text(json.dumps(result,indent=2)+'\n')
                results.append(result)
                print(json.dumps(result),flush=True)
                if label=='after':
                    assert stream['source_frames']==['map']
                    assert stream['distinct_clouds']>=10 and stream['same_stamp_pairs']>=10
                    assert stream['max_cloud_arrival_gap_s']<1.,'Cloud stream stalled'
                    assert not drops,'RViz still drops clouds while moving'
            finally:
                if gui:subprocess.run(['docker','rm','-f',gui],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                if process is not None:process.wait(timeout=10)
                (output/'display.xauth').unlink(missing_ok=True)
    (args.output/'comparison.json').write_text(json.dumps(dict(results=results,scope='Short saved-output playback, no localization'),indent=2)+'\n')


if __name__=='__main__':main()
