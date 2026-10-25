#!/usr/bin/env python3
"""建立明確標記為未校正的 IMU 外參假設設定，不執行定位。"""
import argparse
import json
import math
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__,epilog=(
        'p_ply = R_ply_imu * p_imu + t_ply_imu；xyz 為 IMU 原點在 PLY FLU 的公尺座標。'
        '固定軸 xyz（roll、pitch、yaw）旋轉採 SciPy from_euler("xyz", degrees=True)。'
        '這只是供手動逐次測試的物理假設，不代表已完成校正。'))
    parser.add_argument('--base',type=Path,default=Path(__file__).resolve().parents[1]/'config/track3.json',help='基底設定 JSON')
    parser.add_argument('--rpy-deg',type=float,nargs=3,required=True,metavar=('ROLL','PITCH','YAW'),help='IMU 轉到 PLY 的固定軸旋轉，單位度')
    parser.add_argument('--xyz-m',type=float,nargs=3,required=True,metavar=('X','Y','Z'),help='IMU 原點在 PLY FLU 的位置，單位公尺')
    parser.add_argument('--output',type=Path,required=True,help='新設定檔；已存在時拒絕覆寫')
    args=parser.parse_args()
    if not all(math.isfinite(v) for v in args.rpy_deg+args.xyz_m):
        parser.error('旋轉與平移必須是有限數值')
    cfg=json.loads(args.base.read_text())
    if not isinstance(cfg,dict) or not isinstance(cfg.get('imu'),dict):
        parser.error('基底設定必須含 imu object')
    roll,pitch,yaw=map(math.radians,args.rpy_deg)
    cr,sr=math.cos(roll),math.sin(roll)
    cp,sp=math.cos(pitch),math.sin(pitch)
    cy,sy=math.cos(yaw),math.sin(yaw)
    x,y,z=args.xyz_m
    # Fixed-axis xyz, equivalent to SciPy from_euler('xyz'): Rz @ Ry @ Rx.
    cfg['imu']['T_ply_imu']=[
        [cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr,x],
        [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr,y],
        [-sp,cp*sr,cp*cr,z],[0.,0.,0.,1.]]
    cfg['imu']['extrinsic_status']='assumed_trial_unverified_not_calibrated'
    cfg['imu']['extrinsic_source']='config'
    rendered=json.dumps(cfg,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with args.output.open('x') as out:
        out.write(rendered)
    print(args.output)


if __name__=='__main__':main()
