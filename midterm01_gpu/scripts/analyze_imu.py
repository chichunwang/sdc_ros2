#!/usr/bin/env python3
"""已結束 v5 run 的唯讀 IMU 融合診斷；支援短段，不修改或重估 pose。"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from track3.io import sha256, write_json


def analyze(run):
    started=time.perf_counter()
    names=['test_info.json','observations.jsonl','imu_sources.json','parameters.json']
    sources={name:sha256(run/name) for name in names}
    info=json.loads((run/'test_info.json').read_text())
    if info.get('status') not in ('success','failed') or not info.get('imu_used'):
        raise ValueError('需要已結束的 IMU run；不讀取仍執行中或非 IMU run')
    expected=info.get('output_sha256',{}).get('observations.jsonl')
    if expected is None or expected!=sources['observations.jsonl']:
        raise ValueError('observations SHA 不符已結束 run 的記錄')
    rows=[json.loads(line) for line in (run/'observations.jsonl').read_text().splitlines() if line.strip()]
    if not rows or [r['idx'] for r in rows]!=list(range(len(rows))) or len(rows)!=info.get('frame_count'):
        raise ValueError('需要完整對應 frame_count 的 observations')
    pred=[r['inertial']['prediction'] for r in rows]
    upd=[r['inertial']['update'] for r in rows]
    # Integer subtraction precedes conversion, preserving exact epoch precision.
    stamps=[r['timestamp_ns'] for r in rows]
    if any(type(s) is not int for s in stamps) or any(b<=a for a,b in zip(stamps,stamps[1:])):
        raise ValueError('需要遞增整數 ns timestamps')
    elapsed=np.array([s-stamps[0] for s in stamps])*1e-9
    innovation=np.array([u['innovation'] if u['innovation'] is not None else [np.nan]*6 for u in upd])
    # Bootstrap residual measures the noisy initial guess, not extrinsic quality.
    for i,u in enumerate(upd):
        if u['bootstrap']:innovation[i]=np.nan
    nis=np.array([np.nan if u['nis'] is None else u['nis'] for u in upd])
    limits=np.array([u['nis_limit'] for u in upd])
    velocity=np.array([u['velocity_map_mps'] for u in upd])
    ba=np.array([u['accel_bias_mps2'] for u in upd])
    bg=np.array([u['gyro_bias_radps'] for u in upd])
    cov=np.array([u['covariance_diagonal'] for u in upd])
    age_ms=np.array([p['sample_age_seconds'] for p in pred])*1000
    if (np.any(cov<0) or not np.isfinite(cov).all() or not np.isfinite(velocity).all() or
            not np.isfinite(ba).all() or not np.isfinite(bg).all() or not np.isfinite(age_ms).all()):
        raise ValueError('IMU diagnostics contain invalid state or covariance')
    output=run/'analysis/imu'
    if output.exists():
        manifest=output/'manifest.json'
        previous=json.loads(manifest.read_text()) if manifest.exists() else None
        allowed={'manifest.json','metrics.json','README.md'}|{f'{name}.{ext}' for name in ('01_fusion','02_input_timing') for ext in ('png','pdf')}
        if (previous is None or previous.get('status')=='success' or previous.get('sources')!=sources or
                any(path.name not in allowed for path in output.iterdir())):
            raise FileExistsError('IMU analysis 已存在或來源不同，保留既有證據')
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'manifest.json',dict(status='in_progress',sources=sources))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        fig,axes=plt.subplots(4,2,figsize=(13,12),sharex=True)
        for j,label in enumerate(('x','y','z')):
            axes[0,0].plot(elapsed,innovation[:,j],'.-',label=label,lw=.8,markersize=2)
            axes[0,1].plot(elapsed,innovation[:,j+3],'.-',label=label,lw=.8,markersize=2)
            axes[2,1].plot(elapsed,ba[:,j],label=label,lw=.8)
            axes[3,0].plot(elapsed,bg[:,j],label=label,lw=.8)
        axes[0,0].set(ylabel='Position innovation [m]',title='Candidate registration - IMU prediction (bootstrap excluded)')
        axes[0,1].set(ylabel='Rotation innovation [rad]')
        axes[1,0].plot(elapsed,nis,'.-',label='NIS (registration gate passed)',lw=.8,markersize=3)
        axes[1,0].plot(elapsed,limits,'--',label='6 DoF chi-square gate',lw=.8)
        axes[1,0].set(ylabel='NIS')
        axes[1,1].step(elapsed,[bool(r['registration_accepted']) for r in rows],where='post',label='Registration gate',lw=.8)
        axes[1,1].step(elapsed,[u['measurement_accepted'] for u in upd],where='post',label='Kalman accepted',lw=.8)
        axes[1,1].set(ylabel='Decision [0/1]',ylim=(-.1,1.3))
        axes[2,0].plot(elapsed,np.linalg.norm(velocity,axis=1),label='IMU-origin speed',lw=.8)
        axes[2,0].set(ylabel='Estimated speed [m/s]')
        axes[2,1].set(ylabel='Accel bias [m/s²]')
        axes[3,0].set(ylabel='Gyro bias [rad/s]')
        axes[3,1].plot(elapsed,np.sqrt(cov[:,:3]).max(axis=1),label='Max position std',lw=.8)
        axes[3,1].set(ylabel='Position std [m]')
        fig.suptitle('v5 IMU fusion diagnostics; no ground truth / no extrinsic calibration')
        for ax in axes.flat:ax.grid(alpha=.3);ax.legend(fontsize=7)
        for ax in axes[-1]:ax.set_xlabel('Time since first PLY [s]')
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(output/f'01_fusion.{ext}',dpi=160)
        plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
        axes[0].plot(elapsed,age_ms,lw=.8);axes[0].set(ylabel='Latest past IMU age [ms]')
        axes[1].plot(elapsed,[r['prediction_ms'] for r in rows],label='Read/decode + predict')
        axes[1].plot(elapsed,[r['measurement_update_ms'] for r in rows],label='Measurement update')
        axes[1].set(xlabel='Time since first PLY [s]',ylabel='CPU time [ms]');axes[1].legend()
        for ax in axes:ax.grid(alpha=.3)
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(output/f'02_input_timing.{ext}',dpi=160)
        plt.close(fig)
        finite_nis=nis[np.isfinite(nis)]
        metrics=dict(frames=len(rows),map_accepted=sum(u['measurement_accepted'] for u in upd),
                     nis_rejections=sum(r['registration_reason']=='kalman_nis_gate' for r in rows),
                     nis_median=None if not len(finite_nis) else float(np.median(finite_nis)),
                     max_sample_age_ms=float(age_ms.max()),
                     final_accel_bias_mps2=ba[-1].tolist(),final_gyro_bias_radps=bg[-1].tolist(),
                     extrinsic_status=json.loads((run/'imu_sources.json').read_text())['extrinsic_status'],
                     caveat='Diagnostic consistency only; accepted fraction, NIS and bias do not prove accuracy or calibrate extrinsics.')
        write_json(output/'metrics.json',metrics)
        (output/'README.md').write_text('v5 已結束 run 的唯讀 IMU 診斷。首筆 bootstrap 殘差已排除。\n'
            '圖示殘差、NIS、濾波決策、速度、bias、不確定度與輸入年齡；無 ground truth，不能單憑較小殘差或較高接受率認定外參正確。\n'
            '來源 SHA 在 manifest.json；pose、CSV 與原始資料均未修改。\n')
        if any(sha256(run/name)!=digest for name,digest in sources.items()):
            raise RuntimeError('Source changed during IMU analysis')
        write_json(output/'manifest.json',dict(status='success',sources=sources,
                   script_sha256=sha256(Path(__file__)),elapsed_seconds=time.perf_counter()-started,
                   outputs={p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}))
        return dict(status='success',analysis_dir=str(output),elapsed_seconds=time.perf_counter()-started)
    finally:plt.close('all')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True,type=Path)
    a=p.parse_args();run=a.run_dir.resolve()
    try:
        result=analyze(run)
    except Exception as error:
        result=dict(status='failed',failure=f'{type(error).__name__}: {error}')
        if run.is_dir():
            with (run/'imu_analysis.log').open('a') as out:out.write(traceback.format_exc())
            previous=run/'imu_analysis_status.json'
            if not previous.exists() or json.loads(previous.read_text()).get('status')!='success':
                write_json(previous,result)
        print(json.dumps(result,ensure_ascii=False));return 1
    write_json(run/'imu_analysis_status.json',result)
    print(json.dumps(result,ensure_ascii=False));return 0


if __name__=='__main__':sys.exit(main())
