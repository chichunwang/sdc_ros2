#!/usr/bin/env python3
"""唯讀稽核 IMU 時序與資料範圍，不積分 pose、不執行定位或 GPU。"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from track3.imu import ImuReader
from track3.io import frames, sha256, write_json
from track3.imu_parameters import resolve_imu_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record-dir',required=True,type=Path)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'config/track3.json')
    a=p.parse_args();run=a.record_dir.resolve()
    if not (run/'source_manifest.json').exists() or (run/'test_info.json').exists():
        p.error('需要新的未使用 run（先執行 create_run.py --kind test --track 00）')
    started=time.perf_counter();reader=None
    cfg=json.loads(a.config.read_text());data=Path(cfg['data_dir'])
    cfg['imu']=icfg=resolve_imu_config(cfg['imu'])
    info=dict(status='running',version=cfg['version'],frame_count=0,score=None,score_source=None,
              test_type='read-only IMU and PLY clock audit; no pose estimation, no registration, no GPU')
    write_json(run/'parameters.json',cfg)
    write_json(run/'test_info.json',info)
    try:
        rows=frames(data/cfg['ply_subdir'],expected=cfg['expected_frames'])
        bag=data/icfg['bag_file']
        sources={str(path):sha256(path) for path in (bag,a.config)}
        reader=ImuReader(bag,icfg['topic'],icfg['frame_id'])
        held=None;coverage=[];samples=[]
        with (run/'imu_frame_coverage.csv').open('x',newline='') as out:
            writer=csv.writer(out)
            writer.writerow(['idx','original_idx','timestamp_ns','samples','latest_bag_timestamp_ns','latest_header_timestamp_ns','age_ns'])
            for idx,row in enumerate(rows):
                batch=reader.take_until(row['timestamp_ns']);samples.extend(batch)
                if batch:held=batch[-1]
                age=None if held is None else row['timestamp_ns']-held.timestamp_ns
                coverage.append((len(batch),age))
                writer.writerow([idx,row['original_idx'],row['timestamp_ns'],len(batch),
                                 None if held is None else held.timestamp_ns,
                                 None if held is None else held.header_timestamp_ns,age])
        arrivals=[s.timestamp_ns for s in samples]
        spacing=np.array([b-a for a,b in zip(arrivals,arrivals[1:])],dtype=np.int64)
        offsets=np.array([s.timestamp_ns-s.header_timestamp_ns for s in samples],dtype=np.int64)
        acceleration=np.array([s.linear_acceleration for s in samples])
        angular=np.array([s.angular_velocity for s in samples])
        def stats(values):
            return dict(min=float(np.min(values)),median=float(np.median(values)),p99=float(np.percentile(values,99)),max=float(np.max(values)))
        report=dict(sources=sources,frame_count=len(rows),imu=reader.summary(),
                    first_bag_timestamp_ns=arrivals[0],last_bag_timestamp_ns=arrivals[-1],
                    arrival_spacing_ms=stats(spacing*1e-6),bag_minus_header_ms=stats(offsets*1e-6),
                    frames_without_past_imu=sum(age is None for _,age in coverage),
                    frames_exceeding_configured_age=sum(age is None or age>icfg['max_sample_age_s']*1e9 for _,age in coverage),
                    sample_age_ms=stats([age*1e-6 for _,age in coverage if age is not None]),
                    acceleration_norm_mps2=stats(np.linalg.norm(acceleration,axis=1)),
                    angular_velocity_norm_radps=stats(np.linalg.norm(angular,axis=1)),
                    caveat='Input audit only. Norms do not calibrate axes, gravity, bias, lever arm or localization accuracy.')
        write_json(run/'imu_inspection.json',report)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        analysis=run/'analysis';analysis.mkdir()
        fig,axes=plt.subplots(3,1,figsize=(10,8))
        axes[0].plot([np.nan if age is None else age*1e-6 for _,age in coverage],lw=.7)
        axes[0].axhline(icfg['max_sample_age_s']*1000,color='r',ls='--',label='Configured stale limit')
        axes[0].set(xlabel='PLY idx',ylabel='Latest past IMU age [ms]');axes[0].legend()
        axes[1].plot(offsets*1e-6,lw=.7);axes[1].set(xlabel='IMU sample idx',ylabel='Bag - header [ms]')
        axes[2].plot(np.linalg.norm(acceleration,axis=1),lw=.7);axes[2].set(xlabel='IMU sample idx',ylabel='Specific force norm [m/s²]')
        fig.suptitle('Read-only input audit; no localization / no calibration')
        for ax in axes:ax.grid(alpha=.3)
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(analysis/f'01_imu_input_audit.{ext}',dpi=160)
        plt.close(fig)
        (analysis/'README.md').write_text('IMU 原始輸入與 PLY 時序唯讀圖；沒有積分 pose 或重新配準，不能當成外參校正或定位精度證明。\n')
        write_json(analysis/'manifest.json',dict(sources=sources,script_sha256=sha256(Path(__file__)),
                   inputs={name:sha256(run/name) for name in ('imu_frame_coverage.csv','imu_inspection.json')},
                   outputs={p.name:sha256(p) for p in analysis.iterdir() if p.is_file()}))
        info.update(status='success',inspected_frames=len(rows),imu=reader.summary(),
                    output_sha256={name:sha256(run/name) for name in ('imu_frame_coverage.csv','imu_inspection.json')})
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return 0
    except BaseException as error:
        info.update(status='failed',failure=f'{type(error).__name__}: {error}')
        (run/'failure.log').write_text(traceback.format_exc())
        raise
    finally:
        if reader is not None:reader.close()
        info['elapsed_seconds']=time.perf_counter()-started
        write_json(run/'test_info.json',info)


if __name__=='__main__':sys.exit(main())
