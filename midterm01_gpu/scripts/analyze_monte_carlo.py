#!/usr/bin/env python3
"""Read-only particle diagnostics after a successful v8 full, never re-localize."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

FRAME_COUNT=3142


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(run):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    start=time.perf_counter();run=Path(run)
    names=['test_info.json','observations.jsonl','parameters.json','result_3.csv','poses.txt']
    before={name:digest(run/name) for name in names}
    info=json.loads((run/'test_info.json').read_text())
    if info.get('status')!='success' or info.get('mode')!='full' or info.get('frame_count')!=FRAME_COUNT:
        raise ValueError('Monte Carlo full analysis requires a successful complete full run')
    if info.get('prediction')!='gpu_monte_carlo':raise ValueError('Expected v8 Monte Carlo diagnostics')
    for name in ('observations.jsonl','result_3.csv','poses.txt'):
        if info['output_sha256'].get(name)!=before[name]:raise ValueError(f'Output SHA mismatch: {name}')
    rows=[json.loads(line) for line in (run/'observations.jsonl').read_text().splitlines()]
    if len(rows)!=FRAME_COUNT or [r['idx'] for r in rows]!=list(range(FRAME_COUNT)):
        raise ValueError('Observation frame coverage mismatch')
    stamps=[r['timestamp_ns'] for r in rows]
    if any(type(t) is not int for t in stamps) or any(b<=a for a,b in zip(stamps,stamps[1:])):
        raise ValueError('Expected strictly increasing integer timestamp_ns')
    updates=[r['monte_carlo'] for r in rows]
    if any(u['kalman_used'] or u['posterior_recentered_by_refinement'] for u in updates):
        raise ValueError('Incompatible particle update contract')
    clouds=[u['cloud'] for u in updates]
    def values(key):return np.array([c.get(key,np.nan) for c in clouds],dtype=float)
    x=np.array([(t-stamps[0])*1e-9 for t in stamps]);n=updates[0]['particles']
    out=run/'analysis/monte_carlo';out.mkdir(parents=True,exist_ok=True)
    files=[]
    def save(fig,stem):
        fig.tight_layout()
        for suffix in ('png','pdf'):
            p=out/f'{stem}.{suffix}';fig.savefig(p,dpi=150);files.append(p)
        plt.close(fig)
    fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True)
    ax[0].plot(x,values('ess'),lw=.7);ax[0].set_ylabel('ESS before resampling');ax[0].set_ylim(0,n*1.05)
    threshold=json.loads((run/'parameters.json').read_text())['monte_carlo']['ess_fraction']*n
    ax[0].axhline(threshold,color='tab:red',ls='--',lw=.8,label='resampling threshold');ax[0].legend()
    ax[1].plot(x,values('max_weight'),lw=.7);ax[1].set_ylabel('Maximum weight');ax[1].set_xlabel('Elapsed bag time (s)')
    fig.suptitle('Particle concentration; gaps = cloud update skipped');save(fig,'01_weights_ess')
    fig,ax=plt.subplots(3,1,figsize=(10,7),sharex=True)
    ax[0].plot(x,values('selected_coverage'),lw=.7);ax[0].set_ylabel('Selected coverage')
    ax[1].plot(x,values('valid_particles'),lw=.7);ax[1].set_ylabel('Valid particles')
    ax[2].step(x,[u['resampled'] for u in updates],where='post',lw=.7,label='resampled')
    ax[2].step(x,[u['measurement_accepted'] for u in updates],where='post',lw=.7,alpha=.6,label='map accepted')
    ax[2].legend();ax[2].set_xlabel('Elapsed bag time (s)');save(fig,'02_support_resampling')
    fig,ax=plt.subplots(figsize=(10,4))
    mc=np.array([r['monte_carlo_ms'] for r in rows]);pred=np.array([r['prediction_ms'] for r in rows])
    ax.plot(x,mc,lw=.7,label='particle cloud likelihood')
    ax.plot(x,pred,lw=.7,label='motion prediction')
    ax.set_xlabel('Elapsed bag time (s)');ax.set_ylabel('Wall time (ms)');ax.legend();save(fig,'03_particle_timing')
    metrics=dict(frames=len(rows),particles=n,attempted_cloud_updates=sum(c['applied'] for c in clouds),
                 committed_cloud_updates=sum(c.get('committed',c['applied'] and u['measurement_accepted']) for c,u in zip(clouds,updates)),
                 resampling_frames=sum(u['resampled'] for u in updates),
                 cloud_wall_ms_median=float(np.median(mc)),cloud_wall_ms_p95=float(np.percentile(mc,95)),
                 accuracy_claim=False,kalman_used=False)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');files.append(out/'metrics.json')
    if before!={name:digest(run/name) for name in names}:raise RuntimeError('Source evidence changed during analysis')
    (out/'README.md').write_text('# v8 Monte Carlo 診斷\n\n'
        '這些圖檢查粒子集中、覆蓋率、重採樣及耗時；沒有 ground truth，不能當成精度證明。'
        '空白區表示該幀未評估 cloud likelihood（包含首幀、靜止及候選支持不足）。'
        'ESS 是嘗試更新後、重採樣前的值；配準拒絕時會回復權重，嘗試與實際採用次數分別記於 metrics.json。\n\n'
        'NDT／ICP 精修只產生當幀輸出，不將同一地圖觀測再次當作獨立量測。'
        '來源與 SHA256 見 manifest.json；不修改 pose、CSV 或參數。\n')
    files.append(out/'README.md')
    manifest=dict(source_sha256=before,script_sha256=digest(Path(__file__)),
                  output_sha256={p.name:digest(p) for p in files},accuracy_claim=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    status=dict(status='success',elapsed_seconds=time.perf_counter()-start,outputs=str(out))
    (run/'monte_carlo_analysis_status.json').write_text(json.dumps(status,indent=2)+'\n')
    return metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
    try:analyze(a.run_dir);return 0
    except Exception as error:
        (a.run_dir/'monte_carlo_analysis_status.json').write_text(json.dumps(dict(status='failed',failure=f'{type(error).__name__}: {error}'),indent=2)+'\n')
        (a.run_dir/'monte_carlo_analysis_error.log').write_text(traceback.format_exc())
        return 1


if __name__=='__main__':raise SystemExit(main())
