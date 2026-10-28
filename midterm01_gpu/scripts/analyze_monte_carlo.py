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
    local_search=info.get('monte_carlo_mode')=='causal_local_search_not_calibrated_posterior'
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
    ax[0].plot(x,values('ess'),lw=.7);ax[0].set_ylabel('Search score ESS' if local_search else 'ESS before resampling');ax[0].set_ylim(0,n*1.05)
    if not local_search:
        threshold=json.loads((run/'parameters.json').read_text())['monte_carlo']['ess_fraction']*n
        ax[0].axhline(threshold,color='tab:red',ls='--',lw=.8,label='resampling threshold');ax[0].legend()
    ax[1].plot(x,values('max_weight'),lw=.7);ax[1].set_ylabel('Maximum weight');ax[1].set_xlabel('Elapsed bag time (s)')
    fig.suptitle(('Local search score concentration; not calibrated probabilities' if local_search else 'Particle concentration; gaps = cloud update skipped'));save(fig,'01_weights_ess')
    fig,ax=plt.subplots(3,1,figsize=(10,7),sharex=True)
    ax[0].plot(x,values('selected_coverage'),lw=.7);ax[0].set_ylabel('Selected coverage')
    ax[1].plot(x,values('valid_particles'),lw=.7);ax[1].set_ylabel('Valid particles')
    ax[2].step(x,[u['cloud']['applied'] if local_search else u['resampled'] for u in updates],where='post',lw=.7,label='candidate score evaluated' if local_search else 'resampled')
    ax[2].step(x,[u['measurement_accepted'] for u in updates],where='post',lw=.7,alpha=.6,label='map accepted')
    ax[2].legend();ax[2].set_xlabel('Elapsed bag time (s)');save(fig,'02_support_updates' if local_search else '02_support_resampling')
    fig,ax=plt.subplots(figsize=(10,4))
    mc=np.array([r['monte_carlo_ms'] for r in rows]);pred=np.array([r['prediction_ms'] for r in rows])
    ax.plot(x,mc,lw=.7,label='particle cloud likelihood')
    ax.plot(x,pred,lw=.7,label='motion prediction')
    ax.set_xlabel('Elapsed bag time (s)');ax.set_ylabel('Wall time (ms)');ax.legend();save(fig,'03_particle_timing')
    bias_filter=bool(info.get('kalman_used'))
    bias_updates=0
    if bias_filter:
        if info.get('kalman_scope')!='stationary_gyro_bias_only_no_pose_fusion' or info.get('pose_kalman_used') is not False:
            raise ValueError('Unsupported Kalman scope')
        diagnostics=[r['inertial']['prediction']['gyro_bias_filter'] for r in rows]
        bias=np.array([d['bias_radps'] for d in diagnostics]);std=np.array([d['bias_std_radps'] for d in diagnostics])
        bias_updates=sum(d['updated'] for d in diagnostics)
        fig,ax=plt.subplots(3,1,figsize=(10,7),sharex=True)
        for j,label in enumerate(('IMU X','IMU Y','IMU Z')):
            ax[0].plot(x,bias[:,j],lw=.8,label=label)
            ax[1].plot(x,std[:,j],lw=.8,label=label)
        ax[0].set_ylabel('Gyro bias (rad/s)');ax[0].legend()
        ax[1].set_ylabel('Bias model std (rad/s)');ax[1].set_yscale('log')
        ax[2].step(x,[d['updated'] for d in diagnostics],where='post',lw=.8)
        ax[2].set_ylabel('Stationary bias update');ax[2].set_xlabel('Elapsed bag time (s)')
        fig.suptitle('Causal gyro-bias Kalman only; no pose Kalman or accuracy truth');save(fig,'04_gyro_bias')
    metrics=dict(frames=len(rows),particles=n,mode='causal_local_search' if local_search else 'v8_particle_filter',scores_are_probabilities=False,attempted_cloud_updates=sum(c['applied'] for c in clouds),
                 committed_cloud_updates=sum(c.get('committed',c['applied'] and u['measurement_accepted']) for c,u in zip(clouds,updates)),
                 resampling_frames=sum(u['resampled'] for u in updates),
                 cloud_wall_ms_median=float(np.median(mc)),cloud_wall_ms_p95=float(np.percentile(mc,95)),
                 accuracy_claim=False,kalman_used=bias_filter,pose_kalman_used=False,gyro_bias_update_frames=bias_updates)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');files.append(out/'metrics.json')
    if before!={name:digest(run/name) for name in names}:raise RuntimeError('Source evidence changed during analysis')
    explanation=('v8.1 每幀從已提交狀態產生新的局部候選，不跨幀攜帶 posterior，也沒有重採樣步驟。'
                 'ESS／最大權重僅描述當幀搜尋分數集中程度，不是機率校準或精度。'
                 if local_search else
                 'v8 ESS 是嘗試更新後、重採樣前的值；配準拒絕時會回復權重。')
    if bias_filter:
        explanation+='另圖記錄靜止時的 gyro-bias Kalman 更新；std 是未校正模型的不確定性，不是實際誤差。它不再融合 map pose。'
    (out/'README.md').write_text('# Monte Carlo 診斷\n\n'
        '這些圖檢查候選集中、覆蓋、更新及耗時；沒有 ground truth，不能當成精度證明。'
        '空白區表示該幀未評估 cloud score（包含首幀、靜止及候選支持不足）。'+explanation+'\n\n'
        '幾何精修不是另一筆獨立量測。來源與 SHA256 見 manifest.json；不修改 pose、CSV 或參數。\n')
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
