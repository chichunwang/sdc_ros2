#!/usr/bin/env python3
"""唯讀重播既有候選以檢查 correction filter；不輸出可提交的 pose/CSV。

歷史 observations 沒有完整 data H；此重播明確使用品質膨脹後的量測
floor proxy。正式 v4 用當幀完整 data H，因此這不是 v4 定位/精度結果。
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from track2 import parameters as p
from track2.correction_filter import CausalCorrectionFilter, pose_correction, quality_inflation, wrap


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def check(run, evidence):
    start = time.perf_counter()
    output = run/'analysis'; output.mkdir(exist_ok=False)
    inputs = run/'inputs'; inputs.mkdir(exist_ok=False)
    evidence_info = json.loads((evidence/'jitter_summary.json').read_text())
    source_hashes = {}
    for name in ('pose_trace.jsonl', 'observations.jsonl'):
        path = evidence/'inputs'/name
        expected = evidence_info['sources'][name]['prefix_sha256']
        if sha(path) != expected:
            raise ValueError('Historical prefix SHA mismatch: '+name)
        shutil.copyfile(path, inputs/name)
        source_hashes[name] = dict(path=str(path), sha256=expected)
    traces = [json.loads(s) for s in (inputs/'pose_trace.jsonl').read_text().splitlines()]
    observations = [json.loads(s) for s in (inputs/'observations.jsonl').read_text().splitlines()]
    if len(traces) != len(observations) or len(traces) != evidence_info['prefix_frames']:
        raise ValueError('Mismatched historical prefix lengths')
    base_R = np.diag([p.CORRECTION_MEAS_POSITION_FLOOR_M**2]*2+
                     [np.deg2rad(p.CORRECTION_MEAS_YAW_FLOOR_DEG)**2])
    filt = CausalCorrectionFilter(); records = []
    for trace, obs in zip(traces, observations):
        if trace['idx'] != obs['idx'] or trace['timestamp_ns'] != obs['timestamp_ns']:
            raise ValueError('Historical frame association mismatch')
        prior = np.asarray(trace['predicted']); old = np.asarray(trace['final'])
        available = bool(trace['map_accepted'])
        old_c = pose_correction(old, prior)
        speed = float(np.linalg.norm(obs['sensor_prior']['velocity_xy_mps']))
        filt.predict(prior, int(obs['camera_timestamp_ns']), speed)
        R = base_R*quality_inflation(obs['registration']) if available else None
        c, diag = filt.update(old_c if available else None, R)
        records.append(dict(idx=trace['idx'], camera_timestamp_ns=obs['camera_timestamp_ns'],
            old_correction=old_c.tolist(), replay_correction=c.tolist(), candidate_available=available,
            correction_filter=diag, covariance_model='quality-inflated floors ONLY; historical H unavailable'))
    with (run/'correction_replay.jsonl').open('x') as f:
        for row in records: f.write(json.dumps(row, allow_nan=False)+'\n')
    old_c = np.array([r['old_correction'] for r in records])
    new_c = np.array([r['replay_correction'] for r in records])
    old_xy = np.array([t['final'] for t in traces])[:, :2, 3]
    prior_xy = np.array([t['predicted'] for t in traces])[:, :2, 3]
    # These arrays exist only to display a filter mechanism diagnostic. They
    # are never exported as a trajectory or consumed by a submission runner.
    implied_xy = prior_xy+new_c[:, :2]
    old_step = np.r_[0., np.linalg.norm(np.diff(old_xy, axis=0), axis=1)]
    new_step = np.r_[0., np.linalg.norm(np.diff(implied_xy, axis=0), axis=1)]
    prior_step = np.r_[0., np.linalg.norm(np.diff(prior_xy, axis=0), axis=1)]
    events = []
    def save(fig, stem):
        fig.tight_layout()
        for ext in ('png', 'pdf'): fig.savefig(output/f'{stem}.{ext}', dpi=170)
        plt.close(fig)
    for center in (428, 653):
        lo, hi = max(1, center-20), min(len(records), center+21)
        idx = np.arange(lo, hi)
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        axes[0].plot(idx, prior_step[lo:hi], label='Sensor prior')
        axes[0].plot(idx, old_step[lo:hi], label='Historical v3 output')
        axes[0].plot(idx, new_step[lo:hi], label='Replay-implied step (proxy R)')
        axes[0].set(ylabel='XY step [m]', title=f'Frame {center}: correction mechanism replay; no GT, no registration rerun')
        axes[0].legend(fontsize=8)
        for j, label in enumerate(('X', 'Y')):
            axes[1].plot(idx, old_c[lo:hi, j], ':', label=f'Old {label}')
            axes[1].plot(idx, new_c[lo:hi, j], label=f'Replay {label}')
        axes[1].set_ylabel('Correction [m]'); axes[1].legend(ncol=4, fontsize=8)
        axes[2].plot(idx, np.degrees(old_c[lo:hi, 2]), ':', label='Old')
        axes[2].plot(idx, np.degrees(new_c[lo:hi, 2]), label='Replay')
        axes[2].set_ylabel('Yaw correction [deg]'); axes[2].legend(fontsize=8)
        axes[3].step(idx, [int(r['candidate_available']) for r in records[lo:hi]], where='mid', label='Old geometry accepted')
        axes[3].step(idx, [int(r['correction_filter']['accepted']) for r in records[lo:hi]], where='mid', label='Replay update used')
        axes[3].set(xlabel='Historical frame idx', ylabel='Accepted', yticks=[0, 1], ylim=(-.1, 1.15))
        axes[3].legend(fontsize=8)
        for ax in axes: ax.axvline(center, color='gray', alpha=.5); ax.grid(alpha=.25)
        save(fig, f'01_frame_{center:04d}_correction_replay')
        events.append(dict(idx=center, update_reason=records[center]['correction_filter']['reason'],
            old_xy_step_m=float(old_step[center]), replay_implied_xy_step_m=float(new_step[center]),
            prior_xy_step_m=float(prior_step[center]),
            old_correction_step_m=float(np.linalg.norm(old_c[center, :2]-old_c[center-1, :2])),
            replay_correction_step_m=float(np.linalg.norm(new_c[center, :2]-new_c[center-1, :2]))))
    synthetic = CausalCorrectionFilter(); synthetic_rows = []
    T = np.eye(4)
    for k in range(321):
        synthetic.predict(T, 1_000_000_000+k*50_000_000, 8.)
        c, d = synthetic.update(np.array([.4, -.6, .02]) if k < 20 else None, base_R)
        synthetic_rows.append(dict(seconds=k*.05, correction=c.tolist(), correction_filter=d))
    with (run/'synthetic_dropout.jsonl').open('x') as f:
        for row in synthetic_rows: f.write(json.dumps(row, allow_nan=False)+'\n')
    seconds = np.array([r['seconds'] for r in synthetic_rows])-synthetic_rows[19]['seconds']
    norm = np.array([np.linalg.norm(r['correction'][:2]) for r in synthetic_rows])
    std = np.array([np.sqrt(np.trace(np.array(r['correction_filter']['covariance'])[:2, :2])) for r in synthetic_rows])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(seconds, norm); axes[0].set(ylabel='XY correction norm [m]', title='Generated dropout: hold, then exponential relaxation')
    axes[1].plot(seconds, std); axes[1].set(xlabel='Seconds since last accepted map update', ylabel='XY combined std [m]')
    for ax in axes:
        ax.axvline(0, color='gray', ls=':'); ax.axvline(p.CORRECTION_HOLD_SECONDS, color='tab:red', ls='--'); ax.grid(alpha=.25)
    save(fig, '02_generated_dropout')
    summary = dict(status='completed', kind='correction_filter_diagnostic_only', version=p.VERSION,
        sources=source_hashes, prefix_frames=len(records), events=events,
        decisions=dict(Counter(r['correction_filter']['reason'] for r in records)),
        real_registration_executed=False, submission_outputs_created=False, ground_truth_available=False,
        covariance_limitation='Historical H unavailable: proxy R from floors and quality; production v4 also uses full data H',
        solver_limitation='Historical v3 candidates are fixed; v4 seeding/height feedback can change future candidates',
        elapsed_seconds=time.perf_counter()-start)
    write(run/'correction_summary.json', summary)
    artifacts = ('correction_replay.jsonl', 'synthetic_dropout.jsonl', 'correction_summary.json')
    write(output/'manifest.json', dict(kind=summary['kind'], sources=source_hashes,
        script_sha256=sha(Path(__file__)), artifacts={n: sha(run/n) for n in artifacts},
        real_registration_executed=False, submission_outputs_created=False))
    (output/'README.md').write_text('# Track2 修正記憶機制檢查\n\n'
        '01 使用已凍結的 888 幀 v3 候選，按時間向前重播 correction filter；沒有讀取 bag/PLY、沒有重新配準。'
        '02 是生成資料的短暫保留／長時間衰減。PNG/PDF 位於本目錄。\n\n'
        '歷史資料沒有完整 data Hessian，因此重播只用量測 floor 與品質膨脹作 proxy covariance，'
        '正式 v4 會另用當幀完整 data H。舊候選也不會反映 v4 的新 seed 或高度回饋。'
        '圖上變平順僅證明機制行為，不是 v4 full 精度改善；沒有真值，不產生或修改任何提交軌跡。\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--evidence-run', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.run_dir.resolve(), args.evidence_run.resolve()), ensure_ascii=False, indent=2))
