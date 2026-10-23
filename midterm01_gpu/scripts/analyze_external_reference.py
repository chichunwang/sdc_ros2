#!/usr/bin/env python3
"""Compare existing trajectories offline; external predictions are never truth.

Analysis-only, non-submission diagnostics. Reported scores are source metadata
only. The 2026-10-22 updates prohibit probing hidden truth through leaderboard
scores; legacy crossover and unknown score-component inference options fail.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from accuracy.benchmark import create_run_directory
from analyze_height_score import angle_difference, checked_run, run_diagnostics, statistics
from output_layout import saved_path
from three_camera_source import sha256


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def read_prediction(path, count):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['idx', 'x', 'y', 'yaw']:
            raise ValueError(f'Unexpected CSV columns: {path}')
        rows = list(reader)
    if len(rows) != count or any(int(row['idx']) != i for i, row in enumerate(rows)):
        raise ValueError(f'CSV row count or index mismatch: {path}')
    values = np.array([[float(row[k]) for k in ('x', 'y', 'yaw')] for row in rows])
    if not np.isfinite(values).all():
        raise ValueError(f'Nonfinite CSV: {path}')
    return rows, values


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values)**2)))


def time_median(times, values, window_seconds):
    """Centered physical-time window, with truncated endpoint support."""
    left = np.searchsorted(times, times-window_seconds/2, side='left')
    right = np.searchsorted(times, times+window_seconds/2, side='right')
    return np.array([np.median(values[a:b], axis=0) for a, b in zip(left, right)])


def motion_diagnostic(times, values, window_seconds):
    # Resample before SG derivatives: image intervals are not exactly uniform.
    step = .02
    grid = np.arange(times[0], times[-1], step)
    xy = np.column_stack([np.interp(grid, times, values[:, k]) for k in (0, 1)])
    yaw = np.interp(grid, times, np.unwrap(values[:, 2]))
    width = int(round(window_seconds/step))//2*2+1
    velocity = savgol_filter(xy, width, 3, deriv=1, delta=step, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    course = np.arctan2(velocity[:, 1], velocity[:, 0])
    angle = angle_difference(yaw, course)
    lateral = -np.sin(yaw)*velocity[:, 0]+np.cos(yaw)*velocity[:, 1]
    eligible = (speed > 5.) & (grid > 2.) & (grid < times[-1]-2.)
    if not eligible.any():
        raise ValueError('No moving interior samples for motion diagnostic')
    return dict(window_seconds=window_seconds, eligible_samples=int(eligible.sum()),
                yaw_minus_own_course_deg=statistics(angle[eligible]),
                lateral_velocity_mps=statistics(lateral[eligible])), (grid, np.where(eligible,angle,np.nan), np.where(eligible,lateral,np.nan))


RULES_REFERENCE = 'doc/Competition Regulations/01_Mid01_Updates_Rules.pdf (2026-10-22), pp. 1, 3'
DISABLED_SCORE_PROBING = (
    '依 2026-10-22 因果定位公告（01_Mid01_Updates_Rules.pdf，第 1、3 頁），'
    '本工具已停用交叉提交與由榜分推算未知誤差分項；僅保留歷史檔，不再生成。'
)


def reject_score_probing(args):
    """Reject legacy probing options before reading inputs or creating a run."""
    if (getattr(args, 'write_crossovers', False)
            or getattr(args, 'reported_reference_yaw_rmse', None) is not None):
        raise ValueError(DISABLED_SCORE_PROBING)


def analyze(args):
    reject_score_probing(args)
    data = checked_run(args.baseline_run)
    run, _, poses, frames, prior = data
    count = len(poses)
    if data[1]['track'] != 1:
        raise ValueError('This offline analysis entry currently supports Track1 only')
    baseline_file = run/'diagnostic.csv'
    _, base = read_prediction(baseline_file, count)
    _, external = read_prediction(args.reference, count)
    info_file = args.experiment/'test_info.json'
    if not info_file.is_file() or (args.experiment/'comparison.json').exists():
        raise ValueError('Use a unique analysis run; preserve previous report')
    inputs = [args.reference, baseline_file, run/'result.txt', run/'sensor_prior.csv', run/'frames.csv']
    before = {str(p):sha256(p) for p in inputs}
    if (args.experiment/args.reference.name).resolve() != args.reference.resolve():
        preserved = args.experiment/args.reference.name
        if preserved.exists():
            if sha256(preserved) != sha256(args.reference):
                raise ValueError('Preserved external CSV differs')
        else:
            shutil.copy2(args.reference, preserved)
    ns = np.array([int(r['timestamp_ns']) for r in frames], dtype=np.int64)
    times = (ns-ns[0])/1e9
    delta = external[:, :2]-base[:, :2]
    dyaw = angle_difference(external[:, 2], base[:, 2])
    pxyz = np.array([[float(r[k]) for k in ('x','y','z')] for r in prior])
    pq = np.array([[float(r[k]) for k in ('qx','qy','qz','qw')] for r in prior])
    pyaw = Rotation.from_quat(pq).as_euler('xyz')[:, 2]
    slow = time_median(times, delta, 12.)
    # A best-fitting global SE(2) is descriptive only, never applied to submissions.
    x, y = base[:,:2]-base[:,:2].mean(0), external[:,:2]-external[:,:2].mean(0)
    u, _, vt = np.linalg.svd(x.T@y)
    correction = np.diag([1., np.linalg.det(u@vt)])
    rotation = u@correction@vt
    rigid_error = external[:,:2]-(x@rotation+external[:,:2].mean(0))
    ranges = [('first_12s', times < 12.), ('interior', (times >= 12.) & (times <= times[-1]-12.)),
              ('last_12s', times > times[-1]-12.)]
    segments = [dict(segment=name, rows=int(mask.sum()), xy_version_difference_m=statistics(np.linalg.norm(delta[mask],axis=1)),
                     yaw_version_difference_deg=statistics(dyaw[mask])) for name,mask in ranges]
    motions = []; curves = {}
    for label, values in (('ours',base),('external',external)):
        for seconds in (1.,2.,4.):
            record, values_derived = motion_diagnostic(times,values,seconds)
            motions.append(dict(source=label,**record))
            if seconds == 2.: curves[label] = values_derived
    report = dict(analysis_only=True, non_submission=True, rules_reference=RULES_REFERENCE,
        sources=dict(ours=str(baseline_file), external=str(args.reference), sha256=before),
        rows=count, idx_exact_match=True, csv_finite=True, reference_algorithm_reproduced=False,
        time_basis='Assumes external idx maps to our validated same Track1 frame sequence; external CSV has no timestamps',
        time_span_s=float(times[-1]), input_yaw_unit='radians; comparisons converted to degrees',
        reported_scores=dict(ours=args.baseline_score,external=args.reference_score,uploaded_bytes_independently_verified=False),
        baseline=run_diagnostics(data),
        external_minus_ours=dict(xy_vector_m=statistics(np.linalg.norm(delta,axis=1)),
            dx_m=statistics(delta[:,0]),dy_m=statistics(delta[:,1]),yaw_deg=statistics(dyaw)),
        prior_comparison=dict(ours_minus_motion_prior_xy_m=statistics(np.linalg.norm(base[:,:2]-pxyz[:,:2],axis=1)),
            external_minus_motion_prior_xy_m=statistics(np.linalg.norm(external[:,:2]-pxyz[:,:2],axis=1)),
            ours_minus_motion_prior_yaw_deg=statistics(angle_difference(base[:,2],pyaw)),
            external_minus_motion_prior_yaw_deg=statistics(angle_difference(external[:,2],pyaw))),
        constant_correction_diagnostics=dict(mean_delta_xy_m=delta.mean(0).tolist(),
            residual_after_constant_offset_m=statistics(np.linalg.norm(delta-delta.mean(0),axis=1)),
            fitted_global_rigid_rotation_deg=float(np.degrees(np.arctan2(rotation[0,1],rotation[0,0]))),
            residual_after_global_rigid_fit_m=statistics(np.linalg.norm(rigid_error,axis=1)),applied=False),
        slow_correction_diagnostic=dict(centered_window_seconds=12., estimator='coordinate-wise median, truncated endpoints',
            slow_component_rms_m=rms(np.linalg.norm(slow,axis=1)),
            remaining_component_rms_m=rms(np.linalg.norm(delta-slow,axis=1)),
            interpretation='Descriptive external-minus-ours correction, not an identified GNSS bias, source filter recovery or calibration'),
        segments=segments, motion_consistency=motions,
        motion_consistency_limits='Yaw versus own velocity uses only output trajectories; physical forward-axis/sideslip/lever-arm assumptions apply. It is not yaw truth. SG derivative windows 1/2/4s are fixed diagnostic sensitivities, not algorithm tuning.',
        limits=['No external scales, map correspondences, ICP success masks or factor graph states were supplied.',
            'User method narrative is an unverified source description, not recoverable from CSV.',
            'Reported scores are metadata only; no hidden error component is inferred.',
            'External predictions are not used to calibrate our production trajectory, fit targets or select solver parameters.'])
    with (args.experiment/'per_frame_difference.csv').open('x',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['idx','timestamp_ns','relative_time_s','delta_x_m','delta_y_m','delta_xy_m','delta_yaw_deg','median12_delta_x_m','median12_delta_y_m'])
        writer.writerows((i,int(stamp),t,*d,float(np.linalg.norm(d)),float(h),*s)
            for i,(stamp,t,d,h,s) in enumerate(zip(ns,times,delta,dyaw,slow)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,2,figsize=(13,8),layout='constrained')
    ax=axes[0,0]; ax.plot(times,delta[:,0],label='delta x'); ax.plot(times,delta[:,1],label='delta y')
    ax.plot(times,slow[:,0],color='C0',ls='--',lw=2,label='12s median x');ax.plot(times,slow[:,1],color='C1',ls='--',lw=2,label='12s median y')
    ax.set(title='External minus ours: position',xlabel='Time (s)',ylabel='Difference (m)');ax.legend(ncol=2,fontsize=9)
    ax=axes[0,1];ax.plot(times,dyaw,color='C2');ax.axhline(0,color='gray',lw=.8)
    ax.set(title='External minus ours: yaw',xlabel='Time (s)',ylabel='Difference (degrees)')
    for label,color in (('ours','C3'),('external','C0')):
        g,a,l=curves[label]
        for ax,values in ((axes[1,0],a),(axes[1,1],l)):
            ax.plot(g,values,label=label,color=color,lw=1)
    axes[1,0].set(title='Yaw minus own course (2s derivative, moving interior)',xlabel='Time (s)',ylabel='Angle (degrees)')
    axes[1,1].set(title='Velocity perpendicular to own yaw (moving interior)',xlabel='Time (s)',ylabel='Velocity (m/s)')
    for ax in axes[1,:]: ax.legend()
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Track1 CSV comparison: prediction differences, not truth errors',fontsize=15)
    fig.savefig(args.experiment/'comparison.png',dpi=160);fig.savefig(args.experiment/'comparison.svg');plt.close(fig)
    after={str(p):sha256(p) for p in inputs}
    if before!=after: raise ValueError('Input evidence changed')
    dump(args.experiment/'comparison.json',report)
    dump(args.experiment/'evidence_sha256_unchanged.json',dict(before=before,after=after,unchanged=True))
    snapshot=args.experiment/'sources';snapshot.mkdir()
    for source in (Path(__file__),Path(__file__).with_name('analyze_height_score.py')):shutil.copy2(source,snapshot/source.name)
    dump(snapshot/'sha256.json',{p.name:sha256(p) for p in snapshot.glob('*.py')})
    info=json.loads(info_file.read_text());info.update(status='complete',analysis_only=True,non_submission=True,rules_reference=RULES_REFERENCE,localization_executed=False,
        external_method_description='User supplied narrative; original implementation, per-frame scales, ICP and truth unavailable',
        production_algorithm_changed=False,reference_predictions_used_as_training_truth=False,
        comparison_report='comparison.json',versions=dict(python=sys.version,numpy=np.__version__),
        command=sys.argv)
    dump(info_file,info)
    print(json.dumps(dict(analysis_only=True,non_submission=True,delta=report['external_minus_ours'],slow=report['slow_correction_diagnostic']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=saved_path,required=True)
    parser.add_argument('--reference-score',type=float,required=True,help='Reported source score; metadata only')
    parser.add_argument('--baseline-run',type=saved_path,required=True)
    parser.add_argument('--baseline-score',type=float,required=True,help='Reported source score; metadata only')
    parser.add_argument('--experiment',type=saved_path)
    parser.add_argument('--reported-reference-yaw-rmse',type=float,help='Disabled legacy option: unknown score-component inference is prohibited')
    parser.add_argument('--write-crossovers',action='store_true',help='Disabled legacy option: historical files only; generates no submission CSV')
    args=parser.parse_args()
    try:
        reject_score_probing(args)
    except ValueError as exc:
        parser.error(str(exc))
    for value in (args.reference_score,args.baseline_score):
        if value is not None and (not np.isfinite(value) or value<0):parser.error('Scores/RMSE must be finite and nonnegative')
    root=Path(__file__).resolve().parents[1]
    args.reference=args.reference.resolve();args.baseline_run=args.baseline_run.resolve()
    if args.experiment is None:args.experiment=create_run_directory(root/'outputs/run','external_scored_csv_analysis',track=1)
    args.experiment=args.experiment.resolve();args.experiment.relative_to((root/'outputs/run').resolve())
    analyze(args)
