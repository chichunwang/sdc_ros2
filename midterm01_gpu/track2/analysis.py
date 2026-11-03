"""唯讀 Track2 完整 full 圖表；與 Track3 analyze_run.py 隔離。"""
import csv
import json
from pathlib import Path
import numpy as np
from track3.io import sha256, write_json


def validate(run, expected=3299):
    run = Path(run)
    with (run/'result_2.csv').open() as f:
        reader = csv.reader(f)
        if next(reader) != ['idx', 'x', 'y', 'yaw']:
            raise ValueError('Unexpected result columns')
        result = np.array([[float(v) for v in row] for row in reader])
    poses = np.loadtxt(run/'poses.txt', ndmin=2)
    with (run/'timings.csv').open() as f:
        timing = list(csv.DictReader(f))
    with (run/'frame_manifest.csv').open() as f:
        manifest = list(csv.DictReader(f))
    if result.shape != (expected, 4) or poses.shape != (expected, 12) or len(timing) != expected or len(manifest) != expected:
        raise ValueError('Incomplete output')
    if not np.isfinite(result).all() or not np.isfinite(poses).all() or not np.array_equal(result[:, 0], np.arange(expected)):
        raise ValueError('Invalid result index/data')
    if not np.allclose(result[:, 1:3], poses[:, [3, 7]], atol=1e-8, rtol=0):
        raise ValueError('CSV/TXT positions disagree')
    delta = result[:, 3]-np.arctan2(poses[:, 4], poses[:, 0])
    if not np.all(np.abs(np.arctan2(np.sin(delta), np.cos(delta))) < 1e-8):
        raise ValueError('CSV/TXT yaw mismatch')
    last_stamp = last_camera = None
    for i, (m, t) in enumerate(zip(manifest, timing)):
        stamp = int(m['timestamp_ns']); camera = int(m['camera_timestamp_ns'])
        if (int(m['idx']) != i or int(m['original_idx']) != i or int(t['idx']) != i
                or int(t['original_idx']) != i or int(t['timestamp_ns']) != stamp
                or not 0 < camera <= stamp or (last_stamp is not None and (stamp <= last_stamp or camera <= last_camera))):
            raise ValueError('Frame/time contract mismatch')
        R = poses[i].reshape(3, 4)[:, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-8) or abs(np.linalg.det(R)-1) > 1e-8:
            raise ValueError('Non-rigid TXT rotation')
        last_stamp, last_camera = stamp, camera
    durations = np.array([[float(t[k]) for k in ('load_ms', 'registration_ms', 'total_ms')] for t in timing])
    if not np.isfinite(durations).all() or np.any(durations < 0):
        raise ValueError('Invalid timing')
    return result, poses, durations


def analyze(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run = Path(run)
    info = json.loads((run/'test_info.json').read_text())
    if info['status'] != 'completed' or info['mode'] != 'full' or info['frame_count'] != 3299:
        raise ValueError('Only a completed 3299-frame full receives standard analysis')
    result, poses, durations = validate(run)
    obs = [json.loads(line) for line in (run/'observations.jsonl').read_text().splitlines()]
    has_memory = (len(obs) == len(result) and
                  all('correction_filter' in o.get('registration', {}) for o in obs))
    has_observability = (has_memory and all('pose_covariance_xy_yaw' in o.get('sensor_prior', {}) for o in obs))
    has_quality = len(obs) == len(result) and any('range_selection' in o.get('registration', {}) for o in obs)
    has_heading = len(obs) == len(result) and any(o.get('sensor_prior', {}).get('heading_information_policy') == 'fractional_likelihood_budget_v510' for o in obs)
    has_images = len(obs) == len(result) and any('image_ground_speed' in o for o in obs)
    names = ['result_2.csv', 'poses.txt', 'timings.csv', 'frame_manifest.csv', 'observations.jsonl', 'test_info.json']
    if (run/'reference_contract.json').is_file(): names.append('reference_contract.json')
    sources = {n: sha256(run/n) for n in names}
    output = run/'analysis'
    output.mkdir(exist_ok=True)
    manifest_path = output/'manifest.json'
    stems = ('01_trajectory', '02_pose_changes', '03_registration_timing', '04_fusion_and_scale')
    if has_memory:
        stems += ('05_map_correction_memory',)
    if has_observability:
        stems += ('06_observability_uncertainty',)
    if has_images:
        stems += ('07_image_ground_speed',)
    if has_quality:
        stems += ('08_quality_models',)
    if has_heading:
        stems += ('08_heading_information',)
    recognized = {'manifest.json', 'README.md'} | {f'{s}.{ext}' for s in stems for ext in ('png', 'pdf')}
    if has_images:
        recognized.add('image_speed_summary.json')
    if has_quality: recognized.add('quality_models_summary.json')
    if has_heading: recognized.add('heading_information_summary.json')
    if any(f.name not in recognized for f in output.iterdir()):
        raise FileExistsError('Unrecognized analysis artifacts preserved')
    if any(output.iterdir()):
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()).get('sources') != sources:
            raise FileExistsError('Existing analysis source SHA differs or has no manifest; preserved')
    write_json(manifest_path, dict(status='in_progress', sources=sources,
                                  script_sha256=sha256(Path(__file__))))
    def save(fig, stem):
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(output/f'{stem}.{ext}', dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(result[:, 1], result[:, 2], lw=1, label='Causal fused camera pose')
    ax.scatter(*result[0, 1:3], label='Start'); ax.scatter(*result[-1, 1:3], marker='x', label='End')
    ax.set(xlabel='Map X [m]', ylabel='Map Y [m]', title='Track2 estimated trajectory (no ground truth)')
    ax.axis('equal'); ax.grid(alpha=.3); ax.legend(); save(fig, '01_trajectory')
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(np.linalg.norm(np.diff(poses[:, [3, 7, 11]], axis=0), axis=1))
    d = np.diff(result[:, 3]); axes[1].plot(np.arctan2(np.sin(d), np.cos(d)))
    axes[0].set(ylabel='3D displacement [m]', title='Changes between emitted poses')
    axes[1].set(xlabel='Frame idx', ylabel='Yaw change [rad]')
    for ax in axes: ax.grid(alpha=.3)
    save(fig, '02_pose_changes')
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for i, label in enumerate(('Load + sensors', 'Registration', 'Total frame')):
        axes[i].plot(durations[:, i]); axes[i].set(ylabel=label+' [ms]'); axes[i].grid(alpha=.3)
    axes[-1].set_xlabel('Frame idx'); save(fig, '03_registration_timing')
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot([np.degrees(o['state'][3]) for o in obs]); axes[0].set_ylabel('Heading bias [deg]')
    axes[1].plot([o['state'][4] for o in obs]); axes[1].set_ylabel('Sensor speed [m/s]')
    axes[2].plot([o['scale'] for o in obs]); axes[2].set(xlabel='Frame idx', ylabel='PLY scale')
    for ax in axes: ax.grid(alpha=.3)
    save(fig, '04_fusion_and_scale')
    if has_memory:
        memory = [o['registration']['correction_filter'] for o in obs]
        correction = np.array([d['correction'] for d in memory])
        std = np.sqrt(np.array([np.diag(d['covariance']) for d in memory]))
        idx = np.arange(len(memory))
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        for j, name in enumerate(('Map X', 'Map Y')):
            axes[0].plot(idx, correction[:, j], label=name)
            axes[0].fill_between(idx, correction[:, j]-std[:, j], correction[:, j]+std[:, j], alpha=.15)
        axes[0].set(ylabel='Correction [m]', title='Causal map correction state (bands: model 1-sigma, uncalibrated)')
        axes[0].legend()
        axes[1].plot(idx, np.degrees(correction[:, 2]))
        axes[1].fill_between(idx, np.degrees(correction[:, 2]-std[:, 2]), np.degrees(correction[:, 2]+std[:, 2]), alpha=.15)
        axes[1].set_ylabel('Yaw correction [deg]')
        nis = [np.nan if d['innovation_nis'] is None else d['innovation_nis'] for d in memory]
        axes[2].plot(idx, nis, lw=.8)
        from . import parameters as p
        axes[2].plot(idx, [np.nan if d.get('nis_soft_threshold', p.CORRECTION_NIS_SOFT) is None else d.get('nis_soft_threshold', p.CORRECTION_NIS_SOFT) for d in memory], color='tab:orange', ls='--', label='Soft gate (current DOF)')
        axes[2].plot(idx, [np.nan if d.get('nis_reject_threshold', p.CORRECTION_NIS_REJECT) is None else d.get('nis_reject_threshold', p.CORRECTION_NIS_REJECT) for d in memory], color='tab:red', ls='--', label='Hard gate (current DOF)')
        axes[2].set(ylabel='Innovation NIS', yscale='symlog'); axes[2].legend()
        age = np.array([np.nan if d['seconds_since_update'] is None else d['seconds_since_update'] for d in memory])
        axes[3].plot(idx, age, label='Age of last used map update')
        reject = np.array([not d['accepted'] for d in memory])
        axes[3].scatter(idx[reject], age[reject], s=7, color='tab:red', label='No current map update')
        axes[3].set(xlabel='Frame idx', ylabel='Age [s]'); axes[3].legend()
        for ax in axes: ax.grid(alpha=.3)
        save(fig, '05_map_correction_memory')
    if has_observability:
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        prior_std = np.sqrt(np.array([np.diag(o['sensor_prior']['pose_covariance_xy_yaw']) for o in obs]))
        for j, label in enumerate(('Prior X', 'Prior Y')):
            axes[0].plot(prior_std[:, j], label=label)
        axes[0].set_ylabel('Prior sigma [m]'); axes[0].legend()
        axes[1].plot(np.degrees(prior_std[:, 2]), label='Current prior yaw sigma')
        axes[1].set_ylabel('Yaw sigma [deg]'); axes[1].legend()
        ranks = [0 if o['registration'].get('observability') is None else o['registration']['observability']['rank'] for o in obs]
        axes[2].step(np.arange(len(obs)), ranks, where='post', label='Observable map pose directions')
        axes[2].set(ylabel='Rank', ylim=(-.1, 3.1), yticks=[0, 1, 2, 3]); axes[2].legend()
        for key, label in (('gain_scale_xy', 'XY gain limit'), ('gain_scale_yaw', 'Yaw gain limit')):
            axes[3].plot([o['registration']['correction_filter'].get(key, 1.) for o in obs], label=label, alpha=.8)
        axes[3].set(xlabel='Frame idx', ylabel='Gain multiplier', ylim=(-.05, 1.05)); axes[3].legend()
        for ax in axes: ax.grid(alpha=.3)
        save(fig, '06_observability_uncertainty')
    if has_images:
        records = [o.get('image_ground_speed', {}) for o in obs]
        used = np.array([bool(d.get('filter_accepted')) for d in records])
        moving = np.array([d.get('kind') == 'speed' for d in records])
        stationary = np.array([d.get('kind') == 'stationary' for d in records])
        speed = np.array([float(d['speed_mps']) if d.get('kind') == 'speed'
                          else 0. if d.get('kind') == 'stationary' else np.nan for d in records])
        idx = np.arange(len(records))
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        axes[0].plot(idx, [o['state'][4] for o in obs], label='Sensor state after image update', lw=.8)
        axes[0].scatter(idx[used], speed[used], s=9, label='Used image observation')
        axes[0].scatter(idx[~used & np.isfinite(speed)], speed[~used & np.isfinite(speed)],
                        s=8, marker='x', color='tab:red', label='Unused image estimate')
        axes[0].set(ylabel='Speed [m/s]', title='Image speed observations, not ground truth'); axes[0].legend()
        axes[1].plot(idx, [d.get('innovation_nis', np.nan) for d in records], lw=.8)
        axes[1].set(ylabel='Velocity innovation NIS', yscale='symlog')
        axes[2].plot(idx, [o.get('phase_timings_ms', {}).get('image_ground_speed', 0.) for o in obs], label='Image read/tracking/update')
        axes[2].plot(idx, [o.get('phase_timings_ms', {}).get('image_ground_model_store', 0.) for o in obs], label='CUDA ground-model storage')
        axes[2].set(xlabel='Frame idx', ylabel='Image component [ms]'); axes[2].legend()
        for ax in axes: ax.grid(alpha=.3)
        save(fig, '07_image_ground_speed')
        write_json(output/'image_speed_summary.json', dict(
            accepted_moving_intervals=int(np.sum(used & moving)),
            accepted_stationary_intervals=int(np.sum(used & stationary)),
            accepted_frames=idx[used].tolist(),
            unused_estimate_frames=idx[~used & np.isfinite(speed)].tolist(),
            note='One observation per disjoint interval; no image-derived absolute position/yaw; not accuracy evidence'))
    if has_quality:
        from collections import Counter
        idx = np.arange(len(obs))
        selected = np.array([(o['registration'].get('range_selection') or {}).get('selected_range_m', np.nan) for o in obs])
        bias = np.array([o.get('sensor_prior', {}).get('gyro_bias_body_radps', [np.nan]*3) for o in obs])
        height = np.array([o['registration'].get('height_anchor_after_m', np.nan) for o in obs])
        updated = np.array([bool((o['registration'].get('height_update') or {}).get('accepted')) for o in obs])
        records = [o.get('image_ground_speed', {}) for o in obs]
        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        axes[0].step(idx, selected, where='post'); axes[0].set(ylabel='Selected range [m]', title='Causal quality models; diagnostics, not ground truth')
        axes[1].scatter(idx, [d.get('inliers', np.nan) for d in records], s=4, label='Image consensus inliers')
        axes[1].set_ylabel('Image inlier count'); axes[1].legend(loc='upper left')
        right = axes[1].twinx(); right.scatter(idx, [d.get('inlier_fraction', np.nan) for d in records], s=3, color='tab:orange')
        right.set(ylabel='Whole-mask fraction', ylim=(0, 1.05))
        for j, name in enumerate(('x', 'y', 'z')): axes[2].plot(idx, np.degrees(bias[:, j]), label=name)
        axes[2].set_ylabel('Gyro bias [deg/s]'); axes[2].legend()
        axes[3].plot(idx, height, label='Estimated camera height'); axes[3].scatter(idx[updated], height[updated], s=5, label='Independent data update')
        axes[3].set(xlabel='Frame idx', ylabel='Estimated height [m]'); axes[3].legend()
        for ax in axes: ax.grid(alpha=.3)
        save(fig, '08_quality_models')
        write_json(output/'quality_models_summary.json', dict(
            selected_range_counts=dict(Counter(str(float(v)) for v in selected if np.isfinite(v))),
            independent_height_updates=int(updated.sum()),
            image_reasons=dict(Counter(d.get('reason', 'not_recorded') for d in records)),
            final_gyro_bias_body_radps=None if not np.isfinite(bias[-1]).all() else bias[-1].tolist(),
            reference_contract_recorded=(run/'reference_contract.json').is_file(),
            caveat='Engineering models, not calibrated accuracy; current/past only; no retrospective pose changes'))
    if has_heading:
        from collections import defaultdict, Counter
        prior_rows = [o.get('sensor_prior', {}) for o in obs]
        budget = np.array([d.get('heading_update_budget_used_deg', 0.) for d in prior_rows])
        limit = np.array([d.get('heading_update_budget_limit_deg', np.nan) for d in prior_rows])
        per_frame = []; factors = defaultdict(float); decisions = {}
        for i, d in enumerate(prior_rows):
            updates = d.get('heading_information_updates', [])
            per_frame.append(sum(u.get('information_fraction', 0.) for u in updates))
            for u in updates:
                if 'heading_factor_id' in u:
                    factors[str(u['heading_factor_id'])] += u.get('information_fraction', 0.)
            course = d.get('course_decision')
            if course and 'delivered_timestamp_ns' in course:
                key = (course['delivered_timestamp_ns'], course.get('start_timestamp_ns'), course.get('end_timestamp_ns'))
                if key not in decisions: decisions[key] = (i, course)
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        axes[0].plot(budget, label='Absolute heading calibration used')
        axes[0].plot(limit, '--', label='Post-bootstrap frame budget')
        axes[0].set(ylabel='Calibration step [deg]', title='Causal heading information; diagnostics, not ground truth'); axes[0].legend()
        axes[1].stem(np.arange(len(per_frame)), per_frame, markerfmt=' ', basefmt=' ')
        axes[1].set_ylabel('Likelihood fraction spent')
        for accepted, label in ((False, 'Course held/rejected'), (True, 'Course accepted')):
            frames = [i for i, d in decisions.values() if bool(d.get('accepted')) == accepted]
            axes[2].scatter(frames, [int(accepted)]*len(frames), s=7, label=label)
        axes[2].set(xlabel='Frame idx', ylabel='Course decision', yticks=[0, 1]); axes[2].legend()
        for ax in axes: ax.grid(alpha=.3)
        save(fig, '08_heading_information')
        write_json(output/'heading_information_summary.json', dict(
            factor_information_fraction_totals=dict(factors),
            maximum_factor_fraction=max(factors.values(), default=0.),
            post_bootstrap_budget_exceedances=int(np.sum(budget[1:] > limit[1:]+1e-9)),
            course_decisions=dict(Counter(d.get('reason', 'unknown') for _, d in decisions.values())),
            caveat='One factor may be split across future CURRENT updates; total fraction <=1; first emitted frame may bootstrap; never rewrites past poses'))
    # Never change input files while plotting, including info used as evidence.
    if sources != {n: sha256(run/n) for n in names}:
        raise RuntimeError('Input changed during analysis')
    write_json(manifest_path, dict(status='completed', sources=sources,
        script_sha256=sha256(Path(__file__)), ground_truth_available=False,
        caveat='Acceptance and residuals are not localization accuracy'))
    (output/'README.md').write_text('# Track2 因果定位唯讀圖表\n\n軌跡、相鄰 pose 變化、耗時及融合狀態；沒有獨立真值。\n\n'
        +('05 顯示 world XY/yaw 修正、模型不確定度、innovation NIS 及有效配準的失聯時間；不確定度尚未經真值校準。\n\n' if has_memory else '')
        +('06 顯示逐幀 prior 標準差、scale 邊際化後的 map 可觀測維度及 XY/yaw 分別限幅。\n\n' if has_observability else '')
        +('07 顯示影像速度量測、實際採用影格、velocity innovation 與讀圖／ground model 耗時；image_speed_summary.json 列出採用的 frame，不能當精度證據。\n\n' if has_images else '')
        +('08_heading_information 顯示共用航向預算、已消費資訊及 course 決策；不修改定位輸出。\n\n' if has_heading else '')
        +('08 顯示當幀選用範圍、影像共識支持、gyro零偏和獨立高度更新；quality_models_summary.json保存統計。\n\n' if has_quality else '')
        +'GNSS 是融合輸入，不是真值；接受率與殘差不能證明精度。來源 SHA 見 manifest.json。\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--run-dir', type=Path, required=True)
    analyze(parser.parse_args().run_dir)
