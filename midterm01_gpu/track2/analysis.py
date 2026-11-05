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
    has_candidates = len(obs) == len(result) and any(o.get('registration',{}).get('candidate_selection') is not None for o in obs)
    has_likelihood = len(obs) == len(result) and any(o.get('registration',{}).get('map_likelihood') is not None for o in obs)
    has_position_budget = len(obs) == len(result) and any(o.get('sensor_prior',{}).get('position_information_policy') == 'fractional_likelihood_budget_v5121' for o in obs)
    has_heading = len(obs) == len(result) and any(o.get('sensor_prior', {}).get('heading_information_policy') == 'fractional_likelihood_budget_v510' for o in obs)
    has_images = len(obs) == len(result) and any('image_ground_speed' in o for o in obs)
    has_image_relative = has_images and any('solver_comparison' in (o.get('image_ground_speed') or {}) for o in obs)
    has_geometry = len(obs) == len(result) and any(o.get('registration', {}).get('surface_profile') is not None for o in obs)
    has_turn = len(obs) == len(result) and any(o.get('sensor_prior', {}).get('motion_prediction') is not None for o in obs)
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
    if has_candidates:
        stems += ('09_candidate_validation',)
    if has_likelihood:
        stems += ('10_map_likelihood',)
    if has_position_budget: stems += ('11_position_information',)
    if has_image_relative: stems += ('12_image_relative',)
    if has_geometry: stems += ('13_geometry_diagnostics',)
    if has_turn: stems += ('14_coordinated_turn',)
    recognized = {'manifest.json', 'README.md'} | {f'{s}.{ext}' for s in stems for ext in ('png', 'pdf')}
    if has_images:
        recognized.add('image_speed_summary.json')
    if has_quality: recognized.add('quality_models_summary.json')
    if has_heading: recognized.add('heading_information_summary.json')
    if has_candidates: recognized.add('candidate_validation_summary.json')
    if has_likelihood: recognized.add('map_likelihood_summary.json')
    if has_position_budget: recognized.add('position_information_summary.json')
    if has_image_relative: recognized.add('image_relative_summary.json')
    if has_geometry: recognized.add('geometry_diagnostics_summary.json')
    if has_turn: recognized.add('coordinated_turn_summary.json')
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
        moving = np.array([d.get('kind') in ('speed', 'vector_velocity') for d in records])
        stationary = np.array([d.get('kind') == 'stationary' for d in records])
        speed = np.array([float(d['speed_mps']) if d.get('kind') in ('speed', 'vector_velocity') and d.get('speed_mps') is not None
                          else 0. if d.get('kind') == 'stationary' else np.nan for d in records])
        idx = np.arange(len(records))
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        axes[0].plot(idx, [o['state'][4] for o in obs], label='Sensor state after image update', lw=.8)
        axes[0].scatter(idx[used], speed[used], s=9, label='Used image observation')
        axes[0].scatter(idx[~used & np.isfinite(speed)], speed[~used & np.isfinite(speed)],
                        s=8, marker='x', color='tab:red', label='Unused image estimate')
        axes[0].set(ylabel='Speed [m/s]', title='Image speed observations, not ground truth'); axes[0].legend()
        axes[1].plot(idx, [d.get('innovation_nis', np.nan) for d in records], linestyle='None', marker='.', markersize=3)
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
    if has_candidates:
        selections = [o.get('registration',{}).get('candidate_selection') or {} for o in obs]
        validations = [(o.get('registration',{}).get('correction_filter') or {}).get('proposal_validation') or {} for o in obs]
        counts = [d.get('fully_refined',0) for d in selections]
        rejected = [i for i,d in enumerate(validations) if d and not d['accepted']]
        changes = [100.*(d['proposal_data_cost']-d['baseline_data_cost'])/max(abs(d['baseline_data_cost']),1e-12)
                   if d.get('proposal_data_cost') is not None else np.nan for d in validations]
        fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
        axes[0].step(np.arange(len(obs)),counts,where='post');axes[0].set(ylabel='Complete solves',title='Current-frame hypotheses and posterior checks; not ground truth')
        axes[1].plot(changes,label='Posterior vs prediction, same final planes')
        axes[1].axhline(0.,color='k',lw=.7)
        if rejected:axes[1].scatter(rejected,[changes[i] for i in rejected],color='tab:red',s=10,label='Proposal rejected')
        axes[1].set(xlabel='Frame idx',ylabel='Fixed-plane data-loss change [%]');axes[1].legend()
        for ax in axes:ax.grid(alpha=.3)
        save(fig,'09_candidate_validation')
        write_json(output/'candidate_validation_summary.json',dict(
            maximum_complete_solves=max(counts,default=0),
            startup_frames=[i for i,d in enumerate(selections) if d.get('startup')],
            proposal_reject_frames=rejected,
            caveat='Candidate quality/consistency only; no future poses or accuracy truth; every frame commits once'))
    if has_likelihood:
        likelihoods = [o.get('registration',{}).get('map_likelihood') or {} for o in obs]
        steps = np.array([d.get('likelihood_step_world',[np.nan]*3) for d in likelihoods])
        spectra = np.array([(o.get('registration',{}).get('observability') or {}).get('normalized_eigenvalues',[np.nan]*3) for o in obs])
        validations = [(o.get('registration',{}).get('correction_filter') or {}).get('proposal_validation') or {} for o in obs]
        rematched = np.array([100.*(v['proposal_common_feature_cost']-v['baseline_common_feature_cost'])/max(abs(v['baseline_common_feature_cost']),1e-12)
            if v.get('proposal_common_feature_cost') is not None and v.get('baseline_common_feature_cost') is not None else np.nan for v in validations])
        rejected = [i for i,v in enumerate(validations) if v.get('reason')=='posterior_rematched_feature_loss_increased']
        fig,axes=plt.subplots(4,1,figsize=(11,11),sharex=True)
        for j in range(3):axes[0].plot(np.maximum(spectra[:,j],1e-8),label=f'Mode {j+1}')
        axes[0].set_yscale('log');axes[0].axhline(1.,color='k',ls='--');axes[0].legend();axes[0].set(ylabel='Relative map information',title='Current DATA likelihood; steps and costs are not pose errors')
        axes[1].plot(np.linalg.norm(steps[:,:2],axis=1));axes[1].set_ylabel('Likelihood XY step [m]')
        axes[2].plot(np.degrees(steps[:,2]));axes[2].set_ylabel('Likelihood yaw step [deg]')
        axes[3].plot(rematched,label='Emitted vs prediction, same raw sample');axes[3].axhline(0.,color='k',lw=.7)
        if rejected:axes[3].scatter(rejected,rematched[rejected],s=9,color='tab:red',label='Rematched-cost rejection')
        axes[3].set(xlabel='Frame idx',ylabel='Rematched loss change [%]');axes[3].legend()
        for ax in axes:ax.grid(alpha=.3)
        save(fig,'10_map_likelihood')
        write_json(output/'map_likelihood_summary.json',dict(
            data_likelihood_frames=[i for i,d in enumerate(likelihoods) if d],
            rematched_cost_reject_frames=rejected,
            caveat='DATA gradient removes search prior from measurement center; original rank rules and height model retained; not ground-truth accuracy'))
    if has_position_budget:
        factors=Counter();expired=0.;per_frame=[];steps=[]
        for o in obs:
            d=o.get('sensor_prior',{});spent=0.
            steps.append(np.linalg.norm(d.get('gnss_state_innovation_xy',[0.,0.])))
            for u in d.get('position_information_updates',[]):
                if 'information_fraction' in u:
                    value=u['information_fraction'];spent+=value
                    factors[str(u['position_factor_id'])]+=value
                expired+=u.get('expired_information_fraction',0.)
            per_frame.append(spent)
        fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
        axes[0].plot(steps);axes[0].set(ylabel='GNSS state XY correction [m]',title='Causal GNSS information and estimator increments; not pose error')
        axes[1].stem(np.arange(len(obs)),per_frame,markerfmt=' ',basefmt=' ')
        axes[1].set(xlabel='Frame idx',ylabel='Axis likelihood power spent')
        for ax in axes:ax.grid(alpha=.3)
        save(fig,'11_position_information')
        write_json(output/'position_information_summary.json',dict(
            axis_factor_information_fraction_totals=dict(factors),
            maximum_axis_factor_fraction=max(factors.values(),default=0.),
            expired_unused_axis_fraction=float(expired),
            caveat='Two orthogonal scalar factors equal one isotropic XY likelihood; each axis total <=1. Current CV row with process inflation is approximate; no past pose rewrite or accuracy truth.'))
    # Missing optional diagnostics remain gaps rather than fabricated measurements.
    def diagnostic_number(record, key):
        value = record.get(key)
        return np.nan if value is None else float(value)
    if has_image_relative:
        from collections import Counter
        records = [o.get('image_ground_speed') or {} for o in obs]
        idx = np.arange(len(obs))
        anchors = [d.get('anchor_store') or {} for d in records]
        heading = [d.get('relative_heading_update') or {} for d in records]
        fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
        channels = {}
        kinds=('vector_velocity', 'relative_yaw', 'stationary', 'ground_speed_fallback')
        for j, kind in enumerate(kinds):
            selected = np.array([d.get('relative_channel',d.get('kind')) == kind for d in records])
            accepted = selected & np.array([bool(d.get('filter_accepted')) for d in records])
            axes[0].scatter(idx[accepted], np.full(accepted.sum(), j), s=7, label=kind)
            channels[kind] = dict(frames=idx[selected].tolist(), accepted_frames=idx[accepted].tolist(),
                reasons=dict(Counter(str(d.get('reason', 'not_recorded')) for d, take in zip(records, selected) if take)))
        axes[0].set(ylabel='Accepted channel', yticks=range(len(kinds)), yticklabels=('relative speed / legacy vector', 'relative yaw', 'stationary', 'road speed fallback'),
                    title='Relative image diagnostics; acceptance is not accuracy'); axes[0].legend()
        solvers = {}
        for name in ('pnp', 'gyro_translation'):
            rows = [(d.get('solver_comparison') or {}).get(name) or {} for d in records]
            accepted = np.array([bool(d.get('accepted')) for d in rows])
            axes[1].plot(idx, [diagnostic_number(d, 'common_capped_pixel_cost') for d in rows], label=name, lw=.8)
            solvers[name] = dict(qualified_count=int(accepted.sum()), qualified_frames=idx[accepted].tolist())
        axes[1].set_ylabel('Common capped pixel cost'); axes[1].legend()
        for key, label in (('accepted', 'Anchor available'), ('anchor_retained', 'Previous anchor retained')):
            axes[2].step(idx, [bool(d.get('anchor_available',d.get(key))) if key=='accepted' else bool(d.get(key)) for d in anchors], where='post', label=label, lw=.8)
        axes[2].set_ylabel('Anchor state'); axes[2].legend()
        axes[3].plot(idx, [diagnostic_number(d, 'height_variance_m2') for d in anchors], lw=.8)
        axes[3].set_ylabel('Anchor height variance [m²]')
        axes[4].plot(idx, [np.degrees(diagnostic_number(d, 'innovation_rad')) for d in heading], label='Innovation')
        axes[4].plot(idx, [np.degrees(diagnostic_number(d, 'bias_step_rad')) for d in heading], label='Bias step')
        axes[4].set(xlabel='Frame idx', ylabel='Relative heading [deg]'); axes[4].legend()
        for ax in axes: ax.grid(alpha=.3)
        save(fig, '12_image_relative')
        write_json(output/'image_relative_summary.json', dict(channels=channels, solvers=solvers,
            anchor_available_frames=[i for i, d in enumerate(anchors) if d.get('anchor_available',d.get('accepted'))],
            anchor_retained_frames=[i for i, d in enumerate(anchors) if d.get('anchor_retained')],
            anchor_reasons=dict(Counter(str(d.get('reason', 'not_recorded')) for d in anchors)),
            relative_heading_accepted_frames=[i for i, d in enumerate(heading) if d.get('accepted')],
            absolute_yaw_observed_frames=[i for i, d in enumerate(heading) if d.get('absolute_yaw_observed')],
            caveat='Current/past relative observations only; solver qualification and acceptance are not accuracy evidence.'))
    if has_geometry:
        from collections import Counter
        profiles = [o.get('registration', {}).get('surface_profile') or {} for o in obs]
        idx = np.arange(len(obs))
        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        axes[0].plot(idx, [diagnostic_number(d, 'normal_effective_directions') for d in profiles], label='Effective normal directions')
        axes[0].set(ylabel='Effective directions', title='Geometry and convergence diagnostics; original quality gates retained')
        right = axes[0].twinx()
        right.plot(idx, [diagnostic_number(d, 'dominant_normal_bin_fraction') for d in profiles], color='tab:orange', label='Dominant normal fraction')
        right.set(ylabel='Dominant normal fraction', ylim=(0, 1.05)); axes[0].legend(loc='upper left'); right.legend(loc='upper right')
        band_values = {}
        stage_reasons = Counter(); iterations = []; changes = []
        for i, (profile, observation) in enumerate(zip(profiles, obs)):
            for band in profile.get('range_bands') or []:
                label = f"{band.get('min_m')}–{band.get('max_m')} m"
                band_values.setdefault(label, np.full(len(obs), np.nan))[i] = diagnostic_number(band, 'signed_median_m')
            stages = observation.get('registration', {}).get('stages') or []
            traces = []
            for stage in stages:
                stage_reasons[str(stage.get('reason', 'not_recorded'))] += 1
                traces.extend(stage.get('iteration_trace') or [])
            iterations.append(len(traces))
            changes.append(diagnostic_number(traces[-1], 'association_change_fraction') if traces else np.nan)
        for label, values in band_values.items(): axes[1].plot(idx, values, lw=.8, label=label)
        axes[1].axhline(0., color='black', lw=.5); axes[1].set_ylabel('Range signed median [m]')
        if band_values: axes[1].legend()
        axes[2].plot(idx, iterations); axes[2].set_ylabel('Recorded iterations')
        axes[3].plot(idx, changes); axes[3].set(xlabel='Frame idx', ylabel='Last association change')
        for ax in axes: ax.grid(alpha=.3)
        save(fig, '13_geometry_diagnostics')
        write_json(output/'geometry_diagnostics_summary.json', dict(
            surface_profile_frames=[i for i, d in enumerate(profiles) if d],
            stage_reasons=dict(stage_reasons), recorded_iterations=int(sum(iterations)),
            range_band_labels=list(band_values),
            caveat='Original quality thresholds are unchanged. Normal diversity, signed residuals and convergence describe information and matching, not pose accuracy.'))
    if has_turn:
        records = [o.get('sensor_prior', {}).get('motion_prediction') or {} for o in obs]
        idx = np.arange(len(obs))
        mean = np.array([d.get('gyro_bias_mean_radps', [np.nan]*3) for d in records])
        covariance = np.array([d.get('gyro_bias_covariance_rad2ps2', np.full((3,3), np.nan)) for d in records])
        std = np.sqrt(np.maximum(0., np.diagonal(covariance, axis1=1, axis2=2)))
        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        for j, label in enumerate(('X', 'Y', 'Z')):
            values = np.degrees(mean[:, j]); sigma = np.degrees(std[:, j])
            axes[0].plot(idx, values, label=label)
            axes[0].fill_between(idx, values-sigma, values+sigma, alpha=.12)
        axes[0].set(ylabel='Gyro rest bias [deg/s]', title='Causal turn prediction; shading is model uncertainty, not measured error')
        axes[0].legend()
        axes[1].step(idx, [diagnostic_number(d, 'rest_bias_updates') for d in records], where='post')
        axes[1].set_ylabel('Accepted rest updates')
        axes[2].plot(idx, [np.degrees(diagnostic_number(d, 'signed_turn_since_frame_rad')) for d in records])
        axes[2].set_ylabel('Frame turn [deg]')
        axes[3].plot(idx, [np.degrees(diagnostic_number(d, 'integrated_turn_rad')) for d in records])
        axes[3].set(xlabel='Frame idx', ylabel='Integrated relative turn [deg]')
        for ax in axes: ax.grid(alpha=.3)
        save(fig, '14_coordinated_turn')
        available = [i for i, d in enumerate(records) if d]
        write_json(output/'coordinated_turn_summary.json', dict(
            diagnostic_frames=available, final=records[available[-1]],
            rest_update_frames=[i for i, o in enumerate(obs) if (o.get('image_ground_speed') or {}).get('gyro_motion_bias_update', {}).get('accepted')],
            caveat='Only arrived relative gyro and accepted rest intervals; no absolute yaw constraint, future correction, or independent accuracy measurement.'))
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
        +('09_candidate_validation 顯示完整候選數及實際posterior固定平面損失檢查；不是定位精度。\n\n' if has_candidates else '')
        +('10_map_likelihood 顯示 map 資訊、去除搜尋 prior 後的 likelihood 中心差，以及實際輸出重新對應的共同成本；不是定位誤差。\n\n' if has_likelihood else '')
        +('11_position_information 顯示 GNSS 對當幀 estimator 的修正與各軸資訊使用量；summary 記錄每軸總 power 及過期未使用資訊，不是定位誤差。\n\n' if has_position_budget else '')
        +('12_image_relative 顯示相對影像通道、兩 solver 的共同像素成本、anchor 保存及相對航向更新；summary 列出 frame 與原因，不代表絕對 yaw 或精度。\n\n' if has_image_relative else '')
        +('13_geometry_diagnostics 顯示法向多樣性、各距離有號殘差與迭代／對應變動；summary 統計收斂原因，沒有放寬品質門檻。\n\n' if has_geometry else '')
        +('14_coordinated_turn 顯示靜止估計的 gyro 偏差、模型標準差、已接受靜止更新及累積相對轉角；不代表絕對航向或定位誤差。\n\n' if has_turn else '')
        +('08_heading_information 顯示共用航向預算、已消費資訊及 course 決策；不修改定位輸出。\n\n' if has_heading else '')
        +('08 顯示當幀選用範圍、影像共識支持、gyro零偏和獨立高度更新；quality_models_summary.json保存統計。\n\n' if has_quality else '')
        +'GNSS 是融合輸入，不是真值；接受率與殘差不能證明精度。來源 SHA 見 manifest.json。\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--run-dir', type=Path, required=True)
    analyze(parser.parse_args().run_dir)
