"""幾何 CSV 契約與摘要；只檢查內部數值，不推估定位精度。"""
from output_layout import loads_saved_json
import csv
import json
import math
from collections import Counter


def summarize_guided_updates(rows):
    """拒絕原因來自候選 pose；不可由回退後的 geometry.csv 反推原因。"""
    if not rows or all('update_status' not in row for row in rows):
        return {'update_status_available': False}, []
    geometry_reasons = {'insufficient_reliable_geometry', 'insufficient_unique_targets'}
    reasons = geometry_reasons | {'accepted', 'insufficient_input_points', 'no_map_candidates',
                                 'nonfinite_objective', 'insufficient_matches', 'objective_increased'}
    try:
        counts = Counter()
        for row in rows:
            status = row['update_status']
            if status not in reasons or row['map_accepted'] not in ('0', '1'):
                raise ValueError('Invalid guided update status/flag')
            if (status == 'accepted') != (row['map_accepted'] == '1'):
                raise ValueError('Guided update status differs from map acceptance')
            inliers, reliable, unique = (int(row['candidate_' + key]) for key in
                                          ('inliers', 'reliable_points', 'unique_targets'))
            if not 0 <= unique <= reliable <= inliers:
                raise ValueError('Invalid candidate geometry support counts')
            if status == 'insufficient_reliable_geometry' and not reliable < 10:
                raise ValueError('Reliable geometry rejection differs from candidate support')
            if status == 'insufficient_unique_targets' and not (reliable >= 10 and unique < 6):
                raise ValueError('Unique target rejection differs from candidate support')
            if status != 'accepted' and any(float(row[key]) != 0 for key in ('dx', 'dy', 'dz', 'dyaw')):
                raise ValueError('Rejected map update retained a correction')
            counts[status] += 1
        geometry_fallbacks = sum(counts[reason] for reason in geometry_reasons)
        return {'update_status_available': True, 'update_status_counts': dict(sorted(counts.items())),
                'geometry_support_fallback_frames': geometry_fallbacks,
                'other_map_rejection_frames': len(rows) - counts['accepted'] - geometry_fallbacks,
                'update_status_scope': 'candidate before rejection; geometry.csv describes returned pose'}, []
    except (KeyError, ValueError, TypeError) as error:
        return {}, [str(error)]


def validate_geometry(run, timestamps, mode, normal_gain, tangent_weight):
    errors = []
    summary = {}
    try:
        config = loads_saved_json((run / 'geometry_config.json').read_text())
        with (run / 'geometry.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        if config['mode'] != mode or config['normal_gain'] != normal_gain or config['tangent_weight'] != tangent_weight:
            errors.append('Geometry configuration differs from requested parameters')
        if [int(row['timestamp_ns']) for row in rows] != timestamps or [int(row['idx']) for row in rows] != list(range(len(timestamps))):
            errors.append('Geometry count / frame timestamp / idx mismatch')
        for row in rows:
            values = {key: float(value) for key, value in row.items() if key != 'timestamp_ns'}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError('Nonfinite geometry diagnostics')
            if not (0 <= values['unique_targets'] <= values['reliable_points'] <= values['euclidean_inliers'] <= values['input_points']):
                raise ValueError('Invalid geometry support counts')
            if not (0 <= values['mean_confidence'] <= 1.000001 and 0 <= values['observable_rank'] <= 4):
                raise ValueError('Invalid geometry confidence/rank')
            if values['plane_rmse_m'] < 0 or any(values['span_' + axis + '_m'] < 0 for axis in 'xyz'):
                raise ValueError('Invalid geometry residual/span')
            if values['directional_update_used'] not in (0, 1) or values['map_accepted'] not in (0, 1):
                raise ValueError('Invalid geometry update flags')
            if mode == 'diagnostic' and values['directional_update_used']:
                raise ValueError('Diagnostic-only run changed weighting')
            if mode in ('directional', 'gicp') and values['directional_update_used'] != values['map_accepted']:
                raise ValueError('Directional update flag differs from accepted map update')
            eigenvalues = [values[f'scaled_eigenvalue_{i}'] for i in range(4)]
            if min(eigenvalues) < -1e-9 or eigenvalues != sorted(eigenvalues):
                raise ValueError('Invalid information eigenvalues')
            threshold = max(config['rank_absolute_threshold'], config['rank_relative_threshold'] * max(eigenvalues))
            if values['observable_rank'] != sum(value >= threshold for value in eigenvalues):
                raise ValueError('Geometry observable rank differs from scaled spectrum')
            vectors = [[values[f'scaled_eigenvector_r{r}c{c}'] for c in range(4)] for r in range(4)]
            info = [[values[f'scaled_information_r{r}c{c}'] for c in range(4)] for r in range(4)]
            for r in range(4):
                for c in range(4):
                    orthogonal = sum(vectors[k][r] * vectors[k][c] for k in range(4))
                    reconstructed = sum(vectors[r][k] * eigenvalues[k] * vectors[c][k] for k in range(4))
                    if abs(orthogonal - int(r == c)) > 1e-7 or abs(reconstructed - info[r][c]) > 1e-7 * max(1., abs(info[r][c])):
                        raise ValueError('Geometry eigensystem inconsistent with data-only information')
        if mode == 'gicp':
            with (run / 'gicp.csv').open() as stream:
                source_rows = list(csv.DictReader(stream))
            with (run / 'guidance.csv').open() as stream:
                guidance = list(csv.DictReader(stream))
            if [int(r['timestamp_ns']) for r in source_rows] != timestamps or len(guidance) != len(rows):
                raise ValueError('GICP source diagnostic timestamp/count mismatch')
            for index, (source, geometry, update) in enumerate(zip(source_rows, rows, guidance)):
                if (int(source['idx']) != index or source['input_points'] != geometry['input_points'] or
                    source['map_accepted'] != geometry['map_accepted'] or
                    source['candidate_paired_points'] != update['candidate_reliable_points'] or
                    source['candidate_unique_targets'] != update['candidate_unique_targets'] or
                    not 0 <= int(source['candidate_paired_points']) <= int(source['source_reliable_points']) <= int(source['input_points'])):
                    raise ValueError('GICP source geometry count/acceptance mismatch')
        ratios = [float(row['reliable_points']) / max(1, int(row['input_points'])) for row in rows]
        ranks = [int(row['observable_rank']) for row in rows]
        summary = {'config': config, 'frames': len(rows), 'rank_histogram': {str(i): ranks.count(i) for i in range(5)},
                   'mean_reliable_fraction': sum(ratios) / len(rows) if rows else 0.,
                   'directional_update_frames': sum(int(row['directional_update_used']) for row in rows),
                   'insufficient_geometry_frames': sum(int(row['reliable_points']) < 10 or int(row['unique_targets']) < 6 for row in rows),
                   'insufficient_geometry_scope': 'support at returned pose; not a rejection-reason count; see guidance update_status',
                   'scope': 'model geometry indicators only; neither covariance nor ground-truth accuracy'}
    except (OSError, KeyError, ValueError, TypeError) as error:
        errors.append(str(error))
    return summary, errors
