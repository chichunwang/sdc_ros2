#!/usr/bin/env python3
"""唯讀盤點規則原件與候選資料位置；不啟動定位，不認證定位精度。"""
import argparse
import json
import math
from pathlib import Path
import re
import sqlite3

from competition_rules import ROOT, RULES_VERSION, RULES_REVIEW_DATE, document_identities, track_contract


def read_map_fields(path):
    # Only the ASCII PCD header is inspected; never interpret tl/th as observations.
    with path.open('rb') as stream:
        for _ in range(128):
            line = stream.readline(8192)
            if not line or not line.endswith(b'\n'):
                break
            words = line.decode('ascii').strip().split()
            if words and words[0] == 'FIELDS':
                return words[1:]
            if words and words[0] == 'DATA':
                break
    raise ValueError('PCD header has no readable FIELDS entry')


def inspect_track3_details(directory):
    """Only filenames, bag timestamps/topic metadata, and the supplied init matrix."""
    errors = []
    paths = sorted((directory / 'raw_ply').glob('*.ply'))
    matches = [re.fullmatch(r'(\d+)_ts_(\d+)\.ply', path.name) for path in paths]
    parsed = sorted((int(m[1]), int(m[2])) for m in matches if m)
    indices = [i for i, _ in parsed]
    stamps = [stamp for _, stamp in parsed]
    ordered = len(paths) == 3142 and all(matches) and indices == list(range(3142))
    increasing = all(b > a for a, b in zip(stamps, stamps[1:]))
    if not ordered or not increasing or not all(stamp > 0 for stamp in stamps):
        errors.append('Track3 filenames must contain idx 0..3141 and strictly increasing positive timestamps')
    if any(path.stat().st_size == 0 for path in paths):
        errors.append('Track3 has empty PLY files')
    report = {'filename_indices_complete_0_to_3141': ordered,
              'timestamps_strictly_increasing': increasing,
              'first_timestamp_ns': stamps[0] if stamps else None,
              'last_timestamp_ns': stamps[-1] if stamps else None,
              'bag_payload_decoded': False, 'point_payload_inspected': False,
              'official_sample_verified': False, 'scale_calibrated': False,
              'submission_frame_and_yaw_units_verified': False}
    front_topic = '/lucid_cameras_x00/gige_100_f_hdr/h265'
    try:
        bag = directory / 'Track3.db3'
        with sqlite3.connect(bag.as_uri() + '?mode=ro') as connection:
            connection.execute('PRAGMA query_only=ON')
            topics = connection.execute('SELECT id,name,type FROM topics ORDER BY id').fetchall()
            report['bag_topics'] = []
            front = []
            for topic_id, name, kind in topics:
                count, first, last = connection.execute(
                    'SELECT COUNT(*),MIN(timestamp),MAX(timestamp) FROM messages WHERE topic_id=?',
                    (topic_id,)).fetchone()
                report['bag_topics'].append({'name': name, 'type': kind, 'messages': count,
                                             'first_timestamp_ns': first, 'last_timestamp_ns': last})
                if name == front_topic:
                    if kind != 'sensor_msgs/msg/Image':
                        errors.append('Unexpected front H265 bag message type')
                    front = [row[0] for row in connection.execute(
                        'SELECT timestamp FROM messages WHERE topic_id=? ORDER BY id', (topic_id,))]
            report['ply_timestamps_match_front_bag_in_record_order'] = stamps == front and bool(front)
            if not report['ply_timestamps_match_front_bag_in_record_order']:
                errors.append('Track3 PLY timestamps do not match front bag records one-to-one')
            imu = [row for row in report['bag_topics'] if row['name'] == '/imu/data']
            if not imu or imu[0]['type'] != 'sensor_msgs/msg/Imu' or not imu[0]['messages']:
                errors.append('Track3 /imu/data metadata missing or unexpected')
            report['gnss_pose_topic_present'] = any('gnss_pose' in name.lower() for _, name, _ in topics)
    except (OSError, sqlite3.Error, ValueError) as exc:
        errors.append(f'Track3 bag metadata/timestamp inspection failed: {exc}')
    try:
        pose = [[float(x) for x in line.replace(',', ' ').split()]
                for line in (directory / 'init_guess.txt').read_text().splitlines() if line.strip()]
        if len(pose) != 4 or any(len(row) != 4 for row in pose) or not all(math.isfinite(x) for row in pose for x in row):
            raise ValueError('Expected a finite 4x4 matrix')
        orth = max(abs(sum(pose[k][i] * pose[k][j] for k in range(3)) - int(i == j))
                   for i in range(3) for j in range(3))
        r = pose
        det = (r[0][0] * (r[1][1]*r[2][2] - r[1][2]*r[2][1])
               - r[0][1] * (r[1][0]*r[2][2] - r[1][2]*r[2][0])
               + r[0][2] * (r[1][0]*r[2][1] - r[1][1]*r[2][0]))
        report['initial_matrix'] = {'rotation_orthogonality_max_error': orth, 'rotation_det': det,
                                    'last_row': pose[3], 'translation': [pose[i][3] for i in range(3)],
                                    'atan2_R10_R00_rad': math.atan2(pose[1][0], pose[0][0]),
                                    'frame_direction_verified': False}
        if orth > 1e-5 or abs(det - 1) > 1e-5 or max(abs(a-b) for a,b in zip(pose[3], [0,0,0,1])) > 1e-8:
            errors.append('Track3 initial matrix is not a rigid homogeneous transform')
    except (OSError, ValueError) as exc:
        errors.append(f'Track3 initial matrix inspection failed: {exc}')
    return report, errors


def inspect_track(track, data_root):
    contract = track_contract(track)
    directory = Path(data_root).resolve() / f'Track{track}'
    cloud_dir, map_path = directory / 'raw_ply', directory / 'map.pcd'
    stamps = []
    ignored = 0
    if cloud_dir.is_dir():
        for path in cloud_dir.iterdir():
            if not path.is_file() or path.suffix.lower() != '.ply':
                continue
            match = re.search(r'_ts_(\d+)', path.name)
            if match and int(match[1]) > 0:
                stamps.append(int(match[1]))
            else:
                ignored += 1
    errors = []
    fields = []
    if not cloud_dir.is_dir() or not stamps:
        errors.append('No timestamp-named PLY files at the candidate raw_ply path')
    if len(stamps) != len(set(stamps)):
        errors.append('Duplicate integer nanosecond timestamps')
    expected = contract['expected_frames']
    if expected is not None and len(stamps) != expected:
        errors.append(f'Expected {expected} PLY frames; observed {len(stamps)}')
    if map_path.is_file():
        try:
            fields = read_map_fields(map_path)
            if not {'x', 'y', 'z'}.issubset(fields):
                errors.append('Map is missing XYZ fields')
        except (ValueError, UnicodeError) as exc:
            errors.append(str(exc))
    else:
        errors.append('No map.pcd at the candidate map path')
    documents = document_identities()
    if not all(item['matches_reviewed_source'] for item in documents.values()):
        errors.append('Official PDF missing or differs from the reviewed version')
    details = None
    if track == 3:
        details, extra_errors = inspect_track3_details(directory)
        errors.extend(extra_errors)
    inventory_errors = list(errors)
    if track == 3:
        errors.append('Track3 local filenames/bag timestamps/init matrix inspected; official sample ID/yaw/frame contract and source scale remain unverified')
        errors.append('Track3 map-only legacy entry is statically causal; runtime prefix consistency and localization execution remain unverified')
    else:
        errors.append('Existing sensor_guided priors, including fixed and pose_yaw, use non-causal processing; GNSS input permission does not establish compliance')
        errors.append('Existing CPU/GPU map smoothers and surfel_joint use backward smoothing or offline batch trajectory correction and are not compliant')
    errors.append('No causal localization pipeline or submission compliance has been verified for this Track')
    return {
        'rules_version': RULES_VERSION, 'rules_review_date': RULES_REVIEW_DATE, 'contract': contract, 'documents': documents,
        'candidate_data_directory': str(directory), 'layout_provisional': track == 3 and bool(inventory_errors),
        'track3_local_details': details,
        'observed_ply_count': len(stamps), 'ignored_ply_names': ignored,
        'observed_map_fields': fields,
        'unused_map_fields': [name for name in fields if name not in ('x', 'y', 'z')],
        'inventory_checks_passed': not inventory_errors, 'inventory_errors': inventory_errors,
        'checks_passed': False, 'errors': errors,
        'scope': 'Read-only rules/source identity, filenames and PCD header; Track3 additionally bag topic metadata/integer timestamps and init matrix. No sensor or point payload decoding, localization, submission certification or accuracy verification.',
        'localization_executed': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT.parent / 'midterm01/data')
    parser.add_argument('--inventory-only', action='store_true', help='Exit status reflects data inventory only; does not certify a localization pipeline')
    args = parser.parse_args()
    report = inspect_track(args.track, args.data_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    passed = report['inventory_checks_passed'] if args.inventory_only else report['checks_passed']
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
