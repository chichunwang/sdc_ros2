"""三相機來源與整數時間契約；標準函式庫，可供主機 benchmark 使用。"""
from output_layout import saved_path, loads_saved_json
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import struct

CAMERAS = {
    'f': '/lucid_cameras_x00/gige_100_f_hdr/h265',
    'fl': '/lucid_cameras_x01/gige_100_fl_hdr/h265',
    'fr': '/lucid_cameras_x01/gige_100_fr_hdr/h265',
}
FRAME_COMMENT = 'comment coordinate_frame=front_camera_FLU_x_front_y_left_z_up'


def sha256(path):
    digest = hashlib.sha256()
    with saved_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ply_header(path):
    with saved_path(path).open('rb') as stream:
        lines = []
        for _ in range(100):
            line = stream.readline().decode('ascii').strip()
            lines.append(line)
            if line == 'end_header':
                return lines, stream.tell()
    raise ValueError('Missing PLY end_header')


def load_index(data_root, track, expected_stamps=None):
    root = Path(data_root) / f'kaggle_track0{track}/Track{track}_Mapanything_ply'
    rows = list(csv.DictReader((root / 'index.csv').open()))
    if not rows or [int(r['index']) for r in rows] != list(range(len(rows))):
        raise ValueError('Three-camera index must be contiguous from zero')
    if len({r['ply'] for r in rows}) != len(rows):
        raise ValueError('Duplicate three-camera PLY')
    for tag in CAMERAS:
        stamps = [int(r[f'{tag}_ts']) for r in rows]
        if any(t <= 0 for t in stamps) or any(b <= a for a, b in zip(stamps, stamps[1:])):
            raise ValueError('Camera timestamps must be positive increasing integer ns')
    if expected_stamps is not None and [int(r['f_ts']) for r in rows] != list(expected_stamps):
        raise ValueError('Three-camera/front frame timestamps differ')
    for row in rows:
        path = root / row['ply']
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('PLY escapes source root')
        match = re.fullmatch(r'\d+_ts_(\d+)\.ply', path.name)
        if not match or int(match[1]) != int(row['f_ts']) or not path.is_file():
            raise ValueError('PLY filename/time/file contract mismatch')
        lines, _ = ply_header(path)
        if FRAME_COMMENT not in lines:
            raise ValueError('Three-camera source must explicitly declare front-camera FLU')
    report = loads_saved_json((root / 'report.json').read_text())
    if report.get('coordinate_frame') != 'model-predicted front-camera frame' or report.get('fusion_uses_calibrated_tf') is not False:
        raise ValueError('Unknown three-camera fusion/frame contract')
    return root, rows, {'index_sha256': sha256(root / 'index.csv'),
                        'report_sha256': sha256(root / 'report.json'),
                        'frame': 'front_camera_FLU', 'already_merged': True,
                        'reference_time': 'front_camera_bag_timestamp',
                        'per_camera_deskew_applied': False}


def decode_cdr_header(data):
    """ROS 2 CDR Header at message start; no decoding of image payload."""
    if len(data) < 17 or data[:2] not in (b'\x00\x01', b'\x00\x00'):
        raise ValueError('Unsupported/truncated CDR encapsulation')
    endian = '<' if data[1] == 1 else '>'
    sec, nanos, size = struct.unpack_from(endian + 'iII', data, 4)
    if sec < 0 or nanos >= 1_000_000_000 or size < 1 or 16 + size > len(data) or data[15 + size] != 0:
        raise ValueError('Invalid CDR Header stamp/frame_id')
    return sec * 1_000_000_000 + nanos, data[16:15 + size].decode('utf-8')


def camera_headers(bag):
    output = {}
    with sqlite3.connect(f'file:{Path(bag).resolve()}?mode=ro', uri=True) as connection:
        topics = {name: (ident, kind) for ident, name, kind in connection.execute('SELECT id,name,type FROM topics')}
        for tag, name in CAMERAS.items():
            ident, kind = topics[name]
            if kind != 'sensor_msgs/msg/Image':
                raise ValueError('Unexpected camera type')
            series = {}
            for stamp, blob in connection.execute('SELECT timestamp,substr(data,1,512) FROM messages WHERE topic_id=? ORDER BY timestamp', (ident,)):
                header_ns, frame_id = decode_cdr_header(blob)
                if stamp in series or header_ns <= 0:
                    raise ValueError('Duplicate bag timestamp or invalid header timestamp')
                series[stamp] = (header_ns, frame_id)
            output[tag] = series
    return output


def resolve_camera_timestamps(series, index_stamps):
    """Resolve exact bag OR header stamps; reject ambiguous or missing matches."""
    by_header = {}
    previous_header = -1
    for bag_ns in sorted(series):
        header_ns, _ = series[bag_ns]
        if header_ns <= previous_header:
            raise ValueError('Non-increasing camera header timestamps')
        previous_header = header_ns
        by_header[header_ns] = bag_ns
    result = []
    for stamp in index_stamps:
        bag_match = stamp if stamp in series else None
        header_match = by_header.get(stamp)
        if bag_match is not None and header_match is not None and bag_match != header_match:
            raise ValueError('Ambiguous index timestamp: bag/header match different messages')
        bag_ns = bag_match if bag_match is not None else header_match
        if bag_ns is None:
            raise ValueError('Index timestamp matches neither camera bag nor header')
        basis = 'bag' if bag_match is not None else 'header'
        result.append((bag_ns, *series[bag_ns], basis))
    return result


def reuse_priors(source_run, output_run, stamps, track, fusion):
    """Reuse exact full-run sensor observations for a fair short-window comparison."""
    source_run, output_run = Path(source_run), Path(output_run)
    validation = loads_saved_json((source_run / 'validation.json').read_text())
    if validation.get('track') != track or validation.get('sensor_fusion') != fusion or validation.get('profile') != 'sensor_guided':
        raise ValueError('Prior source Track/profile/fusion mismatch')
    for name in ('sensor_prior.csv', 'sensor_prior.fusion.csv'):
        path = source_run / name
        if not path.exists() and name.endswith('.fusion.csv') and fusion == 'fixed':
            continue
        with path.open() as stream:
            reader = csv.DictReader(stream)
            columns, rows = reader.fieldnames, list(reader)
        lookup = {int(row['timestamp_ns']): row for row in rows}
        if len(lookup) != len(rows) or any(s not in lookup for s in stamps):
            raise ValueError('Missing/duplicate prior timestamps')
        with (output_run / name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(lookup[s] for s in stamps)
    metadata = loads_saved_json((source_run / 'sensor_prior.json').read_text())
    metadata['frames'] = len(stamps)
    metadata['reused_prior'] = {'source_run': str(source_run.resolve()), 'source_sha256': sha256(source_run / 'sensor_prior.csv'),
                               'scope': 'Exact timestamp-selected original sensor prior values; map posterior not reused'}
    (output_run / 'sensor_prior.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')


def load_reconstruction(report_path, outputs_root, track, front_stamps, start, count):
    """Strict small reconstruction manifest; original indices must match front bag times."""
    report_path, outputs_root = Path(report_path).resolve(), Path(outputs_root).resolve()
    report_path.relative_to(outputs_root)
    report = loads_saved_json(report_path.read_text())
    if (report.get('schema') != 'calibrated_rig_v1' or report.get('track') != track
            or report.get('coordinate_frame') != 'front_camera_FLU'
            or report.get('reference_time') != 'front_camera_bag_timestamp' or report.get('ply_scale') != 1.):
        raise ValueError('Reconstruction frame/Track/unit contract mismatch')
    rows = report['frames']
    lookup = {row['original_idx']: row for row in rows}
    if len(lookup) != len(rows) or len({r['filename'] for r in rows}) != len(rows):
        raise ValueError('Duplicate reconstructed index/file')
    selected = []
    for index in range(start, start + count):
        if index not in lookup or index >= len(front_stamps):
            raise ValueError('Requested original frame not reconstructed')
        row = lookup[index]
        stamp = row['timestamp_ns']
        if not isinstance(stamp, int) or stamp != front_stamps[index]:
            raise ValueError('Reconstruction original index/timestamp mismatch')
        path = (report_path.parent / row['filename']).resolve()
        path.relative_to(report_path.parent)
        if (not path.is_file() or sha256(path) != row['sha256']
                or FRAME_COMMENT not in ply_header(path)[0]
                or not re.fullmatch(r'\d+_ts_' + str(stamp) + r'\.ply', path.name)):
            raise ValueError('Reconstruction PLY identity mismatch')
        selected.append((stamp, row['filename']))
    return report_path.parent, selected, report
