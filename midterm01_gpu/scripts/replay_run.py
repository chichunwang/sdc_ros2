"""Read-only saved-run validation and coordinate export (no ROS dependency)."""
from output_layout import latest_full_run, loads_saved_json, relocate_text, saved_path

import csv
import hashlib
import json
import math
import re
from pathlib import Path




def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return loads_saved_json(Path(path).read_text())


def map_container_path(path, root, data_root):
    """Translate only the GPU's two persistent mounts, including container symlinks."""
    path = saved_path(path)
    for prefix, local in [(Path('/root/data'), data_root), (Path('/root/outputs'), root / 'outputs')]:
        if path.is_relative_to(prefix):
            result = local / path.relative_to(prefix)
            # Staged source_ply links contain absolute *container* targets.
            if result.is_symlink():
                return map_container_path(result.readlink(), root, data_root)
            result = result.resolve()
            if not result.is_relative_to(local.resolve()):
                raise ValueError(f'來源路徑超出掛載範圍：{path}')
            return result
    raise ValueError(f'不支援的容器路徑：{path}')


def container_path(path, root, data_root):
    path = Path(path).resolve()
    for local, prefix in [(data_root.resolve(), '/root/data'), ((root / 'outputs').resolve(), '/root/outputs')]:
        if path.is_relative_to(local):
            return str(Path(prefix) / path.relative_to(local))
    raise ValueError(f'檔案未位於 GPU data／outputs 掛載內：{path}')


def resolve_run(root, run, track):
    path = saved_path(run).resolve() if run else latest_full_run(root,track)
    if not (path / 'params.json').is_file():
        if (path / f'track{track}/localization.json').is_file():
            path /= f'track{track}'
        if (path / 'localization.json').is_file():
            path = Path(read_json(path / 'localization.json')['run'])
    path = path.resolve()
    if not path.is_relative_to((root / 'outputs').resolve()) or not (path / 'params.json').is_file():
        raise ValueError('請指定 outputs 下含 params.json、frames.csv、result.txt 的定位 run。')
    return path


def rotation_values(pose):
    """R = Rz(yaw) Ry(pitch) Rx(roll); quaternion order x,y,z,w."""
    r = [[pose[4 * i + j] for j in range(3)] for i in range(3)]
    error = max(abs(sum(r[k][i] * r[k][j] for k in range(3)) - (i == j))
                for i in range(3) for j in range(3))
    det = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
           - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
           + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
    if error > 1e-3 or abs(det - 1) > 1e-3:
        raise ValueError('result.txt 含非正交旋轉或反射矩陣。')
    pitch = math.atan2(-r[2][0], math.hypot(r[0][0], r[1][0]))
    if math.hypot(r[0][0], r[1][0]) > 1e-8:
        roll, yaw = math.atan2(r[2][1], r[2][2]), math.atan2(r[1][0], r[0][0])
    else:
        roll, yaw = 0., math.atan2(-r[0][1], r[1][1])
    trace = sum(r[i][i] for i in range(3))
    if trace > 0:
        s = 2 * math.sqrt(1 + trace)
        q = [(r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s,
             (r[1][0] - r[0][1]) / s, s / 4]
    else:
        i = max(range(3), key=lambda k: r[k][k])
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * math.sqrt(1 + r[i][i] - r[j][j] - r[k][k])
        q = [0.] * 4
        q[i], q[j], q[k], q[3] = s / 4, (r[i][j] + r[j][i]) / s, (r[i][k] + r[k][i]) / s, (r[k][j] - r[j][k]) / s
    norm = math.sqrt(sum(v * v for v in q))
    return [v / norm for v in q], [roll, pitch, yaw]


def load_run(root, data_root, run=None, track=1, start=0, count=0):
    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    run = resolve_run(root, run, track)
    params = read_json(run / 'params.json')
    if params.get('replay_supported') is False or params.get('method') == 'map_scale_slow_v1':
        raise ValueError('此 run 使用逐幀尺度；目前 RViz 回放只支援固定 ply_scale，請選擇固定尺度的既有 run。')
    match = re.fullmatch(r'/root/data/Track([12])', params['map_path'])
    if not match or int(match[1]) != track:
        raise ValueError('run 的 Track 與 --track 不一致。')
    scale = float(params['ply_scale'])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('無效的原始 ply_scale。')
    for key in ('input_voxel_leaf_size', 'map_voxel_leaf_size', 'submap_half_size', 'submap_half_height'):
        value = float(params[key])
        if not math.isfinite(value) or value < 0 or (key.startswith('submap') and value == 0):
            raise ValueError(f'無效的原 run 參數：{key}')
    map_file = map_container_path(str(Path(params['map_path']) / params['map_file']), root, data_root)
    if not map_file.is_file():
        raise ValueError(f'找不到地圖：{map_file}')
    with (run / 'frames.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['idx', 'timestamp_ns', 'filename']:
            raise ValueError('frames.csv 表頭不符。')
        frames = list(reader)
    result_file = run / params['save_file']
    poses = []
    for line in result_file.read_text().splitlines():
        row = [float(v) for v in line.split()]
        if len(row) != 12 or not all(math.isfinite(v) for v in row):
            raise ValueError('result.txt 每列必須是 12 個有限數值。')
        rotation_values(row)
        poses.append(row)
    if not frames or len(frames) != len(poses):
        raise ValueError('frames.csv／result.txt 筆數不符或為空；不回放未完成輸出。')
    source_path = run / 'source_input.json'
    source = read_json(source_path) if source_path.exists() else {}
    if source and source.get('ply_scale') != scale:
        raise ValueError('source_input.json 與 params.json 的尺度不符。')
    selected = source.get('selected_frames')
    if selected is not None and len(selected) != len(frames):
        raise ValueError('source_input.json 幀數不符。')
    timestamps, filenames = set(), set()
    previous = 0
    for index, frame in enumerate(frames):
        stamp = int(frame['timestamp_ns'])
        name = Path(frame['filename'])
        timestamp_match = re.fullmatch(r'.+_ts_([0-9]+)\.ply', name.name, re.IGNORECASE)
        if (int(frame['idx']) != index or stamp <= previous or stamp >= 2**63 or stamp in timestamps
                or frame['filename'] in filenames or name.is_absolute() or '..' in name.parts
                or not timestamp_match or int(timestamp_match[1]) != stamp):
            raise ValueError(f'frame {index} 的索引／整數時間／檔名不一致。')
        previous = stamp
        timestamps.add(stamp)
        filenames.add(frame['filename'])
        if selected is not None:
            identity = selected[index]
            if identity['timestamp_ns'] != stamp or Path(identity['filename']).name != name.name:
                raise ValueError('source_input.json 時間／檔名不符。')
    if start < 0 or count < 0 or start >= len(frames) or (count and start + count > len(frames)):
        raise ValueError('回放區間超出原 run；--start-frame 為 run 內的 0-based 索引。')
    stop = start + count if count else len(frames)
    records = []
    for index in range(start, stop):
        frame, pose = frames[index], poses[index]
        path = map_container_path(str(Path(params['ply_path']) / frame['filename']), root, data_root)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'找不到原始點雲：{path}')
        digest = sha256(path)
        if selected is not None and digest != selected[index]['sha256']:
            raise ValueError(f'原始點雲 SHA256 不符：{path}')
        q, angles = rotation_values(pose)
        records.append(dict(idx=index, original_idx=selected[index]['original_idx'] if selected else index,
                            timestamp_ns=int(frame['timestamp_ns']), path=str(path),
                            container_path=container_path(path, root, data_root), sha256=digest,
                            pose=pose, quaternion=q, rpy=angles))
    map_digest = sha256(map_file)
    display_map, cache_info = map_file, None
    # Existing cache is read only and accepted only with the benchmark's exact
    # Track/map/leaf identity. Saved poses are never recomputed.
    for sidecar in sorted((root / 'outputs/run').glob('*/cache*.pcd.json')):
        try:
            identity = read_json(sidecar)
        except (OSError, ValueError):
            continue  # Never use an unreadable/unverifiable display cache.
        cache = sidecar.with_suffix('')
        if (identity.get('track') == track and identity.get('map_sha256') == map_digest
                and identity.get('map_voxel_leaf_size') == params['map_voxel_leaf_size'] and cache.is_file()):
            display_map = cache
            cache_info = dict(file=str(cache), sha256=sha256(cache), sidecar_sha256=sha256(sidecar), identity=identity)
            break
    identity_files = [run / 'frames.csv', result_file, run / 'params.json'] + ([source_path] if source else [])
    return dict(run=str(run), track=track, total_frames=len(frames), params=params, records=records,
                map_file=container_path(display_map, root, data_root), map_sha256=map_digest, map_cache=cache_info,
                source_sha256={p.name: sha256(p) for p in identity_files},
                cloud_source=source.get('cloud_source', 'front'),
                historical_cloud_hash_checked=selected is not None,
                coordinate_contract='p_map = R_map_source * (ply_scale * p_raw) + t_map_source; source origin, not confirmed body origin')


def write_replay(data, output):
    output = Path(output)
    with (output / 'frames.tsv').open('x') as stream:
        stream.write('idx\toriginal_idx\ttimestamp_ns\tply_file\tpose_3x4\n')
        for row in data['records']:
            if any(c in row['container_path'] for c in '\t\n\r'):
                raise ValueError('點雲路徑不可含 tab／換行。')
            stream.write(f"{row['idx']}\t{row['original_idx']}\t{row['timestamp_ns']}\t{row['container_path']}\t"
                         + ' '.join(format(v, '.17g') for v in row['pose']) + '\n')
    with (output / 'poses.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['idx', 'original_idx', 'timestamp_ns', 'x_m', 'y_m', 'z_m', 'qx', 'qy', 'qz', 'qw',
                         'roll_rad', 'pitch_rad', 'yaw_rad', 'roll_deg', 'pitch_deg', 'yaw_deg'])
        for row in data['records']:
            writer.writerow([row['idx'], row['original_idx'], row['timestamp_ns'],
                             *[row['pose'][i] for i in (3, 7, 11)], *row['quaternion'], *row['rpy'],
                             *[math.degrees(v) for v in row['rpy']]])
    (output / 'replay.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
