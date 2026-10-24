"""Lossless float32 XYZ extraction from ASCII PCD, without pose processing."""
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np

RUN_ROOT = Path(__file__).resolve().parents[1] / 'outputs/run'
BATCH_ROWS = 100_000


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _header(stream):
    header = {}
    while True:
        line = stream.readline()
        if not line:
            raise ValueError('PCD header is truncated')
        words = line.strip().split()
        if not words or words[0].startswith('#'):
            continue
        key = words[0].upper()
        if key in header:
            raise ValueError(f'duplicate PCD header field: {key}')
        header[key] = words[1:]
        if key == 'DATA':
            break
    required = ('FIELDS', 'TYPE', 'SIZE', 'COUNT', 'POINTS', 'WIDTH', 'HEIGHT', 'DATA')
    if any(key not in header for key in required):
        raise ValueError('missing PCD header fields')
    if header['DATA'] != ['ascii']:
        raise ValueError('only DATA ascii is supported')
    fields = header['FIELDS']
    if len(set(fields)) != len(fields):
        raise ValueError('duplicate PCD fields')
    if any(len(header[key]) != len(fields) for key in ('TYPE', 'SIZE', 'COUNT')):
        raise ValueError('PCD field schema lengths differ')
    counts = list(map(int, header['COUNT']))
    sizes = list(map(int, header['SIZE']))
    if any(value <= 0 for value in counts) or any(value <= 0 for value in sizes):
        raise ValueError('PCD size/count must be positive')
    if any(kind not in ('F', 'I', 'U') for kind in header['TYPE']):
        raise ValueError('PCD fields must be numeric')
    for key in ('POINTS', 'WIDTH', 'HEIGHT'):
        if len(header[key]) != 1 or int(header[key][0]) <= 0:
            raise ValueError(f'{key} must be one positive integer')
    points, width, height = (int(header[key][0]) for key in ('POINTS', 'WIDTH', 'HEIGHT'))
    if points != width * height:
        raise ValueError('POINTS differs from WIDTH * HEIGHT')
    indices = []
    for axis in ('x', 'y', 'z'):
        if axis not in fields:
            raise ValueError(f'missing {axis} field')
        index = fields.index(axis)
        if (header['TYPE'][index], sizes[index], counts[index]) != ('F', 4, 1):
            raise ValueError('XYZ must have TYPE F SIZE 4 COUNT 1')
        indices.append(sum(counts[:index]))
    return header, points, indices, sum(counts)


def _output_paths(output_dir):
    output_dir = Path(output_dir).absolute()
    root = RUN_ROOT.resolve()
    resolved = output_dir.resolve()
    relative = resolved.relative_to(root)
    if not relative.parts or not re.fullmatch(r'track0[0-3]_(?:full|test)_\d{10}(?:_\d{4})?', relative.parts[0]):
        raise ValueError('output_dir must be inside a GPU run')
    if not (root / relative.parts[0]).is_dir():
        raise ValueError('owning run must already exist')
    paths = [resolved / name for name in ('map_xyz.partial.npy', 'map_xyz.npy', 'map_xyz.json')]
    if any(path.exists() or path.is_symlink() for path in paths):
        raise FileExistsError('map cache, partial, or metadata already exists')
    resolved.mkdir(parents=True, exist_ok=True)
    return paths


def cache_xyz_map(source: Path, output_dir: Path) -> Path:
    """Stream unmodified XYZ into a new cache; failures preserve partial evidence."""
    partial, output, metadata = _output_paths(output_dir)
    source = Path(source).resolve(strict=True)
    before = _sha(source)
    source_bytes = source.stat().st_size
    with source.open('r', encoding='ascii') as stream:
        schema, points, indices, columns = _header(stream)
        # Reserve the pathname exclusively before NumPy writes the NPY header.
        with partial.open('xb'):
            pass
        mapped = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32,
                                          shape=(points, 3))
        written = 0
        try:
            while written < points:
                count = min(BATCH_ROWS, points - written)
                lines = []
                for _ in range(count):
                    line = stream.readline()
                    if not line:
                        raise ValueError(f'truncated PCD payload at point {written + len(lines)}')
                    lines.append(line)
                # Parse once in NumPy, rather than converting hundreds of
                # millions of individual tokens in a Python inner loop.
                numeric = np.loadtxt(io.StringIO(''.join(lines)), dtype=np.float64, ndmin=2)
                if numeric.shape != (count, columns) or not np.isfinite(numeric).all():
                    raise ValueError('invalid PCD columns or nonfinite payload')
                batch = numeric[:, indices].astype(np.float32)
                if batch.shape != (count, 3) or not np.isfinite(batch).all():
                    raise ValueError('invalid or nonfinite XYZ payload')
                mapped[written:written + count] = batch
                written += count
            if any(line.strip() for line in stream):
                raise ValueError('extra payload after declared POINTS')
            mapped.flush()
        finally:
            del mapped
    if _sha(source) != before or source.stat().st_size != source_bytes:
        raise ValueError('source changed during map extraction')
    record = {'source': str(source), 'source_sha256': before, 'source_bytes': source_bytes,
              'cache_sha256': _sha(partial), 'schema': schema, 'points': points,
              'dtype': 'float32', 'shape': [points, 3], 'voxel_applied': False,
              'crop_applied': False, 'pose_processing_applied': False,
              'point_count_changed': False}
    # Refuse a concurrent writer rather than replacing its output.
    with output.open('xb'):
        pass
    partial.replace(output)
    with metadata.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return output
