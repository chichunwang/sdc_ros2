"""Strict XYZ input; no sensor fusion, pose estimation or historical trajectories."""
import json
from pathlib import Path
import re

import numpy as np

from .contracts import Frame, Similarity

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    import hashlib
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def select_frames(data_root, *, full=False, count=32):
    frames = []
    for p in (Path(data_root)/'Track3/raw_ply').glob('*.ply'):
        match = re.fullmatch(r'(\d+)_ts_(\d+)\.ply', p.name)
        if match is None:
            raise ValueError('Unrecognized PLY filename: '+p.name)
        idx, ns = map(int, match.groups())
        frames.append(Frame(idx, ns, p.resolve()))
    frames.sort(key=lambda f: f.original_idx)
    if (not frames or [f.original_idx for f in frames] != list(range(len(frames)))
            or frames[0].timestamp_ns <= 0
            or any(b.timestamp_ns <= a.timestamp_ns for a, b in zip(frames, frames[1:]))):
        raise ValueError('Need complete 0-based indices and increasing integer nanoseconds')
    if full and len(frames) != 3142:
        raise ValueError('Track3 full requires exactly 3142 frames')
    selected = frames if full else frames[:count]
    if not selected or (not full and (isinstance(count, bool) or count <= 0 or len(selected) != count)):
        raise ValueError('Requested prefix exceeds available frames')
    return selected


def read_initial(path):
    tokens = Path(path).read_text().replace(',', ' ').split()
    if len(tokens) != 16:
        raise ValueError('init_guess must contain exactly 16 numeric values')
    return Similarity(np.array([float(v) for v in tokens]).reshape(4, 4), 1.).pose


def load_ply_xyz(path):
    """Original float32 XYZ promoted to float64; RGB is ignored."""
    with Path(path).open('rb') as stream:
        lines = []
        for _ in range(128):
            line = stream.readline()
            if not line:
                raise ValueError('Truncated PLY header')
            lines.append(line.decode('ascii').strip())
            if lines[-1] == 'end_header':
                break
        else:
            raise ValueError('PLY header too long')
        offset = stream.tell()
    formats = [v for v in lines if v.startswith('format ')]
    elements = [v for v in lines if v.startswith('element ')]
    props = [v for v in lines if v.startswith('property ')]
    if (lines[0] != 'ply' or len(formats) != 1 or len(elements) != 1
            or not re.fullmatch(r'element vertex \d+', elements[0])
            or props != ['property float x', 'property float y', 'property float z',
                         'property uchar red', 'property uchar green', 'property uchar blue']):
        raise ValueError('Unsupported PLY schema')
    count = int(elements[0].split()[2])
    if formats[0] == 'format binary_little_endian 1.0':
        dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', 'u1', (3,))])
        if Path(path).stat().st_size != offset+count*dtype.itemsize:
            raise ValueError('Binary PLY payload length mismatch')
        packed = np.fromfile(path, dtype=dtype, offset=offset, count=count)
        points = np.column_stack([packed[k] for k in 'xyz']).astype(np.float64)
    elif formats[0] == 'format ascii 1.0':
        with Path(path).open() as stream:
            for _ in lines:
                next(stream)
            values = np.loadtxt(stream, ndmin=2)
        if values.shape != (count, 6):
            raise ValueError('ASCII PLY vertex count/schema mismatch')
        points = values[:, :3].astype(np.float64)
    else:
        raise ValueError('Unsupported PLY format')
    if points.shape != (count, 3) or count < 8 or not np.isfinite(points).all():
        raise ValueError('Expected at least eight finite XYZ points')
    return points


def verified_cache(cache, source):
    cache = Path(cache).resolve(strict=True)
    cache.relative_to((ROOT/'outputs/run').resolve())
    metadata = json.loads((cache.parent/'map_xyz.json').read_text())
    if digest(source) != metadata['source_sha256'] or digest(cache) != metadata['cache_sha256']:
        raise ValueError('Map/cache SHA differs from recorded source')
    points = np.load(cache, mmap_mode='r', allow_pickle=False)
    if (list(points.shape) != metadata['shape'] or str(points.dtype) != metadata['dtype']
            or points.ndim != 2 or points.shape[1] != 3 or len(points) != metadata['points']
            or metadata.get('crop_applied') is not False or metadata.get('voxel_applied') is not False
            or metadata.get('pose_processing_applied') is not False):
        raise ValueError('Expected lossless, complete, unprocessed map XYZ cache')
    return points, dict(path=str(cache), metadata=metadata)


def sample_source(points, geometry, *, leaf=.03, max_points=12000):
    """GPU voxel membership, original XYZ rows; small boundary decisions on CPU."""
    import torch
    raw = geometry.tensor(points)
    cells_float = torch.floor(raw/leaf)
    boundary = (abs(raw/leaf-torch.round(raw/leaf)) <= 1e-10).nonzero().cpu().numpy()
    if len(boundary):
        values = np.floor(points[boundary[:, 0], boundary[:, 1]]/leaf)
        cells_float[boundary[:, 0], boundary[:, 1]] = geometry.tensor(values)
    if bool(((cells_float < -(2**63)) | (cells_float >= 2**63)).any()):
        raise ValueError('Voxel coordinates exceed int64')
    cells = cells_float.to(torch.int64)
    low = cells.amin(0)
    sizes = (cells.amax(0)-low+1).cpu().tolist()
    if int(sizes[0])*int(sizes[1])*int(sizes[2]) >= 2**63:
        raise ValueError('Voxel key exceeds int64')
    local = cells-low
    keys = (local[:, 0]*sizes[1]+local[:, 1])*sizes[2]+local[:, 2]
    order = torch.argsort(keys, stable=True)
    ordered = keys[order]
    first = torch.ones(len(order), dtype=torch.bool, device=raw.device)
    first[1:] = ordered[1:] != ordered[:-1]
    keep = torch.sort(order[first]).values.cpu().numpy()
    if len(keep) > max_points:
        keep = keep[np.linspace(0, len(keep)-1, max_points, dtype=int)]
    return np.array(points[keep], dtype=np.float64, copy=True)


def crop_map(points, center, geometry, *, half_size=65., leaf=.25):
    import torch
    low, high = geometry.tensor(np.asarray(center)-half_size), geometry.tensor(np.asarray(center)+half_size)
    chunks = []
    for start in range(0, len(points), 262144):
        block = geometry.tensor(np.array(points[start:start+262144], dtype=np.float64))
        if not bool(torch.isfinite(block).all()):
            raise ValueError('Map cache contains nonfinite XYZ')
        keep = ((block >= low) & (block <= high)).all(1)
        if bool(keep.any()):
            chunks.append(block[keep])
    if not chunks:
        raise ValueError('Empty static map tile')
    cropped = torch.cat(chunks).cpu().numpy()
    sampled = sample_source(cropped, geometry, leaf=leaf, max_points=len(cropped))
    return sampled, dict(center_m=np.asarray(center).tolist(), half_size_m=half_size,
                         raw_crop_points=len(cropped), voxel_points=len(sampled), leaf_m=leaf)
