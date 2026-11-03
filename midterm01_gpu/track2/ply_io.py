"""Track2 ASCII PLY 批次讀取；保留 float32 -> float64 與原始點序。

只為課程實際提供的單一 vertex XYZ/RGB layout 啟用快路徑；其餘格式仍
交給既有 plyfile reader。磁碟解碼是 CPU I/O，配準仍在 CUDA。
"""
from pathlib import Path
import numpy as np
from track3.io import load_ply as legacy_load_ply

_LAYOUT = [b'property float x', b'property float y', b'property float z',
           b'property uchar red', b'property uchar green', b'property uchar blue']
_DTYPE = np.dtype([(n, t) for n, t in zip(
    ('x', 'y', 'z', 'red', 'green', 'blue'), ('f4', 'f4', 'f4', 'u1', 'u1', 'u1'))])


def _known_ascii_vertex_count(stream):
    """嚴格辨識唯一支援的 layout；未辨識時回傳 None。"""
    header = []
    for _ in range(256):
        line = stream.readline()
        if not line or len(line) > 8192:
            return None
        line = line.strip()
        if line.startswith((b'comment ', b'obj_info ')):
            continue
        header.append(line)
        if line == b'end_header':
            break
    else:
        return None
    if len(header) != 10 or header[:2] != [b'ply', b'format ascii 1.0']:
        return None
    vertex = header[2].split()
    if len(vertex) != 3 or vertex[:2] != [b'element', b'vertex'] or not vertex[2].isdigit():
        return None
    if header[3:9] != _LAYOUT or header[9] != b'end_header':
        return None
    return int(vertex[2])


def load_ply(path):
    with Path(path).open('rb') as stream:
        count = _known_ascii_vertex_count(stream)
        if count is None:
            return legacy_load_ply(path)
        if count == 0:
            if stream.read().strip():
                return legacy_load_ply(path)
            return np.empty((0, 3), dtype=np.float64)
        try:
            # Structured dtype retains each original declared field width;
            # extracting XYZ afterwards exactly follows the legacy contract.
            vertices = np.loadtxt(stream, dtype=_DTYPE, comments=None, ndmin=1)
        except (ValueError, TypeError, OverflowError):
            return legacy_load_ply(path)
        if len(vertices) != count:
            return legacy_load_ply(path)
    xyz = np.column_stack([vertices[n] for n in ('x', 'y', 'z')]).astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]
