"""raw_ply 檔名 → (idx, timestamp_ns)。檔名格式 000123_ts_<奈秒>.ply，照時間排序。"""
import re
from pathlib import Path

import numpy as np

_PAT = re.compile(r'(\d+)_ts_(\d+)\.ply$')


def parse_frame_names(names):
    rows = []
    for name in names:
        m = _PAT.search(name)
        if m:
            rows.append((int(m.group(2)), int(m.group(1))))
    rows.sort()
    ts = np.array([r[0] for r in rows], dtype=np.int64)
    file_idx = np.array([r[1] for r in rows], dtype=np.int64)
    if len(ts) and (np.any(np.diff(ts) <= 0) or not np.array_equal(file_idx, np.arange(len(rows)))):
        raise ValueError('frame timestamps not strictly increasing or file idx not contiguous')
    return ts, file_idx


def load_frames(ply_dir):
    return parse_frame_names(p.name for p in Path(ply_dir).glob('*.ply'))
