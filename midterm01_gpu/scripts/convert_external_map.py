#!/usr/bin/env python3
"""binary PCD → 作者 Nx4 float32 cache；只讀 XYZ，第四欄補零作格式相容。"""
from pathlib import Path
import json
import sys


def read_pcd_header(path):
    header = {}
    with Path(path).open("rb") as stream:
        for _ in range(100):
            raw = stream.readline()
            if not raw:
                raise ValueError("PCD 缺少 DATA header")
            line = raw.decode("ascii").strip()
            if not line or line.startswith("#"):
                continue
            key, *values = line.split()
            header[key] = values
            if key == "DATA":
                header["payload_offset"] = stream.tell()
                return header
    raise ValueError("PCD header 過長")


def convert(source, destination):
    import numpy as np
    source, destination = Path(source), Path(destination)
    header = read_pcd_header(source)
    if header["DATA"] != ["binary"]:
        raise ValueError("本介面只支援 uncompressed binary PCD")
    fields, sizes, types = header["FIELDS"], list(map(int, header["SIZE"])), header["TYPE"]
    counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
    if len(set(fields)) != len(fields) or not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("PCD 欄位描述不一致")
    offset = 0
    names, formats, offsets = [], [], []
    for name, size, kind, count in zip(fields, sizes, types, counts):
        if size <= 0 or count <= 0:
            raise ValueError("PCD SIZE/COUNT 必須為正")
        if name in ("x", "y", "z"):
            if kind != "F" or size not in (4, 8) or count != 1:
                raise ValueError("XYZ 必須是單一 float32/64")
            names.append(name); formats.append(f"<f{size}"); offsets.append(offset)
        # tl/th 只計算 byte offset，不建立 timestamp 欄位、不解碼值。
        offset += size * count
    if set(names) != {"x", "y", "z"}:
        raise ValueError("PCD 缺少 XYZ")
    count = int(header["POINTS"][0])
    if count <= 0 or source.stat().st_size != header["payload_offset"] + count * offset:
        raise ValueError("PCD 點數／binary payload 長度不符")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(dict(names=names, formats=formats, offsets=offsets, itemsize=offset))
    records = np.memmap(source, mode="r", dtype=dtype, offset=header["payload_offset"], shape=(count,))
    output = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32, shape=(count, 4))
    for start in range(0, count, 250_000):
        end = min(start + 250_000, count)
        for axis, name in enumerate(("x", "y", "z")):
            values = records[name][start:end]
            if not np.isfinite(values).all():
                raise ValueError("地圖 XYZ 含非有限數值")
            output[start:end, axis] = values
        output[start:end, 3] = 0.0
    output.flush()
    print(json.dumps(dict(points=count, fields_used=["x", "y", "z"],
                          fourth_column="zero compatibility padding; not a measurement",
                          map_timestamps_used=False)), flush=True)


if __name__ == "__main__":
    convert(*sys.argv[1:])
