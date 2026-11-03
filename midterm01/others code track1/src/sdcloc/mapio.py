"""地圖工具：載入 npy、voxel 降取樣、方框裁切。"""
import numpy as np


def load_map(npy_path, mmap=True):
    return np.load(npy_path, mmap_mode='r' if mmap else None)


def voxel_downsample(xyz, size, extra=None):
    """每個 voxel 取質心；extra（N,k）一起平均。回傳 (xyz_ds, extra_ds, counts)。"""
    xyz = np.asarray(xyz, dtype=np.float64)
    keys = np.floor(xyz / size).astype(np.int64)
    keys -= keys.min(axis=0)
    span = keys.max(axis=0) + 1
    flat = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
    uniq, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
    out = np.zeros((len(uniq), 3))
    for k in range(3):
        out[:, k] = np.bincount(inv, weights=xyz[:, k], minlength=len(uniq)) / counts
    ex = None
    if extra is not None:
        extra = np.asarray(extra, dtype=np.float64).reshape(len(xyz), -1)
        ex = np.column_stack([np.bincount(inv, weights=extra[:, k], minlength=len(uniq)) / counts
                              for k in range(extra.shape[1])])
    return out, ex, counts


def voxel_downsample_tiled(xyz, size, extra=None, tile=100.0):
    """大點雲分 x 帶處理（帶邊對齊 voxel 格），省記憶體；結果和一次處理相同（邊界 voxel 最多重複一次）。"""
    x = np.asarray(xyz[:, 0])
    k0, k1 = int(np.floor(x.min() / tile)), int(np.floor(x.max() / tile))
    outs, exs, cnts = [], [], []
    for k in range(k0, k1 + 1):
        m = (x >= k * tile) & (x < (k + 1) * tile)
        if not m.any():
            continue
        ds, ex, c = voxel_downsample(xyz[m], size, None if extra is None else np.asarray(extra)[m])
        outs.append(ds); cnts.append(c)
        if ex is not None:
            exs.append(ex)
    return np.vstack(outs), (np.vstack(exs) if exs else None), np.concatenate(cnts)


def crop_box(xyz, center, half_xy, half_z):
    c = np.asarray(center, dtype=float)
    m = ((np.abs(xyz[:, 0] - c[0]) <= half_xy) & (np.abs(xyz[:, 1] - c[1]) <= half_xy)
         & (np.abs(xyz[:, 2] - c[2]) <= half_z))
    return m
