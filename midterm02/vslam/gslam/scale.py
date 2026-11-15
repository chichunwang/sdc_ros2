"""逐幀尺度正規化：以「相機離地高度固定」當作不漂移的尺度基準。

觀察：LoGeR 點雲的尺度逐幀變動（Track01 前 300 幀，地面到相機的距離在 1.32–1.57 m 間變化，且有週期 3 的跳動），
而且這個變化與影像特徵估到的相鄰幀尺度比正相關（r≈0.57）。
物理事實：相機固定在車上，離地高度不變。所以 尺度 s_i = h_ref / h_i，其中 h_i 是第 i 幀點雲裡地面平面到相機原點的距離。
相鄰幀尺度比連乘會累積誤差（隨機漫步），地面高度則是每幀獨立的絕對量測，不會漂移。
"""
import numpy as np


def ground_height(xyz, cfg, seed=0):
    """在相機前方近處擬合地面平面，回傳 (相機離地高度 [點雲單位], 內點比例)；失敗回傳 (None, 0)。

    候選點：相機下方（y 向下為正）、前方近距離、左右有限範圍。RANSAC 找法向接近 y 軸的平面，再對內點做最小平方。
    """
    m = (xyz[:, 1] > cfg['min_below_m']) & (xyz[:, 2] < cfg['max_forward_m']) & (np.abs(xyz[:, 0]) < cfg['max_lateral_m'])
    p = xyz[m]
    if len(p) < cfg['min_points']:
        return None, 0.0
    rng = np.random.default_rng(seed)
    best, best_mask = 0, None
    for _ in range(cfg['iterations']):
        s = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        if abs(n[1]) < cfg['min_normal_y']:
            continue
        inliers = np.abs(p @ n - n @ s[0]) < cfg['inlier_m']
        count = int(inliers.sum())
        if count > best:
            best, best_mask = count, inliers
    if best_mask is None or best / len(p) < cfg['min_inlier_ratio']:
        return None, 0.0 if best_mask is None else best / len(p)
    q = p[best_mask]
    c = q.mean(axis=0)
    n = np.linalg.svd(q - c, full_matrices=False)[2][2]
    return float(abs(n @ c)), best / len(p)
