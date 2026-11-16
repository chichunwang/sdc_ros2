"""BA 用的學習式特徵軌跡：SuperPoint 或 ALIKED 抽特徵、LightGlue 配對（PyTorch CUDA）。

為什麼需要：基本 BA 的 SIFT 軌跡很短（中位數 4 幀）、在柏油路與樹叢這類低紋理區域幾乎沒有特徵，
跨圈軌跡也只有 7,304 條。學習式特徵在這些區域較穩定，LightGlue 用整張圖的上下文配對，錯配較少。
實測（outputs/cache/dev24_*，2611161129 的 BA 輸入）：觀測 33 萬 → 163 萬、跨圈軌跡 7.3 千 → 5.1 萬，
獨立的 IMU 加速度一致性 0.133 → 0.121 m/s（SuperPoint、ALIKED、與 SIFT 合併三種都約降 9%）。

只用在 BA：追蹤、閉環與 pose graph 仍用已驗證的 SIFT 流程，這裡只為「已決定好的影像對」產生額外的特徵匹配。
影像對 = 相鄰幀（及前 2…skip 幀）＋前端已接受的非相鄰影像對（閉環、跨圈）。每對都做 Essential RANSAC（1 px）驗證，
與 SIFT 的 BA 影像對同一規則。LightGlue 關閉提前停止（depth/width_confidence = −1），結果與輸入順序無關。
"""
import time

import cv2
import numpy as np


def build(load_image, n_frames, extra_pairs, K, cfg, seed=0, log=print):
    """回傳 (kpts: list[(N_i,2)], links: list[(a, b, (M,2))], info)。

    load_image(local) → BGR 影像；extra_pairs：[(a, b)] 非相鄰的影像對（a < b，本地幀索引）。
    """
    import torch
    from lightglue import ALIKED, LightGlue, SuperPoint
    if not torch.cuda.is_available():
        raise RuntimeError('學習式特徵需要 CUDA；不允許退回 CPU')
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    kind = cfg['extractor']
    Extractor = ALIKED if kind == 'aliked' else SuperPoint
    extractor = Extractor(max_num_keypoints=cfg['max_keypoints'], detection_threshold=0.0).eval().cuda()
    matcher = LightGlue(features=kind, depth_confidence=-1, width_confidence=-1).eval().cuda()

    t0 = time.perf_counter()
    feats = []
    for i in range(n_frames):
        img = load_image(i)
        if img is None:
            feats.append(None)
            continue
        x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255.0).cuda()
        with torch.inference_mode():
            feats.append(extractor.extract(x, resize=None))
    kpts = [np.zeros((0, 2)) if f is None else f['keypoints'][0].cpu().numpy().astype(np.float64) for f in feats]
    t_extract = time.perf_counter() - t0

    pairs = {(i - d, i) for i in range(n_frames) for d in range(1, cfg['skip_frames'] + 1) if i - d >= 0}
    pairs |= {(int(a), int(b)) for a, b in extra_pairs if a < b}
    links, failed, n_raw = [], 0, 0
    t1 = time.perf_counter()
    for a, b in sorted(pairs):
        if feats[a] is None or feats[b] is None:
            failed += 1
            continue
        with torch.inference_mode():
            out = matcher({'image0': feats[a], 'image1': feats[b]})
        m = out['matches'][0].cpu().numpy().astype(np.int64)
        n_raw += len(m)
        if len(m) < cfg['min_inliers']:
            failed += 1
            continue
        cv2.setRNGSeed(int(seed))
        _, mask = cv2.findEssentialMat(kpts[a][m[:, 0]], kpts[b][m[:, 1]], K, method=cv2.RANSAC, prob=0.999,
                                       threshold=cfg['ransac_px'])
        if mask is None or mask.sum() < cfg['min_inliers']:
            failed += 1
            continue
        links.append((a, b, m[mask.ravel().astype(bool)]))
    del feats
    torch.cuda.empty_cache()
    info = dict(extractor=kind, max_keypoints=cfg['max_keypoints'], pairs=len(pairs), failed_pairs=failed,
                matches_raw=int(n_raw), matches_kept=int(sum(len(m) for _, _, m in links)),
                extract_seconds=t_extract, match_seconds=time.perf_counter() - t1)
    log('BA：學習式特徵 %s，%d 幀抽取 %.0f s；%d 對影像、%d 個匹配（失敗 %d 對）%.0f s' % (
        kind, n_frames, t_extract, len(pairs), info['matches_kept'], failed, info['match_seconds']))
    return kpts, links, info


def merge(kpts_a, links_a, kpts_b, links_b):
    """兩組特徵併成一組：每幀特徵串接（b 的索引平移 len(a)），匹配各自保留（兩組特徵不互相連）。"""
    shift = [len(k) for k in kpts_a]
    kpts = [np.vstack([np.asarray(a, dtype=np.float64).reshape(-1, 2), np.asarray(b, dtype=np.float64).reshape(-1, 2)])
            for a, b in zip(kpts_a, kpts_b)]
    links = list(links_a) + [(a, b, np.asarray(m) + np.array([shift[a], shift[b]])) for a, b, m in links_b]
    return kpts, links
