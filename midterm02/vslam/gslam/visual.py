"""視覺前端：影像特徵、GPU 描述子比對、幾何驗證、像素→3D 對應、PnP 相對位姿、除錯圖。

哪些在 GPU：描述子相似度矩陣、ratio test、互為最近鄰檢查（CuPy）；閉環檢索的描述子投票（CuPy）。
哪些在 CPU：SIFT／ORB 特徵抽取（pip 版 OpenCV 沒有 CUDA 模組）、RANSAC（Essential、PnP，資料量很小）。
選用的 SuperPoint + LightGlue 則抽取與比對都在 GPU（PyTorch），見 LearnedFrontend。
"""
import threading
import time

import cv2
import numpy as np
import cupy as cp

from . import se3


# ---------------------------------------------------------------- 特徵抽取
class Features:
    """一張影像的特徵：kpts (N,2) 像素座標、desc（GPU，已正規化的 float32）、原始資料 raw（學習式前端用）。"""

    def __init__(self, kpts, desc, raw=None, size=None):
        self.kpts = np.asarray(kpts, dtype=np.float64).reshape(-1, 2)
        # 特徵被偵測到的尺度（px）；BA 用它估計定位誤差。沒有這項資訊的前端一律當 6 px（權重 1）
        self.size = np.full(len(self.kpts), 6.0) if size is None else np.asarray(size, dtype=np.float64)
        self.desc = desc
        self.raw = raw

    def __len__(self):
        return len(self.kpts)


class ClassicFrontend:
    """SIFT（RootSIFT）或 ORB 抽取（CPU）＋ CuPy 比對（GPU）。"""

    cpu_parallel = True    # detect_cpu 可在多個執行緒同時跑（每個執行緒各自一個 detector），供 pipeline 預先抽取

    def __init__(self, cfg):
        self.cfg = cfg
        self.kind = cfg['frontend']
        if self.kind not in ('sift', 'orb'):
            raise ValueError('未知的傳統前端：%s' % self.kind)
        self._local = threading.local()
        self.detector = self._detector()

    def _detector(self):
        """每個執行緒一個 detector（OpenCV 的 Feature2D 物件不保證可跨執行緒共用）；參數相同，輸出相同。"""
        det = getattr(self._local, 'det', None)
        if det is None:
            det = (cv2.SIFT_create(nfeatures=self.cfg['max_keypoints']) if self.kind == 'sift'
                   else cv2.ORB_create(nfeatures=self.cfg['max_keypoints']))
            self._local.det = det
        return det

    def detect_cpu(self, image_bgr):
        """CPU 部分（可在背景執行緒）：回傳 (像素座標, 原始描述子, 尺度)。"""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        kp, desc = self._detector().detectAndCompute(gray, None)
        if desc is None or len(kp) == 0:
            return None
        return np.array([k.pt for k in kp], dtype=np.float64), desc, [k.size for k in kp]

    def from_cpu(self, det):
        """GPU 部分（主執行緒）：描述子正規化後放上 GPU。"""
        if det is None:
            return Features(np.zeros((0, 2)), cp.zeros((0, 128 if self.kind == 'sift' else 256), dtype=cp.float32))
        pts, desc, size = det
        return Features(pts, normalize_descriptors(desc, self.kind), size=size)

    def extract(self, image_bgr):
        return self.from_cpu(self.detect_cpu(image_bgr))

    def match(self, fa, fb):
        return match_descriptors(fa.desc, fb.desc, self.cfg['ratio'], self.cfg['mutual'])


def normalize_descriptors(desc, kind):
    """把描述子轉成 GPU 上的單位向量，之後用內積當相似度。

    SIFT → RootSIFT（L1 正規化後開根號，L2 範數自然為 1）。
    ORB  → 把 256 個 bit 展開成 ±1/16 的向量，內積 = 1 − 2·Hamming/256，排序與 Hamming 距離一致。
    """
    if kind == 'sift':
        d = cp.asarray(desc, dtype=cp.float32)
        d = d / (cp.sum(cp.abs(d), axis=1, keepdims=True) + 1e-7)
        return cp.sqrt(d)
    d = cp.asarray(desc, dtype=cp.uint8)
    bits = cp.unpackbits(d.ravel()).reshape(len(d), -1).astype(cp.float32)
    return (2.0 * bits - 1.0) / 16.0


def match_descriptors(da, db, ratio=0.8, mutual=True):
    """GPU 比對：回傳 (M,2) 的索引對（NumPy）。距離 d² = 2 − 2·sim，ratio test 用 d1/d2 < ratio。"""
    if len(da) < 2 or len(db) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    sim = da @ db.T
    best = cp.argmax(sim, axis=1)
    rows = cp.arange(len(da))
    s1 = sim[rows, best]
    sim2 = sim.copy()
    sim2[rows, best] = -2.0
    s2 = cp.max(sim2, axis=1)
    d1 = cp.sqrt(cp.maximum(2.0 - 2.0 * s1, 0.0))
    d2 = cp.sqrt(cp.maximum(2.0 - 2.0 * s2, 1e-12))
    keep = d1 < ratio * d2
    if mutual:
        back = cp.argmax(sim, axis=0)
        keep &= back[best] == rows
    ia = cp.flatnonzero(keep)
    return cp.asnumpy(cp.stack([ia, best[ia]], axis=1)).astype(np.int64)


class LearnedFrontend:
    """SuperPoint + LightGlue（PyTorch CUDA）。需 `scripts/dev.sh build-learned`（官方規則允許的公開預訓練模型）。"""

    def __init__(self, cfg):
        import torch
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import rbd
        if not torch.cuda.is_available():
            raise RuntimeError('SuperPoint/LightGlue 需要 CUDA；不允許退回 CPU')
        self.torch, self.rbd = torch, rbd
        self.extractor = SuperPoint(max_num_keypoints=cfg['max_keypoints']).eval().cuda()
        self.matcher = LightGlue(features='superpoint').eval().cuda()

    def extract(self, image_bgr):
        torch = self.torch
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).cuda()
        with torch.inference_mode():
            feats = self.extractor.extract(tensor)
        kpts = feats['keypoints'][0].detach().cpu().numpy()
        desc = cp.asarray(feats['descriptors'][0].detach().cpu().numpy(), dtype=cp.float32)
        desc = desc / (cp.linalg.norm(desc, axis=1, keepdims=True) + 1e-7)
        return Features(kpts, desc, raw=feats)

    def match(self, fa, fb):
        with self.torch.inference_mode():
            out = self.matcher({'image0': fa.raw, 'image1': fb.raw})
        return self.rbd(out)['matches'].detach().cpu().numpy().astype(np.int64)


def make_frontend(cfg):
    if cfg['frontend'] == 'superpoint_lightglue':
        return LearnedFrontend(cfg)
    return ClassicFrontend(cfg)


# ---------------------------------------------------------------- 像素 → 3D
class PixelCloud:
    """用相機內參把該幀點雲投影回影像，取得特徵點位置的 3D 座標。

    不假設 PLY 的點順序等於像素順序，只看投影位置：
    - 若投影位置構成規則格點（LoGeR 輸出是低解析度深度圖，約每 4.6 px 一點）：在特徵點周圍四個格點之間
      對深度做雙線性內插，再沿特徵點的視線反投影。四個格點深度差太大（深度不連續的邊緣）則不給 3D。
    - 否則：退回「像素格 → 最近的點」查表。
    這個對應是否可信，由 scripts/inspect_data.py 以「投影位置的影像顏色 vs 點的顏色」檢查。
    """

    def __init__(self, xyz, K, size, stride=5, max_depth=30.0, max_relative_gap=0.08):
        w, h = size
        self.K, self.max_gap = K, max_relative_gap
        z = xyz[:, 2]
        ok = np.all(np.isfinite(xyz), axis=1) & (z > 0.1)
        ids = np.flatnonzero(ok)
        u = K[0, 0] * xyz[ids, 0] / z[ids] + K[0, 2]
        v = K[1, 1] * xyz[ids, 1] / z[ids] + K[1, 2]
        uu, vv = np.unique(np.round(u, 1)), np.unique(np.round(v, 1))
        self.lattice = bool(len(uu) > 1 and len(vv) > 1 and len(uu) * len(vv) == len(ids))
        if self.lattice:
            self.u0, self.v0 = uu[0], vv[0]
            self.du, self.dv = (uu[-1] - uu[0]) / (len(uu) - 1), (vv[-1] - vv[0]) / (len(vv) - 1)
            iu = np.rint((u - self.u0) / self.du).astype(np.int64)
            iv = np.rint((v - self.v0) / self.dv).astype(np.int64)
            self.depth = np.full((len(vv), len(uu)), np.nan)
            self.depth[iv, iu] = np.where(z[ids] < max_depth, z[ids], np.nan)
            self.coverage = float(np.isfinite(self.depth).mean())
            return
        self.stride = stride
        self.shape = (h // stride + 1, w // stride + 1)
        near = z[ids] < max_depth
        ids, u, v = ids[near], u[near], v[near]
        iu = np.rint(u / stride).astype(np.int64)
        iv = np.rint(v / stride).astype(np.int64)
        inside = (iu >= 0) & (iu < self.shape[1]) & (iv >= 0) & (iv < self.shape[0])
        ids, iu, iv = ids[inside], iu[inside], iv[inside]
        order = np.argsort(-z[ids], kind='stable')  # 遠的先寫、近的後寫，重複格子由近的蓋過
        self.table = np.full(self.shape, -1, dtype=np.int64)
        self.table[iv[order], iu[order]] = ids[order]
        self.xyz = xyz
        self.coverage = float(np.count_nonzero(self.table >= 0)) / self.table.size

    def lookup(self, kpts):
        """回傳 (pts3d (N,3), valid (N,))；沒有可靠 3D 的特徵 valid=False。"""
        kpts = np.asarray(kpts, dtype=np.float64).reshape(-1, 2)
        pts = np.zeros((len(kpts), 3))
        if not self.lattice:
            iu = np.clip(np.rint(kpts[:, 0] / self.stride).astype(np.int64), 0, self.shape[1] - 1)
            iv = np.clip(np.rint(kpts[:, 1] / self.stride).astype(np.int64), 0, self.shape[0] - 1)
            ids = self.table[iv, iu]
            valid = ids >= 0
            pts[valid] = self.xyz[ids[valid]]
            return pts, valid
        gx = (kpts[:, 0] - self.u0) / self.du
        gy = (kpts[:, 1] - self.v0) / self.dv
        rows, cols = self.depth.shape
        inside = (gx >= 0) & (gx <= cols - 1) & (gy >= 0) & (gy <= rows - 1)
        x0 = np.clip(np.floor(gx).astype(np.int64), 0, cols - 2)
        y0 = np.clip(np.floor(gy).astype(np.int64), 0, rows - 2)
        fx, fy = gx - x0, gy - y0
        d = np.stack([self.depth[y0, x0], self.depth[y0, x0 + 1], self.depth[y0 + 1, x0], self.depth[y0 + 1, x0 + 1]], 1)
        finite = np.all(np.isfinite(d), axis=1)
        dmin, dmax = np.min(np.where(finite[:, None], d, 1.0), axis=1), np.max(np.where(finite[:, None], d, 1.0), axis=1)
        valid = inside & finite & ((dmax - dmin) <= self.max_gap * dmin)
        zi = (d[:, 0] * (1 - fx) * (1 - fy) + d[:, 1] * fx * (1 - fy) + d[:, 2] * (1 - fx) * fy + d[:, 3] * fx * fy)
        zi = np.where(valid, zi, 0.0)
        pts[:, 0] = (kpts[:, 0] - self.K[0, 2]) / self.K[0, 0] * zi
        pts[:, 1] = (kpts[:, 1] - self.K[1, 2]) / self.K[1, 1] * zi
        pts[:, 2] = zi
        return pts, valid


# ---------------------------------------------------------------- 相對位姿
def translation_length(R, t_dir, pts3d_ref, kpts_cur, K):
    """已知旋轉 R 與平移方向 t_dir（單位向量，x_cur = R·x_ref + λ·t_dir），用 ref 幀的 3D 點估計平移長度 λ。

    每個點的投影式 (X' + λ·t)_xy = uv·(X' + λ·t)_z（X' = R·X，uv 為正規化像素座標）是 λ 的兩條線性方程 a·λ = b。
    各點的最小平方解 λ_i = (a·b)/(a·a)；|a| 小代表該點對 λ 不敏感（視差小），所以只取 |a| 較大的一半，再取中位數。
    回傳 (λ, MAD, 使用點數)。深度有誤差的點只會讓個別 λ_i 偏掉，中位數不受少數離群值影響。
    """
    Xp = pts3d_ref @ R.T
    uv = np.c_[(kpts_cur[:, 0] - K[0, 2]) / K[0, 0], (kpts_cur[:, 1] - K[1, 2]) / K[1, 1]]
    a = t_dir[:2][None, :] - uv * t_dir[2]
    b = uv * Xp[:, 2:3] - Xp[:, :2]
    aa = np.sum(a * a, axis=1)
    keep = aa >= np.median(aa)
    lam = np.sum(a[keep] * b[keep], axis=1) / np.maximum(aa[keep], 1e-12)
    med = float(np.median(lam))
    return med, float(np.median(np.abs(lam - med))), int(keep.sum())


def verify_matches(kpts_a, kpts_b, matches, K, threshold_px, min_inliers, seed=0):
    """只做幾何驗證（Essential matrix RANSAC），回傳通過的匹配 (M,2)；不夠 min_inliers 回傳 None。

    給 BA 補充影像對用（跨幾幀、跨圈）：這裡不需要相對位姿，只要挑出符合同一組對極幾何的匹配。
    """
    if len(matches) < max(8, min_inliers):
        return None
    cv2.setRNGSeed(int(seed))
    _, mask = cv2.findEssentialMat(kpts_a[matches[:, 0]], kpts_b[matches[:, 1]], K, method=cv2.RANSAC, prob=0.999,
                                   threshold=threshold_px)
    if mask is None:
        return None
    keep = mask.ravel().astype(bool)
    return matches[keep] if keep.sum() >= min_inliers else None


def estimate_relative_pose(kpts_ref, kpts_cur, matches, pts3d_ref, valid_ref, K, cfg,
                           pts3d_cur=None, valid_cur=None, seed=0):
    """由兩張影像的匹配估計 T_ref_cur（把 cur 相機座標的點轉到 ref 相機座標）。

    兩種模式，依資料自動選：
    - 'essential'（車輛有移動時）：Essential matrix RANSAC + recoverPose 給旋轉與平移方向——只用 2D 對應，
      不受 LoGeR 深度誤差影響；平移長度再用 ref 幀 3D 點以穩健中位數估計（translation_length）。
    - 'pnp'（幾乎靜止、視差太小、Essential matrix 退化時）：PnP RANSAC，3D 來自 ref 幀點雲。
    另外用兩幀都有 3D 點的 PnP 內點估計相對尺度 = median(ref 點轉到 cur 的深度 / cur 點雲深度)。
    """
    out = dict(ok=False, mode='', matches=int(len(matches)), essential_inliers=0, pnp_inliers=0, inlier_ratio=0.0,
               reprojection_px=None, scale_ratio=None, T_ref_cur=None, essential_mask=None, pnp_mask=None,
               flow_px=0.0, length_m=None, length_mad_m=None, pnp_length_m=None)
    if len(matches) < 8:
        return out
    a = kpts_ref[matches[:, 0]]
    b = kpts_cur[matches[:, 1]]
    out['flow_px'] = float(np.median(np.linalg.norm(a - b, axis=1)))
    cv2.setRNGSeed(int(seed))
    E, mask = cv2.findEssentialMat(a, b, K, method=cv2.RANSAC, prob=0.999, threshold=cfg['essential_threshold_px'])
    emask = np.zeros(len(matches), dtype=bool)
    T_essential = None
    if E is not None and E.shape == (3, 3) and mask is not None:
        # distanceThresh 以「基線長度」為單位；預設 50 會把 0.3 m 基線下 15 m 以外的點全部剔除，所以明確放大
        res = cv2.recoverPose(E, a, b, K, distanceThresh=1e6, mask=mask.copy())
        R_e, t_e = res[1], res[2]
        emask = res[3].ravel().astype(bool)
        use = emask & valid_ref[matches[:, 0]]
        if emask.sum() >= cfg['min_essential_inliers'] and use.sum() >= cfg['min_length_points'] \
                and out['flow_px'] >= cfg['min_flow_px']:
            lam, mad, n_used = translation_length(R_e, t_e.ravel(), pts3d_ref[matches[use, 0]], b[use], K)
            if lam > 0:
                T_essential = se3.make(R_e, lam * t_e.ravel())
                out.update(length_m=lam, length_mad_m=mad)
    out['essential_inliers'] = int(emask.sum())
    out['essential_mask'] = emask

    # PnP：靜止時的主要來源，也用來估計相對尺度與提供內點統計
    has3d = valid_ref[matches[:, 0]]
    sel = np.flatnonzero(has3d)
    T_pnp = None
    R = tvec = None
    if len(sel) >= 6:
        obj = pts3d_ref[matches[sel, 0]].astype(np.float64)
        img = b[sel].astype(np.float64)
        cv2.setRNGSeed(int(seed))
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(obj, img, K, None, iterationsCount=cfg['pnp_iterations'],
                                                     reprojectionError=cfg['pnp_reprojection_px'], confidence=0.999,
                                                     flags=cv2.SOLVEPNP_EPNP)
        if ok and inliers is not None and len(inliers) >= 6:
            inl = inliers.ravel()
            rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)
            R, _ = cv2.Rodrigues(rvec)
            T_pnp = se3.make(R, tvec.ravel())
            out['pnp_length_m'] = float(np.linalg.norm(tvec))
            proj, _ = cv2.projectPoints(obj[inl], rvec, tvec, K, None)
            pmask = np.zeros(len(matches), dtype=bool)
            pmask[sel[inl]] = True
            out.update(pnp_inliers=int(len(inl)), inlier_ratio=float(len(inl)) / len(sel), pnp_mask=pmask,
                       reprojection_px=float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img[inl]) ** 2, axis=1)))))
            if pts3d_cur is not None:
                both = pmask & valid_cur[matches[:, 1]]
                if both.sum() >= 20:
                    z_pred = (pts3d_ref[matches[both, 0]] @ R.T + tvec.ravel())[:, 2]
                    z_cur = pts3d_cur[matches[both, 1], 2]
                    good = (z_pred > 0.1) & (z_cur > 0.1)
                    if good.sum() >= 20:
                        out['scale_ratio'] = float(np.median(z_pred[good] / z_cur[good]))
    pnp_ok = bool(T_pnp is not None and out['pnp_inliers'] >= cfg['min_pnp_inliers']
                  and out['inlier_ratio'] >= cfg['min_pnp_inlier_ratio'])
    if T_essential is not None:
        out.update(ok=True, mode='essential', T_ref_cur=se3.inv(T_essential))
    elif pnp_ok:
        out.update(ok=True, mode='pnp', T_ref_cur=se3.inv(T_pnp))
    elif T_pnp is not None:
        out['T_ref_cur'] = se3.inv(T_pnp)
    return out


# ---------------------------------------------------------------- 閉環檢索（GPU 投票）
def vote_candidates(query_desc, bank_desc, bank_owner, n_owners, similarity, chunk=400000):
    """每個查詢描述子在描述子庫中找最相似者，相似度過門檻就投給它所屬的 keyframe。回傳各 keyframe 票數（NumPy）。"""
    votes = cp.zeros(n_owners, dtype=cp.int64)
    if len(query_desc) == 0 or len(bank_desc) == 0:
        return cp.asnumpy(votes)
    best_sim = cp.full(len(query_desc), -2.0, dtype=cp.float32)
    best_idx = cp.zeros(len(query_desc), dtype=cp.int64)
    for start in range(0, len(bank_desc), chunk):
        sim = query_desc @ bank_desc[start:start + chunk].T
        idx = cp.argmax(sim, axis=1)
        val = sim[cp.arange(len(query_desc)), idx]
        better = val > best_sim
        best_sim = cp.where(better, val, best_sim)
        best_idx = cp.where(better, idx + start, best_idx)
    hit = best_sim >= similarity
    if int(hit.sum()):
        votes = cp.bincount(bank_owner[best_idx[hit]], minlength=n_owners)
    return cp.asnumpy(votes)


# ---------------------------------------------------------------- 除錯圖
def draw_matches(img_ref, img_cur, kpts_ref, kpts_cur, matches, inlier_mask, path, max_lines=400, text=''):
    """左右並排兩張影像：所有特徵點（黃）、RANSAC 內點連線（綠）、外點連線（紅）。"""
    scale = 0.5
    a = cv2.resize(img_ref, None, fx=scale, fy=scale)
    b = cv2.resize(img_cur, None, fx=scale, fy=scale)
    canvas = np.hstack([a, b])
    off = a.shape[1]
    for p in kpts_ref[::3]:
        cv2.circle(canvas, (int(p[0] * scale), int(p[1] * scale)), 1, (0, 255, 255), -1)
    for p in kpts_cur[::3]:
        cv2.circle(canvas, (int(p[0] * scale) + off, int(p[1] * scale)), 1, (0, 255, 255), -1)
    if len(matches):
        mask = np.zeros(len(matches), dtype=bool) if inlier_mask is None else inlier_mask
        step = max(1, len(matches) // max_lines)
        for k in range(0, len(matches), step):
            pa = kpts_ref[matches[k, 0]] * scale
            pb = kpts_cur[matches[k, 1]] * scale
            color = (0, 220, 0) if mask[k] else (0, 0, 255)
            cv2.line(canvas, (int(pa[0]), int(pa[1])), (int(pb[0]) + off, int(pb[1])), color, 1, cv2.LINE_AA)
    if text:
        cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.t) * 1000.0
