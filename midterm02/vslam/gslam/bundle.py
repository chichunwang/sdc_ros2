"""多幀特徵軌跡的重投影最佳化（bundle adjustment，BA）。

為什麼需要：里程計每一步的長度來自 LoGeR 深度，同一地點不同圈可差 2–3 成；pose graph 只能把誤差攤平。
同一個特徵點被連續多幀（以及不同圈）看到時，「它在每張影像的像素位置」本身就是不依賴 LoGeR 深度的幾何量測：
所有相機位姿與特徵點 3D 位置必須同時解釋這些像素位置。BA 就是把這個條件當成最小平方問題來解。

變數：每一幀的相機位姿 T_i（SE(3)，camera-to-world）、每條軌跡的 3D 點 X_j（世界座標）。
殘差：
- 重投影：π(K, T_i⁻¹·X_j) − 觀測像素 uv_ij（單位 px，Huber 穩健核）
- 里程計：相鄰幀的相對位姿（很鬆，只用來固定整體尺度、撐住靜止或特徵太少的幀）
- 閉環：keyframe 之間的相對位姿（沿用 pose graph 的邊）
- 規範：第 0 幀固定
流程：特徵匹配 → 連成軌跡（連通元件）→ 以目前位姿三角化初值並篩選 → GTSAM LM → 丟掉殘差大的觀測 → 再解一次。
三角化與篩選是 NumPy 批次運算；非線性最佳化用 GTSAM（CPU，稀疏 Cholesky）。
"""
import time

import numpy as np

from . import se3


def build_tracks(counts, links, min_length=3):
    """把兩兩影像的特徵匹配連成軌跡。

    counts：每幀的特徵數；links：[(frame_a, frame_b, pairs(M,2))]，pairs 是兩幀各自的特徵索引。
    每個特徵是圖上一個節點、每個匹配是一條邊；一個連通元件就是「同一個 3D 點在各幀的觀測」。
    同一元件若在同一幀出現兩個不同特徵，代表中間有錯誤匹配，整條丟掉。
    回傳 (obs_track, obs_frame, obs_feat, n_tracks)，依軌跡編號排序。
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total = int(offsets[-1])
    if not links:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z, 0
    a = np.concatenate([offsets[fa] + pairs[:, 0] for fa, fb, pairs in links])
    b = np.concatenate([offsets[fb] + pairs[:, 1] for fa, fb, pairs in links])
    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(total, total))
    _, label = connected_components(graph, directed=False)
    nodes = np.unique(np.concatenate([a, b]))
    lab = label[nodes]
    frame = np.searchsorted(offsets, nodes, side='right') - 1
    size = np.bincount(lab, minlength=label.max() + 1)
    # 同一軌跡、同一幀只能有一個特徵
    pair_id = lab.astype(np.int64) * len(counts) + frame
    distinct = np.bincount(lab[np.unique(pair_id, return_index=True)[1]], minlength=len(size))
    good = (size >= min_length) & (distinct == size)
    keep = good[lab]
    nodes, lab, frame = nodes[keep], lab[keep], frame[keep]
    order = np.argsort(lab, kind='stable')
    nodes, lab, frame = nodes[order], lab[order], frame[order]
    _, track = np.unique(lab, return_inverse=True)
    return track.astype(np.int64), frame.astype(np.int64), (nodes - offsets[frame]).astype(np.int64), \
        int(track.max() + 1) if len(track) else 0


def _world_to_camera(poses, obs_frame):
    Twc = np.stack([se3.inv(T) for T in poses])
    return Twc[obs_frame, :3, :3], Twc[obs_frame, :3, 3]


def project(K, R, t, X):
    """R、t：每個觀測的 world→camera；X：對應的世界座標點。回傳 (像素, 深度)。"""
    Xc = np.einsum('nij,nj->ni', R, X) + t
    z = Xc[:, 2]
    zs = np.where(np.abs(z) > 1e-6, z, 1e-6)
    return np.c_[K[0, 0] * Xc[:, 0] / zs + K[0, 2], K[1, 1] * Xc[:, 1] / zs + K[1, 2]], z


def triangulate(K, poses, obs_track, obs_frame, uv, n_tracks, iterations=3, robust_px=3.0):
    """所有軌跡一起做線性三角化（批次解 3×3 正規方程），並用重加權降低離群觀測的影響。

    每個觀測給兩條線性方程：(R₀ − x·R₂)·X = x·t₂ − t₀、(R₁ − y·R₂)·X = y·t₂ − t₁（x、y 為正規化像素座標）。
    方程的殘差 ≈ 深度 × 正規化像素誤差，所以第二輪起除以深度²，並乘上 Cauchy 權重。
    回傳 (X (n_tracks,3), 每個觀測的重投影誤差 px, 深度, 每條軌跡的視差角 deg)。
    """
    R, t = _world_to_camera(poses, obs_frame)
    x = (uv[:, 0] - K[0, 2]) / K[0, 0]
    y = (uv[:, 1] - K[1, 2]) / K[1, 1]
    a = np.concatenate([R[:, 0] - x[:, None] * R[:, 2], R[:, 1] - y[:, None] * R[:, 2]])
    b = np.concatenate([x * t[:, 2] - t[:, 0], y * t[:, 2] - t[:, 1]])
    idx = np.concatenate([obs_track, obs_track])
    w = np.ones(len(a))
    for it in range(iterations + 1):
        AtA = np.zeros((n_tracks, 3, 3))
        Atb = np.zeros((n_tracks, 3))
        for i in range(3):
            Atb[:, i] = np.bincount(idx, w * a[:, i] * b, n_tracks)
            for j in range(i, 3):
                AtA[:, i, j] = AtA[:, j, i] = np.bincount(idx, w * a[:, i] * a[:, j], n_tracks)
        AtA += 1e-9 * np.trace(AtA, axis1=1, axis2=2)[:, None, None] * np.eye(3) + 1e-15 * np.eye(3)
        X = np.linalg.solve(AtA, Atb[..., None])[..., 0]
        proj, z = project(K, R, t, X[obs_track])
        err = np.linalg.norm(proj - uv, axis=1)
        if it == iterations:
            break
        wo = 1.0 / (1.0 + (err / robust_px) ** 2) / np.maximum(z, 0.1) ** 2
        w = np.concatenate([wo, wo])
    # 視差：各觀測「相機→點」的方向與軌跡平均方向的最大夾角 ×2（近似最大兩兩夾角）
    ray = np.einsum('nji,nj->ni', R, np.c_[x, y, np.ones(len(x))])
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    mean = np.stack([np.bincount(obs_track, ray[:, i], n_tracks) for i in range(3)], axis=1)
    mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
    cos = np.sum(ray * mean[obs_track], axis=1)
    min_cos = np.ones(n_tracks)
    np.minimum.at(min_cos, obs_track, cos)
    parallax = 2.0 * np.rad2deg(np.arccos(np.clip(min_cos, -1.0, 1.0)))
    return X, err, z, parallax


def _per_track_median(values, obs_track, n_tracks):
    order = np.lexsort((values, obs_track))
    count = np.bincount(obs_track, minlength=n_tracks)
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    return values[order][np.minimum(start + count // 2, len(values) - 1)]


def select(obs_track, obs_frame, uv, X, err, z, parallax, cfg):
    """篩選可用的軌跡：視差夠、所有觀測都在相機前方、重投影誤差中位數夠小。回傳重新編號後的觀測與點。"""
    n = len(X)
    med = _per_track_median(err, obs_track, n)
    behind = np.bincount(obs_track, z < 0.1, n) > 0
    keep = (parallax >= cfg['min_parallax_deg']) & (~behind) & (med <= cfg['triangulation_gate_px']) \
        & np.all(np.isfinite(X), axis=1)
    return _compact(keep, obs_track, obs_frame, uv, X, np.ones(len(obs_track), dtype=bool), cfg['min_track_length'])


def _compact(keep_track, obs_track, obs_frame, uv, X, keep_obs, min_length):
    keep_obs = keep_obs & keep_track[obs_track]
    count = np.bincount(obs_track[keep_obs], minlength=len(X))
    keep_obs &= (count >= min_length)[obs_track]
    new_id = np.full(len(X), -1, dtype=np.int64)
    alive = np.flatnonzero(np.bincount(obs_track[keep_obs], minlength=len(X)) > 0)
    new_id[alive] = np.arange(len(alive))
    return new_id[obs_track[keep_obs]], obs_frame[keep_obs], uv[keep_obs], X[alive]


def solve(K, poses, obs_track, obs_frame, uv, X, between, cfg, log=print):
    """GTSAM LM。between：[(i, j, T_rel(4×4), noise model)]。回傳 (poses, X, info)。"""
    import gtsam
    n = len(poses)
    cal = gtsam.Cal3_S2(float(K[0, 0]), float(K[1, 1]), 0.0, float(K[0, 2]), float(K[1, 2]))
    pixel = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(cfg['huber_k']),
                                           gtsam.noiseModel.Isotropic.Sigma(2, cfg['pixel_sigma']))
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    for i, T in enumerate(poses):
        values.insert(i, gtsam.Pose3(se3.normalize(T)))
    for j in range(len(X)):
        values.insert(n + j, gtsam.Point3(X[j]))
    graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(se3.normalize(poses[0])),
                                     gtsam.noiseModel.Isotropic.Sigma(6, cfg['prior_sigma'])))
    for i, j, T_rel, noise in between:
        graph.add(gtsam.BetweenFactorPose3(int(i), int(j), gtsam.Pose3(se3.normalize(T_rel)), noise))
    factor = gtsam.GenericProjectionFactorCal3_S2
    for k in range(len(obs_track)):
        graph.add(factor(uv[k], pixel, int(obs_frame[k]), n + int(obs_track[k]), cal))
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(cfg['max_iterations'])
    params.setRelativeErrorTol(cfg['relative_error_tol'])
    try:   # 先消去 3D 點、最後才消去位姿（Schur 補的順序），填入量最少
        last = gtsam.KeyVector(list(range(n)))
        params.setOrdering(gtsam.Ordering.ColamdConstrainedLastNonlinearFactorGraph(graph, last))
    except Exception as exc:
        log('BA：無法指定消去順序（%s），使用預設 COLAMD' % exc)
    t0 = time.perf_counter()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
    e0 = optimizer.error()
    result = optimizer.optimize()
    info = dict(initial_error=float(e0), final_error=float(optimizer.error()), iterations=int(optimizer.iterations()),
                seconds=time.perf_counter() - t0)
    new_poses = [se3.normalize(result.atPose3(i).matrix()) for i in range(n)]
    new_X = np.stack([result.atPoint3(n + j) for j in range(len(X))]) if len(X) else X
    return new_poses, new_X, info


def reprojection(K, poses, obs_track, obs_frame, uv, X):
    R, t = _world_to_camera(poses, obs_frame)
    proj, z = project(K, R, t, X[obs_track])
    return np.linalg.norm(proj - uv, axis=1), z


def run(K, poses, kpts, links, between, cfg, log=print):
    """完整流程。poses：每幀初始位姿（camera-to-world）；kpts：每幀特徵像素座標；links：特徵匹配。

    回傳 (最佳化後每幀位姿, info)。軌跡太少時原樣回傳初始位姿。
    """
    t0 = time.perf_counter()
    counts = [len(k) for k in kpts]
    obs_track, obs_frame, obs_feat, n_tracks = build_tracks(counts, links, cfg['min_track_length'])
    info = dict(tracks_raw=n_tracks, observations_raw=int(len(obs_track)))
    if n_tracks < cfg['min_tracks']:
        info['status'] = 'too_few_tracks'
        return [T.copy() for T in poses], info
    uv = np.concatenate([np.asarray(k, dtype=np.float64).reshape(-1, 2) for k in kpts])[
        np.concatenate([[0], np.cumsum(counts)])[obs_frame] + obs_feat]
    X, err, z, parallax = triangulate(K, poses, obs_track, obs_frame, uv, n_tracks)
    obs_track, obs_frame, uv, X = select(obs_track, obs_frame, uv, X, err, z, parallax, cfg)
    if cfg['max_observations'] and len(obs_track) > cfg['max_observations']:
        # 觀測太多時等間隔抽掉整條軌跡（固定規則，結果可重現）
        step = int(np.ceil(len(obs_track) / cfg['max_observations']))
        keep = np.zeros(len(X), dtype=bool)
        keep[::step] = True
        obs_track, obs_frame, uv, X = _compact(keep, obs_track, obs_frame, uv, X,
                                               np.ones(len(obs_track), dtype=bool), cfg['min_track_length'])
    span = np.zeros(len(X), dtype=np.int64)
    lo = np.full(len(X), 10 ** 9, dtype=np.int64)
    np.maximum.at(span, obs_track, obs_frame)
    np.minimum.at(lo, obs_track, obs_frame)
    e, _ = reprojection(K, poses, obs_track, obs_frame, uv, X)
    info.update(tracks=int(len(X)), observations=int(len(obs_track)),
                tracks_across_laps=int(np.sum(span - lo > cfg['lap_gap_frames'])),
                frames_with_observations=int(len(np.unique(obs_frame))),
                reprojection_px_initial=dict(median=float(np.median(e)), rms=float(np.sqrt(np.mean(e ** 2)))),
                prepare_seconds=time.perf_counter() - t0, rounds=[])
    log('BA：%d 條軌跡、%d 個觀測（跨圈 %d 條），初始重投影誤差中位數 %.2f px' % (
        len(X), len(obs_track), info['tracks_across_laps'], np.median(e)))
    cur = [T.copy() for T in poses]
    for r in range(cfg['rounds']):
        cur, X, round_info = solve(K, cur, obs_track, obs_frame, uv, X, between, cfg, log)
        e, z = reprojection(K, cur, obs_track, obs_frame, uv, X)
        round_info.update(tracks=int(len(X)), observations=int(len(obs_track)),
                          reprojection_px=dict(median=float(np.median(e)), rms=float(np.sqrt(np.mean(e ** 2))),
                                               p95=float(np.percentile(e, 95))))
        info['rounds'].append(round_info)
        log('BA 第 %d 輪：%d 次迭代 %.1f s，重投影誤差中位數 %.2f px、95%% %.2f px' % (
            r + 1, round_info['iterations'], round_info['seconds'], np.median(e), np.percentile(e, 95)))
        if r + 1 < cfg['rounds']:   # 丟掉殘差大或跑到相機後方的觀測，再解一次
            good = (e <= cfg['prune_px']) & (z > 0.1)
            obs_track, obs_frame, uv, X = _compact(np.ones(len(X), dtype=bool), obs_track, obs_frame, uv, X, good,
                                                   cfg['min_track_length'])
    info['status'] = 'ok'
    info['seconds'] = time.perf_counter() - t0
    return cur, info


# ---------------------------------------------------------------- 開發用：把 BA 的輸入存檔，之後可以不重跑前端直接試 BA
def save_input(path, K, poses, kpts, links, between_raw):
    counts = np.array([len(k) for k in kpts], dtype=np.int64)
    np.savez_compressed(
        path, K=K, poses=np.stack(poses), counts=counts,
        kpts=np.concatenate([np.asarray(k, dtype=np.float32).reshape(-1, 2) for k in kpts]),
        link_frames=np.array([[a, b, len(p)] for a, b, p in links], dtype=np.int64).reshape(-1, 3),
        link_pairs=np.concatenate([p for _, _, p in links]) if links else np.zeros((0, 2), dtype=np.int64),
        between_ij=np.array([[i, j, kind] for i, j, _, kind in between_raw], dtype=np.int64).reshape(-1, 3),
        between_T=np.stack([T for _, _, T, _ in between_raw]) if between_raw else np.zeros((0, 4, 4)))


def load_input(path):
    d = np.load(path)
    off = np.concatenate([[0], np.cumsum(d['counts'])])
    kpts = [d['kpts'][off[i]:off[i + 1]].astype(np.float64) for i in range(len(d['counts']))]
    links, s = [], 0
    for a, b, m in d['link_frames']:
        links.append((int(a), int(b), d['link_pairs'][s:s + m]))
        s += m
    between_raw = [(int(i), int(j), T, int(kind)) for (i, j, kind), T in zip(d['between_ij'], d['between_T'])]
    return d['K'], list(d['poses']), kpts, links, between_raw
