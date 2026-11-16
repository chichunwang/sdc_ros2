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


def build_tracks(counts, links, min_length=3, resolve=True):
    """把兩兩影像的特徵匹配連成軌跡。

    counts：每幀的特徵數；links：[(frame_a, frame_b, pairs(M,2))]，pairs 是兩幀各自的特徵索引。
    每個特徵是圖上一個節點、每個匹配是一條邊；一個連通元件就是「同一個 3D 點在各幀的觀測」。
    同一元件若在同一幀出現兩個不同特徵（錯誤匹配，或 SIFT 在同一位置給了兩個方向不同的特徵）：
    - resolve=True：只對這些元件依 links 的順序重新逐條合併，會造成「同一幀兩個特徵」的那條匹配就跳過。
      所以 links 要把比較可靠的匹配排前面（run() 依兩幀的間隔由小到大排）。
    - resolve=False：整條丟掉。
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
    conflict = distinct != size
    if resolve and np.any(conflict & (size >= min_length)):
        lab = lab.copy()
        on = conflict[label[a]]
        root = _merge_without_conflict(a[on], b[on], offsets)
        mine = conflict[lab]
        # 重新編號：衝突元件裡的節點改用「合併後的根節點」當軌跡編號（接在原編號之後，不會撞號）
        lab[mine] = len(size) + np.array([root[int(x)] for x in nodes[mine]], dtype=np.int64)
        size = np.bincount(lab)
        good = size >= min_length
    else:
        good = (size >= min_length) & ~conflict
    keep = good[lab]
    nodes, lab, frame = nodes[keep], lab[keep], frame[keep]
    order = np.argsort(lab, kind='stable')
    nodes, lab, frame = nodes[order], lab[order], frame[order]
    _, track = np.unique(lab, return_inverse=True)
    return track.astype(np.int64), frame.astype(np.int64), (nodes - offsets[frame]).astype(np.int64), \
        int(track.max() + 1) if len(track) else 0


def _merge_without_conflict(a, b, offsets):
    """依序處理每條匹配 (a[k], b[k])：兩邊所屬的軌跡沒有共同的幀才合併。回傳 {節點: 根節點}。"""
    nodes = np.unique(np.concatenate([a, b]))
    frame_of = dict(zip(nodes.tolist(), (np.searchsorted(offsets, nodes, side='right') - 1).tolist()))
    parent = {x: x for x in frame_of}
    frames = {x: {f} for x, f in frame_of.items()}

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    for u, v in zip(a.tolist(), b.tolist()):
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        fu, fv = frames[ru], frames[rv]
        if not fu.isdisjoint(fv):
            continue
        if len(fu) < len(fv):
            ru, rv, fu, fv = rv, ru, fv, fu
        parent[rv] = ru
        fu |= fv
        del frames[rv]
    return {x: find(x) for x in parent}


def revisit_pairs(poses, min_gap, radius_m, min_cos, cluster_gap=30):
    """依目前位姿找「重訪」的影像對：幀 j 比幀 i 早至少 min_gap 幀、相機位置相距 radius_m 以內、
    光軸方向夾角餘弦 ≥ min_cos。符合的 j 若連成好幾段（代表經過同一地點好幾次），每段只取位置最近的一幀。
    回傳 [(j, i)]，j < i。"""
    P = np.stack(poses)
    xyz, fwd = P[:, :3, 3], P[:, :3, 2]
    idx = np.arange(len(P))
    pairs = []
    for i in range(len(P)):
        d = np.linalg.norm(xyz - xyz[i], axis=1)
        js = np.flatnonzero((idx <= i - min_gap) & (d < radius_m) & (fwd @ fwd[i] >= min_cos))
        if len(js) == 0:
            continue
        for grp in np.split(js, np.flatnonzero(np.diff(js) > cluster_gap) + 1):
            pairs.append((int(grp[np.argmin(d[grp])]), i))
    return pairs


def _world_to_camera(poses, obs_frame):
    Twc = np.stack([se3.inv(T) for T in poses])
    return Twc[obs_frame, :3, :3], Twc[obs_frame, :3, 3]


def project(K, R, t, X, dist=None):
    """R、t：每個觀測的 world→camera；X：對應的世界座標點。回傳 (像素, 深度)。
    dist：(k1, k2) 徑向畸變（與 GTSAM Cal3DS2、OpenCV 同一模型：x_d = x·(1 + k1·r² + k2·r⁴)）；None = 針孔。"""
    Xc = np.einsum('nij,nj->ni', R, X) + t
    z = Xc[:, 2]
    zs = np.where(np.abs(z) > 1e-6, z, 1e-6)
    x, y = Xc[:, 0] / zs, Xc[:, 1] / zs
    if dist is not None:
        r2 = x * x + y * y
        g = 1.0 + dist[0] * r2 + dist[1] * r2 * r2
        x, y = x * g, y * g
    return np.c_[K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]], z


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


def observation_sigma(sizes, cfg):
    """每個觀測的像素標準差（相對於 pixel_sigma 的倍數）。

    SIFT 特徵的 size 是它被偵測到的尺度（px）。尺度大的特徵是在模糊、降解析度的影像上找到的，定位比較不準：
    實測重投影誤差中位數由 size<2.5 的 0.23 px 增加到 size>30 的 0.75 px，大致是 size 的 0.4 次方。
    所以 sigma = (size / size_ref) ** power，夾在 [lo, hi]；size_weight 關閉或沒有 size 時全部為 1。
    """
    w = cfg.get('size_weight')
    if not w or not w.get('enabled') or sizes is None:
        return None
    return np.clip((np.maximum(sizes, 1e-3) / w['size_ref']) ** w['power'], w['lo'], w['hi'])


def solve(K, poses, obs_track, obs_frame, uv, X, between, cfg, log=print, rel_sigma=None, extra=None, calib=None):
    """GTSAM LM。between：[(i, j, T_rel(4×4), noise model)]；rel_sigma：各觀測的像素標準差倍數（None = 全部 1）；
    extra：fn(graph, n_poses)，再往圖裡加其他因子（例如 LoGeR 距離因子；3D 點的 key = n_poses + 軌跡編號）。
    calib：dict(value=[fx, fy, s, cx, cy, k1, k2, p1, p2], sigmas=[...])——相機內參與畸變也當變數一起解（自校正，
    GTSAM Cal3DS2，key = n_poses + 軌跡數）；None = 內參固定為 K 的針孔模型。
    回傳 (poses, X, info)；自校正時 info['calibration'] 是解出的 9 個參數。"""
    import gtsam
    n = len(poses)
    cal = gtsam.Cal3_S2(float(K[0, 0]), float(K[1, 1]), 0.0, float(K[0, 2]), float(K[1, 2]))
    ckey = n + len(X)

    def make_pixel(scale):
        return gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(cfg['huber_k']),
                                              gtsam.noiseModel.Isotropic.Sigma(2, cfg['pixel_sigma'] * scale))
    if rel_sigma is None:
        level = np.zeros(len(obs_track), dtype=np.int64)
        models = [make_pixel(1.0)]
    else:   # 量化成 0.05 一級，同一級共用一個 noise model
        level = np.round(rel_sigma / 0.05).astype(np.int64)
        models = {int(v): make_pixel(v * 0.05) for v in np.unique(level)}
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
    if calib is None:
        factor = gtsam.GenericProjectionFactorCal3_S2
        for k in range(len(obs_track)):
            graph.add(factor(uv[k], models[int(level[k])], int(obs_frame[k]), n + int(obs_track[k]), cal))
    else:
        c0 = gtsam.Cal3DS2(*[float(v) for v in calib['value']])
        values.insert(ckey, c0)
        graph.add(gtsam.PriorFactorCal3DS2(ckey, c0, gtsam.noiseModel.Diagonal.Sigmas(np.asarray(calib['sigmas'], float))))
        factor = gtsam.GeneralSFMFactor2Cal3DS2
        for k in range(len(obs_track)):
            graph.add(factor(gtsam.Point2(*uv[k]), models[int(level[k])], int(obs_frame[k]), n + int(obs_track[k]), ckey))
    if extra is not None:
        extra(graph, n)
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(cfg['max_iterations'])
    params.setRelativeErrorTol(cfg['relative_error_tol'])
    try:   # 先消去 3D 點、最後才消去位姿（Schur 補的順序），填入量最少
        last = gtsam.KeyVector(list(range(n)) + ([ckey] if calib is not None else []))
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
    if calib is not None:
        info['calibration'] = [float(v) for v in result.atCal3DS2(ckey).vector()]
    return new_poses, new_X, info


def calib_to_K(c):
    """Cal3DS2 的 9 個參數 → (K 3×3, (k1, k2))。"""
    return np.array([[c[0], c[2], c[3]], [0, c[1], c[4]], [0, 0, 1.0]]), (c[5], c[6])


def reprojection(K, poses, obs_track, obs_frame, uv, X, dist=None):
    R, t = _world_to_camera(poses, obs_frame)
    proj, z = project(K, R, t, X[obs_track], dist)
    return np.linalg.norm(proj - uv, axis=1), z


def run(K, poses, kpts, links, between, cfg, log=print, sizes=None, descs=None, depths=None, imu_windows=None):
    """完整流程。poses：每幀初始位姿（camera-to-world）；kpts：每幀特徵像素座標；links：特徵匹配；
    sizes：每幀特徵的 SIFT 尺度（可省略；有給且 cfg['size_weight'] 開啟時，依尺度給每個觀測不同的權重）；
    descs：每幀描述子（可省略；有給且 cfg['guided'] 開啟時，BA 後用投影引導補觀測再解一次）；
    depths：每幀特徵的 LoGeR 深度（沒有則 NaN；有給且 cfg['depth'] 開啟時，加上 LoGeR 距離因子，見 depth_factors）；
    imu_windows：gslam.imu.Windows（只有 Track01 有 IMU；有給且 cfg['imu'] 開啟時，第 2 輪起加上 IMU 平均速度變化因子）。

    回傳 (最佳化後每幀位姿, info)。軌跡太少時原樣回傳初始位姿。
    """
    t0 = time.perf_counter()
    counts = [len(k) for k in kpts]
    links = sorted(links, key=lambda link: link[1] - link[0])   # 穩定排序：間隔小（較可靠）的匹配先合併
    obs_track, obs_frame, obs_feat, n_tracks = build_tracks(counts, links, cfg['min_track_length'],
                                                            cfg.get('resolve_conflicts', True))
    info = dict(tracks_raw=n_tracks, observations_raw=int(len(obs_track)), image_pairs=len(links))
    if n_tracks < cfg['min_tracks']:
        info['status'] = 'too_few_tracks'
        return [T.copy() for T in poses], info
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    node = offsets[obs_frame] + obs_feat
    all_kpts = np.concatenate([np.asarray(k, dtype=np.float64).reshape(-1, 2) for k in kpts])
    uv = all_kpts[node]
    all_rel = observation_sigma(None if sizes is None else np.concatenate(
        [np.asarray(s, dtype=np.float64).ravel() for s in sizes]), cfg)
    weighted = all_rel is not None
    # 第三欄：每個觀測的 sigma 倍數；第四欄：特徵的全域索引。兩欄跟著篩選一起走；幾何運算只用前兩欄
    uv = np.c_[uv, all_rel[node] if weighted else np.ones(len(uv)), node]
    X, err, z, parallax = triangulate(K, poses, obs_track, obs_frame, uv[:, :2], n_tracks)
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
    e, _ = reprojection(K, poses, obs_track, obs_frame, uv[:, :2], X)
    info.update(size_weighted=bool(weighted), tracks=int(len(X)), observations=int(len(obs_track)),
                tracks_across_laps=int(np.sum(span - lo > cfg['lap_gap_frames'])),
                frames_with_observations=int(len(np.unique(obs_frame))),
                reprojection_px_initial=dict(median=float(np.median(e)), rms=float(np.sqrt(np.mean(e ** 2)))),
                prepare_seconds=time.perf_counter() - t0, rounds=[])
    log('BA：%d 條軌跡、%d 個觀測（跨圈 %d 條），初始重投影誤差中位數 %.2f px' % (
        len(X), len(obs_track), info['tracks_across_laps'], np.median(e)))

    dcfg = cfg.get('depth') or {}
    use_depth = bool(dcfg.get('enabled')) and depths is not None
    all_depth = np.concatenate([np.asarray(d, dtype=np.float64).ravel() for d in depths]) if use_depth else None
    state = dict(s=None)
    icfg = cfg.get('imu') or {}
    use_imu = bool(icfg.get('enabled')) and imu_windows is not None
    if use_imu:
        from . import imu as imu_mod
        hold = icfg.get('holdout_block_frames') or 0
        f_mask = imu_mod.block_mask(hold, keep_even=True) if hold else None   # 驗證用：只用偶數塊，奇數塊留作檢查
        imu_k1, imu_k2 = imu_mod.factor_pairs(len(poses), imu_windows.h, f_mask)
        info['imu'] = dict(pairs=int(len(imu_k1)), holdout_block_frames=int(hold), solves=[])

    ccfg = cfg.get('calibration') or {}
    use_cal = bool(ccfg.get('enabled'))
    state['cal'] = [K[0, 0], K[1, 1], 0.0, K[0, 2], K[1, 2], 0.0, 0.0, 0.0, 0.0]

    def rounds(cur, obs_track, obs_frame, uv, X, label):
        n_solves = cfg['rounds']
        if use_depth:
            n_solves = dcfg['solves']
        if use_imu:
            n_solves = max(n_solves, icfg['solves'])
        if use_cal:
            n_solves = max(n_solves, ccfg['solves'])
        for r in range(n_solves):
            extras, n_depth = [], 0
            # 自校正：第 1 輪用給定的針孔內參；之後把內參與徑向畸變當變數（起點 = 上一輪的解）
            calib = dict(value=state['cal'], sigmas=ccfg['sigmas']) if use_cal and r > 0 else None
            if use_depth:   # 先用目前的 BA 解出每幀的 LoGeR 尺度 s_i，再固定 s_i 解 BA（同一目標函數的交替求解）
                d_obs = all_depth[uv[:, 3].astype(np.int64)]
                state['s'] = frame_log_scale(K, cur, X, obs_track, obs_frame, d_obs, dcfg, state['s'])
                fn, n_depth = depth_factors(K, uv, obs_track, obs_frame, d_obs, state['s'], dcfg)
                extras.append(fn)
            if use_imu and r > 0:
                # 第 1 輪是一般 BA；之後依目前軌跡重新擬合 λ、b、g、槓桿臂，再把 IMU 平均速度變化當因子（交替求解）
                p_cur, R_cur = imu_mod.split_poses(cur)
                fit = imu_windows.fit(p_cur, R_cur, imu_k1, imu_k2)
                if 'sigma_m' not in state:   # 雜訊取第一次擬合的垂直殘差（每軸）：路面近似平面，垂直殘差≈IMU 本身的雜訊上限
                    state['sigma_m'] = icfg['sigma_m'] or max(fit['rms_v'], icfg['min_sigma_m'])
                fn, iinfo = imu_mod.velocity_factors(imu_windows, cur, imu_k1, imu_k2, state['sigma_m'],
                                                     icfg['huber_k'], fit)
                extras.append(fn)
                info['imu']['solves'].append(iinfo)
                log('BA：IMU 因子 %d 個，λ=%.3f、|g|=%.2f、槓桿臂 %s m、擬合水平殘差 %.3f m/s、sigma %.3f m/s' % (
                    iinfo['factors'], iinfo['lam'], iinfo['gravity_norm'], np.round(iinfo['lever'], 2),
                    iinfo['fit_rms_h'], iinfo['sigma_m']))
            extra = None
            if extras:
                def extra(graph, n_poses, extras=tuple(extras)):
                    for fn in extras:
                        fn(graph, n_poses)
            cur, X, round_info = solve(K, cur, obs_track, obs_frame, uv[:, :2], X, between, cfg, log,
                                       uv[:, 2] if weighted else None, extra=extra, calib=calib)
            if 'calibration' in round_info:
                state['cal'] = round_info['calibration']
                c = state['cal']
                log('BA 自校正：fx %.2f fy %.2f cx %.2f cy %.2f k1 %+.5f k2 %+.5f' % (c[0], c[1], c[3], c[4], c[5], c[6]))
            K_r, dist_r = calib_to_K(state['cal'])
            e, z = reprojection(K_r, cur, obs_track, obs_frame, uv[:, :2], X, dist_r)
            if use_depth:
                round_info.update(depth_factors=n_depth, log_scale_std=float(np.std(state['s'])),
                                  log_scale_range=[float(state['s'].min()), float(state['s'].max())])
            round_info.update(stage=label, tracks=int(len(X)), observations=int(len(obs_track)),
                              reprojection_px=dict(median=float(np.median(e)), rms=float(np.sqrt(np.mean(e ** 2))),
                                                   p95=float(np.percentile(e, 95))))
            info['rounds'].append(round_info)
            log('BA %s第 %d 輪：%d 次迭代 %.1f s，重投影誤差中位數 %.2f px、95%% %.2f px' % (
                label, r + 1, round_info['iterations'], round_info['seconds'], np.median(e), np.percentile(e, 95)))
            if r + 1 < n_solves:   # 丟掉殘差大或跑到相機後方的觀測，再解一次
                good = (e <= cfg['prune_px'] * uv[:, 2]) & (z > 0.1)   # 門檻跟著該觀測的 sigma 倍數放大縮小
                obs_track, obs_frame, uv, X = _compact(np.ones(len(X), dtype=bool), obs_track, obs_frame, uv, X,
                                                       good, cfg['min_track_length'])
        return cur, obs_track, obs_frame, uv, X

    cur, obs_track, obs_frame, uv, X = rounds([T.copy() for T in poses], obs_track, obs_frame, uv, X, '')
    gcfg = cfg.get('guided') or {}
    if gcfg.get('enabled') and descs is not None:
        # 用 BA 解出的位姿與 3D 點回頭補觀測（含其他圈經過同地點的幀），再解一次
        tg = time.perf_counter()
        new_t, new_f, new_n, gstat = guided_observations(K, cur, X, obs_track, obs_frame, uv[:, 3].astype(np.int64),
                                                         all_kpts, offsets, descs, gcfg)
        if len(new_t):
            add = np.c_[all_kpts[new_n], all_rel[new_n] if weighted else np.ones(len(new_n)), new_n]
            obs_track, obs_frame, uv = np.r_[obs_track, new_t], np.r_[obs_frame, new_f], np.r_[uv, add]
        lo2 = np.full(len(X), 10 ** 9, dtype=np.int64)
        hi2 = np.zeros(len(X), dtype=np.int64)
        np.minimum.at(lo2, obs_track, obs_frame)
        np.maximum.at(hi2, obs_track, obs_frame)
        gstat.update(seconds=time.perf_counter() - tg, observations_after=int(len(obs_track)),
                     tracks_across_laps_after=int(np.sum(hi2 - lo2 > cfg['lap_gap_frames'])))
        info['guided'] = gstat
        log('BA：投影引導補上 %d 個觀測（跨圈 %d 個），跨圈軌跡 %d → %d 條（%.1f s）' % (
            len(new_t), gstat['cross_lap'], info['tracks_across_laps'], gstat['tracks_across_laps_after'],
            gstat['seconds']))
        cur, obs_track, obs_frame, uv, X = rounds(cur, obs_track, obs_frame, uv, X, '（補觀測後）')
    if use_cal:
        info['calibration'] = dict(zip(['fx', 'fy', 's', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2'],
                                       [float(v) for v in state['cal']]))
    info['status'] = 'ok'
    info['seconds'] = time.perf_counter() - t0
    return cur, info


def frame_log_scale(K, poses, X, obs_track, obs_frame, depth, dcfg, s_init=None):
    """LoGeR 每幀的尺度偏差 s_i（log）：BA 深度 ≈ exp(s_i) × LoGeR 深度。

    min Σ ρ(log z_ij − log d_ij − s_i) / sigma_d² + Σ (s_i − s_{i−1})² / sigma_s²
    ρ 為 Huber（IRLS 五次）；第二項是平滑先驗（實測相鄰幀 s 的變化約 0.018）。三對角稀疏線性系統。
    depth：每個觀測的 LoGeR 深度（沒有則 NaN）。
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    n = len(poses)
    R, t = _world_to_camera(poses, obs_frame)
    z = (np.einsum('nij,nj->ni', R, X[obs_track]) + t)[:, 2]
    m = np.isfinite(depth) & (depth > 0) & (z > 0.3)
    lr = np.log(z[m]) - np.log(depth[m])
    fr = obs_frame[m]
    s = np.zeros(n) if s_init is None else np.asarray(s_init, dtype=np.float64).copy()
    sd = dcfg['sigma_log_depth']
    lam = 1.0 / dcfg['sigma_log_scale_step'] ** 2
    main = np.full(n, 2 * lam)
    main[0] = main[-1] = lam
    L = diags([np.full(n - 1, -lam), main, np.full(n - 1, -lam)], [-1, 0, 1], format='csc')
    for _ in range(5):
        r = lr - s[fr]
        w = np.minimum(1.0, 1.345 * sd / np.maximum(np.abs(r), 1e-9)) / sd ** 2
        s = spsolve(L + diags(np.bincount(fr, w, n) + 1e-6, 0, format='csc'), np.bincount(fr, w * lr, n))
    return s


def depth_factors(K, uv, obs_track, obs_frame, depth, s, dcfg):
    """回傳 fn(graph, n_poses)：每個有 LoGeR 深度的觀測加一個距離因子（相機中心到 3D 點）。

    量測值 = exp(s_i) × LoGeR 深度沿像素視線換算的距離；標準差 = sigma_log_depth × 距離（相對誤差），Huber。
    """
    import gtsam
    m = np.flatnonzero(np.isfinite(depth) & (depth > 0))
    ray = np.linalg.norm(np.c_[(uv[m, 0] - K[0, 2]) / K[0, 0], (uv[m, 1] - K[1, 2]) / K[1, 1], np.ones(len(m))], axis=1)
    meas = depth[m] * ray * np.exp(s[obs_frame[m]])
    level = np.round(np.log(meas * dcfg['sigma_log_depth']) / 0.05).astype(np.int64)   # 標準差分級共用 noise model
    huber = gtsam.noiseModel.mEstimator.Huber.Create(dcfg['huber_k'])
    models = {int(v): gtsam.noiseModel.Robust.Create(huber, gtsam.noiseModel.Isotropic.Sigma(1, float(np.exp(v * 0.05))))
              for v in np.unique(level)}
    frames, tracks = obs_frame[m], obs_track[m]

    def add(graph, n_poses):
        for fi, tj, r, lv in zip(frames.tolist(), tracks.tolist(), meas.tolist(), level.tolist()):
            graph.add(gtsam.RangeFactor3D(fi, n_poses + tj, r, models[lv]))
    return add, int(len(m))


def guided_observations(K, poses, X, obs_track, obs_frame, obs_node, all_kpts, offsets, descs, gcfg):
    """投影引導補觀測：把附近幀看到的 3D 點投影到每一幀，在投影位置附近找描述子相近的特徵。

    為什麼不用「再多配幾對影像＋Essential RANSAC」：對極幾何只檢查點到對極線的距離，沿線方向的錯配擋不掉
    （實測隔 2–3 幀的這類配對，誤差是正常觀測的 2–3 倍，Public 由 0.085 變差為 0.122）。
    這裡驗證的是完整的 3D 位置：投影誤差 ≤ search_px，且描述子相似度 ≥ min_similarity、明顯優於第二名。

    附近的幀：相機相距 radius_m 以內、光軸夾角 ≤ max_angle_deg（同一圈前後幀與其他圈經過同地點的幀）。
    每個特徵最多屬於一條軌跡；已屬於其他軌跡的特徵不動；每條軌跡每幀最多一個觀測。
    descs：每幀描述子（NumPy 或 CuPy，列已正規化）。回傳 (新觀測的軌跡, 幀, 特徵全域索引, 統計)。
    """
    from scipy.spatial import cKDTree
    n_frames, n_tracks = len(poses), len(X)
    P = np.stack(poses)
    xyz, fwd = P[:, :3, 3], P[:, :3, 2]
    Twc = np.stack([se3.inv(T) for T in poses])
    w, h = 2.0 * K[0, 2], 2.0 * K[1, 2]
    order = np.argsort(obs_frame, kind='stable')
    bounds = np.searchsorted(obs_frame[order], np.arange(n_frames + 1))
    tracks_in = [np.unique(obs_track[order[bounds[i]:bounds[i + 1]]]) for i in range(n_frames)]
    dim = None
    tdesc = None
    for i in range(n_frames):   # 軌跡描述子 = 各觀測描述子的平均（再正規化）
        sel = order[bounds[i]:bounds[i + 1]]
        if not len(sel):
            continue
        d = _as_numpy(descs[i])
        if tdesc is None:
            dim = d.shape[1]
            tdesc = np.zeros((n_tracks, dim), dtype=np.float32)
        np.add.at(tdesc, obs_track[sel], d[obs_node[sel] - offsets[i]])
    tdesc /= np.linalg.norm(tdesc, axis=1, keepdims=True) + 1e-9
    owner = np.full(int(offsets[-1]), -1, dtype=np.int64)
    owner[obs_node] = obs_track
    have = set(zip(obs_track.tolist(), obs_frame.tolist()))
    cos_max = np.cos(np.deg2rad(gcfg['max_angle_deg']))
    out_t, out_f, out_n, out_s = [], [], [], []
    stat = dict(candidates=0, taken_by_other_track=0)
    for i in range(n_frames):
        k = int(offsets[i + 1] - offsets[i])
        if k < 2:
            continue
        near = np.flatnonzero((np.linalg.norm(xyz - xyz[i], axis=1) < gcfg['radius_m']) & (fwd @ fwd[i] >= cos_max))
        near = near[near != i]
        if not len(near):
            continue
        cand = np.unique(np.concatenate([tracks_in[j] for j in near]))
        Xc = X[cand] @ Twc[i, :3, :3].T + Twc[i, :3, 3]
        ok = (Xc[:, 2] > 0.5) & (Xc[:, 2] < gcfg['max_depth_m'])
        cand, Xc = cand[ok], Xc[ok]
        pr = np.c_[K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2], K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]]
        ok = (pr[:, 0] >= 0) & (pr[:, 0] < w) & (pr[:, 1] >= 0) & (pr[:, 1] < h)
        cand, pr = cand[ok], pr[ok]
        if not len(cand):
            continue
        stat['candidates'] += len(cand)
        kp = all_kpts[offsets[i]:offsets[i + 1]]
        dist, idx = cKDTree(kp).query(pr, k=4, distance_upper_bound=gcfg['search_px'])
        valid = np.isfinite(dist)
        idx = np.where(valid, idx, 0)
        d = _as_numpy(descs[i])
        sims = np.where(valid, np.einsum('nd,nkd->nk', tdesc[cand], d[idx]), -2.0)
        rows = np.arange(len(cand))
        best = np.argmax(sims, axis=1)
        s1 = sims[rows, best]
        sims[rows, best] = -2.0
        s2 = sims.max(axis=1)
        acc = (s1 >= gcfg['min_similarity']) & ((s2 < -1.0) | (s1 - s2 > gcfg['similarity_margin']))
        for c, ft, s in zip(cand[acc], idx[rows, best][acc], s1[acc]):
            if (int(c), i) in have:
                continue
            if owner[offsets[i] + ft] >= 0:
                stat['taken_by_other_track'] += 1
                continue
            out_t.append(int(c)); out_f.append(i); out_n.append(int(offsets[i] + ft)); out_s.append(float(s))
    if not out_t:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z, dict(stat, accepted=0, cross_lap=0)
    out_t, out_f, out_n, out_s = map(np.array, (out_t, out_f, out_n, out_s))
    seen_n, seen_tf, keep = set(), set(), []
    for q in np.argsort(-out_s, kind='stable'):   # 衝突時留相似度最高的
        a, b = int(out_n[q]), (int(out_t[q]), int(out_f[q]))
        if a in seen_n or b in seen_tf:
            continue
        seen_n.add(a); seen_tf.add(b); keep.append(q)
    keep = np.sort(np.array(keep, dtype=np.int64))
    out_t, out_f, out_n = out_t[keep], out_f[keep], out_n[keep]
    lo = np.full(n_tracks, 10 ** 9, dtype=np.int64)
    hi = np.zeros(n_tracks, dtype=np.int64)
    np.minimum.at(lo, obs_track, obs_frame)
    np.maximum.at(hi, obs_track, obs_frame)
    cross = (np.abs(out_f - lo[out_t]) > gcfg['lap_gap_frames']) & (np.abs(out_f - hi[out_t]) > gcfg['lap_gap_frames'])
    stat.update(accepted=int(len(out_t)), cross_lap=int(cross.sum()))
    return out_t.astype(np.int64), out_f.astype(np.int64), out_n.astype(np.int64), stat


def _as_numpy(d):
    return d.get() if hasattr(d, 'get') else np.asarray(d, dtype=np.float32)


# ---------------------------------------------------------------- 開發用：把 BA 的輸入存檔，之後可以不重跑前端直接試 BA
def save_input(path, K, poses, kpts, links, between_raw, sizes=None):
    counts = np.array([len(k) for k in kpts], dtype=np.int64)
    np.savez_compressed(
        path, K=K, poses=np.stack(poses), counts=counts,
        sizes=np.concatenate([np.asarray(s, dtype=np.float32).ravel() for s in sizes]) if sizes is not None
        else np.zeros(0, dtype=np.float32),
        kpts=np.concatenate([np.asarray(k, dtype=np.float32).reshape(-1, 2) for k in kpts]),
        link_frames=np.array([[a, b, len(p)] for a, b, p in links], dtype=np.int64).reshape(-1, 3),
        link_pairs=np.concatenate([p for _, _, p in links]) if links else np.zeros((0, 2), dtype=np.int64),
        between_ij=np.array([[i, j, kind] for i, j, _, kind in between_raw], dtype=np.int64).reshape(-1, 3),
        between_T=np.stack([T for _, _, T, _ in between_raw]) if between_raw else np.zeros((0, 4, 4)))


def load_input(path):
    d = np.load(path)
    off = np.concatenate([[0], np.cumsum(d['counts'])])
    kp = d['kpts']
    kpts = [kp[off[i]:off[i + 1]].astype(np.float64) for i in range(len(off) - 1)]
    links, s = [], 0
    pairs = d['link_pairs']   # 只讀一次：NpzFile 每次取值都會重新解壓整個陣列，切片 view 又會把整份陣列留在記憶體
    for a, b, m in d['link_frames']:
        links.append((int(a), int(b), pairs[s:s + m].copy()))
        s += m
    between_raw = [(int(i), int(j), T, int(kind)) for (i, j, kind), T in zip(d['between_ij'], d['between_T'])]
    return d['K'], list(d['poses']), kpts, links, between_raw
