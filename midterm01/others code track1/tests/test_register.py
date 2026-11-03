import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from sdcloc.register import estimate_normals, icp_point_to_plane, observability


def make_scene(rng):
    # 地面 + 左右兩面牆 + 一面前牆，點距 0.2 m
    g = np.stack(np.meshgrid(np.arange(-5, 25, .2), np.arange(-8, 8, .2)), -1).reshape(-1, 2)
    ground = np.column_stack([g, np.zeros(len(g))])
    wl = np.stack(np.meshgrid(np.arange(-5, 25, .2), np.arange(0, 4, .2)), -1).reshape(-1, 2)
    wall_l = np.column_stack([wl[:, 0], np.full(len(wl), 8.0), wl[:, 1]])
    wall_r = np.column_stack([wl[:, 0], np.full(len(wl), -8.0), wl[:, 1]])
    wf = np.stack(np.meshgrid(np.arange(-8, 8, .2), np.arange(0, 4, .2)), -1).reshape(-1, 2)
    wall_f = np.column_stack([np.full(len(wf), 25.0), wf[:, 0], wf[:, 1]])
    scene = np.vstack([ground, wall_l, wall_r, wall_f])
    return scene + rng.normal(0, .01, scene.shape)


def test_normals_on_planes():
    rng = np.random.default_rng(0)
    scene = make_scene(rng)
    n, pl = estimate_normals(scene, k=10)
    ground = scene[:, 2] < 0.05
    assert np.mean(n[ground, 2] > 0.95) > 0.95
    walls = np.abs(np.abs(scene[:, 1]) - 8) < 0.05
    assert np.mean(np.abs(n[walls, 1]) > 0.95) > 0.9
    assert np.median(pl) > 0.4      # 規則格點的鄰域各向異性，planarity 不會到 1


def test_icp_recovers_pose():
    rng = np.random.default_rng(1)
    scene = make_scene(rng)
    normals, _ = estimate_normals(scene, k=10)
    tree = cKDTree(scene)
    # 真實 sensor pose：位移 (3, 1, 1.5)，yaw 10°
    R_true = Rotation.from_euler('z', 10, degrees=True).as_matrix()
    t_true = np.array([3.0, 1.0, 1.5])
    src = (scene[rng.choice(len(scene), 4000, replace=False)] - t_true) @ R_true   # sensor 座標
    R0 = Rotation.from_euler('z', 6, degrees=True).as_matrix()
    t0 = t_true + [0.6, -0.4, 0.3]
    out = icp_point_to_plane(src, scene, normals, tree, R0, t0, max_corr=1.5, huber=0.3)
    assert out['converged'] or out['iters'] >= 10
    assert np.linalg.norm(out['t'] - t_true) < 0.03
    assert abs(np.degrees(np.arctan2(out['R'][1, 0], out['R'][0, 0])) - 10) < 0.1
    ob = observability(out['information'])
    assert ob['rank'] == 6 and ob['xy_min'] > 0


def test_icp_estimates_scale():
    rng = np.random.default_rng(3)
    scene = make_scene(rng)
    normals, _ = estimate_normals(scene, k=10)
    tree = cKDTree(scene)
    R_true = Rotation.from_euler('z', 5, degrees=True).as_matrix()
    t_true = np.array([2.0, 0.5, 1.2])
    src = ((scene[rng.choice(len(scene), 5000, replace=False)] - t_true) @ R_true) / 8.0   # 感測器單位 = 1/8 公尺
    out = icp_point_to_plane(src, scene, normals, tree, R_true, t_true + [0.3, -0.2, 0.1], max_corr=1.5, huber=0.3,
                             estimate_scale=True, s0=7.0, max_iter=40)
    assert abs(out['s'] - 8.0) < 0.05
    assert np.linalg.norm(out['t'] - t_true) < 0.05
    assert out['scale_info'] > 0


def test_class_match_ignores_objects_absent_from_map():
    """地圖只有地面＋牆；點雲多了一台不在地圖裡的「車」（1.5 m 高的方塊，離牆 3 m）。
    不分類時車會被拉去貼牆而偏掉；分類配對時車配不到非地面點，解回正確位置。"""
    rng = np.random.default_rng(5)
    scene = make_scene(rng)
    normals, _ = estimate_normals(scene, k=10)
    tree = cKDTree(scene)
    map_ground = normals[:, 2] > 0.9
    R_true = np.eye(3); t_true = np.array([0.0, 0.0, 1.5])
    base = scene[rng.choice(len(scene), 4000, replace=False)]
    car = np.stack(np.meshgrid(np.arange(8, 12, .2), np.arange(4.5, 5.0, .2), np.arange(0.3, 1.6, .2)), -1).reshape(-1, 3)
    src_world = np.vstack([base, car])
    src = src_world - t_true
    src_ground = src_world[:, 2] < 0.3
    t0 = t_true + [0.0, 1.5, 0.0]        # 初值偏 1.5 m，讓車有機會去貼左牆（y=8）
    bad = icp_point_to_plane(src, scene, normals, tree, R_true, t0, max_corr=2.0, huber=0.3, max_iter=40)
    good = icp_point_to_plane(src, scene, normals, tree, R_true, t0, max_corr=2.0, huber=0.3, max_iter=40,
                              src_ground=src_ground, map_ground=map_ground)
    assert np.linalg.norm(good['t'] - t_true) < 0.05
    assert np.linalg.norm(good['t'] - t_true) <= np.linalg.norm(bad['t'] - t_true) + 1e-6


def test_observability_ground_only():
    rng = np.random.default_rng(2)
    g = np.stack(np.meshgrid(np.arange(-5, 25, .2), np.arange(-8, 8, .2)), -1).reshape(-1, 2)
    ground = np.column_stack([g, rng.normal(0, .01, len(g))])
    normals, _ = estimate_normals(ground, k=10)
    tree = cKDTree(ground)
    src = ground[::3] - [0, 0, 1.0]
    out = icp_point_to_plane(src, ground, normals, tree, np.eye(3), np.array([0, 0, 1.0]), max_iter=3)
    ob = observability(out['information'])
    assert ob['rank'] == 3            # 只有 z、roll、pitch
    assert ob['xy_max'] < 1e-2 * ob['z'] and ob['yaw'] < 1e-2 * ob['z']   # 雜訊法向量留下一點假資訊
