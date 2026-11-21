"""所有可調參數集中在這個檔案（依專案規則：改參數就改這裡，不靠命令列選項或多份設定檔）。

數值目前是依模型的物理意義給的初值，尚未在真實資料上驗證；每次 run 會把整份參數寫進 test_info.json。
要做消融實驗時，改 EXPERIMENT 選一組覆寫值即可。
"""
import copy

PARAMS = dict(
    seed=0,
    track=dict(
        # Track 名稱 -> 資料目錄（相對 midterm02/）與輸出檔名
        track01=dict(data='data/data_VSLAM-T1/Track1/data', result='result_1.csv', code='01'),
        track02=dict(data='data/data_VSLAM-T2/Track2/data', result='result_2.csv', code='02'),
    ),
    preprocess=dict(
        min_depth_m=0.3,          # 太近的點多為車體／雜訊
        max_depth_m=40.0,         # LoGeR 天空與極遠處深度不可靠；超過就丟
        voxel_m=0.25,             # 體素降採樣邊長
        radius_outlier=dict(enabled=True, radius_m=0.6, min_neighbors=4),  # 孤立點（飛點）移除
        sky_color=dict(enabled=False, min_blue_ratio=0.40, min_brightness=0.55, min_height_m=1.0),  # 顏色＋高度的天空遮罩
        max_points=60000,         # 降採樣後上限（等間隔取樣，結果可重現）
    ),
    registration=dict(
        method='gicp',            # 'gicp' 或 'icp_plane'
        prior='visual',           # 'visual'（影像特徵 PnP 初值）、'motion'（等速模型）、'identity'（上一幀 pose）
        stages=[dict(max_corr=1.0, iterations=15), dict(max_corr=0.4, iterations=30)],
        recovery_stages=[dict(max_corr=2.5, iterations=20), dict(max_corr=1.0, iterations=20),
                         dict(max_corr=0.4, iterations=30)],
        robust_k_m=0.10,          # Cauchy 尺度（法向距離，公尺）
        normal_radius_m=0.75,     # 估計法向／共變異數的鄰域半徑
        min_neighbors=6,
        planar_eps=1e-3,
        index_cell_m=1.0,
        min_correspondences=300,
        estimate_scale=False,     # True：每幀多解一個尺度（E10 尺度一致性實驗）
        scale_prior_weight=25.0,
    ),
    scale=dict(
        mode='none',              # 'none'：不處理（預設）；'ground_height'：以相機離地高度正規化每幀尺度。
                                  # full Track01 實測後者讓閉環處的 odometry 落差由 2.21 m 變 3.92 m（變差），故預設關閉
        min_below_m=0.6, max_forward_m=12.0, max_lateral_m=4.0, min_points=200,
        iterations=150, inlier_m=0.03, min_normal_y=0.9, min_inlier_ratio=0.3,
        max_change_ratio=0.25,    # 與上一幀尺度差超過這個比例視為地面擬合失敗，沿用上一幀
    ),
    fusion=dict(                  # 視覺位姿當作配準最佳化的先驗（confidence-aware fusion）
        enabled=True,
        point_sigma_m=0.15,       # 點雲對應的殘差標準差（實測 inlier RMSE 約 0.15–0.2 m）
        effective_points=50,      # LoGeR 的誤差在鄰近點之間高度相關，數千個對應只算這麼多個獨立量測
        visual_trans_sigma_m=0.05,
        visual_rot_sigma_deg=0.2,
        reference_inliers=300,    # PnP 內點數少於此值時，視覺標準差依 sqrt(reference/inliers) 放大
    ),
    gate=dict(                    # 品質閘門：全部通過才接受配準結果
        min_fitness=0.30,         # 轉彎時視野轉向新區域，與子圖重疊本來就低（實測 0.45–0.7），不能當成失敗
        max_inlier_rmse_m=0.22,
        max_translation_from_prior_m=1.0,   # 只在使用視覺初值時檢查
        max_rotation_from_prior_deg=4.0,
        max_step_translation_m=2.5,          # 單幀（0.1 s）位移上限，相當於 90 km/h
        max_step_rotation_deg=12.0,          # 單幀轉角上限
        max_visual_rotation_disagreement_deg=3.0,   # 配準出的相對旋轉 vs 影像估計（旋轉不受尺度影響）
        min_normalized_eigenvalue=1e-4,              # 幾何退化指標
    ),
    visual=dict(
        enabled=True,
        frontend='sift',          # 'sift'、'orb'、'superpoint_lightglue'（實驗，需 build-learned）
        max_keypoints=3000,
        ratio=0.8,                # Lowe ratio test
        mutual=True,
        essential_threshold_px=1.0,
        pnp_reprojection_px=2.0,
        pnp_iterations=300,
        min_pnp_inliers=40,
        min_pnp_inlier_ratio=0.35,
        min_essential_inliers=100,   # Essential 模式：內點數下限
        min_length_points=50,        # 估計平移長度至少要有這麼多個帶 3D 的內點
        min_flow_px=2.0,             # 特徵位移中位數低於此值視為幾乎靜止，改用 PnP
        max_depth_m=30.0,         # 取 3D 對應時的深度上限
        assoc_stride_px=5,        # 像素→3D 點查表的格距（LoGeR 點雲約每 4.9 px 一點，見 inspect_data）
        debug_every=50,           # 每幾幀存一張特徵匹配圖（0 = 不存）
    ),
    keyframe=dict(
        translation_m=1.0,
        rotation_deg=5.0,         # 轉彎時更密集加 keyframe，讓子圖跟上新視野
        max_frames_between=30,    # 追蹤失敗太久仍強制加 keyframe，避免地圖落後
    ),
    submap=dict(
        window_keyframes=12,      # 局部子圖使用最近幾個 keyframe
        voxel_m=0.25,
        max_points=400000,
    ),
    loop=dict(
        enabled=True,
        min_keyframe_gap=80,      # 與目前 keyframe 至少相隔這麼多個 keyframe 才算重訪
        min_votes=60,             # 描述子投票數下限
        vote_similarity=0.80,     # RootSIFT cosine 相似度門檻
        top_k=2,                  # 最多幾群候選（每群對應較早的一圈）
        per_cluster=2,            # 每群最多驗證幾個 keyframe
        cluster_separation=40,    # keyframe 編號相差超過這個值才算不同圈
        max_loops_per_keyframe=2,
        scale_dof=dict(enabled=True, prior_weight=0.0),   # 閉環配準時多解一個尺度（不加先驗），吸收兩圈點雲的尺度差
        fuse_visual=True,         # 閉環相對位姿以影像幾何為先驗
        min_essential_inliers=30,  # 影像門檻只負責篩掉明顯不相干的候選；真正把關的是後面的點雲配準與一致性檢查
        min_length_points=20,
        min_pnp_inliers=30,
        submap_half_window=6,
        min_fitness=0.55,
        max_inlier_rmse_m=0.22,
        max_translation_disagreement_m=1.0,   # 影像估計 vs 點雲配準 的一致性
        max_rotation_disagreement_deg=4.0,
        min_keyframes_between_loops=1,   # 1 = 每個 keyframe 都嘗試閉環（密集跨圈約束）
        # 點雲品質門檻（fitness／RMSE）沒過、但影像估計與點雲配準兩個獨立估計幾乎一致時仍接受（None = 關閉）。
        # 由來（2026-11-17，Track2）：起點／終點三個閉環 fitness 0.38（停車起步、重疊少），兩估計卻只差 4–8 cm、0.15–0.3°
        agreement_accept=None,
    ),
    scale_graph=dict(             # 結束時用相鄰幀深度比＋閉環尺度比解每幀尺度，修正里程計長度
        enabled=True,
        sequential_sigma=0.01,    # 相鄰幀深度比（log）的標準差
        loop_sigma=0.03,          # 閉環尺度比（log）的標準差
        missing_sigma=0.05,       # 沒有量測的相鄰幀：假設尺度不變，但信心低
        min_ratio=0.8,            # 深度比超出 [0.8, 1.25] 視為離群
        # 'graph'：上述相鄰幀深度比＋閉環尺度比；'height'：每幀 log 尺度＝log(LoGeR 相機離地高度／中位數)，
        # 高度取 ±height_median_half 幀中位數。Track2 上 'graph' 的相鄰幀深度比有 +0.25%／幀 的偏差，4206 幀累積發散（見 doc/03）
        mode='graph',
        height_median_half=15,
    ),
    graph=dict(
        prior_sigma=1e-4,
        odom_rot_sigma_rad=0.01, odom_trans_sigma_m=0.03,
        odom_scale_sigma=0.10,    # 里程計長度的比例誤差（同地點跨圈的 LoGeR 深度比實測標準差約 8%）
        loop_rot_sigma_rad=0.02, loop_trans_sigma_m=0.15,
        huber_k=1.345,
    ),
    bundle=dict(                  # 收尾時對全部幀做 bundle adjustment（多幀特徵軌跡的重投影最佳化），見 gslam/bundle.py
        enabled=True,
        min_track_length=3,       # 一個特徵至少要在 3 幀被看到才用
        # 以下三項是 gslam 0.4.0 試的「更多影像配對」，三項全開時 Public 由 0.085 變差為 0.097，所以預設關閉（見 E14*）
        skip_frames=1,            # 每幀除了和前一幀，也和前第 2…N 幀比對（1 = 只比前一幀）
        revisit_links=False,      # BA 前依 pose graph 的位姿，替每一幀找其他圈經過同一地點的幀來比對
        revisit_radius_m=3.0, revisit_max_angle_deg=37.0,   # 「同一地點」：相機相距 3 m 內、光軸夾角 37° 內
        link_ransac_px=1.0, link_min_inliers=30,   # 補充影像對的幾何驗證（Essential RANSAC）
        resolve_conflicts=False,  # 軌跡在同一幀出現兩個特徵時：True = 逐條匹配重新合併；False = 整條丟掉
        min_tracks=500,
        min_parallax_deg=1.0,     # 視差太小的點深度定不出來，不用
        triangulation_gate_px=6.0,   # 以初始位姿三角化後，重投影誤差中位數超過就丟（多半是錯誤匹配或動態物體）
        max_observations=0,       # 觀測數上限（0 = 不限）
        pixel_sigma=1.0,          # SIFT 特徵定位誤差（px）
        huber_k=1.345,
        prune_px=4.0,             # 第一輪後丟掉重投影誤差超過這個值的觀測
        # 依 SIFT 尺度給觀測不同權重：sigma = pixel_sigma × clip((size/size_ref)^power, lo, hi)
        size_weight=dict(enabled=False, size_ref=6.0, power=0.4, lo=0.6, hi=2.5),
        # BA 後的投影引導補觀測：把附近幀（相距 radius_m、夾角 ≤ max_angle_deg，含其他圈）看到的 3D 點投影到本幀，
        # 在 search_px 內找描述子相似度 ≥ min_similarity 且比第二名高 similarity_margin 的特徵，加入後再解一次
        # LoGeR 深度因子：相機到 3D 點的距離 ≈ exp(s_i) × LoGeR 距離；s_i 是每幀未知的 LoGeR 尺度偏差，相鄰幀平滑。
        # sigma_log_depth：同一幀內相對深度的誤差（實測 MAD 0.087）；sigma_log_scale_step：相鄰幀 s 的變化（實測 0.018）。
        # solves：交替求解次數（每次先解 s 再解 BA）
        depth=dict(enabled=False, sigma_log_depth=0.10, sigma_log_scale_step=0.02, huber_k=1.345, solves=3),
        # IMU 加速度計的平均速度變化因子（gslam/imu.py；只有 Track01 有 IMU，沒有 imu.csv 時自動略過）。
        # half_window：平均速度的半窗（幀，5 = 1 s）；offset_s：camera time + offset = imu time（陀螺儀實測 −0.02 s）；
        # sigma_m：每軸雜訊（m/s），None = 用第一次擬合的垂直殘差（實測約 0.022）；solves：BA 求解次數（第 1 次不加 IMU）；
        # holdout_block_frames：驗證用，>0 時只在偶數塊加因子，奇數塊留給 IMU 一致性檢查（正式結果用 0）
        imu=dict(enabled=False, half_window=5, offset_s=-0.02, sigma_m=None, min_sigma_m=0.01, huber_k=1.345,
                 solves=3, holdout_block_frames=0),
        # 相機自校正（Cal3DS2）：第 2 輪起把內參與徑向畸變 k1、k2 當變數。sigmas 是先驗標準差，順序
        # [fx, fy, s, cx, cy, k1, k2, p1, p2]：焦距固定（fx 已由陀螺儀驗證）、主點 5 px、k1/k2 寬鬆、切向畸變固定
        calibration=dict(enabled=False, solves=3, sigmas=[1e-6, 1e-6, 1e-6, 5.0, 5.0, 0.5, 0.5, 1e-6, 1e-6]),
        # 學習式特徵軌跡（gslam/learned_tracks.py）：只用在 BA。extractor：'superpoint' 或 'aliked'；
        # combine_sift：True = SIFT 與學習式特徵兩組軌跡一起進 BA；skip_frames：每幀和前 1…N 幀配對（實測 3 沒有幫助）
        learned=dict(enabled=False, extractor='superpoint', max_keypoints=4096, skip_frames=1, combine_sift=True,
                     ransac_px=1.0, min_inliers=30),
        # 跨圈點雲疊合（使用者 2026-11-16 的想法）：BA 解完後，把後面幾圈每個 keyframe 的 LoGeR 點雲，
        # 配準到較早一圈附近 keyframe 拼成的子圖（位姿用 BA 結果、放開尺度），通過品質閘門的當成相對位姿邊，再解一次 BA。
        # 「同一地點」：相機相距 radius_m、光軸夾角 max_angle_deg 內；子圖取最近 keyframe 前後 window_keyframes 個。
        # 離線檢查（outputs/cache/dev39、dev40）：這些修正和影像對極幾何不一致，預期會變差；保留為驗證實驗。
        cloud_overlap=dict(enabled=False, radius_m=3.0, max_angle_deg=30.0, window_keyframes=6, min_fitness=0.6,
                           max_rmse_m=0.25, max_correction_m=2.0, scale_prior_weight=0.5,
                           rot_sigma_deg=0.5, trans_sigma_m=0.15, huber_k=None),
        guided=dict(enabled=False, radius_m=5.0, max_angle_deg=30.0, search_px=3.0, min_similarity=0.85,
                    similarity_margin=0.05, max_depth_m=40.0, lap_gap_frames=200),
        rounds=2,
        max_iterations=40, relative_error_tol=1e-5,
        prior_sigma=1e-4,
        odom_rot_sigma_rad=0.01, odom_trans_sigma_m=0.05,   # 相鄰幀里程計邊：刻意很鬆（長度來自 LoGeR 深度，不可靠）
        use_loop_edges=True,
        lap_gap_frames=200,       # 一條軌跡的首尾幀相隔超過這個數，統計上算「跨圈」
        save_input=True,          # 把 BA 輸入存到 debug/bundle_input.npz（開發時可不重跑前端）
    ),
    viz=dict(                     # RViz 顯示用（只影響畫面，不影響 pose）
        max_depth_m=15.0, frame_point_step=2, keyframe_point_step=6, submap_max_points=60000,
    ),
    debug=dict(cloud_every=100),  # 每幾幀存一張點雲俯視圖（0 = 不存）
    # 只影響速度、不影響結果：背景執行緒預先讀圖＋抽 SIFT 的執行緒數（0 = 關閉，在主執行緒逐幀計算）
    runtime=dict(prefetch_workers=8),
)

# 消融實驗：名稱 -> 要覆寫的鍵（用 'a.b' 路徑）。EXPERIMENT 指定目前要跑哪一組。
EXPERIMENTS = {
    'E2_motion_icp': {'registration.prior': 'motion', 'registration.method': 'icp_plane', 'visual.enabled': False, 'loop.enabled': False},
    'E3_motion_gicp': {'registration.prior': 'motion', 'registration.method': 'gicp', 'visual.enabled': False, 'loop.enabled': False},
    'E5_visual_icp': {'registration.prior': 'visual', 'registration.method': 'icp_plane', 'loop.enabled': False, 'fusion.enabled': False},
    'E6_visual_gicp': {'registration.prior': 'visual', 'registration.method': 'gicp', 'loop.enabled': False, 'fusion.enabled': False},
    'E8_no_loop': {'loop.enabled': False},
    'E9_full': {},
    'E9d_sparse_loops': {'loop.min_keyframes_between_loops': 5, 'loop.max_loops_per_keyframe': 1, 'loop.top_k': 1,
                         'loop.per_cluster': 3, 'loop.fuse_visual': False, 'graph.odom_scale_sigma': 0.0, 'graph.odom_trans_sigma_m': 0.05,
                         'loop.scale_dof': {'enabled': False, 'prior_weight': 0.0}},   # 2611160111（Public 0.369）的設定
    'E9e_isotropic_odom': {'graph.odom_scale_sigma': 0.0, 'graph.odom_trans_sigma_m': 0.05},
    'E9f_loop_icp_only': {'loop.fuse_visual': False, 'loop.scale_dof': {'enabled': False, 'prior_weight': 0.0}},
    'E11_no_scale_graph': {'scale_graph.enabled': False},
    'E12_no_bundle': {'bundle.enabled': False},   # gslam 0.2.0（2611160533，Public 0.144）的設定
    # 預設的 BA（2611161129，Public 0.085）：只用相鄰幀與閉環的匹配。E13 明確寫出來，等同預設
    'E13_bundle_basic': {'bundle.skip_frames': 1, 'bundle.revisit_links': False, 'bundle.resolve_conflicts': False},
    # 更多影像配對：三項全開 = 2611161431（Public 0.097，比預設差）
    'E14_bundle_more_pairs': {'bundle.skip_frames': 3, 'bundle.revisit_links': True, 'bundle.resolve_conflicts': True},
    'E16_size_weight': {'bundle.size_weight': dict(enabled=True, size_ref=6.0, power=0.4, lo=0.6, hi=2.5)},
    'E18_depth': {'bundle.depth': dict(enabled=True, sigma_log_depth=0.10, sigma_log_scale_step=0.02, huber_k=1.345,
                                       solves=3)},
    'E17_guided': {'bundle.guided': dict(enabled=True, radius_m=5.0, max_angle_deg=30.0, search_px=3.0,
                                         min_similarity=0.85, similarity_margin=0.05, max_depth_m=40.0,
                                         lap_gap_frames=200)},
    # IMU 平均速度變化因子（gslam/imu.py）。離線（2611161129 的 BA 輸入）：軌跡只移動 0.004 m（132 幀視窗中位數），
    # IMU 一致性 0.1325 → 0.1282 m/s；影響小於 Public 能分辨的程度，尚未跑 full
    'E19_imu': {'bundle.imu': dict(enabled=True, half_window=5, offset_s=-0.02, sigma_m=None, min_sigma_m=0.01,
                                   huber_k=1.345, solves=3, holdout_block_frames=0)},
    # 學習式特徵（SuperPoint＋LightGlue）與 SIFT 兩組軌跡一起進 BA。離線（2611161129 的 BA 輸入）：
    # 與基本 BA 差 0.028 m（132 幀視窗中位數），IMU 一致性 0.133 → 0.121 m/s
    'E20_learned': {'bundle.learned': dict(enabled=True, extractor='superpoint', max_keypoints=4096, skip_frames=1,
                                           combine_sift=True, ransac_px=1.0, min_inliers=30)},
    'E20a_learned_aliked': {'bundle.learned': dict(enabled=True, extractor='aliked', max_keypoints=4096, skip_frames=1,
                                                   combine_sift=False, ransac_px=1.0, min_inliers=30)},
    # 相機自校正：離線（2611161129 的 BA 輸入）估出 k1≈−0.056、k2≈+0.086、cy −9 px，前後半段各自估計一致；
    # BA 代價 −12.6%、IMU 一致性 0.135 → 0.124、陀螺儀偏航差 0.0059 → 0.0049 rad/s
    'E21_selfcalib': {'bundle.calibration': dict(enabled=True, solves=3,
                                                 sigmas=[1e-6, 1e-6, 1e-6, 5.0, 5.0, 0.5, 0.5, 1e-6, 1e-6])},
    # 自校正＋學習式特徵（SuperPoint＋LightGlue 與 SIFT 兩組軌跡）
    'E22_selfcalib_learned': {
        'bundle.calibration': dict(enabled=True, solves=3, sigmas=[1e-6, 1e-6, 1e-6, 5.0, 5.0, 0.5, 0.5, 1e-6, 1e-6]),
        'bundle.learned': dict(enabled=True, extractor='superpoint', max_keypoints=4096, skip_frames=1,
                               combine_sift=True, ransac_px=1.0, min_inliers=30)},
    # 自校正＋依位姿的跨圈影像配對（E14b 的跨圈配對，在正確的相機模型下重測；Private 看整條路線的跨圈一致性）
    'E24_selfcalib_revisit': {
        'bundle.calibration': dict(enabled=True, solves=3, sigmas=[1e-6, 1e-6, 1e-6, 5.0, 5.0, 0.5, 0.5, 1e-6, 1e-6]),
        'bundle.revisit_links': True, 'bundle.resolve_conflicts': True},
    # E22＋E24：自校正＋學習式特徵＋依位姿的跨圈配對（學習式特徵也會配對跨圈影像對）
    'E25_selfcalib_learned_revisit': {
        'bundle.calibration': dict(enabled=True, solves=3, sigmas=[1e-6, 1e-6, 1e-6, 5.0, 5.0, 0.5, 0.5, 1e-6, 1e-6]),
        'bundle.learned': dict(enabled=True, extractor='superpoint', max_keypoints=4096, skip_frames=1,
                               combine_sift=True, ransac_px=1.0, min_inliers=30),
        'bundle.revisit_links': True, 'bundle.resolve_conflicts': True},
    # E21＋跨圈點雲疊合邊（見 bundle.cloud_overlap）：BA 後依點雲疊合再解一次
    'E26_selfcalib_cloud_overlap': {
        'bundle.calibration': dict(enabled=True, solves=3, sigmas=[1e-6, 1e-6, 1e-6, 5.0, 5.0, 0.5, 0.5, 1e-6, 1e-6]),
        'bundle.cloud_overlap': dict(enabled=True, radius_m=3.0, max_angle_deg=30.0, window_keyframes=6,
                                     min_fitness=0.6, max_rmse_m=0.25, max_correction_m=2.0, scale_prior_weight=0.5,
                                     rot_sigma_deg=0.5, trans_sigma_m=0.15)},
    # E26 的點雲邊（σ 0.15、Huber 1.345）對上幾百萬個影像觀測幾乎沒有作用（軌跡只動約 1 cm）。
    # E26b：讓點雲疊合真的主導跨圈關係——平移 σ 0.02、旋轉 0.1°，Huber 門檻放到 1000（等於不截斷），檢驗想法本身
    'E26b_selfcalib_cloud_overlap_strong': {
        'bundle.calibration': dict(enabled=True, solves=3, sigmas=[1e-6, 1e-6, 1e-6, 5.0, 5.0, 0.5, 0.5, 1e-6, 1e-6]),
        'bundle.cloud_overlap': dict(enabled=True, radius_m=3.0, max_angle_deg=30.0, window_keyframes=6,
                                     min_fitness=0.6, max_rmse_m=0.25, max_correction_m=2.0, scale_prior_weight=0.5,
                                     rot_sigma_deg=0.1, trans_sigma_m=0.02, huber_k=1000.0)},
    # E21 修正（Track2）：尺度圖改用相機離地高度（不漂移的絕對尺度）＋兩個獨立估計一致時接受低 fitness 的閉環
    'E27_selfcalib_height': {
        'bundle.calibration': dict(enabled=True, solves=3, sigmas=[1e-6, 1e-6, 1e-6, 5.0, 5.0, 0.5, 0.5, 1e-6, 1e-6]),
        'scale_graph.mode': 'height',
        'loop.agreement_accept': dict(max_translation_m=0.25, max_rotation_deg=1.0, min_fitness=0.3)},
    # E27＋SuperPoint／LightGlue 軌跡（與 SIFT 合併）。Track2 給 F1 用：scripts/droid_ba.py --experiment 這個名稱時，
    # 在 BA 輸入的同一組影像對上抽學習式特徵（不用重跑 gslam）。Track1 的 E22 比 E21 差 0.009，這裡重測 Track2
    'E28_selfcalib_height_learned': {
        'bundle.calibration': dict(enabled=True, solves=3, sigmas=[1e-6, 1e-6, 1e-6, 5.0, 5.0, 0.5, 0.5, 1e-6, 1e-6]),
        'scale_graph.mode': 'height',
        'loop.agreement_accept': dict(max_translation_m=0.25, max_rotation_deg=1.0, min_fitness=0.3),
        'bundle.learned': dict(enabled=True, extractor='superpoint', max_keypoints=4096, skip_frames=1,
                               combine_sift=True, ransac_px=1.0, min_inliers=30)},
    'E15_more_features': {'visual.max_keypoints': 6000},   # 每張影像的 SIFT 特徵數加倍，BA 的觀測更多
    'E14a_bundle_skip_only': {'bundle.skip_frames': 3, 'bundle.resolve_conflicts': True},
    'E14b_bundle_revisit_only': {'bundle.revisit_links': True, 'bundle.resolve_conflicts': True},
    'E10a_ground_scale': {'scale.mode': 'ground_height'},
    'E9b_no_fusion': {'fusion.enabled': False},
    'E10b_scale_dof': {'registration.estimate_scale': True},
    # （不是消融）速度改動的等價檢查：E21 但關掉背景預先抽 SIFT；短段結果應與 E21 逐位元相同
    'Z21_selfcalib_no_prefetch': {
        'bundle.calibration': dict(enabled=True, solves=3, sigmas=[1e-6, 1e-6, 1e-6, 5.0, 5.0, 0.5, 0.5, 1e-6, 1e-6]),
        'runtime.prefetch_workers': 0},
}
EXPERIMENT = 'E9_full'


def load(experiment=None):
    """回傳套用實驗覆寫後的參數副本與實驗名稱。"""
    name = experiment or EXPERIMENT
    params = copy.deepcopy(PARAMS)
    for path, value in EXPERIMENTS[name].items():
        node = params
        keys = path.split('.')
        for k in keys[:-1]:
            node = node[k]
        if keys[-1] not in node:
            raise KeyError('未知參數 %s' % path)
        node[keys[-1]] = value
    return params, name
