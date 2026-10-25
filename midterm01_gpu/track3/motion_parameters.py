"""v5 因果影像尺度與停車約束；參數集中在 Python，執行指令不變。

pixel 門檻描述可辨識的影像運動，不是定位精度。靜止須同時通過
影像、空間分布、持續時間和 raw gyro 檢查；IMU 低加速度本身不能證明停車。
"""
VISUAL_VELOCITY_STD_MPS = 1.0  # 未校正工程模型；間隔平均速度近似當前速度。
STATIONARY_VELOCITY_STD_MPS = 0.05
ENABLE_MOVING_VISUAL_SCALE = False  # 移動時讓 map Sim(3) 校正尺度，不串乘未驗證的視覺尺度。

PARAMETERS = dict(
    max_features=800, min_tracks=80, min_grid_cells=12,
    forward_backward_px=1.0, depth_projection_px=4.0,
    quiet_median_px=0.25, quiet_p90_px=0.75,
    anchor_median_px=0.5, anchor_p90_px=1.25,
    quiet_duration_s=0.3, max_raw_gyro_radps=0.02,
    scale_max_log_mad=0.06, scale_min_pair_fraction=0.6,
    scale_min_points=60, scale_max_points=256,
    stationary_max_depth_m=80.0,
    keyframe_median_px=20.0, keyframe_max_age_s=2.0,
    pnp_min_inliers=40, pnp_min_fraction=0.5, pnp_reprojection_px=3.0,
    max_speed_mps=50.0, max_rotation_rate_radps=2.0,
)
