"""將已保存結果畫成三張教學圖；曲線不是真值，與 prior 的差也不是 ATE。

輸入是 scipy.io.loadmat(..., simplify_cells=True)['sdc'] 的 dict。
只建立 Figure；顯示、backend 與輸出檔案由呼叫端統一處理。
"""
import numpy as np
import matplotlib.pyplot as plt


BLUE = (0.15, 0.35, 0.75)
ORANGE = (0.85, 0.35, 0.10)
GREEN = (0.10, 0.55, 0.35)
GRAY = (0.65, 0.65, 0.65)


def _vector(d, name):
    return np.asarray(d[name]).reshape(-1)


def _panel(ax, title, xlabel='Time (s)', ylabel=''):
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, aspect='auto')
    ax.grid(True)


def _legend(ax):
    ax.legend(loc='best', fontsize=8)


def _wrapped(value):
    return np.arctan2(np.sin(value), np.cos(value))


def plot_saved_run(d):
    """回傳三組 (stem, Figure)，等價呈現原 MATLAB 圖的觀測與差值。

    map_xy 包含拒絕候選，不能把候選曲線視為實際接受的定位輸出。
    時間由資料匯出端先減整數奈秒 origin 再轉秒，此處不重算時間。
    GNSS 模型 z=p+b，因此 map-GNSS 疊圖使用負 bias。
    """
    t = _vector(d, 'time_s')
    good = _vector(d, 'accepted').astype(bool)
    temporal = _vector(d, 'temporal_ok').astype(bool)
    gnss = np.asarray(d['gnss_xy'])
    prior = np.asarray(d['prior_xy'])
    reference = np.asarray(d['reference_xy'])
    latest = np.asarray(d['latest_xy'])
    origin = gnss[0]
    figures = []

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4), layout='constrained')
    ax = axes[0, 0]
    ax.plot(*(gnss - origin).T, '.', color=GRAY, label='Raw GNSS')
    for values, color, label in ((prior, BLUE, 'Sensor prior'),
                                  (reference, GREEN, 'Previous best'),
                                  (latest, ORANGE, 'Slow-map graph')):
        ax.plot(*(values - origin).T, color=color, label=label)
    _panel(ax, 'Same map coordinates; pan/zoom to inspect',
           'Map X relative to first GNSS (m)', 'Map Y relative to first GNSS (m)')
    ax.set_aspect('equal', adjustable='datalim')
    _legend(ax)
    delta = latest - prior
    old_delta = reference - prior
    for index, ax in enumerate((axes[0, 1], axes[1, 0])):
        ax.plot(t, old_delta[:, index], color=GREEN, label='Previous best - prior')
        ax.plot(t, delta[:, index], color=ORANGE, label='Graph - prior')
        _panel(ax, f'Map {"XY"[index]} correction; not truth error',
               'Time from GNSS origin (s)', 'Position change from prior (m)')
        _legend(ax)
    ax = axes[1, 1]
    prior_yaw = _vector(d, 'prior_yaw_rad')
    ax.plot(t, np.rad2deg(_wrapped(_vector(d, 'reference_yaw_rad') - prior_yaw)),
            color=GREEN, label='Previous best - prior')
    ax.plot(t, np.rad2deg(_wrapped(_vector(d, 'latest_yaw_rad') - prior_yaw)),
            color=ORANGE, label='Graph - prior')
    _panel(ax, 'Yaw changes; not heading RMSE', ylabel='Wrapped heading change (deg)')
    _legend(ax)
    fig.suptitle(f'Track {int(d["track"])}: input, prior and two outputs (no ground truth)')
    figures.append(('01_trajectory_changes', fig))

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4), layout='constrained')
    ax = axes[0, 0]
    scale = _vector(d, 'scale')
    ax.scatter(t[~good], scale[~good], s=8, color=GRAY, label='Rejected')
    ax.scatter(t[good], scale[good], s=8, color=GREEN, label='Geometrically accepted')
    bounds = _vector(d, 'scale_bounds')
    ax.axhline(bounds[0], linestyle='--', color='black', label='Search lower bound')
    ax.axhline(bounds[1], linestyle='--', color='black', label='Search upper bound')
    _panel(ax, 'Fitted scale includes rejected candidates', ylabel='Candidate scale (dimensionless)')
    _legend(ax)
    ax = axes[0, 1]
    # 分成兩列逐幀色帶，避免 3299 個 0/1 階梯互相遮住；沒有平滑旗標。
    from matplotlib.colors import ListedColormap
    edges = np.r_[t[0]-(t[1]-t[0])/2, (t[:-1]+t[1:])/2, t[-1]+(t[-1]-t[-2])/2]
    ax.pcolormesh(edges, [-.5, .5, 1.5], np.vstack([good, temporal]),
                  cmap=ListedColormap(['#dee2e6', BLUE]), vmin=0, vmax=1, shading='flat', rasterized=True)
    ax.set_yticks([0, 1], ['Geometry', 'Temporal'])
    _panel(ax, 'Per-frame pass flags: blue = 1, gray = 0')
    ax.grid(False)
    ax = axes[1, 0]
    ax.scatter(t, _vector(d, 'heldout_median_m'), s=8, color=GRAY)
    ax.axhline(float(d['heldout_limit_m']), linestyle='--', color='black',
               label='Saved acceptance threshold')
    _panel(ax, 'Local fit residual; not pose accuracy',
           ylabel='Held-out median point-to-plane residual (m)')
    _legend(ax)
    ax = axes[1, 1]
    ax.plot(t[1:], _vector(d, 'adjacent_discrepancy_m'), color=BLUE,
            label='Map displacement - GNSS displacement norm')
    ax.plot(t[1:], _vector(d, 'adjacent_gate_m'), '--', color=ORANGE, label='Saved gate')
    _panel(ax, 'Gate requires both frames geometrically accepted',
           'Second frame time of each adjacent pair (s)', 'Displacement discrepancy (m)')
    _legend(ax)
    fig.suptitle('Three stages: geometry -> adjacent consistency -> slow anchors')
    figures.append(('02_map_reliability', fig))

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4), layout='constrained')
    map_delta = np.asarray(d['map_xy']) - np.asarray(d['gnss_xy_at_frames'])
    anchor_bias = np.asarray(d['anchor_bias_xy']).reshape(-1, 2)
    states = np.asarray(d['states'])
    anchor_t = _vector(d, 'anchor_time_s')
    node_t = _vector(d, 'node_time_s')
    for index, ax in enumerate(axes[0]):
        ax.scatter(t[~good], map_delta[~good, index], s=6, color=GRAY,
                   label='Rejected candidate')
        ax.scatter(t[good], map_delta[good, index], s=6, color=BLUE,
                   label='Geometry accepted candidate')
        ax.plot(anchor_t, -anchor_bias[:, index], 'kd', label='Slow anchor (-bias)')
        # MATLAB states(:, 7:8) 對應 Python 的第 6、7 欄。
        ax.plot(node_t, -states[:, 6 + index], color=ORANGE, linewidth=1.4,
                label='Graph (-bias state)')
        _panel(ax, f'{"XY"[index]}: z_GNSS = p + b; correction is -b',
               ylabel='Map minus raw GNSS (m)')
        _legend(ax)
    ax = axes[1, 0]
    # MATLAB 第 6 欄是 effective yaw-rate bias；不是完整三軸 IMU bias。
    ax.plot(node_t, np.rad2deg(states[:, 5]), color=BLUE)
    _panel(ax, 'Estimated state; not calibrated 3-axis IMU bias',
           ylabel='Effective yaw-rate bias (deg/s)')
    ax = axes[1, 1]
    for name, color, label in (('load_ms', GRAY, 'PLY load + voxel'),
                                ('registration_ms', BLUE, 'Registration'),
                                ('total_ms', ORANGE, 'Frame')):
        ax.plot(t, _vector(d, name), color=color, label=label)
    _panel(ax, 'Preparation and batch graph excluded', ylabel='Per-frame wall time (ms)')
    _legend(ax)
    fig.suptitle('Bias sign, sparse anchors and computation boundaries')
    figures.append(('03_bias_and_timing', fig))
    return figures
