"""合成教學模型：只說明可觀測性與 KF/RTS，不讀寫定位設定。"""
import numpy as np


def icp_scene():
    """水平地面與四面牆；已知正確對應，省略完整 ICP 的最近鄰更新。"""
    x, y = np.meshgrid(np.linspace(-6, 6, 13), np.linspace(-5, 5, 11))
    ground = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    a, z = np.meshgrid(np.linspace(-5, 5, 11), np.linspace(0, 4, 9))
    wall_x = np.c_[np.full(a.size, 6.), a.ravel(), z.ravel()]
    a, z = np.meshgrid(np.linspace(-6, 6, 13), np.linspace(0, 4, 9))
    wall_y = np.c_[a.ravel(), np.full(a.size, 5.), z.ravel()]
    room = np.vstack([ground, wall_x, wall_x*[-1, 1, 1], wall_y, wall_y*[1, -1, 1]])
    ground_n = np.tile([0., 0., 1.], (len(ground), 1))
    room_n = np.vstack([ground_n, np.tile([1., 0., 0.], (2*len(wall_x), 1)),
                        np.tile([0., 1., 0.], (2*len(wall_y), 1))])
    return ground, ground_n, room, room_n


def icp_jacobian(points, normal):
    """真值位置 t=[0,0,1.5]、R=I 處，七變數 point-to-plane Jacobian。

    delta=[tx,ty,tz,roll,pitch,yaw,log_scale]；旋轉增量以來源原點為中心。
    這裡只有局部幾何，不加入感測器 prior 或 robust weight。
    """
    rotated = points-np.array([0., 0., 1.5])
    return np.c_[normal, np.cross(rotated, normal), np.sum(rotated*normal, axis=1)]


def observability(jacobian):
    """以展示單位縮放，消去 Z/roll/pitch/scale 後查看 XY/yaw 支持。

    奇異值依單位而變，不能當 covariance 或定位精度。閾值只裁決數值秩。
    """
    units = np.array([1., 1., 1., .05, .05, .05, .1])
    scaled = jacobian*units
    u, singular, _ = np.linalg.svd(scaled[:, [2, 3, 4, 6]], full_matrices=False)
    u = u[:, singular > max(scaled.shape)*np.spacing(singular.max())]
    horizontal = scaled[:, [0, 1, 5]]
    horizontal = horizontal-u@(u.T@horizontal)
    return np.linalg.svd(scaled, compute_uv=False), np.linalg.svd(horizontal, compute_uv=False)


def kalman_demo(seed=21, with_bias=True):
    """一維 [position,velocity] KF + RTS，真值只存在於此合成例。

    展示噪聲/過程模型為固定教學條件，不是 production 參數或感測器標定。
    KF 模型未包含 bias 狀態，所以不能靠平滑識別注入的慢偏差。
    RNG 是本函式局部實例，不影響定位或外部隨機狀態。
    """
    t = np.arange(201)*.2
    truth = 2*t+.8*np.sin(t/5)
    bias = .8*np.sin(t/12) if with_bias else np.zeros_like(t)
    measurement = truth+bias+.3*np.random.default_rng(seed).normal(size=len(t))
    h = np.array([1., 0.]); r = .3**2
    state = np.array([measurement[0], (measurement[1]-measurement[0])/(t[1]-t[0])])
    covariance = np.diag([10., 10.])
    means = np.empty((len(t), 2)); cov = np.empty((len(t), 2, 2))
    predicted = np.empty_like(means); predicted_cov = np.empty_like(cov)
    for k in range(len(t)):
        dt = t[k]-t[k-1] if k else 0.
        f = np.array([[1., dt], [0., 1.]])
        q = .2*np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
        state = f@state; covariance = f@covariance@f.T+q
        predicted[k] = state; predicted_cov[k] = covariance
        gain = covariance@h/(h@covariance@h+r)
        state = state+gain*(measurement[k]-h@state)
        a = np.eye(2)-np.outer(gain, h)
        covariance = a@covariance@a.T+np.outer(gain, gain)*r
        means[k] = state; cov[k] = (covariance+covariance.T)/2
    smooth = means.copy(); smooth_cov = cov.copy()
    for k in range(len(t)-2, -1, -1):
        f = np.array([[1., t[k+1]-t[k]], [0., 1.]])
        gain = np.linalg.solve(predicted_cov[k+1], f@cov[k]).T
        smooth[k] += gain@(smooth[k+1]-predicted[k+1])
        smooth_cov[k] += gain@(smooth_cov[k+1]-predicted_cov[k+1])@gain.T
    return {'time': t, 'truth': truth, 'bias': bias, 'measurement': measurement,
            'filtered': means, 'smoothed': smooth, 'filtered_cov': cov, 'smoothed_cov': smooth_cov}


def bias_cost_grids():
    """單一樣本 z=p+b 的不可辨識性；獨立理想 map 觀測僅為合成例。"""
    position, bias = np.meshgrid(np.linspace(8.5, 11.5, 101), np.linspace(-1, 2.5, 101))
    gnss = ((position+bias-10.8)/.3)**2
    both = gnss+((position-10.)/.2)**2
    return position, bias, gnss, both
