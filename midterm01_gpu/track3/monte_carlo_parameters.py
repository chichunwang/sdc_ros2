"""v8 Monte Carlo 參數入口；標準差均是未標定的工程假設。"""
from copy import deepcopy
import math

# 粒子數控制局部搜尋解析度與運算量；不是精度保證，不做榜分掃參數。
PARTICLES = 256
SEED = 20261025
PARAMETERS = dict(
    particles=PARTICLES, seed=SEED, batch_particles=32, max_points=512,
    source_leaf_m=0.5, radius_m=1.5, robust_m=0.3, min_coverage=0.3,
    # 影像重建點高度相關；用平均 robust loss × 有效觀測數，避免
    # 把數千個相關點誤當獨立量測，導致權重立即塌縮。
    effective_points=8.0, ess_fraction=0.5,
    bootstrap_position_std_m=0.5,
    bootstrap_rotation_std_rad=math.radians(5.0),
    bootstrap_log_scale_std=0.05,
    # 平移 diffusion (m/sqrt(s))、失去 PnP 時速度 random walk (m/s/sqrt(s))。
    position_diffusion_m_sqrt_s=0.1,
    velocity_diffusion_mps_sqrt_s=2.0,
    visual_velocity_std_mps=1.0,
    log_scale_diffusion_sqrt_s=0.15,
)


def resolve_config():
    cfg=deepcopy(PARAMETERS)
    for name in ('particles','seed','batch_particles','max_points'):
        if isinstance(cfg[name],bool) or not isinstance(cfg[name],int):
            raise ValueError(f'{name} must be an integer')
    if cfg['particles']<2 or cfg['max_points']<30 or cfg['batch_particles']<1 or cfg['seed']<0:
        raise ValueError('Invalid Monte Carlo count/seed')
    for name,value in cfg.items():
        if name not in ('particles','seed','batch_particles','max_points'):
            if isinstance(value,bool) or not math.isfinite(value) or value<=0:
                raise ValueError(f'{name} must be finite and positive')
    if not 0<cfg['min_coverage']<=1 or not 0<cfg['ess_fraction']<=1:
        raise ValueError('Invalid coverage/ESS fraction')
    cfg['parameter_source']='track3/monte_carlo_parameters.py'
    cfg['status']='uncalibrated_engineering_priors_not_accuracy_claim'
    return cfg
