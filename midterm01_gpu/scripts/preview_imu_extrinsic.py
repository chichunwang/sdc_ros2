#!/usr/bin/env python3
"""主機執行此檔：凍結設定、唯讀轉換 IMU，另存 CSV 與對照圖。"""

# ===== 反覆測試時，直接修改這裡 =====
# IMU → PLY FLU：固定軸 roll、pitch、yaw，單位為度。
IMU_RPY_DEG = (1.0, 6.0, 0.0)
# IMU 原點在 PLY FLU 的位置，單位為公尺；X 前、Y 左、Z 上。
IMU_XYZ_M = (0.0, 0.0, 0.0)
# True：換算 PLY 原點的 specific force；False：僅旋轉量測座標。
COMPENSATE_LEVER_ARM = True
# angular acceleration：目前與過去 gyro 的回歸時間窗，秒；不讀未來。
ALPHA_WINDOW_S = 0.10
# ==================================

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

VERSION = 'imu_extrinsic_preview_v1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def rotation_from_rpy(rpy_deg):
    """Rz(yaw) @ Ry(pitch) @ Rx(roll)：p_ply = R p_imu + t。"""
    import numpy as np
    rpy = np.asarray(rpy_deg, dtype=float)
    if rpy.shape != (3,) or not np.isfinite(rpy).all():
        raise ValueError('IMU_RPY_DEG 必須是三個有限數值')
    roll, pitch, yaw = np.radians(rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def causal_alpha(timestamps_ns, omega, window_s):
    """以 header ns 對目前及過去 gyro 作線性回歸，估計 body angular acceleration。"""
    import numpy as np
    stamps = np.asarray(timestamps_ns)
    omega = np.asarray(omega, dtype=float)
    if (stamps.ndim != 1 or stamps.dtype.kind not in 'iu' or not len(stamps)
            or np.any(stamps < 0) or np.any(stamps > np.iinfo(np.int64).max)):
        raise ValueError('timestamps 必須是非負 int64 範圍的整數 ns')
    stamps = stamps.astype(np.int64)
    if np.any(np.diff(stamps) <= 0):
        raise ValueError('header timestamps 必須嚴格遞增')
    if omega.shape != (len(stamps), 3) or not np.isfinite(omega).all():
        raise ValueError('gyro 必須是有限的 N×3 數值')
    if not isinstance(window_s, (int, float)) or not 0 < window_s <= 10:
        raise ValueError('ALPHA_WINDOW_S 必須在 (0, 10] 秒內')
    window_ns = int(round(window_s * 1_000_000_000))
    if window_ns < 1:
        raise ValueError('ALPHA_WINDOW_S 不可小於 1 ns')
    alpha = np.full_like(omega, np.nan)
    valid = np.zeros(len(stamps), dtype=bool)
    left = 0
    for i in range(len(stamps)):
        while stamps[i] - stamps[left] > window_ns:
            left += 1
        if left == i:
            continue
        # 先用整數 ns 相減，再轉相對秒，保留 epoch 精度。
        x = (stamps[left:i+1] - stamps[i]) * 1e-9
        x = x - x.mean()
        y = omega[left:i+1] - omega[left:i+1].mean(axis=0)
        alpha[i] = x @ y / (x @ x)
        valid[i] = True
    return alpha, valid


def transform_imu(header_ns, accel, gyro, rpy_deg, xyz_m, compensate=True, window_s=0.10):
    """旋轉量測；可選擇由 IMU 原點移至 PLY 原點，不去重力或 bias。"""
    import numpy as np
    rotation = rotation_from_rpy(rpy_deg)
    offset = np.asarray(xyz_m, dtype=float)
    accel, gyro = np.asarray(accel, dtype=float), np.asarray(gyro, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError('IMU_XYZ_M 必須是三個有限數值')
    if type(compensate) is not bool:
        raise ValueError('COMPENSATE_LEVER_ARM 必須是 True 或 False')
    if accel.shape != gyro.shape or accel.ndim != 2 or accel.shape[1] != 3 or not np.isfinite(accel).all():
        raise ValueError('accel 與 gyro 必須是相同形狀的有限 N×3 數值')
    omega = gyro @ rotation.T
    rotated = accel @ rotation.T
    alpha, alpha_valid = causal_alpha(header_ns, omega, window_s)
    tangential, centripetal = np.zeros_like(accel), np.zeros_like(accel)
    valid = np.ones(len(accel), dtype=bool)
    active = compensate and bool(np.any(offset != 0))
    if active:
        # t 是 PLY→IMU，所以 IMU→PLY 的 lever arm r = -t。
        tangential = -np.cross(alpha, offset)
        centripetal = -np.cross(omega, np.cross(omega, offset))
        valid = alpha_valid.copy()
    return dict(rotation=rotation, accel_rotated=rotated,
                accel_target=rotated+tangential+centripetal, gyro_target=omega,
                alpha=alpha, alpha_valid=alpha_valid, accel_valid=valid,
                tangential=tangential, centripetal=centripetal,
                lever_arm_active=active,
                acceleration_origin='ply_origin' if compensate or not np.any(offset != 0)
                                    else 'imu_origin_rotated_only')


def launch():
    """主機端僅用 stdlib；ROS／NumPy／Matplotlib 在既有容器執行。"""
    root = Path(__file__).resolve().parents[1]
    if (root.parent / 'source_manifest.json').exists():
        raise ValueError('請從專案 scripts/preview_imu_extrinsic.py 建立新 run，勿覆寫舊快照')
    made = subprocess.run([sys.executable, str(root/'scripts/create_run.py'), '--kind', 'test', '--track', '03'],
                          capture_output=True, text=True, check=True)
    run = Path(made.stdout.strip())
    container_run = '/workspace/midterm01_gpu/outputs/run/' + run.name
    command = [str(root/'scripts/dev.sh'), 'exec', 'env', 'PYTHONDONTWRITEBYTECODE=1',
               'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', f'MPLCONFIGDIR={container_run}/matplotlib_cache',
               'python', '-B', container_run+'/source/scripts/preview_imu_extrinsic.py',
               '--record-dir', container_run]
    print(f'Run: {run}', flush=True)
    write_json(run/'launcher_info.json', dict(command=command, launcher_script_sha256=sha256(__file__)))
    with (run/'preview.log').open('x') as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
            code = process.wait()
    write_json(run/'launcher_status.json', dict(returncode=code, status='success' if code == 0 else 'failed'))
    if code:
        print(f'執行失敗，請查看 {run}/preview.log；未啟動／重建容器。', file=sys.stderr)
    else:
        print(f'CSV: {run}/imu_transformed.csv\n圖表與說明: {run}/analysis/README.md')
    return code


def read_inputs(data, topic, expected_frame):
    import numpy as np
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu
    bag = data/'Track3.db3'
    with sqlite3.connect(bag.resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        topics = db.execute('SELECT id,type,serialization_format FROM topics WHERE name=?', (topic,)).fetchall()
        if len(topics) != 1 or topics[0][1:] != ('sensor_msgs/msg/Imu','cdr'):
            raise ValueError('需要唯一的 sensor_msgs/msg/Imu CDR topic')
        records = []
        for mid, arrival, blob in db.execute('SELECT id,timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id', (topics[0][0],)):
            m = deserialize_message(blob, Imu)
            if m.header.frame_id != expected_frame:
                raise ValueError(f'IMU frame 不符: {m.header.frame_id!r}')
            if m.linear_acceleration_covariance[0] < 0 or m.angular_velocity_covariance[0] < 0:
                raise ValueError('IMU 缺少加速度或角速度欄位')
            records.append((int(mid), int(arrival), int(m.header.stamp.sec)*1_000_000_000+int(m.header.stamp.nanosec),
                            [m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z],
                            [m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z],
                            not any(m.linear_acceleration_covariance), not any(m.angular_velocity_covariance)))
    if not records:
        raise ValueError('IMU topic 為空')
    arrivals = np.array([r[1] for r in records], dtype=np.int64)
    if np.any(np.diff(arrivals) <= 0):
        raise ValueError('bag arrival timestamps 必須嚴格遞增')
    frames = []
    for path in (data/'raw_ply').glob('*.ply'):
        match = re.fullmatch(r'(\d+)_ts_(\d+)', path.stem)
        if not match:
            raise ValueError(f'無法解析 PLY 檔名: {path.name}')
        frames.append((int(match[1]), int(match[2]), str(path)))
    frames.sort(key=lambda row: row[1])
    if len(frames) != 3142 or [r[0] for r in frames] != list(range(3142)) or any(b[1] <= a[1] for a,b in zip(frames,frames[1:])):
        raise ValueError('Track3 必須有 3142 個遞增的連續 PLY 索引')
    return records, frames


def save_csv(path, rows, elapsed, transformed):
    import numpy as np
    groups = [('a','raw','mps2'),('w','raw','radps'),('a','rotated','mps2'),
              ('a','target','mps2'),('w','target','radps'),('w','target','degps'),
              ('alpha','','radps2'),('a','tangential','mps2'),('a','centripetal','mps2')]
    columns = ['sample_idx','message_id','timestamp_ns','header_timestamp_ns','elapsed_s',
               'source_frame','target_frame','acceleration_origin','accel_valid','alpha_valid',
               'accel_covariance_unknown','gyro_covariance_unknown']
    for symbol, stage, unit in groups:
        columns.extend('_'.join(filter(None,(symbol+axis,stage,unit))) for axis in 'xyz')
    with path.open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for i, row in enumerate(rows):
            values = [*row[3],*row[4],*transformed['accel_rotated'][i],*transformed['accel_target'][i],
                      *transformed['gyro_target'][i],*np.degrees(transformed['gyro_target'][i]),
                      *transformed['alpha'][i],*transformed['tangential'][i],*transformed['centripetal'][i]]
            writer.writerow([i,*row[:3],float(elapsed[i]),'imu_link','ply_flu',transformed['acceleration_origin'],
                             int(transformed['accel_valid'][i]),int(transformed['alpha_valid'][i]),
                             int(row[5]),int(row[6]),*('' if not np.isfinite(v) else float(v) for v in values)])


def make_plots(output, elapsed, accel, gyro, result):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    title = f'IMU -> PLY FLU | RPY={tuple(IMU_RPY_DEG)} deg | XYZ={tuple(IMU_XYZ_M)} m'
    subtitle = f'Lever arm={COMPENSATE_LEVER_ARM}; alpha window={ALPHA_WINDOW_S:g} s; specific force includes gravity'
    color = ['#0072B2','#D55E00','#009E73']

    def panel(ax, j, kind):
        raw = accel if kind == 'accel' else gyro
        converted = result['accel_target'] if kind == 'accel' else result['gyro_target']
        ax.plot(elapsed,raw[:,j],color='#999999',lw=.65,alpha=.8,label='Raw IMU axes')
        if kind == 'accel' and result['lever_arm_active']:
            ax.plot(elapsed,result['accel_rotated'][:,j],color='#E69F00',lw=.65,alpha=.8,label='Rotated; IMU origin')
        origin = 'PLY origin' if result['acceleration_origin']=='ply_origin' else 'IMU origin'
        ax.plot(elapsed,converted[:,j],color=color[j],lw=.65,label='PLY axes; '+(origin if kind=='accel' else 'angular velocity'))
        ax.set_ylabel(f'{"a" if kind=="accel" else "w"}{"XYZ"[j]} [{"m/s²" if kind=="accel" else "rad/s"}]')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8,loc='upper left')
        if kind == 'gyro':
            ax.secondary_yaxis('right',functions=(np.degrees,np.radians)).set_ylabel('[deg/s]')

    def finish(fig, stem):
        fig.suptitle(title+'\n'+subtitle,fontsize=12)
        fig.supxlabel('Bag arrival time since first PLY [s]')
        fig.tight_layout(rect=(0,.025,1,.93))
        for ext in ('png','pdf'):
            fig.savefig(output/f'{stem}.{ext}',dpi=155)
        plt.close(fig)

    fig,axes = plt.subplots(3,2,figsize=(15,9),sharex=True)
    for j in range(3):
        panel(axes[j,0],j,'accel')
        panel(axes[j,1],j,'gyro')
    finish(fig,'00_comparison')
    for kind,stem in [('accel','01_acceleration'),('gyro','02_angular_velocity')]:
        fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True)
        for j in range(3):
            panel(axes[j],j,kind)
        finish(fig,stem)


def analyze(run):
    import numpy as np
    started = time.perf_counter()
    run = run.resolve(strict=True)
    if Path(__file__).resolve() != run/'source/scripts/preview_imu_extrinsic.py':
        raise ValueError('只能執行本輪凍結的分析程式')
    if (run/'test_info.json').exists() or (run/'analysis').exists():
        raise FileExistsError('run 已使用，保留既有輸出；請以固定命令建立新 run')
    source_manifest=json.loads((run/'source_manifest.json').read_text())
    if sha256(__file__) != source_manifest['sources']['scripts/preview_imu_extrinsic.py']:
        raise ValueError('分析程式 SHA 與快照不符')
    info = dict(status='running',version=VERSION,frame_count=0,test_type='read-only IMU extrinsic preview',
                pose_estimation=False,gpu_used=False,score=None,score_source=None,
                started_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(timespec='seconds'))
    write_json(run/'test_info.json',info)
    try:
        rotation=rotation_from_rpy(IMU_RPY_DEG)
        config_path=run/'source/config/track3.json'
        cfg=json.loads(config_path.read_text())
        data=Path(cfg['data_dir'])
        bag=data/'Track3.db3'
        bag_hash=sha256(bag)
        parameters=dict(IMU_RPY_DEG=IMU_RPY_DEG,IMU_XYZ_M=IMU_XYZ_M,
                        COMPENSATE_LEVER_ARM=COMPENSATE_LEVER_ARM,ALPHA_WINDOW_S=ALPHA_WINDOW_S,
                        R_ply_imu=rotation.tolist(),p_ply_equation='R_ply_imu @ p_imu + t_ply_imu',
                        t_definition='IMU origin in PLY FLU, meters; lever arm IMU->PLY = -t',
                        specific_force_equation='R @ f_imu - alpha cross t - omega cross (omega cross t)',
                        alpha_clock='header ns; causal trailing linear regression',orientation_used=False,
                        bias_correction=False,gravity_removed=False)
        write_json(run/'parameters.json',parameters)
        rows,frames=read_inputs(data,cfg['imu']['topic'],cfg['imu']['frame_id'])
        accel=np.array([r[3] for r in rows])
        gyro=np.array([r[4] for r in rows])
        result=transform_imu([r[2] for r in rows],accel,gyro,IMU_RPY_DEG,IMU_XYZ_M,
                             COMPENSATE_LEVER_ARM,ALPHA_WINDOW_S)
        elapsed=np.array([r[1]-frames[0][1] for r in rows],dtype=np.int64)*1e-9
        with (run/'frame_manifest.csv').open('x',newline='') as handle:
            writer=csv.writer(handle)
            writer.writerow(['idx','original_idx','timestamp_ns','path','content_read'])
            writer.writerows((i,*row,False) for i,row in enumerate(frames))
        save_csv(run/'imu_transformed.csv',rows,elapsed,result)
        extraction_seconds=time.perf_counter()-started
        output=run/'analysis'
        output.mkdir()
        def axis_stats(values):
            clean=values[np.isfinite(values).all(axis=1)]
            return dict(valid_rows=len(clean),mean=None if not len(clean) else clean.mean(axis=0).tolist(),
                        std_population=None if not len(clean) else clean.std(axis=0).tolist())
        accel_error=float(np.max(np.abs(np.linalg.norm(accel,axis=1)-np.linalg.norm(result['accel_rotated'],axis=1))))
        gyro_error=float(np.max(np.abs(np.linalg.norm(gyro,axis=1)-np.linalg.norm(result['gyro_target'],axis=1))))
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-12) or accel_error>1e-10 or gyro_error>1e-10:
            raise ValueError('旋轉矩陣或向量 norm 檢查失敗')
        metrics=dict(sample_count=len(rows),first_ply_timestamp_ns=frames[0][1],
                     samples_after_last_ply=sum(r[1]>frames[-1][1] for r in rows),
                     acceleration_origin=result['acceleration_origin'],lever_arm_active=result['lever_arm_active'],
                     invalid_accel_samples=int((~result['accel_valid']).sum()),
                     invalid_alpha_samples=int((~result['alpha_valid']).sum()),
                     accel_raw=axis_stats(accel),accel_rotated=axis_stats(result['accel_rotated']),
                     accel_target=axis_stats(result['accel_target']),gyro_target=axis_stats(result['gyro_target']),
                     rotation_accel_norm_max_error=accel_error,rotation_gyro_norm_max_error=gyro_error,
                     unknown_zero_accel_covariance_count=sum(r[5] for r in rows),
                     unknown_zero_gyro_covariance_count=sum(r[6] for r in rows))
        write_json(output/'metrics.json',metrics)
        plot_started=time.perf_counter()
        make_plots(output,elapsed,accel,gyro,result)
        info['plot_seconds']=time.perf_counter()-plot_started
        readme=f'''# IMU 外參轉換預覽

本輪設定：RPY = {IMU_RPY_DEG} deg，XYZ = {IMU_XYZ_M} m，lever-arm 補償 = {COMPENSATE_LEVER_ARM}，alpha 時間窗 = {ALPHA_WINDOW_S} s。

- [六軸前後對照](00_comparison.png)、[加速度 XYZ](01_acceleration.png)、[角速度 XYZ](02_angular_velocity.png)，各有同名 PDF。
- [完整資料 CSV](../imu_transformed.csv)：{len(rows)} 筆；timestamp_ns 為 bag arrival，header_timestamp_ns 用於估 alpha。elapsed_s 以首張 PLY 為 0。
- 原始 `ax/ay/az_raw_mps2`、`wx/wy/wz_raw_radps` 保留；`a*_rotated_mps2` 僅轉座標；`a*_target_mps2` 為選定原點的輸出；`w*_target_radps` 及 `w*_target_degps` 是轉換後角速度。
- 加速度輸出原點：`{result['acceleration_origin']}`。平移只經剛體旋轉造成的切向／向心項影響加速度，不影響角速度；沒有把公尺直接加到 m/s²。
- 加速度保持 specific force，含重力效應；沒有去重力、bias 修正或使用 IMU orientation，沒有估計定位 pose。
- alpha 用 header 最近 {ALPHA_WINDOW_S} s 的目前與過去 gyro 回歸。首筆或窗內不足兩筆無法估 alpha；啟用非零 lever arm 時對應加速度留空（本輪 {metrics['invalid_accel_samples']} 筆），見 accel_valid。alpha_valid 獨立保存；不讀未來，不填 0 冒充已知。
- CSV 另存 alpha 與 `a*_tangential_mps2`、`a*_centripetal_mps2`，可追查平移補償。微分可能放大雜訊，0.10 s 是工程預設，非已校正濾波器；時間窗回歸在運動突變時有延遲。
- frame_manifest 只解析 PLY 檔名，不讀點雲內容；最後 PLY 之後的 {metrics['samples_after_last_ply']} 筆 IMU 僅作描述，未輸入定位。
- [參數](../parameters.json)、[統計](metrics.json)、[來源及輸出 SHA](manifest.json)。每次改 live scripts/preview_imu_extrinsic.py 頂端參數，再從 Ubuntu 主機執行 `python3 scripts/preview_imu_extrinsic.py`，會另建新 run。
- 定位用 track3/imu_parameters.py 保持原值；本工具是獨立的外參假設檢視，不能僅憑曲線平坦宣稱外參標定成功。

公式及操作說明見快照 [doc54](../source/doc/54_IMU_EXTRINSIC_PREVIEW.md)。剛體加速度關係來源：[Drake SpatialAcceleration](https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_spatial_acceleration.html)。
'''
        (output/'README.md').write_text(readme)
        if sha256(bag)!=bag_hash:
            raise ValueError('來源 bag SHA 在分析期間改變')
        inputs={str(bag):bag_hash,str(config_path):sha256(config_path)}
        outputs={str(p.relative_to(run)):sha256(p) for p in output.iterdir() if p.is_file()}
        outputs.update({name:sha256(run/name) for name in ('imu_transformed.csv','frame_manifest.csv','parameters.json')})
        write_json(output/'manifest.json',dict(status='success',version=VERSION,sources=inputs,
                   source_manifest_sha256=sha256(run/'source_manifest.json'),script_sha256=sha256(__file__),
                   bag_sha_unchanged=True,outputs=outputs))
        info.update(status='success',inspected_frames=len(frames),imu_sample_count=len(rows),
                    invalid_accel_samples=metrics['invalid_accel_samples'],extraction_seconds=extraction_seconds,
                    bag_sha_unchanged=True,source_sha256=inputs,output_sha256=outputs)
        print(json.dumps(dict(status='success',RPY_deg=IMU_RPY_DEG,XYZ_m=IMU_XYZ_M,
                             sample_count=len(rows),invalid_accel_samples=metrics['invalid_accel_samples']),ensure_ascii=False))
        return 0
    except BaseException as error:
        info.update(status='failed',failure=f'{type(error).__name__}: {error}')
        (run/'failure.log').write_text(traceback.format_exc())
        raise
    finally:
        info['elapsed_seconds']=time.perf_counter()-started
        write_json(run/'test_info.json',info)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-dir',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    return launch() if args.record_dir is None else analyze(args.record_dir)


if __name__ == '__main__':
    sys.exit(main())
