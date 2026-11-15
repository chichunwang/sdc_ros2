#!/usr/bin/env python3
"""檢查資料集與「像素 ↔ LoGeR 3D 點」對應是否可信（唯讀；結果寫到 outputs/run/track00_test_*/）。

    python scripts/inspect_data.py --track 01            # 預設抽 8 幀

回答的問題：
- 影像、點雲、sample_submission 的 timestamp 是否一一對應；影像間隔是否約 0.1 s。
- PLY 點數是否等於影像像素數（是否為有序點雲）。
- 把點用內參投影回影像後，投影位置的影像顏色與點的顏色差多少（差很小 → 投影對應可信）。
- 深度分布、相鄰幀深度比（尺度變化的粗略指標）、像素格被點覆蓋的比例。
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vslam'))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--track', choices=['01', '02'], required=True)
    ap.add_argument('--samples', type=int, default=8)
    a = ap.parse_args()
    import cv2
    import create_run
    from gslam import params as params_mod
    from gslam.dataset import Dataset

    params, _ = params_mod.load()
    ds = Dataset(ROOT / params['track']['track' + a.track]['data'])
    run = create_run.create_run('test', '00')
    w, h = ds.size
    ts = np.array([int(t) for t in ds.timestamps], dtype=np.int64)
    dt = np.diff(ts) / 1e9
    report = dict(frames=len(ds), images=len(ds.img), plys=len(ds.ply), missing_images=len(ds.missing_images),
                  image_size=[w, h], K=ds.K.tolist(), camera_info_keys=sorted(ds.camera_info),
                  has_imu=ds.imu_path is not None,
                  dt_seconds=dict(median=float(np.median(dt)), min=float(dt.min()), max=float(dt.max())), samples=[])
    previous_median = None
    for i in np.linspace(0, len(ds) - 1, a.samples).astype(int):
        xyz, rgb = ds.load_cloud(int(i))
        image = ds.load_image(int(i))
        finite = np.all(np.isfinite(xyz), axis=1)
        z = xyz[finite, 2]
        row = dict(idx=int(i), timestamp=ds.timestamps[i], points=int(len(xyz)), organized=bool(len(xyz) == w * h),
                   finite_ratio=float(finite.mean()), depth_percentiles={str(p): float(np.percentile(z, p)) for p in (1, 25, 50, 75, 99)},
                   y_percentiles={str(p): float(np.percentile(xyz[finite, 1], p)) for p in (1, 50, 99)})
        ok = finite & (xyz[:, 2] > 0.1)
        u = ds.K[0, 0] * xyz[ok, 0] / xyz[ok, 2] + ds.K[0, 2]
        v = ds.K[1, 1] * xyz[ok, 1] / xyz[ok, 2] + ds.K[1, 2]
        inside = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
        row['projected_inside_ratio'] = float(inside.mean())
        if image is not None and inside.any():
            iu, iv = np.rint(u[inside]).astype(int), np.rint(v[inside]).astype(int)
            img_rgb = image[iv, iu, ::-1].astype(np.float32) / 255.0
            diff = np.abs(img_rgb - rgb[ok][inside])
            row['rgb_mean_abs_diff'] = float(diff.mean())          # 0 = 完全一致；隨機配對約 0.25–0.35
            shuffled = np.abs(img_rgb - rgb[ok][inside][np.random.default_rng(0).permutation(int(inside.sum()))])
            row['rgb_mean_abs_diff_shuffled'] = float(shuffled.mean())
            if len(xyz) == w * h:  # 有序點雲：再檢查點索引是否就是像素索引（row-major）
                pix = np.flatnonzero(ok)[inside]
                row['index_equals_pixel_ratio'] = float(np.mean(pix == iv * w + iu))
            # 把投影深度畫成圖，疊在影像旁邊供目視
            depth = np.zeros((h, w), np.float32)
            depth[iv, iu] = xyz[ok, 2][inside]
            vis = cv2.applyColorMap(np.clip(depth / 40.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            cv2.imwrite(str(run / ('projection_%05d.jpg' % i)), np.hstack([cv2.resize(image, None, fx=0.5, fy=0.5),
                                                                            cv2.resize(vis, None, fx=0.5, fy=0.5)]))
        if previous_median is not None:
            row['median_depth_ratio_to_previous_sample'] = float(np.median(z) / previous_median)
        previous_median = float(np.median(z))
        report['samples'].append(row)
        print(json.dumps(row, ensure_ascii=False))
    (run / 'data_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('report:', run / 'data_report.json')


if __name__ == '__main__':
    main()
