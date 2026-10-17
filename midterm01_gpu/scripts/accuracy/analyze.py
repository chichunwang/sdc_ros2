#!/usr/bin/env python3
"""在容器內彙整共用地圖診斷與 pose 差異；不計算無真值的正式 ATE。"""
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1])
config = json.loads((root / "config.json").read_text())
colors = {"CPU": "#2864BD", "CUDA": "#D77812"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .18, "figure.dpi": 150})


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"median": float(np.median(a)), "p95": float(np.quantile(a, .95)), "max": float(np.max(a))}


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def load_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


report = {"scope": "First 50 matching real frames per Track; no verified ground truth/Kaggle score.",
          "common_map": "Full fixed LiDAR map downsampled at 1 m; same target for CPU/CUDA, no per-backend crop.",
          "transform": "p_map = result_pose * voxel_0.3m(3 * p_raw); no extra init/extrinsics/trajectory alignment.",
          "tracks": {}}
fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
quality_fig, quality_axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
overlay_fig, overlay_axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

for column, track in enumerate([1, 2]):
    directory = root / f"track{track}"
    frames = load_csv(directory / "frames.csv")
    stamps = np.array([int(row["timestamp_ns"]) for row in frames], dtype=np.int64)
    seconds = (stamps - stamps[0]) / 1e9
    dt = np.diff(stamps) / 1e9
    assert len(frames) == 50 and np.all(dt > 0)
    poses = {"CPU": np.loadtxt(directory / "cpu_result.txt").reshape(-1, 3, 4),
             "CUDA": np.loadtxt(directory / "cuda_result.txt").reshape(-1, 3, 4)}
    residuals = load_csv(directory / "map_residuals.csv")
    pooled = {row["backend"]: {k: float(v) for k, v in row.items() if k not in ("backend", "idx")}
              for row in residuals if int(row["idx"]) == -1}
    metrics = {"frames": 50, "duration_seconds": float(seconds[-1]), "backends": {}}
    points_by_frame = {label: [int(row["input_points"]) for row in residuals
                              if row["backend"] == label and int(row["idx"]) >= 0] for label in poses}
    assert points_by_frame["CPU"] == points_by_frame["CUDA"]
    assert pooled["CPU"]["map_points"] == pooled["CUDA"]["map_points"]
    assert pooled["CPU"]["input_points"] == pooled["CUDA"]["input_points"]
    for label, pose in poses.items():
        assert pose.shape == (50, 3, 4) and np.all(np.isfinite(pose))
        xyz = pose[:, :, 3]
        yaw = np.arctan2(pose[:, 1, 0], pose[:, 0, 0])
        steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        yaw_steps = np.degrees(np.abs(wrap(np.diff(yaw))))
        metrics["backends"][label] = {
            "map_residual": pooled[label], "step_translation_m": stats(steps),
            "step_yaw_deg": stats(yaw_steps), "apparent_speed_mps_not_verified_vehicle_speed": stats(steps / dt),
            "path_length_m": float(np.sum(steps)), "start_end_displacement_m": float(np.linalg.norm(xyz[-1] - xyz[0])),
        }
        axes[0, column].plot(xyz[:, 0], xyz[:, 1], ".-", lw=1.4, markersize=3, color=colors[label], label=label)
        axes[0, column].scatter(xyz[0, 0], xyz[0, 1], marker="s", s=55, color=colors[label], zorder=4)
        axes[1, column].plot(seconds, np.degrees(np.unwrap(yaw)), color=colors[label], label=label)
        axes[2, column].plot(seconds[1:], steps, color=colors[label], label=label)
        rows = [row for row in residuals if row["backend"] == label and int(row["idx"]) >= 0]
        quality_axes[0, column].plot(seconds, [float(row["rmse_m"]) for row in rows], color=colors[label], label=label)
        quality_axes[1, column].plot(seconds, [100 * float(row["inlier_0_5"]) for row in rows], color=colors[label], label=label)

    cpu, gpu = poses["CPU"], poses["CUDA"]
    delta = gpu[:, :, 3] - cpu[:, :, 3]
    yaw_delta = np.degrees(wrap(np.arctan2(gpu[:, 1, 0], gpu[:, 0, 0]) - np.arctan2(cpu[:, 1, 0], cpu[:, 0, 0])))
    relative_rot = np.einsum("nji,njk->nik", cpu[:, :, :3], gpu[:, :, :3])
    rotation_delta = np.degrees(np.arccos(np.clip((np.trace(relative_rot, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    translation_delta = np.linalg.norm(delta, axis=1)
    metrics["cpu_cuda_difference_not_ground_truth_error"] = {
        "translation_rms_m": float(np.sqrt(np.mean(translation_delta ** 2))), "translation_max_m": float(np.max(translation_delta)),
        "yaw_rms_deg": float(np.sqrt(np.mean(yaw_delta ** 2))), "yaw_max_deg": float(np.max(np.abs(yaw_delta))),
        "rotation3d_rms_deg": float(np.sqrt(np.mean(rotation_delta ** 2))),
    }
    with (directory / "pose_differences.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["idx", "timestamp_ns", "filename", "translation_difference_m", "yaw_difference_deg", "rotation3d_difference_deg"])
        writer.writerows((i, int(stamps[i]), frames[i]["filename"], translation_delta[i], yaw_delta[i], rotation_delta[i]) for i in range(50))

    axes[0, column].set(title=f"Track{track}: map-frame pose trajectory", xlabel="Map X (m)", ylabel="Map Y (m)")
    axes[0, column].set_aspect("equal", adjustable="datalim")
    axes[0, column].ticklabel_format(useOffset=False, style="plain")
    axes[1, column].set(title=f"Track{track}: unwrapped yaw", xlabel="Elapsed data time (s)", ylabel="Yaw (deg)")
    axes[2, column].set(title=f"Track{track}: adjacent pose translation", xlabel="Elapsed data time (s)", ylabel="Step length (m)")
    quality_axes[0, column].set(title=f"Track{track}: same-map NN residual", xlabel="Elapsed data time (s)", ylabel="Point-to-map RMS (m)")
    quality_axes[1, column].set(title=f"Track{track}: same-map NN inliers", xlabel="Elapsed data time (s)", ylabel="Points within 0.5 m (%)")
    for ax in [axes[0, column], axes[1, column], axes[2, column], quality_axes[0, column], quality_axes[1, column]]:
        ax.legend(frameon=False)

    map_rows = load_csv(directory / "map_view.csv")
    map_xy = np.array([[float(row["x"]), float(row["y"])] for row in map_rows])
    clouds = load_csv(directory / "frame_clouds.csv")
    for row_index, frame_idx in enumerate([0, 49]):
        ax = overlay_axes[row_index, column]
        ax.scatter(map_xy[:, 0], map_xy[:, 1], s=.5, color="#888888", alpha=.25, rasterized=True, label="Common map")
        extents = []
        for label in poses:
            xy = np.array([[float(row["x"]), float(row["y"])] for row in clouds if int(row["idx"]) == frame_idx and row["backend"] == label])
            ax.scatter(xy[:, 0], xy[:, 1], s=3, alpha=.65, color=colors[label], rasterized=True, label=label)
            extents.append(xy)
        all_xy = np.concatenate(extents)
        low, high = np.quantile(all_xy, .02, axis=0), np.quantile(all_xy, .98, axis=0)
        ax.set_xlim(low[0] - 3, high[0] + 3)
        ax.set_ylim(low[1] - 3, high[1] + 3)
        ax.set_aspect("equal", adjustable="box")
        ax.set(title=f"Track{track}, frame {frame_idx}: transformed source XY", xlabel="Map X (m)", ylabel="Map Y (m)")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.legend(frameon=False, markerscale=2)
    report["tracks"][str(track)] = metrics

for figure, filename in [(fig, "trajectory_comparison"), (quality_fig, "map_residual_comparison"), (overlay_fig, "pointcloud_overlay")]:
    figure.savefig(root / (filename + ".png"), dpi=180)
    figure.savefig(root / (filename + ".pdf"))
    plt.close(figure)
(root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

lines = ["# CPU／CUDA 前 50 幀精度診斷", "", "沒有已確認的真值或 Kaggle 分數；以下比較軌跡穩定性與共同地圖的幾何貼合，不能當成正式 ATE。", "",
         "## 相同 frame 的 CPU／CUDA 差異", "", "| Track | 平移差異 RMS m | yaw 差異 RMS ° | 最大平移差異 m |", "| --- | ---: | ---: | ---: |"]
for track, info in report["tracks"].items():
    difference = info["cpu_cuda_difference_not_ground_truth_error"]
    lines.append(f"| {track} | {difference['translation_rms_m']:.3f} | {difference['yaw_rms_deg']:.3f} | {difference['translation_max_m']:.3f} |")
lines += ["", "## 幾何貼合與穩定性", "", "| Track | 版本 | 共同地圖殘差 RMS m | 中位數 m | 0.5m 內比例 | 最大相鄰平移 m | 最大相鄰 yaw ° |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
for track, info in report["tracks"].items():
    for label, values in info["backends"].items():
        r = values["map_residual"]
        lines.append(f"| {track} | {label} | {r['rmse_m']:.3f} | {r['median_m']:.3f} | {100*r['inlier_0_5']:.1f}% | {values['step_translation_m']['max']:.3f} | {values['step_yaw_deg']['max']:.3f} |")
lines += ["", "## 方法與限制", "",
          "- 每條 Track 的 CPU／CUDA 都使用完全相同的 50 個檔名與奈秒 timestamp；結果未做額外剛體或軌跡對齊。CPU 為已驗證與原 R0 逐 byte 相同的 Release 輸出。",
          "- 每幀僅將原始 PLY XYZ 乘 3.0、清除非有限點並以 0.3m voxel 濾波，再套該版本輸出的 source→map 4×4 矩陣一次；未另套初始矩陣或外參。",
          "- 最近鄰查詢使用同一份完整 1m 下採樣 LiDAR 地圖，避免兩個後端各自不同 submap 影響比較。RMS／中位數／比例彙整 50 幀所有處理後 source 點；這是單向點到地圖殘差，並非車輛位置／yaw 真值誤差。",
          "- 重複結構、點雲重建／尺度誤差、地圖下採樣與局部極小值都可能影響殘差；較低殘差不保證車體 pose 較準。平滑或收斂也不能證明精度。",
          "- CPU PCL NDT 與 CUDA D2D／LM 的目標函數、voxel 鄰域和最佳化不同。這次比較的是兩套完整估計結果，不能把差異僅歸因於 float 或 GPU 硬體。",
          "- 官方提交 frame／車體外參、ply_scale 校準仍待確認；GNSS 沒有被當成真值。合成契約測試的 truth.json 也未納入真實資料評估。",
          "- 只有兩條 Track 的起始約 1.8 秒，不能代表整條 Track 或 Kaggle 精度。未進行提交或修改定位演算法。", "",
          "## 圖表", "", "![軌跡與跳動](trajectory_comparison.png)", "", "![共同地圖殘差](map_residual_comparison.png)", "", "![點雲疊圖](pointcloud_overlay.png)", "",
          "疊圖只顯示 XY 投影，不能取代三維殘差；圖像中的 map 點有抽樣，數值評估使用完整的固定下採樣 map。完整參數／來源在 config.json、各 Track 的 *_params.json、frames.csv、map_residuals.csv 與 pose_differences.csv。"]
(root / "REPORT.md").write_text("\n".join(lines) + "\n")
print(json.dumps({track: {"difference": info["cpu_cuda_difference_not_ground_truth_error"],
                         "map_rms": {label: data["map_residual"]["rmse_m"] for label, data in info["backends"].items()}}
                  for track, info in report["tracks"].items()}, indent=2))
