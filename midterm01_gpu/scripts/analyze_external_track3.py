#!/usr/bin/env python3
"""作者 CPU 結果的唯讀驗證與圖表；不補造 pose、逐幀耗時或精度。"""
from pathlib import Path
import csv
import json
import sys

from external_track3 import sha256, write_json
from external_parameters import get_parameters


def run_parameters(run):
    stored = json.loads((run / "parameters.json").read_text())
    parameters = get_parameters(stored.get("track", 3))
    parameters.update(stored)
    return parameters


def validate_result(run):
    import numpy as np
    run = Path(run)
    p = run_parameters(run)
    expected = p["expected_frames"]
    result_path = run / p["result_name"]
    with result_path.open(newline="") as stream:
        reader = csv.reader(stream)
        if next(reader) != ["idx", "x", "y", "yaw"]:
            raise ValueError("CSV 欄位不符")
        rows = list(reader)
    if len(rows) != expected or any(len(r) != 4 or r[0] != str(i) for i, r in enumerate(rows)):
        raise ValueError(f"CSV 必須為 {expected} 筆、idx 0..{expected-1}，且每列 4 欄")
    values = np.array([r[1:] for r in rows], dtype=float)
    if not np.isfinite(values).all() or np.any(np.abs(values[:, 2]) > np.pi + 1e-9):
        raise ValueError("非有限數值或 yaw 超出作者弧度輸出範圍")
    for name in ["track_causal", *[f"reg_causal_r{r}" for r in p["ranges_m"]], "reg_causal_hybrid"]:
        with np.load(run / "work" / f"{name}.npz") as data:
            if not np.array_equal(data["frame"], np.arange(expected)):
                raise ValueError(f"{name} frame 順序不符")
    with (run / "frame_manifest.csv").open(newline="") as stream:
        manifest = list(csv.DictReader(stream))
    stamps = [int(r["timestamp_ns"]) for r in manifest]
    if len(stamps) != expected or any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("frame manifest 時間／數量錯誤")
    if any(r["idx"] != str(i) or r["original_idx"] != str(i) for i, r in enumerate(manifest)):
        raise ValueError("frame manifest 索引錯誤")
    same = sha256(result_path) == sha256(run / "work" / p["author_output_name"])
    if not same:
        raise ValueError("複製結果與作者原始 CSV bytes 不同")
    audit = dict(status="passed", track=p["track"], frames=expected, finite=True, indices_contiguous=True,
                 intermediate_frame_order=True, timestamps_integer_ns=True, author_csv_bytes_identical=same,
                 yaw_unit="radian_by_author_implementation", evaluation_contract_verified=False,
                 accuracy_verified=False, author_scored_csv_reproduced=False)
    write_json(run / "contract_audit.json", audit)
    return audit


def analyze(run):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(run)
    p = run_parameters(run)
    info = json.loads((run / "analysis_input_info.json").read_text())
    if info["status"] not in ("checks_completed", "completed"):
        raise ValueError("只分析成功檢查或完整完成的作者 run")
    analysis = run / "analysis"
    analysis.mkdir(exist_ok=False)
    sources = [run / name for name in ("analysis_input_info.json", "parameters.json", "external_source_manifest.json", "environment.json", "data_check.json", "stage_timings.csv")]
    def save(fig, stem):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(analysis / f"{stem}.{ext}", dpi=160)
        plt.close(fig)
    if info["mode"] == "check":
        data = json.loads((run / "data_check.json").read_text())
        counts = info["unit_tests"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
        axes[0].bar(["Passed", "Failed", "Errors", "Skipped"], [passed, counts["failures"], counts["errors"], counts["skipped"]])
        axes[0].set_title("Author synthetic unit tests")
        axes[0].set_ylabel("Test count")
        axes[1].axis("off")
        axes[1].text(.02, .9, f"Track: {p['track']}\nData ready: {data['ready']}\nReal localization executed: False\nAccuracy verified: False\n\nMissing: {', '.join(data['missing']) or 'None'}", va="top", wrap=True)
        save(fig, "01_environment_checks")
        sources.append(run / "pytest.xml")
        if (run / "wrapper_pytest.xml").exists():
            sources.append(run / "wrapper_pytest.xml")
        text = f"# 作者流程 Track{p['track']} 環境檢查\n\n執行合成單元／契約測試與真實輸入檔頭盤點，沒有執行真實定位。\n\n" + f"作者測試總數 {counts['tests']}，失敗 {counts['failures']}，錯誤 {counts['errors']}；資料就緒：{data['ready']}。\n"
    else:
        validate_result(run)
        pose = np.loadtxt(run / p["result_name"], delimiter=",", skiprows=1)
        with (run / "frame_manifest.csv").open(newline="") as stream:
            stamps = np.array([int(r["timestamp_ns"]) for r in csv.DictReader(stream)], dtype=np.int64)
        t = (stamps-stamps[0])*1e-9
        reg = np.load(run / "work/reg_causal_hybrid.npz")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(pose[:, 1], pose[:, 2], lw=1)
        ax.scatter(pose[[0, -1], 1], pose[[0, -1], 2], c=["green", "red"])
        ax.set(xlabel="Map X (m)", ylabel="Map Y (m)", title=f"Author pipeline Track{p['track']} output; no ground truth")
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(True)
        save(fig, "01_trajectory")
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(t[1:], np.linalg.norm(np.diff(pose[:, 1:3], axis=0), axis=1)); axes[0].set_ylabel("XY step (m)")
        axes[1].plot(t, np.degrees(np.unwrap(pose[:, 3]))); axes[1].set_ylabel("Yaw (deg)")
        axes[2].plot(t[1:], np.degrees(np.diff(np.unwrap(pose[:, 3])))); axes[2].set_ylabel("Yaw step (deg)")
        axes[2].set_xlabel("Elapsed bag time (s)")
        for ax in axes: ax.grid(True)
        save(fig, "02_pose_changes")
        with (run / "stage_timings.csv").open(newline="") as stream:
            stages = list(csv.DictReader(stream))
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh([r["stage"] for r in stages], [float(r["elapsed_seconds"]) for r in stages])
        ax.set(xlabel="Measured stage wall time (s)", title="Stage timings; per-frame timings unavailable")
        save(fig, "03_registration_timing")
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        axes[0].plot(t, np.load(run / "work/scale_hc.npy")); axes[0].set_ylabel("Scale")
        axes[1].plot(t, reg["ng_frac03"]); axes[1].set_ylabel("NG fraction <0.3m")
        axes[2].plot(t, reg["xy_min_info"]); axes[2].set_ylabel("XY min info")
        axes[3].step(t, np.array(p["ranges_m"])[reg["range_pick"]], where="post"); axes[3].set_ylabel("Range (m)")
        axes[3].set_xlabel("Elapsed bag time (s)")
        for ax in axes: ax.grid(True)
        save(fig, "04_registration_and_scale")
        sources += [run / p["result_name"], run / "frame_manifest.csv", run / "contract_audit.json",
                    run / "work/scale_hc.npy", run / "work/reg_causal_hybrid.npz"]
        text = f"# 作者流程 Track{p['track']} 本機結果\n\nCSV 與作者程式本輪輸出 bytes 相同；不是與作者已評分 CSV 比對。\n\n圖表呈現軌跡、相鄰變化、實測階段耗時與配準診斷。沒有逐幀耗時，03 圖只畫階段耗時。\n\n沒有獨立真值；幾何指標與格式通過不代表定位精度。\n"
    text += "\n限制：\n\n" + "\n".join("- " + item for item in info["limitations"]) + "\n"
    (analysis / "README.md").write_text(text)
    write_json(analysis / "manifest.json", dict(sources={str(p.relative_to(run)): sha256(p) for p in sources}, read_only_analysis=True))


if __name__ == "__main__":
    analyze(Path(sys.argv[1]))
