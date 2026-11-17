#!/usr/bin/env python3
"""外部系統 run 的唯讀圖表（DROID／MASt3R／VGGT／融合）：這些 runner 只輸出 result_*.csv 與 test_info.json，
沒有 gslam 的 odometry、timings.csv、keyframes、閉環檔，所以圖表內容依可取得的資料調整。

    python scripts/analyze_external_run.py outputs/run/<run> [outputs/run/<run> ...]

輸出到 <run>/analysis/：
- 01_trajectory：俯視軌跡（x–z）；融合 run 另畫各來源（Sim(3) 對齊到 base 後）。
- 02_pose_changes：逐幀位移（run 自己的單位）與轉角。
- 03_registration_timing：各階段耗時（外部系統沒有逐幀計時）＋ summary 數值。
- 04_compare_reference：和參考軌跡（E21，Public 0.073）的 132 幀滑動視窗 Sim(3) 位置差，E21 單位 ×1.25 ≈ 公尺。
- manifest.json（來源 SHA）、README.md。失敗 run 不畫；畫圖失敗寫 analysis_status.json。
"""
import csv
import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from lap_consistency import umeyama  # noqa: E402

# 各 Track 的參考軌跡（同 Track 的 E21_selfcalib：我們的 BA＋自校正）、單位換算與比較視窗長度
# Track01：E21 單位 ×1.25 ≈ 公尺（IMU 擬合的 λ）；視窗 132 幀＝Public 長度
# Track02：沒有 IMU，無法換算公尺，差值以參考軌跡自己的單位表示；視窗取全長約 10%（Public 比例，實際切法未公布）
REFERENCES = {
    '01': dict(run='outputs/run/track01_full_2611161931', unit_to_m=1.25, window=132),
    '02': dict(run='outputs/run/track02_full_2611171739', unit_to_m=None, window=421),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load(fn):
    rows = list(csv.reader(open(fn)))[1:]
    return [r[0] for r in rows], np.array([[float(x) for x in r[1:]] for r in rows]).reshape(-1, 4, 4)


def rot_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def analyze(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run = Path(run)
    info = json.loads((run / 'test_info.json').read_text())
    if info.get('status') != 'success':
        print('%s：status=%s，不畫圖' % (run.name, info.get('status')))
        return
    result = sorted(run.glob('result_*.csv'))[0]
    ts, T = load(result)
    P = T[:, :3, 3]
    exp = info.get('experiment', '?')
    out = run / 'analysis'
    out.mkdir(exist_ok=True)

    def save(fig, stem):
        for ext in ('png', 'pdf'):
            fig.savefig(str(out / ('%s.%s' % (stem, ext))), dpi=130, bbox_inches='tight')
        plt.close(fig)

    # 01 軌跡
    fig, ax = plt.subplots(figsize=(9, 8))
    sources = []
    if exp == 'F_fuse_mean':
        par = info['parameters']
        other = par['other'] if isinstance(par['other'], list) else [par['other']]   # 早期融合 run 存成字串
        for src in [par['base']] + other:
            _, S = load(ROOT / src / result.name)
            s, R, t = umeyama(S[:, :3, 3], P)
            sources.append((Path(src).name, s * S[:, :3, 3] @ R.T + t))
        for name, Q in sources:
            ax.plot(Q[:, 0], Q[:, 2], lw=0.8, alpha=0.6, label='source %s (Sim3-aligned)' % name)
    ax.plot(P[:, 0], P[:, 2], color='k' if sources else 'tab:blue', lw=1.0, label='%s result' % exp)
    ax.scatter(P[:1, 0], P[:1, 2], c='k', s=30, zorder=3, label='start')
    ax.set_aspect('equal'); ax.set_xlabel('x (run units)'); ax.set_ylabel('z (run units)')
    ax.set_title('%s  %s  top view (world frame and scale arbitrary)' % (run.name, exp))
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    save(fig, '01_trajectory')

    # 02 逐幀位姿變化
    step = np.r_[np.nan, np.linalg.norm(np.diff(P, axis=0), axis=1)]
    rot = np.r_[np.nan, [rot_deg(T[i - 1, :3, :3], T[i, :3, :3]) for i in range(1, len(T))]]
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(step, lw=0.8); ax[0].set_ylabel('translation per frame (run units)')
    ax[1].plot(rot, lw=0.8, color='tab:red'); ax[1].set_ylabel('rotation per frame [deg]')
    ax[1].set_xlabel('frame idx')
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].set_title('%s  %s' % (run.name, exp))
    save(fig, '02_pose_changes')

    # 03 耗時與 summary
    tim = info.get('timings_seconds') or {}
    fig, ax = plt.subplots(figsize=(9, 4))
    if tim:
        ax.barh(list(tim), [float(v) for v in tim.values()], color='tab:gray')
        for i, v in enumerate(tim.values()):
            ax.text(float(v), i, ' %.0f s' % float(v), va='center')
    ax.set_xlabel('seconds')
    summ = info.get('summary') or {}
    txt = ', '.join('%s=%s' % (k, (('%.4g' % v) if isinstance(v, float) else v)) for k, v in summ.items()
                    if not isinstance(v, (list, dict)))
    ax.set_title('%s stage timings (external system: no per-frame timing)\n%s' % (exp, txt), fontsize=9)
    save(fig, '03_registration_timing')

    # 04 和參考軌跡比較
    ref = REFERENCES[str(info.get('track', '01'))]
    REFERENCE, W = ref['run'], ref['window']
    UNIT_TO_M = ref['unit_to_m'] or 1.0
    unit = 'm' if ref['unit_to_m'] else 'ref units'
    refp = sorted((ROOT / REFERENCE).glob('result_*.csv'))[0]
    _, Rf = load(refp)
    A = Rf[:, :3, 3]
    s, R, t = umeyama(P, A)
    Pa = s * P @ R.T + t
    starts = np.arange(0, len(P) - W + 1, 11)
    win = []
    for i in starts:
        s2, R2, t2 = umeyama(P[i:i + W], A[i:i + W])
        win.append(np.sqrt(np.mean(np.sum((s2 * P[i:i + W] @ R2.T + t2 - A[i:i + W]) ** 2, 1))) * UNIT_TO_M)
    per = np.linalg.norm(Pa - A, axis=1) * UNIT_TO_M
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(per, lw=0.8); ax[0].set_ylabel('per-frame distance after\nglobal Sim3 [%s]' % unit)
    ax[1].plot(starts + W // 2, win, lw=1.0, color='tab:green'); ax[1].set_ylabel('%d-frame window\nSim3 RMS [%s]' % (W, unit))
    ax[1].set_xlabel('frame idx (window centre)')
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].set_title('%s vs reference E21 (%s): global RMS %.3f %s, window median %.3f %s (not ground truth)' % (
        exp, Path(REFERENCE).name, np.sqrt(np.mean(per ** 2)), unit, np.median(win), unit), fontsize=9)
    save(fig, '04_compare_reference')

    man = {result.name: sha256(result), 'reference_result': {str(refp.relative_to(ROOT)): sha256(refp)}}
    for name, _ in sources:
        man.setdefault('fusion_sources', []).append(name)
    (out / 'manifest.json').write_text(json.dumps(dict(sources_sha256=man, experiment=exp,
                                                       script='scripts/analyze_external_run.py'), indent=2) + '\n')
    (out / 'README.md').write_text(
        '# 圖表說明（外部系統 run：%s）\n\n'
        '這個 run 由外部系統的 runner 產生（DROID／MASt3R／VGGT／融合），只有 `%s` 與 `test_info.json`，'
        '沒有 gslam 的 odometry、逐幀計時與閉環紀錄，所以圖表內容和 gslam run 不同：\n\n'
        '- `01_trajectory`：俯視軌跡（x–z 平面，run 自己的單位與世界座標）。融合 run 另畫各來源（Sim(3) 對齊後）。\n'
        '- `02_pose_changes`：逐幀位移與轉角。\n'
        '- `03_registration_timing`：各階段耗時與 summary（外部系統沒有逐幀計時）。\n'
        '- `04_compare_reference`：和同 Track 的 E21（`%s`）的位置差：上圖為全域 Sim(3) 對齊後逐幀距離，下圖為 %d 幀滑動視窗各自對齊後的 RMS。'
        '單位：%s。E21 不是真值，這只是「兩個估計彼此差多少」。\n\n'
        '圖表只讀取本 run 的輸出（SHA 見 manifest.json），不修改 pose；沒有 ground truth，不能當作精度證明。\n'
        % (exp, result.name, Path(REFERENCE).name, W,
           'E21 ×1.25 換算成約公尺' if ref['unit_to_m'] else '參考軌跡自己的單位（Track02 沒有 IMU，無法換算公尺）'))
    print('%s：%s 圖表完成（全域差 %.3f %s、視窗中位數 %.3f %s）' % (
        run.name, exp, np.sqrt(np.mean(per ** 2)), unit, np.median(win), unit))


def main():
    for r in sys.argv[1:]:
        try:
            analyze(ROOT / r if not Path(r).is_absolute() else r)
        except Exception:
            err = traceback.format_exc()
            p = Path(r) / 'analysis_status.json'
            p.write_text(json.dumps(dict(status='failed', error=err[-2000:]), ensure_ascii=False, indent=2) + '\n')
            print('%s：圖表失敗，見 analysis_status.json\n%s' % (r, err[-500:]))


if __name__ == '__main__':
    main()
