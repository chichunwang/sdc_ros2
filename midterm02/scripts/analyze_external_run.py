#!/usr/bin/env python3
"""外部系統 run 的唯讀圖表（DROID／MASt3R／VGGT／融合）：這些 runner 只輸出 result_*.csv 與 test_info.json，
沒有 gslam 的 odometry、timings.csv、keyframes、閉環檔，所以圖表內容依可取得的資料調整。

    python scripts/analyze_external_run.py outputs/run/<run> [outputs/run/<run> ...]

輸出到 <run>/analysis/：
- 01_trajectory：俯視軌跡（x–z）；融合 run 另畫各來源（Sim(3) 對齊到 base 後）。
- 02_pose_changes：逐幀位移（run 自己的單位）與轉角。
- 03_registration_timing：各階段耗時（外部系統沒有逐幀計時）＋ summary 數值。
- 04_compare_reference：和參考軌跡（E21，Public 0.073）的 132 幀滑動視窗 Sim(3) 位置差，E21 單位 ×1.25 ≈ 公尺。
- 05–09（只有 Track02）：和目前最佳 F1 的分解差、俯視差異、高度剖面、重訪一致性、局部尺度。
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
# Track02 的延伸圖表（05–09）：對照目前最佳（F1，Public 0.254）、反向／同方向重訪配對（P1 的 edges.npz）、
# 每幀 LoGeR 地面法向（定義「上」，outputs/cache/dev53_normals.npy）。路長縮放成 2.1 km（與 dev64 相同），單位約公尺。
TRACK2_EXTRA = dict(best='outputs/run/track02_full_2611172344', edges='outputs/run/track02_full_2611172340/debug/edges.npz',
                    normals='outputs/cache/dev53_normals.npy', route_m=2100.0, window=421)

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

    extra = []
    if str(info.get('track')) in ('2', '02'):
        try:
            extra = track2_extra(run, T, save, plt)
        except Exception:
            (out / 'analysis_extra_status.json').write_text(json.dumps(dict(status='failed', error=traceback.format_exc()[-2000:]),
                                                                       ensure_ascii=False, indent=2) + '\n')
            print('%s：延伸圖表失敗，見 analysis/analysis_extra_status.json' % run.name)
    man = {result.name: sha256(result), 'reference_result': {str(refp.relative_to(ROOT)): sha256(refp)}}
    for f in extra:
        man.setdefault('extra_inputs', {})[f] = sha256(ROOT / f)
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
        % (exp, result.name, Path(REFERENCE).name, W,
           'E21 ×1.25 換算成約公尺' if ref['unit_to_m'] else '參考軌跡自己的單位（Track02 沒有 IMU，無法換算公尺）')
        + (('- `05_compare_best`：和目前最佳 F1（`%s`）的逐幀差，分解成沿線／橫向／垂直（Sim(3) 對齊、路長縮放 2.1 km），下圖為 421 幀視窗 RMS。\n'
            '- `06_topview_diff`：俯視軌跡，顏色＝和最佳 F1 的逐幀差；灰線＝最佳 F1。\n'
            '- `07_height_profile`：沿「上」（LoGeR 地面法向平均）的高度剖面，和最佳 F1 疊圖。\n'
            '- `08_revisit_consistency`：同方向（閉環）與反向重訪配對的高度差與水平距離（沿路線、直方圖）；反向的水平距離約一個車道寬才合理。\n'
            '- `09_speed_ratio`：每 100 幀路長 ÷ 最佳 F1 的同段路長（局部尺度差）。\n' % Path(TRACK2_EXTRA['best']).name)
           if str(info.get('track')) in ('2', '02') else '') +
        '\n圖表只讀取本 run 的輸出（SHA 見 manifest.json），不修改 pose；沒有 ground truth，不能當作精度證明。\n')
    print('%s：%s 圖表完成（全域差 %.3f %s、視窗中位數 %.3f %s）' % (
        run.name, exp, np.sqrt(np.mean(per ** 2)), unit, np.median(win), unit))


def track2_extra(run, T, save, plt):
    """Track02 延伸圖表 05–09。回傳用到的外部輸入（相對 ROOT，寫進 manifest）。"""
    cfg = TRACK2_EXTRA
    _, Tb = load(sorted((ROOT / cfg['best']).glob('result_*.csv'))[0])
    nrm = np.load(ROOT / cfg['normals'])
    ok = np.all(np.isfinite(nrm[:, 1:]), 1)
    E = dict(np.load(ROOT / cfg['edges']))

    def scaled(TT):
        X = TT[:, :3, 3]
        return X * cfg['route_m'] / np.sum(np.linalg.norm(np.diff(X, axis=0), axis=1))
    Xb = scaled(Tb)
    s, R, t = umeyama(T[:, :3, 3], Xb)
    X = s * T[:, :3, 3] @ R.T + t
    Rw = np.einsum('ij,njk->nik', R, T[:, :3, :3])
    up = np.einsum('nij,nj->ni', Tb[nrm[ok, 0].astype(int), :3, :3], nrm[ok, 1:]).mean(0)
    up /= np.linalg.norm(up)
    v = np.gradient(Xb, axis=0)
    v -= np.outer(v @ up, up)
    fwd = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    lat = np.cross(up, fwd)
    d = X - Xb
    comp = dict(along=np.sum(d * fwd, 1), lateral=np.sum(d * lat, 1), vertical=d @ up)
    name, best = run.name, Path(cfg['best']).name
    W = cfg['window']
    starts = np.arange(0, len(X) - W + 1, 11)
    win = []
    for i in starts:
        s2, R2, t2 = umeyama(X[i:i + W], Xb[i:i + W])
        win.append(np.sqrt(np.mean(np.sum((s2 * X[i:i + W] @ R2.T + t2 - Xb[i:i + W]) ** 2, 1))))

    # 05 和最佳 F1 的差（分解）
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for k, c in (('along', 'tab:blue'), ('lateral', 'tab:orange'), ('vertical', 'tab:green')):
        ax[0].plot(comp[k], lw=0.8, color=c, label='%s (RMS %.2f m)' % (k, np.sqrt(np.mean(comp[k] ** 2))))
    ax[0].set_ylabel('difference to best F1 [m]'); ax[0].legend(fontsize=8)
    ax[1].plot(starts + W // 2, win, color='tab:red'); ax[1].set_ylabel('%d-frame window Sim3 RMS [m]' % W)
    ax[1].set_xlabel('frame idx')
    for a_ in ax:
        a_.grid(alpha=0.3)
    ax[0].set_title('%s vs best %s: global RMS %.2f m, window median %.2f m (route scaled to 2.1 km; not ground truth)' % (
        name, best, np.sqrt(np.mean(np.sum(d ** 2, 1))), np.median(win)), fontsize=9)
    save(fig, '05_compare_best')

    # 06 俯視差異
    e1 = lat / np.linalg.norm(lat, axis=1, keepdims=True)
    h1 = np.cross(up, [1.0, 0, 0]); h1 /= np.linalg.norm(h1); h2 = np.cross(up, h1)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.plot(Xb @ h1, Xb @ h2, color='0.7', lw=2.5, label='best F1 %s' % best)
    sc = ax.scatter(X @ h1, X @ h2, c=np.linalg.norm(d, axis=1), s=2, cmap='viridis', label=name)
    fig.colorbar(sc, ax=ax, label='per-frame difference to best F1 [m]')
    for k in range(0, len(X), 500):
        ax.annotate(str(k), (X[k] @ h1, X[k] @ h2), fontsize=7)
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title('%s top view (horizontal plane, Sim3-aligned to best F1), frame numbers every 500' % name, fontsize=9)
    _ = e1
    save(fig, '06_topview_diff')

    # 07 高度剖面
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(Xb @ up - Xb[0] @ up, color='0.5', lw=1.5, label='best F1')
    ax[0].plot(X @ up - Xb[0] @ up, lw=0.8, label=name)
    ax[0].set_ylabel('height along up [m]'); ax[0].legend(fontsize=8)
    ax[1].plot(comp['vertical'], lw=0.8, color='tab:green'); ax[1].set_ylabel('height difference [m]')
    ax[1].set_xlabel('frame idx')
    for a_ in ax:
        a_.grid(alpha=0.3)
    ax[0].set_title('height profile (up = mean LoGeR ground normal)', fontsize=9)
    save(fig, '07_height_profile')

    # 08 重訪一致性
    fig, ax = plt.subplots(2, 2, figsize=(13, 7))
    for col, (kind, label) in enumerate((('same', 'same direction (loop closures)'), ('opp', 'opposite direction'))):
        ij = E[kind + '_ij'].astype(int)
        dz = (X[ij[:, 0]] - X[ij[:, 1]]) @ up
        dh = X[ij[:, 0]] - X[ij[:, 1]]
        dh = np.linalg.norm(dh - np.outer(dh @ up, up), axis=1)
        ax[0, col].scatter(ij[:, 0], dz, s=6, label='height diff (|median| %.3f m)' % np.median(np.abs(dz)))
        ax[0, col].scatter(ij[:, 0], dh, s=6, label='horizontal dist (median %.2f m)' % np.median(dh))
        ax[0, col].set_xlabel('frame a'); ax[0, col].set_ylabel('[m]'); ax[0, col].legend(fontsize=8)
        ax[0, col].set_title(label, fontsize=9); ax[0, col].grid(alpha=0.3)
        ax[1, col].hist(dz, bins=40, alpha=0.7, label='height diff'); ax[1, col].set_xlabel('height difference [m]')
        ax[1, col].legend(fontsize=8); ax[1, col].grid(alpha=0.3)
    fig.suptitle('%s revisit consistency (pairs from P1 edges)' % name, fontsize=10)
    save(fig, '08_revisit_consistency')

    # 09 局部尺度
    sp = np.linalg.norm(np.diff(X, axis=0), axis=1)
    spb = np.linalg.norm(np.diff(Xb, axis=0), axis=1)
    c = np.arange(50, len(sp) - 50, 10)
    ratio = np.array([sp[i - 50:i + 50].sum() / max(spb[i - 50:i + 50].sum(), 1e-9) for i in c])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(c, 100 * (ratio - 1), lw=0.9)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('frame idx'); ax.set_ylabel('100-frame path length vs best F1 [%]'); ax.grid(alpha=0.3)
    ax.set_title('%s local scale vs best F1 (std %.2f%%)' % (name, 100 * np.std(ratio)), fontsize=9)
    save(fig, '09_speed_ratio')
    _ = Rw
    return [str(Path(cfg['best']) / 'result_2.csv'), cfg['edges'], cfg['normals']]


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
