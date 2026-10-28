#!/bin/bash
# Track3 無 GNSS 定位（離線版，public 0.48）：整段一起平滑，每一幀會參考前後的幀；輸出前相機位姿
# 用法：bash run_track3.sh <Track3 資料夾> [輸出資料夾]
#   Track3 資料夾裡要有 raw_ply/、map.pcd、init_guess.txt、Track3.db3
set -e
export PYTHONUTF8=1   # Windows 中文主控台（cp950）印不出程式說明裡的「−」，強制用 UTF-8
cd "$(dirname "$0")"
TRACK=${1:?請給 Track3 資料夾}
OUT=${2:-./work}
PY=${PYTHON:-python}
PROCS=${PROCS:-2}
mkdir -p "$OUT"

[ -f "$OUT/map_xyzi.npy" ] || { echo "[1/9] 地圖轉檔（約 20 秒）"; $PY scripts/convert_map.py "$TRACK/map.pcd" "$OUT/map_xyzi.npy"; }
[ -f "$OUT/map_v020.npz" ] || { echo "[2/9] 地圖降取樣＋法向量（約 30 秒）"; $PY scripts/prepare_map.py "$OUT/map_xyzi.npy" "$OUT/map_v020.npz" --voxel 0.2; }
[ -f "$OUT/plys_xyz.npy" ] || { echo "[3/9] PLY 讀成快取（約 1 分鐘）"; $PY scripts/cache_plys.py "$TRACK/raw_ply" "$OUT/plys.npz"; }
echo "[4/9] 用相機離地高度算逐幀尺度（前後各 3 幀中位數，約 1 分鐘）"
$PY scripts/scale_from_height.py --plys "$OUT/plys" --H 1.573 --nominal 8.6 --out "$OUT/scale_h.npy"
echo "[5/9] 逐幀追蹤（約 10 分鐘；第 0 幀在 init_guess 附近格點搜尋，跑完再從第 100 幀往回追）"
$PY scripts/track_sequential.py --map "$OUT/map_v020.npz" --plys "$OUT/plys" --init-guess "$TRACK/init_guess.txt" \
    --scale-file "$OUT/scale_h.npy" --init-search --out "$OUT/track.npz"
echo "[6/9] 單幀配準四種範圍 12／15／18／25 m（各 2–7 分鐘）"
for R in 12 15 18 25; do
    $PY scripts/register_all.py --map "$OUT/map_v020.npz" --plys "$OUT/plys" --prior "$OUT/track.cam.csv" \
        --scale-file "$OUT/scale_h.npy" --init "$OUT/track.npz" --class-match --max-range $R --frames all --procs "$PROCS" \
        --out "$OUT/reg_r$R.npz"
done
echo "[7/9] 逐幀挑配準來源：12 m 夠用就用，不夠依序退到 15／18／25 m"
$PY scripts/merge_regs.py --out "$OUT/reg_hybrid.npz" --min-xy-info 2.0 "$OUT/reg_r12.npz" "$OUT/reg_r15.npz" "$OUT/reg_r18.npz" "$OUT/reg_r25.npz"
echo "[8/9] 整段平滑＋gyro＋停車零速（約 1 分鐘）"
$PY scripts/build_nognss.py "$TRACK" --reg "$OUT/reg_hybrid.npz" --init "$OUT/track.npz" \
    --gyro --gyro-dt 0.16 --nhc-lever 1 --zupt --lever 0,0 --cam-yaw-offset-deg 0 --out "$OUT/smoothed.csv"
echo "[9/9] yaw：形狀用 gyro、水平用行進方向，輸出 CSV"
$PY scripts/yaw_from_motion.py "$TRACK" --csv "$OUT/smoothed.csv" --out "$OUT/track3_cam.csv"
echo "完成：$OUT/track3_cam.csv"
