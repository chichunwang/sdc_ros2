#!/bin/bash
# Track1 定位（因果版）：和 Track3 正式成績那份同一套九步，每一幀只用到該幀為止的資料，車上可以即時跑
# 和 Track3 只差兩個旗標：追蹤器用 GNSS 當車速預測與弱位置先驗（--gnss）；輸出車體參考點（--lever、--cam-yaw-offset-deg）
# 用法：bash run_track1_causal.sh <Track1 資料夾> [輸出資料夾]
#   Track1 資料夾裡要有 raw_ply/、map.pcd、init_guess.txt、Track1.db3（GNSS 與 IMU 都在 bag 裡）
set -e
export PYTHONUTF8=1   # Windows 中文主控台（cp950）印不出程式說明裡的「−」，強制用 UTF-8
cd "$(dirname "$0")"
TRACK=${1:?請給 Track1 資料夾}
OUT=${2:-./work_track1}
PY=${PYTHON:-python}
PROCS=${PROCS:-2}      # 配準的程序數；記憶體 32 GB 以上可以用 3
mkdir -p "$OUT"

[ -f "$OUT/map_xyzi.npy" ] || { echo "[1/9] 地圖轉檔（約 20 秒）"; $PY scripts/convert_map.py "$TRACK/map.pcd" "$OUT/map_xyzi.npy"; }
[ -f "$OUT/map_v020.npz" ] || { echo "[2/9] 地圖降取樣＋法向量（約 30 秒）"; $PY scripts/prepare_map.py "$OUT/map_xyzi.npy" "$OUT/map_v020.npz" --voxel 0.2; }
[ -f "$OUT/plys_xyz.npy" ] || { echo "[3/9] PLY 讀成快取（約 1 分鐘）"; $PY scripts/cache_plys.py "$TRACK/raw_ply" "$OUT/plys.npz"; }
echo "[4/9] 逐幀尺度，只用當幀和之前 6 幀（約 1 分鐘）"
$PY scripts/scale_from_height.py --plys "$OUT/plys" --H 1.573 --nominal 8.6 --causal --out "$OUT/scale_hc.npy"
echo "[5/9] 逐幀追蹤，正向、不往回追（約 13 分鐘）：不開 --init-search（Track1 的 init_guess 只差 0.35 m／1°，開了會貼到平行結構）；"
echo "      --gnss：車速用幀時間之前最後兩筆 GNSS 差分、GNSS 位置當 σ 1 m 弱先驗（沒有它加速段會靜悄悄跑掉幾十公尺）"
$PY scripts/track_sequential.py --map "$OUT/map_v020.npz" --plys "$OUT/plys" --init-guess "$TRACK/init_guess.txt" \
    --scale-file "$OUT/scale_hc.npy" --backfill 0 --gnss "$TRACK" --out "$OUT/track_causal.npz"
echo "[6/9] 單幀配準四種範圍 12／15／18／25 m（各 4–7 分鐘）"
for R in 12 15 18 25; do
    $PY scripts/register_all.py --map "$OUT/map_v020.npz" --plys "$OUT/plys" --prior "$OUT/track_causal.cam.csv" \
        --scale-file "$OUT/scale_hc.npy" --init "$OUT/track_causal.npz" --class-match --max-range $R --frames all --procs "$PROCS" \
        --out "$OUT/reg_causal_r$R.npz"
done
echo "[7/9] 逐幀挑配準來源：12 m 夠用就用，不夠依序退到 15／18／25 m"
$PY scripts/merge_regs.py --out "$OUT/reg_causal_hybrid.npz" --min-xy-info 2.0 \
    "$OUT/reg_causal_r12.npz" "$OUT/reg_causal_r15.npz" "$OUT/reg_causal_r18.npz" "$OUT/reg_causal_r25.npz"
echo "[8/9] 前向 EKF（gyro＋配準；輸出車體參考點：lever 0.7086,-0.5995、偏角 -0.4455°；Track1 相機對 IMU 的時間差量過是 16 ms，當 0）"
$PY scripts/build_causal.py "$TRACK" --reg "$OUT/reg_causal_hybrid.npz" --gyro-dt 0 \
    --lever 0.7086,-0.5995 --cam-yaw-offset-deg -0.4455 --out "$OUT/ekf_causal.csv"
echo "[9/9] yaw：形狀用 gyro 扣零偏、水平用過去 2 秒的行進方向，輸出 CSV"
$PY scripts/yaw_from_motion.py "$TRACK" --csv "$OUT/ekf_causal.csv" --causal --bias-file "$OUT/ekf_causal.state.npz" \
    --out "$OUT/track1_causal_hybrid_yawmotion.csv"
echo "完成：$OUT/track1_causal_hybrid_yawmotion.csv"
echo "有假真值時可以比：$PY scripts/compare_csv.py $OUT/track1_causal_hybrid_yawmotion.csv --ref candidates/track1_I_ns.csv"
