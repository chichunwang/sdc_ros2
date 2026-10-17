# Ubuntu 已建立的 Midterm01 資料

目前唯一原始資料仍在本目錄；後續由 [GPU 工作區](../../midterm01_gpu/README.md) 唯讀共用，不建立 `midterm01_gpu/data`。GPU `.env` 使用 `MIDTERM_GPU_DATA_DIR=../midterm01/data`，CPU 舊 `.env` 的 `MIDTERM_DATA_DIR` 不會自動設定 GPU。GPU 的新報告／cache 寫 `midterm01_gpu/outputs`，CPU 舊結果保留 `midterm01/outputs`。

2026-10-17 已直接檢查 Ubuntu 資料：兩條前相機及 MapAnything 各為 3011／3299 幀，無空 PLY、無重複或不可解析時間。MapAnything index 與前相機時間／編號逐列一致；`camera_info` 已找到，位置為 `TrackN/TrackN_images/camera_info/`。兩份 bag SQLite quick_check 通過；前相機每個 PLY timestamp 均存在於 bag 的前相機 **bag timestamp**，且有同名影像。完整檢查範圍、幾何抽樣與限制見 [實測報告](../docs/UBUNTU_FIRST_RUN_20261017.md)。未找到官方 submission sample。

以下為歷史交接與檢查命令。

2026-10-16：使用者確認實際位置為 **`~/sdc_ros2/midterm01/data`**。下列資料樹依使用者截圖記錄；本次助理只能存取 Windows checkout，未直接讀取 Ubuntu 的大型檔案。

```text
data/
├── Track1/
│   ├── raw_ply/
│   ├── Track1_images/
│   ├── init_guess.txt
│   ├── map.pcd
│   ├── metadata.yaml
│   └── Track1.db3
├── Track2/
│   ├── raw_ply/
│   ├── Track2_images/
│   ├── init_guess.txt
│   ├── map.pcd
│   ├── metadata.yaml
│   └── Track2.db3
├── kaggle_track01/Track1_Mapanything_ply/
│   ├── ply/
│   ├── index.csv
│   └── report.json
├── kaggle_track02/Track2_Mapanything_ply/
│   ├── ply/
│   ├── index.csv
│   └── report.json
└── README.md
```

## Ubuntu 主機：基本檢查

先確認已同步程式、目前位置及資料存在。以下操作讀取資料，報告另寫到 outputs：

```bash
cd ~/sdc_ros2/midterm01
realpath data
check_run="$(mktemp -d ../midterm01_gpu/outputs/runs/data_check_host_XXXXXXXX)"
python3 scripts/check_data.py --data-root "$(realpath data)" \
  --strict-count --output "$check_run/data_check.json"
```

預期 raw_ply 為 Track01 **3011**、Track02 **3299**。此工具只檢查 map／init_guess 存在且非空，以及第一層 PLY 數量與非空，不檢查 bag、metadata、點雲內容、時間或 MapAnything。

## Ubuntu 主機：PLY 時間、索引與報告

以下可直接在 `midterm01/` 執行，僅讀取資料並印出摘要。時間以整數處理；這裡用一般 `_ts_<十進位奈秒>` 命名檢查，正式執行仍核對程式索引 log 與 frame manifest。

```bash
python3 - <<'PY'
from pathlib import Path
import csv
import json
import re

root = Path('data').resolve()
for track, expected in [(1, 3011), (2, 3299)]:
    base = root / f'Track{track}'
    print(f'\nTrack{track}, expected={expected}, root={base}')
    for name in ['map.pcd', 'init_guess.txt', 'metadata.yaml', f'Track{track}.db3']:
        p = base / name
        print(name, f'{p.stat().st_size} bytes' if p.is_file() else 'MISSING')
    for directory in [base / 'raw_ply',
                      root / f'kaggle_track0{track}' / f'Track{track}_Mapanything_ply' / 'ply']:
        if not directory.is_dir():
            print(directory, 'MISSING')
            continue
        clouds = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == '.ply')
        timestamps, invalid = [], []
        for p in clouds:
            match = re.search(r'_ts_([0-9]+)$', p.stem)
            if match and int(match[1]) > 0:
                timestamps.append(int(match[1]))
            else:
                invalid.append(p.name)
        print(directory, 'PLY=', len(clouds), 'empty=', sum(p.stat().st_size == 0 for p in clouds),
              'timestamped=', len(timestamps), 'duplicate_times=', len(timestamps) - len(set(timestamps)))
        print('examples=', [p.name for p in clouds[:3]], 'unmatched_examples=', invalid[:3])
    multi = root / f'kaggle_track0{track}' / f'Track{track}_Mapanything_ply'
    index = multi / 'index.csv'
    if index.is_file():
        with index.open(encoding='utf-8-sig', newline='') as stream:
            records = list(csv.reader(stream))
        print('index CSV total records (including possible header)=', len(records))
        print('index first records=', records[:3])
    else:
        print(index, 'MISSING')
    report = multi / 'report.json'
    if report.is_file():
        value = json.loads(report.read_text(encoding='utf-8-sig'))
        print('report preview=', json.dumps(value, ensure_ascii=False)[:2000])
    else:
        print(report, 'MISSING')
PY
```

若 MapAnything 檔名不同，先讀 index 的欄位與 report；不要只為通過檔名規則就批次改名。上方摘要不會解析 PLY 幾何或驗證 index 每列指向的檔案；仍需抽樣讀 PLY header／點數／尺度／座標、確認索引唯一性、檔案存在與 raw_ply／官方 ID 的對應。資料有不同命名時，調整讀取介面或明確建立對應表。

另外檢查：

```bash
# Ubuntu 主機，midterm01/；查看初始矩陣、校正與提交範例位置
head -n 6 data/Track1/init_guess.txt
head -n 6 data/Track2/init_guess.txt
find data -maxdepth 5 -type d -iname '*camera*info*'
find data -maxdepth 5 -type f \( -iname '*submission*.csv' -o -iname '*sample*.csv' \)
```

`camera_info/`、sample 在截圖未見，搜尋無結果後再核對原始下載包／Kaggle；範例也可能使用其他名稱。MapAnything `index.csv` 不能直接當 sample。bag topics／訊息型態可在已配置的 ROS 容器用 `ros2 bag info /root/data/Track1` 與 Track2 核對，再抽查 timestamp、frame_id、感測器訊息品質。不能只因 `.db3` 非空就認定 topic 正確。

## 容器路徑與切換點雲

GPU `.env` 沿用 `MIDTERM_GPU_DATA_DIR=../midterm01/data`（相對 GPU 目錄）時，本目錄唯讀掛在 GPU `/root/data`；CPU 的 `MIDTERM_DATA_DIR=./data` 則是 CPU 容器設定。前相機定位目前使用：

| Track | map／init_guess 所在目錄 | 前相機 ply_path | 三相機候選 ply_path |
| --- | --- | --- | --- |
| 01 | `/root/data/Track1` | `/root/data/Track1/raw_ply` | `/root/data/kaggle_track01/Track1_Mapanything_ply/ply` |
| 02 | `/root/data/Track2` | `/root/data/Track2/raw_ply` | `/root/data/kaggle_track02/Track2_Mapanything_ply/ply` |

三相機路徑是依掛載與截圖推得，尚未實跑。正式切換前先確認時間、尺度、座標與外參；不直接沿用目前 `ply_scale=3.0`。`config/tracks.json` 只用於基本檢查，不會自動選擇上述三相機目錄。

原始資料與 bag 被 Git 忽略，僅保留本 README。新輸出與 map cache 寫到 GPU `midterm01_gpu/outputs/`；不把資料掛載改成可寫來繞過程式問題。HW2 的 bag 是另一份作業資料，保留原位置。
