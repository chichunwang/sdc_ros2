**2026-10-23 現行規則：**請以 GPU 文件中的 [00 整合新版（31 頁）](../../midterm01_gpu/doc/Competition%20Regulations/00_SDC_2026_Midterm01_Localization_%E6%95%B4%E5%90%88%E6%96%B0%E7%89%88_20261022.pdf)、[01 因果定位細則（4 頁）](../../midterm01_gpu/doc/Competition%20Regulations/01_Mid01_Updates_Rules.pdf) 及 [48 規則與程式稽核](../../midterm01_gpu/doc/48_RULES_AND_CODE_AUDIT_20261023.md) 為準。所有 Track 禁止未來資訊回修、RTS／backward smoothing 及定位後離線軌跡修正；Track1／2 GNSS 輸入合法不代表 RTS 合規。以下歷史內容不作現行提交推薦。

# Kaggle 資料集與 CSV 提交規格

**2026-10-23 本機核對接續：**已在 `data/Track3` 唯讀確認 3142 個 PLY 的檔名索引 0–3141、唯一遞增時間戳，並與 bag 前相機訊息整數 timestamp 逐項相同；bag 含三相機各 3142 筆與 IMU 11307 筆、無 GNSS pose，map／初始矩陣已盤點。GPU `legacy` map-only 的 50 幀入口已完成乾跑，未執行定位。詳見 [49｜本機證據與手動命令](../../midterm01_gpu/doc/49_TRACK3_CAUSAL_BASELINE_20261023.md)。下列「未實讀 bag」限定原公告轉貼的來源範圍；本機核對另有證據，不把本機 0-based 索引當成官方 sample 的提交 ID／yaw／frame 確認。

記錄日期：2026-10-23。Track01／Track02 資料集說明保留歷史來源紀錄；Track03 規格依使用者於 2026-10-23 貼上的 Kaggle Dataset Description、Topics 與 Submission 文字整理，未登入網頁或實讀 bag 核驗。規則以[00 整合新版](../../midterm01_gpu/doc/Competition%20Regulations/00_SDC_2026_Midterm01_Localization_%E6%95%B4%E5%90%88%E6%96%B0%E7%89%88_20261022.pdf)及[01 因果定位細則](../../midterm01_gpu/doc/Competition%20Regulations/01_Mid01_Updates_Rules.pdf)為準；程式盤點見[48](../../midterm01_gpu/doc/48_RULES_AND_CODE_AUDIT_20261023.md)。

對應競賽：[Track01 Data](https://www.kaggle.com/competitions/sdc-2026-midterm-01-track-01/data)、[Track02 Data](https://www.kaggle.com/competitions/sdc-2026-localization-track-02/data)、[Track03 Data](https://www.kaggle.com/competitions/sdc-2026-localization-track-03/data)。Track03 URL／slug 依使用者提供來源位置記錄，未登入核對頁面。

Track01／Track02 的 Dataset Description、Topics 與提交資料列數紀錄保留在下方歷史小節，不能套用到 Track03。Track03 的文字規格已由使用者貼出；這不代表本機已下載資料、資料齊全或定位 runner 已支援。

## Track03：Dataset Description（使用者貼文原文）

> The provided codebase includes a directory containing extracted and preprocessed image point clouds in .ply format.
>
> Additionally, a ROS bag is provided with the following topics, which you may use to perform your own data preprocessing:

中文整理：提供的 codebase 含一個目錄，放置已擷取及前處理的影像點雲，格式為 `.ply`；另提供 ROS bag，可使用公告列出的 topics 自行進行資料前處理。這是轉貼文字內容，未核驗本機檔案或 bag。

### Track03：Topics（使用者貼文原文與中譯）

| Topic | 原文用途 | 中文說明 |
| --- | --- | --- |
| `/lucid_cameras_x00/gige_100_f_hdr/h265` | Front camera data | 前方相機資料 |
| `/lucid_cameras_x01/gige_100_fl_hdr/h265` | Front-left camera data | 左前方相機資料 |
| `/lucid_cameras_x01/gige_100_fr_hdr/h265` | Front-right camera data | 右前方相機資料 |
| `/imu/data` | IMU data, including orientation and acceleration. | IMU 資料，包含姿態與加速度 |

以上 Track03 貼文未列 GNSS topic；[00 整合新版 p.28](../../midterm01_gpu/doc/Competition%20Regulations/00_SDC_2026_Midterm01_Localization_%E6%95%B4%E5%90%88%E6%96%B0%E7%89%88_20261022.pdf)確認 Track03 不提供 GNSS pose。Track01／02 的 GNSS topic 是下方歷史資料，不能併入 Track03 topics。

## Track01／Track02 歷史 Dataset Description

原先取得的 Track01／Track02 官方說明表示，已提供擷取並前處理完成的影像點雲，格式為 `.ply`；另外提供 ROS bag，可用其中 topics 自行進行前處理。以下記錄其當時貼文，不代表 Track03 有相同 GNSS topic。

使用者貼上的原文：

> The provided codebase includes a directory containing extracted and preprocessed image point clouds in .ply format.
>
> Additionally, a ROS bag is provided with the following topics, which you may use to perform your own data preprocessing:

### ROS bag topics（Track01／Track02 歷史規格）

| Topic | 原文用途 | 中文說明 |
| --- | --- | --- |
| `/lucid_cameras_x00/gige_100_f_hdr/h265` | Front camera data | 前方相機資料 |
| `/lucid_cameras_x01/gige_100_fl_hdr/h265` | Front-left camera data | 左前方相機資料 |
| `/lucid_cameras_x01/gige_100_fr_hdr/h265` | Front-right camera data | 右前方相機資料 |
| `/imu/data` | IMU data, including orientation and acceleration. | IMU 資料，包含姿態與加速度 |
| `/localizer_node/gnss_pose` | GPS data, including position [x, y, z] | GPS 資料，包含位置 `[x, y, z]` |

此段官方說明沒有將 GNSS topic 標示為評分 ground truth；後續使用時不直接視為定位真值。

Track1／Track2 的 GNSS 仍是合法輸入。Track2 公告對 `tl/th` 的文字屬警示；本專案採僅使用 XYZ 位置、不使用 `tl/th` 的政策。所有 Track 均須遵守因果定位規範；GNSS 可用性不使 RTS 或定位後離線修正合規。

## CSV 提交規格

三條 Track 的 CSV 共同表頭與欄位順序為：

```csv
idx,x,y,yaw
```

| Track | 資料列數（不含表頭） | 含表頭的 CSV 紀錄數 |
| --- | ---: | ---: |
| Track01 | **3011** | 3012 |
| Track02 | **3299** | 3300 |
| Track03 | **3142** | 3143 |

Track03 原文 Submission：

> submissions should in csv format, with columns [idx,x,y,yaw]
>
> in this track, your submission should contain exactly 3142 rows (excluding the header)

中文整理：提交格式為 CSV，欄位依序為 `idx,x,y,yaw`；Track03 恰有 3142 筆資料列，不含表頭，故含表頭共 3143 筆 CSV 紀錄。Track01／02 的 3011／3299 也均指資料列數，不是下載檔案總數。

### Track01：原文 Submission

- submissions should in csv format, with columns [idx,x,y,yaw]
- in this track, your submission should contain exactly 3011 rows (excluding the header)

### Track02：原文 Submission

- submissions should in csv format, with columns [idx,x,y,yaw]
- in this track, your submission should contain exactly 3299 rows (excluding the header)

### Track03：來源與未確認欄位

來源為使用者於 2026-10-23 轉貼的 Track03 Kaggle 文字，以及其提供的頁面位置：[Track03 Data](https://www.kaggle.com/competitions/sdc-2026-localization-track-03/data)。未登入網頁或實讀 bag 核驗。公告已給欄位名稱／順序與 3142 筆資料列數，但沒有在所貼文字中指定 `idx` 起算值或完整 ID 集合、`yaw` 單位、座標系，也未提供已核對的 official sample。不得由 Track01／02 範例或本機現有程式自行推定這些契約。

Track03 貼文未指定 `idx` 起算值與完整 ID 集合、`yaw` 角度單位、座標系或評分方式。這些須另依正式範例及課程說明確認，不由欄名自行推定。

此文件只記錄公告文字，不代表本機已取得 Track03 點雲／bag、資料完整性已檢查，或現有定位 runner 已支援 Track03。已知 CSV header 為 `idx,x,y,yaw`；仍待官方 sample 核對的是 idx 起算與完整 ID 集合、yaw 單位，以及座標 frame 契約。
