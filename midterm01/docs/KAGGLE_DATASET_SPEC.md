# Kaggle 資料集與 CSV 提交規格

記錄日期：2026-10-15。來源為使用者在對話中貼上的 Kaggle Track01／Track02 說明；本次依貼文記錄，未另行登入查驗網頁最新版本。

對應競賽：[Track01 Data](https://www.kaggle.com/competitions/sdc-2026-midterm-01-track-01/data)、[Track02 Data](https://www.kaggle.com/competitions/sdc-2026-localization-track-02/data)。

兩條 Track 的 Dataset Description 與 Topics 相同，提交資料列數不同。

## 資料集說明（兩條 Track 共用）

官方說明表示已提供擷取並前處理完成的影像點雲，格式為 `.ply`。另外提供 ROS bag，可以使用其中的 topics 自行進行資料前處理。

使用者貼上的原文：

> The provided codebase includes a directory containing extracted and preprocessed image point clouds in .ply format.
>
> Additionally, a ROS bag is provided with the following topics, which you may use to perform your own data preprocessing:

## ROS bag topics（兩條 Track 共用）

| Topic | 原文用途 | 中文說明 |
| --- | --- | --- |
| `/lucid_cameras_x00/gige_100_f_hdr/h265` | Front camera data | 前方相機資料 |
| `/lucid_cameras_x01/gige_100_fl_hdr/h265` | Front-left camera data | 左前方相機資料 |
| `/lucid_cameras_x01/gige_100_fr_hdr/h265` | Front-right camera data | 右前方相機資料 |
| `/imu/data` | IMU data, including orientation and acceleration. | IMU 資料，包含姿態與加速度 |
| `/localizer_node/gnss_pose` | GPS data, including position [x, y, z] | GPS 資料，包含位置 `[x, y, z]` |

此段官方說明沒有將 GNSS topic 標示為評分 ground truth；後續使用時不直接視為定位真值。

## CSV 提交規格

兩條 Track 都使用 CSV，欄位及順序為：

```csv
idx,x,y,yaw
```

| Track | 資料列數（不含表頭） | 含表頭的 CSV 紀錄數 |
| --- | ---: | ---: |
| Track01 | **3011** | 3012 |
| Track02 | **3299** | 3300 |

上表含表頭的紀錄數由規定的資料列數加 1 得出；3011／3299 指提交資料列數，不是下載檔案總數。

### Track01：原文 Submission

- submissions should in csv format, with columns [idx,x,y,yaw]
- in this track, your submission should contain exactly 3011 rows (excluding the header)

### Track02：原文 Submission

- submissions should in csv format, with columns [idx,x,y,yaw]
- in this track, your submission should contain exactly 3299 rows (excluding the header)

此段文字未指定 `idx` 起算值與完整 ID 集合、`yaw` 角度單位、座標系或評分方式。這些須另依正式範例及課程說明確認，不由欄名自行推定。

目前 `config/tracks.json` 中的公告幀數同樣為 Track01 3011、Track02 3299；但 `scripts/check_data.py` 只檢查資料檔案與 PLY 數量，尚未驗證提交 CSV 的欄位、ID 或資料列數。
