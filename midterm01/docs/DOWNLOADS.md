**2026-10-23 現行規則：**請以 GPU 文件中的 [00 整合新版（31 頁）](../../midterm01_gpu/doc/Competition%20Regulations/00_SDC_2026_Midterm01_Localization_%E6%95%B4%E5%90%88%E6%96%B0%E7%89%88_20261022.pdf)、[01 因果定位細則（4 頁）](../../midterm01_gpu/doc/Competition%20Regulations/01_Mid01_Updates_Rules.pdf) 及 [48 規則與程式稽核](../../midterm01_gpu/doc/48_RULES_AND_CODE_AUDIT_20261023.md) 為準。所有 Track 禁止未來資訊回修、RTS／backward smoothing 及定位後離線軌跡修正；Track1／2 GNSS 輸入合法不代表 RTS 合規。以下歷史內容不作現行提交推薦。

# 官方資源與查漏補缺

更新：2026-10-23。**Track1／2 本機資料曾完成首輪檢查；Track3 的公告規格依使用者轉貼文字記錄，本機資料完整性與定位 runner 尚未驗證。** 現行規則見[00 整合新版](../../midterm01_gpu/doc/Competition%20Regulations/00_SDC_2026_Midterm01_Localization_%E6%95%B4%E5%90%88%E6%96%B0%E7%89%88_20261022.pdf)、[01 因果定位細則](../../midterm01_gpu/doc/Competition%20Regulations/01_Mid01_Updates_Rules.pdf)，程式盤點見[48](../../midterm01_gpu/doc/48_RULES_AND_CODE_AUDIT_20261023.md)。資料根目錄為 `/home/ricky/sdc_ros2/midterm01/data`；原始資料保留於各自 Track 目錄。後續建置／測試使用 [GPU 工作區](../../midterm01_gpu/doc/27_PROJECT_OVERVIEW.md)，資料與歷史檢查範圍見 [data/README](../data/README.md) 及 [首跑報告](UBUNTU_FIRST_RUN_20261017.md)。

## 已取得與待核對

| 項目 | 目前狀態 |
| --- | --- |
| 現行整合規則 | [PDF（00，31 頁）](../../midterm01_gpu/doc/Competition%20Regulations/00_SDC_2026_Midterm01_Localization_%E6%95%B4%E5%90%88%E6%96%B0%E7%89%88_20261022.pdf)；因果細則見 [01（4 頁）](../../midterm01_gpu/doc/Competition%20Regulations/01_Mid01_Updates_Rules.pdf) |
| 歷史原競賽規則 | [PDF（90，24 頁）](../../midterm01_gpu/doc/Competition%20Regulations/90_%E8%88%8A%E7%89%88_%E5%B7%B2%E5%8F%96%E4%BB%A3_%E5%8F%AF%E5%88%AA%E9%99%A4_SDC_2026_Midterm01_Localization_24%E9%A0%81.pdf)，原檔已搬移，SHA 與搬移前相同 |
| 歷史規則補充 | [PDF（91，10 頁）](../../midterm01_gpu/doc/Competition%20Regulations/91_%E8%88%8A%E7%89%88_%E5%B7%B2%E5%8F%96%E4%BB%A3_%E5%8F%AF%E5%88%AA%E9%99%A4_Mid01_Updates_Adjustments_20261021.pdf)，2026-10-21；歷史解讀見 [45](../../midterm01_gpu/doc/45_COMPETITION_UPDATES_20261021.md) |
| Localization package | 原 CPU 在 `workspace/src/localization/`；目前開發版在 `../../midterm01_gpu/workspace/src/`，含 localization 與 cuda_ndt_backend |
| Track1、Track2 | 已直接核對 map、init_guess、raw_ply、影像、bag 及 metadata；前相機 3011／3299 幀，詳細範圍見首跑報告 |
| Track3 | 使用者於 2026-10-23 轉貼 Kaggle 說明：提供預處理 `.ply` 與三相機／IMU topics，提交欄位 `idx,x,y,yaw`、3142 筆資料（不含表頭）。本機資料完整性與定位 runner 尚未驗證。原文與待核對項目見 [Kaggle 規格](KAGGLE_DATASET_SPEC.md)。 |
| MapAnything 點雲 | Ubuntu 有兩組 `kaggle_track0N/TrackN_Mapanything_ply/{ply,index.csv,report.json}` |
| 校正資料 camera_info | 已找到 `TrackN/TrackN_images/camera_info/`，含三相機內參／相機間外參；不等於車體外參 |
| 官方 sample submission | Track1／2 首跑本地搜尋尚未找到；Track3 sample 尚未核對。不可用 MapAnything index.csv 替代 |

Track1／2 的欄位順序為 `idx,x,y,yaw`，資料列數為 3011／3299，均不含表頭；兩 Track 的 GNSS 仍合法。Track3 無 GNSS pose；使用者轉貼的公告文字列出三個相機 topic、`/imu/data`（orientation、acceleration），以及 3142 筆提交資料（含表頭共 3143 筆紀錄）。完整 ID 集合、yaw 單位／座標系、official sample 與本機資料完整性尚待核對；原文及來源範圍見 [Kaggle 規格](KAGGLE_DATASET_SPEC.md)。

## 官方來源索引

連結取自 PDF p.9、11；本次未重新登入 Kaggle／Drive 查驗網站最新內容或重新下載。

| 資源 | 官方來源 | 用途 |
| --- | --- | --- |
| Localization | [Google Drive](https://drive.google.com/file/d/1dAnE_-QwlmxVQIdNu13cNBjqEZSL6vCp/view?usp=sharing) | 缺少原始模板時查對，不覆寫目前 package |
| Track01 | [Kaggle Data](https://www.kaggle.com/competitions/sdc-2026-midterm-01-track-01/data?select=Track1) | 核對缺檔、取得該 Track sample |
| Track02 | [Kaggle Data](https://www.kaggle.com/competitions/sdc-2026-localization-track-02/data?select=Track2) | 同上，使用對應地圖／sample |
| Track03 | [Kaggle Data](https://www.kaggle.com/competitions/sdc-2026-localization-track-03/data) | 使用者提供的公告來源位置；本機資料統一放 `/home/ricky/sdc_ros2/midterm01/data/Track3/`，規格見 [Kaggle 原文紀錄](KAGGLE_DATASET_SPEC.md) |
| Track01 三相機 | [Kaggle 3cameras_ply](https://www.kaggle.com/competitions/sdc-2026-midterm-01-track-01/data?select=3cameras_ply) | 核對已取得的 MapAnything 來源與結構 |
| Track02 三相機 | [Kaggle 3cameras_ply](https://www.kaggle.com/competitions/sdc-2026-localization-track-02/data?select=3cameras_ply) | 同上 |
| LoGeR 原始碼（選用） | [Google Drive](https://drive.google.com/file/d/1ZHeWxVrf86GDtHtFY7JyXY7w1vge3Fsa/view?usp=sharing) | 需要修改影像重建流程時再研究 |

加入邀請在 PDF p.16，隊名要求見 p.18。保留原始下載內容及來源紀錄；sample 的實際檔名不預設為 `sample_submission.csv`。官方 PDF 對後處理目錄有 process／processing_codes 的差異，本 package 實際使用 `processing_codes/`。

## 使用既有資料

目前 GPU 專案 `midterm01_gpu/.env` 可沿用（相對 GPU 目錄）：

```dotenv
MIDTERM_GPU_DATA_DIR=../midterm01/data
```

原 CPU `midterm01/.env` 的 `MIDTERM_DATA_DIR=./data` 保留，兩者是不同設定。

若改用其他資料位置，填同時含 `Track1/` 與 `Track2/` 的真實 Ubuntu 根目錄。它會唯讀掛載到容器 `/root/data`。`config/tracks.json` 只影響資料檢查工具；執行定位另核對 `map_path`／`data_path`、`ply_path`、`init_guess_file`。保留 HW2 的資料與工作區。

舊的 `UBUNTU_DATA_DOWNLOAD.md` 與 `scripts/download_kaggle.py` 已移除。歷史上 Kaggle CLI 2.2.4 的 ACCESS_TOKEN／單檔與清單成功，Track01 整包與網頁 ZIP 曾 404；這不再是當前「尚無資料」的證據，也不構成重設 token 或重下載的理由。若實際缺檔，依現在可用的 Kaggle 網頁／CLI 補取；不強制使用自訂腳本。

## 已核對與待補證據

- [x] Ubuntu checkout 可讀取模板／官方 PDF；原規則 PDF 已搬至 GPU 文件目錄，SHA 保持，原始資料保持一份。
- [x] 主機／容器 `check_data.py --strict-count` 通過並保存 JSON。
- [x] 前相機／MapAnything PLY 時間數量、唯一性與 index 對應完成核對；限制見首跑報告。
- [x] camera_info 已找到；相機間外參有檢查，不代表車體外參已確認。
- [x] GPU／CPU 容器實際掛載分別核對，均唯讀共用同一資料根目錄。
- [ ] 核對 Track01／02／03 官方 sample，確認完整 ID、車體 frame 與 yaw 單位；Track03 頁面文字已提供 3142 列契約，但尚未登入核驗 sample。

被 Git 忽略的大型資料、輸出與 `.env` 不會隨 Git 同步；程式／PDF／Markdown 依實際追蹤狀態交接。不要將既有 Windows 空資料夾或歷史整包 404 當成 Ubuntu 目前缺少資料。
