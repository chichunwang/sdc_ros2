# 官方資源與查漏補缺

更新：2026-10-17。**模板與 Ubuntu 資料已建立並完成首輪檢查，確認缺檔才補下載。** 資料保留 `/home/ricky/sdc_ros2/midterm01/data`；後續建置／測試使用 [GPU 工作區](../../midterm01_gpu/doc/27_PROJECT_OVERVIEW.md)，資料與檢查範圍見 [data/README](../data/README.md) 及 [首跑報告](UBUNTU_FIRST_RUN_20261017.md)。

## 已取得與待核對

| 項目 | 目前狀態 |
| --- | --- |
| 官方題目 | 位於本目錄 [PDF](<SDC 2026 Midterm01 Competition -  Localization.pdf>)，不再使用 localreference 舊路徑 |
| Localization package | 原 CPU 在 `workspace/src/localization/`；目前開發版在 `../../midterm01_gpu/workspace/src/`，含 localization 與 cuda_ndt_backend |
| Track1、Track2 | 已直接核對 map、init_guess、raw_ply、影像、bag 及 metadata；前相機 3011／3299 幀，詳細範圍見首跑報告 |
| MapAnything 點雲 | Ubuntu 有兩組 `kaggle_track0N/TrackN_Mapanything_ply/{ply,index.csv,report.json}` |
| 校正資料 camera_info | 已找到 `TrackN/TrackN_images/camera_info/`，含三相機內參／相機間外參；不等於車體外參 |
| 官方 sample submission | 首跑本地搜尋尚未找到；不可用 MapAnything index.csv 替代 |

官方欄位順序為 `idx,x,y,yaw`，兩條資料列數為 3011／3299，均不含表頭；使用者提供的原文與 topics 見 [Kaggle 規格](KAGGLE_DATASET_SPEC.md)。完整 ID、座標與 yaw 單位仍需核對正式範例／公告。

## 官方來源索引

連結取自 PDF p.9、11；本次未重新登入 Kaggle／Drive 查驗網站最新內容或重新下載。

| 資源 | 官方來源 | 用途 |
| --- | --- | --- |
| Localization | [Google Drive](https://drive.google.com/file/d/1dAnE_-QwlmxVQIdNu13cNBjqEZSL6vCp/view?usp=sharing) | 缺少原始模板時查對，不覆寫目前 package |
| Track01 | [Kaggle Data](https://www.kaggle.com/competitions/sdc-2026-midterm-01-track-01/data?select=Track1) | 核對缺檔、取得該 Track sample |
| Track02 | [Kaggle Data](https://www.kaggle.com/competitions/sdc-2026-localization-track-02/data?select=Track2) | 同上，使用對應地圖／sample |
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

- [x] Ubuntu checkout 可讀取模板／官方 PDF，原始資料保持一份。
- [x] 主機／容器 `check_data.py --strict-count` 通過並保存 JSON。
- [x] 前相機／MapAnything PLY 時間數量、唯一性與 index 對應完成核對；限制見首跑報告。
- [x] camera_info 已找到；相機間外參有檢查，不代表車體外參已確認。
- [x] GPU／CPU 容器實際掛載分別核對，均唯讀共用同一資料根目錄。
- [ ] 取得官方 sample／明確公告，確認完整 ID、車體 frame 與 yaw 單位。

被 Git 忽略的大型資料、輸出與 `.env` 不會隨 Git 同步；程式／PDF／Markdown 依實際追蹤狀態交接。不要將既有 Windows 空資料夾或歷史整包 404 當成 Ubuntu 目前缺少資料。
